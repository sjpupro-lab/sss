"""SSS ingestion — file → SSS memory cells.

Accepts an arbitrary file (image / text / video), converts it into the
shape that CEMemory understands (per-block edge map + average colour,
or text-as-grid), and writes the resulting cells into a memory
instance. The edge map is a Sobel-magnitude threshold, not a strict
Canny — the cell stash uses it as a structure signature, so the
distinction matters.

Pure-Python image work: no cv2, no Pillow. Image decoding goes through
tools/sss_image_io for PNG/PPM and through ffmpeg → PPM for everything
else (JPEG, BMP, video). All resampling, gradient and tag work uses
numpy directly.

Public surface:
    ingest_file(filepath, memory, *, filename=None) -> dict
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from tools import sss_image_io

W, H = 256, 256
ROW_BLOCKS = {
    "hair":  (0.00, 0.15),
    "face":  (0.15, 0.40),
    "upper": (0.40, 0.65),
    "lower": (0.65, 0.88),
    "bg":    (0.88, 1.00),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
# .csv used to flow through the text path; it now drives ingest_csv()
# (batch image ingest via `path,label` rows). Plain prose still goes
# through .txt / .md.
TEXT_EXTS  = {".txt", ".md"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ────────────────────────────────────────────────────────────
# numpy-only helpers (no cv2)
# ────────────────────────────────────────────────────────────
def _resize_nn(img, target_h=H, target_w=W):
    """Nearest-neighbour resize to target_h × target_w.

    Uses linspace endpoint mapping so the corners of the source image
    map to the corners of the output (no consistent crop bias at the
    bottom/right edges).
    """
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    if target_h <= 1:
        yi = np.zeros(target_h, dtype=np.int64)
    else:
        yi = np.linspace(0, h - 1, target_h).round().astype(np.int64)
    if target_w <= 1:
        xi = np.zeros(target_w, dtype=np.int64)
    else:
        xi = np.linspace(0, w - 1, target_w).round().astype(np.int64)
    yi = np.clip(yi, 0, max(0, h - 1))
    xi = np.clip(xi, 0, max(0, w - 1))
    if img.ndim == 2:
        return img[yi[:, None], xi[None, :]]
    return img[yi[:, None], xi[None, :], :]


def _to_gray(img):
    """BGR uint8 → grayscale uint8 (B=0.114, G=0.587, R=0.299)."""
    if img.ndim == 2:
        return img
    a = img.astype(np.uint16)
    return ((a[..., 0] * 29 + a[..., 1] * 150 + a[..., 2] * 77) >> 8
            ).astype(np.uint8)


def _sobel_edges(gray, threshold=80):
    """Sobel-magnitude threshold. Same shape as `cv2.Canny` uses
    elsewhere in SSS (uint8 0/255)."""
    g = gray.astype(np.int32)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.abs(gx) + np.abs(gy)
    return ((mag >= int(threshold)).astype(np.uint8)) * 255


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _ffmpeg_to_ppm(src_path: str, dst_ppm: str):
    """Convert any ffmpeg-readable image into a single PPM. Used for
    JPEG / BMP and any other format we can't decode in pure Python."""
    cp = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
         "-frames:v", "1", "-f", "image2", "-pix_fmt", "rgb24",
         dst_ppm],
        capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {src_path}: "
            f"{cp.stderr.decode('utf-8', errors='replace').strip()}")


def _read_image_any(filepath: str) -> np.ndarray:
    """Decode an image to a BGR uint8 ndarray. Routes by extension:
    PNG / PPM go through sss_image_io; everything else goes through
    ffmpeg → PPM → sss_image_io."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".png":
        img = sss_image_io.read_png(filepath)
        if img.ndim == 2:                 # grayscale → BGR
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]            # drop alpha
        return img
    if ext == ".ppm":
        return sss_image_io.read_ppm(filepath)

    # Anything else: ffmpeg → temp PPM.
    if not _ffmpeg_available():
        raise RuntimeError(
            f"cannot decode {ext} without ffmpeg "
            f"(install via `pkg install ffmpeg` on Termux)")
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        tmp_ppm = f.name
    try:
        _ffmpeg_to_ppm(filepath, tmp_ppm)
        return sss_image_io.read_ppm(tmp_ppm)
    finally:
        try:
            os.unlink(tmp_ppm)
        except OSError:
            pass


# ────────────────────────────────────────────────────────────
# Fingerprint — Python port of sss_fingerprint() in
# ce_core/sss_rowvae.c. Byte-histogram with a position-mixed
# neighbour bump, then L2-normalised. Must stay byte-identical to
# the C version so future fp-keyed search lines up.
# ────────────────────────────────────────────────────────────
SSS_FP_LEN = 256


def fingerprint(text: str) -> np.ndarray:
    fp = np.zeros(SSS_FP_LEN, dtype=np.float32)
    if not text:
        return fp
    data = text.encode("utf-8")
    for i, b in enumerate(data):
        fp[b] += 1.0
        n = (b + 1 + i) & 0xFF
        fp[n] += 0.3
    s = float(np.sqrt(np.sum(fp * fp)))
    if s > 1e-9:
        fp /= s
    return fp


# ────────────────────────────────────────────────────────────
# Image ingest
# ────────────────────────────────────────────────────────────
def _ingest_image_array(img: np.ndarray, memory, *, tags, source) -> dict:
    """Split a 256×256 BGR image into row blocks, write one cell per
    block, return a per-block summary. Used by the legacy unlabeled
    path; the labeled path goes through ingest_labeled_image()."""
    img = _resize_nn(img, H, W)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)

    per_block = {}
    cells_added = 0
    for bname, (s, e) in ROW_BLOCKS.items():
        y0, y1 = int(H * s), int(H * e)
        block = img[y0:y1]
        gray = _to_gray(block)
        canny = _sobel_edges(gray, 80)
        color = np.mean(block, axis=(0, 1))
        memory.add(bname, canny, color, 0.45, list(tags), source)
        per_block[bname] = per_block.get(bname, 0) + 1
        cells_added += 1
    return {"cells_added": cells_added, "blocks": per_block}


def _block_fft(block: np.ndarray) -> tuple:
    """Per-row, per-channel rfft of a (bh, W, 3) uint8 block normalised
    into [0, 1]. Returns two contiguous float32 ndarrays of shape
    `(bh, 3, NF)` for amp and phase. NF = W // 2 + 1.

    The compact ndarray representation avoids the Python list +
    per-row ndarray-object overhead the earlier list-of-arrays form
    paid (≈ bh * 3 small allocations per cell × 5 cells per image)."""
    block_f = block.astype(np.float32) / 255.0
    # rfft along axis=1 (the W axis): result is (bh, NF, 3) complex.
    spec = np.fft.rfft(block_f, axis=1)
    # Reorder to (bh, 3, NF) so reconstruction can index `[row, ch]`.
    spec = np.transpose(spec, (0, 2, 1))
    amp = np.ascontiguousarray(np.abs(spec).astype(np.float32))
    phase = np.ascontiguousarray(np.angle(spec).astype(np.float32))
    return amp, phase


def _block_subband_25x4(block: np.ndarray) -> dict:
    """1-level 2D Haar wavelet decomposition of a (bh, W, 3) uint8
    block into the four subbands

        LL — average of (rows, cols)         "큰 구조"
        LH — average row × diff col          "가로 디테일"
        HL — diff row    × average col       "세로 디테일"
        HH — diff (rows, cols)               "대각선 디테일"

    Each subband is (bh//2, W//2, 3) float32 in roughly [-0.5, 0.5]
    for the high bands and [0, 1] for LL (normalised the same way
    `_block_fft` normalises). Together the four subbands cover 100 %
    of the signal — the inverse `_inverse_subband_25x4` round-trips
    bit-for-bit (within float32 precision).

    Numpy-only; no pywt / cv2 / scipy. Memory is exactly one
    intermediate (bh//2, W, 3) plus the four output arrays."""
    block_f = block.astype(np.float32) / 255.0
    bh, w, ch = block_f.shape
    # Drop the trailing row / column when the dimension is odd so
    # every subband has the same shape. Callers slicing the source
    # image at ROW_BLOCKS already get even-height slabs in the
    # default 256×256 layout, but the guard keeps arbitrary block
    # sizes safe.
    bh2 = bh - (bh & 1)
    w2  = w  - (w  & 1)
    if bh2 == 0 or w2 == 0:
        empty = np.zeros((max(1, bh2 // 2), max(1, w2 // 2), ch), np.float32)
        return {"LL": empty, "LH": empty.copy(),
                "HL": empty.copy(), "HH": empty.copy()}
    a = block_f[:bh2:2, :w2]            # even rows, even-width crop
    b = block_f[1:bh2:2, :w2]           # odd rows
    low_row  = (a + b) * 0.5            # (bh/2, w, 3)
    high_row = (a - b) * 0.5
    LL = (low_row[:, 0:w2:2]  + low_row[:, 1:w2:2])  * 0.5
    LH = (low_row[:, 0:w2:2]  - low_row[:, 1:w2:2])  * 0.5
    HL = (high_row[:, 0:w2:2] + high_row[:, 1:w2:2]) * 0.5
    HH = (high_row[:, 0:w2:2] - high_row[:, 1:w2:2]) * 0.5
    return {
        "LL": np.ascontiguousarray(LL.astype(np.float32)),
        "LH": np.ascontiguousarray(LH.astype(np.float32)),
        "HL": np.ascontiguousarray(HL.astype(np.float32)),
        "HH": np.ascontiguousarray(HH.astype(np.float32)),
    }


def _inverse_subband_25x4(subbands: dict) -> np.ndarray:
    """Inverse of `_block_subband_25x4`. Returns a (2*bh, 2*w, 3)
    float32 array in [0, 1] (un-clamped — caller maps to uint8 if
    needed). Implements the symmetric Haar synthesis so a
    decompose / synthesise round-trip is exact within float32
    precision."""
    LL = np.asarray(subbands["LL"], dtype=np.float32)
    LH = np.asarray(subbands["LH"], dtype=np.float32)
    HL = np.asarray(subbands["HL"], dtype=np.float32)
    HH = np.asarray(subbands["HH"], dtype=np.float32)
    bh, w, ch = LL.shape
    # Synthesise the two row-half bands first.
    low_row  = np.empty((bh, w * 2, ch), np.float32)
    high_row = np.empty((bh, w * 2, ch), np.float32)
    low_row[:,  0::2] = LL + LH
    low_row[:,  1::2] = LL - LH
    high_row[:, 0::2] = HL + HH
    high_row[:, 1::2] = HL - HH
    # Then the two column-half bands.
    out = np.empty((bh * 2, w * 2, ch), np.float32)
    out[0::2] = low_row + high_row
    out[1::2] = low_row - high_row
    return out


def ingest_labeled_image(filepath, memory, label_text: str,
                         *, source: Optional[str] = None) -> dict:
    """Ingest one image with a required text label.

    The label drives the cell's tags AND its fingerprint (Python port
    of sss_fingerprint). For each row block the cell stores:
        canny  — Sobel edge map
        color  — mean BGR
        amp    — per-row, per-channel rfft amplitudes
        phase  — per-row, per-channel rfft phases
        tags   — every word in the label (lowercased, len > 1)
        fp     — 256-bin fingerprint of the bare label text
    """
    if not label_text or not label_text.strip():
        return {"error": "라벨이 필요합니다"}

    tags = [w.lower() for w in label_text.split() if len(w) > 1]
    if not tags:
        return {"error": "라벨이 필요합니다"}
    fp = fingerprint(label_text.strip())

    img = _read_image_any(str(filepath))
    img = _resize_nn(img, H, W)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)

    src = source if source is not None else str(filepath)
    per_block = {}
    cells_added = 0
    for bname, (s, e) in ROW_BLOCKS.items():
        y0, y1 = int(H * s), int(H * e)
        block = img[y0:y1]
        gray = _to_gray(block)
        canny = _sobel_edges(gray, 80)
        color = np.mean(block, axis=(0, 1))
        amps, phases = _block_fft(block)
        # Per-block 25%×4 subband decomposition. Lives alongside the
        # FFT amp/phase pair on the same memory cell so the restore
        # path can score row + column attention against LL/LH/HL/HH
        # without re-decomposing every retrieval.
        subbands = _block_subband_25x4(block)
        memory.add(bname, canny, color, 0.5, list(tags), src,
                   amp=amps, phase=phases, fp=fp, subbands=subbands)
        per_block[bname] = per_block.get(bname, 0) + 1
        cells_added += 1

    # Pose / radar perception: produces a sidecar entry on the same
    # memory instance (memory.pose_cells via add_pose) when the
    # underlying CEMemory subclass supports it. Failures are
    # swallowed — pose perception is additive, never blocking the
    # primary 5 row-block cells.
    pose_summary = None
    pose_cells_added = 0
    if hasattr(memory, "add_pose"):
        try:
            from tools import sss_pose_radar
            pose_filename = os.path.basename(str(filepath)) or "<image>"
            pose_result = sss_pose_radar.analyze_pose_radar(
                img, label_text.strip(), filename=pose_filename)
            pose_result.pop("_arrays", None)
            pose_tags = ["pose", "radar", "motion"] + list(tags)
            memory.add_pose(
                pose_result, pose_tags,
                source=f"pose_radar:{pose_filename}",
                quality=float(pose_result.get("quality", 0.5)))
            pose_cells_added = 1
            pose_summary = {
                "bbox": pose_result.get("bbox"),
                "joint_count": len(pose_result.get("joints") or {}),
                "motion_count": len(pose_result.get("motions") or []),
                "dirty_zone_count": len(pose_result.get("dirty_zones") or {}),
                "quality": pose_result.get("quality"),
                "area_frac": pose_result.get("area_frac"),
            }
        except Exception as e:
            pose_summary = {"error": str(e)}

    out = {
        "type": "image",
        "cells_added": cells_added,
        "blocks": per_block,
        "tags": tags,
        "label": label_text.strip(),
        "pose_cells_added": pose_cells_added,
    }
    if pose_summary is not None:
        out["pose_radar"] = pose_summary
    return out


def _resolve_csv_path(fname: str, base_dir: str) -> str:
    """Resolve a CSV path against base_dir, refusing anything that
    escapes the sandbox.

    Rules:
        - When `base_dir` is non-empty, absolute paths are refused
          and relative paths must resolve (after symlink follow) to
          a location inside `realpath(base_dir)`. Any escape via
          `..`, symlinks, or `/etc/...` raises `ValueError`.
        - When `base_dir` is empty (the caller has opted out of the
          sandbox — e.g. running ingest_csv directly from a script),
          the path is returned as-is via abspath. The /api/ingest
          endpoint always passes a non-empty base_dir, so the
          server is sandboxed by default.
    """
    if not base_dir:
        return os.path.abspath(fname)
    if os.path.isabs(fname):
        raise ValueError(f"absolute path not allowed: {fname!r}")
    abs_base = os.path.realpath(base_dir)
    abs_target = os.path.realpath(os.path.join(base_dir, fname))
    sep = os.path.sep
    if abs_target != abs_base and not abs_target.startswith(abs_base + sep):
        raise ValueError(
            f"path escapes base_dir: {fname!r} → {abs_target!r}")
    return abs_target


def ingest_csv(csv_path, memory, base_dir: str = "") -> dict:
    """Batch-ingest from a 2-column CSV: `path,label`.

    Lines starting with `#` are skipped. A header row whose first
    column is `filepath` / `file` / `filename` (case-insensitive) is
    also skipped. Relative paths resolve under `base_dir` (or the cwd
    if base_dir is falsy). When base_dir is set, absolute paths and
    `..`/symlink escapes are refused per row — see
    `_resolve_csv_path`."""
    rows = []
    errors = []
    with open(csv_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                errors.append({"line": lineno, "error": "missing label"})
                continue
            fname = parts[0].strip()
            label = parts[1].strip()
            # Skip an obvious header row.
            if (lineno == 1
                    and fname.lower() in ("filepath", "file", "filename", "path")):
                continue
            if not fname or not label:
                errors.append({"line": lineno,
                               "error": "empty filepath or label"})
                continue
            rows.append((lineno, fname, label))

    cells_total = 0
    files_ok = 0
    files_err = []
    for lineno, fname, label in rows:
        try:
            path = _resolve_csv_path(fname, base_dir)
        except ValueError as e:
            files_err.append({"line": lineno, "file": fname,
                              "error": str(e)})
            continue
        if not os.path.exists(path):
            files_err.append({"line": lineno, "file": fname,
                              "error": "file not found"})
            continue
        try:
            r = ingest_labeled_image(path, memory, label,
                                     source=f"csv:{os.path.basename(csv_path)}#{lineno}")
        except Exception as e:
            files_err.append({"line": lineno, "file": fname,
                              "error": str(e)})
            continue
        if "error" in r:
            files_err.append({"line": lineno, "file": fname,
                              "error": r["error"]})
            continue
        cells_total += r["cells_added"]
        files_ok += 1

    return {
        "type": "csv",
        "rows": len(rows),
        "files_ok": files_ok,
        "files_err": files_err,
        "errors": errors,
        "cells_added": cells_total,
    }


# ────────────────────────────────────────────────────────────
# Text ingest
# ────────────────────────────────────────────────────────────
def _ingest_text(filepath: str, memory, filename: str) -> dict:
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    lines = text.splitlines()
    cells_added = 0
    block_keys = list(ROW_BLOCKS.keys())

    for line_idx, line in enumerate(lines):
        line_bytes = line.encode("utf-8", errors="replace")
        if not line_bytes:
            continue
        # Each chunk of up to 256 bytes becomes one cell. Lines ≤ 256
        # bytes → one row; longer lines → split across multiple rows.
        for start in range(0, len(line_bytes), 256):
            chunk = line_bytes[start:start + 256]
            row = np.zeros(256, dtype=np.uint8)
            row[:len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
            # Reshape to 16×16 so the cell's canny field has a real
            # 2-D shape — matches what CE generators expect downstream.
            grid = row.reshape(16, 16)
            mean_byte = float(np.mean(row[:len(chunk)])) if len(chunk) else 0.0
            color = np.array([mean_byte, mean_byte, mean_byte], np.float32)
            block = block_keys[(line_idx + (start // 256)) % len(block_keys)]
            memory.add(block, grid, color, 0.40,
                       ["text", "ingest", filename],
                       f"ingest_text:{filename}:line{line_idx}_off{start}")
            cells_added += 1
    return {"type": "text", "cells_added": cells_added,
            "lines": len(lines)}


# ────────────────────────────────────────────────────────────
# Video ingest
# ────────────────────────────────────────────────────────────
def _ingest_video(filepath: str, memory, filename: str,
                  frame_step: int = 12, max_frames: int = 24) -> dict:
    if not _ffmpeg_available():
        raise RuntimeError(
            "video ingest requires ffmpeg "
            "(install via `pkg install ffmpeg` on Termux)")

    workdir = tempfile.mkdtemp(prefix="sss_ingest_video_")
    try:
        # Sample every `frame_step`-th frame, scale to 256×256,
        # cap at `max_frames`. PPM output → sss_image_io.read_ppm.
        pattern = os.path.join(workdir, "f_%04d.ppm")
        cp = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", filepath,
             "-vf", f"select=not(mod(n\\,{int(frame_step)})),scale=256:256",
             "-vsync", "vfr", "-frames:v", str(int(max_frames)),
             "-pix_fmt", "rgb24", pattern],
            capture_output=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract frames: "
                f"{cp.stderr.decode('utf-8', errors='replace').strip()}")

        frame_paths = sorted(
            os.path.join(workdir, f) for f in os.listdir(workdir)
            if f.endswith(".ppm"))
        cells_added = 0
        prev_frame = None
        for fi, fp in enumerate(frame_paths):
            frame = sss_image_io.read_ppm(fp)
            summary = _ingest_image_array(
                frame, memory,
                tags=["ingest", "video", filename, f"frame{fi}"],
                source=f"ingest_video:{filename}:f{fi}")
            cells_added += summary["cells_added"]

            # Optical flow proxy: per-pixel abs diff between consecutive
            # frames. Cheap, no cv2, captures motion energy.
            if prev_frame is not None:
                diff = np.abs(frame.astype(np.int16)
                              - prev_frame.astype(np.int16)
                              ).astype(np.uint8)
                # Store the diff itself as a "motion" cell, parked in
                # the "bg" block since flow tends to be background-ish.
                gray = _to_gray(diff)
                canny = _sobel_edges(gray, 40)
                color = np.mean(diff, axis=(0, 1))
                memory.add("bg", canny, color, 0.40,
                           ["ingest", "video", "flow", filename],
                           f"ingest_video_flow:{filename}:f{fi-1}->{fi}")
                cells_added += 1
            prev_frame = frame
        return {"type": "video",
                "frames_processed": len(frame_paths),
                "cells_added": cells_added}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ────────────────────────────────────────────────────────────
# Public entry
# ────────────────────────────────────────────────────────────
def ingest_file(filepath, memory, *,
                filename: Optional[str] = None,
                label: Optional[str] = None,
                base_dir: str = "") -> dict:
    """Ingest one file into `memory`. Returns a per-type summary dict.

    Routes by extension:
      .png / .jpg / .jpeg / .ppm / .bmp  → labeled image ingest (label
                                            is required — unlabeled
                                            images are refused)
      .csv                               → batch via ingest_csv
      .txt / .md                         → text ingest
      .mp4 / .avi / .mov / .mkv / .webm  → video ingest
    Unknown extensions raise ValueError.
    """
    p = str(filepath)
    name = filename if filename is not None else os.path.basename(p)
    ext = os.path.splitext(name)[1].lower()

    if ext in IMAGE_EXTS:
        if not label or not label.strip():
            return {"type": "image", "error": "라벨이 필요합니다",
                    "filename": name}
        return ingest_labeled_image(p, memory, label, source=name)
    if ext == ".csv":
        return ingest_csv(p, memory, base_dir=base_dir)
    if ext in TEXT_EXTS:
        return _ingest_text(p, memory, name)
    if ext in VIDEO_EXTS:
        return _ingest_video(p, memory, name)
    raise ValueError(f"unsupported extension {ext!r} for {name!r}")
