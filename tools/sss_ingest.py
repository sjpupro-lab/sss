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
TEXT_EXTS  = {".txt", ".md", ".csv"}
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
# Image ingest
# ────────────────────────────────────────────────────────────
def _ingest_image_array(img: np.ndarray, memory, *, tags, source) -> dict:
    """Split a 256×256 BGR image into row blocks, write one cell per
    block, return a per-block summary."""
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


def _ingest_image(filepath: str, memory, filename: str) -> dict:
    img = _read_image_any(filepath)
    summary = _ingest_image_array(
        img, memory,
        tags=["ingest", "image", filename],
        source=f"ingest:{filename}")
    summary["type"] = "image"
    return summary


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
def ingest_file(filepath, memory, *, filename: Optional[str] = None) -> dict:
    """Ingest one file into `memory`. Returns a per-type summary dict.

    Routes by extension:
      .png / .jpg / .jpeg / .ppm / .bmp  → image ingest
      .txt / .md / .csv                  → text ingest
      .mp4 / .avi / .mov / .mkv / .webm  → video ingest
    Unknown extensions raise ValueError.
    """
    p = str(filepath)
    name = filename if filename is not None else os.path.basename(p)
    ext = os.path.splitext(name)[1].lower()

    if ext in IMAGE_EXTS:
        return _ingest_image(p, memory, name)
    if ext in TEXT_EXTS:
        return _ingest_text(p, memory, name)
    if ext in VIDEO_EXTS:
        return _ingest_video(p, memory, name)
    raise ValueError(f"unsupported extension {ext!r} for {name!r}")
