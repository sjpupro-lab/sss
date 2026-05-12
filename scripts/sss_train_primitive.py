#!/usr/bin/env python3
"""sss_train_primitive — Phase 3 primitive motif trainer.

Design philosophy (Phase 3 alignment with Phase 2 .sfb):

The .sfb is a **feature spectral dictionary**. Each motif is "an
abstract visual token characterised by a particular frequency
shape" — and absolutely nothing else.

  * No phase is stored. The frequency envelope is amplitude-only.
  * No per-row FFT bytes are stored. We condense the (H × NF × 3)
    per-row FFT amplitudes that a Phase 1 trainer would have
    written into a single 128-bin row_freq envelope per motif.
  * No original pixel coordinates. `position_heatmap` is a coarse
    16×16 probability surface — not pixel-addressable.
  * The envelope encodes *frequency shape*, not "energy total".
    DC (mean brightness / per-channel mean) is explicitly removed
    before condensation so that two folders with similar mean
    luminance but different texture come out distinct.

Algorithm (per folder ↔ motif):

  (1) Normalise every image to 64×64 RGB (aspect-preserving, zero-pad)
      float32 ∈ [0, 1]. Channel-wise mean subtraction: R -= mean(R),
      G -= mean(G), B -= mean(B).

  (2) row_freq[128]: per-image, per-row rfft → 33 bins; zero bin 0
      (kill DC); resample 33 → 128 (linear); average across rows;
      average across images; L2-normalise.

  (3) col_freq[128]: symmetric to (2) but along the column axis.

  (4) color_freq[3][128]: flatten each channel (length 4096) → rfft
      → 2049 bins; zero bin 0; resample to 128; folder-average;
      L2-normalise per channel.

  (5) position_heatmap[16][16]: per-image Sobel magnitude on the
      mean-subtracted grayscale, downsample to 16×16 by 4×4
      averaging; folder-average; min-max normalise to [0, 1]
      (final uint8 quantisation happens at .sfb write time).

  (6) coherence: per-image row_freq's cosine vs the folder mean
      row_freq (after DC removal). Average across images. Single-
      image folder → coherence = 1.0.

  (7) confidence = coherence × min(image_count / 10, 1.0).

  (8) variation_cluster_id = 0 (single cluster per folder for v1).

Output is a .npz intermediate with one row per motif. Phase 3
finalises into .sfb via `sss_build_feature_bank.py`.

CLI:
    python scripts/sss_train_primitive.py \\
        --data dataset/primitive \\
        --out  build/primitive_motifs.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import sss_image_io                     # noqa: E402


# ── Constants (must match .sfb v1) ─────────────────────────────────
FREQ_BINS    = 128
HEATMAP_DIM  = 16
PATCH_SIZE   = 64

# PNG / PPM only — that's what `_read_image_rgb` supports. Listing
# .jpg / .bmp here would invite folder scans that crash on the
# first non-PNG/PPM hit. Convert upstream via ffmpeg if you need
# the other formats.
IMAGE_EXTS = (".png", ".ppm")


# ── Image normalisation ───────────────────────────────────────────

def _read_image_rgb(path: str) -> np.ndarray:
    """Decode `path` to a (H, W, 3) uint8 RGB array. Routes through
    `tools.sss_image_io` for PNG / PPM; everything else goes through
    ffmpeg → PPM the same way `tools.sss_ingest` does, but we avoid
    that dep here by only accepting PNG / PPM (the supported primitive
    dataset formats).

    `sss_image_io.read_png` returns BGR; we flip to RGB so the channel
    layout matches the .sfb convention (R, G, B)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        img = sss_image_io.read_png(path)
    elif ext == ".ppm":
        img = sss_image_io.read_ppm(path)
    else:
        raise ValueError(
            f"sss_train_primitive only accepts PNG / PPM, got {ext!r} "
            f"for {path!r}. Convert via ffmpeg upstream.")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    # sss_image_io returns BGR; flip channel axis to RGB for envelope work.
    return np.ascontiguousarray(img[..., ::-1], dtype=np.uint8)


def _resize_pad_64(img: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize into a 64×64 canvas with zero-pad.

    The longer side scales to 64; the shorter side is zero-padded
    (top/bottom or left/right symmetrically). Stays numpy-only and
    nearest-neighbour to match `tools.sss_ingest._resize_nn` precision.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE, 3), np.uint8)
    s = PATCH_SIZE / float(max(h, w))
    nh = max(1, int(round(h * s)))
    nw = max(1, int(round(w * s)))
    if nh == 1:
        yi = np.zeros(nh, dtype=np.int64)
    else:
        yi = np.linspace(0, h - 1, nh).round().astype(np.int64)
    if nw == 1:
        xi = np.zeros(nw, dtype=np.int64)
    else:
        xi = np.linspace(0, w - 1, nw).round().astype(np.int64)
    scaled = img[yi[:, None], xi[None, :], :]
    out = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    y0 = (PATCH_SIZE - nh) // 2
    x0 = (PATCH_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw, :] = scaled
    return out


def _to_float_demeaned(img_u8: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3) RGB → float32 (H, W, 3), per-channel
    mean-subtracted. The DC bin of the per-row / per-col / per-channel
    rfft will then be (close to) zero — we still zero it explicitly
    below in case rounding leaves a fragment."""
    f = img_u8.astype(np.float32) / 255.0
    for c in range(3):
        f[..., c] -= float(f[..., c].mean())
    return f


# ── Envelope extraction ───────────────────────────────────────────

def _resample_linear(amp: np.ndarray, target: int) -> np.ndarray:
    """Linear resample of a 1-D amplitude array to `target` length.
    The input is already non-negative (FFT amplitude). Endpoints map
    to endpoints (np.interp on linspace), matching the `numpy.interp`
    behaviour used elsewhere in SSS."""
    src = amp.shape[-1]
    if src == target:
        return amp.astype(np.float32, copy=False)
    src_x    = np.linspace(0.0, 1.0, src,    endpoint=True)
    target_x = np.linspace(0.0, 1.0, target, endpoint=True)
    return np.interp(target_x, src_x, amp).astype(np.float32)


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector. Zero vector stays zero (cosine
    against it returns 0 in our matchers)."""
    n = float(np.sqrt((v.astype(np.float64) ** 2).sum()))
    if n < 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32)


def _zero_mean_l2_normalise(v: np.ndarray) -> np.ndarray:
    """Zero-mean across the 128 envelope bins, then L2-normalise.
    Equivalent to the Pearson form of cosine similarity.

    Amplitude envelopes live in the non-negative orthant, where raw
    cosine has a high floor (~0.7 between broadband and any
    structured peak vector) that smears the off-diagonal of the
    motif-vs-motif similarity matrix. Subtracting the per-envelope
    mean signs the deviation from "flat broadband": a peaked
    spectrum becomes positive at peaks and negative elsewhere,
    while a flat noise spectrum collapses to ≈0. The resulting
    cosine spans [-1, 1] and discriminates structured-vs-flat
    cleanly. Zero-norm inputs are passed through (cosine against
    them returns 0)."""
    v64 = v.astype(np.float64)
    centred = v64 - v64.mean()
    n = float(np.sqrt((centred ** 2).sum()))
    if n < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (centred / n).astype(np.float32)


def _envelope_row_freq(img_f: np.ndarray) -> np.ndarray:
    """row_freq[128] for one image. img_f shape (H, W, 3), already
    mean-subtracted per channel. We sum the RGB channels into a
    luminance signal first so the row spectrum reflects structural
    texture rather than chroma-specific energy (chroma lives in
    color_freq). DC bin is zeroed; bins are resampled 33 → 128 and
    averaged across the H rows.

    Returns a (128,) float32 vector. Not L2-normalised — the caller
    averages across images then normalises once."""
    luma = img_f.sum(axis=2)                           # (H, W)
    spec = np.fft.rfft(luma, axis=1)
    amp = np.abs(spec).astype(np.float32)              # (H, NF)
    amp[:, 0] = 0.0
    # Resample row-wise then average across rows.
    resamp = np.empty((amp.shape[0], FREQ_BINS), np.float32)
    for r in range(amp.shape[0]):
        resamp[r] = _resample_linear(amp[r], FREQ_BINS)
    return resamp.mean(axis=0)


def _envelope_col_freq(img_f: np.ndarray) -> np.ndarray:
    """col_freq[128]. Mirror of _envelope_row_freq along the column
    axis. The input is (H, W, 3); we transpose to (W, H, 3) and reuse
    the row helper."""
    return _envelope_row_freq(np.transpose(img_f, (1, 0, 2)))


def _envelope_color_freq(img_f: np.ndarray) -> np.ndarray:
    """color_freq[3][128]. Per channel, flatten the 64×64 plane into
    a length-4096 1-D signal, rfft, kill bin 0, resample to 128.
    Returns (3, 128) float32, NOT L2-normalised yet."""
    out = np.zeros((3, FREQ_BINS), np.float32)
    for c in range(3):
        flat = img_f[..., c].ravel()                   # (H*W,)
        spec = np.fft.rfft(flat)
        amp = np.abs(spec).astype(np.float32)
        amp[0] = 0.0
        out[c] = _resample_linear(amp, FREQ_BINS)
    return out


def _sobel_heatmap_16(img_u8: np.ndarray) -> np.ndarray:
    """16×16 heatmap of edge energy. Sobel-magnitude on grayscale,
    downsampled by 4×4 averaging. Returns (16, 16) float32 in arbitrary
    units — the caller folder-averages and min-max normalises."""
    # Luma from BGR-ish RGB array. Coefficients match
    # tools.sss_ingest._to_gray (B=0.114, G=0.587, R=0.299) but we
    # only care about an edge-sensitive scalar — the exact coefficient
    # choice doesn't change which pixels are edges.
    a = img_u8.astype(np.int32)
    gray = (a[..., 0] * 77 + a[..., 1] * 150 + a[..., 2] * 29) >> 8
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = (np.abs(gx) + np.abs(gy)).astype(np.float32)  # (64, 64)
    # 64 / 16 == 4, exact 4×4 block average.
    blk = mag.reshape(HEATMAP_DIM, 4, HEATMAP_DIM, 4)
    return blk.mean(axis=(1, 3)).astype(np.float32)


# ── Per-folder motif aggregation ──────────────────────────────────

def _scan_folder(folder: Path) -> list:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def train_motif_from_folder(folder: Path) -> dict | None:
    """Aggregate every image in `folder` into a single motif dict.
    Returns None if the folder has no usable images. The dict shape
    matches what `sss_build_feature_bank.py` consumes:

        {
          "label":            <folder name>,
          "row_freq":         (128,) f32 (L2-normalised),
          "col_freq":         (128,) f32 (L2-normalised),
          "color_freq":       (3, 128) f32 (per-channel L2-normalised),
          "position_heatmap": (16, 16) f32 in [0, 1],
          "coherence":        float in [0, 1],
          "confidence":       float in [0, 1],
          "activation_count": int (== image count at training time),
          "variation_cluster_id": int (= 0 for v1),
          "image_count":      int,
        }
    """
    paths = _scan_folder(folder)
    if not paths:
        return None

    per_image_row = []
    per_image_col = []
    per_image_color = []
    per_image_heat = []
    heat_accum = np.zeros((HEATMAP_DIM, HEATMAP_DIM), np.float32)

    for p in paths:
        raw = _read_image_rgb(str(p))
        norm = _resize_pad_64(raw)
        f = _to_float_demeaned(norm)
        per_image_row.append(_envelope_row_freq(f))
        per_image_col.append(_envelope_col_freq(f))
        per_image_color.append(_envelope_color_freq(f))
        hm = _sobel_heatmap_16(norm)
        per_image_heat.append(hm)
        heat_accum += hm

    per_image_row   = np.stack(per_image_row,   axis=0)   # (N, 128)
    per_image_col   = np.stack(per_image_col,   axis=0)   # (N, 128)
    per_image_color = np.stack(per_image_color, axis=0)   # (N, 3, 128)
    per_image_heat  = np.stack(per_image_heat,  axis=0)   # (N, 16, 16)
    n = len(paths)

    mean_row   = per_image_row.mean(axis=0)
    mean_col   = per_image_col.mean(axis=0)
    mean_color = per_image_color.mean(axis=0)

    # Zero-mean + L2 (Pearson form). Amplitude envelopes are non-
    # negative; raw cosine on those vectors floors around 0.7 even
    # for unrelated motifs and erodes the off-diagonal separation
    # the Phase 3 spec calls for (< 0.5 between groups). Subtracting
    # the per-envelope mean turns the comparison into "shape of the
    # spectrum, ignoring its DC offset" — broadband-vs-peaked
    # cosines drop to near zero, motif-vs-itself stays at 1.
    row_freq = _zero_mean_l2_normalise(mean_row)
    col_freq = _zero_mean_l2_normalise(mean_col)
    color_freq = np.stack([_zero_mean_l2_normalise(mean_color[c])
                           for c in range(3)], axis=0)

    # Heatmap: min-max normalise the folder-averaged Sobel magnitude.
    heat_mean = heat_accum / float(n)
    hmin = float(heat_mean.min())
    hmax = float(heat_mean.max())
    if hmax - hmin > 1e-9:
        heat_norm = (heat_mean - hmin) / (hmax - hmin)
    else:
        heat_norm = np.zeros_like(heat_mean)
    heat_norm = heat_norm.astype(np.float32)

    # Coherence — "how consistent is this folder?" using the same
    # weighted-cosine metric the matcher uses at retrieval time. A
    # single axis is allowed to be degenerate (zero envelope) because
    # some texture classes — like horizontal stripes with row-wise
    # uniform colour — really do have a zero row spectrum. The
    # weighted average treats a zero-vs-zero comparison as 0 (not
    # NaN), so a degenerate axis only weakens coherence instead of
    # poisoning it.
    if n <= 1:
        coherence = 1.0
    else:
        sims = []
        for i in range(n):
            sims.append(_weighted_self_similarity(
                per_image_row[i], per_image_col[i], per_image_color[i],
                per_image_heat[i],
                mean_row, mean_col, mean_color, heat_norm))
        coherence = float(np.mean(sims))

    confidence = float(np.clip(coherence * min(n / 10.0, 1.0), 0.0, 1.0))

    return {
        "label": folder.name,
        "row_freq": row_freq,
        "col_freq": col_freq,
        "color_freq": color_freq,
        "position_heatmap": heat_norm,
        "coherence": float(np.clip(coherence, 0.0, 1.0)),
        "confidence": confidence,
        "activation_count": int(n),
        "variation_cluster_id": 0,
        "image_count": int(n),
    }


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine that returns 0 (not NaN) when either side has zero
    norm — matches what the .sfb matcher needs for textures with a
    degenerate axis."""
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _weighted_self_similarity(
        img_row, img_col, img_color, img_heat,
        mean_row, mean_col, mean_color, heat_norm) -> float:
    """Phase 3 weighted-cosine formula, fully tolerant of degenerate
    inputs. The four weights mirror the retrieval-time matcher:
    0.35 row + 0.35 col + 0.20 color + 0.10 heatmap.

    A degenerate axis (zero envelope) contributes 0 to the average —
    so a horizontal-stripe texture with zero row_freq still earns a
    high coherence from col_freq + heatmap agreement instead of
    collapsing to NaN."""
    row   = _safe_cosine(img_row, mean_row)
    col   = _safe_cosine(img_col, mean_col)
    color = (_safe_cosine(img_color[0], mean_color[0])
             + _safe_cosine(img_color[1], mean_color[1])
             + _safe_cosine(img_color[2], mean_color[2])) / 3.0
    heat  = _safe_cosine(img_heat, heat_norm)
    return 0.35 * row + 0.35 * col + 0.20 * color + 0.10 * heat


# ── Driver ────────────────────────────────────────────────────────

def _save_npz(motifs: list, out_path: Path) -> None:
    """Persist a list of motif dicts as a single .npz. Stored as
    aligned ndarrays (one row per motif) so downstream consumers can
    index by motif id without rebuilding dicts."""
    n = len(motifs)
    if n == 0:
        # Empty bank: write a 0-row file so the build script still
        # produces a valid (empty) .sfb without special-casing.
        np.savez(out_path,
                 labels=np.array([], dtype="U64"),
                 row_freq=np.zeros((0, FREQ_BINS), np.float32),
                 col_freq=np.zeros((0, FREQ_BINS), np.float32),
                 color_freq=np.zeros((0, 3, FREQ_BINS), np.float32),
                 position_heatmap=np.zeros((0, HEATMAP_DIM, HEATMAP_DIM), np.float32),
                 coherence=np.zeros((0,), np.float32),
                 confidence=np.zeros((0,), np.float32),
                 activation_count=np.zeros((0,), np.uint32),
                 variation_cluster_id=np.zeros((0,), np.uint16),
                 image_count=np.zeros((0,), np.uint32))
        return

    np.savez(
        out_path,
        labels=np.array([m["label"] for m in motifs], dtype="U64"),
        row_freq=np.stack([m["row_freq"] for m in motifs], axis=0),
        col_freq=np.stack([m["col_freq"] for m in motifs], axis=0),
        color_freq=np.stack([m["color_freq"] for m in motifs], axis=0),
        position_heatmap=np.stack([m["position_heatmap"] for m in motifs], axis=0),
        coherence=np.array([m["coherence"]  for m in motifs], np.float32),
        confidence=np.array([m["confidence"] for m in motifs], np.float32),
        activation_count=np.array([m["activation_count"] for m in motifs], np.uint32),
        variation_cluster_id=np.array(
            [m["variation_cluster_id"] for m in motifs], np.uint16),
        image_count=np.array([m["image_count"] for m in motifs], np.uint32),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="dataset/primitive root: one subfolder per motif label")
    ap.add_argument("--out",  required=True, type=Path,
                    help="output .npz path")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-folder timing and stats")
    args = ap.parse_args()

    if not args.data.is_dir():
        print(f"error: --data {args.data} is not a directory",
              file=sys.stderr)
        return 2

    folders = sorted(p for p in args.data.iterdir() if p.is_dir())
    motifs = []
    t_start = time.perf_counter()
    for folder in folders:
        t0 = time.perf_counter()
        m = train_motif_from_folder(folder)
        if m is None:
            if args.verbose:
                print(f"  skip {folder.name}: no images")
            continue
        motifs.append(m)
        if args.verbose:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  {folder.name:<32s} "
                  f"n={m['image_count']:>4d}  "
                  f"coh={m['coherence']:.3f}  "
                  f"conf={m['confidence']:.3f}  "
                  f"({dt:.1f} ms)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(motifs, args.out)
    dt_total = time.perf_counter() - t_start
    print(f"wrote {args.out}  ({len(motifs)} motifs from "
          f"{len(folders)} folders in {dt_total:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
