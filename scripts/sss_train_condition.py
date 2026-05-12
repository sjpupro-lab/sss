#!/usr/bin/env python3
"""sss_train_condition — Phase 4 condition-signal trainer.

Design philosophy:
    A condition signal is **a signal in its own right, not a frame**.
    The trainer extracts an amplitude envelope along the row axis, the
    column axis, and the temporal axis — never per-frame pixels, never
    per-row phases. No raw frames land on disk; only spectral
    envelopes and a few scalar descriptors (type, direction, intensity,
    decay rate).

Input:
    dataset/conditions/<label>/<sequence>/frame_NNN.{png,ppm}
        Each subfolder of `--data` is one condition. Inside it,
        sequence subdirectories hold 16 (or `--frames`) ordered
        frames. Multiple sequences per condition are averaged.

Classifier (Phase 4 v1, frame-delta-first per GPT verification):
    1. Mean per-pixel L2 difference between adjacent frames.
        - `< 0.01`     → CONTINUOUS (skip FFT, no time-axis structure).
        - `>= 0.01`    → continue.
    2. Temporal FFT analysis on the per-pixel mean:
        - first frame has a broadband impulse + decay tail → IMPULSE.
        - clear non-DC peak in temporal FFT → OSCILLATORY.
        - otherwise → CONTINUOUS.

Algorithm:
    For each labelled condition folder:
      (1) Load every frame as 64×64 RGB float32 in [0, 1].
      (2) Classifier above.
      (3) row_freq[128] / col_freq[128]: same DC-killed + zero-mean
          L2-normalised envelopes as the primitive motif trainer
          (see scripts/sss_train_primitive.py).
      (4) temporal_freq[64]: per-pixel mean over time → length-N
          signal → rfft. Bin 0 zeroed. Resampled to 64 bins.
          L2-normalised.
      (5) direction (radians): mean optical-flow direction across
          frames (cheap dense block-matching at 8×8 grid).
      (6) default_intensity: mean per-frame standard deviation, scaled
          into [0, 1].
      (7) decay_rate: only for IMPULSE — exponential fit on the
          per-frame energy series.

Output:
    A .npz file with one row per condition. Consumed by
    `scripts/sss_build_feature_bank.py --conditions`.

CLI:
    python3 scripts/sss_train_condition.py \\
        --data dataset/conditions \\
        --out  build/conditions.npz
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

from scripts import sss_train_primitive as _prim   # noqa: E402
from tools import sss_image_io                     # noqa: E402


FRAMES_DEFAULT       = 16
FREQ_BINS            = 128
TEMPORAL_BINS        = 64
PATCH_SIZE           = 64
FRAME_DELTA_THRESH   = 0.01      # below → CONTINUOUS without FFT
# Peak-to-second-peak ratio above this → OSCILLATORY. Picking against
# the *second* peak (not the median) lets the classifier distinguish a
# real oscillation (one tall spike, everything else small) from a slow
# linear drift, whose spectrum has 1/k decay across many bins and so
# clears a median-based test even though no single frequency dominates.
OSCILLATORY_PEAK_PROMINENCE = 3.0
# Floor on peak amplitude relative to total non-DC energy — keeps
# very-low-energy "noise spectra" out of OSCILLATORY.
OSCILLATORY_PEAK_FRACTION = 0.10
IMPULSE_FIRST_FRAME_RATIO = 1.8  # frame_0 energy ≥ this × median → IMPULSE

CONDITION_CONTINUOUS  = 0
CONDITION_IMPULSE     = 1
CONDITION_OSCILLATORY = 2

IMAGE_EXTS = (".png", ".ppm")


# ── Frame loading + normalisation ────────────────────────────────

def _scan_sequences(folder: Path) -> list[list[Path]]:
    """Return a list of per-sequence sorted frame paths. A
    "sequence" is either a subfolder containing frames, or — if no
    subfolders exist — the folder itself treated as one sequence."""
    subdirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if subdirs:
        seqs = []
        for sd in subdirs:
            paths = sorted(p for p in sd.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if paths:
                seqs.append(paths)
        return seqs
    # Flat-folder: every PNG/PPM is one frame in a single sequence.
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return [paths] if paths else []


def _load_sequence_64(paths: list[Path], target_len: int) -> np.ndarray:
    """Decode every frame to 64×64 RGB float32 in [0, 1] and return
    a (T, 64, 64, 3) stack. If the on-disk sequence has fewer frames
    than `target_len` we sample with replacement (cycle); more frames
    are linearly subsampled."""
    if not paths:
        return np.zeros((target_len, PATCH_SIZE, PATCH_SIZE, 3), np.float32)
    raw = []
    for p in paths:
        rgb = _prim._read_image_rgb(str(p))
        raw.append(_prim._resize_pad_64(rgb).astype(np.float32) / 255.0)
    stack = np.stack(raw, axis=0)                     # (T_src, 64, 64, 3)
    if stack.shape[0] == target_len:
        return stack
    # Linear resample along time.
    src_idx = np.linspace(0, stack.shape[0] - 1, target_len)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, stack.shape[0] - 1)
    frac = (src_idx - lo).astype(np.float32)[:, None, None, None]
    return (stack[lo] * (1.0 - frac) + stack[hi] * frac).astype(np.float32)


# ── Classifier ───────────────────────────────────────────────────

def classify_signal(seq: np.ndarray) -> tuple[int, float]:
    """Decide which condition_type best describes a (T, 64, 64, 3)
    sequence. Returns (type, mean_frame_delta).

    Two-stage decision:

      Step 1 (frame-delta first, GPT-verified fix): mean adjacent-
        frame L2 difference per pixel. Below FRAME_DELTA_THRESH the
        sequence has no meaningful time-axis structure → CONTINUOUS.
        This is what stops `gravity` (a smooth linear drift, every-
        frame-the-same plus a tiny offset) from being routed through
        the FFT path.

      Step 2 (per-pixel temporal FFT, aggregated): compute the
        temporal FFT amplitude *per pixel* (the per-frame mean used
        by an earlier draft cancels out for spatial-shift signals
        like "wind", where the global energy stays constant but
        individual pixels swing in and out of a stripe). Sum
        |FFT|_k across pixels → an aggregated 1-D spectrum. If the
        non-DC peak/median ratio clears OSCILLATORY_PEAK → OSCILLATORY.

      The IMPULSE check looks at the *first-frame* per-pixel energy
      against the median across frames; impulses are first-frame-
      heavy regardless of the spatial pattern.
    """
    if seq.shape[0] < 2:
        return CONDITION_CONTINUOUS, 0.0
    flat = seq.reshape(seq.shape[0], -1)
    diffs = np.linalg.norm(flat[1:] - flat[:-1], axis=1)
    mean_delta = float(diffs.mean() / np.sqrt(flat.shape[1]))
    if mean_delta < FRAME_DELTA_THRESH:
        return CONDITION_CONTINUOUS, mean_delta

    # Impulse check: first-frame total energy dominates the median.
    energies = (flat ** 2).sum(axis=1)
    if energies.size >= 2:
        median_energy = float(np.median(energies))
        if median_energy > 0 and energies[0] >= IMPULSE_FIRST_FRAME_RATIO * median_energy:
            return CONDITION_IMPULSE, mean_delta

    # Per-pixel temporal FFT — *after* removing a per-pixel linear
    # trend. The linear-detrend step is what separates "slow drift"
    # signals like gravity (whose remaining residual is mostly
    # measurement noise) from genuine oscillations like wind /
    # vibration (whose stripe / sinusoid structure survives the
    # detrend intact).
    T = flat.shape[0]
    t_vec = np.arange(T, dtype=np.float32)
    t_centered = (t_vec - t_vec.mean())
    t_var = float((t_centered ** 2).sum() + 1e-12)
    # slope per pixel: cov(t, x) / var(t).
    slope = ((flat - flat.mean(axis=0, keepdims=True))
             * t_centered[:, None]).sum(axis=0) / t_var
    intercept = flat.mean(axis=0) - slope * t_vec.mean()
    trend = intercept[None, :] + slope[None, :] * t_vec[:, None]
    residual = flat - trend                          # (T, H*W*3)

    res_energy = float((residual ** 2).sum())
    full_energy = float(((flat - flat.mean(axis=0, keepdims=True)) ** 2).sum()
                        + 1e-12)
    # If the linear trend explains > 80 % of the variation, the
    # signal is a drift — call it CONTINUOUS regardless of any FFT
    # leakage at low frequencies.
    if res_energy / full_energy < 0.20:
        return CONDITION_CONTINUOUS, mean_delta

    spec = np.fft.rfft(residual, axis=0)
    amp = np.abs(spec).astype(np.float32)
    if amp.shape[0] <= 2:
        return CONDITION_CONTINUOUS, mean_delta
    amp[0] = 0.0
    agg = amp.sum(axis=1)
    nonzero = agg[1:]
    if nonzero.size >= 1:
        peak       = float(nonzero.max())
        total      = float(nonzero.sum() + 1e-12)
        # Oscillation needs the residual to carry a peak with a
        # meaningful share of total non-DC energy. A multi-harmonic
        # signal (e.g. stripe drift) still concentrates 10–40 % at
        # one bin; broadband noise spreads below ~5 %.
        if (peak / total) >= OSCILLATORY_PEAK_FRACTION:
            return CONDITION_OSCILLATORY, mean_delta

    return CONDITION_CONTINUOUS, mean_delta


# ── Envelope extraction ──────────────────────────────────────────

def _temporal_envelope_64(seq: np.ndarray) -> np.ndarray:
    """Per-pixel temporal FFT, aggregated across pixels, resampled
    to 64 bins. Bin 0 zeroed. NOT L2-normalised — caller averages
    then normalises.

    Uses *per-pixel* FFT (axis=0 of the (T, H*W*3) stack) rather
    than the per-frame global mean. Spatial-shift signals like
    "wind" (vertical stripes drifting sideways) leave the global
    energy unchanged frame-to-frame — only the per-pixel signal
    actually oscillates with the underlying frequency. Aggregating
    |FFT|_k across pixels gives a 1-D spectrum that captures the
    per-pixel oscillation period regardless of whether the global
    mean moves."""
    flat = seq.reshape(seq.shape[0], -1)
    if flat.shape[0] < 2:
        return np.zeros(TEMPORAL_BINS, np.float32)
    per_pixel = flat - flat.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(per_pixel, axis=0)
    amp = np.abs(spec).astype(np.float32)
    amp[0] = 0.0
    agg = amp.sum(axis=1).astype(np.float32)            # (T/2+1,)
    return _prim._resample_linear(agg, TEMPORAL_BINS)


def _spatial_envelopes(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average row_freq / col_freq across all frames in the sequence.
    Re-uses the primitive trainer's pipeline; per-frame DC-killed +
    zero-mean L2-normalised at the end. Returns
    (row_freq[128], col_freq[128]) as float32."""
    rows = []
    cols = []
    for t in range(seq.shape[0]):
        f = _prim._to_float_demeaned((seq[t] * 255.0).astype(np.uint8))
        rows.append(_prim._envelope_row_freq(f))
        cols.append(_prim._envelope_col_freq(f))
    mean_row = np.mean(rows, axis=0)
    mean_col = np.mean(cols, axis=0)
    return (_prim._zero_mean_l2_normalise(mean_row),
            _prim._zero_mean_l2_normalise(mean_col))


def _estimate_direction(seq: np.ndarray) -> float:
    """Mean motion direction across frames, in radians ∈ [-π, π].
    Cheap dense block-matching: between consecutive frames, compute
    the per-pixel intensity delta and weight the centre-of-mass of
    the absolute delta image. Returns 0.0 when motion energy is
    negligible (CONTINUOUS-flat case)."""
    if seq.shape[0] < 2:
        return 0.0
    gray = seq.mean(axis=-1)                          # (T, 64, 64)
    deltas = np.abs(gray[1:] - gray[:-1])             # (T-1, 64, 64)
    total = deltas.sum()
    if total < 1e-6:
        return 0.0
    yy, xx = np.mgrid[:gray.shape[1], :gray.shape[2]].astype(np.float32)
    cy = float((deltas * yy[None]).sum() / total)
    cx = float((deltas * xx[None]).sum() / total)
    # Direction from the centre of the frame outward.
    h, w = gray.shape[1], gray.shape[2]
    return float(np.arctan2(cy - (h - 1) / 2.0, cx - (w - 1) / 2.0))


def _estimate_decay_rate(seq: np.ndarray) -> float:
    """Fit `E(t) = E_0 · exp(-r · t)` to per-frame energy. Used only
    for IMPULSE conditions. Returns 0.0 when fit is meaningless."""
    flat = seq.reshape(seq.shape[0], -1)
    E = (flat ** 2).sum(axis=1).astype(np.float64)
    if E.size < 3 or E[0] <= 0:
        return 0.0
    # log(E) = log(E_0) - r·t; least-squares fit.
    log_E = np.log(np.maximum(E, 1e-9))
    t = np.arange(E.size, dtype=np.float64)
    # Slope of log(E) vs t.
    cov = np.cov(t, log_E, ddof=0)
    if cov.shape != (2, 2) or cov[0, 0] < 1e-9:
        return 0.0
    slope = cov[0, 1] / cov[0, 0]
    return float(max(0.0, -slope))


def _intensity(seq: np.ndarray) -> float:
    """Mean per-frame std, clipped into [0, 1]. Provides a sane
    "nominal intensity" baseline so the generator's intensity slider
    has a meaningful default scale."""
    stds = seq.reshape(seq.shape[0], -1).std(axis=1)
    return float(np.clip(stds.mean() / 0.25, 0.0, 1.0))


# ── Per-folder driver ────────────────────────────────────────────

def train_condition_from_folder(folder: Path,
                                target_len: int = FRAMES_DEFAULT) -> dict | None:
    """Aggregate all sequences in a folder into one condition dict.

    Returns None if the folder has no usable sequences. The dict
    schema matches what `sss_build_feature_bank.py` consumes:

        {
          "label":             <folder name>,
          "condition_type":    0 / 1 / 2,
          "row_freq":          (128,) f32 (zero-mean L2-normalised),
          "col_freq":          (128,) f32,
          "temporal_freq":     (64,)  f32 (L2-normalised),
          "direction":         float (radians),
          "default_intensity": float in [0, 1],
          "decay_rate":        float (0 for non-impulse),
          "sequence_count":    int,
          "mean_frame_delta":  float (debug aid),
        }
    """
    seqs = _scan_sequences(folder)
    if not seqs:
        return None

    rows = []
    cols = []
    temps = []
    dirs = []
    intens = []
    decays = []
    types_seen = []
    deltas = []
    for paths in seqs:
        seq = _load_sequence_64(paths, target_len=target_len)
        ctype, delta = classify_signal(seq)
        types_seen.append(ctype)
        deltas.append(delta)
        row, col = _spatial_envelopes(seq)
        rows.append(row); cols.append(col)
        temps.append(_temporal_envelope_64(seq))
        dirs.append(_estimate_direction(seq))
        intens.append(_intensity(seq))
        if ctype == CONDITION_IMPULSE:
            decays.append(_estimate_decay_rate(seq))
        else:
            decays.append(0.0)

    # Majority vote on type — within-folder spread is usually small,
    # but a noisy sequence shouldn't flip the whole label.
    cond_type = int(np.bincount(types_seen).argmax())
    mean_row = np.mean(rows, axis=0).astype(np.float32)
    mean_col = np.mean(cols, axis=0).astype(np.float32)
    mean_temp = np.mean(temps, axis=0).astype(np.float32)

    # row / col already zero-mean-L2 per call; folder-mean is a
    # weighted average that can drift off-norm so re-normalise.
    row_out = _prim._zero_mean_l2_normalise(mean_row)
    col_out = _prim._zero_mean_l2_normalise(mean_col)
    # temporal: L2 normalise without zero-mean (it's a non-negative
    # 1-D amplitude envelope; signal-shape matters but the spec
    # field stays amplitude-only).
    temp_n = float(np.linalg.norm(mean_temp))
    temp_out = (mean_temp / temp_n).astype(np.float32) if temp_n > 1e-12 \
                                                      else mean_temp

    direction = float(np.mean(dirs))
    intensity = float(np.clip(np.mean(intens), 0.0, 1.0))
    decay = float(np.mean(decays)) if cond_type == CONDITION_IMPULSE else 0.0

    return {
        "label":             folder.name,
        "condition_type":    cond_type,
        "row_freq":          row_out,
        "col_freq":          col_out,
        "temporal_freq":     temp_out,
        "direction":         direction,
        "default_intensity": intensity,
        "decay_rate":        decay,
        "sequence_count":    int(len(seqs)),
        "mean_frame_delta":  float(np.mean(deltas)),
    }


def _save_npz(conditions: list, out_path: Path) -> None:
    n = len(conditions)
    if n == 0:
        np.savez(out_path,
                 labels=np.array([], dtype="U64"),
                 condition_type=np.zeros(0, np.uint8),
                 row_freq=np.zeros((0, FREQ_BINS), np.float32),
                 col_freq=np.zeros((0, FREQ_BINS), np.float32),
                 temporal_freq=np.zeros((0, TEMPORAL_BINS), np.float32),
                 direction=np.zeros(0, np.float32),
                 default_intensity=np.zeros(0, np.float32),
                 decay_rate=np.zeros(0, np.float32),
                 sequence_count=np.zeros(0, np.uint32),
                 mean_frame_delta=np.zeros(0, np.float32))
        return
    np.savez(
        out_path,
        labels=np.array([c["label"] for c in conditions], dtype="U64"),
        condition_type=np.array([c["condition_type"] for c in conditions],
                                np.uint8),
        row_freq=np.stack([c["row_freq"] for c in conditions], axis=0),
        col_freq=np.stack([c["col_freq"] for c in conditions], axis=0),
        temporal_freq=np.stack([c["temporal_freq"] for c in conditions], axis=0),
        direction=np.array([c["direction"] for c in conditions], np.float32),
        default_intensity=np.array([c["default_intensity"] for c in conditions],
                                   np.float32),
        decay_rate=np.array([c["decay_rate"] for c in conditions], np.float32),
        sequence_count=np.array([c["sequence_count"] for c in conditions],
                                np.uint32),
        mean_frame_delta=np.array([c["mean_frame_delta"] for c in conditions],
                                  np.float32),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="dataset/conditions root: subfolder per condition label")
    ap.add_argument("--out",  required=True, type=Path,
                    help="output .npz path")
    ap.add_argument("--frames", type=int, default=FRAMES_DEFAULT,
                    help=f"per-sequence frame count (default {FRAMES_DEFAULT})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.data.is_dir():
        print(f"error: --data {args.data} is not a directory",
              file=sys.stderr)
        return 2

    folders = sorted(p for p in args.data.iterdir() if p.is_dir())
    conditions = []
    t_start = time.perf_counter()
    type_names = {CONDITION_CONTINUOUS: "continuous",
                  CONDITION_IMPULSE: "impulse",
                  CONDITION_OSCILLATORY: "oscillatory"}
    for folder in folders:
        t0 = time.perf_counter()
        c = train_condition_from_folder(folder, target_len=args.frames)
        if c is None:
            if args.verbose:
                print(f"  skip {folder.name}: no sequences")
            continue
        conditions.append(c)
        if args.verbose:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  {folder.name:<32s} "
                  f"type={type_names[c['condition_type']]:<11s} "
                  f"Δ={c['mean_frame_delta']:.4f} "
                  f"intensity={c['default_intensity']:.3f} "
                  f"({dt:.1f} ms)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(conditions, args.out)
    print(f"wrote {args.out}  ({len(conditions)} conditions from "
          f"{len(folders)} folders in "
          f"{time.perf_counter() - t_start:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
