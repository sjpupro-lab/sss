"""Regression for the FFT-level phase-ablation finding.

Origin: the Phase 1 design decision was justified by an offline GPT
verification run that compared three phase regimes on the same
amplitude spectrum:

    A) original amp + original phase   → PSNR 321.7 dB (essentially
                                          a perfect round-trip)
    B) original amp + phase = 0        → PSNR 10.56 dB  (deterministic
                                          but unrelated to the original)
    C) original amp + random phase     → PSNR 10.34 dB; 10-way mean
                                          pairwise MSE 0.0852 → seed
                                          diversity without amp change

This regression locks in (C). It deliberately does NOT exercise the
full sss_generate path — the goal is to prove the *FFT-level*
invariant the engine now relies on: given a fixed amp envelope, a
distinct uniform[-π, π) phase per seed produces visually distinct
inverse images. If this ever drops below 0.05 mean pairwise MSE,
the generator's per-seed diversity guarantee is in trouble.

Threshold: 0.05 (≈ 60 % of the measured 0.0852, safety margin).

Run:
    python3 tools/test_phase_ablation_regression.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.sss_image_io import read_ppm


SEED_COUNT  = 10
MSE_THRESH  = 0.05  # 60 % of GPT-measured 0.0852


def _load_reference() -> np.ndarray:
    """Pick the first PPM under `data/sanrio` (sorted lexicographically
    — currently a keroppi sample) and resize to 64×64 as float32 in
    [0, 1]. The identity of the image doesn't matter for the FFT-level
    ablation: any non-trivial reference produces the same amp-only
    diversity claim. BGR/RGB also doesn't matter — both sides of every
    MSE go through the same channel order."""
    candidates = []
    sanrio_dir = os.path.join(REPO, "data", "sanrio")
    for fname in sorted(os.listdir(sanrio_dir)):
        if fname.endswith(".ppm"):
            candidates.append(os.path.join(sanrio_dir, fname))
    if not candidates:
        raise SystemExit("no PPMs under data/sanrio for ablation reference")
    img = read_ppm(candidates[0])
    h, w = img.shape[:2]
    yi = np.linspace(0, h - 1, 64).round().astype(int)
    xi = np.linspace(0, w - 1, 64).round().astype(int)
    return (img[yi[:, None], xi[None, :]].astype(np.float32) / 255.0)


def _amp_only_inverse(amp: np.ndarray, seed: int) -> np.ndarray:
    """Round-trip an image's amp spectrum with a per-seed random phase.

    amp shape: (H, NF, 3) — the same layout the trainer / radio path
    uses. Returns a (H, W, 3) float32 image in [0, 1] after clipping."""
    rng = np.random.default_rng(seed)
    H, NF, C = amp.shape
    W = (NF - 1) * 2
    phase = rng.uniform(-np.pi, np.pi, size=amp.shape).astype(np.float32)
    spec  = amp * np.exp(1j * phase)
    img   = np.fft.irfft(spec, n=W, axis=1).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def main() -> int:
    ref = _load_reference()
    # Per-row rfft (axis=1), matching sss_ingest._block_fft / sss_train.row_fft.
    spec  = np.fft.rfft(ref, axis=1)
    amp   = np.abs(spec).astype(np.float32)
    orig_phase = np.angle(spec).astype(np.float32)

    # ── Scenario A: original amp + original phase ─────────────
    recon_A = np.fft.irfft(amp * np.exp(1j * orig_phase),
                           n=ref.shape[1], axis=1).astype(np.float32)
    mse_A = float(np.mean((ref - recon_A) ** 2))
    psnr_A = float("inf") if mse_A < 1e-12 else 10.0 * np.log10(1.0 / mse_A)
    print(f"A) amp+orig_phase : PSNR = {psnr_A:.2f} dB "
          f"(should be ≫ 100; exact round-trip)")
    assert psnr_A > 100.0, f"FAIL A: PSNR {psnr_A:.2f} dB"

    # ── Scenario B: amp + zero phase ──────────────────────────
    recon_B = np.fft.irfft(amp + 0j, n=ref.shape[1], axis=1).astype(np.float32)
    mse_B = float(np.mean((ref - np.clip(recon_B, 0, 1)) ** 2))
    psnr_B = float("inf") if mse_B < 1e-12 else 10.0 * np.log10(1.0 / mse_B)
    print(f"B) amp+zero_phase : PSNR = {psnr_B:.2f} dB "
          f"(should be far below A; deterministic but unrelated)")
    assert psnr_B < 30.0, f"FAIL B: PSNR {psnr_B:.2f} dB unexpectedly high"

    # ── Scenario C: amp + random phase per seed ───────────────
    imgs = [_amp_only_inverse(amp, seed) for seed in range(SEED_COUNT)]
    mses = []
    for i in range(SEED_COUNT):
        for j in range(i + 1, SEED_COUNT):
            mses.append(float(np.mean((imgs[i] - imgs[j]) ** 2)))
    mean_mse = float(np.mean(mses))
    min_mse  = float(np.min(mses))
    max_mse  = float(np.max(mses))
    print(f"C) amp+rand_phase : pairwise MSE mean={mean_mse:.4f}, "
          f"min={min_mse:.4f}, max={max_mse:.4f}")
    if mean_mse <= MSE_THRESH:
        sys.stderr.write(
            f"FAIL C: mean pairwise MSE {mean_mse:.4f} ≤ {MSE_THRESH}; "
            "per-seed phase diversity has regressed.\n")
        return 2
    print(f"OK   diversity   : mean pairwise MSE > {MSE_THRESH}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
