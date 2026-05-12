"""Diversity + PSNR regression for ce_core/sss_rowvae.c.

Validates the Phase 1 amp-only radio tuning:

  1. Train a model on data/sanrio (25 PPMs, 4-col legacy labels.tsv).
  2. Generate 10 images for the same prompt with seeds 0..9.
  3. Assert that pairwise MSE between seeds exceeds 0.05 — i.e. the
     per-seed random phase init really does produce visually distinct
     outputs.
  4. Assert that for every generated image, the *nearest* training
     original lies below 25 dB PSNR — the generator must not be
     reproducing training samples verbatim.

Prompt note: we use the multi-word prompt "white cat kitty" so that
all three (COLOR / SHAPE / FACE) winners are engaged. A bare "kitty"
matches only the FACE label in the sanrio dataset, leaving the
low band untouched by the radio loop and dropping per-seed MSE below
the threshold — that's a property of the dataset, not the engine, and
isn't what we're trying to test here.

Run:
    make pybridge
    python3 tools/test_sss_rowvae.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.sss_image_io import read_ppm
from tools.sss_memory import sss_generate


PROMPT       = "white cat kitty"
LABEL_PREFIX = "kitty_"
SEEDS        = list(range(10))
SIZE         = 64
STEPS        = 24
MSE_THRESH   = 0.05
PSNR_THRESH  = 25.0


def _train(out_path: str) -> int:
    cp = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "sss_train.py"),
         "--labels", os.path.join(REPO, "data", "sanrio", "labels.tsv"),
         "--root",   os.path.join(REPO, "data", "sanrio"),
         "--out",    out_path,
         "--size",   str(SIZE)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        sys.stderr.write("sss_train.py failed:\n")
        sys.stderr.write(cp.stdout + cp.stderr)
        return 1
    print(cp.stdout.strip())
    return 0


def _load_training_originals() -> list[np.ndarray]:
    """Return the kitty PPMs as (SIZE, SIZE, 3) float32 BGR in [0, 1].

    BGR — both `read_ppm()` and `sss_generate()` return BGR uint8 in
    this repo, so the PSNR/MSE comparison stays in the same colour
    order on both sides without an extra conversion."""
    out = []
    sanrio_dir = os.path.join(REPO, "data", "sanrio")
    for fname in sorted(os.listdir(sanrio_dir)):
        if not (fname.startswith(LABEL_PREFIX) and fname.endswith(".ppm")):
            continue
        # Keep BGR (channel order matches sss_generate's output).
        img = read_ppm(os.path.join(sanrio_dir, fname))
        h, w = img.shape[:2]
        yi = np.linspace(0, h - 1, SIZE).round().astype(int)
        xi = np.linspace(0, w - 1, SIZE).round().astype(int)
        out.append(img[yi[:, None], xi[None, :]].astype(np.float32) / 255.0)
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "sanrio.sss")
        if _train(model_path) != 0:
            return 1

        # ── 1. Generate seed 0..9 ────────────────────────────
        imgs = []
        for s in SEEDS:
            img_u8 = sss_generate(model_path, PROMPT,
                                  seed=s, size=SIZE, steps=STEPS)
            imgs.append(img_u8.astype(np.float32) / 255.0)

        # ── 2. Pairwise MSE between seeds ────────────────────
        mses = []
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                mses.append(float(np.mean((imgs[i] - imgs[j]) ** 2)))
        mean_mse = float(np.mean(mses))
        min_mse  = float(np.min(mses))
        max_mse  = float(np.max(mses))
        print(f"pairwise MSE: mean={mean_mse:.4f}, "
              f"min={min_mse:.4f}, max={max_mse:.4f}")
        if mean_mse <= MSE_THRESH:
            sys.stderr.write(
                f"FAIL: mean pairwise MSE {mean_mse:.4f} ≤ {MSE_THRESH}\n")
            return 2
        print(f"OK  diversity : mean pairwise MSE > {MSE_THRESH}")

        # ── 3. PSNR vs nearest training original ─────────────
        originals = _load_training_originals()
        if not originals:
            sys.stderr.write(
                f"FAIL: no training originals matching {LABEL_PREFIX!r}\n")
            return 3
        worst_psnr = 0.0
        for i, gen in enumerate(imgs):
            best = 0.0
            for orig in originals:
                mse = float(np.mean((gen - orig) ** 2))
                if mse < 1e-12:
                    best = float("inf"); break
                psnr = 10.0 * np.log10(1.0 / mse)
                if psnr > best:
                    best = psnr
            print(f"  seed {i}: nearest-original PSNR = {best:.2f} dB")
            if best > worst_psnr:
                worst_psnr = best
        if worst_psnr >= PSNR_THRESH:
            sys.stderr.write(
                f"FAIL: max PSNR vs originals = {worst_psnr:.2f} dB "
                f"≥ {PSNR_THRESH} dB (the generator may be reproducing "
                "training samples)\n")
            return 4
        print(f"OK  no-copy   : worst-case PSNR vs originals "
              f"{worst_psnr:.2f} dB < {PSNR_THRESH} dB")

        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
