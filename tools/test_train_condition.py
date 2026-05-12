"""Regression for scripts/sss_train_condition.

Synthesises five condition signals and verifies the Phase 4 classifier
routes each one correctly. The headline check is *gravity*: a smooth
linear drift whose adjacent-frame L2 difference falls below the
frame-delta threshold and should therefore land in CONTINUOUS without
ever reaching the temporal-FFT branch (the GPT-validated fix).

Synthesised signals (16 frames @ 64×64 RGB):

  wind        — oscillatory: stripes drift sideways with sinusoidal
                amplitude over t → strong non-DC peak in temporal FFT.
  pulse       — impulse: frame 0 is bright noise, every later frame
                is near-zero.
  static      — continuous: every frame is identical → frame-delta 0.
  gravity     — continuous (smooth linear drift): every frame is the
                previous frame plus a tiny constant offset → frame
                delta < 0.01.
  vibration   — oscillatory: every pixel oscillates with a fast
                temporal sine → strong non-DC peak.

Run:
    python3 tools/test_train_condition.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import sss_image_io                                # noqa: E402
from scripts.sss_train_condition import (                     # noqa: E402
    CONDITION_CONTINUOUS, CONDITION_IMPULSE, CONDITION_OSCILLATORY,
    FRAMES_DEFAULT, classify_signal, train_condition_from_folder,
)


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# ── Synthetic generators (T, 64, 64, 3 uint8) ─────────────────────

def _wind(seed: int, T: int = 16) -> np.ndarray:
    """Vertical stripes drifting sideways with a sinusoidal envelope
    in t. Classifier should land on OSCILLATORY."""
    rng = np.random.default_rng(seed)
    H = W = 64
    base = np.zeros((H, W, 3), np.float32)
    for x in range(W):
        v = 128 + 80 * np.cos(2 * np.pi * x / 8)
        base[:, x, :] = v
    frames = []
    for t in range(T):
        amp = 0.5 + 0.5 * np.cos(2 * np.pi * t / 8)
        shift = int(round(amp * 8))
        f = np.roll(base, shift, axis=1).copy()
        f += rng.normal(0, 4, size=f.shape).astype(np.float32)
        frames.append(_u8(f))
    return np.stack(frames, axis=0)


def _pulse(seed: int, T: int = 16) -> np.ndarray:
    """Frame 0 is bright noise; later frames are dark + faint
    residual decay. Classifier should pick IMPULSE."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(T):
        amp = 200 if t == 0 else int(200 * np.exp(-t * 0.7))
        f = rng.integers(0, amp + 1, size=(64, 64, 3))
        frames.append(_u8(f))
    return np.stack(frames, axis=0)


def _static(seed: int, T: int = 16) -> np.ndarray:
    """Single random texture cloned T times. Frame-delta ≈ 0 →
    CONTINUOUS. JPEG-style per-frame jitter is suppressed so the
    classifier exercises the early-out path explicitly."""
    rng = np.random.default_rng(seed)
    f = rng.integers(60, 200, size=(64, 64, 3))
    f = _u8(f)
    return np.stack([f] * T, axis=0)


def _gravity(seed: int, T: int = 16) -> np.ndarray:
    """A very gentle linear drift — barely visible — so the classifier
    must take the GPT-validated frame-delta early-out
    (FRAME_DELTA_THRESH = 0.01) instead of running through the temporal
    FFT path. The contract being tested: a slow, monotonic motion is
    indistinguishable from "static with imperceptible drift" at the
    classifier level, and should be tagged CONTINUOUS rather than
    routed as a periodic signal that doesn't exist.

    Per-frame drift is small enough that mean per-pixel L2 stays
    below 0.01 (~1/255 in 8-bit pixel units). The image otherwise
    looks like the static motif — exactly the scenario the GPT
    verification flagged when gravity was being mis-routed."""
    rng = np.random.default_rng(seed)
    # Low contrast → small absolute pixel differences as it drifts.
    base = rng.integers(100, 130, size=(64, 64, 3)).astype(np.float32)
    frames = []
    for t in range(T):
        # ~0.02 px / frame shift — far below the threshold's reach.
        shift_px = 0.02 * t
        yi = np.arange(64, dtype=np.float32) - shift_px
        yi = np.clip(yi, 0, 63)
        lo = np.floor(yi).astype(np.int64)
        hi = np.minimum(lo + 1, 63)
        frac = (yi - lo)[:, None, None]
        f = base[lo] * (1 - frac) + base[hi] * frac
        # Imperceptible noise floor.
        f = f + rng.normal(0, 0.1, size=f.shape).astype(np.float32)
        frames.append(_u8(f))
    return np.stack(frames, axis=0)


def _vibration(seed: int, T: int = 16) -> np.ndarray:
    """Every pixel oscillates with a fast temporal sine, no spatial
    pattern variation. Classifier should pick OSCILLATORY."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 160, size=(64, 64, 3)).astype(np.float32)
    frames = []
    for t in range(T):
        amp = 40 * np.sin(2 * np.pi * t / 4)
        f = base + amp
        frames.append(_u8(f))
    return np.stack(frames, axis=0)


GENERATORS = {
    "wind":      (_wind,      CONDITION_OSCILLATORY),
    "pulse":     (_pulse,     CONDITION_IMPULSE),
    "static":    (_static,    CONDITION_CONTINUOUS),
    "gravity":   (_gravity,   CONDITION_CONTINUOUS),
    "vibration": (_vibration, CONDITION_OSCILLATORY),
}


# ── Filesystem layout helper ─────────────────────────────────────

def _build_dataset(root: str, seed: int = 0) -> None:
    """Write each (label, sequence) as `root/<label>/seq_NN/frame_NN.ppm`
    so the trainer's _scan_sequences() picks them up. Two sequences per
    label so the trainer exercises folder-averaging."""
    for label, (gen, _) in GENERATORS.items():
        for seq_idx in range(2):
            seq_dir = os.path.join(root, label, f"seq_{seq_idx:02d}")
            os.makedirs(seq_dir, exist_ok=True)
            stack = gen(seed + seq_idx)
            for t, frame in enumerate(stack):
                # sss_image_io.write_ppm expects BGR; flip channels.
                sss_image_io.write_ppm(
                    os.path.join(seq_dir, f"frame_{t:03d}.ppm"),
                    frame[..., ::-1])


# ── Tests ─────────────────────────────────────────────────────────

def _condition_label(ct: int) -> str:
    return {CONDITION_CONTINUOUS: "continuous",
            CONDITION_IMPULSE:    "impulse",
            CONDITION_OSCILLATORY: "oscillatory"}[ct]


def test_classifier_directly() -> None:
    """Exercise classify_signal() on every generator's first
    sequence — independent of folder walking + filesystem I/O."""
    print("classifier on raw float32 sequences:")
    for label, (gen, expected) in GENERATORS.items():
        seq = gen(0).astype(np.float32) / 255.0
        got, delta = classify_signal(seq)
        ok = "OK " if got == expected else "FAIL"
        print(f"  {ok} {label:<12s} Δ={delta:.4f} "
              f"got={_condition_label(got)} "
              f"want={_condition_label(expected)}")
        assert got == expected, \
            f"{label}: classifier returned {_condition_label(got)}, " \
            f"expected {_condition_label(expected)}"
    print("OK  classifier  : 5/5 signals routed correctly "
          "(gravity uses frame-delta early-out)")


def test_trainer_end_to_end() -> None:
    """Run the full trainer (CLI subprocess) over a synthetic
    dataset and check every produced condition has the expected
    type + a non-trivial envelope."""
    with tempfile.TemporaryDirectory() as tmp:
        data_root = os.path.join(tmp, "conditions")
        out_path  = os.path.join(tmp, "conditions.npz")
        _build_dataset(data_root, seed=0)

        cp = subprocess.run([
            sys.executable, os.path.join(REPO, "scripts",
                                         "sss_train_condition.py"),
            "--data", data_root, "--out", out_path, "--verbose",
        ], capture_output=True, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stdout + cp.stderr)
            raise SystemExit(1)
        print(cp.stdout.strip())

        npz = np.load(out_path, allow_pickle=False)
        labels = list(npz["labels"])
        assert len(labels) == 5, f"expected 5 conditions, got {len(labels)}"

        for i, lab in enumerate(labels):
            got = int(npz["condition_type"][i])
            _, want = GENERATORS[lab]
            assert got == want, (
                f"{lab}: trainer wrote type={_condition_label(got)}, "
                f"expected {_condition_label(want)}")

            # Sanity: oscillatory / impulse signals should produce a
            # non-zero temporal_freq envelope; continuous can be zero.
            t_norm = float(np.linalg.norm(npz["temporal_freq"][i]))
            if want != CONDITION_CONTINUOUS:
                assert t_norm > 1e-3, \
                    f"{lab}: expected non-trivial temporal envelope, " \
                    f"got |t| = {t_norm:.6f}"

            print(f"  {lab:<12s} type={_condition_label(got)}  "
                  f"|temp|={t_norm:.3f}  "
                  f"dir={float(npz['direction'][i]):+.3f} "
                  f"intensity={float(npz['default_intensity'][i]):.3f}")

        print("OK  trainer     : all 5 conditions classified + envelopes "
              "carry non-trivial spectral content for non-CONTINUOUS")


def main() -> int:
    test_classifier_directly()
    test_trainer_end_to_end()
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
