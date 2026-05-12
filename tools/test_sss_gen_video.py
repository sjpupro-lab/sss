"""End-to-end test for the Phase 5 unified `sss_gen` entry point.

Covers both the C binary (`build/sss_gen`) and the Python wrapper
(`scripts/sss_gen.py`):

  1. C path, frames = 1: writes one PPM matching the documented
     dimensions.
  2. C path, frames = 4: writes 4 PPMs under --out-dir; consecutive
     frames have non-zero pixel difference (the seed-variation
     baseline must vary frame-to-frame).
  3. C path rejects --condition / --feature-bank / --atmos-from with
     exit code 7 and points at scripts/sss_gen.py.
  4. Python path on `.sfb` + `--condition wind 0.0` vs `--condition
     wind 0.5`: output L2 difference > 0.3 (Phase 4 condition-
     intensity verification — high enough to confirm the response is
     applied, low enough to allow for Phase 1 per-seed jitter).
  5. Python path on `.sfb` with --frames 8: writes 8 PPMs and
     consecutive frames differ.

Run:
    make pybridge && make sss_gen
    python3 tools/test_sss_gen_video.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import sss_image_io                             # noqa: E402
from tools.sss_feature_bank import (                       # noqa: E402
    SSSFeatureBank, SSSMotif, SSSCondition, SSSInteractionResponse,
    CONDITION_OSCILLATORY,
    SFB_FREQ_BINS, SFB_TEMPORAL_BINS, SFB_WARP_DIM, SFB_HEATMAP_DIM,
)


C_BIN = REPO / "build" / "sss_gen"
PY_BIN = [sys.executable, str(REPO / "scripts" / "sss_gen.py")]


# ── Helpers ───────────────────────────────────────────────────────

def _train_sanrio_sss(out_path: Path) -> None:
    """Re-use the Phase 1 trainer on the in-tree sanrio dataset to
    produce a small .sss for the C-path tests."""
    cp = subprocess.run([
        sys.executable, str(REPO / "scripts" / "sss_train.py"),
        "--labels", str(REPO / "data" / "sanrio" / "labels.tsv"),
        "--root",   str(REPO / "data" / "sanrio"),
        "--out",    str(out_path),
        "--size",   "64",
    ], capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(cp.stdout + cp.stderr)
        raise SystemExit(1)


def _build_test_sfb(path: Path) -> None:
    """Synthesise a one-motif, one-condition, one-response .sfb
    bank so the Phase 4 path has trainable content without going
    through the full Phase 3 + Phase 4 trainer pipeline."""
    rng = np.random.default_rng(0)
    motif = SSSMotif(
        label="flag",
        row_freq=rng.random(SFB_FREQ_BINS, dtype=np.float32),
        col_freq=rng.random(SFB_FREQ_BINS, dtype=np.float32),
        color_freq=rng.random((3, SFB_FREQ_BINS), dtype=np.float32),
        position_heatmap=rng.random((SFB_HEATMAP_DIM, SFB_HEATMAP_DIM),
                                     dtype=np.float32),
        coherence=0.9, confidence=0.9,
        activation_count=10, variation_cluster_id=0,
        temporal_length=16,
        response_count_in_motif=1, response_offset=0,
        cross_resonance=np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32),
        warp_self_x=np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32),
        warp_self_y=np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32),
    )
    # Oscillatory condition with a clear non-DC peak.
    temp = np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)
    temp[16] = 1.0
    cond = SSSCondition(
        label="wind", condition_type=CONDITION_OSCILLATORY,
        row_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        col_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        temporal_freq=temp,
        direction=0.0, default_intensity=1.0, decay_rate=0.0,
    )
    # Response with a strong row-band shift + non-zero warp.
    row_resp = np.zeros(SFB_FREQ_BINS, dtype=np.float32)
    row_resp[5:25] = 1.0
    wx = np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32)
    wy = np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32)
    for gx in range(SFB_WARP_DIM):
        wy[:, gx] = 0.5 * np.sin(2 * np.pi * gx / SFB_WARP_DIM)
    response = SSSInteractionResponse(
        motif_id=0, condition_id=0,
        row_response=row_resp,
        col_response=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        cross_response=temp.copy(),
        warp_x_response=wx, warp_y_response=wy,
        response_strength=1.0, response_delay=0.0,
        accumulated_samples=1,
    )
    SSSFeatureBank(motifs=[motif], conditions=[cond],
                   responses=[response]).save(str(path))


def _read_ppm_flat(path: Path) -> np.ndarray:
    """Read a PPM into a float32 [0, 1] flat vector for L2 comparison."""
    img = sss_image_io.read_ppm(str(path))
    return img.astype(np.float32).ravel() / 255.0


def _run(cmd: list, expect_code: int = 0) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != expect_code:
        sys.stderr.write(f"unexpected exit {cp.returncode} "
                         f"(want {expect_code}) for: {' '.join(cmd)}\n")
        sys.stderr.write(cp.stdout); sys.stderr.write(cp.stderr)
        raise SystemExit(1)
    return cp


# ── Tests ─────────────────────────────────────────────────────────

def test_c_single_image(tmp: Path, sss_model: Path) -> None:
    out = tmp / "single.ppm"
    _run([str(C_BIN), str(sss_model), "white cat kitty", str(out),
          "1", "1.5", "12"])
    img = sss_image_io.read_ppm(str(out))
    assert img.shape == (64, 64, 3), img.shape
    print(f"OK  C single   : wrote {out.name} ({img.shape})")


def test_c_frames_loop(tmp: Path, sss_model: Path) -> None:
    out_dir = tmp / "c_frames"
    _run([str(C_BIN), str(sss_model), "white cat kitty",
          str(tmp / "unused.ppm"),
          "1", "1.5", "12", "--frames", "4", "--out-dir", str(out_dir)])
    frames = sorted(out_dir.glob("frame_*.ppm"))
    assert len(frames) == 4, [p.name for p in frames]
    # Consecutive frames must differ — Phase 1 seed-variation rule.
    pixs = [_read_ppm_flat(p) for p in frames]
    diffs = [float(np.linalg.norm(pixs[i] - pixs[i + 1]) / np.sqrt(pixs[i].size))
             for i in range(len(pixs) - 1)]
    assert min(diffs) > 0.01, (
        f"C --frames N produced near-identical frames; per-frame "
        f"diffs {diffs}")
    print(f"OK  C frames=4 : 4 PPMs written, consecutive-frame diff "
          f"min={min(diffs):.4f}")


def test_c_phase4_flags_rejected(tmp: Path, sss_model: Path) -> None:
    """The C binary must refuse Phase 4 flags with exit 7 and a
    pointer to scripts/sss_gen.py — not silently drop them."""
    for flag, extra in [("--condition", ["wind", "0.5"]),
                        ("--feature-bank", [str(sss_model)]),
                        ("--atmos-from", [str(sss_model)])]:
        cp = subprocess.run(
            [str(C_BIN), str(sss_model), "kitty", str(tmp / "x.ppm"),
             flag, *extra],
            capture_output=True, text=True)
        assert cp.returncode == 7, \
            f"{flag} should exit 7, got {cp.returncode}\n{cp.stderr}"
        assert "scripts/sss_gen.py" in cp.stderr, \
            f"{flag} stderr missing pointer to scripts/sss_gen.py:\n{cp.stderr}"
    print("OK  C rejects  : --condition / --feature-bank / --atmos-from "
          "exit 7 with pointer to scripts/sss_gen.py")


def test_python_condition_intensity(tmp: Path, sfb_path: Path) -> None:
    """Phase 4 condition-intensity check: same prompt and seed, with
    intensity 0.0 vs 0.5, must produce materially different output.
    Phase 4's test_synthesizer already proves the synthesizer math;
    this test confirms the wrapper CLI plumbs intensity through."""
    out_lo = tmp / "intensity_0.ppm"
    out_hi = tmp / "intensity_0.5.ppm"
    _run(PY_BIN + [str(sfb_path), "flag", str(out_lo),
                   "42", "1.5", "12",
                   "--condition", "wind", "0.0"])
    _run(PY_BIN + [str(sfb_path), "flag", str(out_hi),
                   "42", "1.5", "12",
                   "--condition", "wind", "0.5"])
    a = _read_ppm_flat(out_lo)
    b = _read_ppm_flat(out_hi)
    l2 = float(np.linalg.norm(a - b))
    assert l2 > 0.3, (
        f"intensity 0.0 vs 0.5 produced near-identical output "
        f"(L2 = {l2:.4f} ≤ 0.3); condition pipeline isn't applying "
        "the response.")
    print(f"OK  py intensity: 0.0 vs 0.5 L2 = {l2:.3f} (> 0.3)")


def test_python_frames_with_condition(tmp: Path, sfb_path: Path) -> None:
    """frames = 8 with a condition: 8 PPMs, consecutive frames
    differ."""
    out_dir = tmp / "py_frames"
    _run(PY_BIN + [str(sfb_path), "flag", str(tmp / "unused.ppm"),
                   "1", "1.5", "12",
                   "--frames", "8", "--out-dir", str(out_dir),
                   "--condition", "wind", "0.5"])
    frames = sorted(out_dir.glob("frame_*.ppm"))
    assert len(frames) == 8, [p.name for p in frames]
    pixs = [_read_ppm_flat(p) for p in frames]
    diffs = [float(np.linalg.norm(pixs[i] - pixs[i + 1]) / np.sqrt(pixs[i].size))
             for i in range(len(pixs) - 1)]
    assert min(diffs) > 0.001, \
        f"frames near-identical; consecutive diff min = {min(diffs)}"
    print(f"OK  py frames=8: 8 PPMs written, consecutive-frame diff "
          f"min={min(diffs):.4f}")


def main() -> int:
    if not C_BIN.is_file():
        print("error: build/sss_gen missing — run `make sss_gen` first",
              file=sys.stderr); return 1
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sss_model = tmp / "model.sss"
        sfb_path  = tmp / "test.sfb"
        _train_sanrio_sss(sss_model)
        _build_test_sfb(sfb_path)
        test_c_single_image(tmp, sss_model)
        test_c_frames_loop(tmp, sss_model)
        test_c_phase4_flags_rejected(tmp, sss_model)
        test_python_condition_intensity(tmp, sfb_path)
        test_python_frames_with_condition(tmp, sfb_path)
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
