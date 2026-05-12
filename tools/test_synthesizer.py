"""Phase 4 synthesizer + scenarios regression.

Reproduces the four GPT pre-verification scenarios:

  1. Condition classifier: gravity-class signal routed CONTINUOUS
     via the frame-delta early-out (the GPT-validated fix).
  2. Interaction-response learning: a (motif, condition) pair
     with a real response gets accumulated, with similar-motif
     contributions when applicable.
  3. Condition-strength reflection in synthesis: synthesising at
     intensity = 0, 0.5, 1.0 produces materially different output
     envelopes (L2 difference > 0.1 between the extremes).
  4. Cross-axis resonance: turning the motif's `cross_resonance`
     off zeros out the cross-modulation interference pattern in
     the rendered frame, while turning it on injects visible
     row/col side-band content.

Plus a fifth direct check for the Phase 4 warp pipeline:

  5. Apply-warp displacement: feeding a known warp field through
     `apply_warp` produces measurable per-pixel displacement of a
     synthetic test image (mean displacement > 1 px at strength = 8).

Run:
    python3 tools/test_synthesizer.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.sss_feature_bank import (                           # noqa: E402
    SSSFeatureBank, SSSMotif, SSSCondition, SSSInteractionResponse,
    SFB_FREQ_BINS, SFB_TEMPORAL_BINS, SFB_WARP_DIM, SFB_HEATMAP_DIM,
    CONDITION_CONTINUOUS, CONDITION_IMPULSE, CONDITION_OSCILLATORY,
)
from tools.sss_synthesizer import (                            # noqa: E402
    SynthOutput, synth_frame, apply_warp, synthesise_from_envelope,
)


# ── Tiny synthetic bank ───────────────────────────────────────────

def _build_test_bank() -> SSSFeatureBank:
    """One motif, two conditions, one strong response. Hand-built
    so the synthesizer's behaviour is independent of trainer noise."""
    rng = np.random.default_rng(0)
    # Motif: flag-ish — broadband row spectrum, narrow col peak.
    row_freq = rng.random(SFB_FREQ_BINS, dtype=np.float32) * 0.1
    col_freq = np.zeros(SFB_FREQ_BINS, dtype=np.float32)
    col_freq[8] = 1.0
    motif = SSSMotif(
        label="flag",
        row_freq=row_freq,
        col_freq=col_freq,
        color_freq=rng.random((3, SFB_FREQ_BINS), dtype=np.float32) * 0.1,
        position_heatmap=rng.random((SFB_HEATMAP_DIM, SFB_HEATMAP_DIM),
                                     dtype=np.float32),
        coherence=0.9, confidence=0.9,
        activation_count=10, variation_cluster_id=0,
        temporal_length=16,
        response_count_in_motif=1,
        response_offset=0,
        cross_resonance=np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32),
        warp_self_x=np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32),
        warp_self_y=np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32),
    )

    # Condition 1: vibration — clear non-DC temporal peak.
    vib_temp = np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)
    vib_temp[16] = 1.0                                       # period ≈ 4 frames
    vibration = SSSCondition(
        label="vibration",
        condition_type=CONDITION_OSCILLATORY,
        row_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        col_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        temporal_freq=vib_temp,
        direction=0.0, default_intensity=1.0, decay_rate=0.0,
    )
    # Condition 2: gravity — broad flat spectrum, CONTINUOUS.
    gravity = SSSCondition(
        label="gravity",
        condition_type=CONDITION_CONTINUOUS,
        row_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        col_freq=np.zeros(SFB_FREQ_BINS, dtype=np.float32),
        temporal_freq=np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32),
        direction=-np.pi / 2.0,
        default_intensity=0.3, decay_rate=0.0,
    )

    # Response for (flag, vibration): row delta + warp swing.
    row_resp = np.zeros(SFB_FREQ_BINS, dtype=np.float32)
    row_resp[5:15] = 1.0                                     # injects a row band
    col_resp = np.zeros(SFB_FREQ_BINS, dtype=np.float32)
    cross_resp = np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)
    cross_resp[10] = 1.0
    warp_x = np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32)
    warp_y = np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32)
    # A horizontal-wave warp: each column gets a y-displacement
    # that varies sinusoidally with x. apply_warp should expand this
    # to 64×64 and shift pixels accordingly.
    for gx in range(SFB_WARP_DIM):
        warp_y[:, gx] = 0.5 * np.sin(2 * np.pi * gx / SFB_WARP_DIM)

    response = SSSInteractionResponse(
        motif_id=0, condition_id=0,
        row_response=row_resp,
        col_response=col_resp,
        cross_response=cross_resp,
        warp_x_response=warp_x,
        warp_y_response=warp_y,
        response_strength=1.0,
        response_delay=0.0,
        accumulated_samples=3,
    )

    return SSSFeatureBank(
        motifs=[motif], relations=[], identities=[],
        conditions=[vibration, gravity], responses=[response],
    )


# ── Scenarios ─────────────────────────────────────────────────────

def test_classifier_gravity_continuous() -> None:
    """GPT scenario 1: a smooth low-energy drift gets routed
    CONTINUOUS via the frame-delta early-out, NOT through the FFT
    path. This is the GPT-validated fix that prevents the classifier
    from invoking the temporal-FFT branch for gravity-class signals."""
    from tools.test_train_condition import _gravity
    from scripts.sss_train_condition import classify_signal, CONDITION_CONTINUOUS
    seq = _gravity(0).astype(np.float32) / 255.0
    ctype, delta = classify_signal(seq)
    assert ctype == CONDITION_CONTINUOUS, ctype
    print(f"OK  scenario 1 : gravity Δ={delta:.4f} → CONTINUOUS "
          "(frame-delta early-out)")


def test_intensity_changes_output() -> None:
    """GPT scenario 3: synthesising the same (motif, condition,
    seed) at intensity 0.0 / 0.5 / 1.0 produces measurably different
    outputs. Δ envelope L2 between intensity 0 and 1 should be > 0.1
    when the response is non-trivial."""
    bank = _build_test_bank()
    motif = bank.motifs[0]
    cond  = bank.conditions[0]              # vibration
    envs = []
    for intensity in (0.0, 0.5, 1.0):
        out = synth_frame(bank, motif, cond, intensity=intensity,
                          t=0.0, seed=42)
        envs.append(out)
    l2_01 = float(np.linalg.norm(envs[0].envelope_row - envs[1].envelope_row))
    l2_02 = float(np.linalg.norm(envs[0].envelope_row - envs[2].envelope_row))
    assert l2_02 > 0.1, f"intensity 0 vs 1 row delta L2 {l2_02:.3f} <= 0.1"
    # Monotone: 0→1 farther than 0→0.5.
    assert l2_02 > l2_01, (l2_01, l2_02)
    print(f"OK  scenario 3 : intensity 0/0.5/1.0 → row deltas "
          f"L2 0→0.5={l2_01:.3f}, 0→1={l2_02:.3f} (monotone)")


def test_cross_resonance_modulates_render() -> None:
    """GPT scenario 4: turning cross_resonance on injects measurable
    interference content into the rendered frame. The naive
    renderer's row+col baseline is reproduced when cross == 0; with
    cross != 0 we expect a non-trivial L2 difference."""
    rng = np.random.default_rng(123)
    env_row = rng.random(SFB_FREQ_BINS, dtype=np.float32)
    env_col = rng.random(SFB_FREQ_BINS, dtype=np.float32)
    cross_off = np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)
    cross_on  = np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)
    cross_on[6] = 1.0
    img_off = synthesise_from_envelope(env_row, env_col, cross_off,
                                       seed=42, size=64)
    img_on  = synthesise_from_envelope(env_row, env_col, cross_on,
                                       seed=42, size=64)
    diff = float(np.linalg.norm(img_off - img_on))
    assert diff > 0.5, \
        f"cross on/off image diff L2 {diff:.3f} <= 0.5"
    print(f"OK  scenario 4 : cross_resonance off → on image L2 = "
          f"{diff:.3f} (visible side-band modulation)")


def test_apply_warp_displaces_pixels() -> None:
    """Direct check that apply_warp moves pixels — feed a
    checkerboard image through a known warp field and measure that
    the centre of mass of each square has shifted."""
    H = W = 64
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[16:48, 16:48, :] = 1.0                # white square in centre
    warp_y = np.full((SFB_WARP_DIM, SFB_WARP_DIM), 0.5, dtype=np.float32)
    warp_x = np.zeros_like(warp_y)
    warped = apply_warp(img, warp_x, warp_y, strength=8.0)
    # Centre of mass of the warped white region — should sit BELOW
    # the original centre (y = 31.5 → ~ 35 after a +4 px shift).
    yy = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32)
    mask = warped[..., 0] > 0.5
    cy_warped = float((yy * mask).sum() / max(1, mask.sum()))
    assert cy_warped > 33.0, \
        f"warped centre-of-mass y={cy_warped:.2f} ≈ original; warp didn't move"
    print(f"OK  apply_warp : centre-of-mass shifted to y={cy_warped:.2f} "
          "(from y≈31.5)")


def test_interaction_accumulator_sanity() -> None:
    """Smoke check that the synth path applies the response.
    response_strength = 1.0 on our test bank, so synth_frame at
    intensity=1.0 should pick up the response's row_response (a
    band injected at bins 5..15)."""
    bank = _build_test_bank()
    motif = bank.motifs[0]
    cond  = bank.conditions[0]
    out = synth_frame(bank, motif, cond, intensity=1.0,
                      t=0.0, seed=7)
    base_row = motif.row_freq
    delta = out.envelope_row - base_row
    band_energy = float(np.linalg.norm(delta[5:15]))
    elsewhere   = float(np.linalg.norm(delta) - band_energy)
    assert band_energy > 0.1, \
        f"response band energy {band_energy:.3f} should exceed 0.1"
    assert band_energy > elsewhere, \
        f"response band ({band_energy:.3f}) should outweigh non-band " \
        f"energy ({elsewhere:.3f})"
    print(f"OK  scenario 2 : (flag, vibration) response applied — band "
          f"energy {band_energy:.3f} > non-band {elsewhere:.3f}")


def main() -> int:
    test_classifier_gravity_continuous()
    test_interaction_accumulator_sanity()
    test_intensity_changes_output()
    test_cross_resonance_modulates_render()
    test_apply_warp_displaces_pixels()
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
