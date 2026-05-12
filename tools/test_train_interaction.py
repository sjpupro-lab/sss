"""End-to-end regression for scripts/sss_train_interaction.

Builds a tiny world from scratch:

  motifs:     fur, flag, rock           (three primitive motifs;
              fur ≈ flag in row/col envelopes so the similar-motif
              accumulator has a non-trivial donor for fur once flag
              is trained.)
  conditions: wind, gravity              (one oscillatory, one
              continuous; the gravity pair should be rejected
              because the active sequence isn't responding.)
  pairs:      flag_wind, fur_wind, rock_gravity

Expectations:

  - `flag_wind` and `fur_wind` produce a usable response with
    correlation > 0.7 (the trainer's threshold). The fur response
    benefits from the similar-motif accumulation rule because fur
    and flag have overlapping envelopes — `accumulated_samples`
    on fur should reflect at least the contribution from flag.
  - `rock_gravity` is rejected (correlation below threshold, or
    motion energy too small to fit anyway). It should NOT appear
    in the interactions output.
  - The aggregated cross_response carries non-zero energy for the
    wind responses.
  - Resulting interactions.npz is loadable and feeds cleanly into
    `sss_build_feature_bank.py --interactions`, producing a v2
    `.sfb` whose Phase 2 round-trip still holds.

Run:
    python3 tools/test_train_interaction.py
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
from tools.sss_feature_bank import SSSFeatureBank             # noqa: E402


# ── Synthetic generators ──────────────────────────────────────────

def _u8(arr): return np.clip(arr, 0, 255).astype(np.uint8)


def _flag_frame(rng, t=0, condition_amp=0.0):
    """Bright horizontal-stripe pattern. The condition modulates the
    *stripe contrast* — multiplying the stripe's amplitude by
    (1 + condition_amp). The trainer's envelope (which kills DC)
    only sees spectral content from the stripes themselves, so the
    contrast modulation is exactly what produces a measurable
    row_freq delta when the condition is active. A pure brightness
    shift would be invisible because mean subtraction kills it."""
    h = w = 64
    y = np.arange(h, dtype=np.float32)
    stripe = ((y // 8).astype(int) & 1).astype(np.float32)  # 0/1
    contrast = 1.0 + condition_amp                          # 0..2
    band = (stripe * 0.6 * contrast + 0.2)
    base = band[:, None, None].repeat(w, axis=1).repeat(3, axis=2) * 255.0
    base += rng.normal(0, 3, size=base.shape).astype(np.float32)
    return _u8(base)


def _fur_frame(rng, t=0, condition_amp=0.0):
    """Fur shares flag's stripe family but adds per-pixel texture
    noise. The same vibration condition shifts intensity uniformly."""
    base = _flag_frame(rng, t=t, condition_amp=condition_amp).astype(np.float32)
    base += rng.normal(0, 20, size=base.shape).astype(np.float32)
    return _u8(base)


def _rock_frame(rng, t=0, condition_amp=0.0):
    """Vertical-stripe rock pattern that *ignores* the condition
    (no condition_amp coupling). Response should be flat → trainer
    rejects the pair via the correlation threshold."""
    h = w = 64
    x = np.arange(w, dtype=np.float32)
    stripe = ((x // 12).astype(int) & 1).astype(np.float32)
    base = (stripe * 0.4 + 0.3)[None, :, None].repeat(h, axis=0).repeat(3, axis=2)
    base = base * 255.0
    base += rng.normal(0, 3, size=base.shape).astype(np.float32)
    return _u8(base)


def _sequence(generator, n_frames=16, active=False, seed=0):
    """Build a 16-frame sequence. When `active`, modulate the
    condition intensity per-pixel using cos(2π t / 4). cos (not
    sin) so the active-signal phase matches the trainer's zero-
    phase irfft reconstruction of the stored temporal_freq
    spectrum — that's what lets the (intensity, delta) energy
    correlation clear the trainer's threshold. Phase information
    is dropped on disk by the Phase 1 amp-only rule, so test
    fixtures need to assume zero-phase round-trip."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        c = float(np.cos(2 * np.pi * t / 4.0)) if active else 0.0
        frames.append(generator(rng, t=t, condition_amp=c))
    return np.stack(frames, axis=0)


def _write_seq(folder: str, seq: np.ndarray) -> None:
    os.makedirs(folder, exist_ok=True)
    for t in range(seq.shape[0]):
        sss_image_io.write_ppm(
            os.path.join(folder, f"frame_{t:03d}.ppm"),
            seq[t][..., ::-1])


# ── Test ──────────────────────────────────────────────────────────

def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prim_root = os.path.join(tmp, "primitive")
        cond_root = os.path.join(tmp, "conditions")
        inter_root = os.path.join(tmp, "interactions")
        motifs_p = os.path.join(tmp, "motifs.npz")
        cond_p   = os.path.join(tmp, "conditions.npz")
        inter_p  = os.path.join(tmp, "interactions.npz")
        sfb_p    = os.path.join(tmp, "bank.sfb")

        # ── 1. Primitive motifs: flag / fur / rock ──────────────
        for label, gen in [("flag", _flag_frame),
                           ("fur",  _fur_frame),
                           ("rock", _rock_frame)]:
            d = os.path.join(prim_root, label)
            os.makedirs(d, exist_ok=True)
            for i in range(20):
                rng = np.random.default_rng(100 + i)
                frame = gen(rng, t=0, condition_amp=0.0)
                sss_image_io.write_ppm(
                    os.path.join(d, f"s{i:03d}.ppm"),
                    frame[..., ::-1])

        cp = subprocess.run([sys.executable, os.path.join(REPO, "scripts",
                                                          "sss_train_primitive.py"),
                             "--data", prim_root, "--out", motifs_p],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stdout + cp.stderr); return 1
        print(cp.stdout.strip())

        # ── 2. Conditions: vibration + gravity ────────────────────
        # `vibration` is a per-pixel uniform oscillation — same
        # signal shape as the active sequences below, so the
        # condition's temporal spectrum and the active delta-norm
        # series share the same peak frequency. That's what lets
        # the trainer's correlation clear the 0.7 threshold.
        from tools.test_train_condition import (
            _vibration as gen_vibration, _gravity as gen_gravity,
        )
        for label, gen in [("vibration", gen_vibration),
                           ("gravity",   gen_gravity)]:
            for seq_i in range(2):
                d = os.path.join(cond_root, label, f"seq_{seq_i:02d}")
                seq = gen(seq_i)
                _write_seq(d, seq)

        cp = subprocess.run([sys.executable, os.path.join(REPO, "scripts",
                                                          "sss_train_condition.py"),
                             "--data", cond_root, "--out", cond_p],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stdout + cp.stderr); return 1
        print(cp.stdout.strip())

        # ── 3. Interaction pairs ─────────────────────────────────
        # flag_vibration / fur_vibration: motif clearly responds.
        # rock_gravity: rock ignores condition (active==static),
        # so correlation should fall below 0.7 → trainer rejects.
        for pair, (gen, condition_on) in [
                ("flag_vibration", (_flag_frame, True)),
                ("fur_vibration",  (_fur_frame,  True)),
                ("rock_gravity",   (_rock_frame, False)),
        ]:
            base = os.path.join(inter_root, pair)
            _write_seq(os.path.join(base, "static"),
                       _sequence(gen, n_frames=16, active=False, seed=1))
            _write_seq(os.path.join(base, "active"),
                       _sequence(gen, n_frames=16, active=condition_on, seed=2))

        cp = subprocess.run([sys.executable, os.path.join(REPO, "scripts",
                                                          "sss_train_interaction.py"),
                             "--motifs", motifs_p,
                             "--conditions", cond_p,
                             "--data", inter_root,
                             "--out", inter_p, "--verbose"],
                            capture_output=True, text=True)
        sys.stdout.write(cp.stdout)
        if cp.returncode != 0:
            sys.stderr.write(cp.stderr); return 1

        # ── 4. Inspect interactions.npz ─────────────────────────
        npz = np.load(inter_p, allow_pickle=False)
        motif_ids = list(npz["motif_id"])
        cond_ids  = list(npz["condition_id"])
        n_inter   = len(motif_ids)
        print(f"\nInteractions emitted: {n_inter}")

        motifs_npz = np.load(motifs_p, allow_pickle=False)
        cond_npz   = np.load(cond_p, allow_pickle=False)
        motif_labels = [str(s) for s in motifs_npz["labels"]]
        cond_labels  = [str(s) for s in cond_npz["labels"]]
        vib_idx = cond_labels.index("vibration")

        # flag_vibration + fur_vibration must both be present.
        flag_idx = motif_labels.index("flag")
        fur_idx  = motif_labels.index("fur")
        pairs_seen = {(int(motif_ids[i]), int(cond_ids[i])): i
                      for i in range(n_inter)}
        assert (flag_idx, vib_idx) in pairs_seen, \
            f"flag_vibration missing; have {list(pairs_seen)}"
        assert (fur_idx, vib_idx) in pairs_seen, \
            f"fur_vibration missing; have {list(pairs_seen)}"
        print(f"OK  vib pairs  : flag_vibration + fur_vibration both present")

        # rock_gravity must be REJECTED (low correlation, or rock
        # ignoring wind). It should not appear in the interactions
        # output at all.
        rock_idx = motif_labels.index("rock")
        gravity_idx = cond_labels.index("gravity")
        assert (rock_idx, gravity_idx) not in pairs_seen, (
            "rock_gravity should be rejected — rock doesn't respond "
            "to wind/gravity, so correlation falls below 0.7")
        print(f"OK  rejection  : rock_gravity not emitted (correlation "
              "below threshold)")

        # Similar-motif accumulation: fur should be flagged with
        # accumulated_samples > 1 (it picked up flag's contribution
        # because fur≈flag in row/col envelopes).
        fur_resp_idx = pairs_seen[(fur_idx, vib_idx)]
        fur_accum = int(npz["accumulated_samples"][fur_resp_idx])
        flag_resp_idx = pairs_seen[(flag_idx, vib_idx)]
        flag_accum = int(npz["accumulated_samples"][flag_resp_idx])
        print(f"  accumulated samples: flag={flag_accum}, fur={fur_accum}")
        # Either fur picked up flag (accum > 1) or flag picked up fur
        # (depending on processing order). At least one of them must
        # be > 1 for the "similar-motif accumulation" rule to be live.
        assert max(fur_accum, flag_accum) > 1, (
            "similar-motif accumulator never activated; expected fur "
            "and flag (cosine ≥ 0.85) to share at least one contributor")
        print("OK  accumulator: similar-motif (fur≈flag) accumulation live")

        # cross_response should carry non-zero energy on wind pairs.
        cross_flag = float(np.linalg.norm(npz["cross_response"][flag_resp_idx]))
        cross_fur  = float(np.linalg.norm(npz["cross_response"][fur_resp_idx]))
        assert cross_flag > 1e-3 and cross_fur > 1e-3, (
            f"cross_response unexpectedly zero (flag={cross_flag}, "
            f"fur={cross_fur})")
        print(f"OK  cross spec : ||cross_response|| = "
              f"{cross_flag:.3f} (flag), {cross_fur:.3f} (fur)")

        # ── 5. Build v2 .sfb from this and verify round-trip ────
        cp = subprocess.run([sys.executable, os.path.join(REPO, "scripts",
                                                          "sss_build_feature_bank.py"),
                             "--motifs", motifs_p,
                             "--conditions", cond_p,
                             "--interactions", inter_p,
                             "--out", sfb_p],
                            capture_output=True, text=True)
        sys.stdout.write(cp.stdout)
        if cp.returncode != 0:
            sys.stderr.write(cp.stderr); return 1

        bank = SSSFeatureBank.load(sfb_p)
        assert len(bank.motifs) == 3
        assert len(bank.conditions) == 2
        assert len(bank.responses) == n_inter
        print(f"OK  v2 .sfb     : {len(bank.motifs)} motifs, "
              f"{len(bank.conditions)} conditions, "
              f"{len(bank.responses)} responses; round-trip clean")

        print("\nPASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
