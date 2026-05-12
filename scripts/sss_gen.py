#!/usr/bin/env python3
"""sss_gen — single Python entry point for the sss image / video engine.

Wraps two underlying paths so the UI and downstream scripts only ever
talk to one CLI surface:

  * Phase 1–3 path  (`.sss` model, no conditions, no .sfb features):
    shells out to the C binary `build/sss_gen`. The fast path; same
    code that the Phase 1–3 regression tests already cover.

  * Phase 4 path    (`.sfb` feature bank, conditions, atmos warm-
    starts, or frames > 1 with a real motion response): drives
    `tools/sss_synthesizer.py` directly. Synthesises N frames into
    an output directory, writing one PPM per frame.

The CLI is identical to the C binary (see `tools/sss_gen.c`); the
Python wrapper just adds the routing layer.

Examples
--------
Image (Phase 1-3):
    python3 scripts/sss_gen.py model.sss "kitty" out.ppm

Video (seed-variation baseline, no Phase 4 features):
    python3 scripts/sss_gen.py model.sss "kitty" out.ppm \\
        --frames 8 --out-dir /tmp/frames

Video (Phase 4 — conditions drive per-frame warp + envelope):
    python3 scripts/sss_gen.py feature_bank.sfb "flag" out.ppm \\
        --frames 16 --out-dir /tmp/frames \\
        --condition vibration 0.5

Atmos warm-start (Phase 4):
    python3 scripts/sss_gen.py feature_bank.sfb "scene" out.ppm \\
        --frames 16 --out-dir /tmp/frames --atmos-from reference.ppm
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ── Helpers ───────────────────────────────────────────────────────

def _is_sfb(path: Path) -> bool:
    """A .sfb file starts with `SFB1` magic — content-sniff so the
    user doesn't have to remember which extension matches which
    pipeline."""
    if not path.is_file(): return False
    try:
        return path.read_bytes()[:4] == b"SFB1"
    except OSError:
        return False


def _write_ppm(path: Path, image_bgr: np.ndarray) -> None:
    """Write a (H, W, 3) uint8 BGR image (matches the sss conventions)
    as a P6 PPM. PPM expects RGB so we flip the channel order."""
    img = np.ascontiguousarray(image_bgr[..., ::-1])
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path,
                    help="path to .sss (Phase 1-3) or .sfb (Phase 4)")
    ap.add_argument("prompt", type=str)
    ap.add_argument("out", type=Path,
                    help="output PPM (frames=1) or template path "
                         "(frames>1 — see --out-dir)")
    # Numeric positionals (parsed permissively so the CLI matches
    # the C binary's `[seed] [detail] [steps]` triplet).
    ap.add_argument("seed",   type=int,   nargs="?", default=1)
    ap.add_argument("detail", type=float, nargs="?", default=1.5)
    ap.add_argument("steps",  type=int,   nargs="?", default=120)

    ap.add_argument("--out-w", type=int, default=0)
    ap.add_argument("--out-h", type=int, default=0)
    ap.add_argument("--atmos", action="store_true")

    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--feature-bank", type=Path, default=None,
                    help="alias for the positional model path (.sfb)")
    ap.add_argument("--condition", nargs=2, action="append",
                    metavar=("LABEL", "INTENSITY"),
                    help="repeatable; LABEL must match a condition in "
                         "the .sfb, INTENSITY ∈ [0, 1]")
    ap.add_argument("--atmos-from", type=Path, default=None,
                    help="reference PPM for Atmos warm-start")
    ap.add_argument("--temporal-mode", choices=("static", "video"),
                    default=None,
                    help="optional; defaults to video when frames > 1")

    args = ap.parse_args()
    # --feature-bank overrides the positional model if both given.
    if args.feature_bank is not None:
        args.model = args.feature_bank
    # Conditions come in as [(LABEL, INTENSITY_STR), ...]; convert.
    if args.condition:
        args.condition = [(lab, float(it)) for lab, it in args.condition]
    return args


# ── Phase 1-3 path: shell out to C `sss_gen` ─────────────────────

def _run_c_sss_gen(args: argparse.Namespace) -> int:
    """Forward to `build/sss_gen` for the fast Phase 1-3 path. We do
    this only when (a) the model is .sss / non-.sfb, AND (b) no
    Phase-4 features were requested. The C binary itself rejects
    --condition / --atmos-from / --feature-bank, so reaching it
    here would be a misroute."""
    binary = REPO / "build" / "sss_gen"
    if not binary.is_file():
        print("error: build/sss_gen not found — run `make sss_gen`",
              file=sys.stderr)
        return 1
    cmd = [str(binary), str(args.model), args.prompt, str(args.out),
           str(args.seed), f"{args.detail:.6f}", str(args.steps)]
    if args.out_w > 0: cmd += ["--out-w", str(args.out_w)]
    if args.out_h > 0: cmd += ["--out-h", str(args.out_h)]
    if args.atmos:      cmd += ["--atmos"]
    if args.frames > 1:
        cmd += ["--frames", str(args.frames)]
        if args.out_dir is None:
            print("error: --frames > 1 requires --out-dir", file=sys.stderr)
            return 1
        cmd += ["--out-dir", str(args.out_dir)]
    if args.temporal_mode:
        cmd += ["--temporal-mode", args.temporal_mode]
    return subprocess.call(cmd)


# ── Phase 4 path: drive the Python synthesiser ───────────────────

def _run_python_synth(args: argparse.Namespace) -> int:
    """Phase 4: load the .sfb, resolve the prompt to a motif, walk
    the condition list, synthesise N frames via
    `tools.sss_synthesizer.synth_frame` + `synthesise_from_envelope`
    + `apply_warp`. Writes one PPM per frame.

    Per-frame seed is `args.seed ^ t` to preserve Phase 1's per-seed
    diversity rule at the frame level."""
    if not _is_sfb(args.model):
        print(f"error: --condition / --atmos-from / .sfb model "
              f"required, got {args.model}", file=sys.stderr)
        return 1
    if args.frames > 1 and args.out_dir is None:
        print("error: --frames > 1 requires --out-dir", file=sys.stderr)
        return 1

    from tools.sss_feature_bank import SSSFeatureBank
    from tools.sss_synthesizer import (
        synth_frame, synthesise_from_envelope, apply_warp,
    )
    bank = SSSFeatureBank.load(str(args.model))
    if not bank.motifs:
        print("error: .sfb has no motifs", file=sys.stderr); return 1

    # Resolve the prompt to a motif. Phase 5 doesn't ship the
    # weighted-cosine matcher from sss_unified yet, so we use a
    # simple label/substring match — falling back to motif 0
    # ("the first motif") only when nothing else fires. This
    # matches the audit report's "small surface" promise; richer
    # matching can come in a follow-up.
    prompt_l = args.prompt.lower()
    motif = None
    for m in bank.motifs:
        if m.label.lower() in prompt_l or prompt_l in m.label.lower():
            motif = m; break
    if motif is None:
        motif = bank.motifs[0]
    motif_label = motif.label

    # Resolve conditions (--condition LABEL INTENSITY, repeatable).
    # All conditions specified must exist in the .sfb.
    cond_list: list[tuple[object, float]] = []
    for label, intensity in (args.condition or []):
        cond = next((c for c in bank.conditions if c.label == label), None)
        if cond is None:
            print(f"error: condition {label!r} not in .sfb (have: "
                  f"{[c.label for c in bank.conditions]})", file=sys.stderr)
            return 1
        cond_list.append((cond, intensity))

    # frames=1 with no conditions: still allowed — uses the motif's
    # baseline envelopes only. A typical "image from .sfb" call.
    if not cond_list:
        # Synthesise a neutral SynthOutput from the motif itself.
        # Use the first stored condition (if any) at intensity 0
        # to drive the synthesiser API without injecting motion.
        if bank.conditions:
            cond_list = [(bank.conditions[0], 0.0)]

    out_w = args.out_w if args.out_w > 0 else 64
    out_h = args.out_h if args.out_h > 0 else 64

    def render_one(t_norm: float, seed: int) -> np.ndarray:
        # Each frame uses the first condition (Phase 4 v1 single-
        # condition path); multi-condition support is an
        # accumulation Phase 5+ may extend. For now we sum the
        # envelope deltas from each (cond, intensity) tuple.
        env_row = motif.row_freq.copy()
        env_col = motif.col_freq.copy()
        env_x   = motif.cross_resonance.copy()
        wx      = motif.warp_self_x.copy()
        wy      = motif.warp_self_y.copy()
        for cond, intensity in cond_list:
            out = synth_frame(bank, motif, cond,
                              intensity=intensity, t=t_norm,
                              seed=seed, n_frames=max(args.frames, 16))
            env_row += (out.envelope_row - motif.row_freq)
            env_col += (out.envelope_col - motif.col_freq)
            env_x   += (out.cross        - motif.cross_resonance)
            wx      += (out.warp_x       - motif.warp_self_x)
            wy      += (out.warp_y       - motif.warp_self_y)
        img_f = synthesise_from_envelope(env_row, env_col, env_x,
                                         seed=seed, size=max(out_w, out_h))
        # Resize to (out_h, out_w) and apply the per-frame warp.
        if img_f.shape[0] != out_h or img_f.shape[1] != out_w:
            yi = np.linspace(0, img_f.shape[0] - 1, out_h).round().astype(int)
            xi = np.linspace(0, img_f.shape[1] - 1, out_w).round().astype(int)
            img_f = img_f[yi[:, None], xi[None, :]]
        img_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        img_u8 = apply_warp(img_u8, wx, wy, strength=4.0)
        # synthesise_from_envelope returns RGB; sss conventions
        # treat output as BGR for the writer. Flip channel axis
        # so `_write_ppm` produces RGB on disk after its own flip.
        return img_u8[..., ::-1]

    if args.frames == 1:
        img = render_one(0.0, int(args.seed))
        _write_ppm(args.out, img)
        print(f"wrote {args.out}  ({out_w}x{out_h}, prompt={args.prompt!r}, "
              f"seed={args.seed}, motif={motif_label!r}, "
              f"conditions={[(c.label, i) for c, i in cond_list]})")
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in range(args.frames):
            t_norm = float(t) / max(1, args.frames - 1)
            seed = (int(args.seed) ^ t) & 0xFFFFFFFF
            img = render_one(t_norm, seed)
            _write_ppm(out_dir / f"frame_{t:04d}.ppm", img)
        print(f"wrote {args.frames} frames to {out_dir}  "
              f"({out_w}x{out_h}, prompt={args.prompt!r}, "
              f"base_seed={args.seed}, motif={motif_label!r}, "
              f"conditions={[(c.label, i) for c, i in cond_list]})")
    return 0


# ── Driver ───────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    if args.frames < 1:
        print("error: --frames must be >= 1", file=sys.stderr); return 1
    if args.temporal_mode == "video" and args.frames < 2:
        print("error: --temporal-mode video requires --frames >= 2",
              file=sys.stderr); return 1

    is_phase4 = (
        _is_sfb(args.model)
        or args.condition
        or args.atmos_from is not None
    )
    if is_phase4:
        return _run_python_synth(args)
    return _run_c_sss_gen(args)


if __name__ == "__main__":
    sys.exit(main())
