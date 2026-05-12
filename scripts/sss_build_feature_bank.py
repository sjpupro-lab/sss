#!/usr/bin/env python3
"""sss_build_feature_bank — Phase 3 .sfb compositor.

Combines the two intermediates produced upstream into a single
Phase 2 .sfb file:

  * `primitive_motifs.npz` (from `scripts/sss_train_primitive.py`):
    one motif per dataset folder, holding zero-mean L2-normalised
    row / col / color frequency envelopes + a 16×16 position
    heatmap derived from per-folder Sobel magnitude.

  * `motif_memory.npz` (from `scripts/sss_train_motif_memory.py`):
    relations + identities + an updated per-motif 16×16 position
    heatmap derived from where each motif actually fired across the
    labelled-image scan. When present, this heatmap *replaces* the
    primitive heatmap because the labelled scan provides truer
    spatial statistics ("where does the motif appear in real
    scenes?") than the primitive folder's edge-density proxy.

The result is a `.sfb` v1 file. Every Phase 2 invariant continues
to hold: no phase, no row order, no original pixel coordinates;
only motif-level frequency envelopes + 16×16 heatmaps.

CLI:
    python scripts/sss_build_feature_bank.py \\
        --motifs build/primitive_motifs.npz \\
        --memory build/motif_memory.npz \\
        --out    build/feature_bank.sfb

`--memory` is optional. When omitted the .sfb is built from the
primitive-only intermediate (no relations, no identities, primitive
heatmap retained). This keeps the build pipeline usable while the
labelled-image trainer is still being assembled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.sss_feature_bank import (                       # noqa: E402
    SSSFeatureBank, SSSMotif, SSSRelation, SSSIdentity,
    SSSCondition, SSSInteractionResponse,
)


def _motifs_from_npz(motifs_npz: np.lib.npyio.NpzFile,
                     heatmap_update: np.ndarray | None) -> list[SSSMotif]:
    """Turn the primitive .npz arrays into SSSMotif dataclass
    instances. When `heatmap_update` is provided (Phase 3 memory
    stage), replace the primitive heatmap with the one derived from
    labelled-image scans — but only for motifs that actually fired
    in at least one labelled image. Motifs not seen by the memory
    builder keep their primitive heatmap; overwriting with the
    memory builder's all-zero row would erase valid spatial prior."""
    n = int(motifs_npz["row_freq"].shape[0])
    out = []
    for i in range(n):
        primitive_hm = motifs_npz["position_heatmap"][i].astype(np.float32)
        if heatmap_update is not None and i < heatmap_update.shape[0]:
            candidate = heatmap_update[i].astype(np.float32)
            heatmap = candidate if float(candidate.max()) > 0.0 \
                                else primitive_hm
        else:
            heatmap = primitive_hm
        out.append(SSSMotif(
            label=str(motifs_npz["labels"][i]),
            row_freq=motifs_npz["row_freq"][i].astype(np.float32),
            col_freq=motifs_npz["col_freq"][i].astype(np.float32),
            color_freq=motifs_npz["color_freq"][i].astype(np.float32),
            position_heatmap=heatmap,
            coherence=float(motifs_npz["coherence"][i]),
            confidence=float(motifs_npz["confidence"][i]),
            activation_count=int(motifs_npz["activation_count"][i]),
            variation_cluster_id=int(motifs_npz["variation_cluster_id"][i]),
        ))
    return out


def _relations_from_memory(mem_npz: np.lib.npyio.NpzFile,
                           motif_count: int) -> list[SSSRelation]:
    rels_raw = mem_npz["relations"]
    out = []
    for row in rels_raw:
        src, dst, dx, dy, rtype, weight = row
        src_i = int(src); dst_i = int(dst)
        if src_i >= motif_count or dst_i >= motif_count:
            # Skip stale references — should never happen if the
            # memory was built from the same motif .npz, but defend
            # the .sfb-side index validation that runs at save.
            continue
        out.append(SSSRelation(
            src_motif_id=src_i,
            dst_motif_id=dst_i,
            dx=float(dx),
            dy=float(dy),
            relation_type=int(rtype),
            weight=float(weight),
        ))
    return out


def _identities_from_memory(mem_npz: np.lib.npyio.NpzFile,
                            motif_count: int) -> list[SSSIdentity]:
    labels   = list(mem_npz["identities_labels"])
    flat_ids = list(mem_npz["identities_motif_flat"])
    lens     = list(mem_npz["identities_motif_lens"])
    out = []
    off = 0
    for lab, ln in zip(labels, lens):
        ids = [int(i) for i in flat_ids[off:off + int(ln)]]
        off += int(ln)
        # Filter out stale motif ids; never re-emit an empty identity
        # because the .sfb format rejects motif_count == 0.
        ids = [i for i in ids if 0 <= i < motif_count]
        if not ids:
            continue
        out.append(SSSIdentity(
            label=str(lab),
            motif_ids=ids,
            confidence=1.0,                 # placeholder; Phase 4 may compute
        ))
    return out


def _conditions_from_npz(cond_npz: np.lib.npyio.NpzFile) -> list[SSSCondition]:
    n = int(cond_npz["labels"].shape[0])
    out = []
    for i in range(n):
        out.append(SSSCondition(
            label=str(cond_npz["labels"][i]),
            condition_type=int(cond_npz["condition_type"][i]),
            row_freq=cond_npz["row_freq"][i].astype(np.float32),
            col_freq=cond_npz["col_freq"][i].astype(np.float32),
            temporal_freq=cond_npz["temporal_freq"][i].astype(np.float32),
            direction=float(cond_npz["direction"][i]),
            default_intensity=float(cond_npz["default_intensity"][i]),
            decay_rate=float(cond_npz["decay_rate"][i]),
        ))
    return out


def _responses_from_npz(resp_npz: np.lib.npyio.NpzFile,
                        motif_count: int,
                        condition_count: int) -> list[SSSInteractionResponse]:
    n = int(resp_npz["motif_id"].shape[0])
    out = []
    for i in range(n):
        m = int(resp_npz["motif_id"][i])
        c = int(resp_npz["condition_id"][i])
        if m >= motif_count or c >= condition_count:
            # Stale reference — skip rather than crash the build.
            continue
        out.append(SSSInteractionResponse(
            motif_id=m,
            condition_id=c,
            row_response=resp_npz["row_response"][i].astype(np.float32),
            col_response=resp_npz["col_response"][i].astype(np.float32),
            cross_response=resp_npz["cross_response"][i].astype(np.float32),
            warp_x_response=resp_npz["warp_x_response"][i].astype(np.float32),
            warp_y_response=resp_npz["warp_y_response"][i].astype(np.float32),
            response_strength=float(resp_npz["response_strength"][i]),
            response_delay=float(resp_npz["response_delay"][i]),
            accumulated_samples=int(resp_npz["accumulated_samples"][i]),
        ))
    return out


def build(motifs_path: Path, memory_path: Path | None,
          out_path: Path,
          conditions_path: Path | None = None,
          interactions_path: Path | None = None) -> SSSFeatureBank:
    """Top-level build. Returns the built bank for inspection by
    callers (e.g. the test) before/after the save.

    Phase 4 v2 additions: `conditions_path` and `interactions_path`
    are optional. When omitted, the bank ships with `conditions == []`
    and `responses == []` and behaves like a Phase 3 build."""
    motifs_npz = np.load(motifs_path, allow_pickle=False)
    motif_count = int(motifs_npz["row_freq"].shape[0])

    relations: list[SSSRelation] = []
    identities: list[SSSIdentity] = []
    heatmap_update: np.ndarray | None = None
    if memory_path is not None:
        mem_npz = np.load(memory_path, allow_pickle=False)
        mem_motif_count = int(mem_npz["motif_count"][0])
        if mem_motif_count != motif_count:
            print(f"WARN: memory has {mem_motif_count} motifs but "
                  f"primitives have {motif_count}; using primitives", file=sys.stderr)
        relations = _relations_from_memory(mem_npz, motif_count)
        identities = _identities_from_memory(mem_npz, motif_count)
        heatmap_update = mem_npz["position_heatmap_update"]

    motifs = _motifs_from_npz(motifs_npz, heatmap_update)

    conditions: list[SSSCondition] = []
    responses:  list[SSSInteractionResponse] = []
    if conditions_path is not None:
        cond_npz = np.load(conditions_path, allow_pickle=False)
        conditions = _conditions_from_npz(cond_npz)
    if interactions_path is not None:
        if not conditions:
            print("WARN: --interactions requires --conditions; skipping "
                  "responses", file=sys.stderr)
        else:
            resp_npz = np.load(interactions_path, allow_pickle=False)
            responses = _responses_from_npz(resp_npz, motif_count,
                                            len(conditions))

    bank = SSSFeatureBank(motifs=motifs, relations=relations,
                          identities=identities,
                          conditions=conditions, responses=responses)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(str(out_path))
    return bank


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motifs", required=True, type=Path,
                    help="primitive_motifs.npz from sss_train_primitive")
    ap.add_argument("--memory", type=Path, default=None,
                    help="optional motif_memory.npz from sss_train_motif_memory")
    ap.add_argument("--conditions", type=Path, default=None,
                    help="optional conditions.npz from sss_train_condition (v2)")
    ap.add_argument("--interactions", type=Path, default=None,
                    help="optional interactions.npz from sss_train_interaction (v2)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output .sfb path")
    args = ap.parse_args()

    if not args.motifs.is_file():
        print(f"error: --motifs {args.motifs} not found", file=sys.stderr)
        return 2
    if args.memory is not None and not args.memory.is_file():
        print(f"error: --memory {args.memory} not found", file=sys.stderr)
        return 2
    if args.conditions is not None and not args.conditions.is_file():
        print(f"error: --conditions {args.conditions} not found",
              file=sys.stderr); return 2
    if args.interactions is not None and not args.interactions.is_file():
        print(f"error: --interactions {args.interactions} not found",
              file=sys.stderr); return 2

    bank = build(args.motifs, args.memory, args.out,
                 conditions_path=args.conditions,
                 interactions_path=args.interactions)
    size = args.out.stat().st_size
    print(f"wrote {args.out}  "
          f"({len(bank.motifs)} motifs, {len(bank.relations)} relations, "
          f"{len(bank.identities)} identities, "
          f"{len(bank.conditions)} conditions, "
          f"{len(bank.responses)} responses, {size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
