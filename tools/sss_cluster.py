#!/usr/bin/env python3
"""sss_cluster — discover morpheme ↔ cell-cluster associations.

The .sss file holds one cell per (image, word) pair: a COLOR cell, a
SHAPE cell, and a FACE cell, each carrying a Korean/English label
(e.g. ``red_0``, ``circle_3``, ``mymelody_2``) plus an FFT amp/phase
block. The trainer doesn't try to *understand* the labels — it just
stores them.

This tool runs the post-training discovery step the integration spec
calls for:

  1. Load the .sss.
  2. Per cell type (COLOR / SHAPE / FACE), cluster cells by FFT-amp
     similarity. Cells whose amps are close share an underlying
     pattern — "all the rose-coloured COLOR cells", "all the round
     SHAPE cells".
  3. For each cluster, aggregate the bare-stem labels (split on
     ``_`` to drop the trainer's per-image suffix) and pick the
     most-frequent stem as the cluster name.
  4. Emit a JSON sidecar mapping stem → cluster id per type, plus
     per-cluster member rosters and centroids.

The sidecar lives next to the model (``model.sss`` →
``model.sss.cluster.json``) so downstream generators can route
prompt morphemes straight to a cluster id and synthesise from the
centroid amp without re-running ce_distance against every cell.

Pure stdlib + numpy. No cv2, no scipy, no scikit-learn — the
clustering routine is a couple dozen lines of Euclidean
greedy-nearest-neighbour, integer-friendly enough to run in Termux.

Usage:
    python3 -m tools.sss_cluster MODEL.sss \\
        [--out PATH.json] [--top-k N] [--threshold F]

Defaults:
    --out          MODEL.sss.cluster.json
    --top-k        3      labels reported per cluster (most-frequent first)
    --threshold    0.35   normalised radius — anything within this fraction
                          of the mean inter-cell distance joins the
                          cluster. Lower = more clusters.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Use the stable load_sss_model + SSSCell types that tools/sss_restore.py
# already exposes — saves us redoing the v8/v9 parser.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np                                          # noqa: E402
from tools.sss_restore import (                             # noqa: E402
    CE_COLOR, CE_FACE, CE_SHAPE, SSSCell, SSSModel, load_sss_model,
)


_TYPE_NAMES = {CE_COLOR: "color", CE_SHAPE: "shape", CE_FACE: "face"}


def _flatten_amp(amp: np.ndarray) -> np.ndarray:
    """Cells of different sizes (COLOR vs FACE vs SHAPE) need a common
    shape before we can compare them. Flatten everything to a 1-D
    float vector. The per-type clustering loop already keeps types
    separate, so cross-type vector lengths never collide."""
    a = np.asarray(amp, dtype=np.float32).reshape(-1)
    # Empty-amp fallback (e.g. truncated v8 model) → zero vector keeps
    # downstream norms finite.
    if a.size == 0:
        return np.zeros(1, dtype=np.float32)
    return a


def _stem(label: str) -> str:
    """The 2-column trainer suffixes labels with ``_<idx>`` to keep
    same-word cells unique on disk (`red_0`, `red_1`, ...). Stripping
    the trailing ``_<digits>`` recovers the morpheme.

    The 4-column legacy trainer (one cell per (label, type)) doesn't
    use the suffix, so the regex is a no-op there. Either way the
    stem is what we cluster on."""
    m = re.match(r"^(.*?)_(\d+)$", label)
    return m.group(1) if m else label


def _pairwise_dists(vectors: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix.

    `vectors` is (N, D) float32. Returns (N, N) float32. We use the
    expanded form ‖a−b‖² = ‖a‖² − 2·a·b + ‖b‖² so the heavy work is
    one matmul; the matrix is symmetric and the diagonal is forced to
    zero (numerical noise can leave -1e-7 there, which would NaN the
    sqrt). For the model sizes we care about (< 1000 cells per type),
    this is well below 1 GB."""
    sq = (vectors * vectors).sum(axis=1)              # (N,)
    cross = vectors @ vectors.T                       # (N, N)
    sq_dist = sq[:, None] - 2.0 * cross + sq[None, :]
    np.fill_diagonal(sq_dist, 0.0)
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.sqrt(sq_dist).astype(np.float32)


def _cluster_threshold(distmat: np.ndarray, base: float) -> float:
    """Convert the user's normalised radius into an absolute distance.

    `base` is the pre-clip user threshold (typically 0.35 = 35 %% of
    the mean off-diagonal distance). For an N×N distance matrix the
    mean is taken over the strictly-upper triangle (skips the zero
    diagonal). When fewer than two cells participate we fall back to
    the smallest non-zero element — a one-cell pool always becomes
    a one-cell cluster anyway."""
    n = distmat.shape[0]
    if n <= 1:
        return float("inf")
    iu = np.triu_indices(n, k=1)
    upper = distmat[iu]
    if upper.size == 0:
        return float("inf")
    mean_d = float(upper.mean())
    if mean_d <= 0.0:
        return 1e-6
    return base * mean_d


def _greedy_cluster(distmat: np.ndarray,
                    threshold: float) -> list[list[int]]:
    """Single-link greedy clustering.

    Walk cells in original order; each cell joins the nearest cluster
    whose nearest member is within `threshold`. With a fresh cell
    that has no cluster within range, start a new cluster.

    Single-link can chain into long thin clusters under tight data —
    that's a feature here: it means cells along a smooth gradient
    (red → orange → yellow) end up in one cluster instead of forcing
    arbitrary split points. The user can tighten `--threshold` to
    break the chain when the resulting groups are too coarse."""
    n = distmat.shape[0]
    clusters: list[list[int]] = []
    cell_to_cluster = [-1] * n
    for i in range(n):
        best_c = -1
        best_d = threshold
        for ci, members in enumerate(clusters):
            # Single-link: distance to cluster = min distance to any member.
            d = distmat[i, members].min()
            if d < best_d:
                best_d = d
                best_c = ci
        if best_c >= 0:
            clusters[best_c].append(i)
            cell_to_cluster[i] = best_c
        else:
            clusters.append([i])
            cell_to_cluster[i] = len(clusters) - 1
    return clusters


def _label_clusters(cells: list[SSSCell],
                    clusters: list[list[int]],
                    top_k: int) -> list[dict]:
    """For each cluster compute centroid + top-k stem labels.

    `top_k` is reported in descending frequency. Ties broken
    deterministically by alphabetical order so two runs over the
    same model produce the same JSON."""
    out: list[dict] = []
    for ci, members in enumerate(clusters):
        stems = collections.Counter(_stem(cells[i].label) for i in members)
        # Stable sort: (-count, label)
        ranked = sorted(stems.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [{"morpheme": s, "count": c} for s, c in ranked[:top_k]]

        # Centroid: mean of flat amps. Stored as a Python list so the
        # JSON sidecar is human-readable; consumers can np.asarray it.
        amps = np.stack([_flatten_amp(cells[i].amp) for i in members], axis=0)
        centroid = amps.mean(axis=0).tolist()

        out.append({
            "cluster_id":  ci,
            "size":        len(members),
            "top_labels":  top,
            "members":     [cells[i].label for i in members],
            "centroid_dim": len(centroid),
            "centroid":    centroid,
        })
    return out


def _cells_of(model: SSSModel, ctype: int) -> list[SSSCell]:
    return [c for c in model.cells if c.type == ctype]


def cluster_model(model_path: str | os.PathLike,
                  *,
                  threshold: float = 0.35,
                  top_k: int = 3) -> dict:
    """Run the full cluster discovery pass over a .sss model.

    Returns a dict ready for json.dump: per-type cluster lists plus a
    `morpheme_index` table mapping each discovered stem to the cluster
    id it most strongly belongs to (resolves ties by largest cluster
    size). The morpheme_index is what the prompt-routing path consumes
    — given a token, look it up and you get back the type → cluster_id
    pair to synthesise from."""
    model = load_sss_model(str(model_path))

    by_type: dict[str, list[dict]] = {}
    morpheme_index: dict[str, dict[str, int]] = {}

    for ctype, ctype_name in _TYPE_NAMES.items():
        cells = _cells_of(model, ctype)
        if not cells:
            by_type[ctype_name] = []
            continue

        amps = np.stack([_flatten_amp(c.amp) for c in cells], axis=0)
        # NF varies by type, but all cells of the same type share the
        # same amp layout (sss_train.py guarantees this), so vectors
        # are commensurate inside each type.
        if amps.shape[0] == 1:
            # Single-cell shortcut: don't bother with the distance
            # matrix.
            clusters = [[0]]
        else:
            distmat = _pairwise_dists(amps)
            tau = _cluster_threshold(distmat, threshold)
            clusters = _greedy_cluster(distmat, tau)

        labelled = _label_clusters(cells, clusters, top_k=top_k)
        by_type[ctype_name] = labelled

        # Build per-type morpheme → cluster_id index. A morpheme can
        # appear in several clusters (rare but possible, e.g. if
        # "red" labels diverged into two amp groups during training);
        # we point it at the cluster where it dominates by frequency.
        per_morph = collections.defaultdict(list)
        for cl in labelled:
            for stem_entry in cl["top_labels"]:
                per_morph[stem_entry["morpheme"]].append(
                    (cl["cluster_id"], stem_entry["count"], cl["size"])
                )
        winner: dict[str, int] = {}
        for stem, hits in per_morph.items():
            # Choose the cluster where this stem appears most often;
            # break ties by largest cluster (more evidence).
            hits.sort(key=lambda t: (-t[1], -t[2]))
            winner[stem] = hits[0][0]
        morpheme_index[ctype_name] = winner

    return {
        "model_path":     str(model_path),
        "model_height":   model.height,
        "model_width":    model.width,
        "model_nf":       model.nf,
        "model_nf_low":   model.nf_low,
        "threshold":      threshold,
        "top_k":          top_k,
        "by_type":        by_type,
        "morpheme_index": morpheme_index,
    }


def _strip_centroid(report: dict) -> dict:
    """Return a deep-ish copy of `report` with each cluster's centroid
    list dropped. The centroid pushes the JSON into multi-megabyte
    territory for medium-sized models, which is fine for a sidecar
    consumed by Python tooling but bad for terminal `--print` output.
    """
    slim = dict(report)
    slim["by_type"] = {}
    for tname, clusters in report["by_type"].items():
        slim["by_type"][tname] = [
            {k: v for k, v in cl.items() if k != "centroid"}
            for cl in clusters
        ]
    return slim


def _format_summary(report: dict) -> str:
    """Plain-text summary for terminal output."""
    lines = []
    lines.append(f"model: {report['model_path']}")
    lines.append(
        f"  H={report['model_height']} W={report['model_width']} "
        f"NF={report['model_nf']} NF_LOW={report['model_nf_low']}"
    )
    for tname, clusters in report["by_type"].items():
        lines.append(f"  type={tname}  clusters={len(clusters)}")
        for cl in clusters[:8]:
            tags = ", ".join(
                f"{t['morpheme']}×{t['count']}" for t in cl["top_labels"]
            )
            lines.append(f"    [{cl['cluster_id']:>2}] size={cl['size']:>3}  {tags}")
        if len(clusters) > 8:
            lines.append(f"    … {len(clusters) - 8} more")
        idx = report["morpheme_index"].get(tname, {})
        if idx:
            lines.append("    morpheme_index: "
                         + ", ".join(f"{k}→{v}" for k, v in
                                     sorted(idx.items())[:8])
                         + (" …" if len(idx) > 8 else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m tools.sss_cluster",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="path to a .sss model file")
    p.add_argument("--out",
                   help="output JSON path (default: <model>.cluster.json)")
    p.add_argument("--top-k", type=int, default=3,
                   help="labels reported per cluster (default 3)")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="cluster radius as fraction of mean inter-cell "
                        "distance (default 0.35; lower = more, smaller "
                        "clusters)")
    p.add_argument("--print", action="store_true",
                   help="print a human summary to stdout in addition to "
                        "writing the JSON sidecar")
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"sss_cluster: model not found: {args.model}", file=sys.stderr)
        return 2

    report = cluster_model(
        args.model, threshold=args.threshold, top_k=args.top_k,
    )

    out_path = args.out or (str(args.model) + ".cluster.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"sss_cluster: wrote {out_path}", file=sys.stderr)

    if args.print:
        print(_format_summary(_strip_centroid(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
