#!/usr/bin/env python3
"""sss_train_motif_memory — Phase 3 motif relation + identity builder.

Design philosophy (same as primitive trainer):

The .sfb is a **feature spectral dictionary**. Relations and
identities describe how the abstract frequency-shape motifs co-occur
across larger labelled images — NOT how pixels of the training image
flow into a generator. The output preserves the Phase 2 invariants:

  * No phase ever stored.
  * No per-row FFT; we only re-run the primitive trainer's envelope
    pipeline on 32×32-grid patches and match each patch against the
    motif dictionary by amplitude shape.
  * No pixel coordinates land in the .sfb — only normalised
    (dx, dy) ∈ [-1, 1] direction vectors and a 16×16 position
    heatmap (downsampled from the 32×32 grid).

Algorithm:

  (1) Scan each labelled image (any size; auto-resized to 256×256
      for grid scanning so the matcher operates at a fixed scale).
  (2) Split into a 32×32 grid; extract a 64×64 patch around each
      grid cell's centre (with edge clamping).
  (3) Run the patch through the SAME envelope pipeline used by
      `sss_train_primitive` (DC-killed, zero-mean L2-normalised
      row / col / color envelopes + Sobel heatmap).
  (4) Weighted-cosine match against every motif. Above the
      MATCH_THRESHOLD (default 0.85) the motif is considered
      "activated" at this grid cell.

      final_sim = 0.35 × row + 0.35 × col + 0.20 × color + 0.10 × heat

  (5) For every pair (a, b) of activated motifs in the same image,
      compute the normalised centre delta:
          dx = (cx_b - cx_a) / image_width
          dy = (cy_b - cy_a) / image_height
      and classify into a Phase 2 relation_type:
          |dy| > 2|dx| and dy < 0  → SFB_RELATION_TYPE_ABOVE
          |dy| > 2|dx| and dy > 0  → SFB_RELATION_TYPE_BELOW
          |dx| > 2|dy| and dx < 0  → SFB_RELATION_TYPE_LEFT
          |dx| > 2|dy| and dx > 0  → SFB_RELATION_TYPE_RIGHT
          dx² + dy² < 0.05          → SFB_RELATION_TYPE_NEAR
          otherwise                 → SFB_RELATION_TYPE_AROUND
      Co-occurrence is counted per (src, dst, type) triple.

  (6) weight = co_occurrence_count / total_image_count. Triples with
      weight < WEIGHT_FLOOR (default 0.3) are dropped.

  (7) Position heatmap update: each motif's per-image grid-cell
      activations accumulate into a 32×32 surface (one heatmap per
      motif), downsampled by 2×2 to the 16×16 .sfb heatmap. The
      Phase 3 spec says "update the primitive heatmap" — we
      *replace* the primitive heatmap with the relation-time
      heatmap because the labelled-image scan provides truer
      spatial statistics (the primitive folder gave coarse Sobel
      density, this gives "where did the motif actually fire").

  (8) Identity extraction: every labelled image becomes an identity
      record whose `motif_ids` are the union of motifs that fired
      at any grid cell. The identity's label is the image's label
      (caller supplied) or its filename stem when no label map is
      present.

CLI:
    python scripts/sss_train_motif_memory.py \\
        --motifs   build/primitive_motifs.npz \\
        --labeled  dataset/labeled \\
        --out      build/motif_memory.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import sss_train_primitive as _prim   # noqa: E402
from tools import sss_image_io                     # noqa: E402


# ── Constants ─────────────────────────────────────────────────────
GRID_DIM       = 32                # 32×32 patch grid across the image
PATCH_SIZE     = _prim.PATCH_SIZE  # 64
SCAN_RES       = 256               # resize every labelled image to 256×256
HEATMAP_DIM    = _prim.HEATMAP_DIM # 16, downsamples 32→16 (2×2 blocks)

DEFAULT_THRESHOLD   = 0.85
DEFAULT_WEIGHT_FLOOR = 0.3
NEAR_RADIUS_SQ      = 0.05         # dx² + dy² < this → NEAR

# Phase 2 relation_type enums.
RELATION_ABOVE  = 0
RELATION_BELOW  = 1
RELATION_LEFT   = 2
RELATION_RIGHT  = 3
RELATION_NEAR   = 4
RELATION_AROUND = 5

IMAGE_EXTS = (".png", ".ppm", ".jpg", ".jpeg", ".bmp")


# ── Patch envelope ─────────────────────────────────────────────────

def _patch_envelope(patch_u8: np.ndarray) -> dict:
    """Run the primitive trainer's envelope pipeline on a single 64×64
    RGB patch. Returns the zero-mean L2-normalised envelopes plus the
    raw 16×16 Sobel heatmap. Same shape contract as one entry of the
    `_prim.train_motif_from_folder` per-image accumulators."""
    f = _prim._to_float_demeaned(patch_u8)
    row_raw   = _prim._envelope_row_freq(f)          # (128,) non-neg
    col_raw   = _prim._envelope_col_freq(f)          # (128,) non-neg
    color_raw = _prim._envelope_color_freq(f)        # (3, 128)
    heat      = _prim._sobel_heatmap_16(patch_u8)    # (16, 16)
    return {
        "row":   _prim._zero_mean_l2_normalise(row_raw),
        "col":   _prim._zero_mean_l2_normalise(col_raw),
        "color": np.stack(
            [_prim._zero_mean_l2_normalise(color_raw[c]) for c in range(3)],
            axis=0),
        # Heatmap stays in raw min-max form for similarity comparison;
        # zero-meaning a 256-cell heatmap is fine on the float scale.
        "heat":  _prim._zero_mean_l2_normalise(heat.ravel()),
        "heat_raw": heat,
    }


def _motif_envelope_view(motifs_npz: np.lib.npyio.NpzFile) -> list[dict]:
    """Wrap the stored motif arrays into per-motif env dicts that
    match `_patch_envelope`'s output shape. The motif file already
    holds zero-meaned L2-normalised envelopes (Phase 3 trainer), so
    no further transform is needed — we only flatten the heatmap to
    match the per-patch comparison shape."""
    n = int(motifs_npz["row_freq"].shape[0])
    out = []
    for i in range(n):
        heat_raw = motifs_npz["position_heatmap"][i]
        out.append({
            "label": str(motifs_npz["labels"][i]),
            "row":   motifs_npz["row_freq"][i],
            "col":   motifs_npz["col_freq"][i],
            "color": motifs_npz["color_freq"][i],
            "heat":  _prim._zero_mean_l2_normalise(heat_raw.ravel()),
            "heat_raw": heat_raw,
        })
    return out


def _weighted_similarity(a: dict, b: dict) -> float:
    """Phase 3 weighted-cosine matcher.
       0.35 row + 0.35 col + 0.20 color (avg of 3 channels) + 0.10 heat.
    Vectors are zero-mean L2-normalised, so plain np.dot gives cosine."""
    row   = float(np.dot(a["row"],   b["row"]))
    col   = float(np.dot(a["col"],   b["col"]))
    color = float(
        np.dot(a["color"][0], b["color"][0])
        + np.dot(a["color"][1], b["color"][1])
        + np.dot(a["color"][2], b["color"][2])
    ) / 3.0
    heat  = float(np.dot(a["heat"], b["heat"]))
    return 0.35 * row + 0.35 * col + 0.20 * color + 0.10 * heat


# ── Image scan ────────────────────────────────────────────────────

def _read_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        img = sss_image_io.read_png(path)
    elif ext == ".ppm":
        img = sss_image_io.read_ppm(path)
    else:
        raise ValueError(f"unsupported extension {ext!r} for {path!r}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    # sss_image_io returns BGR; flip to RGB for trainer alignment.
    return np.ascontiguousarray(img[..., ::-1], dtype=np.uint8)


def _resize_256_rgb(img: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize to 256×256 RGB. Keeps the scan grid
    at a fixed scale across heterogeneous input resolutions."""
    h, w = img.shape[:2]
    if h == SCAN_RES and w == SCAN_RES:
        return np.ascontiguousarray(img, dtype=np.uint8)
    yi = np.linspace(0, h - 1, SCAN_RES).round().astype(np.int64)
    xi = np.linspace(0, w - 1, SCAN_RES).round().astype(np.int64)
    yi = np.clip(yi, 0, max(0, h - 1))
    xi = np.clip(xi, 0, max(0, w - 1))
    return np.ascontiguousarray(img[yi[:, None], xi[None, :], :], dtype=np.uint8)


def _grid_centres() -> np.ndarray:
    """The (cy, cx) centre of every 32×32 grid cell on a SCAN_RES×SCAN_RES
    image, as integer pixel positions. Returns (32, 32, 2) → [cy, cx]."""
    step = SCAN_RES / GRID_DIM
    coords = (np.arange(GRID_DIM) + 0.5) * step
    cy = coords.astype(np.int64)
    cx = coords.astype(np.int64)
    grid = np.stack(np.meshgrid(cy, cx, indexing="ij"), axis=-1)
    return grid                                          # (32, 32, 2)


def _extract_patch(img: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """64×64 patch centred at (cy, cx). Edge clamping (mirror would
    be cleaner spectrally but the SCAN_RES=256 grid has enough margin
    that clamping rarely activates and the bias is uniform across
    candidate motifs)."""
    half = PATCH_SIZE // 2
    y0 = max(0, cy - half); y1 = min(SCAN_RES, y0 + PATCH_SIZE)
    x0 = max(0, cx - half); x1 = min(SCAN_RES, x0 + PATCH_SIZE)
    # Adjust if at the boundary so the patch is always 64×64.
    if y1 - y0 < PATCH_SIZE: y0 = y1 - PATCH_SIZE
    if x1 - x0 < PATCH_SIZE: x0 = x1 - PATCH_SIZE
    return img[y0:y1, x0:x1]


def _classify_relation(dx: float, dy: float) -> int:
    """(dx, dy) in normalised [-1, 1] → relation_type enum. Order
    matters: NEAR is checked LAST so a slight pixel offset that still
    lies inside the near radius doesn't get tagged as AROUND."""
    adx = abs(dx); ady = abs(dy)
    if ady > 2 * adx and dy < 0: return RELATION_ABOVE
    if ady > 2 * adx and dy > 0: return RELATION_BELOW
    if adx > 2 * ady and dx < 0: return RELATION_LEFT
    if adx > 2 * ady and dx > 0: return RELATION_RIGHT
    if dx * dx + dy * dy < NEAR_RADIUS_SQ: return RELATION_NEAR
    return RELATION_AROUND


# ── Per-image scan ────────────────────────────────────────────────

def scan_image(img_u8_rgb: np.ndarray, motif_views: list,
               threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Return everything one image contributes to the memory:

        {
          "activations": list of (motif_id, grid_y, grid_x, score),
          "fired_motifs": set of motif_id (any-cell activation),
          "heatmap_32": (32, 32) uint16 per-motif activation count …
            actually a dict {motif_id: (32, 32) np.uint16}, mostly
            sparse, keyed only by motifs that fired.
        }

    `motif_views` is the list produced by `_motif_envelope_view`."""
    img = _resize_256_rgb(img_u8_rgb)
    centres = _grid_centres()

    activations = []
    fired = set()
    heat_per_motif: dict[int, np.ndarray] = {}

    for gy in range(GRID_DIM):
        for gx in range(GRID_DIM):
            cy, cx = int(centres[gy, gx, 0]), int(centres[gy, gx, 1])
            patch = _extract_patch(img, cy, cx)
            env = _patch_envelope(patch)
            best_score = 0.0
            best_id = -1
            for mid, m in enumerate(motif_views):
                s = _weighted_similarity(env, m)
                if s > best_score:
                    best_score = s
                    best_id = mid
            if best_id >= 0 and best_score >= threshold:
                activations.append((best_id, gy, gx, best_score))
                fired.add(best_id)
                if best_id not in heat_per_motif:
                    heat_per_motif[best_id] = np.zeros(
                        (GRID_DIM, GRID_DIM), np.uint16)
                heat_per_motif[best_id][gy, gx] += 1

    return {
        "activations": activations,
        "fired_motifs": fired,
        "heat_per_motif": heat_per_motif,
    }


# ── Driver ────────────────────────────────────────────────────────

def _load_labels_tsv(path: Path | None,
                     base_dir: Path) -> dict[str, str]:
    """Optional labels.tsv: rows are `<filename>\\t<label>`. When
    absent the identity label defaults to the filename stem."""
    if not path or not path.exists():
        return {}
    labels = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            labels[parts[0].strip()] = parts[1].strip()
    return labels


def _aggregate_relations(per_image: list, motif_count: int,
                         weight_floor: float) -> tuple[list, np.ndarray]:
    """Pass 2: roll up per-image activations into the relation list
    + cumulative per-motif heatmap. Returns:

        (relations_list, heatmap_32_per_motif)

    relations_list: [(src, dst, dx_mean, dy_mean, type, weight), ...]
    heatmap shape: (motif_count, 32, 32) float32.

    Per-image relation counting uses the **centroid** of each fired
    motif's activated grid cells. Without centroid-based reduction we'd
    count each grid-cell pair separately and `weight = count / n_images`
    would blow past 1.0 (with O(activations²) pairs per image). The
    spec's `weight ∈ [0, 1]` only makes sense when each
    (src, dst, type) triple is counted at most once per image."""
    cooc: dict = defaultdict(lambda: [0, 0.0, 0.0])
    heatmap_32 = np.zeros((motif_count, GRID_DIM, GRID_DIM), np.float32)

    n_images = len(per_image)
    centres = _grid_centres()
    cell_xy = centres[..., ::-1].astype(np.float32)        # (32, 32, [x, y])

    for sample in per_image:
        # Accumulate per-motif heatmap.
        for mid, hm in sample["heat_per_motif"].items():
            heatmap_32[mid] += hm.astype(np.float32)

        # Centroid of each fired motif's grid cells (in pixel space).
        per_motif_xs: dict[int, list[float]] = defaultdict(list)
        per_motif_ys: dict[int, list[float]] = defaultdict(list)
        for mid, gy, gx, _ in sample["activations"]:
            per_motif_xs[mid].append(float(cell_xy[gy, gx, 0]))
            per_motif_ys[mid].append(float(cell_xy[gy, gx, 1]))

        centroids = {
            mid: (sum(xs) / len(xs), sum(per_motif_ys[mid]) / len(per_motif_ys[mid]))
            for mid, xs in per_motif_xs.items()
        }
        fired = sorted(centroids.keys())
        if len(fired) < 2:
            continue
        # All ordered pairs (mi, mj) with mi != mj — each pair counts
        # once per image. With ordered pairs we get both ABOVE and
        # BELOW for the same physical layout (a single (mi → mj) and
        # the inverse (mj → mi)), which the matcher can look up from
        # either side.
        for i, mi in enumerate(fired):
            xi, yi = centroids[mi]
            for j, mj in enumerate(fired):
                if i == j: continue
                xj, yj = centroids[mj]
                dx = (xj - xi) / SCAN_RES
                dy = (yj - yi) / SCAN_RES
                rtype = _classify_relation(dx, dy)
                key = (mi, mj, rtype)
                slot = cooc[key]
                slot[0] += 1
                slot[1] += dx
                slot[2] += dy

    relations = []
    for (src, dst, rtype), (count, sum_dx, sum_dy) in cooc.items():
        weight = count / float(max(1, n_images))
        if weight < weight_floor:
            continue
        relations.append((
            int(src), int(dst),
            float(sum_dx / count), float(sum_dy / count),
            int(rtype), float(weight),
        ))

    return relations, heatmap_32


def _downsample_32_to_16(heat_32: np.ndarray) -> np.ndarray:
    """(M, 32, 32) → (M, 16, 16) via 2×2 averaging, then per-motif
    min-max normalisation into [0, 1]. Returns float32."""
    M = heat_32.shape[0]
    if M == 0:
        return np.zeros((0, HEATMAP_DIM, HEATMAP_DIM), np.float32)
    blk = heat_32.reshape(M, HEATMAP_DIM, 2, HEATMAP_DIM, 2)
    avg = blk.mean(axis=(2, 4)).astype(np.float32)
    out = np.empty_like(avg)
    for i in range(M):
        lo = float(avg[i].min())
        hi = float(avg[i].max())
        if hi - lo > 1e-9:
            out[i] = (avg[i] - lo) / (hi - lo)
        else:
            out[i] = 0.0
    return out


def build_memory(motifs_path: Path, labeled_dir: Path,
                 labels_tsv: Path | None = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 weight_floor: float = DEFAULT_WEIGHT_FLOOR,
                 verbose: bool = False) -> dict:
    """Top-level entry. Returns a dict serialised by `_save_npz`.

    Keys:
        relations      — (R, 6) f32  [src, dst, dx, dy, type, weight]
        identities_*   — parallel arrays (labels + motif_ids list + conf)
        position_heatmap_update — (M, 16, 16) f32 in [0, 1]
        motif_activation_count  — (M,) u32 total per-motif firing counts
        scanned_images          — int total images that contributed
    """
    motifs_npz = np.load(motifs_path, allow_pickle=False)
    motif_views = _motif_envelope_view(motifs_npz)
    motif_count = len(motif_views)

    image_paths = sorted(p for p in Path(labeled_dir).rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    label_map = _load_labels_tsv(labels_tsv, labeled_dir)

    per_image_samples = []
    identities_labels: list[str] = []
    identities_motifs: list[list[int]] = []
    activation_count = np.zeros(motif_count, np.uint32)

    t_start = time.perf_counter()
    for ipath in image_paths:
        rgb = _read_image(str(ipath))
        sample = scan_image(rgb, motif_views, threshold=threshold)
        per_image_samples.append(sample)
        for mid in sample["fired_motifs"]:
            activation_count[mid] += 1
        if sample["fired_motifs"]:
            ident_label = label_map.get(ipath.name, ipath.stem)
            identities_labels.append(ident_label)
            identities_motifs.append(sorted(int(m) for m in sample["fired_motifs"]))
        if verbose:
            n_act = len(sample["activations"])
            n_fired = len(sample["fired_motifs"])
            print(f"  {ipath.name:<48s} activations={n_act:>4d}  "
                  f"fired={n_fired:>3d}")

    relations, heatmap_32 = _aggregate_relations(
        per_image_samples, motif_count, weight_floor)
    heatmap_16 = _downsample_32_to_16(heatmap_32)
    dt = time.perf_counter() - t_start

    return {
        "motif_count":    int(motif_count),
        "scanned_images": int(len(image_paths)),
        "scan_seconds":   float(dt),
        "relations":      np.array(relations, dtype=np.float32) if relations
                          else np.zeros((0, 6), np.float32),
        "identities_labels": np.array(identities_labels, dtype="U64"),
        "identities_motifs": identities_motifs,        # ragged → save below
        "position_heatmap_update": heatmap_16,
        "motif_activation_count":  activation_count,
    }


def _save_npz(mem: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ragged motif_ids → save as (id_arr, len_arr) for safe load.
    flat_ids = []
    lens = []
    for ids in mem["identities_motifs"]:
        flat_ids.extend(int(i) for i in ids)
        lens.append(len(ids))
    np.savez(
        out_path,
        relations=mem["relations"],
        identities_labels=mem["identities_labels"],
        identities_motif_flat=np.array(flat_ids, dtype=np.uint16),
        identities_motif_lens=np.array(lens, dtype=np.uint8),
        position_heatmap_update=mem["position_heatmap_update"],
        motif_activation_count=mem["motif_activation_count"],
        motif_count=np.array([mem["motif_count"]], dtype=np.uint32),
        scanned_images=np.array([mem["scanned_images"]], dtype=np.uint32),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motifs", required=True, type=Path,
                    help="primitive_motifs.npz from sss_train_primitive")
    ap.add_argument("--labeled", required=True, type=Path,
                    help="directory of labelled images")
    ap.add_argument("--labels-tsv", type=Path, default=None,
                    help="optional <filename>TAB<label> mapping; "
                         "default uses the filename stem as the label")
    ap.add_argument("--out", required=True, type=Path,
                    help="output .npz path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"weighted-cosine match threshold "
                         f"(default {DEFAULT_THRESHOLD})")
    ap.add_argument("--weight-floor", type=float,
                    default=DEFAULT_WEIGHT_FLOOR,
                    help=f"drop relations with weight below this "
                         f"(default {DEFAULT_WEIGHT_FLOOR})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.motifs.is_file():
        print(f"error: --motifs {args.motifs} not found", file=sys.stderr)
        return 2
    if not args.labeled.is_dir():
        print(f"error: --labeled {args.labeled} is not a directory",
              file=sys.stderr)
        return 2

    mem = build_memory(
        args.motifs, args.labeled, args.labels_tsv,
        threshold=args.threshold,
        weight_floor=args.weight_floor,
        verbose=args.verbose,
    )
    _save_npz(mem, args.out)
    print(f"wrote {args.out}  ({len(mem['relations'])} relations, "
          f"{len(mem['identities_labels'])} identities, "
          f"{mem['scanned_images']} images scanned in "
          f"{mem['scan_seconds']:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
