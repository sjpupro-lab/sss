"""End-to-end regression for scripts/sss_train_motif_memory.

Pipeline under test:
  1. Generate per-motif training data (re-uses the test_train_primitive
     synthetic generators).
  2. Train primitive motifs into a .npz.
  3. Generate compound scenes: each scene is 256×256 with two motifs
     stacked vertically — "horizontal_stripes on top, circular_pattern
     below" and "circular_pattern on top, noise_texture below".
  4. Run the memory builder against those scenes.
  5. Confirm the extracted relations contain the right (src, dst,
     ABOVE/BELOW, |dy| ≈ 0.3) triples with high weight.

The relations test is the headline check; the heatmap update and
identity extraction are validated by sanity (heatmap_update non-zero
for activated motifs, one identity per scene).

Run:
    python3 tools/test_train_motif_memory.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools import sss_image_io                                     # noqa: E402
from tools.test_train_primitive import (                           # noqa: E402
    GENERATORS as PRIMITIVE_GENERATORS,
    _generate_dataset,
)
from scripts import sss_train_motif_memory as _mem                 # noqa: E402


SCAN_RES = _mem.SCAN_RES


# ── Compound-scene synthesis ──────────────────────────────────────

def _paste_64_into_256(canvas: np.ndarray, patch: np.ndarray,
                       cy: int, cx: int) -> None:
    """Paste a 64×64 patch centered at (cy, cx) into the 256×256
    canvas, clipping safely if the patch crosses the canvas edge."""
    ph, pw, _ = patch.shape
    y0 = cy - ph // 2; y1 = y0 + ph
    x0 = cx - pw // 2; x1 = x0 + pw
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    sy1 = ph - max(0, y1 - canvas.shape[0])
    sx1 = pw - max(0, x1 - canvas.shape[1])
    y0c = max(0, y0); y1c = min(canvas.shape[0], y1)
    x0c = max(0, x0); x1c = min(canvas.shape[1], x1)
    canvas[y0c:y1c, x0c:x1c] = patch[sy0:sy1, sx0:sx1]


def _make_scene(top_label: str, bot_label: str,
                rng: np.random.Generator,
                size: int = SCAN_RES,
                patch_dim: int = 128) -> np.ndarray:
    """Compound 256×256 scene: top half holds a patch of `top_label`,
    bottom half holds a patch of `bot_label`. Patches are generated
    at 128×128 (not 64×64) so the matcher's 64×64 scan window can sit
    *entirely inside* a planted motif — otherwise scan windows pick
    up grey-background edges and the envelope shifts enough to break
    recognition. The trained motifs represent full 64×64 frames of
    one texture; the scene must give the scan window the same.

    Generators accept (h, w) and produce a fresh texture at any size,
    so we ask for 128×128 directly rather than tiling a 64×64 sample."""
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    for label, cy in ((top_label, size // 4),
                      (bot_label, 3 * size // 4)):
        patch = PRIMITIVE_GENERATORS[label](rng, h=patch_dim, w=patch_dim)
        _paste_64_into_256(canvas, patch, cy=cy, cx=size // 2)
    return canvas


def _write_scenes(scene_dir: str, pairs: list[tuple[str, str]],
                  per_pair: int = 10, seed: int = 1) -> dict[str, str]:
    """Write `per_pair` PPMs for every (top, bottom) pair, plus a
    labels.tsv that names each scene "<top>_above_<bottom>". Returns
    the labels map for downstream checks."""
    os.makedirs(scene_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = {}
    for top, bot in pairs:
        for i in range(per_pair):
            scene = _make_scene(top, bot, rng)
            fname = f"{top}__{bot}__{i:02d}.ppm"
            sss_image_io.write_ppm(os.path.join(scene_dir, fname),
                                   scene[..., ::-1])  # write expects BGR
            labels[fname] = f"{top}_above_{bot}"
    return labels


# ── Test ──────────────────────────────────────────────────────────

def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prim_root = os.path.join(tmp, "primitive")
        scenes    = os.path.join(tmp, "labeled")
        labels_t  = os.path.join(tmp, "labels.tsv")
        motifs_p  = os.path.join(tmp, "primitive_motifs.npz")
        memory_p  = os.path.join(tmp, "motif_memory.npz")

        # ── 1+2. Primitive training -------------------------------
        _generate_dataset(prim_root, samples_per_class=30, seed=0)
        cp = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts",
                                          "sss_train_primitive.py"),
             "--data", prim_root, "--out", motifs_p],
            capture_output=True, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stdout + cp.stderr)
            return 1
        print(cp.stdout.strip())

        # ── 3. Compound scenes ------------------------------------
        pairs = [
            ("horizontal_stripes", "circular_pattern"),
            ("circular_pattern",   "noise_texture"),
        ]
        labels = _write_scenes(scenes, pairs, per_pair=10, seed=1)
        with open(labels_t, "w", encoding="utf-8") as f:
            for k, v in labels.items():
                f.write(f"{k}\t{v}\n")
        print(f"OK  scenes      : {len(labels)} compound 256×256 images")

        # ── 4. Memory build ---------------------------------------
        cp = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts",
                                          "sss_train_motif_memory.py"),
             "--motifs",  motifs_p,
             "--labeled", scenes,
             "--labels-tsv", labels_t,
             "--out", memory_p,
             "--threshold", "0.7",
             "--weight-floor", "0.3",
             "--verbose"],
            capture_output=True, text=True)
        sys.stdout.write(cp.stdout)
        if cp.returncode != 0:
            sys.stderr.write(cp.stderr)
            return 1

        # ── 5. Inspect the memory ---------------------------------
        mem = np.load(memory_p, allow_pickle=False)
        relations = mem["relations"]
        idents_label = list(mem["identities_labels"])
        idents_lens  = list(mem["identities_motif_lens"])
        flat_ids     = list(mem["identities_motif_flat"])
        print(f"\nrelations.shape={relations.shape}, "
              f"identities={len(idents_label)}, "
              f"scanned={int(mem['scanned_images'][0])}")

        # Resolve motif labels for readable output.
        motifs_npz = np.load(motifs_p, allow_pickle=False)
        motif_labels = [str(s) for s in motifs_npz["labels"]]
        print("motif labels:", motif_labels)

        # ── Relations must contain the expected pair triples. ─────
        # Spec convention: dy = (center_dst − center_src). dy > 0 →
        # dst is below src → relation_type is BELOW. So a scene
        # "horizontal_stripes (top) over circular_pattern (bottom)"
        # produces the pair (src=horizontal, dst=circular) with
        # dy ≈ +0.5 → BELOW. Reading the tuple naturally: (A, B,
        # BELOW) means "B is BELOW A".
        wanted = {
            ("horizontal_stripes", "circular_pattern", _mem.RELATION_BELOW),
            ("circular_pattern",   "noise_texture",    _mem.RELATION_BELOW),
        }
        rel_by_key = {}
        for r in relations:
            src, dst, dx, dy, rtype, w = r
            key = (motif_labels[int(src)], motif_labels[int(dst)],
                   int(rtype))
            rel_by_key[key] = (float(dx), float(dy), float(w))

        print("\nExpected BELOW relations (dst is below src):")
        # The wanted pair appears in 10 of 20 scenes (the other 10
        # scenes use the *other* pair), so the maximum achievable
        # weight per relation is 10 / 20 = 0.5. We assert ≥ 0.45
        # (small slack for the rare case where centroid-based
        # classification happens to land in the NEAR/AROUND band).
        for src_lab, dst_lab, rtype in wanted:
            hit = rel_by_key.get((src_lab, dst_lab, rtype))
            assert hit is not None, (
                f"missing relation ({src_lab!r}, {dst_lab!r}, type={rtype}); "
                f"available keys: {list(rel_by_key.keys())}")
            dx_m, dy_m, w_m = hit
            print(f"  {src_lab:<24s} → {dst_lab:<24s} "
                  f"dx={dx_m:+.3f} dy={dy_m:+.3f} w={w_m:.2f}")
            assert w_m >= 0.45, \
                f"({src_lab}, {dst_lab}, BELOW) weight {w_m:.2f} < 0.45"
            assert abs(dx_m) < 0.05, \
                f"({src_lab}, {dst_lab}) dx {dx_m:+.3f} | should be ≈ 0"
            assert 0.2 < dy_m < 0.7, \
                f"({src_lab}, {dst_lab}) dy {dy_m:+.3f} | should be ≈ +0.5"
        print("OK  BELOW rels  : both expected (src, dst, BELOW) found, "
              "weight ≥ 0.45 (max possible 0.5 with two scene types), "
              "dx ≈ 0, dy in (0.2, 0.7)")

        # ── Inverse (B, A, ABOVE) must also exist since the matcher
        #    emits ordered pairs (both directions per image). ─────
        for src_lab, dst_lab, _ in wanted:
            inv_key = (dst_lab, src_lab, _mem.RELATION_ABOVE)
            hit = rel_by_key.get(inv_key)
            assert hit is not None, (
                f"missing inverse {inv_key} — matcher should emit both "
                "ordered pairs per image")
            assert hit[2] >= 0.45, (
                f"inverse ABOVE weight {hit[2]:.2f} < 0.45 for {inv_key}")
        print("OK  ABOVE rels  : inverse (dst, src, ABOVE) found for each")

        # ── Heatmap update has non-zero rows for activated motifs.
        hm = mem["position_heatmap_update"]
        n_motifs_with_heat = sum(1 for i in range(hm.shape[0])
                                 if float(hm[i].max()) > 0.0)
        assert n_motifs_with_heat >= 3, (
            f"only {n_motifs_with_heat} motifs picked up a heatmap; "
            "the compound scenes should activate at least 3 of 4")
        print(f"OK  heatmap upd : {n_motifs_with_heat} motifs picked up "
              "non-zero position heatmaps")

        # ── Identity extraction: one identity per scanned image,
        #    each with at least 2 fired motifs. ────────────────────
        assert len(idents_label) == int(mem["scanned_images"][0]), (
            f"identities={len(idents_label)} but scanned="
            f"{int(mem['scanned_images'][0])}")
        # Walk flat motif-ids array using lens.
        off = 0
        for i, n in enumerate(idents_lens):
            ids = flat_ids[off:off + int(n)]
            assert int(n) >= 2, \
                f"identity[{i}] {idents_label[i]!r} has {n} motifs; "\
                "expected ≥ 2 for a 2-motif compound scene"
            off += int(n)
        print(f"OK  identities  : {len(idents_label)} identities, "
              "all with ≥ 2 motif_ids")

        print("\nPASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
