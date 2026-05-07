"""Smoke test for tools.sss_cluster.

Verifies that the morpheme ↔ cluster discovery pass:
  * loads a real .sss model produced by scripts/sss_train.py
  * groups cells of each type into one or more clusters
  * surfaces the dominant stem ("red", "circle", ...) per cluster
  * builds a morpheme_index that points back to a cluster id

The test trains a tiny 2-column model on the bundled sss_demo
fixture so we exercise the per-image-cell path (the only path where
clustering has anything to do — the 4-column path emits one cell
per label and every cluster is trivially a singleton).

Run:
    make pybridge        # not strictly required (cluster doesn't use ctypes)
    python3 -m tools.test_sss_cluster
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Repo root has data/sss_demo/labels.tsv that the v1 demo trainer ships.
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO  = os.path.join(ROOT, "data", "sss_demo")


def _train_2col_model(out_dir: str) -> str:
    """Build a 2-column labels.tsv from the bundled 4-column file and
    train a tiny .sss for the test. Two-column rows produce one cell
    per (image, word) so the clustering pass actually has something
    to merge."""
    src = os.path.join(DEMO, "labels.tsv")
    dst = os.path.join(out_dir, "labels.tsv")
    with open(src, encoding="utf-8") as fp_in, \
         open(dst, "w", encoding="utf-8") as fp_out:
        for line in fp_in:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                # filename + "color shape face" → 2-column
                fp_out.write(f"{parts[0]}\t{parts[1]} {parts[2]} {parts[3]}\n")

    model = os.path.join(out_dir, "test_cluster.sss")
    cmd = [
        "python3", os.path.join(ROOT, "scripts", "sss_train.py"),
        "--labels", dst,
        "--root",   DEMO,
        "--out",    model,
        "--size",   "64",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return model


def main() -> int:
    fails = 0
    def check(cond, msg):
        nonlocal fails
        if cond:
            print(f"OK  {msg}")
        else:
            print(f"FAIL {msg}")
            fails += 1

    with tempfile.TemporaryDirectory() as tmp:
        try:
            model_path = _train_2col_model(tmp)
        except subprocess.CalledProcessError as e:
            print(f"FAIL train: {e}")
            return 1

        # cluster_model returns a dict; we don't need the .json sidecar
        # for the assertions, only the in-memory report.
        from tools.sss_cluster import cluster_model
        report = cluster_model(model_path, threshold=0.5, top_k=4)

        # Trainer mints 18 images × 3 words = 18 cells per type — clustering
        # must produce *some* groups, not the trivial 1-per-cluster fallback.
        color_clusters = report["by_type"]["color"]
        shape_clusters = report["by_type"]["shape"]
        face_clusters  = report["by_type"]["face"]

        check(len(color_clusters) >= 2,
              f"color: {len(color_clusters)} clusters (≥ 2)")
        check(len(shape_clusters) >= 2,
              f"shape: {len(shape_clusters)} clusters (≥ 2)")
        check(len(face_clusters)  >= 2,
              f"face:  {len(face_clusters)} clusters (≥ 2)")

        # At least one cluster per type must contain multiple cells —
        # otherwise the discovery effectively did nothing useful.
        check(any(c["size"] >= 2 for c in color_clusters),
              "color: at least one multi-cell cluster")
        check(any(c["size"] >= 2 for c in shape_clusters),
              "shape: at least one multi-cell cluster")

        # The morpheme_index for COLOR must contain stems we put in
        # (red/green/blue). The trainer suffixes per image, so the stem
        # is what _stem() recovers — verify the round-trip lands.
        idx_color = report["morpheme_index"].get("color", {})
        check("red"   in idx_color, f"morpheme_index[color] has red:   {idx_color}")
        check("green" in idx_color, f"morpheme_index[color] has green: {idx_color}")
        check("blue"  in idx_color, f"morpheme_index[color] has blue:  {idx_color}")

        # Same for SHAPE: circle / square / triangle must appear.
        idx_shape = report["morpheme_index"].get("shape", {})
        check("circle"   in idx_shape, "morpheme_index[shape] has circle")
        check("square"   in idx_shape, "morpheme_index[shape] has square")
        check("triangle" in idx_shape, "morpheme_index[shape] has triangle")

        # Centroid sanity: every cluster's centroid_dim must be > 0
        # (empty-amp guard fires only on truncated v8 models).
        bad_centroid = [
            (tname, cl["cluster_id"]) for tname, cls in report["by_type"].items()
            for cl in cls if cl["centroid_dim"] <= 0
        ]
        check(not bad_centroid, f"every cluster has non-empty centroid {bad_centroid}")

    print()
    print(f"{'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s))")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
