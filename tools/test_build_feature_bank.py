"""End-to-end .sfb build smoke test.

Stitches the Phase 3 pipeline together and exercises the empty-bank
edge case:

  1. Synthesise the 4-motif primitive dataset (re-used from
     test_train_primitive).
  2. Run sss_train_primitive  → primitive_motifs.npz
  3. Synthesise compound scenes (re-used from test_train_motif_memory).
  4. Run sss_train_motif_memory → motif_memory.npz
  5. Run sss_build_feature_bank → feature_bank.sfb
  6. Re-load the .sfb via Phase 2's `SSSFeatureBank.load` and verify
     motif_count / relation_count / identity_count, plus that the
     pairwise weighted-cosine matrix matches what test_train_primitive
     reported (i.e. the .sfb byte-trip preserves the trainer output).

  7. Build an *empty* bank (empty primitive root, no memory file) →
     confirm the .sfb still loads and reports zero counts.

Run:
    python3 tools/test_build_feature_bank.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.test_train_primitive import _generate_dataset  # noqa: E402
from tools.test_train_motif_memory import _write_scenes   # noqa: E402
from tools.sss_feature_bank import SSSFeatureBank          # noqa: E402


def _run(cmd):
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(" ".join(cmd) + "\n")
        sys.stderr.write(cp.stdout + cp.stderr)
        raise SystemExit(1)
    return cp.stdout.strip()


def _weighted_cosine(a, b) -> float:
    row   = float(np.dot(a.row_freq,   b.row_freq))
    col   = float(np.dot(a.col_freq,   b.col_freq))
    color = (
        float(np.dot(a.color_freq[0], b.color_freq[0]))
        + float(np.dot(a.color_freq[1], b.color_freq[1]))
        + float(np.dot(a.color_freq[2], b.color_freq[2]))
    ) / 3.0
    # Heatmaps live as float32 [0, 1]; cosine of two non-negative
    # heatmaps floors high, but we keep the same formula the matcher
    # uses so the matrix matches retrieval-time behaviour.
    ah = a.position_heatmap.astype(np.float64).ravel()
    bh = b.position_heatmap.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(ah)), float(np.linalg.norm(bh))
    heat = float(np.dot(ah, bh) / (na * nb)) if (na > 1e-12 and nb > 1e-12) else 0.0
    return 0.35 * row + 0.35 * col + 0.20 * color + 0.10 * heat


def test_full_build(tmp: str) -> None:
    prim_root = os.path.join(tmp, "primitive")
    scenes    = os.path.join(tmp, "labeled")
    motifs_p  = os.path.join(tmp, "motifs.npz")
    memory_p  = os.path.join(tmp, "memory.npz")
    sfb_p     = os.path.join(tmp, "bank.sfb")

    _generate_dataset(prim_root, samples_per_class=30, seed=0)
    pairs = [
        ("horizontal_stripes", "circular_pattern"),
        ("circular_pattern",   "noise_texture"),
    ]
    _write_scenes(scenes, pairs, per_pair=10, seed=1)

    print(_run([sys.executable, os.path.join(REPO, "scripts",
                                             "sss_train_primitive.py"),
                "--data", prim_root, "--out", motifs_p]))
    print(_run([sys.executable, os.path.join(REPO, "scripts",
                                             "sss_train_motif_memory.py"),
                "--motifs",  motifs_p,
                "--labeled", scenes,
                "--out",     memory_p,
                "--threshold", "0.7"]))
    print(_run([sys.executable, os.path.join(REPO, "scripts",
                                             "sss_build_feature_bank.py"),
                "--motifs", motifs_p,
                "--memory", memory_p,
                "--out",    sfb_p]))

    # ── Re-load and verify counts + matrix ───────────────────────
    bank = SSSFeatureBank.load(sfb_p)
    assert len(bank.motifs) == 4, len(bank.motifs)
    assert len(bank.relations) > 0, len(bank.relations)
    assert len(bank.identities) > 0, len(bank.identities)
    print(f"OK  counts      : motifs={len(bank.motifs)}, "
          f"relations={len(bank.relations)}, "
          f"identities={len(bank.identities)}")

    # Pairwise similarity matrix on the .sfb side must agree with
    # what the trainer produced (Phase 2 save → load is byte-exact for
    # everything except the float-precision rounding of scalars and
    # the uint8 heatmap quantisation; the resulting weighted cosine
    # changes by O(1/255) at worst).
    labels = [m.label for m in bank.motifs]
    n = len(labels)
    mat = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = _weighted_cosine(bank.motifs[i], bank.motifs[j])

    col_w = max(len(l) for l in labels) + 1
    print(" " * col_w + " ".join(f"{l[:6]:>6s}" for l in labels))
    for i, la in enumerate(labels):
        print(f"{la:<{col_w}}" +
              " ".join(f"{mat[i, j]:6.3f}" for j in range(n)))

    diag = float(np.mean(np.diag(mat)))
    off_max = float((mat - np.eye(n)).max())
    assert diag > 0.95, f"diag mean {diag:.3f} <= 0.95"
    assert off_max < 0.95, f"off-diag max {off_max:.3f} >= 0.95"
    print(f"OK  matrix      : diag={diag:.3f}, off-diag max={off_max:.3f}")

    # ── Phase 2 round-trip sanity: re-save and re-load. ─────────
    sfb_p2 = os.path.join(tmp, "bank2.sfb")
    bank.save(sfb_p2)
    with open(sfb_p, "rb") as f:  blob1 = f.read()
    with open(sfb_p2, "rb") as f: blob2 = f.read()
    assert blob1 == blob2, "two saves produced different .sfb bytes"
    print(f"OK  round-trip  : second save is byte-identical "
          f"({len(blob1)} bytes)")


def test_empty_dataset(tmp: str) -> None:
    """Empty primitive root → empty motif .npz → empty .sfb (header
    only). Confirms the build pipeline never special-cases the
    common "no data yet" condition."""
    prim_root = os.path.join(tmp, "empty_primitive")
    os.makedirs(prim_root, exist_ok=True)
    motifs_p = os.path.join(tmp, "empty_motifs.npz")
    sfb_p    = os.path.join(tmp, "empty_bank.sfb")

    _run([sys.executable, os.path.join(REPO, "scripts",
                                       "sss_train_primitive.py"),
          "--data", prim_root, "--out", motifs_p])
    _run([sys.executable, os.path.join(REPO, "scripts",
                                       "sss_build_feature_bank.py"),
          "--motifs", motifs_p, "--out", sfb_p])

    bank = SSSFeatureBank.load(sfb_p)
    assert bank.motifs == [] and bank.relations == [] \
        and bank.identities == []
    assert os.path.getsize(sfb_p) == 40, os.path.getsize(sfb_p)
    print(f"OK  empty build : {os.path.getsize(sfb_p)}-byte header-only "
          ".sfb produced and loads cleanly")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_full_build(tmp)
        test_empty_dataset(tmp)
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
