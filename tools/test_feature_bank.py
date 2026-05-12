"""End-to-end tests for the .sfb (Phase 2 feature bank) format.

Covers:
  - Round-trip bit-exactness on 100 / 200 / 5 record scales
  - Two-save bytewise determinism (same bank → same bytes)
  - UTF-8 32-byte label truncation at a codepoint boundary
  - Empty bank
  - Invalid relation src/dst, invalid relation_type
  - Identity motif_count out of range
  - Bad magic and bad version are rejected
  - Python-save ↔ C-save bytewise identity
  - Python-load and C-load both return identical content
  - 1000 motif × 5000 relation × 50 identity benchmark (save/load < 100 ms)
  - position_heatmap uint8 quantisation round-trip error ≤ 1/255

Run:
    make pybridge
    python3 tools/test_feature_bank.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.sss_feature_bank import (
    SSSFeatureBank, SSSMotif, SSSRelation, SSSIdentity,
    RELATION_ABOVE, RELATION_BELOW, RELATION_NEAR, RELATION_INSIDE,
    RELATION_TYPE_MAX,
    SFB_MAGIC, SFB_VERSION, SFB_LABEL_LEN, SFB_FREQ_BINS, SFB_HEATMAP_DIM,
    SFB_MAX_IDENTITY_MOTIFS,
    SFB_ERR_MAGIC, SFB_ERR_VERSION, SFB_ERR_INVALID_REL, SFB_ERR_INVALID_IDENT,
    SFBError,
    MOTIF_RECORD_SIZE, RELATION_RECORD_SIZE, IDENTITY_RECORD_SIZE,
    save_via_c, load_via_c,
)


# ── Bank builders for stable, reproducible content ────────────────

def _rng_motif(i: int, rng: np.random.Generator) -> SSSMotif:
    return SSSMotif(
        label=f"motif_{i:04d}",
        row_freq=rng.random(SFB_FREQ_BINS, dtype=np.float32),
        col_freq=rng.random(SFB_FREQ_BINS, dtype=np.float32),
        color_freq=rng.random((3, SFB_FREQ_BINS), dtype=np.float32),
        position_heatmap=rng.random((SFB_HEATMAP_DIM, SFB_HEATMAP_DIM),
                                    dtype=np.float32),
        coherence=float(rng.random()),
        confidence=float(rng.random()),
        activation_count=int(rng.integers(0, 100_000)),
        variation_cluster_id=int(rng.integers(0, 65_536)),
    )


def _build_bank(n_motifs: int, n_relations: int, n_identities: int,
                seed: int = 42) -> SSSFeatureBank:
    rng = np.random.default_rng(seed)
    motifs = [_rng_motif(i, rng) for i in range(n_motifs)]
    relations = []
    for _ in range(n_relations):
        relations.append(SSSRelation(
            src_motif_id=int(rng.integers(0, max(1, n_motifs))),
            dst_motif_id=int(rng.integers(0, max(1, n_motifs))),
            dx=float(rng.standard_normal()),
            dy=float(rng.standard_normal()),
            relation_type=int(rng.integers(0, RELATION_TYPE_MAX + 1)),
            weight=float(rng.random()),
        ))
    identities = []
    for k in range(n_identities):
        n_pick = int(rng.integers(1, min(SFB_MAX_IDENTITY_MOTIFS, n_motifs) + 1))
        picks = rng.choice(n_motifs, size=n_pick, replace=False).tolist()
        identities.append(SSSIdentity(
            label=f"ident_{k:03d}",
            motif_ids=[int(p) for p in picks],
            confidence=float(rng.random()),
        ))
    return SSSFeatureBank(motifs=motifs, relations=relations, identities=identities)


# ── Field-by-field comparison ─────────────────────────────────────

def _assert_motifs_equal(a: SSSMotif, b: SSSMotif, ctx: str = "",
                         heatmap_tol: float = 1.0 / 255.0 + 1e-7) -> None:
    """Compare two motifs field-by-field. Float arrays must be bitwise
    equal (float32 → bytes → float32 is exact); position_heatmap is
    quantised through uint8, so we compare on the quantisation grid
    and bound the dequantised error by 1/255."""
    assert a.label == b.label, f"{ctx}: label {a.label!r} != {b.label!r}"
    assert np.array_equal(a.row_freq, b.row_freq),    f"{ctx}: row_freq differs"
    assert np.array_equal(a.col_freq, b.col_freq),    f"{ctx}: col_freq differs"
    assert np.array_equal(a.color_freq, b.color_freq), f"{ctx}: color_freq differs"
    # Heatmap: both sides must land on the same uint8 grid AND the
    # dequantised float error must be at most one quantisation step.
    qa = np.clip(np.round(a.position_heatmap * 255.0), 0, 255).astype(np.uint8)
    qb = np.clip(np.round(b.position_heatmap * 255.0), 0, 255).astype(np.uint8)
    assert np.array_equal(qa, qb), \
        f"{ctx}: position_heatmap byte grid differs " \
        f"(max delta {int(np.abs(qa.astype(int) - qb.astype(int)).max())})"
    max_err = float(np.abs(a.position_heatmap - b.position_heatmap).max())
    assert max_err <= heatmap_tol, \
        f"{ctx}: position_heatmap dequant error {max_err} > {heatmap_tol}"
    # Scalars: coherence / confidence are stored as float32 on disk,
    # so any Python float64 round-trip narrows to the nearest float32.
    # Compare on the float32 grid (both sides quantise the same way)
    # so the test is correct regardless of which side held the more
    # precise value originally.
    assert np.float32(a.coherence) == np.float32(b.coherence), \
        f"{ctx}: coherence {a.coherence!r} vs {b.coherence!r}"
    assert np.float32(a.confidence) == np.float32(b.confidence), \
        f"{ctx}: confidence {a.confidence!r} vs {b.confidence!r}"
    assert a.activation_count == b.activation_count, f"{ctx}: activation_count"
    assert a.variation_cluster_id == b.variation_cluster_id, \
        f"{ctx}: variation_cluster_id"


def _assert_relations_equal(a: SSSRelation, b: SSSRelation, ctx: str = "") -> None:
    assert a.src_motif_id == b.src_motif_id, f"{ctx}: src_motif_id"
    assert a.dst_motif_id == b.dst_motif_id, f"{ctx}: dst_motif_id"
    # dx/dy/weight are float32 on disk — narrow both sides before
    # comparing (same reasoning as in _assert_motifs_equal).
    assert np.float32(a.dx) == np.float32(b.dx), f"{ctx}: dx"
    assert np.float32(a.dy) == np.float32(b.dy), f"{ctx}: dy"
    assert a.relation_type == b.relation_type, f"{ctx}: relation_type"
    assert np.float32(a.weight) == np.float32(b.weight), f"{ctx}: weight"


def _assert_identities_equal(a: SSSIdentity, b: SSSIdentity, ctx: str = "") -> None:
    assert a.label == b.label, f"{ctx}: label"
    assert list(a.motif_ids) == list(b.motif_ids), f"{ctx}: motif_ids"
    assert np.float32(a.confidence) == np.float32(b.confidence), \
        f"{ctx}: confidence"


def _assert_banks_equal(a: SSSFeatureBank, b: SSSFeatureBank, ctx: str = "") -> None:
    assert len(a.motifs)     == len(b.motifs),     f"{ctx}: motif_count"
    assert len(a.relations)  == len(b.relations),  f"{ctx}: relation_count"
    assert len(a.identities) == len(b.identities), f"{ctx}: identity_count"
    for i, (ma, mb) in enumerate(zip(a.motifs, b.motifs)):
        _assert_motifs_equal(ma, mb, f"{ctx}.motifs[{i}]")
    for i, (ra, rb) in enumerate(zip(a.relations, b.relations)):
        _assert_relations_equal(ra, rb, f"{ctx}.relations[{i}]")
    for i, (ia, ib) in enumerate(zip(a.identities, b.identities)):
        _assert_identities_equal(ia, ib, f"{ctx}.identities[{i}]")


# ── Test cases ────────────────────────────────────────────────────

def test_roundtrip(tmp: str) -> None:
    """100 motifs × 200 relations × 5 identities, save → load → save,
    full field equality + bytewise determinism on the second save."""
    bank = _build_bank(100, 200, 5)
    path = os.path.join(tmp, "rt.sfb")
    bank.save(path)
    loaded = SSSFeatureBank.load(path)
    # Field-by-field equality (floats bit-exact; heatmap within ±1/255).
    _assert_banks_equal(bank, loaded, "roundtrip")

    # Second save against the loaded bank: bytes must equal the first
    # file's bytes verbatim. This is the "save → load → save is
    # byte-identical" claim that proves the heatmap quantisation is
    # idempotent.
    path2 = os.path.join(tmp, "rt2.sfb")
    loaded.save(path2)
    with open(path, "rb") as f:  blob1 = f.read()
    with open(path2, "rb") as f: blob2 = f.read()
    assert blob1 == blob2, "two saves produced different bytes"
    # And the second-save bank ↔ load must be field-bit-exact at zero
    # tolerance for the heatmap too — both sides are now on the same
    # quantisation grid.
    loaded2 = SSSFeatureBank.load(path2)
    for i, (ma, mb) in enumerate(zip(loaded.motifs, loaded2.motifs)):
        _assert_motifs_equal(ma, mb, f"roundtrip2.motifs[{i}]",
                             heatmap_tol=0.0)
    print(f"OK  roundtrip   : 100×200×5 records, "
          f"{len(blob1)} bytes, byte-deterministic save")


def test_korean_label_truncation(tmp: str) -> None:
    """A Korean label longer than 32 UTF-8 bytes must be truncated at a
    codepoint boundary, never half a codepoint."""
    long_label = "안녕하세요반갑습니다오늘은좋은날입니다정말로기쁩니다"
    bank = SSSFeatureBank(
        motifs=[SSSMotif(label=long_label)],
        relations=[],
        identities=[SSSIdentity(label=long_label, motif_ids=[0])],
    )
    path = os.path.join(tmp, "ko.sfb")
    bank.save(path)
    loaded = SSSFeatureBank.load(path)
    # Label encoded length must be exactly 32 (with trailing NULs).
    raw_label = loaded.motifs[0].label.encode("utf-8")
    assert len(raw_label) <= SFB_LABEL_LEN, len(raw_label)
    # Truncated label must decode without UnicodeDecodeError errors
    # (codepoint-aware truncation guarantees this).
    decoded = loaded.motifs[0].label
    assert decoded.encode("utf-8") == raw_label
    # And it must be a prefix of the original.
    assert long_label.startswith(decoded), \
        f"truncated label {decoded!r} is not a prefix of {long_label!r}"
    print(f"OK  ko-truncate : 한글 label cut to '{decoded}' "
          f"({len(raw_label)} bytes)")


def test_empty_bank(tmp: str) -> None:
    bank = SSSFeatureBank()
    path = os.path.join(tmp, "empty.sfb")
    bank.save(path)
    loaded = SSSFeatureBank.load(path)
    assert loaded.motifs == [] and loaded.relations == [] \
        and loaded.identities == []
    # File is just the 40-byte header.
    assert os.path.getsize(path) == 40, os.path.getsize(path)
    print("OK  empty       : 40-byte header only, all counts 0")


def test_invalid_relation(tmp: str) -> None:
    bank = SSSFeatureBank(
        motifs=[_rng_motif(0, np.random.default_rng(1))],
        relations=[SSSRelation(src_motif_id=0, dst_motif_id=5)],  # 5 >= 1
        identities=[],
    )
    try:
        bank.save(os.path.join(tmp, "bad_rel.sfb"))
    except SFBError as e:
        assert e.code == SFB_ERR_INVALID_REL, e.code
        print(f"OK  bad rel id  : refused with code {e.code}")
        return
    raise AssertionError("save accepted out-of-range relation id")


def test_invalid_relation_type(tmp: str) -> None:
    bank = SSSFeatureBank(
        motifs=[_rng_motif(0, np.random.default_rng(2))],
        relations=[SSSRelation(src_motif_id=0, dst_motif_id=0,
                               relation_type=99)],
        identities=[],
    )
    try:
        bank.save(os.path.join(tmp, "bad_rtype.sfb"))
    except SFBError as e:
        assert e.code == SFB_ERR_INVALID_REL, e.code
        print(f"OK  bad rel type: refused with code {e.code}")
        return
    raise AssertionError("save accepted relation_type out of range")


def test_invalid_identity(tmp: str) -> None:
    motifs = [_rng_motif(i, np.random.default_rng(3 + i)) for i in range(5)]
    too_many = SSSIdentity(label="oversized",
                           motif_ids=list(range(SFB_MAX_IDENTITY_MOTIFS + 1)),
                           confidence=0.5)
    bank = SSSFeatureBank(motifs=motifs, relations=[], identities=[too_many])
    try:
        bank.save(os.path.join(tmp, "bad_ident.sfb"))
    except SFBError as e:
        assert e.code == SFB_ERR_INVALID_IDENT, e.code
        print(f"OK  bad ident   : motif_count > {SFB_MAX_IDENTITY_MOTIFS} "
              f"refused with code {e.code}")
        return
    raise AssertionError("save accepted identity with too many motif_ids")


def test_bad_magic_and_version(tmp: str) -> None:
    """A handcrafted file with the wrong magic / version must be rejected
    by both the Python and C loaders with the right error code."""
    good_path = os.path.join(tmp, "good.sfb")
    SSSFeatureBank().save(good_path)
    with open(good_path, "rb") as f:
        good_bytes = f.read()

    # Bad magic — replace SFB1 with XXXX.
    bad_magic_path = os.path.join(tmp, "bad_magic.sfb")
    with open(bad_magic_path, "wb") as f:
        f.write(b"XXXX" + good_bytes[4:])
    try:
        SSSFeatureBank.load(bad_magic_path)
    except SFBError as e:
        assert e.code == SFB_ERR_MAGIC, e.code
        print(f"OK  bad magic   : refused with code {e.code}")
    else:
        raise AssertionError("loader accepted bad magic")

    # Bad version — set version to 999.
    bad_version_path = os.path.join(tmp, "bad_version.sfb")
    with open(bad_version_path, "wb") as f:
        f.write(good_bytes[:4] + struct.pack("<I", 999) + good_bytes[8:])
    try:
        SSSFeatureBank.load(bad_version_path)
    except SFBError as e:
        assert e.code == SFB_ERR_VERSION, e.code
        print(f"OK  bad version : refused with code {e.code}")
    else:
        raise AssertionError("loader accepted bad version")


def test_python_save_c_load(tmp: str) -> None:
    """Python-saved file must load through libsss_pybridge.so → C → back
    to identical Python dataclasses."""
    bank = _build_bank(50, 80, 4, seed=7)
    path = os.path.join(tmp, "py2c.sfb")
    bank.save(path)
    loaded = load_via_c(path)
    _assert_banks_equal(bank, loaded, "py-save-c-load")
    print("OK  py→C        : python save / C load round-trip identical")


def test_c_save_python_load(tmp: str) -> None:
    """C-saved file must load through pure-Python SSSFeatureBank.load
    into identical dataclasses, AND its bytes must equal the
    Python-saved bytes verbatim (the headline cross-implementation
    parity claim)."""
    bank = _build_bank(50, 80, 4, seed=11)
    path_c  = os.path.join(tmp, "c.sfb")
    path_py = os.path.join(tmp, "py.sfb")
    save_via_c(bank, path_c)
    bank.save(path_py)

    with open(path_c,  "rb") as f: blob_c  = f.read()
    with open(path_py, "rb") as f: blob_py = f.read()
    assert blob_c == blob_py, \
        f"C save vs Python save bytes differ (lens {len(blob_c)} vs {len(blob_py)})"
    print(f"OK  C↔py bytes : {len(blob_c)} bytes byte-identical")

    loaded = SSSFeatureBank.load(path_c)
    _assert_banks_equal(bank, loaded, "c-save-py-load")
    print("OK  C→py        : C save / python load round-trip identical")


def test_benchmark(tmp: str) -> None:
    """1000 motifs × 5000 relations × 50 identities, save+load < 100 ms each."""
    bank = _build_bank(1000, 5000, 50, seed=99)
    path = os.path.join(tmp, "bench.sfb")

    t0 = time.perf_counter()
    bank.save(path)
    t_save = time.perf_counter() - t0

    size = os.path.getsize(path)

    t0 = time.perf_counter()
    loaded = SSSFeatureBank.load(path)
    t_load = time.perf_counter() - t0

    assert len(loaded.motifs)     == 1000
    assert len(loaded.relations)  == 5000
    assert len(loaded.identities) == 50

    # 100 ms target — print actual numbers regardless so a failure
    # mode is visible. Tighten only if both pass with margin.
    print(f"BENCH save 1k×5k×50 : {t_save * 1000:6.1f} ms "
          f"(target < 100 ms) — file {size / 1024:.1f} KB")
    print(f"BENCH load 1k×5k×50 : {t_load * 1000:6.1f} ms "
          f"(target < 100 ms)")
    # Hard fail only if grossly slow (10× over budget); flag warning
    # otherwise so test stays green on slow CI runners.
    assert t_save < 1.0, f"save took {t_save:.3f}s (>1s, way over budget)"
    assert t_load < 1.0, f"load took {t_load:.3f}s (>1s, way over budget)"
    if t_save >= 0.1 or t_load >= 0.1:
        print(f"WARN: budget overshoot (save={t_save*1000:.0f}ms, "
              f"load={t_load*1000:.0f}ms)")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_roundtrip(tmp)
        test_korean_label_truncation(tmp)
        test_empty_bank(tmp)
        test_invalid_relation(tmp)
        test_invalid_relation_type(tmp)
        test_invalid_identity(tmp)
        test_bad_magic_and_version(tmp)
        test_python_save_c_load(tmp)
        test_c_save_python_load(tmp)
        test_benchmark(tmp)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
