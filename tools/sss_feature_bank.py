"""sss_feature_bank — Python I/O for the .sfb (SSS Feature Bank) format.

The .sfb file is a **feature spectral dictionary**, not an image
repository. The unit of storage is a *motif*, never a pixel or a row.
A motif represents "an abstract feature with a particular frequency
tendency" — and absolutely nothing else:

  * No phase is ever written to disk (Phase 1 established this for
    the spectrogram generator; .sfb extends the rule to every higher
    feature).
  * No row indices, no row order, no per-row spectra. The legacy
    .sss v9 (H × NF × 3) per-row layout collapses into a single
    128-bin row_freq envelope per motif.
  * No original pixel coordinates. position_heatmap is a coarse
    16×16 probability surface, not a pixel-addressable map.

Consequence at generate time: the generator synthesises a **new
signal** following the motif's envelope. No row, pixel, or fragment
of a training image is ever directly invoked. The same (motif, seed)
pair produces a fresh waveform each call by design.

Where .sss stored raw per-cell FFT amplitudes inside the model,
.sfb stores higher-level *motif* records (token-scoped amplitude
envelopes + spatial heatmap), directed *relations* between motifs, and
*identity* clusters that bundle motifs into named concepts.

See `ce_core/sss_feature_bank.h` for the full byte-map. This module is
the exact Python mirror — both ends produce byte-identical files for
the same content, verified by `tools/test_feature_bank.py`.

Key design choices that the Python side must respect to stay
bit-compatible with the C writer:

  * pack(1), little-endian (`<` everywhere in struct format strings)
  * position_heatmap is float32 [0, 1] in memory but uint8 on disk —
    save quantises with `*255` + round, load dequantises with `/255`.
    Round-trip error per cell is ≤ 1/255, matching the C side.
  * Labels are byte-clean UTF-8 with a 32-byte hard cap. Strings
    longer than 32 bytes are truncated at a UTF-8 codepoint boundary
    (we never emit a half-codepoint).
  * Empty arrays save / load to zero-count headers without writing
    record bytes.

Typical use::

    from tools.sss_feature_bank import (
        SSSFeatureBank, SSSMotif, SSSRelation, SSSIdentity,
        RELATION_ABOVE,
    )
    bank = SSSFeatureBank(
        motifs=[SSSMotif(label="ear", row_freq=np.zeros(128), ...)],
        relations=[SSSRelation(0, 0, 0.0, 0.0, RELATION_ABOVE, 1.0)],
        identities=[SSSIdentity(label="kitty", motif_ids=[0], confidence=0.8)],
    )
    bank.save("/tmp/mybank.sfb")
    loaded = SSSFeatureBank.load("/tmp/mybank.sfb")
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

# ── Wire-format constants (must match sss_feature_bank.h) ──────────
SFB_MAGIC                 = b"SFB1"
SFB_VERSION               = 1
SFB_LABEL_LEN             = 32
SFB_FREQ_BINS             = 128
SFB_HEATMAP_DIM           = 16
SFB_MAX_IDENTITY_MOTIFS   = 32

# Relation type enum (matches SFB_RELATION_TYPE_* in the header).
RELATION_ABOVE   = 0
RELATION_BELOW   = 1
RELATION_LEFT    = 2
RELATION_RIGHT   = 3
RELATION_NEAR    = 4
RELATION_AROUND  = 5
RELATION_INSIDE  = 6
RELATION_TYPE_MAX = RELATION_INSIDE

# Header struct format: 40 bytes total. See sss_feature_bank.h for the
# byte map. Keep `<` (little-endian, no native alignment) to match
# pack(1) on the C side.
_HEADER_FMT     = "<4sI III HHHHQI"
_HEADER_SIZE    = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 40, _HEADER_SIZE

# Record sizes — must match the C structs exactly.
MOTIF_RECORD_SIZE     = 2864
RELATION_RECORD_SIZE  = 20
IDENTITY_RECORD_SIZE  = 104

# Error code mirror of SFB_ERR_*.
SFB_OK              = 0
SFB_ERR_IO          = -1
SFB_ERR_MAGIC       = -2
SFB_ERR_VERSION     = -3
SFB_ERR_INVALID_REL = -4
SFB_ERR_INVALID_IDENT = -5
SFB_ERR_ALLOC       = -6


class SFBError(IOError):
    """All on-disk failures (bad magic, version mismatch, validation,
    truncated read) raise this with `code` set to one of the SFB_ERR_*
    constants for parity with the C return-code surface."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


# ── Label encoding helpers ─────────────────────────────────────────

def _encode_label(label: str) -> bytes:
    """Encode a label to exactly SFB_LABEL_LEN bytes (UTF-8, zero-padded
    on the right, truncated at the last whole codepoint that still fits).

    Truncation is codepoint-aware: a 3-byte Hangul codepoint that would
    cross the 32-byte boundary is dropped entirely rather than emitted
    as two stray bytes. This matches the GPT-validated trimming rule."""
    b = label.encode("utf-8")
    if len(b) <= SFB_LABEL_LEN:
        return b + b"\x00" * (SFB_LABEL_LEN - len(b))
    # Truncate at the largest UTF-8 prefix that fits; decode-with-ignore
    # finds the natural codepoint boundary.
    truncated = b[:SFB_LABEL_LEN]
    # Walk backwards from byte 32 until the prefix decodes cleanly.
    for cut in range(SFB_LABEL_LEN, 0, -1):
        prefix = b[:cut]
        try:
            prefix.decode("utf-8")
            return prefix + b"\x00" * (SFB_LABEL_LEN - cut)
        except UnicodeDecodeError:
            continue
    # Pathological: even the first byte isn't a valid leader; pad zeros.
    return b"\x00" * SFB_LABEL_LEN


def _decode_label(raw: bytes) -> str:
    """Decode a 32-byte label field. Reads up to the first NUL or the
    end of the field, then decodes UTF-8 with `replace` so a truncated
    multibyte sequence (theoretically impossible from our own writer
    after codepoint-aware truncation, but defensive) doesn't raise."""
    end = raw.find(b"\x00")
    if end < 0:
        end = len(raw)
    return raw[:end].decode("utf-8", errors="replace")


# ── Dataclasses ────────────────────────────────────────────────────

def _zero_freq() -> np.ndarray:
    return np.zeros(SFB_FREQ_BINS, dtype=np.float32)


def _zero_color_freq() -> np.ndarray:
    return np.zeros((3, SFB_FREQ_BINS), dtype=np.float32)


def _zero_heatmap() -> np.ndarray:
    return np.zeros((SFB_HEATMAP_DIM, SFB_HEATMAP_DIM), dtype=np.float32)


@dataclass
class SSSMotif:
    """One motif record. Numeric arrays are float32 in memory; the
    heatmap is float32 in [0, 1] and quantises to uint8 on disk."""
    label: str = ""
    row_freq: np.ndarray   = field(default_factory=_zero_freq)
    col_freq: np.ndarray   = field(default_factory=_zero_freq)
    color_freq: np.ndarray = field(default_factory=_zero_color_freq)
    position_heatmap: np.ndarray = field(default_factory=_zero_heatmap)
    coherence: float = 0.0
    confidence: float = 0.0
    activation_count: int = 0
    variation_cluster_id: int = 0

    RECORD_SIZE: ClassVar[int] = MOTIF_RECORD_SIZE


@dataclass
class SSSRelation:
    src_motif_id: int = 0
    dst_motif_id: int = 0
    dx: float = 0.0
    dy: float = 0.0
    relation_type: int = RELATION_ABOVE
    weight: float = 1.0

    RECORD_SIZE: ClassVar[int] = RELATION_RECORD_SIZE


@dataclass
class SSSIdentity:
    label: str = ""
    motif_ids: list = field(default_factory=list)
    confidence: float = 0.0

    RECORD_SIZE: ClassVar[int] = IDENTITY_RECORD_SIZE


# ── Per-record packers / unpackers ─────────────────────────────────

def _pack_motif(m: SSSMotif) -> bytes:
    label = _encode_label(m.label)
    row = np.ascontiguousarray(m.row_freq,   dtype=np.float32)
    col = np.ascontiguousarray(m.col_freq,   dtype=np.float32)
    if row.size != SFB_FREQ_BINS:
        raise ValueError(f"row_freq must be {SFB_FREQ_BINS} bins, got {row.size}")
    if col.size != SFB_FREQ_BINS:
        raise ValueError(f"col_freq must be {SFB_FREQ_BINS} bins, got {col.size}")
    cf  = np.ascontiguousarray(m.color_freq, dtype=np.float32)
    if cf.shape != (3, SFB_FREQ_BINS):
        raise ValueError(
            f"color_freq must be (3, {SFB_FREQ_BINS}), got {cf.shape}")
    heat = np.asarray(m.position_heatmap, dtype=np.float32)
    if heat.shape != (SFB_HEATMAP_DIM, SFB_HEATMAP_DIM):
        raise ValueError(
            f"position_heatmap must be ({SFB_HEATMAP_DIM}, {SFB_HEATMAP_DIM}), "
            f"got {heat.shape}")
    # Quantise float[0,1] → uint8[0,255] with rounding. clip first so
    # noisy callers can't write out-of-range bytes.
    heat_u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)

    parts = [
        label,
        row.tobytes(order="C"),
        col.tobytes(order="C"),
        cf.tobytes(order="C"),
        heat_u8.tobytes(order="C"),
        struct.pack("<ffIHH",
                    float(m.coherence),
                    float(m.confidence),
                    int(m.activation_count) & 0xFFFFFFFF,
                    int(m.variation_cluster_id) & 0xFFFF,
                    0),                    # _pad
    ]
    out = b"".join(parts)
    if len(out) != MOTIF_RECORD_SIZE:
        raise AssertionError(
            f"motif record size mismatch: {len(out)} != {MOTIF_RECORD_SIZE}")
    return out


def _unpack_motif(buf: bytes) -> SSSMotif:
    if len(buf) != MOTIF_RECORD_SIZE:
        raise ValueError(f"motif record must be {MOTIF_RECORD_SIZE} bytes, "
                         f"got {len(buf)}")
    off = 0
    label = _decode_label(buf[off:off + SFB_LABEL_LEN]); off += SFB_LABEL_LEN
    row = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    col = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    cf = np.frombuffer(buf[off:off + 4 * 3 * SFB_FREQ_BINS],
                       dtype="<f4").astype(np.float32).reshape(3, SFB_FREQ_BINS)
    off += 4 * 3 * SFB_FREQ_BINS
    heat_u8 = np.frombuffer(
        buf[off:off + SFB_HEATMAP_DIM * SFB_HEATMAP_DIM],
        dtype=np.uint8).reshape(SFB_HEATMAP_DIM, SFB_HEATMAP_DIM)
    off += SFB_HEATMAP_DIM * SFB_HEATMAP_DIM
    coh, conf, act, vcid, _pad = struct.unpack_from("<ffIHH", buf, off)
    off += struct.calcsize("<ffIHH")

    return SSSMotif(
        label=label,
        row_freq=row.copy(),
        col_freq=col.copy(),
        color_freq=cf.copy(),
        position_heatmap=(heat_u8.astype(np.float32) / 255.0),
        coherence=float(coh),
        confidence=float(conf),
        activation_count=int(act),
        variation_cluster_id=int(vcid),
    )


def _pack_relation(r: SSSRelation) -> bytes:
    out = struct.pack(
        "<HHffB3sf",
        int(r.src_motif_id) & 0xFFFF,
        int(r.dst_motif_id) & 0xFFFF,
        float(r.dx),
        float(r.dy),
        int(r.relation_type) & 0xFF,
        b"\x00\x00\x00",            # _pad[3]
        float(r.weight),
    )
    assert len(out) == RELATION_RECORD_SIZE, (len(out), RELATION_RECORD_SIZE)
    return out


def _unpack_relation(buf: bytes) -> SSSRelation:
    if len(buf) != RELATION_RECORD_SIZE:
        raise ValueError(
            f"relation record must be {RELATION_RECORD_SIZE} bytes, "
            f"got {len(buf)}")
    src, dst, dx, dy, rtype, _pad, w = struct.unpack("<HHffB3sf", buf)
    return SSSRelation(
        src_motif_id=int(src),
        dst_motif_id=int(dst),
        dx=float(dx),
        dy=float(dy),
        relation_type=int(rtype),
        weight=float(w),
    )


def _pack_identity(idn: SSSIdentity) -> bytes:
    label = _encode_label(idn.label)
    motif_ids = list(idn.motif_ids or [])
    if not (1 <= len(motif_ids) <= SFB_MAX_IDENTITY_MOTIFS):
        raise ValueError(
            f"identity {idn.label!r}: motif_ids must have 1..32 entries, "
            f"got {len(motif_ids)}")
    padded_ids = motif_ids + [0] * (SFB_MAX_IDENTITY_MOTIFS - len(motif_ids))
    ids_bytes = struct.pack("<" + "H" * SFB_MAX_IDENTITY_MOTIFS,
                            *[mi & 0xFFFF for mi in padded_ids])
    out = label + ids_bytes + struct.pack(
        "<B3sf",
        len(motif_ids) & 0xFF,
        b"\x00\x00\x00",            # _pad[3]
        float(idn.confidence),
    )
    assert len(out) == IDENTITY_RECORD_SIZE, (len(out), IDENTITY_RECORD_SIZE)
    return out


def _unpack_identity(buf: bytes) -> SSSIdentity:
    if len(buf) != IDENTITY_RECORD_SIZE:
        raise ValueError(
            f"identity record must be {IDENTITY_RECORD_SIZE} bytes, "
            f"got {len(buf)}")
    off = 0
    label = _decode_label(buf[off:off + SFB_LABEL_LEN]); off += SFB_LABEL_LEN
    ids_full = struct.unpack_from("<" + "H" * SFB_MAX_IDENTITY_MOTIFS, buf, off)
    off += 2 * SFB_MAX_IDENTITY_MOTIFS
    mc, _pad, conf = struct.unpack_from("<B3sf", buf, off)
    if not (1 <= mc <= SFB_MAX_IDENTITY_MOTIFS):
        raise SFBError(SFB_ERR_INVALID_IDENT,
                       f"identity motif_count {mc} out of range")
    return SSSIdentity(
        label=label,
        motif_ids=list(ids_full[:mc]),
        confidence=float(conf),
    )


# ── Validation ─────────────────────────────────────────────────────

def _validate(motifs, relations, identities) -> None:
    n = len(motifs)
    for i, r in enumerate(relations):
        if not (0 <= r.src_motif_id < n):
            raise SFBError(SFB_ERR_INVALID_REL,
                f"relation[{i}].src_motif_id={r.src_motif_id} >= "
                f"motif_count={n}")
        if not (0 <= r.dst_motif_id < n):
            raise SFBError(SFB_ERR_INVALID_REL,
                f"relation[{i}].dst_motif_id={r.dst_motif_id} >= "
                f"motif_count={n}")
        if not (0 <= r.relation_type <= RELATION_TYPE_MAX):
            raise SFBError(SFB_ERR_INVALID_REL,
                f"relation[{i}].relation_type={r.relation_type} > "
                f"{RELATION_TYPE_MAX}")
    for i, idn in enumerate(identities):
        mc = len(idn.motif_ids or [])
        if not (1 <= mc <= SFB_MAX_IDENTITY_MOTIFS):
            raise SFBError(SFB_ERR_INVALID_IDENT,
                f"identity[{i}] {idn.label!r}: motif_ids has {mc} entries, "
                f"must be 1..{SFB_MAX_IDENTITY_MOTIFS}")
        for j, mid in enumerate(idn.motif_ids):
            if not (0 <= mid < n):
                raise SFBError(SFB_ERR_INVALID_IDENT,
                    f"identity[{i}].motif_ids[{j}]={mid} >= "
                    f"motif_count={n}")


# ── Bank ──────────────────────────────────────────────────────────

@dataclass
class SSSFeatureBank:
    motifs: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    identities: list = field(default_factory=list)

    def save(self, path: str) -> None:
        _validate(self.motifs, self.relations, self.identities)
        header = struct.pack(
            _HEADER_FMT,
            SFB_MAGIC,
            SFB_VERSION,
            len(self.motifs),
            len(self.relations),
            len(self.identities),
            MOTIF_RECORD_SIZE,
            RELATION_RECORD_SIZE,
            IDENTITY_RECORD_SIZE,
            0,                             # flags
            0,                             # reserved u64
            0,                             # reserved2 u32
        )
        chunks = [header]
        for m in self.motifs:
            chunks.append(_pack_motif(m))
        for r in self.relations:
            chunks.append(_pack_relation(r))
        for idn in self.identities:
            chunks.append(_pack_identity(idn))
        # Single write keeps save bit-identical to the C side, which
        # fwrite()s the whole header then each array contiguously.
        with open(path, "wb") as f:
            f.write(b"".join(chunks))

    @classmethod
    def load(cls, path: str) -> "SSSFeatureBank":
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < _HEADER_SIZE:
            raise SFBError(SFB_ERR_IO, ".sfb: truncated header")
        (magic, version,
         n_motifs, n_relations, n_identities,
         motif_rs, relation_rs, identity_rs,
         _flags, _reserved, _reserved2) = struct.unpack_from(_HEADER_FMT, blob, 0)

        if magic != SFB_MAGIC:
            raise SFBError(SFB_ERR_MAGIC, f".sfb: bad magic {magic!r}")
        if version != SFB_VERSION:
            raise SFBError(SFB_ERR_VERSION, f".sfb: unsupported version {version}")
        if (motif_rs < MOTIF_RECORD_SIZE
                or relation_rs < RELATION_RECORD_SIZE
                or identity_rs < IDENTITY_RECORD_SIZE):
            raise SFBError(SFB_ERR_VERSION,
                ".sfb: record_size smaller than current minimum "
                f"(motif {motif_rs}/{MOTIF_RECORD_SIZE}, "
                f"relation {relation_rs}/{RELATION_RECORD_SIZE}, "
                f"identity {identity_rs}/{IDENTITY_RECORD_SIZE})")

        off = _HEADER_SIZE
        need = off + motif_rs * n_motifs + relation_rs * n_relations \
                   + identity_rs * n_identities
        if need > len(blob):
            raise SFBError(SFB_ERR_IO,
                f".sfb: truncated body (need {need}, have {len(blob)})")

        motifs = []
        for _ in range(n_motifs):
            # When record_size > current struct size (v2+), trim the
            # excess. v1 readers must never crash on a v2 file.
            motifs.append(_unpack_motif(blob[off:off + MOTIF_RECORD_SIZE]))
            off += motif_rs

        relations = []
        for _ in range(n_relations):
            relations.append(_unpack_relation(blob[off:off + RELATION_RECORD_SIZE]))
            off += relation_rs

        identities = []
        for _ in range(n_identities):
            identities.append(_unpack_identity(blob[off:off + IDENTITY_RECORD_SIZE]))
            off += identity_rs

        bank = cls(motifs=motifs, relations=relations, identities=identities)
        _validate(bank.motifs, bank.relations, bank.identities)
        return bank

    def find_motif(self, label: str) -> int:
        """Return the first motif index matching `label`, or -1."""
        for i, m in enumerate(self.motifs):
            if m.label == label:
                return i
        return -1


# ── C bridge round-trip helpers ────────────────────────────────────
#
# These functions go through libsss_pybridge.so, exercising the C
# sss_feature_bank_save / load implementations end-to-end. They're the
# how-do-we-know-they-agree path for tools/test_feature_bank.py and
# any downstream training tool that wants the C writer's speed.

def _c_lib():
    """Lazy-import the pybridge lib. Imported here (not at module top)
    so this module can be used without ctypes available — pure-Python
    save/load works without libsss_pybridge.so."""
    import ctypes
    from tools.sss_memory import _lib       # type: ignore[attr-defined]
    return ctypes, _lib()


def save_via_c(bank: SSSFeatureBank, path: str) -> None:
    """Round-trip a bank through libsss_pybridge.so's
    sss_pybridge_feature_bank_save. Bytes-identical with `SSSFeatureBank.save`
    for the same `bank`, by construction (the C side fwrite()s the
    record bytes we hand it, and the bytes we pack match the C struct
    byte map)."""
    ctypes, lib = _c_lib()
    if not hasattr(lib, "sss_pybridge_feature_bank_save"):
        raise RuntimeError(
            "libsss_pybridge.so is missing sss_pybridge_feature_bank_save; "
            "rebuild with `make pybridge`.")
    _validate(bank.motifs, bank.relations, bank.identities)

    motif_blob = b"".join(_pack_motif(m) for m in bank.motifs)
    rel_blob   = b"".join(_pack_relation(r) for r in bank.relations)
    ident_blob = b"".join(_pack_identity(i) for i in bank.identities)

    def _buf(blob: bytes):
        if not blob:
            return ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
        return (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)

    rc = lib.sss_pybridge_feature_bank_save(
        path.encode("utf-8"),
        _buf(motif_blob), ctypes.c_uint32(len(bank.motifs)),
        _buf(rel_blob),   ctypes.c_uint32(len(bank.relations)),
        _buf(ident_blob), ctypes.c_uint32(len(bank.identities)),
    )
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_save returned {rc}")


def load_via_c(path: str) -> SSSFeatureBank:
    """Round-trip the on-disk file through libsss_pybridge.so's
    sss_pybridge_feature_bank_load."""
    ctypes, lib = _c_lib()
    for name in ("sss_pybridge_feature_bank_probe",
                 "sss_pybridge_feature_bank_load"):
        if not hasattr(lib, name):
            raise RuntimeError(
                f"libsss_pybridge.so is missing {name}; rebuild with "
                "`make pybridge`.")

    nm = ctypes.c_uint32(0); nr = ctypes.c_uint32(0); ni = ctypes.c_uint32(0)
    rc = lib.sss_pybridge_feature_bank_probe(
        path.encode("utf-8"),
        ctypes.byref(nm), ctypes.byref(nr), ctypes.byref(ni))
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_probe returned {rc}")

    motif_buf = (ctypes.c_uint8 * (MOTIF_RECORD_SIZE * nm.value))()
    rel_buf   = (ctypes.c_uint8 * (RELATION_RECORD_SIZE * nr.value))()
    ident_buf = (ctypes.c_uint8 * (IDENTITY_RECORD_SIZE * ni.value))()
    nm_cap = ctypes.c_uint32(nm.value)
    nr_cap = ctypes.c_uint32(nr.value)
    ni_cap = ctypes.c_uint32(ni.value)
    rc = lib.sss_pybridge_feature_bank_load(
        path.encode("utf-8"),
        motif_buf, ctypes.byref(nm_cap),
        rel_buf,   ctypes.byref(nr_cap),
        ident_buf, ctypes.byref(ni_cap),
    )
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_load returned {rc}")

    motifs     = [_unpack_motif(bytes(motif_buf[i * MOTIF_RECORD_SIZE
                                                :(i + 1) * MOTIF_RECORD_SIZE]))
                  for i in range(nm_cap.value)]
    relations  = [_unpack_relation(bytes(rel_buf[i * RELATION_RECORD_SIZE
                                                 :(i + 1) * RELATION_RECORD_SIZE]))
                  for i in range(nr_cap.value)]
    identities = [_unpack_identity(bytes(ident_buf[i * IDENTITY_RECORD_SIZE
                                                   :(i + 1) * IDENTITY_RECORD_SIZE]))
                  for i in range(ni_cap.value)]
    return SSSFeatureBank(
        motifs=motifs, relations=relations, identities=identities)
