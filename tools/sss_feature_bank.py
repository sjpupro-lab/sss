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
of a training image is ever directly invoked. Generation is
deterministic per seed — a fixed seed reproduces the same output —
but varying the seed varies the waveform (the per-seed random phase
init from Phase 1's spectrogram generator continues to hold at the
motif level).

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
SFB_VERSION               = 2
SFB_LABEL_LEN             = 32
SFB_FREQ_BINS             = 128
SFB_HEATMAP_DIM           = 16
SFB_MAX_IDENTITY_MOTIFS   = 32

# Phase 4 v2 dimensions.
SFB_TEMPORAL_BINS         = 64
SFB_WARP_DIM              = 16
SFB_MAX_RESPONSES_PER_MOTIF = 16

# Relation type enum (matches SFB_RELATION_TYPE_* in the header).
RELATION_ABOVE   = 0
RELATION_BELOW   = 1
RELATION_LEFT    = 2
RELATION_RIGHT   = 3
RELATION_NEAR    = 4
RELATION_AROUND  = 5
RELATION_INSIDE  = 6
RELATION_TYPE_MAX = RELATION_INSIDE

# Phase 4 condition type enum (matches SFB_CONDITION_*).
CONDITION_CONTINUOUS  = 0
CONDITION_IMPULSE     = 1
CONDITION_OSCILLATORY = 2
CONDITION_TYPE_MAX    = CONDITION_OSCILLATORY

# Header struct format: 40 bytes total. v2 reinterprets the previous
# `flags + reserved + reserved2` byte range as
# `condition_count(u16) + response_count(u32) + reserved1(u32) +
# reserved2(u32)`. Byte offsets are identical so v1 readers see
# their old (random) values where the v2 writer placed meaningful
# counts — that's fine because v1 readers ignored those fields.
_HEADER_FMT     = "<4sI III HHHH III"
_HEADER_SIZE    = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 40, _HEADER_SIZE

# Record sizes — must match the C structs exactly. The v1 motif
# size is kept as a named constant so the v2 loader can still parse
# v1 files on the way in.
V1_MOTIF_RECORD_SIZE   = 2864
MOTIF_RECORD_SIZE      = 3744          # v2
RELATION_RECORD_SIZE   = 20
IDENTITY_RECORD_SIZE   = 104
CONDITION_RECORD_SIZE  = 1332
RESPONSE_RECORD_SIZE   = 1808

# Error code mirror of SFB_ERR_*.
SFB_OK              = 0
SFB_ERR_IO          = -1
SFB_ERR_MAGIC       = -2
SFB_ERR_VERSION     = -3
SFB_ERR_INVALID_REL = -4
SFB_ERR_INVALID_IDENT = -5
SFB_ERR_ALLOC       = -6
SFB_ERR_INVALID_COND = -7
SFB_ERR_INVALID_RESP = -8


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


def _zero_temporal() -> np.ndarray:
    return np.zeros(SFB_TEMPORAL_BINS, dtype=np.float32)


def _zero_warp() -> np.ndarray:
    """Default warp field — float32 in [-1, +1]. Zero means "no
    displacement". The on-disk uint8 is `round(127*v) + 128` so 0
    in memory decodes to 128 on disk."""
    return np.zeros((SFB_WARP_DIM, SFB_WARP_DIM), dtype=np.float32)


@dataclass
class SSSMotif:
    """One motif record (v2). Numeric arrays are float32 in memory;
    the heatmap and warp_self_* fields are float32 in [-1, +1] (warp)
    or [0, 1] (heatmap) and quantise to uint8 on disk.

    Phase 4 v2 fields:
        temporal_length     — 0 for static, N for N-frame motifs
        response_count_in_motif — informational count of
            InteractionResponse rows keyed by this motif (set by
            the build step, ignored at save time when 0)
        response_offset     — index of the motif's first response
            in `bank.responses` (set by the build step)
        cross_resonance     — 64-bin envelope; the synthesiser
            interprets bin k as a modulation side-band at
            (row_freq[k] + col_freq[k]) × (row_freq[k] − col_freq[k]).
        warp_self_x / warp_self_y — 16×16 baseline warp fields
            applied even with no condition. uint8 [0, 255] on disk
            with the centre-128 / scale-127 convention used by
            interaction response warps.

    `temporal_envelope` is intentionally absent — per-condition
    `temporal_freq[64]` carries the time-axis spectrum. See the
    Phase 4 design philosophy in the C header."""
    label: str = ""
    row_freq: np.ndarray   = field(default_factory=_zero_freq)
    col_freq: np.ndarray   = field(default_factory=_zero_freq)
    color_freq: np.ndarray = field(default_factory=_zero_color_freq)
    position_heatmap: np.ndarray = field(default_factory=_zero_heatmap)
    coherence: float = 0.0
    confidence: float = 0.0
    activation_count: int = 0
    variation_cluster_id: int = 0
    # v2 extension
    temporal_length: int = 0
    response_count_in_motif: int = 0
    response_offset: int = 0
    cross_resonance: np.ndarray = field(default_factory=_zero_temporal)
    warp_self_x: np.ndarray     = field(default_factory=_zero_warp)
    warp_self_y: np.ndarray     = field(default_factory=_zero_warp)

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


@dataclass
class SSSCondition:
    """Phase 4 condition signal — 1332 B on disk. A condition is a
    *signal*, never a frame: row / col / temporal amplitude envelopes
    + direction (radians) + nominal intensity + decay rate."""
    label: str = ""
    condition_type: int = CONDITION_CONTINUOUS
    row_freq: np.ndarray = field(default_factory=_zero_freq)
    col_freq: np.ndarray = field(default_factory=_zero_freq)
    temporal_freq: np.ndarray = field(default_factory=_zero_temporal)
    direction: float = 0.0
    default_intensity: float = 1.0
    decay_rate: float = 0.0

    RECORD_SIZE: ClassVar[int] = CONDITION_RECORD_SIZE


@dataclass
class SSSInteractionResponse:
    """Phase 4 interaction response — 1808 B. Per (motif, condition)
    we accumulate envelope-side response (row_response / col_response
    / cross_response) and warp-side response (warp_x / warp_y, each
    float32 [-1, +1] in memory, uint8 on disk)."""
    motif_id: int = 0
    condition_id: int = 0
    row_response: np.ndarray = field(default_factory=_zero_freq)
    col_response: np.ndarray = field(default_factory=_zero_freq)
    cross_response: np.ndarray = field(default_factory=_zero_temporal)
    warp_x_response: np.ndarray = field(default_factory=_zero_warp)
    warp_y_response: np.ndarray = field(default_factory=_zero_warp)
    response_strength: float = 0.0
    response_delay: float = 0.0
    accumulated_samples: int = 0

    RECORD_SIZE: ClassVar[int] = RESPONSE_RECORD_SIZE


# ── Per-record packers / unpackers ─────────────────────────────────

def _encode_warp(field_f32: np.ndarray) -> np.ndarray:
    """float32 warp field in [-1, +1] → uint8 [0, 255] using the
    centre-128 / scale-127 convention shared with the C side:
        u = round(127 * clip(v, -1, 1)) + 128
    decoded later via _decode_warp().

    Zero displacement → 128 on disk (matching `warp_self_*` defaults
    in motif initialisation)."""
    v = np.clip(np.asarray(field_f32, dtype=np.float32), -1.0, 1.0)
    return (np.round(v * 127.0) + 128).astype(np.uint8)


def _decode_warp(u8: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] → float32 [-1, +1] (inverse of _encode_warp).
    A stored 128 decodes to 0 exactly; saturating values land at
    ±127/127 ≈ ±1.0."""
    return ((u8.astype(np.float32) - 128.0) / 127.0).astype(np.float32)


def _pack_motif(m: SSSMotif) -> bytes:
    """Pack a v2 SSSMotif to 3744 bytes. Layout (matching the C
    pack(1) struct in sss_feature_bank.h):

        [v1 motif: 2864 bytes]
        uint16  temporal_length
        uint16  response_count_in_motif
        uint32  response_offset
        float32 cross_resonance[64]              (256 B)
        uint8   warp_self_x[16][16]              (256 B)
        uint8   warp_self_y[16][16]              (256 B)
        uint8   _pad[104]                        (zero-fill)
    """
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
    heat_u8 = np.clip(np.round(heat * 255.0), 0, 255).astype(np.uint8)

    # v2 fields
    cross = np.ascontiguousarray(m.cross_resonance, dtype=np.float32)
    if cross.size != SFB_TEMPORAL_BINS:
        raise ValueError(
            f"cross_resonance must be {SFB_TEMPORAL_BINS} bins, got {cross.size}")
    wsx = np.asarray(m.warp_self_x, dtype=np.float32)
    wsy = np.asarray(m.warp_self_y, dtype=np.float32)
    if wsx.shape != (SFB_WARP_DIM, SFB_WARP_DIM) \
       or wsy.shape != (SFB_WARP_DIM, SFB_WARP_DIM):
        raise ValueError(
            f"warp_self_x/_y must be ({SFB_WARP_DIM}, {SFB_WARP_DIM})")

    parts = [
        # v1 head (2864 B)
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
        # v2 extension (880 B)
        struct.pack("<HHI",
                    int(m.temporal_length)         & 0xFFFF,
                    int(m.response_count_in_motif) & 0xFFFF,
                    int(m.response_offset)         & 0xFFFFFFFF),
        cross.tobytes(order="C"),
        _encode_warp(wsx).tobytes(order="C"),
        _encode_warp(wsy).tobytes(order="C"),
        b"\x00" * 104,                     # _pad_v2
    ]
    out = b"".join(parts)
    if len(out) != MOTIF_RECORD_SIZE:
        raise AssertionError(
            f"motif record size mismatch: {len(out)} != {MOTIF_RECORD_SIZE}")
    return out


def _unpack_motif(buf: bytes) -> SSSMotif:
    """Decode a 3744-byte v2 motif record. Also accepts a 2864-byte
    v1 record (zero-padded extension fields) so the v2 loader can
    consume legacy files transparently."""
    if len(buf) == V1_MOTIF_RECORD_SIZE:
        buf = buf + b"\x00" * (MOTIF_RECORD_SIZE - V1_MOTIF_RECORD_SIZE)
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

    # v2 extension
    tlen, rcnt, roff = struct.unpack_from("<HHI", buf, off)
    off += struct.calcsize("<HHI")
    cross = np.frombuffer(buf[off:off + 4 * SFB_TEMPORAL_BINS],
                          dtype="<f4").astype(np.float32)
    off += 4 * SFB_TEMPORAL_BINS
    wsx_u8 = np.frombuffer(buf[off:off + SFB_WARP_DIM * SFB_WARP_DIM],
                           dtype=np.uint8).reshape(SFB_WARP_DIM, SFB_WARP_DIM)
    off += SFB_WARP_DIM * SFB_WARP_DIM
    wsy_u8 = np.frombuffer(buf[off:off + SFB_WARP_DIM * SFB_WARP_DIM],
                           dtype=np.uint8).reshape(SFB_WARP_DIM, SFB_WARP_DIM)
    off += SFB_WARP_DIM * SFB_WARP_DIM
    # _pad_v2 is ignored.

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
        temporal_length=int(tlen),
        response_count_in_motif=int(rcnt),
        response_offset=int(roff),
        cross_resonance=cross.copy(),
        warp_self_x=_decode_warp(wsx_u8),
        warp_self_y=_decode_warp(wsy_u8),
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


def _pack_condition(c: SSSCondition) -> bytes:
    """Pack a Phase 4 condition record (1332 B)."""
    label = _encode_label(c.label)
    row = np.ascontiguousarray(c.row_freq,      dtype=np.float32)
    col = np.ascontiguousarray(c.col_freq,      dtype=np.float32)
    tmp = np.ascontiguousarray(c.temporal_freq, dtype=np.float32)
    if row.size != SFB_FREQ_BINS or col.size != SFB_FREQ_BINS:
        raise ValueError("condition row/col_freq must be 128 bins")
    if tmp.size != SFB_TEMPORAL_BINS:
        raise ValueError(
            f"condition temporal_freq must be {SFB_TEMPORAL_BINS} bins")
    if not (0 <= c.condition_type <= CONDITION_TYPE_MAX):
        raise SFBError(SFB_ERR_INVALID_COND,
                       f"condition_type {c.condition_type} out of range")
    parts = [
        label,
        struct.pack("<B3s",
                    int(c.condition_type) & 0xFF,
                    b"\x00\x00\x00"),
        row.tobytes(order="C"),
        col.tobytes(order="C"),
        tmp.tobytes(order="C"),
        struct.pack("<fffI",
                    float(c.direction),
                    float(c.default_intensity),
                    float(c.decay_rate),
                    0),                              # _pad2
    ]
    out = b"".join(parts)
    if len(out) != CONDITION_RECORD_SIZE:
        raise AssertionError(
            f"condition record size mismatch: {len(out)} != {CONDITION_RECORD_SIZE}")
    return out


def _unpack_condition(buf: bytes) -> SSSCondition:
    if len(buf) != CONDITION_RECORD_SIZE:
        raise ValueError(
            f"condition record must be {CONDITION_RECORD_SIZE} bytes, "
            f"got {len(buf)}")
    off = 0
    label = _decode_label(buf[off:off + SFB_LABEL_LEN]); off += SFB_LABEL_LEN
    ctype, _pad = struct.unpack_from("<B3s", buf, off)
    off += struct.calcsize("<B3s")
    row = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    col = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    tmp = np.frombuffer(buf[off:off + 4 * SFB_TEMPORAL_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_TEMPORAL_BINS
    dir_, dintens, decay, _pad2 = struct.unpack_from("<fffI", buf, off)
    return SSSCondition(
        label=label,
        condition_type=int(ctype),
        row_freq=row.copy(),
        col_freq=col.copy(),
        temporal_freq=tmp.copy(),
        direction=float(dir_),
        default_intensity=float(dintens),
        decay_rate=float(decay),
    )


def _pack_response(r: SSSInteractionResponse) -> bytes:
    """Pack a Phase 4 interaction response record (1808 B)."""
    row = np.ascontiguousarray(r.row_response,   dtype=np.float32)
    col = np.ascontiguousarray(r.col_response,   dtype=np.float32)
    cross = np.ascontiguousarray(r.cross_response, dtype=np.float32)
    if row.size != SFB_FREQ_BINS or col.size != SFB_FREQ_BINS:
        raise ValueError("response row/col_response must be 128 bins")
    if cross.size != SFB_TEMPORAL_BINS:
        raise ValueError(
            f"response cross_response must be {SFB_TEMPORAL_BINS} bins")
    wx = np.asarray(r.warp_x_response, dtype=np.float32)
    wy = np.asarray(r.warp_y_response, dtype=np.float32)
    if wx.shape != (SFB_WARP_DIM, SFB_WARP_DIM) \
       or wy.shape != (SFB_WARP_DIM, SFB_WARP_DIM):
        raise ValueError("response warp_x/_y must be 16×16")
    parts = [
        struct.pack("<HH",
                    int(r.motif_id)     & 0xFFFF,
                    int(r.condition_id) & 0xFFFF),
        row.tobytes(order="C"),
        col.tobytes(order="C"),
        cross.tobytes(order="C"),
        _encode_warp(wx).tobytes(order="C"),
        _encode_warp(wy).tobytes(order="C"),
        struct.pack("<ffB3s",
                    float(r.response_strength),
                    float(r.response_delay),
                    int(r.accumulated_samples) & 0xFF,
                    b"\x00\x00\x00"),
    ]
    out = b"".join(parts)
    if len(out) != RESPONSE_RECORD_SIZE:
        raise AssertionError(
            f"response record size mismatch: {len(out)} != {RESPONSE_RECORD_SIZE}")
    return out


def _unpack_response(buf: bytes) -> SSSInteractionResponse:
    if len(buf) != RESPONSE_RECORD_SIZE:
        raise ValueError(
            f"response record must be {RESPONSE_RECORD_SIZE} bytes, "
            f"got {len(buf)}")
    off = 0
    mid, cid = struct.unpack_from("<HH", buf, off)
    off += struct.calcsize("<HH")
    row = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    col = np.frombuffer(buf[off:off + 4 * SFB_FREQ_BINS],
                        dtype="<f4").astype(np.float32)
    off += 4 * SFB_FREQ_BINS
    cross = np.frombuffer(buf[off:off + 4 * SFB_TEMPORAL_BINS],
                          dtype="<f4").astype(np.float32)
    off += 4 * SFB_TEMPORAL_BINS
    wx_u8 = np.frombuffer(buf[off:off + SFB_WARP_DIM * SFB_WARP_DIM],
                          dtype=np.uint8).reshape(SFB_WARP_DIM, SFB_WARP_DIM)
    off += SFB_WARP_DIM * SFB_WARP_DIM
    wy_u8 = np.frombuffer(buf[off:off + SFB_WARP_DIM * SFB_WARP_DIM],
                          dtype=np.uint8).reshape(SFB_WARP_DIM, SFB_WARP_DIM)
    off += SFB_WARP_DIM * SFB_WARP_DIM
    strength, delay, accum, _pad = struct.unpack_from("<ffB3s", buf, off)
    return SSSInteractionResponse(
        motif_id=int(mid),
        condition_id=int(cid),
        row_response=row.copy(),
        col_response=col.copy(),
        cross_response=cross.copy(),
        warp_x_response=_decode_warp(wx_u8),
        warp_y_response=_decode_warp(wy_u8),
        response_strength=float(strength),
        response_delay=float(delay),
        accumulated_samples=int(accum),
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


def _validate_v2(conditions, responses, motif_count) -> None:
    n_c = len(conditions)
    for i, c in enumerate(conditions):
        if not (0 <= c.condition_type <= CONDITION_TYPE_MAX):
            raise SFBError(SFB_ERR_INVALID_COND,
                f"condition[{i}] {c.label!r}: condition_type "
                f"{c.condition_type} > {CONDITION_TYPE_MAX}")
    for i, r in enumerate(responses):
        if not (0 <= r.motif_id < motif_count):
            raise SFBError(SFB_ERR_INVALID_RESP,
                f"response[{i}].motif_id={r.motif_id} >= "
                f"motif_count={motif_count}")
        if not (0 <= r.condition_id < n_c):
            raise SFBError(SFB_ERR_INVALID_RESP,
                f"response[{i}].condition_id={r.condition_id} >= "
                f"condition_count={n_c}")


# ── Bank ──────────────────────────────────────────────────────────

@dataclass
class SSSFeatureBank:
    motifs: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    identities: list = field(default_factory=list)
    # Phase 4 v2 additions
    conditions: list = field(default_factory=list)
    responses: list = field(default_factory=list)

    def save(self, path: str) -> None:
        _validate(self.motifs, self.relations, self.identities)
        _validate_v2(self.conditions, self.responses, len(self.motifs))
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
            len(self.conditions) & 0xFFFF,          # condition_count
            len(self.responses) & 0xFFFFFFFF,       # response_count u32
            0,                                       # reserved1 u32
            0,                                       # reserved2 u32
        )
        chunks = [header]
        for m in self.motifs:
            chunks.append(_pack_motif(m))
        for r in self.relations:
            chunks.append(_pack_relation(r))
        for idn in self.identities:
            chunks.append(_pack_identity(idn))
        for c in self.conditions:
            chunks.append(_pack_condition(c))
        for r in self.responses:
            chunks.append(_pack_response(r))
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
         condition_count_or_flags,
         response_count_or_reserved,
         _reserved1, _reserved2) = struct.unpack_from(_HEADER_FMT, blob, 0)

        if magic != SFB_MAGIC:
            raise SFBError(SFB_ERR_MAGIC, f".sfb: bad magic {magic!r}")

        # v2 reader accepts both versions; v1 files have implicit
        # condition_count = 0 / response_count = 0 (those bytes were
        # `flags + reserved` and don't carry meaningful counts).
        if version == 1:
            n_conditions = 0
            n_responses  = 0
            # v1 motif on-disk size — special-case so the
            # zero-padding load can lift v1 records into v2 structs.
            v1_motif_layout = (motif_rs == V1_MOTIF_RECORD_SIZE)
        elif version == SFB_VERSION:
            n_conditions = int(condition_count_or_flags)
            n_responses  = int(response_count_or_reserved)
            v1_motif_layout = False
        else:
            raise SFBError(SFB_ERR_VERSION,
                f".sfb: unsupported version {version}")

        # Record size sanity. v1 motif (2864) is allowed only on v1
        # files; anything else must be ≥ current struct size.
        if not v1_motif_layout and motif_rs < MOTIF_RECORD_SIZE:
            raise SFBError(SFB_ERR_VERSION,
                f".sfb: motif_record_size {motif_rs} < {MOTIF_RECORD_SIZE}")
        if relation_rs < RELATION_RECORD_SIZE \
                or identity_rs < IDENTITY_RECORD_SIZE:
            raise SFBError(SFB_ERR_VERSION,
                f".sfb: relation/identity record_size below current minimum")

        off = _HEADER_SIZE
        need = off + motif_rs * n_motifs + relation_rs * n_relations \
                   + identity_rs * n_identities \
                   + CONDITION_RECORD_SIZE * n_conditions \
                   + RESPONSE_RECORD_SIZE * n_responses
        if need > len(blob):
            raise SFBError(SFB_ERR_IO,
                f".sfb: truncated body (need {need}, have {len(blob)})")

        motifs = []
        for _ in range(n_motifs):
            if v1_motif_layout:
                # v1 record on disk → zero-pad to v2 in _unpack_motif.
                motifs.append(_unpack_motif(blob[off:off + V1_MOTIF_RECORD_SIZE]))
            else:
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

        conditions = []
        for _ in range(n_conditions):
            conditions.append(_unpack_condition(
                blob[off:off + CONDITION_RECORD_SIZE]))
            off += CONDITION_RECORD_SIZE

        responses = []
        for _ in range(n_responses):
            responses.append(_unpack_response(
                blob[off:off + RESPONSE_RECORD_SIZE]))
            off += RESPONSE_RECORD_SIZE

        bank = cls(
            motifs=motifs, relations=relations, identities=identities,
            conditions=conditions, responses=responses,
        )
        _validate(bank.motifs, bank.relations, bank.identities)
        _validate_v2(bank.conditions, bank.responses, len(bank.motifs))
        return bank

    def find_motif(self, label: str) -> int:
        """Return the first motif index matching `label`, or -1."""
        for i, m in enumerate(self.motifs):
            if m.label == label:
                return i
        return -1

    def find_condition(self, label: str) -> int:
        """Return the first condition index matching `label`, or -1."""
        for i, c in enumerate(self.conditions):
            if c.label == label:
                return i
        return -1

    def responses_for_motif(self, motif_id: int) -> list:
        """All InteractionResponse rows whose motif_id matches.
        Linear scan — Phase 4 v1 doesn't index this for speed since
        typical banks have < 1000 responses."""
        return [r for r in self.responses if r.motif_id == motif_id]


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
    """Round-trip a bank through libsss_pybridge.so's v2 save path.
    Byte-identical with `SSSFeatureBank.save` for the same `bank`."""
    ctypes, lib = _c_lib()
    if not hasattr(lib, "sss_pybridge_feature_bank_save_v2"):
        raise RuntimeError(
            "libsss_pybridge.so is missing sss_pybridge_feature_bank_save_v2; "
            "rebuild with `make pybridge`.")
    _validate(bank.motifs, bank.relations, bank.identities)
    _validate_v2(bank.conditions, bank.responses, len(bank.motifs))

    motif_blob = b"".join(_pack_motif(m)     for m in bank.motifs)
    rel_blob   = b"".join(_pack_relation(r)  for r in bank.relations)
    ident_blob = b"".join(_pack_identity(i)  for i in bank.identities)
    cond_blob  = b"".join(_pack_condition(c) for c in bank.conditions)
    resp_blob  = b"".join(_pack_response(r)  for r in bank.responses)

    def _buf(blob: bytes):
        if not blob:
            return ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
        return (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)

    rc = lib.sss_pybridge_feature_bank_save_v2(
        path.encode("utf-8"),
        _buf(motif_blob), ctypes.c_uint32(len(bank.motifs)),
        _buf(rel_blob),   ctypes.c_uint32(len(bank.relations)),
        _buf(ident_blob), ctypes.c_uint32(len(bank.identities)),
        _buf(cond_blob),  ctypes.c_uint32(len(bank.conditions)),
        _buf(resp_blob),  ctypes.c_uint32(len(bank.responses)),
    )
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_save_v2 returned {rc}")


def load_via_c(path: str) -> SSSFeatureBank:
    """Round-trip the on-disk file through libsss_pybridge.so's v2
    load path. Accepts both v1 and v2 files; v1 files come back with
    `conditions == [] and responses == []` and motifs lifted into
    v2 structs with `temporal_length = 0`."""
    ctypes, lib = _c_lib()
    for name in ("sss_pybridge_feature_bank_probe_v2",
                 "sss_pybridge_feature_bank_load_v2"):
        if not hasattr(lib, name):
            raise RuntimeError(
                f"libsss_pybridge.so is missing {name}; rebuild with "
                "`make pybridge`.")

    nm = ctypes.c_uint32(0); nr = ctypes.c_uint32(0); ni = ctypes.c_uint32(0)
    nc = ctypes.c_uint32(0); np_ = ctypes.c_uint32(0)
    rc = lib.sss_pybridge_feature_bank_probe_v2(
        path.encode("utf-8"),
        ctypes.byref(nm), ctypes.byref(nr), ctypes.byref(ni),
        ctypes.byref(nc), ctypes.byref(np_))
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_probe_v2 returned {rc}")

    motif_buf = (ctypes.c_uint8 * (MOTIF_RECORD_SIZE * nm.value))()
    rel_buf   = (ctypes.c_uint8 * (RELATION_RECORD_SIZE * nr.value))()
    ident_buf = (ctypes.c_uint8 * (IDENTITY_RECORD_SIZE * ni.value))()
    cond_buf  = (ctypes.c_uint8 * (CONDITION_RECORD_SIZE * nc.value))()
    resp_buf  = (ctypes.c_uint8 * (RESPONSE_RECORD_SIZE * np_.value))()
    nm_cap = ctypes.c_uint32(nm.value)
    nr_cap = ctypes.c_uint32(nr.value)
    ni_cap = ctypes.c_uint32(ni.value)
    nc_cap = ctypes.c_uint32(nc.value)
    np_cap = ctypes.c_uint32(np_.value)
    rc = lib.sss_pybridge_feature_bank_load_v2(
        path.encode("utf-8"),
        motif_buf, ctypes.byref(nm_cap),
        rel_buf,   ctypes.byref(nr_cap),
        ident_buf, ctypes.byref(ni_cap),
        cond_buf,  ctypes.byref(nc_cap),
        resp_buf,  ctypes.byref(np_cap),
    )
    if rc != SFB_OK:
        raise SFBError(rc, f"sss_pybridge_feature_bank_load_v2 returned {rc}")

    motifs     = [_unpack_motif(bytes(motif_buf[i * MOTIF_RECORD_SIZE
                                                :(i + 1) * MOTIF_RECORD_SIZE]))
                  for i in range(nm_cap.value)]
    relations  = [_unpack_relation(bytes(rel_buf[i * RELATION_RECORD_SIZE
                                                 :(i + 1) * RELATION_RECORD_SIZE]))
                  for i in range(nr_cap.value)]
    identities = [_unpack_identity(bytes(ident_buf[i * IDENTITY_RECORD_SIZE
                                                   :(i + 1) * IDENTITY_RECORD_SIZE]))
                  for i in range(ni_cap.value)]
    conditions = [_unpack_condition(bytes(cond_buf[i * CONDITION_RECORD_SIZE
                                                    :(i + 1) * CONDITION_RECORD_SIZE]))
                  for i in range(nc_cap.value)]
    responses  = [_unpack_response(bytes(resp_buf[i * RESPONSE_RECORD_SIZE
                                                   :(i + 1) * RESPONSE_RECORD_SIZE]))
                  for i in range(np_cap.value)]
    return SSSFeatureBank(
        motifs=motifs, relations=relations, identities=identities,
        conditions=conditions, responses=responses,
    )
