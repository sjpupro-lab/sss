/* sss_feature_bank.h — .sfb (SSS Feature Bank) on-disk format,
 * version 2 (Phase 4 Condition-Interaction Paradigm).
 *
 * ── Design philosophy (Phase 4) ──────────────────────────────────
 *
 * The sss engine is a **resonance-based visual-signal synthesiser**
 * that treats "time-frozen" signals (images) and "time-flowing"
 * signals (videos) through the same motif dictionary. A motif
 * carries an abstract frequency tendency; a condition signal
 * carries a time-axis stimulus; and an interaction response
 * describes how a motif's envelope and spatial warp react when
 * that condition is applied at a given intensity.
 *
 * Core principles inherited from Phase 1–3 and reasserted here:
 *
 *   - An image is the temporal_length = 0 special case of a video.
 *     The synthesiser uses the same code path for both; only the
 *     condition signal's time axis distinguishes the two.
 *   - The motif learns by *accumulating responses from similar
 *     motifs*, not by stochastic weight updates. Phase 4's
 *     accumulator stores a coherence-weighted average of every
 *     observed (motif, condition) response.
 *   - Time lives only on the condition signal's time axis. The
 *     motif itself is static; "movement" is what happens when a
 *     condition is applied over a sequence of t-values.
 *   - **No phase, no per-row FFT, no original pixel / frame data**
 *     ever lands on disk. Phase 1 established the rule; Phase 2
 *     extended it; Phase 4 holds the line.
 *
 * The .sfb file is a **feature spectral dictionary**, not an image
 * repository. A motif represents "an abstract feature with a
 * particular frequency tendency" and absolutely nothing else:
 *
 *   - No phase information is ever written to disk. Phase 1 already
 *     established this for the spectrogram generator; .sfb extends
 *     the same rule to every higher-level feature.
 *   - No row indices, no row order, no per-row spectra. The legacy
 *     .sss v9 layout's (H × NF × 3) per-row FFT amplitudes are
 *     collapsed into a single 128-bin row_freq envelope per motif.
 *   - No original pixel coordinates. position_heatmap is a coarse
 *     16×16 probability surface, not a pixel-addressable map.
 *   - **No raw frames.** Conditions store an amplitude envelope of
 *     the time axis (`temporal_freq[64]`); responses store the
 *     envelope and warp *delta* the motif exhibits under the
 *     condition. The training pipeline never persists pixel-level
 *     deltas.
 *
 * Consequence at generate time: the generator synthesises **a new
 * signal** that follows the motif's envelope. No row, pixel, or
 * fragment of a training image is ever directly invoked. Generation
 * is deterministic per seed — a fixed seed reproduces the same
 * output bit-for-bit — but varying the seed varies the waveform
 * (the per-seed random phase init from Phase 1's spectrogram
 * generator continues to hold at the motif level). Reproducibility
 * for a fixed seed is an engine guarantee; nondeterminism is not
 * introduced.
 *
 * This is enforced by the on-disk schema below: there is simply no
 * field in which a writer could smuggle phase, row order, or pixel
 * data without bumping the file version.
 *
 * ── Layout overview ──────────────────────────────────────────────
 *
 * A feature bank is a flat, self-describing collection of three
 * record arrays:
 *   - Motifs:     per-token amplitude envelopes (row/col/colour
 *                 spectra) + spatial heatmap + scalar quality fields.
 *   - Relations:  directed (src → dst) edges with a typed spatial
 *                 hint (above / below / left / right / near / around /
 *                 inside) and a learned weight.
 *   - Identities: named clusters of up to 32 motifs forming a single
 *                 high-level concept (e.g. "kitty" → ear motif +
 *                 eye motif + bow motif).
 *
 * The format is pack(1) little-endian. Every record size is also
 * stored in the header so future Phase 4 motion fields can grow the
 * record without bumping the file version — readers stride by the
 * header-supplied record_size and ignore unknown trailing bytes.
 *
 * Byte map of the header (40 bytes total):
 *
 *   off  size  field
 *     0     4  magic         "SFB1"
 *     4     4  version       1
 *     8     4  motif_count
 *    12     4  relation_count
 *    16     4  identity_count
 *    20     2  motif_record_size       (v1: 2864)
 *    22     2  relation_record_size    (v1: 20)
 *    24     2  identity_record_size    (v1: 104)
 *    26     2  flags                   reserved, currently 0
 *    28     8  reserved                u64 zero
 *    36     4  reserved2               u32 zero (header padding to 40)
 *    40 ─── end of header
 *
 * Then come the three record arrays in order: motifs, relations,
 * identities. Each array is a tight (no padding) run of
 * `*_record_size` bytes.
 *
 * Motif (2864 bytes):
 *    char     label[32]                  utf-8, null- or len-terminated
 *    float32  row_freq[128]              512 B
 *    float32  col_freq[128]              512 B
 *    float32  color_freq[3][128]         1536 B (R, G, B)
 *    uint8    position_heatmap[16][16]   256 B   (0..255 quantised)
 *    float32  coherence
 *    float32  confidence
 *    uint32   activation_count
 *    uint16   variation_cluster_id
 *    uint16   _pad
 *
 * Relation (20 bytes):
 *    uint16   src_motif_id
 *    uint16   dst_motif_id
 *    float32  dx
 *    float32  dy
 *    uint8    relation_type              SFB_RELATION_TYPE_*
 *    uint8    _pad[3]
 *    float32  weight
 *
 * Identity (104 bytes):
 *    char     label[32]
 *    uint16   motif_ids[32]              64 B, only first motif_count used
 *    uint8    motif_count                1..32
 *    uint8    _pad[3]
 *    float32  confidence
 */
#ifndef SSS_FEATURE_BANK_H
#define SSS_FEATURE_BANK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SFB_MAGIC                 "SFB1"     /* 4 bytes, no NUL */
#define SFB_VERSION               2u

#define SFB_LABEL_LEN             32
#define SFB_FREQ_BINS             128
#define SFB_HEATMAP_DIM           16
#define SFB_MAX_IDENTITY_MOTIFS   32

/* Phase 4 v2 dimensions */
#define SFB_TEMPORAL_BINS         64
#define SFB_WARP_DIM              16
#define SFB_MAX_RESPONSES_PER_MOTIF 16

#define SFB_RELATION_TYPE_ABOVE   0u
#define SFB_RELATION_TYPE_BELOW   1u
#define SFB_RELATION_TYPE_LEFT    2u
#define SFB_RELATION_TYPE_RIGHT   3u
#define SFB_RELATION_TYPE_NEAR    4u
#define SFB_RELATION_TYPE_AROUND  5u
#define SFB_RELATION_TYPE_INSIDE  6u
#define SFB_RELATION_TYPE_MAX     SFB_RELATION_TYPE_INSIDE

/* Phase 4 condition types */
#define SFB_CONDITION_CONTINUOUS  0u
#define SFB_CONDITION_IMPULSE     1u
#define SFB_CONDITION_OSCILLATORY 2u
#define SFB_CONDITION_TYPE_MAX    SFB_CONDITION_OSCILLATORY

/* Result codes. SFB_OK on success; negative on any error. */
#define SFB_OK                     0
#define SFB_ERR_IO                (-1)
#define SFB_ERR_MAGIC             (-2)
#define SFB_ERR_VERSION           (-3)
#define SFB_ERR_INVALID_REL       (-4)
#define SFB_ERR_INVALID_IDENT     (-5)
#define SFB_ERR_ALLOC             (-6)
#define SFB_ERR_INVALID_COND     (-7)
#define SFB_ERR_INVALID_RESP     (-8)

#pragma pack(push, 1)

/* Header layout — 40 bytes total. The v2 format reuses the byte
 * offsets the Phase 2 header carved out for `flags` (u16) and
 * `reserved` (u64) but reinterprets them as `condition_count` (u16)
 * + `response_count` (u32) + `reserved1` (u32). v1 readers see
 * those bytes as their reserved fields and ignore them; v2 readers
 * see meaningful counts. The 40 B header is therefore stable across
 * the version bump. */
typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t motif_count;
    uint32_t relation_count;
    uint32_t identity_count;
    uint16_t motif_record_size;
    uint16_t relation_record_size;
    uint16_t identity_record_size;
    uint16_t condition_count;     /* v1 was: flags (now reinterpreted) */
    uint32_t response_count;      /* v1 was: first 4 B of reserved u64 */
    uint32_t reserved1;           /* v1 was: last  4 B of reserved u64 */
    uint32_t reserved2;
} SSSFeatureBankHeader;

/* Motif v2 (3744 B) = v1 motif (2864 B) + 880 B Phase 4 extension.
 *
 * The 880 B extension carries:
 *   uint16   temporal_length          0 = static motif, N = N-frame motif
 *   uint16   response_count_in_motif  # of InteractionResponse rows
 *                                     keyed by this motif (informational
 *                                     — the actual lookup walks the
 *                                     responses[] array by motif_id).
 *   uint32   response_offset          first index in responses[] that
 *                                     belongs to this motif (set at
 *                                     save time so the loader can group
 *                                     responses by motif without a scan).
 *   float32  cross_resonance[64]      256 B — cross-axis modulation
 *                                     side-band; the synthesiser
 *                                     interprets k-th bin as
 *                                     cos(2π(row_k + col_k)·i) ·
 *                                     cos(2π(row_k - col_k)·j).
 *   uint8    warp_self_x[16][16]      256 B — motif's baseline shape
 *   uint8    warp_self_y[16][16]      256 B   warp (without any
 *                                     condition applied). Phase-4 spec
 *                                     option A choice; uint8 with a
 *                                     centre-128 + scale-128 convention
 *                                     so the field decodes to
 *                                     [-1, +1] displacement.
 *   uint8    _pad[104]                Phase 5 / motion field hint.
 *
 * Temporal envelope is **not** stored on the motif (Phase 4 design
 * decision): per-condition `temporal_freq[64]` carries the time-axis
 * spectrum where it belongs, and storing a redundant per-motif copy
 * was rejected during v2 spec review. */
typedef struct {
    char     label[SFB_LABEL_LEN];
    float    row_freq[SFB_FREQ_BINS];
    float    col_freq[SFB_FREQ_BINS];
    float    color_freq[3][SFB_FREQ_BINS];
    uint8_t  position_heatmap[SFB_HEATMAP_DIM][SFB_HEATMAP_DIM];
    float    coherence;
    float    confidence;
    uint32_t activation_count;
    uint16_t variation_cluster_id;
    uint16_t _pad;
    /* ── v2 extension (880 B) ───────────────────────────────────── */
    uint16_t temporal_length;
    uint16_t response_count_in_motif;
    uint32_t response_offset;
    float    cross_resonance[SFB_TEMPORAL_BINS];
    uint8_t  warp_self_x[SFB_WARP_DIM][SFB_WARP_DIM];
    uint8_t  warp_self_y[SFB_WARP_DIM][SFB_WARP_DIM];
    uint8_t  _pad_v2[104];
} SSSMotif;

typedef struct {
    uint16_t src_motif_id;
    uint16_t dst_motif_id;
    float    dx;
    float    dy;
    uint8_t  relation_type;        /* SFB_RELATION_TYPE_* */
    uint8_t  _pad[3];
    float    weight;
} SSSRelation;

typedef struct {
    char     label[SFB_LABEL_LEN];
    uint16_t motif_ids[SFB_MAX_IDENTITY_MOTIFS];
    uint8_t  motif_count;          /* 1..SFB_MAX_IDENTITY_MOTIFS */
    uint8_t  _pad[3];
    float    confidence;
} SSSIdentity;

/* Phase 4 condition signal — 1332 B. Per the design philosophy,
 * conditions are *signals*, not pixels: each carries an amplitude
 * envelope along row, column, and the temporal axis plus a
 * direction hint and decay parameters. No phase, no row order. */
typedef struct {
    char     label[SFB_LABEL_LEN];
    uint8_t  condition_type;       /* SFB_CONDITION_* */
    uint8_t  _pad[3];
    float    row_freq[SFB_FREQ_BINS];
    float    col_freq[SFB_FREQ_BINS];
    float    temporal_freq[SFB_TEMPORAL_BINS];
    float    direction;            /* radians, -π..π */
    float    default_intensity;    /* 0..1 nominal scale */
    float    decay_rate;           /* impulse-only */
    uint32_t _pad2;
} SSSCondition;

/* Phase 4 interaction response — 1808 B. Per (motif, condition) we
 * accumulate an envelope-side response (row_response / col_response /
 * cross_response) and a warp-side response (warp_*_response, uint8
 * with the [-1, +1] decode convention shared with warp_self). */
typedef struct {
    uint16_t motif_id;
    uint16_t condition_id;
    float    row_response[SFB_FREQ_BINS];
    float    col_response[SFB_FREQ_BINS];
    float    cross_response[SFB_TEMPORAL_BINS];
    uint8_t  warp_x_response[SFB_WARP_DIM][SFB_WARP_DIM];
    uint8_t  warp_y_response[SFB_WARP_DIM][SFB_WARP_DIM];
    float    response_strength;
    float    response_delay;
    uint8_t  accumulated_samples;
    uint8_t  _pad[3];
} SSSInteractionResponse;

#pragma pack(pop)

/* In-memory bank. The save / load functions own the five arrays
 * (loaded from disk via malloc); callers building a bank in memory
 * are free to use stack / arena allocation as long as they set the
 * counts and pointers correctly. sss_feature_bank_free() only
 * touches malloc'd pointers — passing NULL pointers + 0 counts is a
 * no-op, and callers can mix-and-match (e.g. own motifs externally
 * while letting the loader own relations).
 *
 * `conditions` and `responses` are Phase 4 v2 additions. v1 files
 * load with condition_count == 0 and response_count == 0 (the
 * loader simply skips the trailing v2 arrays). */
typedef struct {
    uint32_t                motif_count;
    SSSMotif               *motifs;
    uint32_t                relation_count;
    SSSRelation            *relations;
    uint32_t                identity_count;
    SSSIdentity            *identities;
    uint32_t                condition_count;
    SSSCondition           *conditions;
    uint32_t                response_count;
    SSSInteractionResponse *responses;
} SSSFeatureBank;

/* Serialise `bank` to `path`. Records are validated:
 *   - relation.src/dst < motif_count
 *   - relation.relation_type <= SFB_RELATION_TYPE_MAX
 *   - identity.motif_count in [1, SFB_MAX_IDENTITY_MOTIFS]
 *   - identity.motif_ids[k] < motif_count for k < motif_count
 *   - condition.condition_type <= SFB_CONDITION_TYPE_MAX
 *   - response.motif_id < motif_count
 *   - response.condition_id < condition_count
 * Returns SFB_OK or a negative SFB_ERR_*. On failure the output
 * file may be partially written; callers that care about atomicity
 * should write to a temp path and rename. */
int  sss_feature_bank_save(const char *path, const SSSFeatureBank *bank);

/* Load `path` into `*out`. On success *out owns the three record
 * arrays via malloc — release with sss_feature_bank_free(). On
 * failure *out is zero-initialised; no allocations remain. */
int  sss_feature_bank_load(const char *path, SSSFeatureBank *out);

/* Free the malloc'd record arrays inside `bank` (set by
 * sss_feature_bank_load). Zero-initialises the struct after. */
void sss_feature_bank_free(SSSFeatureBank *bank);

/* Linear search: returns the first index whose motif label matches
 * `label` exactly (NUL-terminated, byte-compare), or -1 if none.
 * `label` longer than SFB_LABEL_LEN-1 will never match because the
 * on-disk label is truncated. */
int  sss_feature_bank_find_motif(const SSSFeatureBank *bank,
                                 const char *label);

#ifdef __cplusplus
}
#endif

#endif /* SSS_FEATURE_BANK_H */
