/* sss_feature_bank.h — Phase 2 .sfb (SSS Feature Bank) on-disk format.
 *
 * ── Design philosophy ─────────────────────────────────────────────
 *
 * The .sfb file is a **feature spectral dictionary**, not an image
 * repository. The unit of storage is a *motif*, never a pixel or a
 * row. A motif represents "an abstract feature with a particular
 * frequency tendency" and absolutely nothing else:
 *
 *   - No phase information is ever written to disk. Phase 1 already
 *     established this for the spectrogram generator; .sfb extends
 *     the same rule to every higher-level feature.
 *   - No row indices, no row order, no per-row spectra. The legacy
 *     .sss v9 layout's (H × NF × 3) per-row FFT amplitudes are
 *     collapsed into a single 128-bin row_freq envelope per motif.
 *   - No original pixel coordinates. position_heatmap is a coarse
 *     16×16 probability surface, not a pixel-addressable map.
 *
 * Consequence at generate time: the generator synthesises **a new
 * signal** that follows the motif's envelope. No row, pixel, or
 * fragment of a training image is ever directly invoked. The same
 * (motif, seed) pair produces a fresh waveform each call by design —
 * the diversity that Phase 1's per-seed random phase added at the
 * spectrogram level continues to hold here at the motif level.
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
#define SFB_VERSION               1u

#define SFB_LABEL_LEN             32
#define SFB_FREQ_BINS             128
#define SFB_HEATMAP_DIM           16
#define SFB_MAX_IDENTITY_MOTIFS   32

#define SFB_RELATION_TYPE_ABOVE   0u
#define SFB_RELATION_TYPE_BELOW   1u
#define SFB_RELATION_TYPE_LEFT    2u
#define SFB_RELATION_TYPE_RIGHT   3u
#define SFB_RELATION_TYPE_NEAR    4u
#define SFB_RELATION_TYPE_AROUND  5u
#define SFB_RELATION_TYPE_INSIDE  6u
#define SFB_RELATION_TYPE_MAX     SFB_RELATION_TYPE_INSIDE

/* Result codes. SFB_OK on success; negative on any error. */
#define SFB_OK                     0
#define SFB_ERR_IO                (-1)
#define SFB_ERR_MAGIC             (-2)
#define SFB_ERR_VERSION           (-3)
#define SFB_ERR_INVALID_REL       (-4)
#define SFB_ERR_INVALID_IDENT     (-5)
#define SFB_ERR_ALLOC             (-6)

#pragma pack(push, 1)

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t motif_count;
    uint32_t relation_count;
    uint32_t identity_count;
    uint16_t motif_record_size;
    uint16_t relation_record_size;
    uint16_t identity_record_size;
    uint16_t flags;
    uint64_t reserved;
    uint32_t reserved2;            /* header padding to 40 bytes */
} SSSFeatureBankHeader;

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

#pragma pack(pop)

/* In-memory bank. The save / load functions own the three arrays
 * (loaded from disk via malloc); callers building a bank in memory
 * are free to use stack / arena allocation as long as they set the
 * counts and pointers correctly. sss_feature_bank_free() only
 * touches malloc'd pointers — passing NULL pointers + 0 counts is a
 * no-op, and callers can mix-and-match (e.g. own motifs externally
 * while letting the loader own relations). */
typedef struct {
    uint32_t     motif_count;
    SSSMotif    *motifs;
    uint32_t     relation_count;
    SSSRelation *relations;
    uint32_t     identity_count;
    SSSIdentity *identities;
} SSSFeatureBank;

/* Serialise `bank` to `path`. Records are validated:
 *   - relation.src/dst < motif_count
 *   - relation.relation_type <= SFB_RELATION_TYPE_MAX
 *   - identity.motif_count in [1, SFB_MAX_IDENTITY_MOTIFS]
 *   - identity.motif_ids[k] < motif_count for k < motif_count
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
