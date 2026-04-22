#ifndef SPATIAL_MATCH_H
#define SPATIAL_MATCH_H

#include "spatial_grid.h"
#include <math.h>

/* Directional RGB update parameters */
#define ALPHA_R 0.05f
#define BETA_G  0.08f
#define GAMMA_B 0.03f

/* Top-K and matching constants */
#define TOP_K 8
#define MAX_CANDIDATES 4096
#define BUCKET_THRESHOLD 100
#define NUM_BUCKETS 256

/* Matching candidate */
typedef struct {
    uint32_t id;
    float    score;
} Candidate;

/* Block summary for 16x16 block skip (SPEC-ENGINE Phase B) */
#define BLOCK 16
#define BLOCKS 16

typedef struct {
    uint32_t sum[BLOCKS][BLOCKS];
} BlockSummary;

/* Hash bucket for large-scale search (SPEC-ENGINE Phase C).
 * `ids` grows dynamically; capacity starts at 0 and doubles on demand.
 * bucket_index_destroy must be called before freeing a BucketIndex
 * that has had ids registered. */
typedef struct {
    uint32_t* ids;
    uint32_t  count;
    uint32_t  capacity;
} Bucket;

typedef struct {
    Bucket buckets[NUM_BUCKETS];
} BucketIndex;

/* ── Directional RGB ── */

/* Update RGB channels with directional diffusion */
void update_rgb_directional(SpatialGrid* grid);

/* ── Overlap (Coarse filter) ── */

/* Count active pixels that overlap between two grids */
uint32_t overlap_score(const SpatialGrid* a, const SpatialGrid* b);

/* ── Cosine similarity ── */

/* RGB weight factor for a single pixel pair */
float rgb_weight(uint8_t r1, uint8_t r2,
                 uint8_t g1, uint8_t g2,
                 uint8_t b1, uint8_t b2);

/* RGB-weighted cosine similarity */
float cosine_rgb_weighted(const SpatialGrid* a, const SpatialGrid* b);

/* A-channel only cosine similarity */
float cosine_a_only(const SpatialGrid* a, const SpatialGrid* b);

/* ── Block skip cosine (SPEC-ENGINE Phase B) ── */

/* Compute block sums for a grid */
void compute_block_sums(const SpatialGrid* g, BlockSummary* bs);

/* Cosine with block skip optimization */
float cosine_block_skip(const SpatialGrid* a, const SpatialGrid* b,
                        const BlockSummary* bs_a, const BlockSummary* bs_b);

/* ── Top-K selection ── */

/* Select top-k candidates in-place (partial sort) */
void topk_select(Candidate* pool, uint32_t pool_size, uint32_t k);

/* ── Hash bucket (SPEC-ENGINE Phase C) ── */

/* Compute grid hash based on active X coordinates */
uint32_t grid_hash(const SpatialGrid* g);

/* Initialize bucket index. Zeroes all bucket slots; ids arrays are
 * lazy-allocated by bucket_index_add. */
void bucket_index_init(BucketIndex* idx);

/* Free every bucket's ids array. Safe to call on a zero-init index. */
void bucket_index_destroy(BucketIndex* idx);

/* Add a keyframe id to the bucket for `g`'s hash. Grows the bucket
 * if needed (capacity 0 → 64 → 128 → …). */
void bucket_index_add(BucketIndex* idx, const SpatialGrid* g, uint32_t kf_id);

/* Collect candidate keyframe ids from buckets [hash - expand,
 * hash + expand]. Writes at most `max_out` ids; *out_count receives
 * the number written. */
void bucket_candidates(BucketIndex* idx, uint32_t hash, int expand,
                       uint32_t* out, uint32_t* out_count,
                       uint32_t max_out);

/* ── Channel cascade matching ─────────────────────────────
 * Lego-block style staged matching: channels are combined in
 * ordered pairs rather than all at once. Top-K from one pair is
 * re-scored by another pair for the final match.
 *
 *   CASCADE_SEARCH    A-only baseline (overlap → cosine_a_only)
 *   CASCADE_QA        A → RG pair → BA rematch
 *                       R (diagonal/semantic) × G (vertical/substitution)
 *                       then fix top-K and re-score with
 *                       B (horizontal/co-occurrence) × A (activation).
 *   CASCADE_GENERATE  A → BG pair → RA rematch
 *                       B × G then R × A on top-K.
 *
 * Step 1 early-return: if the A-only cosine on the matched clause is
 * already ≥ CASCADE_STEP1_THRESHOLD (0.5), we've found a
 * structurally-identical clause and return immediately.
 */
typedef enum {
    CASCADE_SEARCH   = 0,
    CASCADE_QA       = 1,
    CASCADE_GENERATE = 2
} CascadeMode;

#define CASCADE_STEP1_THRESHOLD 0.5f

/* Forward declaration — full struct lives in spatial_keyframe.h.
 * Callers that invoke match_cascade must include spatial_keyframe.h
 * (which brings in the full SpatialAI definition).  C11 allows the
 * same typedef to appear in multiple headers. */
typedef struct SpatialAI_ SpatialAI;

/* Best-match cascade over ai->keyframes.
 * out_similarity:
 *   CASCADE_SEARCH       → A-only cosine of best match
 *   CASCADE_QA/GENERATE  → A-only cosine if step 1 fired; otherwise
 *                          RGB-weighted cosine of the final match. */
uint32_t match_cascade(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    float* out_similarity
);

/* Top-K variant. Fills out_ids / out_scores (both capacity >= k) sorted
 * by final cascade score descending. Returns actual count written. */
uint32_t match_cascade_topk(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    uint32_t k,
    uint32_t* out_ids,
    float* out_scores
);

/* Expose channel-pair scoring primitives (used by cascade, also useful
 * for tests and offline analysis). All iterate cells where BOTH a.A>0
 * and b.A>0. */
float rg_score(const SpatialGrid* a, const SpatialGrid* b);
float bg_score(const SpatialGrid* a, const SpatialGrid* b);
float ba_score(const SpatialGrid* a, const SpatialGrid* b);
float ra_score(const SpatialGrid* a, const SpatialGrid* b);

/* ── Unified matching entry point (see below, after ChannelWeight) ── */

/* ── Adaptive channel weights ───────────────────────────
 *
 *   ChannelWeight holds per-channel scalar weights used to combine
 *   A/R/G/B similarities into a single adaptive score.
 *
 *   Invariant maintained by weight_normalize:
 *      w_A + w_R + w_G + w_B == 4.0f   (average weight = 1.0)
 *
 *   Update rule (weight_update):
 *     For each channel k in {A, R, G, B}, we receive a per-channel
 *     similarity sim_k ∈ [0, 1] observed on a successful match. The
 *     channel with the highest similarity is given a small reward and
 *     the others a small decay, then weights are renormalised so the
 *     sum remains 4. This causes the engine to trust the channels
 *     that actually discriminate.
 *
 *   Default learning rate is 0.05 (tuned for visible convergence on
 *   hundreds of training steps).
 */
typedef struct {
    float w_A;
    float w_R;
    float w_G;
    float w_B;
} ChannelWeight;

#define WEIGHT_LEARNING_RATE 0.05f

/* Initialise to (1, 1, 1, 1) — fully uniform. */
void  weight_init(ChannelWeight* w);

/* Rescale so w_A + w_R + w_G + w_B == 4 while preserving ratios. */
void  weight_normalize(ChannelWeight* w);

/* Single feedback update. Given per-channel observed similarities on
 * the winning match, boost the channel with the highest sim by
 * WEIGHT_LEARNING_RATE and decay the others by LR/3, then renormalise. */
void  weight_update(ChannelWeight* w,
                    float sim_A, float sim_R, float sim_G, float sim_B);

/* Per-channel average similarity over cells where BOTH grids are active. */
float channel_sim_A(const SpatialGrid* a, const SpatialGrid* b);
float channel_sim_R(const SpatialGrid* a, const SpatialGrid* b);
float channel_sim_G(const SpatialGrid* a, const SpatialGrid* b);
float channel_sim_B(const SpatialGrid* a, const SpatialGrid* b);

/* Weighted combination:
 *   adaptive_score = (w_A * sim_A + w_R * sim_R + w_G * sim_G + w_B * sim_B) / 4
 *
 * Returns a value in roughly [0, 1]. Used by match_cascade Step 2/3
 * when ChannelWeight is supplied (non-NULL). */
float adaptive_score(const SpatialGrid* a, const SpatialGrid* b,
                     const ChannelWeight* w);

/* Cascade variants that accept ChannelWeight for adaptive scoring.
 * If w == NULL, behave exactly like match_cascade / match_cascade_topk. */
uint32_t match_cascade_weighted(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    const ChannelWeight* w,
    float* out_similarity);

uint32_t match_cascade_topk_weighted(
    SpatialAI* ai,
    SpatialGrid* input,
    CascadeMode mode,
    const ChannelWeight* w,
    uint32_t k,
    uint32_t* out_ids,
    float* out_scores);

/* ── Unified matching entry point (Mod 1) ──
 *
 * Pass NULL for ctx to use default settings. When ctx->bucket_idx is
 * supplied and ai->kf_count >= BUCKET_THRESHOLD, the coarse stage uses
 * the hash bucket instead of a full overlap scan. block_cache and
 * frame_cache are accepted for API symmetry but are currently unused
 * — the underlying cosine paths are already cache-free.
 *
 * frame_cache is held as an opaque pointer to keep spatial_match.h
 * independent of spatial_context.h. */
typedef struct {
    BucketIndex*         bucket_idx;
    BlockSummary*        block_cache;
    void*                frame_cache;  /* opaque; FrameCache* in practice */
    const ChannelWeight* weights;
} MatchContext;

typedef struct {
    uint32_t  best_id;
    float     best_score;
    Candidate topk[TOP_K];
    uint32_t  topk_count;
} MatchResult;

typedef enum {
    MATCH_PREDICT  = 0,  /* overlap → cosine_rgb_weighted (default retrieval) */
    MATCH_SEARCH   = 1,  /* overlap → cosine_a_only (structural) */
    MATCH_QA       = 2,  /* overlap → rg_score (content-channel pair) */
    MATCH_GENERATE = 3   /* overlap → bg_score (extended-channel pair) */
} MatchMode;

MatchResult spatial_match(SpatialAI* ai,
                          const SpatialGrid* input,
                          MatchMode mode,
                          const MatchContext* ctx);

#endif /* SPATIAL_MATCH_H */
