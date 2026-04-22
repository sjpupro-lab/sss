#ifndef SPATIAL_KEYFRAME_H
#define SPATIAL_KEYFRAME_H

#include "spatial_grid.h"
#include "spatial_match.h"

/* Keyframe (I-Frame): full snapshot.
 * topic_hash + seq_in_topic drive the topic-aware "next frame"
 * lookup in ai_generate_next. Both default to 0; ai_store_auto and
 * ai_force_keyframe derive topic_hash from the label (djb2) and
 * assign the next seq within the same topic. Legacy v3 files
 * load with topic_hash=0 / seq_in_topic=0 and still work via the
 * id+1 fallback path. */
typedef struct {
    uint32_t    id;
    char        label[64];
    SpatialGrid grid;
    uint32_t    text_byte_count;
    uint32_t    topic_hash;
    uint32_t    seq_in_topic;
} Keyframe;

/* Delta entry: sparse format (SPEC-ENGINE Phase D).
 * Bumped to 9 logical bytes with diff_B. On-disk is written field-by-
 * field so struct padding and cross-version size changes don't break
 * compatibility — see write_delta_record / read_delta_body. */
typedef struct {
    uint32_t index;     /* y * 256 + x */
    int16_t  diff_A;
    int8_t   diff_R;
    int8_t   diff_G;
    int8_t   diff_B;
} DeltaEntry;

/* RLE delta entry (for 4096 scale) */
typedef struct {
    uint32_t start;
    uint16_t length;
    int16_t  diff;
} RLEDelta;  /* 8 bytes */

/* Delta frame (P-Frame) */
typedef struct {
    uint32_t     id;
    uint32_t     parent_id;
    char         label[64];
    uint32_t     count;
    DeltaEntry*  entries;
    float        change_ratio;
} DeltaFrame;

/* Main AI engine structure.
 * Named struct (SpatialAI_) so spatial_match.h can forward-declare it
 * for the cascade API without a circular include. */
struct SpatialCanvasPool_;  /* fwd-decl; see spatial_subtitle.h */
typedef struct SpatialAI_ {
    Keyframe*     keyframes;
    uint32_t      kf_count;
    uint32_t      kf_capacity;
    DeltaFrame*   deltas;
    uint32_t      df_count;
    uint32_t      df_capacity;

    /* Adaptive channel weights (SPEC §5: dynamic RGB embedding).
     * Initialised to (1, 1, 1, 1) by spatial_ai_create; updated
     * automatically after each ai_store_auto by the engine. */
    ChannelWeight global_weights;

    /* Optional canvas pool (SPEC §6 + SubtitleTrack). NULL until
     * ai_get_canvas_pool(ai) is called. */
    struct SpatialCanvasPool_* canvas_pool;

    /* Hash bucket index over all keyframes. Populated by
     * ai_store_auto/ai_force_keyframe so large-corpus retrieval
     * (kf_count >= BUCKET_THRESHOLD) can skip the O(N) overlap scan.
     * Managed entirely in-memory; not serialized — rebuilt on load. */
    BucketIndex bucket_idx;

    /* Morpheme-dictionary readiness flag. Set by spatial_ai_create; all
     * hot paths (ai_store_auto / ai_predict / ai_generate_next /
     * ai_force_keyframe) assume this is already true and skip the
     * morpheme_init() call. */
    int morpheme_ready;

    /* RGB EMA tables indexed by (y * 256 + x). Accumulated across
     * every stored clause so R/G/B values stabilize at each bitmap
     * position as training progresses. ema_count is the running
     * number of times the cell was active; used to skip cells that
     * haven't been seen enough to be trustworthy. Serialized as an
     * optional SPAI_TAG_EMA trailing record. Size: 4 * GRID_TOTAL * 4
     * bytes = 1 MB. */
    float ema_R    [GRID_SIZE * GRID_SIZE];
    float ema_G    [GRID_SIZE * GRID_SIZE];
    float ema_B    [GRID_SIZE * GRID_SIZE];
    float ema_count[GRID_SIZE * GRID_SIZE];
} SpatialAI;

/* Blend EMA into a newly-encoded grid. Called right after
 * update_rgb_directional and before matching. Cells whose
 * ema_count[i] < 2 are left untouched (not enough evidence). */
void apply_ema_to_grid(const SpatialAI* ai, SpatialGrid* grid);

/* Update the EMA tables from a stored grid. Called once per
 * ai_store_auto / ai_force_keyframe after the frame is committed. */
void ema_update(SpatialAI* ai, const SpatialGrid* grid);

/* Tune the similarity threshold ai_store_auto uses to decide
 * delta vs new keyframe. Default 0.30 (engine-wide, process-local).
 * Lower values (e.g. 0.15 for wiki abstracts) produce more deltas
 * and a smaller model. Clamped to [0, 1]. */
void  ai_set_store_threshold(float t);
float ai_get_store_threshold(void);

/* Resolve the topic hash that ai_store_auto would assign to this
 * clause/label pair. Used by calibration code that needs to look up
 * the same topic bucket without committing the clause yet. Returns
 * 0 when both inputs yield no bytes. */
uint32_t ai_resolve_topic(const char* clause_text, const char* label);

/* Forward declarations that avoid pulling spatial_subtitle.h into
 * every translation unit that needs SpatialAI. */
struct SpatialCanvasPool_* ai_get_canvas_pool(SpatialAI* ai);  /* lazy create */
void                       ai_release_canvas_pool(SpatialAI* ai); /* destroy+NULL */

/* Create/destroy engine */
SpatialAI* spatial_ai_create(void);
void       spatial_ai_destroy(SpatialAI* ai);

/* Store a clause: auto-detect keyframe vs delta (threshold 0.3).
   Returns the stored frame ID. */
uint32_t ai_store_auto(SpatialAI* ai,
                       const char* clause_text,
                       const char* label);

/* Always store a clause as a new keyframe, bypassing the delta
   decision. Needed when callers require a 1-1 clause ↔ keyframe
   mapping (e.g. context frames for QA retrieval).
   Returns the new keyframe ID (== ai->kf_count - 1 on success),
   or UINT32_MAX on failure. */
uint32_t ai_force_keyframe(SpatialAI* ai,
                           const char* clause_text,
                           const char* label);

/* Compute delta between two grids.
   Returns number of changed pixels. entries must be pre-allocated. */
uint32_t compute_delta(const SpatialGrid* base, const SpatialGrid* target,
                       DeltaEntry* entries, uint32_t max_entries);

/* Apply delta to reconstruct target from base */
void apply_delta(const SpatialGrid* base, const DeltaEntry* entries,
                 uint32_t count, SpatialGrid* out);

/* Predict: find best matching keyframe for input text.
   Returns keyframe ID, writes similarity to out_similarity. */
uint32_t ai_predict(SpatialAI* ai,
                    const char* input_text,
                    float* out_similarity);

#endif /* SPATIAL_KEYFRAME_H */
