#ifndef SPATIAL_RECLUSTER_H
#define SPATIAL_RECLUSTER_H

/* v4 Task C — periodic / offline re-clustering of a SpatialCanvasPool.
 *
 * Rearranges the *physical* order of same-topic canvases to improve
 * delta/RLE compression while leaving every slot's sequence_id
 * untouched, so the original utterance order is still recoverable via
 * pool_iterate_in_sequence_order.
 *
 * Design rules (see SPEC v4 §ContextPool + §Recluster):
 *   - Grouping uses topic_hash EXACT MATCH only. Do NOT interpret
 *     bit/hamming distance as semantic distance.
 *   - Cost is measured by the real canvas_delta_rle_bytes, not a
 *     hash-based proxy.
 *   - A reorder commits only if
 *       (bytes_before - bytes_after) / bytes_before >= min_gain_ratio;
 *     otherwise the pool is left byte-identical.
 *   - SubtitleTrack entries are re-pointed so canvas_id matches the
 *     new physical index.
 *   - Offline / checkpoint path — not a hot-loop primitive. */

#include <stdint.h>
#include "spatial_subtitle.h"  /* SpatialCanvasPool */
#include "spatial_canvas.h"    /* SpatialCanvas */

typedef struct {
    uint32_t canvases_reordered;  /* number of canvases that moved */
    uint32_t bytes_before;        /* sum of adjacent-pair RLE bytes before */
    uint32_t bytes_after;         /* same metric after the candidate order */
    float    compression_gain;    /* (before - after) / before; 0 when before == 0 */
    uint32_t groups_examined;     /* distinct topic_hash groups with >= 2 canvases */
    int      committed;           /* 1 if the new order was applied, 0 if rolled back */
} ReclusterReport;

/* Topic-grouped greedy re-ordering. min_gain_ratio is the minimum
 * relative RLE byte reduction (e.g. 0.10 for 10%); if the candidate
 * order falls short, pool layout is unchanged and committed = 0. */
ReclusterReport pool_recluster_by_topic(SpatialCanvasPool* p,
                                        float min_gain_ratio);

/* Visit every occupied slot in ascending sequence_id order,
 * regardless of current physical canvas/slot layout. `user` is an
 * opaque pointer forwarded to the callback. Safe on NULL pool (no-op). */
void pool_iterate_in_sequence_order(
    const SpatialCanvasPool* p,
    void (*visit)(const SpatialCanvas* c, uint32_t slot, void* user),
    void* user);

#endif /* SPATIAL_RECLUSTER_H */
