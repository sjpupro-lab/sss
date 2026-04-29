#include "ce_extend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Sampler -------------------------------------------------------- */

static int idx_min_distance(const CEUnit *target, const CEUnit *cands, int count) {
    int best = 0;
    uint32_t bd = ce_distance(target, &cands[0]);
    for (int i = 1; i < count; ++i) {
        uint32_t d = ce_distance(target, &cands[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

static int idx_max_pairwise_distance(const CEUnit *cands, int count) {
    /* Pick the candidate with the largest minimum distance to any other —
     * i.e., the most "outlier" element, useful for diversity sampling. */
    int best = 0;
    uint32_t best_min = 0;
    for (int i = 0; i < count; ++i) {
        uint32_t mind = (uint32_t)-1;
        for (int j = 0; j < count; ++j) {
            if (i == j) continue;
            uint32_t d = ce_distance(&cands[i], &cands[j]);
            if (d < mind) mind = d;
        }
        if (i == 0 || mind > best_min) { best_min = mind; best = i; }
    }
    return best;
}

static int idx_best_context(const CEUnit *cands, int count, const CECellContext *ctx) {
    /* Score each candidate by how well it fits the local context: the
     * candidate that is closest to the centre AND the average of its
     * neighbours (without averaging values — we sum distances). */
    int best = 0;
    uint32_t best_score = (uint32_t)-1;
    for (int i = 0; i < count; ++i) {
        uint32_t s = ce_distance(&cands[i], &ctx->center);
        for (int n = 0; n < ctx->neighbor_count; ++n) {
            s += ce_distance(&cands[i], &ctx->neighbors[n]) / 2u;
        }
        if (s < best_score) { best_score = s; best = i; }
    }
    return best;
}

void ce_sample(CEUnit *out,
               const CEUnit *candidates, int count,
               enum CESamplerMode mode,
               const CECellContext *ctx) {
    if (count <= 0) { ce_init(out); return; }
    if (count == 1) { *out = candidates[0]; return; }

    int pick = 0;
    switch (mode) {
        case CE_SAMPLE_CONFIDENCE:
            /* Closest candidate to its own neighbour-collapse. Without an
             * external query, we treat candidate[0] as the search anchor. */
            pick = idx_min_distance(&candidates[0], candidates, count);
            break;
        case CE_SAMPLE_DIVERSITY:
            pick = idx_max_pairwise_distance(candidates, count);
            break;
        case CE_SAMPLE_CONTEXT:
            if (ctx) pick = idx_best_context(candidates, count, ctx);
            else     pick = idx_min_distance(&candidates[0], candidates, count);
            break;
        case CE_SAMPLE_ADAPTIVE:
            /* Use diversity early, context mid, confidence late.
             * Without a step indicator we approximate by candidate count. */
            if (count > 8)      pick = idx_max_pairwise_distance(candidates, count);
            else if (ctx)       pick = idx_best_context(candidates, count, ctx);
            else                pick = idx_min_distance(&candidates[0], candidates, count);
            break;
        default:
            pick = 0;
    }
    *out = candidates[pick];
}

/* --- Inpaint -------------------------------------------------------- */

void ce_inpaint(CELatentGrid *z,
                const CEInpaintMask *mask,
                const CEStorage *storage,
                const CEUnit *prompt, int prompt_count,
                const CEGenConfig *config) {
    if (!mask || !storage || !config) return;

    /* Snapshot anchor cells so we can restore them after denoise. */
    static CEUnit anchor_snap[CE_GRID_N];
    for (int i = 0; i < CE_GRID_N; ++i) anchor_snap[i] = z->cells[i];

    ce_denoise_loop(z, storage, prompt, prompt_count, config);

    /* Restore anchors (mask byte 0 means "keep original"). */
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (mask->mask[i] == 0) z->cells[i] = anchor_snap[i];
    }
}
