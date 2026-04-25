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

/* --- Memo ----------------------------------------------------------- */

void ce_memo_apply(CELatentGrid *z, const CEMemoLayer *memo) {
    if (!memo || !memo->adjustments || memo->count == 0) return;
    for (int i = 0; i < CE_GRID_N; ++i) {
        const CEUnit *adj = &memo->adjustments[i % memo->count];
        CEUnit out;
        ce_apply(&out, &z->cells[i], adj);
        z->cells[i] = out;
    }
}

void ce_memo_remove(CELatentGrid *z, const CEMemoLayer *memo) {
    if (!memo || !memo->adjustments || memo->count == 0) return;
    for (int i = 0; i < CE_GRID_N; ++i) {
        const CEUnit *adj = &memo->adjustments[i % memo->count];
        /* Subtract: out = z - adj  ==  z + (-adj). We compute via ce_delta. */
        CEUnit zero; ce_init(&zero);
        CEUnit neg;
        ce_delta(&neg, adj, &zero); /* neg = 0 - adj = -adj */
        CEUnit out;
        ce_apply(&out, &z->cells[i], &neg);
        z->cells[i] = out;
    }
}

/* --- Hint ----------------------------------------------------------- */

void ce_hint_apply(CELatentGrid *z, const CEHintLayer *hint) {
    if (!hint || hint->strength == 0.0f) return;
    for (int i = 0; i < CE_GRID_N; ++i) {
        CEUnit d, ds, out;
        ce_delta(&d, &z->cells[i], &hint->cells[i]);
        ce_delta_scale(&ds, &d, hint->strength);
        ce_apply(&out, &z->cells[i], &ds);
        z->cells[i] = out;
    }
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

    ce_denoise_loop(z, storage, prompt, prompt_count, NULL, config);

    /* Restore anchors (mask byte 0 means "keep original"). */
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (mask->mask[i] == 0) z->cells[i] = anchor_snap[i];
    }
}

/* --- Upscale -------------------------------------------------------- */

void ce_upscale(CEHiresGrid *hi_res,
                const CELatentGrid *lo_res,
                const CEStorage *storage,
                const CEGenConfig *config) {
    hi_res->width = CE_HIRES_GRID_W;
    hi_res->height = CE_HIRES_GRID_H;
    int factor = CE_HIRES_GRID_W / CE_GRID_W; /* = 4 */

    /* Step 1: nearest-neighbour seed from lo_res. */
    for (int hy = 0; hy < CE_HIRES_GRID_H; ++hy) {
        int ly = hy / factor;
        for (int hx = 0; hx < CE_HIRES_GRID_W; ++hx) {
            int lx = hx / factor;
            hi_res->cells[hy * CE_HIRES_GRID_W + hx] =
                lo_res->cells[ly * CE_GRID_W + lx];
        }
    }
    /* Step 2: per-subcell perturbation using a deterministic seed derived
     * from the (hx, hy) location and the cell's own bytes. */
    for (int hy = 0; hy < CE_HIRES_GRID_H; ++hy) {
        for (int hx = 0; hx < CE_HIRES_GRID_W; ++hx) {
            int sub_x = hx % factor;
            int sub_y = hy % factor;
            if (sub_x == 0 && sub_y == 0) continue; /* leave centre untouched */
            int hi_idx = hy * CE_HIRES_GRID_W + hx;
            uint8_t b[8];
            uint64_t mix = ((uint64_t)hx * 0x9E3779B97F4A7C15ULL)
                         ^ ((uint64_t)hy * 0xBF58476D1CE4E5B9ULL)
                         ^ ((uint64_t)ce_read(&hi_res->cells[hi_idx], CE_CH_R) & 0xFFFFu);
            for (int k = 0; k < 8; ++k) b[k] = (uint8_t)(mix >> (k * 8));
            CEUnit perturb; ce_init(&perturb);
            ce_feed(&perturb, b, 8);
            CEUnit d, ds, out;
            ce_delta(&d, &hi_res->cells[hi_idx], &perturb);
            ce_delta_scale(&ds, &d, 0.15f);
            ce_apply(&out, &hi_res->cells[hi_idx], &ds);
            hi_res->cells[hi_idx] = out;
        }
    }

    /* Step 3: storage-guided refinement — pull each hi-res cell toward
     * its nearest stored keyframe. */
    if (storage && storage->count > 0 && config) {
        float w = 0.10f * config->cfg_scale;
        for (int i = 0; i < CE_HIRES_N; ++i) {
            CESearchResult r;
            ce_search_topk(storage, &hi_res->cells[i], 1, &r);
            const CEStorageEntry *e = &storage->entries[r.entry_idx];
            CEUnit d, ds, out;
            ce_delta(&d, &hi_res->cells[i], &e->keyframe);
            ce_delta_scale(&ds, &d, w);
            ce_apply(&out, &hi_res->cells[i], &ds);
            hi_res->cells[i] = out;
        }
    }
}

/* --- Audio ---------------------------------------------------------- */

void ce_audio_load(CEAudioTrack *track,
                   const uint8_t *audio_data, uint32_t len,
                   int segments_per_step) {
    if (segments_per_step <= 0) segments_per_step = 1;
    /* For simplicity, derive a fixed number of segments proportional to
     * input length. If len is small we still produce >= 1. */
    uint32_t n = len / 64; if (n == 0) n = 1; if (n > 4096) n = 4096;
    track->count = n;
    track->segments = (CEUnit *)calloc(n, sizeof(CEUnit));
    track->energy   = (float  *)calloc(n, sizeof(float));
    if (!track->segments || !track->energy) {
        track->count = 0;
        free(track->segments); track->segments = NULL;
        free(track->energy);   track->energy = NULL;
        return;
    }
    uint32_t chunk = (len + n - 1) / n;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t off = i * chunk;
        uint32_t c   = (off + chunk <= len) ? chunk : (len - off);
        ce_init(&track->segments[i]);
        if (c > 0 && audio_data) ce_feed(&track->segments[i], audio_data + off, c);
        /* Energy = normalised sum of sample magnitudes. */
        uint64_t sum = 0;
        for (uint32_t k = 0; k < c; ++k) sum += (uint8_t)audio_data[off + k];
        track->energy[i] = (c > 0) ? (float)sum / (float)c / 128.0f : 1.0f;
        if (track->energy[i] < 0.5f) track->energy[i] = 0.5f;
        if (track->energy[i] > 2.0f) track->energy[i] = 2.0f;
    }
    (void)segments_per_step; /* reserved for future fine-grain mapping */
}

void ce_audio_free(CEAudioTrack *track) {
    free(track->segments); track->segments = NULL;
    free(track->energy);   track->energy = NULL;
    track->count = 0;
}

float ce_audio_energy_at(const CEAudioTrack *track, int step, int total_steps) {
    if (!track || track->count == 0 || total_steps <= 0) return 1.0f;
    if (step < 0) step = 0;
    if (step >= total_steps) step = total_steps - 1;
    /* Map [0, total_steps) -> [0, count). */
    uint32_t idx = (uint32_t)((int64_t)step * (int64_t)track->count / total_steps);
    if (idx >= track->count) idx = track->count - 1;
    return track->energy[idx];
}
