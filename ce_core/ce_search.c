#include "ce_search.h"
#include <stdlib.h>
#include <string.h>

/* SplitMix64 — small, fast, deterministic PRNG. */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void ce_noise_init(CEUnit *out, uint64_t seed) {
    uint8_t buf[64];
    uint64_t state = seed ^ 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < 8; ++i) {
        uint64_t r = splitmix64(&state);
        for (int j = 0; j < 8; ++j) buf[i * 8 + j] = (uint8_t)(r >> (j * 8));
    }
    ce_init(out);
    ce_feed(out, buf, 64);
}

void ce_query_from_noise(CEQueryGrid *grid, uint64_t seed) {
    grid->width = CE_GRID_W;
    grid->height = CE_GRID_H;
    for (int i = 0; i < CE_GRID_N; ++i) {
        /* Per-cell sub-seed: (seed, i) -> distinct stream. */
        uint64_t sub = seed ^ (((uint64_t)i + 1) * 0x9E3779B97F4A7C15ULL);
        ce_noise_init(&grid->cells[i], sub);
    }
}

/* Insertion into a max-heap of size k by distance (so the worst entry sits
 * at index 0 and gets evicted when a better candidate comes). */
static void heap_sift_down(CESearchResult *h, int n, int i) {
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < n && h[l].distance > h[m].distance) m = l;
        if (r < n && h[r].distance > h[m].distance) m = r;
        if (m == i) break;
        CESearchResult t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static int cmp_result(const void *a, const void *b) {
    uint32_t da = ((const CESearchResult *)a)->distance;
    uint32_t db = ((const CESearchResult *)b)->distance;
    if (da < db) return -1;
    if (da > db) return  1;
    /* tie-break on entry_idx for determinism */
    uint32_t ia = ((const CESearchResult *)a)->entry_idx;
    uint32_t ib = ((const CESearchResult *)b)->entry_idx;
    return (ia < ib) ? -1 : (ia > ib);
}

/* Common worker: top-k over storage with an optional modality filter.
 * `match_type < 0` disables the filter (matches all entries). */
static int search_topk_filtered(const CEStorage *storage, const CEUnit *query,
                                int match_type, int k,
                                CESearchResult *results) {
    if (k <= 0 || storage->count == 0) return 0;
    int n = (int)storage->count;

    /* Streaming heap fill: only keep at most k candidates that pass the
     * filter. Without a filter, this is identical to the unfiltered path. */
    int kept = 0;
    for (int i = 0; i < n; ++i) {
        if (match_type >= 0 && (int)storage->entries[i].type != match_type) continue;
        uint32_t d = ce_distance(query, &storage->entries[i].keyframe);
        if (kept < k) {
            results[kept].entry_idx = (uint32_t)i;
            results[kept].distance  = d;
            ++kept;
            if (kept == k) {
                /* Build max-heap once we hit k entries. */
                for (int j = k / 2 - 1; j >= 0; --j) heap_sift_down(results, k, j);
            }
        } else if (d < results[0].distance) {
            results[0].distance  = d;
            results[0].entry_idx = (uint32_t)i;
            heap_sift_down(results, k, 0);
        }
    }
    /* Final sort ascending for caller convenience. */
    qsort(results, kept, sizeof(*results), cmp_result);
    return kept;
}

int ce_search_topk(const CEStorage *storage, const CEUnit *query,
                   int k, CESearchResult *results) {
    return search_topk_filtered(storage, query, /*match_type=*/-1, k, results);
}

int ce_search_by_type(const CEStorage *storage, const CEUnit *query,
                      CEType type, int k, CESearchResult *results) {
    return search_topk_filtered(storage, query, (int)type, k, results);
}

void ce_extract_context(const CEStorage *storage, uint32_t canvas_id,
                        uint16_t slot, uint16_t center_block,
                        uint16_t row_stride,
                        CECellContext *ctx) {
    int32_t ci = ce_storage_find(storage, canvas_id, slot, center_block);
    if (ci < 0) {
        ce_init(&ctx->center);
        ctx->neighbor_count = 0;
        return;
    }
    ctx->center = storage->entries[ci].keyframe;
    ctx->neighbor_count = 0;

    /* 8 candidate offsets in row-major order: N, S, W, E, NW, NE, SW, SE. */
    int dx[8] = { 0,  0, -1, +1, -1, +1, -1, +1 };
    int dy[8] = {-1, +1,  0,  0, -1, -1, +1, +1 };
    int rs = row_stride ? row_stride : 32;
    for (int n = 0; n < 8; ++n) {
        int nb = (int)center_block + dy[n] * rs + dx[n];
        if (nb < 0 || nb > 0xFFFF) continue;
        /* avoid wrap-around at row edges */
        int cx = (int)center_block % rs, nx = nb % rs;
        if (dx[n] != 0 && (nx - cx) != dx[n]) continue;
        int32_t idx = ce_storage_find(storage, canvas_id, slot, (uint16_t)nb);
        if (idx >= 0) {
            ctx->neighbors[ctx->neighbor_count++] = storage->entries[idx].keyframe;
        }
    }
}

void ce_select_kd_pair(const CEStorage *storage,
                       const CESearchResult *results, int k,
                       CERetrievedPair *pairs) {
    /* Higher relevance for smaller distance. Normalise per-query so that
     * the closest result gets relevance 1.0 and far ones tail toward 0. */
    uint32_t best = results[0].distance;
    if (best == 0) best = 1; /* avoid div-by-zero on identity match */
    for (int i = 0; i < k; ++i) {
        uint32_t idx = results[i].entry_idx;
        const CEStorageEntry *e = &storage->entries[idx];
        pairs[i].keyframe_latent = e->keyframe;
        pairs[i].delta_latent    = e->delta;
        float rel = (float)best / (float)(results[i].distance + 1u);
        if (rel > 1.0f) rel = 1.0f;
        pairs[i].relevance = rel;
    }
}
