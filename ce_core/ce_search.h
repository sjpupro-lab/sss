/* ce_search.h — deterministic noise, query grid, top-k retrieval, context.
 *
 * Steps 1-5 of the generation pipeline:
 *   1. ce_noise_init        : seed -> CEUnit (deterministic noise)
 *   2. ce_query_from_noise  : seed -> 32x32 query grid
 *   3. ce_search_topk       : nearest keyframes by ce_distance
 *   4. ce_extract_context   : center cell + spatial neighbours
 *   5. ce_select_kd_pair    : keyframe + delta with relevance
 */
#ifndef CE_SEARCH_H
#define CE_SEARCH_H

#include "ce_core.h"
#include "ce_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_GRID_W 32
#define CE_GRID_H 32
#define CE_GRID_N (CE_GRID_W * CE_GRID_H)  /* 1024 */

typedef struct {
    CEUnit cells[CE_GRID_N];
    uint16_t width;
    uint16_t height;
} CEQueryGrid;

typedef struct {
    uint32_t entry_idx;
    uint32_t distance;
} CESearchResult;

typedef struct {
    CEUnit  center;
    CEUnit  neighbors[8]; /* N, S, E, W, NE, NW, SE, SW (when valid) */
    int     neighbor_count;
} CECellContext;

typedef struct {
    CEUnit keyframe_latent;
    CEUnit delta_latent;
    float  relevance;
} CERetrievedPair;

/* Step 1: deterministic 64-byte noise from seed. */
void ce_noise_init(CEUnit *out, uint64_t seed);

/* Step 2: 32x32 grid of independent noise cells. */
void ce_query_from_noise(CEQueryGrid *grid, uint64_t seed);

/* Step 3: top-k keyframe search (linear scan, sorted by ascending distance).
 * `results` must have room for k entries. Returns the number written. */
int ce_search_topk(const CEStorage *storage, const CEUnit *query,
                   int k, CESearchResult *results);

/* Step 3 (typed variant): top-k keyframe search filtered to entries with
 * the given modality. Same return convention as ce_search_topk. Useful
 * for image generation, which must not retrieve text keyframes. */
int ce_search_by_type(const CEStorage *storage, const CEUnit *query,
                      CEType type, int k, CESearchResult *results);

/* Step 4: extract a cell + its <=8 spatial neighbours from the storage.
 * Neighbours share canvas_id and slot, with block_idx = center +/- {1, W, W+/-1}.
 * Falls back to the center cell when neighbours are missing. */
void ce_extract_context(const CEStorage *storage, uint32_t canvas_id,
                        uint16_t slot, uint16_t center_block,
                        uint16_t row_stride,
                        CECellContext *ctx);

/* Step 5: build keyframe/delta pairs with a relevance score from a top-k
 * search result list. `pairs` must have room for `k`. */
void ce_select_kd_pair(const CEStorage *storage,
                       const CESearchResult *results, int k,
                       CERetrievedPair *pairs);

#ifdef __cplusplus
}
#endif
#endif /* CE_SEARCH_H */
