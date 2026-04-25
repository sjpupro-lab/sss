/* ce_engine.h — UNet-equivalent operations on a 32x32 CE latent grid.
 *
 * Steps 6-13:
 *   ce_latent_init       : combine noise + keyframe + delta + prompt + audio
 *   ce_down              : pull each cell toward the keyframe (one step)
 *   ce_conv              : weighted-sum spatial neighbour deltas
 *   ce_self_attention    : weighted-sum top-k pattern neighbour deltas
 *   ce_cross_attention   : prompt cells modulated by cfg_scale
 *   ce_residual          : preserve information from before to after
 *   ce_skip_connect      : feed Down's snapshot back to Up
 *   ce_up                : decode-side detail restoration
 *
 * NEVER averages bytes — every operation is a weighted ce_apply chain.
 */
#ifndef CE_ENGINE_H
#define CE_ENGINE_H

#include "ce_core.h"
#include "ce_search.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CEUnit cells[CE_GRID_N];
    uint16_t width;
    uint16_t height;
    int      current_step;
    int      total_steps;
} CELatentGrid;

typedef struct {
    uint16_t cell_idx;
    uint32_t distance;
} CEAttentionLink;

/* Step 6 — initial latent built from up to 5 sources.
 * `weights[0..4]` correspond to (noise, keyframe, delta, prompt, audio).
 * `audio` may be NULL (its weight is then ignored). */
void ce_latent_init(CELatentGrid *z,
                    const CEUnit *noise,
                    const CEUnit *keyframe,
                    const CEUnit *delta,
                    const CEUnit *prompt,
                    const CEUnit *audio,
                    const float weights[5]);

/* Step 7 — Down. Each cell takes one step toward `pairs[*].keyframe_latent`.
 * `skip_buffer` (CE_GRID_N entries) receives the pre-Down snapshot. */
void ce_down(CELatentGrid *z,
             const CERetrievedPair *pairs, int pair_count,
             CEUnit *skip_buffer);

/* Step 8 — Convolution. Each cell pulls toward weighted near neighbours
 * (radius 1 = 3x3, 2 = 5x5). `contexts` may be NULL — if non-NULL it
 * supplies per-cell hint neighbours that augment the in-grid neighbours. */
void ce_conv(CELatentGrid *z,
             const CECellContext *contexts,
             int radius);

/* Step 9 — Self attention. Each cell pulls toward `topk_per_cell` distant
 * pattern matches found anywhere in the grid (excluding the immediate
 * 3x3 neighbourhood handled by conv). `link_buf` must have size
 * >= topk_per_cell (reused across cells). */
void ce_self_attention(CELatentGrid *z,
                       int topk_per_cell,
                       CEAttentionLink *link_buf);

/* Step 10 — Cross attention. Each cell is pulled toward the closest
 * prompt cell, scaled by cfg_scale and inverse distance. */
void ce_cross_attention(CELatentGrid *z,
                        const CEUnit *prompt_cells, int prompt_count,
                        float cfg_scale);

/* Step 11 — Residual. out = before + (after - before) * residual_weight. */
void ce_residual(CEUnit *out,
                 const CEUnit *before, const CEUnit *after,
                 float residual_weight);

/* Step 12 — Skip connection. Apply scaled delta(skip -> z) onto z. */
void ce_skip_connect(CELatentGrid *z,
                     const CEUnit *skip_buffer,
                     float skip_weight);

/* Step 13 — Up. Combine z, skip, and retrieved pairs to restore detail. */
void ce_up(CELatentGrid *z,
           const CEUnit *skip_buffer,
           const CERetrievedPair *pairs, int pair_count);

#ifdef __cplusplus
}
#endif
#endif /* CE_ENGINE_H */
