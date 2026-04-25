#include "ce_engine.h"
#include <string.h>
#include <stdlib.h>

/* Apply a delta(anchor -> target) scaled by `w` onto `z`. Net effect:
 *   z := z + w * (target - anchor)   (mod 256, byte-wise) */
static void apply_pull(CEUnit *z, const CEUnit *anchor, const CEUnit *target, float w) {
    if (w == 0.0f) return;
    CEUnit d, ds;
    ce_delta(&d, anchor, target);
    ce_delta_scale(&ds, &d, w);
    CEUnit out;
    ce_apply(&out, z, &ds);
    *z = out;
}

void ce_latent_init(CELatentGrid *z,
                    const CEUnit *noise,
                    const CEUnit *keyframe,
                    const CEUnit *delta,
                    const CEUnit *prompt,
                    const CEUnit *audio,
                    const float weights[5]) {
    z->width = CE_GRID_W;
    z->height = CE_GRID_H;
    z->current_step = 0;
    z->total_steps = 0;

    /* Use the noise unit as the per-cell base; for inter-cell variety we
     * fold the cell index into the byte stream. */
    for (int i = 0; i < CE_GRID_N; ++i) {
        CEUnit base;
        ce_init(&base);
        if (noise) {
            apply_pull(&base, &base, noise, weights[0]);
            /* per-cell perturbation: XOR cell index into one byte position */
            uint8_t *b = ce_bytes(&base);
            b[i & 63] ^= (uint8_t)(i + 1);
            b[(i + 17) & 63] ^= (uint8_t)((i >> 5) + 1);
        }
        if (keyframe) apply_pull(&base, ce_cbytes(keyframe) ? &base : &base, keyframe, weights[1]);
        if (delta) {
            CEUnit out;
            CEUnit ds;
            ce_delta_scale(&ds, delta, weights[2]);
            ce_apply(&out, &base, &ds);
            base = out;
        }
        if (prompt) apply_pull(&base, &base, prompt, weights[3]);
        if (audio)  apply_pull(&base, &base, audio,  weights[4]);
        z->cells[i] = base;
    }
}

void ce_down(CELatentGrid *z,
             const CERetrievedPair *pairs, int pair_count,
             CEUnit *skip_buffer) {
    int total = z->total_steps > 0 ? z->total_steps : 1;
    int step = z->current_step;
    if (step < 0) step = 0;
    /* Larger pull early, smaller late. step+1 keeps weight in [1/total, 1]. */
    float w = 1.0f / (float)(total - step);
    if (w > 1.0f) w = 1.0f;
    if (w < 0.0f) w = 0.0f;

    for (int i = 0; i < CE_GRID_N; ++i) {
        if (skip_buffer) skip_buffer[i] = z->cells[i];
        if (pair_count <= 0) continue;

        /* Pick a deterministic keyframe target for this cell. */
        const CERetrievedPair *p = &pairs[i % pair_count];
        apply_pull(&z->cells[i], &z->cells[i], &p->keyframe_latent, w * p->relevance);
    }
}

static int idx_at(int x, int y) {
    if (x < 0 || y < 0 || x >= CE_GRID_W || y >= CE_GRID_H) return -1;
    return y * CE_GRID_W + x;
}

void ce_conv(CELatentGrid *z,
             const CECellContext *contexts,
             int radius) {
    if (radius < 1) radius = 1;
    if (radius > 3) radius = 3;

    /* Snapshot so all cells see the same input. */
    static CEUnit snap[CE_GRID_N];
    memcpy(snap, z->cells, sizeof(snap));

    for (int y = 0; y < CE_GRID_H; ++y) {
        for (int x = 0; x < CE_GRID_W; ++x) {
            int ci = y * CE_GRID_W + x;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ni = idx_at(x + dx, y + dy);
                    if (ni < 0) continue;
                    uint32_t d = ce_distance(&snap[ci], &snap[ni]);
                    /* inverse-distance weight; clamp range so it stays useful */
                    float w = 0.25f / (1.0f + (float)d * 0.01f);
                    apply_pull(&z->cells[ci], &snap[ci], &snap[ni], w);
                }
            }
            /* Optional context neighbours (from storage). */
            if (contexts) {
                const CECellContext *ctx = &contexts[ci];
                for (int n = 0; n < ctx->neighbor_count; ++n) {
                    uint32_t d = ce_distance(&snap[ci], &ctx->neighbors[n]);
                    float w = 0.10f / (1.0f + (float)d * 0.01f);
                    apply_pull(&z->cells[ci], &snap[ci], &ctx->neighbors[n], w);
                }
            }
        }
    }
}

void ce_self_attention(CELatentGrid *z,
                       int topk_per_cell,
                       CEAttentionLink *link_buf) {
    if (topk_per_cell <= 0) return;

    static CEUnit snap[CE_GRID_N];
    memcpy(snap, z->cells, sizeof(snap));

    for (int i = 0; i < CE_GRID_N; ++i) {
        /* Build top-k farthest-but-similar list (excluding 3x3 neighbours
         * to avoid double-counting with conv). We use a max-heap of size k
         * keyed on distance to keep the smallest distances. */
        int k = topk_per_cell;
        int filled = 0;
        int xi = i % CE_GRID_W, yi = i / CE_GRID_W;
        for (int j = 0; j < CE_GRID_N; ++j) {
            if (j == i) continue;
            int xj = j % CE_GRID_W, yj = j / CE_GRID_W;
            int dx = xj - xi, dy = yj - yi;
            if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) continue; /* skip immediate 3x3 */
            uint32_t d = ce_distance(&snap[i], &snap[j]);
            if (filled < k) {
                link_buf[filled].cell_idx = (uint16_t)j;
                link_buf[filled].distance = d;
                ++filled;
                /* sift up */
                int p = filled - 1;
                while (p > 0) {
                    int par = (p - 1) / 2;
                    if (link_buf[par].distance < link_buf[p].distance) {
                        CEAttentionLink t = link_buf[par];
                        link_buf[par] = link_buf[p];
                        link_buf[p] = t;
                        p = par;
                    } else break;
                }
            } else if (d < link_buf[0].distance) {
                link_buf[0].cell_idx = (uint16_t)j;
                link_buf[0].distance = d;
                /* sift down */
                int p = 0;
                while (1) {
                    int l = 2 * p + 1, r = 2 * p + 2, m = p;
                    if (l < filled && link_buf[l].distance > link_buf[m].distance) m = l;
                    if (r < filled && link_buf[r].distance > link_buf[m].distance) m = r;
                    if (m == p) break;
                    CEAttentionLink t = link_buf[m];
                    link_buf[m] = link_buf[p];
                    link_buf[p] = t;
                    p = m;
                }
            }
        }
        /* Pull current cell toward the matched pattern cells. */
        for (int n = 0; n < filled; ++n) {
            uint32_t d = link_buf[n].distance;
            float w = 0.05f / (1.0f + (float)d * 0.005f);
            apply_pull(&z->cells[i], &snap[i], &snap[link_buf[n].cell_idx], w);
        }
    }
}

void ce_cross_attention(CELatentGrid *z,
                        const CEUnit *prompt_cells, int prompt_count,
                        float cfg_scale) {
    if (!prompt_cells || prompt_count <= 0 || cfg_scale == 0.0f) return;

    static CEUnit snap[CE_GRID_N];
    memcpy(snap, z->cells, sizeof(snap));

    for (int i = 0; i < CE_GRID_N; ++i) {
        /* Pick the closest prompt cell for this latent cell. */
        int best = 0;
        uint32_t bestd = ce_distance(&snap[i], &prompt_cells[0]);
        for (int p = 1; p < prompt_count; ++p) {
            uint32_t d = ce_distance(&snap[i], &prompt_cells[p]);
            if (d < bestd) { bestd = d; best = p; }
        }
        /* Closer prompt cells get stronger pull (high relevance + cfg). */
        float w = cfg_scale * 0.20f / (1.0f + (float)bestd * 0.005f);
        apply_pull(&z->cells[i], &snap[i], &prompt_cells[best], w);
    }
}

void ce_residual(CEUnit *out,
                 const CEUnit *before, const CEUnit *after,
                 float residual_weight) {
    /* out = before + w * (after - before) */
    CEUnit d, ds;
    ce_delta(&d, before, after);
    ce_delta_scale(&ds, &d, residual_weight);
    ce_apply(out, before, &ds);
}

void ce_skip_connect(CELatentGrid *z,
                     const CEUnit *skip_buffer,
                     float skip_weight) {
    if (!skip_buffer || skip_weight == 0.0f) return;
    for (int i = 0; i < CE_GRID_N; ++i) {
        /* Move z toward skip_buffer by skip_weight. Restores pre-Down structure. */
        apply_pull(&z->cells[i], &z->cells[i], &skip_buffer[i], skip_weight);
    }
}

void ce_up(CELatentGrid *z,
           const CEUnit *skip_buffer,
           const CERetrievedPair *pairs, int pair_count) {
    /* Up restores detail by combining (z, skip, deltas-from-pairs). */
    for (int i = 0; i < CE_GRID_N; ++i) {
        /* light skip blend */
        if (skip_buffer) {
            apply_pull(&z->cells[i], &z->cells[i], &skip_buffer[i], 0.25f);
        }
        if (pair_count > 0) {
            const CERetrievedPair *p = &pairs[i % pair_count];
            /* apply the stored delta (variation) on top of current z */
            CEUnit ds, out;
            ce_delta_scale(&ds, &p->delta_latent, 0.50f * p->relevance);
            ce_apply(&out, &z->cells[i], &ds);
            z->cells[i] = out;
        }
    }
}
