#include "spatial_match.h"
#include "spatial_keyframe.h"   /* full SpatialAI definition for cascade */
#include <stdlib.h>
#include <string.h>

/* ── Directional RGB update (§9.2) ── */

void update_rgb_directional(SpatialGrid* grid) {
    if (!grid) return;

    uint8_t oldR[GRID_TOTAL], oldG[GRID_TOTAL], oldB[GRID_TOTAL];
    uint8_t newR[GRID_TOTAL], newG[GRID_TOTAL], newB[GRID_TOTAL];
    memcpy(oldR, grid->R, GRID_TOTAL);
    memcpy(oldG, grid->G, GRID_TOTAL);
    memcpy(oldB, grid->B, GRID_TOTAL);
    memcpy(newR, grid->R, GRID_TOTAL);
    memcpy(newG, grid->G, GRID_TOTAL);
    memcpy(newB, grid->B, GRID_TOTAL);

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            uint32_t idx = (uint32_t)(y * GRID_SIZE + x);
            if (grid->A[idx] == 0) continue;

            /* R: diagonal (morpheme/semantic) */
            int dx[4] = {1, 1, -1, -1};
            int dy[4] = {1, -1, 1, -1};
            int diff_r_sum = 0;
            int r_neighbors = 0;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(ny * GRID_SIZE + nx);
                    if (grid->A[nidx] > 0) {
                        diff_r_sum += (int)oldR[nidx] - (int)oldR[idx];
                        r_neighbors++;
                    }
                }
            }
            if (r_neighbors > 0) {
                float d = ALPHA_R * ((float)diff_r_sum / (float)r_neighbors);
                int delta = (int)lrintf(d);
                if (delta == 0 && diff_r_sum != 0) {
                    delta = (diff_r_sum > 0) ? 1 : -1;
                }
                int new_val = (int)oldR[idx] + delta;
                if (new_val < 0) new_val = 0;
                if (new_val > 255) new_val = 255;
                newR[idx] = (uint8_t)new_val;
            }

            /* G: vertical (word substitution) */
            int diff_g_sum = 0;
            int g_neighbors = 0;
            for (int d = -1; d <= 1; d += 2) {
                int ny = y + d;
                if (ny >= 0 && ny < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(ny * GRID_SIZE + x);
                    if (grid->A[nidx] > 0) {
                        diff_g_sum += (int)oldG[nidx] - (int)oldG[idx];
                        g_neighbors++;
                    }
                }
            }
            if (g_neighbors > 0) {
                float d = BETA_G * ((float)diff_g_sum / (float)g_neighbors);
                int delta = (int)lrintf(d);
                if (delta == 0 && diff_g_sum != 0) {
                    delta = (diff_g_sum > 0) ? 1 : -1;
                }
                int new_val = (int)oldG[idx] + delta;
                if (new_val < 0) new_val = 0;
                if (new_val > 255) new_val = 255;
                newG[idx] = (uint8_t)new_val;
            }

            /* B: horizontal (clause order) */
            int diff_b_sum = 0;
            int b_neighbors = 0;
            for (int d = -1; d <= 1; d += 2) {
                int nx = x + d;
                if (nx >= 0 && nx < GRID_SIZE) {
                    uint32_t nidx = (uint32_t)(y * GRID_SIZE + nx);
                    if (grid->A[nidx] > 0) {
                        diff_b_sum += (int)oldB[nidx] - (int)oldB[idx];
                        b_neighbors++;
                    }
                }
            }
            if (b_neighbors > 0) {
                float d = GAMMA_B * ((float)diff_b_sum / (float)b_neighbors);
                int delta = (int)lrintf(d);
                if (delta == 0 && diff_b_sum != 0) {
                    delta = (diff_b_sum > 0) ? 1 : -1;
                }
                int new_val = (int)oldB[idx] + delta;
                if (new_val < 0) new_val = 0;
                if (new_val > 255) new_val = 255;
                newB[idx] = (uint8_t)new_val;
            }
        }
    }

    memcpy(grid->R, newR, GRID_TOTAL);
    memcpy(grid->G, newG, GRID_TOTAL);
    memcpy(grid->B, newB, GRID_TOTAL);
}

/* ── Overlap score (Coarse filter §9.3) ── */

uint32_t overlap_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] > 0 && b->A[i] > 0) count++;
    }
    return count;
}

/* ── RGB weight (§9.4) ── */

float rgb_weight(uint8_t r1, uint8_t r2,
                 uint8_t g1, uint8_t g2,
                 uint8_t b1, uint8_t b2) {
    float dr = fabsf((float)r1 - (float)r2) / 255.0f;
    float dg = fabsf((float)g1 - (float)g2) / 255.0f;
    float db = fabsf((float)b1 - (float)b2) / 255.0f;
    return 1.0f - (0.5f * dr + 0.3f * dg + 0.2f * db);
}

/* ── A-channel only cosine ── */

float cosine_a_only(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        double va = (double)a->A[i];
        double vb = (double)b->A[i];
        dot    += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── RGB-weighted cosine (§9.4) ── */

float cosine_rgb_weighted(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        double va = (double)a->A[i];
        double vb = (double)b->A[i];
        if (va > 0.0 && vb > 0.0) {
            float w = rgb_weight(a->R[i], b->R[i],
                                 a->G[i], b->G[i],
                                 a->B[i], b->B[i]);
            dot += va * vb * (double)w;
        }
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── Block summary (SPEC-ENGINE Phase B) ── */

void compute_block_sums(const SpatialGrid* g, BlockSummary* bs) {
    if (!g || !bs) return;

    for (int by = 0; by < BLOCKS; by++) {
        for (int bx = 0; bx < BLOCKS; bx++) {
            uint32_t s = 0;
            for (int y = 0; y < BLOCK; y++) {
                for (int x = 0; x < BLOCK; x++) {
                    s += g->A[(by * BLOCK + y) * GRID_SIZE + (bx * BLOCK + x)];
                }
            }
            bs->sum[by][bx] = s;
        }
    }
}

/* ── Block-skip cosine (SPEC-ENGINE Phase B.3) ── */

float cosine_block_skip(const SpatialGrid* a, const SpatialGrid* b,
                        const BlockSummary* bs_a, const BlockSummary* bs_b) {
    if (!a || !b || !bs_a || !bs_b) return 0.0f;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

    for (int by = 0; by < BLOCKS; by++) {
        for (int bx = 0; bx < BLOCKS; bx++) {
            /* Both blocks contribute to norms even if one is zero */
            int y_start = by * BLOCK;
            int x_start = bx * BLOCK;

            if (bs_a->sum[by][bx] == 0 && bs_b->sum[by][bx] == 0) {
                /* Both zero: no contribution to anything */
                continue;
            }

            for (int y = 0; y < BLOCK; y++) {
                for (int x = 0; x < BLOCK; x++) {
                    uint32_t idx = (uint32_t)((y_start + y) * GRID_SIZE + x_start + x);
                    double va = (double)a->A[idx];
                    double vb = (double)b->A[idx];
                    norm_a += va * va;
                    norm_b += vb * vb;

                    if (bs_a->sum[by][bx] == 0 || bs_b->sum[by][bx] == 0) {
                        /* One side zero: dot product contribution is 0 */
                        continue;
                    }
                    dot += va * vb;
                }
            }
        }
    }

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── Top-K selection (partial sort) ── */

void topk_select(Candidate* pool, uint32_t pool_size, uint32_t k) {
    if (!pool || pool_size <= k) return;

    /* Simple selection sort for top-k */
    for (uint32_t i = 0; i < k && i < pool_size; i++) {
        uint32_t max_idx = i;
        for (uint32_t j = i + 1; j < pool_size; j++) {
            if (pool[j].score > pool[max_idx].score) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            Candidate tmp = pool[i];
            pool[i] = pool[max_idx];
            pool[max_idx] = tmp;
        }
    }
}

/* ── Hash bucket (SPEC-ENGINE Phase C) ── */

uint32_t grid_hash(const SpatialGrid* g) {
    if (!g) return 0;
    uint32_t h = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (g->A[i] > 0) {
            uint32_t x = i % GRID_SIZE;
            h = h * 31 + x;
        }
    }
    return h % NUM_BUCKETS;
}

void bucket_index_init(BucketIndex* idx) {
    if (!idx) return;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        idx->buckets[i].ids      = NULL;
        idx->buckets[i].count    = 0;
        idx->buckets[i].capacity = 0;
    }
}

void bucket_index_destroy(BucketIndex* idx) {
    if (!idx) return;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        free(idx->buckets[i].ids);
        idx->buckets[i].ids      = NULL;
        idx->buckets[i].count    = 0;
        idx->buckets[i].capacity = 0;
    }
}

void bucket_index_add(BucketIndex* idx, const SpatialGrid* g, uint32_t kf_id) {
    if (!idx || !g) return;
    uint32_t h = grid_hash(g);
    Bucket* b = &idx->buckets[h];

    if (b->count >= b->capacity) {
        uint32_t new_cap = b->capacity ? b->capacity * 2 : 64;
        uint32_t* tmp = (uint32_t*)realloc(b->ids, new_cap * sizeof(uint32_t));
        if (!tmp) return;  /* allocation failure: silently drop */
        b->ids      = tmp;
        b->capacity = new_cap;
    }
    b->ids[b->count++] = kf_id;
}

void bucket_candidates(BucketIndex* idx, uint32_t hash, int expand,
                       uint32_t* out, uint32_t* out_count,
                       uint32_t max_out) {
    if (!idx || !out || !out_count) return;
    *out_count = 0;

    for (int d = -expand; d <= expand; d++) {
        uint32_t bi = (uint32_t)((int)hash + d + (int)NUM_BUCKETS) % NUM_BUCKETS;
        Bucket* b = &idx->buckets[bi];
        for (uint32_t i = 0; i < b->count; i++) {
            if (*out_count >= max_out) return;
            out[(*out_count)++] = b->ids[i];
        }
    }
}

/* ── Channel-pair scoring primitives ───────────────────────
 *
 * All four helpers return a mean in [0, 1] over the co-active cells
 * (A[i] > 0 on both grids). The previous implementation returned a
 * raw Σ, which scaled with the number of co-active cells and yielded
 * values like 28.0 / 42.0 when fed through cascade modes — breaking
 * threshold comparisons against cosine scores (which are already
 * normalized). ba_score / ra_score additionally switch from a raw
 * min(A_a, A_b) to a min/max ratio so the A-channel contribution is
 * also bounded to [0, 1]. */

float rg_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    double s = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] == 0 || b->A[i] == 0) continue;
        double r_sim = 1.0 - fabs((double)a->R[i] - b->R[i]) / 255.0;
        double g_sim = 1.0 - fabs((double)a->G[i] - b->G[i]) / 255.0;
        if (r_sim < 0) r_sim = 0;
        if (g_sim < 0) g_sim = 0;
        s += r_sim * g_sim;
        n++;
    }
    if (n == 0) return 0.0f;
    return (float)(s / (double)n);
}

float bg_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    double s = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] == 0 || b->A[i] == 0) continue;
        double b_sim = 1.0 - fabs((double)a->B[i] - b->B[i]) / 255.0;
        double g_sim = 1.0 - fabs((double)a->G[i] - b->G[i]) / 255.0;
        if (b_sim < 0) b_sim = 0;
        if (g_sim < 0) g_sim = 0;
        s += b_sim * g_sim;
        n++;
    }
    if (n == 0) return 0.0f;
    return (float)(s / (double)n);
}

float ba_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    double s = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] == 0 || b->A[i] == 0) continue;
        double b_sim = 1.0 - fabs((double)a->B[i] - b->B[i]) / 255.0;
        double a_min = (double)((a->A[i] < b->A[i]) ? a->A[i] : b->A[i]);
        double a_max = (double)((a->A[i] > b->A[i]) ? a->A[i] : b->A[i]);
        double a_sim = (a_max > 0.0) ? (a_min / a_max) : 0.0;
        if (b_sim < 0) b_sim = 0;
        s += b_sim * a_sim;
        n++;
    }
    if (n == 0) return 0.0f;
    return (float)(s / (double)n);
}

float ra_score(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    double s = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a->A[i] == 0 || b->A[i] == 0) continue;
        double r_sim = 1.0 - fabs((double)a->R[i] - b->R[i]) / 255.0;
        double a_min = (double)((a->A[i] < b->A[i]) ? a->A[i] : b->A[i]);
        double a_max = (double)((a->A[i] > b->A[i]) ? a->A[i] : b->A[i]);
        double a_sim = (a_max > 0.0) ? (a_min / a_max) : 0.0;
        if (r_sim < 0) r_sim = 0;
        s += r_sim * a_sim;
        n++;
    }
    if (n == 0) return 0.0f;
    return (float)(s / (double)n);
}

/* ── Unified matching entry point (Mod 1) ───────────────── */

MatchResult spatial_match(SpatialAI* ai,
                          const SpatialGrid* input,
                          MatchMode mode,
                          const MatchContext* ctx) {
    MatchResult result;
    memset(&result, 0, sizeof(result));
    if (!ai || !input || ai->kf_count == 0) return result;

    uint32_t n = ai->kf_count;
    Candidate* pool = (Candidate*)malloc(n * sizeof(Candidate));
    if (!pool) return result;

    /* ── Step 1: coarse candidate pool ──
     * Bucket path activates only when the caller provides an index AND
     * the corpus is large enough (>= BUCKET_THRESHOLD). If the bucket
     * returns fewer than TOP_K candidates, fall back to a full scan. */
    uint32_t pool_size = 0;

    if (ctx && ctx->bucket_idx && n >= BUCKET_THRESHOLD) {
        uint32_t cand_ids[1024];
        uint32_t cand_count = 0;
        uint32_t h = grid_hash(input);
        bucket_candidates(ctx->bucket_idx, h, 5, cand_ids, &cand_count,
                          (uint32_t)(sizeof(cand_ids) / sizeof(cand_ids[0])));

        if (cand_count >= TOP_K) {
            for (uint32_t i = 0; i < cand_count; i++) {
                pool[i].id = cand_ids[i];
                pool[i].score = (float)overlap_score(input,
                                      &ai->keyframes[cand_ids[i]].grid);
            }
            pool_size = cand_count;
        }
    }

    if (pool_size == 0) {
        for (uint32_t i = 0; i < n; i++) {
            pool[i].id = i;
            pool[i].score = (float)overlap_score(input, &ai->keyframes[i].grid);
        }
        pool_size = n;
    }

    uint32_t k = (pool_size < TOP_K) ? pool_size : TOP_K;
    topk_select(pool, pool_size, k);

    /* ── Step 2: precise scoring on the top-K ── */
    for (uint32_t i = 0; i < k; i++) {
        const SpatialGrid* kf = &ai->keyframes[pool[i].id].grid;
        switch (mode) {
            case MATCH_PREDICT:  pool[i].score = cosine_rgb_weighted(input, kf); break;
            case MATCH_SEARCH:   pool[i].score = cosine_a_only(input, kf);       break;
            case MATCH_QA:       pool[i].score = rg_score(input, kf);            break;
            case MATCH_GENERATE: pool[i].score = bg_score(input, kf);            break;
            default:             pool[i].score = cosine_rgb_weighted(input, kf); break;
        }
    }

    /* Sort the rescored top-K descending. topk_select's fast-path returns
     * when pool_size == k, so do it inline — k ≤ TOP_K (8), O(k²) trivial. */
    for (uint32_t i = 0; i < k; i++) {
        uint32_t max_i = i;
        for (uint32_t j = i + 1; j < k; j++) {
            if (pool[j].score > pool[max_i].score) max_i = j;
        }
        if (max_i != i) {
            Candidate tmp = pool[i];
            pool[i]       = pool[max_i];
            pool[max_i]   = tmp;
        }
    }

    /* ── Assemble result ── */
    result.best_id    = pool[0].id;
    result.best_score = pool[0].score;
    result.topk_count = k;
    for (uint32_t i = 0; i < k && i < TOP_K; i++) {
        result.topk[i] = pool[i];
    }

    free(pool);
    return result;
}


/* ── Public: match_cascade (thin wrapper, spec v2) ──
 *
 * The old 3-stage cascade (A-only early return → channel-pair rerank
 * → cross-pair rematch) has been replaced with the unified 2-stage
 * spatial_match core. The CascadeMode values map directly to the new
 * MatchMode enum; callers keep their existing API. */
uint32_t match_cascade(SpatialAI* ai, SpatialGrid* input,
                       CascadeMode mode, float* out_similarity) {
    MatchMode mm;
    switch (mode) {
        case CASCADE_SEARCH:   mm = MATCH_SEARCH;   break;
        case CASCADE_QA:       mm = MATCH_QA;       break;
        case CASCADE_GENERATE: mm = MATCH_GENERATE; break;
        default:               mm = MATCH_PREDICT;  break;
    }
    MatchResult r = spatial_match(ai, input, mm, NULL);
    if (out_similarity) *out_similarity = r.best_score;
    return r.best_id;
}

/* ── Public: match_cascade_topk ────────────────────────── */

/* ── Adaptive channel weights ─────────────────────────── */

void weight_init(ChannelWeight* w) {
    if (!w) return;
    w->w_A = 1.0f;
    w->w_R = 1.0f;
    w->w_G = 1.0f;
    w->w_B = 1.0f;
}

void weight_normalize(ChannelWeight* w) {
    if (!w) return;
    float s = w->w_A + w->w_R + w->w_G + w->w_B;
    if (s <= 0.0f) { weight_init(w); return; }
    float k = 4.0f / s;
    w->w_A *= k;
    w->w_R *= k;
    w->w_G *= k;
    w->w_B *= k;
    /* Guard small drift to keep weights non-negative in [0, 4] */
    if (w->w_A < 0.0f) w->w_A = 0.0f;
    if (w->w_R < 0.0f) w->w_R = 0.0f;
    if (w->w_G < 0.0f) w->w_G = 0.0f;
    if (w->w_B < 0.0f) w->w_B = 0.0f;
}

void weight_update(ChannelWeight* w,
                   float sim_A, float sim_R, float sim_G, float sim_B) {
    if (!w) return;

    /* Winner takes the reward, others get a small decay. */
    float best = sim_A; int idx = 0;
    if (sim_R > best) { best = sim_R; idx = 1; }
    if (sim_G > best) { best = sim_G; idx = 2; }
    if (sim_B > best) { best = sim_B; idx = 3; }

    const float lr       = WEIGHT_LEARNING_RATE;
    const float decay    = lr / 3.0f;

    /* Apply to all four, subtract LR/3 from losers */
    w->w_A -= decay;
    w->w_R -= decay;
    w->w_G -= decay;
    w->w_B -= decay;

    /* Add LR to winner (so net effect for winner = +LR + decay extra = +LR - decay?
       simpler: undo subtract for winner and add LR directly.) */
    switch (idx) {
        case 0: w->w_A += (lr + decay); break;
        case 1: w->w_R += (lr + decay); break;
        case 2: w->w_G += (lr + decay); break;
        case 3: w->w_B += (lr + decay); break;
    }
    weight_normalize(w);
}

/* ── Per-channel similarity helpers ───────────────────── */

/* Average (1 - |Δ|/255) over cells where both A's are > 0.
 * Returns 0 if no co-active cells. */
static float avg_rgb_sim(const uint8_t* a_ch, const uint8_t* b_ch,
                          const uint16_t* a_A, const uint16_t* b_A) {
    double sum = 0.0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (a_A[i] == 0 || b_A[i] == 0) continue;
        sum += 1.0 - fabs((double)a_ch[i] - b_ch[i]) / 255.0;
        n++;
    }
    if (n == 0) return 0.0f;
    return (float)(sum / n);
}

float channel_sim_A(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    return cosine_a_only(a, b);
}

float channel_sim_R(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    return avg_rgb_sim(a->R, b->R, a->A, b->A);
}

float channel_sim_G(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    return avg_rgb_sim(a->G, b->G, a->A, b->A);
}

float channel_sim_B(const SpatialGrid* a, const SpatialGrid* b) {
    if (!a || !b) return 0.0f;
    return avg_rgb_sim(a->B, b->B, a->A, b->A);
}

float adaptive_score(const SpatialGrid* a, const SpatialGrid* b,
                     const ChannelWeight* w) {
    if (!a || !b) return 0.0f;
    ChannelWeight def;
    weight_init(&def);
    if (!w) w = &def;

    float sA = channel_sim_A(a, b);
    float sR = channel_sim_R(a, b);
    float sG = channel_sim_G(a, b);
    float sB = channel_sim_B(a, b);

    return (w->w_A * sA + w->w_R * sR + w->w_G * sG + w->w_B * sB) / 4.0f;
}

/* ── Weighted cascade variants ────────────────────────── */

/* Weighted variants: top-K from adaptive_score over the whole engine.
 * Skipping the unified spatial_match cascade here because the adaptive
 * scoring rule combines all four channels uniformly and needs every
 * keyframe scored once — no coarse filter is cheaper than just running
 * the scorer. */
uint32_t match_cascade_weighted(SpatialAI* ai, SpatialGrid* input,
                                CascadeMode mode, const ChannelWeight* w,
                                float* out_similarity) {
    if (!w) return match_cascade(ai, input, mode, out_similarity);
    if (!ai || !input || ai->kf_count == 0) {
        if (out_similarity) *out_similarity = 0.0f;
        return 0;
    }
    (void)mode; /* adaptive score is mode-agnostic */

    uint32_t n = ai->kf_count;
    Candidate* pool = (Candidate*)malloc(n * sizeof(Candidate));
    if (!pool) {
        if (out_similarity) *out_similarity = 0.0f;
        return 0;
    }
    for (uint32_t i = 0; i < n; i++) {
        pool[i].id    = i;
        pool[i].score = adaptive_score(input, &ai->keyframes[i].grid, w);
    }
    uint32_t k = (TOP_K < n) ? TOP_K : n;
    topk_select(pool, n, k);

    uint32_t final_id = pool[0].id;
    float    final_score = pool[0].score;
    free(pool);

    if (out_similarity) *out_similarity = final_score;
    return final_id;
}

uint32_t match_cascade_topk_weighted(SpatialAI* ai, SpatialGrid* input,
                                     CascadeMode mode, const ChannelWeight* w,
                                     uint32_t k, uint32_t* out_ids,
                                     float* out_scores) {
    if (!w) return match_cascade_topk(ai, input, mode, k, out_ids, out_scores);
    if (!ai || !input || !out_ids || !out_scores || ai->kf_count == 0 || k == 0) return 0;
    (void)mode;

    uint32_t n = ai->kf_count;
    if (k > n) k = n;
    Candidate* pool = (Candidate*)malloc(n * sizeof(Candidate));
    if (!pool) return 0;
    for (uint32_t i = 0; i < n; i++) {
        pool[i].id    = i;
        pool[i].score = adaptive_score(input, &ai->keyframes[i].grid, w);
    }
    topk_select(pool, n, k);
    for (uint32_t i = 0; i < k; i++) {
        out_ids[i]    = pool[i].id;
        out_scores[i] = pool[i].score;
    }
    free(pool);
    return k;
}

/* match_cascade_topk becomes a thin spatial_match wrapper (spec v2). */
uint32_t match_cascade_topk(SpatialAI* ai, SpatialGrid* input,
                            CascadeMode mode, uint32_t k,
                            uint32_t* out_ids, float* out_scores) {
    if (!ai || !input || !out_ids || !out_scores || ai->kf_count == 0 || k == 0) {
        return 0;
    }

    MatchMode mm;
    switch (mode) {
        case CASCADE_SEARCH:   mm = MATCH_SEARCH;   break;
        case CASCADE_QA:       mm = MATCH_QA;       break;
        case CASCADE_GENERATE: mm = MATCH_GENERATE; break;
        default:               mm = MATCH_PREDICT;  break;
    }
    MatchResult r = spatial_match(ai, input, mm, NULL);
    uint32_t n = r.topk_count < k ? r.topk_count : k;
    for (uint32_t i = 0; i < n; i++) {
        out_ids[i]    = r.topk[i].id;
        out_scores[i] = r.topk[i].score;
    }
    return n;
}

