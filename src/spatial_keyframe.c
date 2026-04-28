#include "spatial_keyframe.h"
#include "spatial_layers.h"
#include "spatial_subtitle.h"   /* SpatialCanvasPool for ai_get_canvas_pool */
#include "slig_signal.h"        /* SLIG v2: slig_decompose for image cells */
#include "slig_codebook.h"      /* SLIG v2.3: codebook lookup */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 64
#define SIMILARITY_THRESHOLD 0.3f
#define EMA_ALPHA            0.1f   /* new value weight per update */
#define EMA_MIN_EVIDENCE     2.0f   /* skip cells with fewer observations */

/* Runtime-overridable store threshold. Set by ai_set_store_threshold().
 * On real wiki clauses the default 0.30 rarely triggers — see the
 * practical test in tools/run_practical_test.sh — so callers (e.g.
 * stream_train --threshold 0.15) can lower it for delta-heavy runs. */
static float g_store_threshold = SIMILARITY_THRESHOLD;

void ai_set_store_threshold(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    g_store_threshold = t;
}

float ai_get_store_threshold(void) { return g_store_threshold; }

/* ── Topic hashing ──
 *
 * Keyframes carry a topic_hash + seq_in_topic. The hash groups
 * clauses that likely belong to the same document so ai_store_auto
 * can skip the O(KF) flat scan and only compare the new clause
 * against same-topic keyframes. When enough same-topic clauses
 * cluster, deltas trigger instead of exploding the keyframe count.
 *
 * Derivation precedence:
 *   1. non-empty label: djb2(label) — explicit document tag
 *   2. else:            djb2(first space-delimited token of clause)
 *   3. else:            0 (legacy sequential behavior)
 *
 * Using only the first token is intentional: wiki abstracts often
 * lead every clause with the article subject ("Anarchism is ...",
 * "Anarchism advocates ..."), so a 1-token fingerprint clusters
 * them cheaply without the cost of a full doc classifier. */
static uint32_t djb2_bytes(const unsigned char* p, size_t n) {
    uint32_t h = 5381u;
    for (size_t i = 0; i < n; i++) h = h * 33u + (uint32_t)p[i];
    return h;
}

static uint32_t topic_hash_from_text(const char* text) {
    if (!text) return 0;
    const unsigned char* p = (const unsigned char*)text;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    const unsigned char* start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
           *p != '.' && *p != ',' && *p != '!' && *p != '?') p++;
    size_t n = (size_t)(p - start);
    if (n == 0) return 0;
    uint32_t h = djb2_bytes(start, n);
    return h ? h : 1;
}

static uint32_t topic_hash_from_label(const char* label) {
    if (!label || !*label) return 0;
    uint32_t h = djb2_bytes((const unsigned char*)label, strlen(label));
    return h ? h : 1;
}

/* Resolve the topic for a store call: label wins if provided, else the
 * clause's first token. Returns 0 only when neither yields any bytes,
 * which we treat as "no topic" (legacy flat-scan fallback). */
uint32_t ai_resolve_topic(const char* clause_text, const char* label) {
    uint32_t h = topic_hash_from_label(label);
    if (h == 0) h = topic_hash_from_text(clause_text);
    return h;
}

static uint32_t resolve_topic(const char* clause_text, const char* label) {
    return ai_resolve_topic(clause_text, label);
}

static uint32_t next_seq_in_topic(const SpatialAI* ai, uint32_t topic) {
    if (topic == 0) return 0;
    uint32_t max_seq = 0;
    int seen = 0;
    for (uint32_t i = 0; i < ai->kf_count; i++) {
        if (ai->keyframes[i].topic_hash == topic) {
            if (!seen || ai->keyframes[i].seq_in_topic > max_seq) {
                max_seq = ai->keyframes[i].seq_in_topic;
            }
            seen = 1;
        }
    }
    return seen ? max_seq + 1 : 1;
}

/* Scan only keyframes carrying `topic` and return the best
 * cosine_a_only match. Returns UINT32_MAX (no candidates) when no
 * keyframe shares this topic. */
static uint32_t topic_bucket_best_match(const SpatialAI* ai,
                                        const SpatialGrid* input,
                                        uint32_t topic,
                                        float* out_sim) {
    float best = -1.0f;
    uint32_t best_id = UINT32_MAX;
    if (topic != 0) {
        for (uint32_t i = 0; i < ai->kf_count; i++) {
            if (ai->keyframes[i].topic_hash != topic) continue;
            float sim = cosine_a_only(input, &ai->keyframes[i].grid);
            if (sim > best) { best = sim; best_id = i; }
        }
    }
    if (out_sim) *out_sim = (best < 0.0f) ? 0.0f : best;
    return best_id;
}

/* ── RGB EMA ──
 *
 * apply_ema_to_grid blends the accumulated EMA means into an input
 * grid's R/G/B wherever the EMA has enough evidence. The intent is to
 * stabilize each (y, x) cell's channel values across a large corpus:
 * individual clauses are noisy, but the mean over thousands of
 * clauses converges to the POS / context prior for that bitmap slot.
 *
 * ema_update walks the active cells of a freshly stored grid and
 * folds their R/G/B into the running means with weight EMA_ALPHA. A
 * companion count table lets apply_ema_to_grid know how many
 * observations back each cell; cells below EMA_MIN_EVIDENCE are left
 * alone. The update is commutative and doesn't depend on the order
 * in which clauses are stored.
 */

void apply_ema_to_grid(const SpatialAI* ai, SpatialGrid* grid) {
    if (!ai || !grid) return;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (grid->A[i] == 0) continue;
        if (ai->ema_count[i] < EMA_MIN_EVIDENCE) continue;

        float er = ai->ema_R[i];
        float eg = ai->ema_G[i];
        float eb = ai->ema_B[i];

        /* Cells that came out of the encoder with no POS seed get
         * overwritten; cells with a seed are blended 50/50 so the
         * local signal still matters. */
        if (grid->R[i] == 0 && grid->G[i] == 0 && grid->B[i] == 0) {
            grid->R[i] = (uint8_t)(er > 255.0f ? 255 : (er < 0.0f ? 0 : er));
            grid->G[i] = (uint8_t)(eg > 255.0f ? 255 : (eg < 0.0f ? 0 : eg));
            grid->B[i] = (uint8_t)(eb > 255.0f ? 255 : (eb < 0.0f ? 0 : eb));
        } else {
            float r = 0.5f * (float)grid->R[i] + 0.5f * er;
            float g = 0.5f * (float)grid->G[i] + 0.5f * eg;
            float b = 0.5f * (float)grid->B[i] + 0.5f * eb;
            if (r > 255.0f) r = 255.0f;
            if (r < 0.0f)   r = 0.0f;
            if (g > 255.0f) g = 255.0f;
            if (g < 0.0f)   g = 0.0f;
            if (b > 255.0f) b = 255.0f;
            if (b < 0.0f)   b = 0.0f;
            grid->R[i] = (uint8_t)r;
            grid->G[i] = (uint8_t)g;
            grid->B[i] = (uint8_t)b;
        }
    }
}

void ema_update(SpatialAI* ai, const SpatialGrid* grid) {
    if (!ai || !grid) return;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (grid->A[i] == 0) continue;
        float r = (float)grid->R[i];
        float g = (float)grid->G[i];
        float b = (float)grid->B[i];

        if (ai->ema_count[i] == 0.0f) {
            ai->ema_R[i] = r;
            ai->ema_G[i] = g;
            ai->ema_B[i] = b;
        } else {
            ai->ema_R[i] = (1.0f - EMA_ALPHA) * ai->ema_R[i] + EMA_ALPHA * r;
            ai->ema_G[i] = (1.0f - EMA_ALPHA) * ai->ema_G[i] + EMA_ALPHA * g;
            ai->ema_B[i] = (1.0f - EMA_ALPHA) * ai->ema_B[i] + EMA_ALPHA * b;
        }
        ai->ema_count[i] += 1.0f;
    }
}

SpatialAI* spatial_ai_create(void) {
    SpatialAI* ai = (SpatialAI*)malloc(sizeof(SpatialAI));
    if (!ai) return NULL;

    ai->kf_count = 0;
    ai->kf_capacity = INITIAL_CAPACITY;
    ai->keyframes = (Keyframe*)calloc(INITIAL_CAPACITY, sizeof(Keyframe));

    ai->df_count = 0;
    ai->df_capacity = INITIAL_CAPACITY;
    ai->deltas = (DeltaFrame*)calloc(INITIAL_CAPACITY, sizeof(DeltaFrame));

    /* Adaptive channel weights start uniform */
    weight_init(&ai->global_weights);

    /* Canvas pool is lazily created on first ai_get_canvas_pool() call */
    ai->canvas_pool = NULL;

    /* v4 Task A — separate working-memory pool, lazily created */
    ai->context_pool = NULL;

    /* Hash bucket index for large-corpus retrieval; populated as
     * keyframes are added. Rebuilt in ai_load after reading KFs. */
    bucket_index_init(&ai->bucket_idx);

    /* Morpheme dictionary is embedded, but we still pay the per-engine
     * init cost up-front so hot paths can skip it. */
    morpheme_init();
    ai->morpheme_ready = 1;

    /* EMA tables start at zero (no evidence yet). */
    memset(ai->ema_R,     0, sizeof(ai->ema_R));
    memset(ai->ema_G,     0, sizeof(ai->ema_G));
    memset(ai->ema_B,     0, sizeof(ai->ema_B));
    memset(ai->ema_count, 0, sizeof(ai->ema_count));

    /* SLIG v2.3: empty codebook. Populated lazily as ai_store_grid
     * decomposes images and registers their pattern sets. */
    slig_codebook_init(&ai->codebook);

    if (!ai->keyframes || !ai->deltas) {
        spatial_ai_destroy(ai);
        return NULL;
    }

    /* Initialize keyframe grids */
    for (uint32_t i = 0; i < ai->kf_capacity; i++) {
        ai->keyframes[i].grid.A = NULL;
    }

    return ai;
}

void spatial_ai_destroy(SpatialAI* ai) {
    if (!ai) return;

    if (ai->keyframes) {
        for (uint32_t i = 0; i < ai->kf_count; i++) {
            SpatialGrid* g = &ai->keyframes[i].grid;
            if (g->A) { free(g->A); g->A = NULL; }
            if (g->R) { free(g->R); g->R = NULL; }
            if (g->G) { free(g->G); g->G = NULL; }
            if (g->B) { free(g->B); g->B = NULL; }
        }
        free(ai->keyframes);
    }

    if (ai->deltas) {
        for (uint32_t i = 0; i < ai->df_count; i++) {
            free(ai->deltas[i].entries);
        }
        free(ai->deltas);
    }

    if (ai->canvas_pool) {
        pool_destroy(ai->canvas_pool);
        ai->canvas_pool = NULL;
    }

    if (ai->context_pool) {
        pool_destroy(ai->context_pool);
        ai->context_pool = NULL;
    }

    bucket_index_destroy(&ai->bucket_idx);

    free(ai);
}

/* ── Lazy canvas-pool accessors ── */

SpatialCanvasPool* ai_get_canvas_pool(SpatialAI* ai) {
    if (!ai) return NULL;
    if (!ai->canvas_pool) ai->canvas_pool = pool_create();
    return ai->canvas_pool;
}

void ai_release_canvas_pool(SpatialAI* ai) {
    if (!ai || !ai->canvas_pool) return;
    pool_destroy(ai->canvas_pool);
    ai->canvas_pool = NULL;
}

/* ── v4 Task A: ContextPool accessors ──
 *
 * Working-memory pool separate from both the long-term keyframes and
 * the (corpus-oriented) canvas_pool. Intended to be populated with
 * session-scoped clauses — conversation history, recently-browsed
 * prompts, scratch notes — and read back through
 * agg_build_from_pool() during retrieval/refinement so the generator
 * can bias toward the current context without mutating anything in
 * ai->keyframes / ai->deltas / ai->ema_*.
 *
 * Lifecycle:
 *   ai_get_context_pool      lazy-create. Idempotent. Never NULL on
 *                            a live SpatialAI* unless allocation fails.
 *   ai_clear_context_pool    drop contents but keep a fresh empty pool.
 *                            SpatialCanvasPool has no public clear
 *                            primitive, so we destroy-and-recreate.
 *                            Safe to call on a NULL-pool ai (no-op
 *                            beyond lazy create).
 *   ai_release_context_pool  destroy pool and NULL the pointer. After
 *                            this a subsequent ai_get_context_pool
 *                            call lazy-creates again.
 *
 * .spai serialization: by default the context_pool is NOT written.
 * See the roadmap item ai_save_with_context(ai, path) which would add
 * a new SPAI_TAG_* tag without reordering existing ones. */
SpatialCanvasPool* ai_get_context_pool(SpatialAI* ai) {
    if (!ai) return NULL;
    if (!ai->context_pool) ai->context_pool = pool_create();
    return ai->context_pool;
}

void ai_clear_context_pool(SpatialAI* ai) {
    if (!ai) return;
    if (ai->context_pool) {
        pool_destroy(ai->context_pool);
        ai->context_pool = NULL;
    }
    ai->context_pool = pool_create();
}

void ai_release_context_pool(SpatialAI* ai) {
    if (!ai || !ai->context_pool) return;
    pool_destroy(ai->context_pool);
    ai->context_pool = NULL;
}

/* Grow keyframe array if needed */
static int ensure_kf_capacity(SpatialAI* ai) {
    if (ai->kf_count < ai->kf_capacity) return 1;

    uint32_t new_cap = ai->kf_capacity * 2;
    Keyframe* new_kf = (Keyframe*)realloc(ai->keyframes, new_cap * sizeof(Keyframe));
    if (!new_kf) return 0;

    ai->keyframes = new_kf;
    /* Zero new entries */
    memset(&ai->keyframes[ai->kf_capacity], 0,
           (new_cap - ai->kf_capacity) * sizeof(Keyframe));
    ai->kf_capacity = new_cap;
    return 1;
}

/* Grow delta array if needed */
static int ensure_df_capacity(SpatialAI* ai) {
    if (ai->df_count < ai->df_capacity) return 1;

    uint32_t new_cap = ai->df_capacity * 2;
    DeltaFrame* new_df = (DeltaFrame*)realloc(ai->deltas, new_cap * sizeof(DeltaFrame));
    if (!new_df) return 0;

    ai->deltas = new_df;
    memset(&ai->deltas[ai->df_capacity], 0,
           (new_cap - ai->df_capacity) * sizeof(DeltaFrame));
    ai->df_capacity = new_cap;
    return 1;
}

/* Allocate and copy grid channels into keyframe inline grid */
static void keyframe_alloc_grid(Keyframe* kf, const SpatialGrid* src) {
    kf->grid.A = (uint16_t*)malloc(GRID_TOTAL * sizeof(uint16_t));
    kf->grid.R = (uint8_t*)malloc(GRID_TOTAL);
    kf->grid.G = (uint8_t*)malloc(GRID_TOTAL);
    kf->grid.B = (uint8_t*)malloc(GRID_TOTAL);

    memcpy(kf->grid.A, src->A, GRID_TOTAL * sizeof(uint16_t));
    memcpy(kf->grid.R, src->R, GRID_TOTAL);
    memcpy(kf->grid.G, src->G, GRID_TOTAL);
    memcpy(kf->grid.B, src->B, GRID_TOTAL);
}

uint32_t compute_delta(const SpatialGrid* base, const SpatialGrid* target,
                       DeltaEntry* entries, uint32_t max_entries) {
    if (!base || !target || !entries) return 0;

    uint32_t count = 0;
    for (uint32_t i = 0; i < GRID_TOTAL && count < max_entries; i++) {
        int16_t dA = (int16_t)target->A[i] - (int16_t)base->A[i];
        int8_t  dR = (int8_t)((int)target->R[i] - (int)base->R[i]);
        int8_t  dG = (int8_t)((int)target->G[i] - (int)base->G[i]);
        int8_t  dB = (int8_t)((int)target->B[i] - (int)base->B[i]);

        if (dA != 0 || dR != 0 || dG != 0 || dB != 0) {
            entries[count].index  = i;
            entries[count].diff_A = dA;
            entries[count].diff_R = dR;
            entries[count].diff_G = dG;
            entries[count].diff_B = dB;
            count++;
        }
    }
    return count;
}

void apply_delta(const SpatialGrid* base, const DeltaEntry* entries,
                 uint32_t count, SpatialGrid* out) {
    if (!base || !entries || !out) return;

    grid_copy(out, base);

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = entries[i].index;
        if (idx >= GRID_TOTAL) continue;

        int val_A = (int)out->A[idx] + entries[i].diff_A;
        if (val_A < 0) val_A = 0;
        if (val_A > 65535) val_A = 65535;
        out->A[idx] = (uint16_t)val_A;

        int val_R = (int)out->R[idx] + entries[i].diff_R;
        if (val_R < 0) val_R = 0;
        if (val_R > 255) val_R = 255;
        out->R[idx] = (uint8_t)val_R;

        int val_G = (int)out->G[idx] + entries[i].diff_G;
        if (val_G < 0) val_G = 0;
        if (val_G > 255) val_G = 255;
        out->G[idx] = (uint8_t)val_G;

        int val_B = (int)out->B[idx] + entries[i].diff_B;
        if (val_B < 0) val_B = 0;
        if (val_B > 255) val_B = 255;
        out->B[idx] = (uint8_t)val_B;
    }
}

uint32_t ai_store_auto(SpatialAI* ai, const char* clause_text, const char* label) {
    if (!ai || !clause_text) return UINT32_MAX;

    /* Encode clause into grid */
    SpatialGrid* input = grid_create();
    if (!input) return UINT32_MAX;

    layers_encode_clause(clause_text, NULL, input);
    update_rgb_directional(input);
    apply_ema_to_grid(ai, input);

    uint32_t topic = resolve_topic(clause_text, label);

    /* If no keyframes yet, store as first keyframe */
    if (ai->kf_count == 0) {
        if (!ensure_kf_capacity(ai)) { grid_destroy(input); return UINT32_MAX; }

        Keyframe* kf = &ai->keyframes[0];
        kf->id = 0;
        if (label) strncpy(kf->label, label, 63);
        kf->label[63] = '\0';
        kf->text_byte_count = (uint32_t)strlen(clause_text);
        kf->topic_hash      = topic;
        kf->seq_in_topic    = topic ? 1 : 0;
        keyframe_alloc_grid(kf, input);

        ai->kf_count = 1;
        bucket_index_add(&ai->bucket_idx, input, 0);
        ema_update(ai, input);
        grid_destroy(input);
        return 0;
    }

    /* Matching strategy (spec v3 — topic-bucketed):
     *   1. If the clause has a topic, first walk only same-topic
     *      keyframes. Same-topic clauses typically share a lot of
     *      byte-position overlap (wiki abstracts often lead with
     *      the article subject) so this small linear scan finds
     *      deltas the flat scan missed.
     *   2. Fall back to the unified spatial_match(MATCH_SEARCH)
     *      cascade if the topic bucket is empty OR the best
     *      same-topic sim failed to clear the threshold. The
     *      cascade respects the bucket index when kf_count is
     *      large. */
    float    best_sim = 0.0f;
    uint32_t best_id  = UINT32_MAX;
    uint32_t bucket_best = topic_bucket_best_match(ai, input, topic, &best_sim);
    if (bucket_best != UINT32_MAX) best_id = bucket_best;

    if (best_sim < g_store_threshold) {
        MatchContext ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.bucket_idx = &ai->bucket_idx;
        MatchResult mr = spatial_match(ai, input, MATCH_SEARCH, &ctx);
        if (mr.best_score > best_sim) {
            best_sim = mr.best_score;
            best_id  = mr.best_id;
        }
    }

    if (best_sim >= g_store_threshold && best_id < ai->kf_count) {
        /* Store as delta frame */
        if (!ensure_df_capacity(ai)) { grid_destroy(input); return UINT32_MAX; }

        DeltaEntry* entries = (DeltaEntry*)malloc(GRID_TOTAL * sizeof(DeltaEntry));
        if (!entries) { grid_destroy(input); return UINT32_MAX; }

        uint32_t delta_count = compute_delta(&ai->keyframes[best_id].grid, input,
                                             entries, GRID_TOTAL);

        DeltaFrame* df = &ai->deltas[ai->df_count];
        df->id = ai->df_count;
        df->parent_id = best_id;
        if (label) strncpy(df->label, label, 63);
        df->label[63] = '\0';
        df->count = delta_count;
        if (delta_count > 0) {
            /* Shrink the scratch buffer to actual size. If realloc can't
             * shrink in place and returns NULL, keep the original — which
             * is still a valid free-able pointer. */
            DeltaEntry* shrunk = (DeltaEntry*)realloc(entries,
                                   delta_count * sizeof(DeltaEntry));
            df->entries = shrunk ? shrunk : entries;
        } else {
            /* Identical clauses: delta has zero entries. realloc(ptr, 0)
             * is implementation-defined (may free ptr and return NULL),
             * so free explicitly and leave entries NULL. df->count == 0
             * means readers/writers skip the array. */
            free(entries);
            df->entries = NULL;
        }
        uint32_t active = grid_active_count(input);
        df->change_ratio = active ? (float)delta_count / (float)active : 0.0f;

        /* Adaptive feedback: good structural match → boost the channel
         * that most contributed. Compute per-channel similarities
         * between input and matched parent keyframe. */
        {
            SpatialGrid* parent = &ai->keyframes[best_id].grid;
            float sA = channel_sim_A(input, parent);
            float sR = channel_sim_R(input, parent);
            float sG = channel_sim_G(input, parent);
            float sB = channel_sim_B(input, parent);
            weight_update(&ai->global_weights, sA, sR, sG, sB);
        }

        ai->df_count++;
        ema_update(ai, input);
        grid_destroy(input);
        return df->id | 0x80000000u; /* high bit indicates delta */
    } else {
        /* Store as new keyframe */
        if (!ensure_kf_capacity(ai)) { grid_destroy(input); return UINT32_MAX; }

        uint32_t new_id = ai->kf_count;
        Keyframe* kf = &ai->keyframes[new_id];
        kf->id = new_id;
        if (label) strncpy(kf->label, label, 63);
        kf->label[63] = '\0';
        kf->text_byte_count = (uint32_t)strlen(clause_text);
        kf->topic_hash      = topic;
        kf->seq_in_topic    = next_seq_in_topic(ai, topic);
        keyframe_alloc_grid(kf, input);

        /* Adaptive feedback for "novel" input: compare against the
         * nearest existing keyframe to learn which channel saw it as
         * novel (i.e. produced low similarity). Channels that were
         * already low-similarity have done their job in distinguishing
         * the new pattern; they earn a small boost. */
        if (best_id < ai->kf_count - 1) {  /* -1 since we just added */
            SpatialGrid* nearest = &ai->keyframes[best_id].grid;
            float sA = 1.0f - channel_sim_A(input, nearest);
            float sR = 1.0f - channel_sim_R(input, nearest);
            float sG = 1.0f - channel_sim_G(input, nearest);
            float sB = 1.0f - channel_sim_B(input, nearest);
            weight_update(&ai->global_weights, sA, sR, sG, sB);
        }

        ai->kf_count++;
        bucket_index_add(&ai->bucket_idx, input, new_id);
        ema_update(ai, input);
        grid_destroy(input);
        return new_id;
    }
}

uint32_t ai_force_keyframe(SpatialAI* ai, const char* clause_text, const char* label) {
    if (!ai || !clause_text) return UINT32_MAX;

    SpatialGrid* input = grid_create();
    if (!input) return UINT32_MAX;

    layers_encode_clause(clause_text, NULL, input);
    update_rgb_directional(input);
    apply_ema_to_grid(ai, input);

    if (!ensure_kf_capacity(ai)) { grid_destroy(input); return UINT32_MAX; }

    uint32_t new_id = ai->kf_count;
    Keyframe* kf = &ai->keyframes[new_id];
    kf->id = new_id;
    if (label) strncpy(kf->label, label, 63);
    kf->label[63] = '\0';
    kf->text_byte_count = (uint32_t)strlen(clause_text);
    kf->topic_hash      = resolve_topic(clause_text, label);
    kf->seq_in_topic    = next_seq_in_topic(ai, kf->topic_hash);
    keyframe_alloc_grid(kf, input);

    ai->kf_count++;
    bucket_index_add(&ai->bucket_idx, input, new_id);
    ema_update(ai, input);
    grid_destroy(input);
    return new_id;
}

uint32_t ai_store_grid(SpatialAI* ai, const SpatialGrid* input, const char* label) {
    if (!ai || !input) return UINT32_MAX;
    if (!ensure_kf_capacity(ai)) return UINT32_MAX;

    uint32_t new_id = ai->kf_count;
    Keyframe* kf = &ai->keyframes[new_id];
    kf->id = new_id;
    if (label) strncpy(kf->label, label, 63);
    kf->label[63] = '\0';
    kf->text_byte_count = 0;
    kf->topic_hash      = ai_resolve_topic(NULL, label);
    kf->seq_in_topic    = next_seq_in_topic(ai, kf->topic_hash);
    keyframe_alloc_grid(kf, input);

    /* SLIG v2.3 (codebook): split RGB → YCbCr at native 256², build
     * a 3-level image pyramid (32 / 128 / 256) for each plane,
     * decompose each (level, channel) slot, then register the
     * resulting SligCellSet in the engine codebook. The keyframe
     * stores only the byte-sized index — repeated textures across
     * keyframes converge onto a single pattern. */
    memset(kf->image_idx, 0, sizeof(kf->image_idx));
    kf->has_image = 0;

    uint8_t *plane_y_full  = (uint8_t*)malloc(GRID_TOTAL);
    uint8_t *plane_cb_full = (uint8_t*)malloc(GRID_TOTAL);
    uint8_t *plane_cr_full = (uint8_t*)malloc(GRID_TOTAL);
    /* Scratch for downsampled scales: max needed = 128² = 16384 B. */
    uint8_t *scratch       = (uint8_t*)malloc(GRID_TOTAL);
    if (plane_y_full && plane_cb_full && plane_cr_full && scratch) {
        uint32_t active = 0;
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            int32_t R = input->R[i];
            int32_t G = input->G[i];
            int32_t B = input->B[i];
            int32_t Y  = ( 9798 * R + 19235 * G +  3735 * B + 16384) >> 15;
            int32_t Cb = (-5537 * R - 10846 * G + 16384 * B + 16384) >> 15;
            int32_t Cr = (16384 * R - 13720 * G -  2664 * B + 16384) >> 15;
            Cb += 128; Cr += 128;
            if (Y  < 0) Y  = 0; else if (Y  > 255) Y  = 255;
            if (Cb < 0) Cb = 0; else if (Cb > 255) Cb = 255;
            if (Cr < 0) Cr = 0; else if (Cr > 255) Cr = 255;
            plane_y_full [i] = (uint8_t)Y;
            plane_cb_full[i] = (uint8_t)Cb;
            plane_cr_full[i] = (uint8_t)Cr;
            if (R || G || B) active++;
        }

        if (active > 0) {
            uint8_t *full_planes[SLIG_NUM_CHANNELS] = {
                plane_y_full, plane_cb_full, plane_cr_full
            };
            uint8_t channels[SLIG_NUM_CHANNELS] = {
                SLIG_CH_Y, SLIG_CH_CB, SLIG_CH_CR
            };
            uint32_t budgets[SLIG_NUM_CHANNELS] = {
                SLIG_CELL_BUDGET_Y, SLIG_CELL_BUDGET_CB, SLIG_CELL_BUDGET_CR
            };

            /* Residual pyramid encode (per-channel chain):
             *   coarse = decompose(downsample(full, 32))
             *   recon32 = reconstruct(coarse, 32)
             *   up128   = upsample(recon32, 128)
             *   mid     = decompose(downsample(full, 128) − up128 + 128)
             *   recon_mid_128 = reconstruct(mid, 128) − 128
             *   pred128 = clamp(up128 + recon_mid_128, 0, 255)
             *   up256   = upsample(pred128, 256)
             *   fine    = decompose(full − up256 + 128)
             * Each set goes into the codebook; the keyframe carries
             * only the 9 indices. The mid/fine sets store *residuals*
             * shifted to uint8 (offset 128), so reconstruction must
             * undo the shift before adding back. */
            int dim_c = slig_scale_dim(SLIG_LEVEL_COARSE);   /* 32  */
            int dim_m = slig_scale_dim(SLIG_LEVEL_MID);      /* 128 */
            int dim_f = slig_scale_dim(SLIG_LEVEL_FINE);     /* 256 */
            uint8_t *plane_32   = (uint8_t*)malloc((size_t)dim_c * dim_c);
            uint8_t *plane_128  = (uint8_t*)malloc((size_t)dim_m * dim_m);
            uint8_t *recon_32   = (uint8_t*)malloc((size_t)dim_c * dim_c);
            uint8_t *up_128     = (uint8_t*)malloc((size_t)dim_m * dim_m);
            uint8_t *res_128    = (uint8_t*)malloc((size_t)dim_m * dim_m);
            uint8_t *recon_mid  = (uint8_t*)malloc((size_t)dim_m * dim_m);
            uint8_t *pred_128   = (uint8_t*)malloc((size_t)dim_m * dim_m);
            uint8_t *up_256     = (uint8_t*)malloc((size_t)dim_f * dim_f);
            uint8_t *res_256    = (uint8_t*)malloc((size_t)dim_f * dim_f);
            int alloc_ok = plane_32 && plane_128 && recon_32 && up_128 &&
                           res_128 && recon_mid && pred_128 && up_256 && res_256;

            uint32_t total = 0;
            SligCellSet probe;
            if (alloc_ok) {
                for (int ch = 0; ch < SLIG_NUM_CHANNELS; ch++) {
                    const uint8_t *full = full_planes[ch];

                    /* Level 0 — coarse base */
                    slig_image_downsample(plane_32, dim_c, full, dim_f);
                    slig_decompose_channel(&probe, plane_32, dim_c, dim_c,
                                           channels[ch], SLIG_LEVEL_COARSE,
                                           budgets[ch]);
                    kf->image_idx[SLIG_LEVEL_COARSE][ch] =
                        slig_codebook_add_or_lookup(&ai->codebook, &probe,
                                                    SLIG_CODEBOOK_DEFAULT_THRESHOLD);
                    total += probe.num_cells;

                    /* Reconstruct → upsample → mid residual input */
                    slig_reconstruct_at_dim(recon_32, dim_c, &probe);
                    slig_image_upsample(up_128, dim_m, recon_32, dim_c);
                    slig_image_downsample(plane_128, dim_m, full, dim_f);
                    int n_m = dim_m * dim_m;
                    for (int i = 0; i < n_m; i++) {
                        int v = (int)plane_128[i] - (int)up_128[i] + 128;
                        if (v < 0) v = 0; else if (v > 255) v = 255;
                        res_128[i] = (uint8_t)v;
                    }

                    /* Level 1 — mid residual */
                    slig_decompose_channel(&probe, res_128, dim_m, dim_m,
                                           channels[ch], SLIG_LEVEL_MID,
                                           budgets[ch]);
                    kf->image_idx[SLIG_LEVEL_MID][ch] =
                        slig_codebook_add_or_lookup(&ai->codebook, &probe,
                                                    SLIG_CODEBOOK_DEFAULT_THRESHOLD);
                    total += probe.num_cells;

                    /* Predict 128, upsample to 256, compute fine residual */
                    slig_reconstruct_at_dim(recon_mid, dim_m, &probe);
                    for (int i = 0; i < n_m; i++) {
                        int v = (int)up_128[i] + (int)recon_mid[i] - 128;
                        if (v < 0) v = 0; else if (v > 255) v = 255;
                        pred_128[i] = (uint8_t)v;
                    }
                    slig_image_upsample(up_256, dim_f, pred_128, dim_m);
                    int n_f = dim_f * dim_f;
                    for (int i = 0; i < n_f; i++) {
                        int v = (int)full[i] - (int)up_256[i] + 128;
                        if (v < 0) v = 0; else if (v > 255) v = 255;
                        res_256[i] = (uint8_t)v;
                    }

                    /* Level 2 — fine residual */
                    slig_decompose_channel(&probe, res_256, dim_f, dim_f,
                                           channels[ch], SLIG_LEVEL_FINE,
                                           budgets[ch]);
                    kf->image_idx[SLIG_LEVEL_FINE][ch] =
                        slig_codebook_add_or_lookup(&ai->codebook, &probe,
                                                    SLIG_CODEBOOK_DEFAULT_THRESHOLD);
                    total += probe.num_cells;
                }
            }
            kf->has_image = (total > 0) ? 1 : 0;
            free(plane_32);  free(plane_128); free(recon_32);
            free(up_128);    free(res_128);   free(recon_mid);
            free(pred_128);  free(up_256);    free(res_256);
        }
    }
    free(plane_y_full); free(plane_cb_full); free(plane_cr_full); free(scratch);

    ai->kf_count++;
    bucket_index_add(&ai->bucket_idx, input, new_id);
    ema_update(ai, input);
    return new_id;
}

uint32_t ai_predict(SpatialAI* ai, const char* input_text, float* out_similarity) {
    if (!ai || !input_text || ai->kf_count == 0) {
        if (out_similarity) *out_similarity = 0.0f;
        return UINT32_MAX;
    }

    SpatialGrid* input = grid_create();
    if (!input) {
        if (out_similarity) *out_similarity = 0.0f;
        return UINT32_MAX;
    }

    layers_encode_clause(input_text, NULL, input);
    update_rgb_directional(input);
    apply_ema_to_grid(ai, input);

    /* Delegate to the unified 2-stage core. */
    MatchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.bucket_idx = &ai->bucket_idx;
    MatchResult r = spatial_match(ai, input, MATCH_PREDICT, &ctx);

    grid_destroy(input);
    if (out_similarity) *out_similarity = r.best_score;
    return r.best_id;
}
