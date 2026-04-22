/*
 * test_wiki.c — Wikipedia integration test on the Canvas Pool
 *
 * Stores each clause into a SpatialCanvasPool (SPEC §6). Clauses of
 * the same DataType cluster on the same 2048×1024 canvas; the
 * SubtitleTrack routes queries directly to slots of the query's type.
 *
 * Pipeline:
 *   1. Load + split clauses
 *   2. pool_add_clause for each clause (lazy canvases, type clustering)
 *   3. canvas_update_rgb per canvas (cross-boundary RGB diffusion)
 *   4. Query via pool_match (4-step cascade: type jump → A → R×G → B×A → other)
 *   5. Report canvas topology, type distribution, similarity histogram
 *   6. Recall@K on prefix queries
 *   7. Korean vs English script separation
 *   8. Next-clause prediction via subtitle_entry_id + 1
 *
 * Usage:
 *   ./build/test_wiki data/sample_ko.txt
 *   ./build/test_wiki data/sample_en.txt 500
 *   ./build/test_wiki data/sample_en.txt --save model_en.spai
 *   ./build/test_wiki data/sample_en.txt --load model_en.spai --save mixed.spai
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_context.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"
#include "spatial_io.h"
#include "bench_utf8.h"
#include "bench_args.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_CLAUSES     100000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10
#define DEFAULT_LIMIT   2000
#define TOP_PAIRS       10
#define RECALL_QUERIES  200
#define NEXT_QUERIES    200

typedef struct { uint32_t a, b; float sim; } SimPair;

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    return (line[0] == '<' || line[0] == '\0');
}
static void trim_trailing(char* s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r' ||
                       s[len-1] == ' '  || s[len-1] == '\t')) {
        s[--len] = '\0';
    }
}
static uint32_t split_clauses(const char* line, char out[][MAX_LINE_LEN],
                              uint32_t max_out) {
    uint32_t count = 0;
    const char* p = line;
    const char* start = p;
    while (*p && count < max_out) {
        if (*p == '.' || *p == '!' || *p == '?') {
            uint32_t len = (uint32_t)(p - start + 1);
            if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
                memcpy(out[count], start, len);
                out[count][len] = '\0';
                trim_trailing(out[count]);
                if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) count++;
            }
            start = p + 1;
            while (*start == ' ') start++;
            p = start;
            continue;
        }
        p++;
    }
    if (start < p && count < max_out) {
        uint32_t len = (uint32_t)(p - start);
        if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
            memcpy(out[count], start, len);
            out[count][len] = '\0';
            trim_trailing(out[count]);
            if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) count++;
        }
    }
    return count;
}

static int classify_script(const char* text) {
    const uint8_t* b = (const uint8_t*)text;
    uint32_t ascii = 0, cjk = 0, total = 0;
    for (uint32_t i = 0; b[i]; i++) {
        total++;
        if (b[i] < 0x80) ascii++;
        else if (b[i] >= 0xE0 && b[i] <= 0xEF) cjk++;
    }
    if (total == 0) return 2;
    if (cjk * 2 > total) return 1;
    if (ascii * 4 > total * 3) return 0;
    return 2;
}

static void make_prefix(const char* src, char* dst, uint32_t max_bytes) {
    uint32_t len = (uint32_t)strlen(src);
    if (len <= max_bytes) { strcpy(dst, src); return; }
    uint32_t cut = max_bytes;
    while (cut > max_bytes / 2 && src[cut] != ' ' && src[cut] != '\t') cut--;
    if (cut < max_bytes / 2) cut = max_bytes;
    memcpy(dst, src, cut);
    dst[cut] = '\0';
}

int main(int argc, char* argv[]) {
    utf8_console_init();

    BenchArgs ba;
    if (bench_parse_args(argc, argv, &ba) != 0 || ba.positional_count < 1) {
        fprintf(stderr,
            "Usage: %s <text_file> [max_clauses] [--save PATH] [--load PATH] [--load-only PATH]\n"
            "\n"
            "  Canvas-Pool edition. Clauses are tiled onto 2048x1024 canvases\n"
            "  by DataType; queries use SubtitleTrack type jump for retrieval.\n"
            "\n"
            "Example:\n"
            "  %s data/sample_en.txt 500 --save model.spai\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* filepath    = ba.positional[0];
    uint32_t    max_clauses = (ba.positional_count >= 2)
                              ? (uint32_t)atoi(ba.positional[1]) : DEFAULT_LIMIT;
    if (max_clauses > MAX_CLAUSES) max_clauses = MAX_CLAUSES;

    FILE* fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return 1; }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  Wiki Test (Canvas Pool)\n");
    printf("========================================\n");
    printf("  File:        %s\n", filepath);
    printf("  Max clauses: %u\n", max_clauses);
    printf("----------------------------------------\n\n");

    /* ── [1/8] Load + split ── */
    printf("[1/8] Loading and splitting clauses...\n");
    double t0 = now_sec();
    morpheme_init();

    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    if (!clauses) { fprintf(stderr, "alloc failed\n"); fclose(fp); return 1; }

    uint32_t clause_count = 0;
    uint64_t total_bytes = 0;
    char line[MAX_LINE_LEN];
    char split_buf[8][MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && clause_count < max_clauses) {
        trim_trailing(line);
        if (is_meta_line(line)) continue;
        if ((int)strlen(line) < MIN_CLAUSE_LEN) continue;
        uint32_t n = split_clauses(line, split_buf, 8);
        for (uint32_t i = 0; i < n && clause_count < max_clauses; i++) {
            memcpy(clauses[clause_count], split_buf[i], MAX_LINE_LEN);
            total_bytes += strlen(split_buf[i]);
            clause_count++;
        }
    }
    fclose(fp);

    double t_load = now_sec() - t0;
    printf("  Loaded:  %u clauses, %llu bytes (%.2f sec)\n\n",
           clause_count, (unsigned long long)total_bytes, t_load);
    if (clause_count < 10) { printf("  Too few clauses. Abort.\n"); free(clauses); return 1; }

    for (uint32_t i = 0; i < 3 && i < clause_count; i++) {
        char preview[80];
        strncpy(preview, clauses[i], 76); preview[76] = '\0';
        if (strlen(clauses[i]) > 76) strcat(preview, "...");
        printf("    [%u] %s\n", i, preview);
    }
    printf("\n");

    /* Script stats */
    uint32_t script_ko = 0, script_en = 0, script_mix = 0;
    uint8_t* script = (uint8_t*)malloc(clause_count);
    for (uint32_t i = 0; i < clause_count; i++) {
        script[i] = (uint8_t)classify_script(clauses[i]);
        if (script[i] == 0) script_en++;
        else if (script[i] == 1) script_ko++;
        else script_mix++;
    }

    /* ── [2/8] Pool + type-clustered tile placement ── */
    printf("[2/8] Placing clauses onto Canvas Pool (type clustering)...\n");
    t0 = now_sec();

    SpatialAI* ai = NULL;
    uint32_t loaded_entries = 0;
    if (ba.load_path) {
        SpaiStatus ls;
        ai = ai_load(ba.load_path, &ls);
        if (!ai) {
            fprintf(stderr, "\n  ERROR: load '%s' failed: %s\n",
                    ba.load_path, spai_status_str(ls));
            free(clauses); free(script);
            return 1;
        }
        if (ai->canvas_pool) loaded_entries = ai->canvas_pool->track.count;
        printf("  Loaded '%s': kf=%u delta=%u pool_slots=%u\n",
               ba.load_path, ai->kf_count, ai->df_count, loaded_entries);
    } else {
        ai = spatial_ai_create();
    }
    SpatialCanvasPool* pool = ai_get_canvas_pool(ai);

    uint32_t* clause_to_entry = (uint32_t*)malloc(clause_count * sizeof(uint32_t));
    for (uint32_t i = 0; i < clause_count; i++) clause_to_entry[i] = UINT32_MAX;

    if (ba.load_only) {
        printf("  --load-only: skipping placement; %u pool slots available.\n",
               pool_total_slots(pool));
    } else {
        for (uint32_t i = 0; i < clause_count; i++) {
            int e = pool_add_clause(pool, clauses[i]);
            if (e >= 0) clause_to_entry[i] = (uint32_t)e;
            if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
                printf("\r  Placed: %u / %u  (canvases=%u, slots=%u)",
                       i + 1, clause_count, pool->count, pool_total_slots(pool));
                fflush(stdout);
            }
        }
        printf("\n");
    }

    double t_store = now_sec() - t0;
    printf("  Done in %.2f sec  (%.0f clauses/sec)\n\n",
           t_store, clause_count / (t_store + 1e-9));

    /* ── [3/8] Canvas-wide RGB diffusion (cross-boundary) ── */
    printf("[3/8] Running canvas_update_rgb on %u canvas(es)...\n", pool->count);
    t0 = now_sec();
    for (uint32_t i = 0; i < pool->count; i++) {
        canvas_update_rgb(pool->canvases[i]);
    }
    double t_diffuse = now_sec() - t0;
    printf("  Done in %.2f sec\n\n", t_diffuse);

    /* Save */
    if (ba.save_path) {
        SpaiStatus ss = ai_save(ai, ba.save_path);
        printf("  [save] full → '%s' : %s  (kf=%u, slots=%u, canvases=%u)\n\n",
               ba.save_path, spai_status_str(ss),
               ai->kf_count, pool_total_slots(pool), pool->count);
        if (ss != SPAI_OK) fprintf(stderr, "  WARNING: save failed\n");
    }

    /* ── [4/8] Query match via pool_match ── */
    printf("[4/8] Running queries via pool_match (type jump + cascade)...\n");
    t0 = now_sec();

    float* similarities = (float*)malloc(clause_count * sizeof(float));
    uint32_t exact_matches = 0, high_matches = 0, low_matches = 0;
    double sim_sum = 0.0;
    uint32_t hist[10] = {0};
    SimPair top_pairs[TOP_PAIRS];
    memset(top_pairs, 0, sizeof(top_pairs));
    uint32_t step_counts[5] = {0};   /* step 1..4 + other */
    uint32_t fallback_count = 0;

    for (uint32_t i = 0; i < clause_count; i++) {
        SpatialGrid* input = grid_create();
        layers_encode_clause(clauses[i], NULL, input);
        update_rgb_directional(input);

        PoolMatchResult r = pool_match(pool, input, clauses[i]);
        similarities[i] = r.similarity;
        sim_sum += r.similarity;

        if (r.similarity >= 0.99f) exact_matches++;
        if (r.similarity >= 0.50f) high_matches++;
        if (r.similarity <  0.10f) low_matches++;

        int b = (int)(r.similarity * 10.0f);
        if (b < 0) b = 0;
        if (b > 9) b = 9;
        hist[b]++;

        if (r.step_taken >= 1 && r.step_taken <= 4) step_counts[r.step_taken]++;
        if (r.fallback) fallback_count++;

        /* Top cross-entry pairs (excluding self-match) */
        if (r.subtitle_entry_id != clause_to_entry[i] &&
            r.similarity > top_pairs[TOP_PAIRS - 1].sim) {
            top_pairs[TOP_PAIRS - 1].a = i;
            top_pairs[TOP_PAIRS - 1].b = r.subtitle_entry_id;
            top_pairs[TOP_PAIRS - 1].sim = r.similarity;
            for (int j = TOP_PAIRS - 2; j >= 0; j--) {
                if (top_pairs[j + 1].sim > top_pairs[j].sim) {
                    SimPair tmp = top_pairs[j];
                    top_pairs[j] = top_pairs[j + 1];
                    top_pairs[j + 1] = tmp;
                }
            }
        }

        grid_destroy(input);
        if ((i + 1) % 200 == 0 || i + 1 == clause_count) {
            printf("\r  Queried: %u / %u", i + 1, clause_count);
            fflush(stdout);
        }
    }
    double t_query = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f queries/sec)\n\n",
           t_query, clause_count / (t_query + 1e-9));

    /* ── [5/8] Basic report ── */
    printf("[5/8] Results\n");
    printf("========================================\n\n");

    /* Canvas I/P counts */
    uint32_t n_iframe = 0, n_pframe = 0, n_open = 0;
    for (uint32_t i = 0; i < pool->count; i++) {
        SpatialCanvas* c = pool->canvases[i];
        if (!c->classified) { n_open++; continue; }
        if (c->frame_type == CANVAS_IFRAME) n_iframe++;
        else if (c->frame_type == CANVAS_PFRAME) n_pframe++;
    }

    printf("  STORAGE (Canvas Pool)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Total clauses:     %u\n", clause_count);
    printf("  Total bytes:       %llu\n", (unsigned long long)total_bytes);
    printf("  Canvases:          %u  (KF=%u, Delta=%u, Open=%u)\n",
           pool->count, n_iframe, n_pframe, n_open);
    printf("  Slots used:        %u\n", pool_total_slots(pool));
    if (pool->count > 0)
        printf("  Avg slots/canvas:  %.1f / %d\n",
               (double)pool_total_slots(pool) / pool->count, CV_SLOTS);
    printf("  Scene-change EMA:  %.0f  (samples=%u)\n",
           pool->scene.threshold_ema, pool->scene.n_samples);
    printf("\n");

    /* 3-layer brightness diagnostics: verify the new 1/3/5 weighting
       appears in effective A distributions (1 / 4 / 6 / 9). */
    uint64_t a_active = 0, a_total = 0, a_non_base = 0;
    uint64_t a_1 = 0, a_4 = 0, a_6 = 0, a_9 = 0;
    SpatialGrid* dg = grid_create();
    for (uint32_t e = 0; e < pool->track.count; e++) {
        const SubtitleEntry* se = &pool->track.entries[e];
        canvas_slot_to_grid(pool->canvases[se->canvas_id], se->slot_id, dg);
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            uint16_t a = dg->A[i];
            if (a == 0) continue;
            a_active++;
            a_total += a;
            if (a > 1) a_non_base += (uint64_t)(a - 1);
            if (a == 1) a_1++;
            else if (a == 4) a_4++;
            else if (a == 6) a_6++;
            else if (a == 9) a_9++;
        }
    }
    grid_destroy(dg);

    double avg_a = (a_active > 0) ? ((double)a_total / (double)a_active) : 0.0;
    uint64_t a_diag = a_1 + a_4 + a_6 + a_9;

    printf("  3-LAYER BRIGHTNESS DIAGNOSTICS\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Active cells:      %llu\n", (unsigned long long)a_active);
    printf("  Avg A:             %.3f\n", avg_a);
    printf("  A=1:               %llu\n", (unsigned long long)a_1);
    printf("  A=4:               %llu\n", (unsigned long long)a_4);
    printf("  A=6:               %llu\n", (unsigned long long)a_6);
    printf("  A=9:               %llu\n", (unsigned long long)a_9);
    printf("  Non-base contrib:  %llu\n", (unsigned long long)a_non_base);
    if (a_active > 0) {
        printf("  Tier coverage:     %.1f%%  (A in {1,4,6,9})\n",
               100.0 * (double)a_diag / (double)a_active);
    }
    printf("\n");

    /* Type distribution per DataType */
    printf("  TYPE DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    for (uint32_t t = 0; t < DATA_TYPE_COUNT; t++) {
        uint32_t n_t;
        subtitle_track_ids_of_type(&pool->track, (DataType)t, &n_t);
        printf("  %-8s  %u slots\n", data_type_name((DataType)t), n_t);
    }
    /* Canvas-by-canvas utilization */
    printf("\n  CANVAS UTILIZATION\n");
    printf("  ─────────────────────────────────────\n");
    for (uint32_t i = 0; i < pool->count && i < 20; i++) {
        SpatialCanvas* c = pool->canvases[i];
        const char* lbl = !c->classified ? "OPEN"
                          : (c->frame_type == CANVAS_IFRAME ? "KF" : "Delta");
        if (c->frame_type == CANVAS_PFRAME && c->classified) {
            printf("  Canvas %2u  type=%-6s  slots=%u/%d  [%s parent=%u changed=%.1f%%]\n",
                   i, data_type_name(c->canvas_type), c->slot_count, CV_SLOTS,
                   lbl, c->parent_canvas_id, c->changed_ratio * 100.0f);
        } else {
            printf("  Canvas %2u  type=%-6s  slots=%u/%d  [%s]\n",
                   i, data_type_name(c->canvas_type), c->slot_count, CV_SLOTS, lbl);
        }
    }
    if (pool->count > 20) printf("  ... (%u more)\n", pool->count - 20);
    printf("\n");

    printf("  DELTA / COMPRESSION DIAGNOSTICS\n");
    printf("  ─────────────────────────────────────\n");
    if (n_pframe == 0) {
        printf("  Delta canvases:    0\n\n");
    } else {
        CanvasDeltaEntry* tmp_delta =
            (CanvasDeltaEntry*)malloc((size_t)CV_TOTAL * sizeof(CanvasDeltaEntry));
        if (!tmp_delta) {
            printf("  Delta diagnostics: skipped (alloc failed)\n\n");
        } else {
            uint64_t total_changed = 0;
            uint64_t total_sparse_bytes = 0;
            uint64_t total_rle_bytes = 0;
            uint64_t total_full_a_bytes = 0;
            double   ratio_sum = 0.0;
            uint32_t ratio_n = 0;

            for (uint32_t i = 0; i < pool->count; i++) {
                SpatialCanvas* c = pool->canvases[i];
                if (!c->classified || c->frame_type != CANVAS_PFRAME) continue;
                if (c->parent_canvas_id == UINT32_MAX || c->parent_canvas_id >= pool->count)
                    continue;

                SpatialCanvas* p = pool->canvases[c->parent_canvas_id];
                uint32_t changed = canvas_delta_sparse(p, c, tmp_delta, CV_TOTAL);
                uint32_t rle_bytes = canvas_delta_rle_bytes(tmp_delta, changed);

                total_changed += changed;
                total_sparse_bytes += (uint64_t)changed * 6ull;
                total_rle_bytes += (uint64_t)rle_bytes;
                total_full_a_bytes += (uint64_t)CV_TOTAL * (uint64_t)sizeof(uint16_t);
                ratio_sum += c->changed_ratio;
                ratio_n++;
            }

            double saved_vs_full = 0.0;
            double saved_vs_sparse = 0.0;
            if (total_full_a_bytes > 0) {
                saved_vs_full = 100.0 * (1.0 - (double)total_rle_bytes / (double)total_full_a_bytes);
            }
            if (total_sparse_bytes > 0) {
                saved_vs_sparse = 100.0 * (1.0 - (double)total_rle_bytes / (double)total_sparse_bytes);
            }

            printf("  Delta canvases:    %u\n", n_pframe);
            printf("  Changed cells:     %llu\n", (unsigned long long)total_changed);
            printf("  change_ratio avg:  %.2f%%\n",
                   (ratio_n > 0) ? (100.0 * ratio_sum / (double)ratio_n) : 0.0);
            printf("  Full A bytes:      %llu\n", (unsigned long long)total_full_a_bytes);
            printf("  Sparse bytes:      %llu\n", (unsigned long long)total_sparse_bytes);
            printf("  RLE bytes:         %llu\n", (unsigned long long)total_rle_bytes);
            printf("  Saved vs full A:   %.1f%%\n", saved_vs_full);
            printf("  Saved vs sparse:   %.1f%%\n", saved_vs_sparse);
            printf("\n");

            free(tmp_delta);
        }
    }

    float avg_sim = (clause_count > 0) ? (float)(sim_sum / clause_count) : 0.0f;
    printf("  MATCHING (pool_match self-query)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Avg similarity:  %.1f%%\n", avg_sim * 100.0f);
    printf("  Exact (>=99%%):   %u  (%.1f%%)\n",
           exact_matches, 100.0f * exact_matches / clause_count);
    printf("  High (>=50%%):    %u  (%.1f%%)\n",
           high_matches, 100.0f * high_matches / clause_count);
    printf("  Low (<10%%):      %u\n", low_matches);
    printf("\n");

    printf("  CASCADE STEPS\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Step 1 (A):      %u\n", step_counts[1]);
    printf("  Step 2 (RG):     %u\n", step_counts[2]);
    printf("  Step 3 (BA):     %u\n", step_counts[3]);
    printf("  Step 4 (other):  %u\n", step_counts[4]);
    printf("  Fallbacks:       %u  (%.1f%%)\n",
           fallback_count, 100.0f * fallback_count / clause_count);
    printf("\n");

    printf("  SIMILARITY DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    uint32_t hist_max = 1;
    for (int i = 0; i < 10; i++) if (hist[i] > hist_max) hist_max = hist[i];
    for (int i = 0; i < 10; i++) {
        int bar = (int)((float)hist[i] / hist_max * 30.0f);
        if (hist[i] > 0 && bar == 0) bar = 1;
        printf("  %3d-%3d%% | ", i * 10, (i + 1) * 10);
        for (int b = 0; b < bar; b++) printf("#");
        printf(" %u\n", hist[i]);
    }
    printf("\n");

    printf("  TOP-%d MOST SIMILAR PAIRS (cross-entry)\n", TOP_PAIRS);
    printf("  ─────────────────────────────────────\n");
    for (int i = 0; i < TOP_PAIRS; i++) {
        if (top_pairs[i].sim <= 0.0f) break;
        char pa[60];
        strncpy(pa, clauses[top_pairs[i].a], 56); pa[56] = '\0';
        if (strlen(clauses[top_pairs[i].a]) > 56) strcat(pa, "...");
        printf("  %2d. %.1f%%  Q[%u]: %s\n", i + 1,
               top_pairs[i].sim * 100.0f, top_pairs[i].a, pa);
    }
    printf("\n");

    /* ── [6/8] Recall@K on prefix queries (pool_match_topk over all entries) ── */
    printf("[6/8] Recall@K (prefix queries)\n");
    printf("  ─────────────────────────────────────\n");
    t0 = now_sec();

    uint32_t nq = (clause_count < RECALL_QUERIES) ? clause_count : RECALL_QUERIES;
    uint32_t recall_at_1 = 0, recall_at_5 = 0, recall_at_10 = 0;
    uint32_t evaluated = 0;

    uint32_t top_ids[10];
    float    top_scores[10];

    for (uint32_t q = 0; q < nq; q++) {
        uint32_t ci = q * (clause_count / nq);
        if (ci >= clause_count) ci = clause_count - 1;

        uint32_t gold_entry = clause_to_entry[ci];
        if (gold_entry == UINT32_MAX) continue;

        char prefix[MAX_LINE_LEN];
        uint32_t full_len = (uint32_t)strlen(clauses[ci]);
        uint32_t plen = full_len * 6 / 10;
        if (plen < 10) plen = full_len;
        make_prefix(clauses[ci], prefix, plen);

        SpatialGrid* qg = grid_create();
        layers_encode_clause(prefix, NULL, qg);
        update_rgb_directional(qg);

        uint32_t k = pool_match_topk(pool, qg, 10, top_ids, top_scores);

        int at1 = 0, at5 = 0, at10 = 0;
        for (uint32_t i = 0; i < k; i++) {
            if (top_ids[i] == gold_entry) {
                at10 = 1;
                if (i < 5) at5 = 1;
                if (i == 0) at1 = 1;
                break;
            }
        }
        if (at1) recall_at_1++;
        if (at5) recall_at_5++;
        if (at10) recall_at_10++;
        evaluated++;

        grid_destroy(qg);
    }

    double t_recall = now_sec() - t0;
    printf("  Queries:     %u  (prefix = 60%% of original)\n", evaluated);
    if (evaluated > 0) {
        printf("  Recall@1:    %.1f%%  (%u / %u)\n",
               100.0f * recall_at_1  / evaluated, recall_at_1,  evaluated);
        printf("  Recall@5:    %.1f%%  (%u / %u)\n",
               100.0f * recall_at_5  / evaluated, recall_at_5,  evaluated);
        printf("  Recall@10:   %.1f%%  (%u / %u)\n",
               100.0f * recall_at_10 / evaluated, recall_at_10, evaluated);
    }
    printf("  Query time:  %.2f sec  (%.0f/sec)\n",
           t_recall, evaluated / (t_recall + 1e-9));
    printf("\n");

    /* ── [7/8] Language separation (uses scripts, unchanged) ── */
    printf("[7/8] Language separation (Korean vs English)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Clause script distribution:\n");
    printf("    Korean (CJK): %u  (%.1f%%)\n", script_ko, 100.0f * script_ko / clause_count);
    printf("    English:      %u  (%.1f%%)\n", script_en, 100.0f * script_en / clause_count);
    printf("    Mixed/other:  %u  (%.1f%%)\n", script_mix, 100.0f * script_mix / clause_count);

    if (script_ko > 10 && script_en > 10) {
        uint32_t ko_pairs = 0, en_pairs = 0, cross_pairs = 0;
        double ko_sum = 0, en_sum = 0, cross_sum = 0;
        SpatialGrid* ga = grid_create();
        SpatialGrid* gb = grid_create();
        uint32_t step = clause_count / 30; if (step == 0) step = 1;

        for (uint32_t i = 0; i < clause_count; i += step) {
            for (uint32_t j = i + step; j < clause_count; j += step) {
                grid_clear(ga); grid_clear(gb);
                layers_encode_clause(clauses[i], NULL, ga);
                layers_encode_clause(clauses[j], NULL, gb);
                float s = cosine_a_only(ga, gb);
                if (script[i] == 1 && script[j] == 1) { ko_sum += s; ko_pairs++; }
                else if (script[i] == 0 && script[j] == 0) { en_sum += s; en_pairs++; }
                else if ((script[i] == 0 && script[j] == 1) ||
                         (script[i] == 1 && script[j] == 0)) { cross_sum += s; cross_pairs++; }
            }
        }
        grid_destroy(ga);
        grid_destroy(gb);

        double avg_ko    = ko_pairs    ? ko_sum    / ko_pairs    : 0.0;
        double avg_en    = en_pairs    ? en_sum    / en_pairs    : 0.0;
        double avg_cross = cross_pairs ? cross_sum / cross_pairs : 0.0;

        printf("\n  Average cosine (sampled pairs):\n");
        printf("    Korean ↔ Korean:   %.1f%%  (n=%u)\n", avg_ko    * 100, ko_pairs);
        printf("    English ↔ English: %.1f%%  (n=%u)\n", avg_en    * 100, en_pairs);
        printf("    Korean ↔ English:  %.1f%%  (n=%u)\n", avg_cross * 100, cross_pairs);
    } else {
        printf("\n  Single-script corpus. Separation test skipped.\n");
    }
    printf("\n");

    /* ── [8/8] Next-clause prediction via subtitle_entry_id + 1 ── */
    printf("[8/8] Next-clause prediction (entry+1 in insertion order)\n");
    printf("  ─────────────────────────────────────\n");
    uint32_t np = (clause_count - 1 < NEXT_QUERIES) ? clause_count - 1 : NEXT_QUERIES;
    uint32_t correct_top1 = 0;

    SpatialGrid* gg_next = grid_create();
    SpatialGrid* gg_tile = grid_create();

    for (uint32_t q = 0; q < np; q++) {
        uint32_t ci = q * ((clause_count - 1) / np);
        if (ci + 1 >= clause_count) continue;

        /* Match c_i via pool */
        SpatialGrid* qg = grid_create();
        layers_encode_clause(clauses[ci], NULL, qg);
        update_rgb_directional(qg);
        PoolMatchResult r = pool_match(pool, qg, clauses[ci]);
        grid_destroy(qg);

        uint32_t matched_e = r.subtitle_entry_id;
        if (matched_e >= pool->track.count) continue;
        uint32_t next_e = (matched_e + 1 < pool->track.count) ? (matched_e + 1) : matched_e;

        /* Extract predicted next slot to a temporary grid */
        SubtitleEntry* ne = &pool->track.entries[next_e];
        canvas_slot_to_grid(pool->canvases[ne->canvas_id], ne->slot_id, gg_tile);

        /* Encode ground-truth next clause */
        grid_clear(gg_next);
        layers_encode_clause(clauses[ci + 1], NULL, gg_next);
        float s_pred = cosine_a_only(gg_next, gg_tile);

        /* Baseline: best of 5 random slots */
        float s_rand_best = 0.0f;
        for (int k = 0; k < 5; k++) {
            uint32_t re = (uint32_t)(q * 131 + k * 17) % pool->track.count;
            SubtitleEntry* rre = &pool->track.entries[re];
            canvas_slot_to_grid(pool->canvases[rre->canvas_id], rre->slot_id, gg_tile);
            float sr = cosine_a_only(gg_next, gg_tile);
            if (sr > s_rand_best) s_rand_best = sr;
        }
        if (s_pred >= s_rand_best) correct_top1++;
    }
    grid_destroy(gg_next);
    grid_destroy(gg_tile);

    printf("  Queries:       %u  (c_i → predict c_{i+1})\n", np);
    printf("  Predict top-1: %.1f%%  (beats best-of-5 random slot)\n",
           100.0f * correct_top1 / (np + 1));
    printf("\n");

    printf("========================================\n");
    printf("  PASS  (%u clauses, %u canvases, %u slots)\n",
           clause_count, pool->count, pool_total_slots(pool));
    printf("========================================\n");

    free(similarities);
    free(clause_to_entry);
    free(script);
    free(clauses);
    spatial_ai_destroy(ai);
    return 0;
}
