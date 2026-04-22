/*
 * bench_full_v2.c — CANVAS comprehensive benchmark (corrected)
 *
 * Fixes from v1:
 *   - Word prediction uses A-channel argmax (3-layer sum = word importance)
 *     NOT the broken A*R_sim*G_sim*B_sim multiplication
 *   - Loads pre-trained .spai model instead of re-training
 *   - Includes grade criteria in report
 *
 * Usage:
 *   bench_full_v2 <input.txt> [max] --load <model.spai> --report <output.txt>
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"
#include "spatial_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#  define MKDIR(p) mkdir(p)
#else
#  include <unistd.h>
#  include <sys/resource.h>
#  define MKDIR(p) mkdir((p), 0755)
#endif

#define MAX_LINE        4096
#define MIN_CLAUSE_LEN  10
#define DEFAULT_MAX     25000
#define TRAIN_RATIO     0.70f
#define GEN_TESTS       100
#define WORD_PRED_TESTS 500

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double get_memory_mb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (double)ru.ru_maxrss / 1024.0;
    return 0.0;
#endif
}

static long file_size_bytes(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return (long)st.st_size;
    return 0;
}

static int is_valid_utf8(const char* s, uint32_t len) {
    uint32_t i = 0;
    while (i < len) {
        uint8_t b = (uint8_t)s[i];
        int seq;
        if (b < 0x80) seq = 1;
        else if ((b & 0xE0) == 0xC0) seq = 2;
        else if ((b & 0xF0) == 0xE0) seq = 3;
        else if ((b & 0xF8) == 0xF0) seq = 4;
        else return 0;
        if (i + seq > len) return 0;
        for (int k = 1; k < seq; k++)
            if (((uint8_t)s[i+k] & 0xC0) != 0x80) return 0;
        i += seq;
    }
    return 1;
}

static const char* grade_word(float pct) {
    if (pct >= 60) return "EXCELLENT";
    if (pct >= 40) return "GOOD";
    if (pct >= 20) return "USABLE";
    if (pct >= 5)  return "MINIMAL";
    return "BELOW BASELINE";
}

static const char* grade_match(float avg_sim) {
    if (avg_sim >= 0.80) return "EXCELLENT";
    if (avg_sim >= 0.60) return "GOOD";
    if (avg_sim >= 0.40) return "USABLE";
    if (avg_sim >= 0.20) return "MINIMAL";
    return "FAIL";
}

static const char* grade_recall(float pct) {
    if (pct >= 99) return "EXCELLENT";
    if (pct >= 95) return "GOOD";
    if (pct >= 80) return "MINIMAL";
    return "FAIL";
}

static const char* grade_gen(float utf8_pct, float unique_pct) {
    if (utf8_pct >= 99 && unique_pct >= 80) return "GOOD";
    if (utf8_pct >= 95 && unique_pct >= 60) return "USABLE";
    if (utf8_pct >= 80 && unique_pct >= 30) return "MINIMAL";
    return "FAIL";
}

static const char* grade_speed(double cps) {
    if (cps >= 200) return "VERY FAST";
    if (cps >= 50)  return "FAST";
    if (cps >= 10)  return "NORMAL";
    return "SLOW";
}

static const char* grade_efficiency(float kb_per_clause, float delta_pct) {
    if (kb_per_clause < 20 && delta_pct > 70) return "VERY EFFICIENT";
    if (kb_per_clause < 100 && delta_pct > 40) return "EFFICIENT";
    if (kb_per_clause < 500 && delta_pct > 10) return "NORMAL";
    return "INEFFICIENT";
}

typedef struct {
    const char* input;
    uint32_t    max_clauses;
    const char* load_path;
    const char* report_path;
} BenchArgs;

static BenchArgs parse_args(int argc, char** argv) {
    BenchArgs a = {0};
    a.max_clauses = DEFAULT_MAX;
    a.report_path = "bench_report.txt";
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0 && i+1 < argc) {
            a.load_path = argv[++i];
        } else if (strcmp(argv[i], "--report") == 0 && i+1 < argc) {
            a.report_path = argv[++i];
        } else if (argv[i][0] != '-') {
            if (positional == 0) { a.input = argv[i]; positional++; }
            else { int v = atoi(argv[i]); if (v > 0) a.max_clauses = (uint32_t)v; positional++; }
        }
    }
    return a;
}

static char** load_clauses(const char* path, uint32_t max, uint32_t* out_count) {
    FILE* fp = fopen(path, "r");
    if (!fp) { *out_count = 0; return NULL; }
    char** lines = malloc(max * sizeof(char*));
    uint32_t n = 0;
    char buf[MAX_LINE];
    while (n < max && fgets(buf, MAX_LINE, fp)) {
        uint32_t len = (uint32_t)strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = '\0';
        if (len < MIN_CLAUSE_LEN) continue;
        lines[n++] = strdup(buf);
    }
    fclose(fp);
    *out_count = n;
    return lines;
}

static void free_clauses(char** lines, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) free(lines[i]);
    free(lines);
}

int main(int argc, char** argv) {
    BenchArgs args = parse_args(argc, argv);
    if (!args.input) {
        fprintf(stderr, "Usage: bench_full_v2 <input.txt> [max] --load <model.spai> --report <output.txt>\n");
        return 1;
    }

    FILE* rpt = fopen(args.report_path, "w");
    if (!rpt) { fprintf(stderr, "Cannot open report: %s\n", args.report_path); return 1; }

    #define RPT(fmt, ...) do { fprintf(rpt, fmt, ##__VA_ARGS__); fprintf(stdout, fmt, ##__VA_ARGS__); } while(0)

    RPT("================================================================\n");
    RPT("  SPATIAL-PATTERN-AI  Comprehensive Benchmark v2\n");
    RPT("================================================================\n\n");

    /* Load data */
    uint32_t total = 0;
    char** clauses = load_clauses(args.input, args.max_clauses, &total);
    if (!clauses || total == 0) {
        RPT("ERROR: no clauses from %s\n", args.input);
        fclose(rpt);
        return 2;
    }

    uint32_t train_n = (uint32_t)(total * TRAIN_RATIO);
    uint32_t test_n  = total - train_n;
    RPT("Input:          %s\n", args.input);
    RPT("Total clauses:  %u (train %u / test %u)\n\n", total, train_n, test_n);

    morpheme_init();
    SpatialAI* ai = NULL;
    double t_train = 0;
    double train_cps = 0;

    /* ── LOAD or TRAIN ── */
    if (args.load_path) {
        RPT("[MODE] Loading pre-trained model: %s\n", args.load_path);
        SpaiStatus st = 0;
        ai = ai_load(args.load_path, &st);
        if (!ai) {
            RPT("ERROR: load failed (%s)\n", spai_status_str(st));
            fclose(rpt);
            return 3;
        }
        RPT("  KF: %u  Delta: %u\n\n", ai->kf_count, ai->df_count);
    } else {
        RPT("[MODE] Training from scratch\n");
        ai = spatial_ai_create();
        double t0 = now_sec();
        for (uint32_t i = 0; i < train_n; i++) {
            ai_store_auto(ai, clauses[i], NULL);
        }
        t_train = now_sec() - t0;
        train_cps = train_n / t_train;
        RPT("  Trained: %u clauses in %.1f s (%.0f c/s)\n", train_n, t_train, train_cps);
        RPT("  KF: %u  Delta: %u\n\n", ai->kf_count, ai->df_count);
    }

    float delta_ratio = (ai->kf_count + ai->df_count > 0)
        ? (float)ai->df_count / (float)(ai->kf_count + ai->df_count) * 100.0f : 0;

    /* ================================================================
     *  1. MATCHING / RETRIEVAL
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  1. Matching & Retrieval\n");
    RPT("----------------------------------------------------------------\n");

    double t_match_start = now_sec();
    uint32_t match_count = 0;
    float sim_sum = 0, sim_min = 1.0f, sim_max = 0.0f;
    uint32_t hits_90 = 0, hits_50 = 0, hits_10 = 0;
    uint32_t test_limit = (test_n < 500) ? test_n : 500;

    for (uint32_t i = 0; i < test_limit; i++) {
        float sim = 0;
        uint32_t mid = ai_predict(ai, clauses[train_n + i], &sim);
        if (mid != UINT32_MAX) {
            match_count++;
            sim_sum += sim;
            if (sim < sim_min) sim_min = sim;
            if (sim > sim_max) sim_max = sim;
            if (sim >= 0.90f) hits_90++;
            if (sim >= 0.50f) hits_50++;
            if (sim >= 0.10f) hits_10++;
        }
    }
    double t_match = now_sec() - t_match_start;
    float sim_avg = match_count ? sim_sum / match_count : 0;
    double ms_query = test_limit ? t_match / test_limit * 1000 : 0;

    RPT("  Unseen queries:    %u\n", test_limit);
    RPT("  Avg similarity:    %.1f%%\n", sim_avg * 100);
    RPT("  Min / Max:         %.1f%% / %.1f%%\n", sim_min*100, sim_max*100);
    RPT("  Hits >= 90%%:       %u (%.1f%%)\n", hits_90, 100.0*hits_90/test_limit);
    RPT("  Hits >= 50%%:       %u (%.1f%%)\n", hits_50, 100.0*hits_50/test_limit);
    RPT("  Hits >= 10%%:       %u (%.1f%%)\n", hits_10, 100.0*hits_10/test_limit);
    RPT("  ms/query:          %.2f\n", ms_query);
    RPT("  >>> GRADE: %s\n\n", grade_match(sim_avg));

    /* Self-recall */
    uint32_t recall_n = (train_n < 200) ? train_n : 200;
    uint32_t recall_90 = 0;
    double t_recall_start = now_sec();
    for (uint32_t i = 0; i < recall_n; i++) {
        uint32_t idx = (i * 37) % train_n;
        float sim = 0;
        ai_predict(ai, clauses[idx], &sim);
        if (sim >= 0.90f) recall_90++;
    }
    double t_recall = now_sec() - t_recall_start;
    float recall_pct = 100.0f * recall_90 / recall_n;
    RPT("  Self-Recall@1:     %u/%u (%.1f%%)\n", recall_90, recall_n, recall_pct);
    RPT("  >>> GRADE: %s\n\n", grade_recall(recall_pct));

    /* ================================================================
     *  2. WORD PREDICTION (A-channel argmax)
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  2. Word Prediction (A-channel Top-1)\n");
    RPT("     Baseline: random = 0.4%% (1/256)\n");
    RPT("----------------------------------------------------------------\n");

    /* Build aggregated A table across all keyframes (heap — 512KB) */
    double* agg_A = (double*)calloc(GRID_TOTAL, sizeof(double));
    if (!agg_A) { RPT("ERROR: alloc agg_A\n"); fclose(rpt); return 4; }
    for (uint32_t k = 0; k < ai->kf_count; k++) {
        const SpatialGrid* g = &ai->keyframes[k].grid;
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            agg_A[i] += (double)g->A[i];
        }
    }

    uint32_t wp_tested = 0, wp_top1 = 0, wp_top3 = 0;
    double t_wp_start = now_sec();

    for (uint32_t ti = 0; ti < test_limit && wp_tested < WORD_PRED_TESTS; ti++) {
        const char* clause = clauses[train_n + ti];
        const uint8_t* bytes = (const uint8_t*)clause;
        uint32_t clen = (uint32_t)strlen(clause);
        if (clen < 20) continue;

        /* Find a word to mask */
        uint32_t ws = 0, we = 0;
        for (uint32_t p = 5; p < clen - 5; p++) {
            if (clause[p] == ' ') {
                ws = p + 1;
                for (we = ws; we < clen && clause[we] != ' '; we++);
                if (we - ws >= 3 && we - ws <= 15) break;
                ws = 0;
            }
        }
        if (ws == 0 || we == 0 || we <= ws) continue;

        uint32_t wlen = we - ws;
        int all_match = 1;
        int any_in_top3 = 1;

        for (uint32_t bi = 0; bi < wlen; bi++) {
            uint32_t y = (ws + bi) % GRID_SIZE;
            uint8_t actual = bytes[ws + bi];

            /* Top-1 and Top-3 by A value at this row */
            uint8_t top_bytes[3] = {0};
            double top_scores[3] = {0};

            for (uint32_t x = 0; x < GRID_SIZE; x++) {
                double a = agg_A[y * GRID_SIZE + x];
                if (a <= 0) continue;
                for (int k = 0; k < 3; k++) {
                    if (a > top_scores[k]) {
                        for (int j = 2; j > k; j--) {
                            top_bytes[j] = top_bytes[j-1];
                            top_scores[j] = top_scores[j-1];
                        }
                        top_bytes[k] = (uint8_t)x;
                        top_scores[k] = a;
                        break;
                    }
                }
            }

            if (top_bytes[0] != actual) all_match = 0;
            if (top_bytes[0] != actual && top_bytes[1] != actual && top_bytes[2] != actual)
                any_in_top3 = 0;
        }

        if (all_match) wp_top1++;
        if (any_in_top3) wp_top3++;
        wp_tested++;
    }
    double t_wp = now_sec() - t_wp_start;
    float wp1_pct = wp_tested ? 100.0f * wp_top1 / wp_tested : 0;
    float wp3_pct = wp_tested ? 100.0f * wp_top3 / wp_tested : 0;

    RPT("  Words tested:      %u\n", wp_tested);
    RPT("  Top-1 (all bytes): %u (%.1f%%)\n", wp_top1, wp1_pct);
    RPT("  Top-3 (any byte):  %u (%.1f%%)\n", wp_top3, wp3_pct);
    RPT("  ms/word:           %.2f\n", wp_tested ? t_wp/wp_tested*1000 : 0);
    RPT("  >>> GRADE: %s\n\n", grade_word(wp1_pct));

    /* ================================================================
     *  3. TEXT GENERATION
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  3. Text Generation\n");
    RPT("----------------------------------------------------------------\n");

    uint32_t gen_tested = 0, gen_valid = 0, gen_nonempty = 0;
    uint32_t gen_total_len = 0;
    char gen_out[GEN_TESTS][512];
    uint32_t gen_unique = 0;
    double t_gen_start = now_sec();

    for (uint32_t i = 0; i < test_limit && gen_tested < GEN_TESTS; i++) {
        char out[512] = {0};
        float sim = 0;
        uint32_t len = ai_generate_next(ai, clauses[train_n + i], out, 511, &sim);
        if (len > 0) {
            gen_nonempty++;
            gen_total_len += len;
            if (is_valid_utf8(out, len)) gen_valid++;
        }
        strncpy(gen_out[gen_tested], out, 511);
        gen_tested++;
    }
    double t_gen = now_sec() - t_gen_start;

    for (uint32_t i = 0; i < gen_tested; i++) {
        int dup = 0;
        for (uint32_t j = 0; j < i; j++) {
            if (strcmp(gen_out[i], gen_out[j]) == 0) { dup = 1; break; }
        }
        if (!dup && gen_out[i][0]) gen_unique++;
    }

    float utf8_pct = gen_nonempty ? 100.0f * gen_valid / gen_nonempty : 0;
    float uniq_pct = gen_nonempty ? 100.0f * gen_unique / gen_nonempty : 0;
    float avg_len = gen_nonempty ? (float)gen_total_len / gen_nonempty : 0;

    RPT("  Prompts:           %u\n", gen_tested);
    RPT("  Non-empty:         %u (%.1f%%)\n", gen_nonempty, 100.0*gen_nonempty/gen_tested);
    RPT("  Valid UTF-8:       %u (%.1f%%)\n", gen_valid, utf8_pct);
    RPT("  Unique outputs:    %u (%.1f%%)\n", gen_unique, uniq_pct);
    RPT("  Avg length:        %.1f bytes\n", avg_len);
    RPT("  ms/gen:            %.2f\n", gen_tested ? t_gen/gen_tested*1000 : 0);
    RPT("  >>> GRADE: %s\n\n", grade_gen(utf8_pct, uniq_pct));

    /* Sample outputs */
    RPT("  Samples:\n");
    int shown = 0;
    for (uint32_t i = 0; i < gen_tested && shown < 5; i++) {
        if (gen_out[i][0]) {
            RPT("    [%u] \"%.80s\"\n", i, gen_out[i]);
            shown++;
        }
    }
    RPT("\n");

    /* ================================================================
     *  4. SPATIAL / RGB ANALYSIS
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  4. Spatial & RGB Channel Analysis\n");
    RPT("----------------------------------------------------------------\n");

    double r_sum = 0, g_sum = 0, b_sum = 0;
    double r_sq = 0, g_sq = 0, b_sq = 0;
    uint32_t rgb_n = 0, total_active = 0;
    uint16_t a_max_val = 0;
    double a_sum_all = 0;

    for (uint32_t k = 0; k < ai->kf_count; k++) {
        const SpatialGrid* g = &ai->keyframes[k].grid;
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            if (g->A[i] == 0) continue;
            total_active++;
            if (g->A[i] > a_max_val) a_max_val = g->A[i];
            a_sum_all += g->A[i];
            double r = g->R[i], gv = g->G[i], b = g->B[i];
            r_sum += r; g_sum += gv; b_sum += b;
            r_sq += r*r; g_sq += gv*gv; b_sq += b*b;
            rgb_n++;
        }
    }

    double r_mean = rgb_n ? r_sum/rgb_n : 0;
    double g_mean = rgb_n ? g_sum/rgb_n : 0;
    double b_mean = rgb_n ? b_sum/rgb_n : 0;
    double r_std = rgb_n ? sqrt(fabs(r_sq/rgb_n - r_mean*r_mean)) : 0;
    double g_std = rgb_n ? sqrt(fabs(g_sq/rgb_n - g_mean*g_mean)) : 0;
    double b_std = rgb_n ? sqrt(fabs(b_sq/rgb_n - b_mean*b_mean)) : 0;

    uint32_t ema_active = 0;
    double ema_max = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (ai->ema_count[i] > 0) ema_active++;
        if (ai->ema_count[i] > ema_max) ema_max = ai->ema_count[i];
    }

    const char* rgb_grade = (r_std > 20 && g_std > 15 && b_std > 20) ? "GOOD" :
        (r_std > 10 || g_std > 10 || b_std > 10) ? "PARTIAL" : "WEAK";

    RPT("  Active cells:      %u\n", total_active);
    RPT("  A max / mean:      %u / %.2f\n", a_max_val, total_active ? a_sum_all/total_active : 0);
    RPT("  R mean/std:        %.1f / %.1f\n", r_mean, r_std);
    RPT("  G mean/std:        %.1f / %.1f\n", g_mean, g_std);
    RPT("  B mean/std:        %.1f / %.1f\n", b_mean, b_std);
    RPT("  EMA active:        %u (max count %.0f)\n", ema_active, ema_max);
    RPT("  Weights:           A=%.3f R=%.3f G=%.3f B=%.3f\n",
        ai->global_weights.w_A, ai->global_weights.w_R,
        ai->global_weights.w_G, ai->global_weights.w_B);
    RPT("  >>> RGB GRADE: %s\n\n", rgb_grade);

    /* ================================================================
     *  5. SPEED SUMMARY
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  5. Speed\n");
    RPT("----------------------------------------------------------------\n");

    double t_enc_start = now_sec();
    uint32_t enc_n = (train_n < 1000) ? train_n : 1000;
    for (uint32_t i = 0; i < enc_n; i++) {
        SpatialGrid* g = grid_create();
        layers_encode_clause(clauses[i], NULL, g);
        update_rgb_directional(g);
        grid_destroy(g);
    }
    double t_enc = now_sec() - t_enc_start;
    double enc_cps = enc_n / t_enc;

    RPT("  Encoding:          %.2f ms/clause (%.0f c/s) %s\n",
        t_enc/enc_n*1000, enc_cps, grade_speed(enc_cps));
    RPT("  Matching:          %.2f ms/query\n", ms_query);
    RPT("  Generation:        %.2f ms/gen\n", gen_tested ? t_gen/gen_tested*1000 : 0);
    if (t_train > 0)
        RPT("  Training:          %.2f ms/clause (%.0f c/s) %s\n",
            t_train/train_n*1000, train_cps, grade_speed(train_cps));
    RPT("\n");

    /* ================================================================
     *  6. RESOURCE USAGE
     * ================================================================ */
    RPT("----------------------------------------------------------------\n");
    RPT("  6. Resource Usage\n");
    RPT("----------------------------------------------------------------\n");

    double mem = get_memory_mb();
    long model_sz = args.load_path ? file_size_bytes(args.load_path) : 0;
    uint32_t total_clauses = ai->kf_count + ai->df_count;
    float kb_per_clause = total_clauses ? (float)(model_sz / 1024) / total_clauses : 0;

    RPT("  Memory:            %.1f MB\n", mem);
    RPT("  Model file:        %ld MB\n", model_sz / (1024*1024));
    RPT("  KF count:          %u\n", ai->kf_count);
    RPT("  Delta count:       %u\n", ai->df_count);
    RPT("  Delta ratio:       %.1f%%\n", delta_ratio);
    RPT("  KB/clause:         %.1f\n", kb_per_clause);
    RPT("  >>> GRADE: %s\n\n", grade_efficiency(kb_per_clause, delta_ratio));

    /* ================================================================
     *  FINAL SCORECARD
     * ================================================================ */
    RPT("================================================================\n");
    RPT("  SCORECARD\n");
    RPT("================================================================\n");
    RPT("  Matching:       %.1f%% avg sim      %s\n", sim_avg*100, grade_match(sim_avg));
    RPT("  Self-Recall:    %.1f%%              %s\n", recall_pct, grade_recall(recall_pct));
    RPT("  Word Top-1:     %.1f%%              %s\n", wp1_pct, grade_word(wp1_pct));
    RPT("  Generation:     UTF8=%.0f%% Uniq=%.0f%%  %s\n", utf8_pct, uniq_pct, grade_gen(utf8_pct, uniq_pct));
    RPT("  RGB Channels:                      %s\n", rgb_grade);
    RPT("  Efficiency:     %.0f KB/cl Delta=%.0f%%  %s\n", kb_per_clause, delta_ratio, grade_efficiency(kb_per_clause, delta_ratio));
    RPT("================================================================\n");

    free(agg_A);
    spatial_ai_destroy(ai);
    free_clauses(clauses, total);
    fclose(rpt);
    printf("\nReport: %s\n", args.report_path);
    return 0;
}
