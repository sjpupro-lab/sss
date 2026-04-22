/*
 * bench_perplexity.c — byte-level perplexity benchmark
 *
 * SPEC.md §2-3:  grid[y][x] accumulates 3-layer weights where X = byte value,
 * Y = position in stream. The aggregated distribution over all training
 * keyframes gives a position-dependent byte model:
 *
 *     count[y, v] = sum_k  keyframe_k.A[y, v]
 *     p(byte=v | pos y) = (count[y, v] + epsilon) /
 *                         (sum_x count[y, x] + 256*epsilon)
 *
 * Perplexity is computed over the test-set byte sequence:
 *
 *     log_p_sum = sum_i log p(byte_i | y_i = i % 256)
 *     PPL       = exp(-log_p_sum / N)
 *
 * Range: [1, 256]. Lower is better. Random byte source ≈ 256.
 *
 * Usage:
 *   ./build/bench_perplexity data/sample_ko.txt [max_clauses]
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "bench_utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CLAUSES     20000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10
#define DEFAULT_LIMIT   2000
#define TEST_SPLIT      0.10f   /* 10% held out */
#define EPSILON         0.5     /* Laplace smoothing */

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Strip wiki meta lines */
static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] == '<') return 1;
    if (line[0] == '\0') return 1;
    return 0;
}

static void trim_trailing(char* s) {
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r' ||
                       s[len-1] == ' '  || s[len-1] == '\t')) {
        s[--len] = '\0';
    }
}

/* Split by sentence-ending punctuation */
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
    return count;
}

int main(int argc, char* argv[]) {
    utf8_console_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <text_file> [max_clauses]\n"
            "\n"
            "Example:\n"
            "  %s data/sample_ko.txt\n"
            "  %s data/sample_en.txt 1000\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    uint32_t max_clauses = (argc >= 3) ? (uint32_t)atoi(argv[2]) : DEFAULT_LIMIT;
    if (max_clauses > MAX_CLAUSES) max_clauses = MAX_CLAUSES;

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", filepath);
        return 1;
    }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  Perplexity Bench\n");
    printf("========================================\n");
    printf("  File:        %s\n", filepath);
    printf("  Max clauses: %u\n", max_clauses);
    printf("  Test split:  %.0f%%\n", TEST_SPLIT * 100.0f);
    printf("  Smoothing:   Laplace (eps=%.1f)\n", EPSILON);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* ── Phase 1: Load ── */
    printf("[1/4] Loading clauses...\n");
    double t0 = now_sec();

    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    uint32_t clause_count = 0;
    char line[MAX_LINE_LEN];
    char split_buf[8][MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && clause_count < max_clauses) {
        trim_trailing(line);
        if (is_meta_line(line)) continue;
        if ((int)strlen(line) < MIN_CLAUSE_LEN) continue;

        uint32_t k = split_clauses(line, split_buf, 8);
        for (uint32_t i = 0; i < k && clause_count < max_clauses; i++) {
            memcpy(clauses[clause_count], split_buf[i], MAX_LINE_LEN);
            clause_count++;
        }
    }
    fclose(fp);

    printf("  Loaded: %u clauses (%.2f sec)\n\n", clause_count, now_sec() - t0);

    if (clause_count < 20) {
        fprintf(stderr, "ERROR: need at least 20 clauses\n");
        free(clauses);
        return 1;
    }

    /* ── Phase 2: Split train/test ── */
    uint32_t test_count  = (uint32_t)(clause_count * TEST_SPLIT);
    uint32_t train_count = clause_count - test_count;
    if (test_count < 5) test_count = 5;
    if (train_count < 10) train_count = 10;
    if (train_count + test_count > clause_count)
        train_count = clause_count - test_count;

    printf("[2/4] Training on %u clauses...\n", train_count);
    t0 = now_sec();

    /* Build aggregated byte distribution from training keyframes.
       count[y][x] = sum over train clauses of (3-layer combined) A channel. */
    double (*count)[256] = calloc(256 * 256, sizeof(double));
    if (!count) {
        fprintf(stderr, "ERROR: cannot allocate count table\n");
        free(clauses);
        return 1;
    }

    SpatialGrid* work = grid_create();
    for (uint32_t i = 0; i < train_count; i++) {
        grid_clear(work);
        layers_encode_clause(clauses[i], NULL, work);
        for (uint32_t y = 0; y < 256; y++) {
            for (uint32_t x = 0; x < 256; x++) {
                count[y][x] += (double)work->A[y * 256 + x];
            }
        }
        if ((i + 1) % 500 == 0 || i + 1 == train_count) {
            printf("\r  Processed: %u / %u", i + 1, train_count);
            fflush(stdout);
        }
    }
    grid_destroy(work);

    /* Row totals for normalization */
    double row_total[256];
    for (uint32_t y = 0; y < 256; y++) {
        double s = 0.0;
        for (uint32_t x = 0; x < 256; x++) s += count[y][x];
        row_total[y] = s;
    }

    double t_train = now_sec() - t0;
    printf("\n  Done in %.2f sec\n\n", t_train);

    /* ── Phase 3: Evaluate PPL on test set ── */
    printf("[3/4] Evaluating on %u held-out clauses...\n", test_count);
    t0 = now_sec();

    double log_p_sum = 0.0;
    uint64_t total_bytes = 0;
    uint64_t unseen_bytes = 0;

    /* Per-byte-range breakdown */
    double ascii_logp = 0.0;  uint64_t ascii_cnt = 0;
    double korean_logp = 0.0; uint64_t korean_cnt = 0;
    double other_logp = 0.0;  uint64_t other_cnt = 0;

    for (uint32_t ci = train_count; ci < train_count + test_count; ci++) {
        const uint8_t* bytes = (const uint8_t*)clauses[ci];
        uint32_t len = (uint32_t)strlen((const char*)bytes);

        for (uint32_t i = 0; i < len; i++) {
            uint8_t v = bytes[i];
            uint32_t y = i % 256;
            double c_vy = count[y][v];
            double total = row_total[y];
            double p = (c_vy + EPSILON) / (total + 256.0 * EPSILON);
            double lp = log(p);

            log_p_sum += lp;
            total_bytes++;
            if (c_vy == 0.0) unseen_bytes++;

            if (v < 0x80) {
                ascii_logp += lp; ascii_cnt++;
            } else if (v >= 0xE0 && v <= 0xEF) {
                /* 3-byte UTF-8 leader (Korean/CJK) */
                korean_logp += lp; korean_cnt++;
            } else {
                other_logp += lp; other_cnt++;
            }
        }
    }

    double t_eval = now_sec() - t0;
    printf("  Done in %.2f sec (%.0f bytes/sec)\n\n",
           t_eval, total_bytes / t_eval);

    /* ── Phase 4: Report ── */
    printf("[4/4] Results\n");
    printf("========================================\n\n");

    double avg_nll = -log_p_sum / (double)total_bytes;
    double ppl = exp(avg_nll);
    double entropy_bits = avg_nll / log(2.0);

    printf("  TEST SET\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Test clauses:       %u\n", test_count);
    printf("  Total bytes:        %llu\n", (unsigned long long)total_bytes);
    printf("  Unseen byte events: %llu  (%.2f%%)\n",
           (unsigned long long)unseen_bytes,
           100.0 * unseen_bytes / total_bytes);
    printf("\n");

    printf("  PERPLEXITY  (byte-level)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Avg NLL:            %.4f nats/byte\n", avg_nll);
    printf("  Entropy:            %.4f bits/byte\n", entropy_bits);
    printf("  Perplexity:         %.2f\n", ppl);
    printf("  (uniform 256-byte baseline = 256.00)\n\n");

    printf("  PER-BYTE-RANGE PERPLEXITY\n");
    printf("  ─────────────────────────────────────\n");
    if (ascii_cnt > 0) {
        double p = exp(-ascii_logp / (double)ascii_cnt);
        printf("  ASCII   (< 0x80)  n=%llu  PPL=%.2f\n",
               (unsigned long long)ascii_cnt, p);
    }
    if (korean_cnt > 0) {
        double p = exp(-korean_logp / (double)korean_cnt);
        printf("  CJK UTF-8 leaders n=%llu  PPL=%.2f\n",
               (unsigned long long)korean_cnt, p);
    }
    if (other_cnt > 0) {
        double p = exp(-other_logp / (double)other_cnt);
        printf("  Other UTF-8 bytes n=%llu  PPL=%.2f\n",
               (unsigned long long)other_cnt, p);
    }
    printf("\n");

    /* Training stats */
    double total_mass = 0.0;
    uint32_t active_cells = 0;
    for (uint32_t y = 0; y < 256; y++) {
        for (uint32_t x = 0; x < 256; x++) {
            total_mass += count[y][x];
            if (count[y][x] > 0) active_cells++;
        }
    }
    printf("  TRAINING DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Total mass:         %.0f\n", total_mass);
    printf("  Active cells:       %u / 65536  (%.1f%%)\n",
           active_cells, 100.0 * active_cells / 65536.0);
    printf("  Train clauses:      %u  (%.1f sec)\n", train_count, t_train);
    printf("\n");

    /* Interpretation */
    printf("  INTERPRETATION\n");
    printf("  ─────────────────────────────────────\n");
    if (ppl < 20.0)
        printf("  Strong language model (PPL < 20).\n");
    else if (ppl < 60.0)
        printf("  Useful byte model (PPL < 60).\n");
    else if (ppl < 128.0)
        printf("  Better than random (PPL < 128).\n");
    else
        printf("  Near-random (increase train data).\n");
    printf("\n");

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    free(count);
    free(clauses);
    return 0;
}
