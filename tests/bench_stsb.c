/*
 * bench_stsb.c — STS-B sentence similarity benchmark
 *
 * Input TSV format (one pair per line):
 *   <score>\t<sentence1>\t<sentence2>
 *
 *   score       float in [0, 5]   (human-labeled similarity)
 *   sentence1   UTF-8 text
 *   sentence2   UTF-8 text
 *
 * Pipeline per pair:
 *   each sentence → 3-layer encode → RGB directional update
 *   engine_sim = cosine_rgb_weighted(s1, s2)
 *
 * Metrics:
 *   Pearson correlation   (linear)
 *   Spearman correlation  (rank-based, tie-averaged)
 *   mean abs error on normalized scales
 *
 * Usage:
 *   ./build/bench_stsb data/stsb.tsv
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "bench_utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_PAIRS     16384
#define MAX_LINE_LEN  4096

typedef struct {
    float  gold;       /* 0..5 */
    float  engine;     /* 0..1 */
    char*  s1;
    char*  s2;
} Pair;

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── TSV parsing ── */

/* Parse one line: "<score>\t<s1>\t<s2>".
   Returns 1 on success, 0 on skip. */
static int parse_line(const char* line, Pair* out) {
    const char* p = line;

    /* score */
    char* endp = NULL;
    float score = strtof(p, &endp);
    if (endp == p) return 0;
    if (*endp != '\t') return 0;

    /* s1 */
    const char* s1_start = endp + 1;
    const char* tab2 = strchr(s1_start, '\t');
    if (!tab2) return 0;
    size_t s1_len = (size_t)(tab2 - s1_start);

    /* s2 */
    const char* s2_start = tab2 + 1;
    size_t s2_len = strlen(s2_start);
    while (s2_len > 0 && (s2_start[s2_len - 1] == '\n' ||
                          s2_start[s2_len - 1] == '\r' ||
                          s2_start[s2_len - 1] == ' ')) {
        s2_len--;
    }

    if (s1_len < 2 || s2_len < 2) return 0;

    out->gold = score;
    out->s1 = (char*)malloc(s1_len + 1);
    out->s2 = (char*)malloc(s2_len + 1);
    if (!out->s1 || !out->s2) return 0;
    memcpy(out->s1, s1_start, s1_len); out->s1[s1_len] = '\0';
    memcpy(out->s2, s2_start, s2_len); out->s2[s2_len] = '\0';
    return 1;
}

/* ── Correlation ── */

static double pearson(const double* x, const double* y, int n) {
    if (n < 2) return 0.0;
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;

    double num = 0, dx = 0, dy = 0;
    for (int i = 0; i < n; i++) {
        double a = x[i] - mx, b = y[i] - my;
        num += a * b;
        dx  += a * a;
        dy  += b * b;
    }
    if (dx == 0 || dy == 0) return 0.0;
    return num / sqrt(dx * dy);
}

/* Compute ranks with tie averaging.
   Fills out[] with ranks [1..n]. */
typedef struct { double v; int idx; } RankItem;

static int cmp_rank(const void* a, const void* b) {
    double va = ((const RankItem*)a)->v;
    double vb = ((const RankItem*)b)->v;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

static void rankify(const double* vals, double* out, int n) {
    RankItem* r = (RankItem*)malloc((size_t)n * sizeof(RankItem));
    for (int i = 0; i < n; i++) { r[i].v = vals[i]; r[i].idx = i; }
    qsort(r, (size_t)n, sizeof(RankItem), cmp_rank);

    /* Average ranks for ties */
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && r[j + 1].v == r[i].v) j++;
        double avg_rank = ((double)(i + 1) + (double)(j + 1)) * 0.5;
        for (int k = i; k <= j; k++) out[r[k].idx] = avg_rank;
        i = j + 1;
    }
    free(r);
}

static double spearman(const double* x, const double* y, int n) {
    double* rx = (double*)malloc((size_t)n * sizeof(double));
    double* ry = (double*)malloc((size_t)n * sizeof(double));
    rankify(x, rx, n);
    rankify(y, ry, n);
    double r = pearson(rx, ry, n);
    free(rx); free(ry);
    return r;
}

/* ── Main ── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <stsb.tsv> [max_pairs]\n"
            "\n"
            "  TSV format:  <score>\\t<sentence1>\\t<sentence2>\n"
            "  score: float in [0,5]\n"
            "\n"
            "Example:\n"
            "  %s data/stsb.tsv\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    int max_pairs = (argc >= 3) ? atoi(argv[2]) : MAX_PAIRS;
    if (max_pairs > MAX_PAIRS) max_pairs = MAX_PAIRS;

    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", filepath);
        return 1;
    }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  STS-B Benchmark\n");
    printf("========================================\n");
    printf("  File:       %s\n", filepath);
    printf("  Max pairs:  %d\n", max_pairs);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* Load pairs */
    printf("[1/3] Loading sentence pairs...\n");
    double t0 = now_sec();

    Pair* pairs = (Pair*)calloc((size_t)max_pairs, sizeof(Pair));
    char line[MAX_LINE_LEN];
    int n = 0;

    while (fgets(line, sizeof(line), fp) && n < max_pairs) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (parse_line(line, &pairs[n])) n++;
    }
    fclose(fp);

    printf("  Loaded: %d pairs (%.2f sec)\n\n", n, now_sec() - t0);

    if (n < 2) {
        fprintf(stderr, "ERROR: need at least 2 pairs\n");
        return 1;
    }

    /* Compute engine similarity per pair */
    printf("[2/3] Encoding + computing cosine similarity...\n");
    t0 = now_sec();

    double* gold   = (double*)malloc((size_t)n * sizeof(double));
    double* engine = (double*)malloc((size_t)n * sizeof(double));

    SpatialGrid* g1 = grid_create();
    SpatialGrid* g2 = grid_create();

    for (int i = 0; i < n; i++) {
        grid_clear(g1);
        grid_clear(g2);
        layers_encode_clause(pairs[i].s1, NULL, g1);
        layers_encode_clause(pairs[i].s2, NULL, g2);
        update_rgb_directional(g1);
        update_rgb_directional(g2);

        float sim = cosine_rgb_weighted(g1, g2);
        pairs[i].engine = sim;
        gold[i]   = (double)pairs[i].gold;
        engine[i] = (double)sim;

        if ((i + 1) % 100 == 0 || i + 1 == n) {
            printf("\r  Processed: %d / %d", i + 1, n);
            fflush(stdout);
        }
    }

    grid_destroy(g1);
    grid_destroy(g2);

    double t_encode = now_sec() - t0;
    printf("\n  Done in %.2f sec (%.0f pairs/sec)\n\n", t_encode, n / t_encode);

    /* Correlations */
    printf("[3/3] Computing correlations...\n\n");

    double r_pearson  = pearson(gold, engine, n);
    double r_spearman = spearman(gold, engine, n);

    /* MAE on normalized scales: gold/5 vs engine (already 0..1) */
    double mae = 0.0;
    double bin_totals[6] = {0}; /* gold in [0,1),[1,2),... */
    double bin_engine[6] = {0};
    int    bin_count[6]  = {0};
    for (int i = 0; i < n; i++) {
        double g_norm = gold[i] / 5.0;
        mae += fabs(g_norm - engine[i]);
        int b = (int)gold[i];
        if (b < 0) b = 0;
        if (b > 5) b = 5;
        bin_totals[b] += gold[i];
        bin_engine[b] += engine[i];
        bin_count[b]++;
    }
    mae /= n;

    /* Quartile analysis */
    int q_low = 0, q_mid = 0, q_high = 0;
    for (int i = 0; i < n; i++) {
        if (gold[i] < 1.67) q_low++;
        else if (gold[i] < 3.33) q_mid++;
        else q_high++;
    }

    /* ── Report ── */
    printf("========================================\n");
    printf("  RESULTS\n");
    printf("========================================\n\n");

    printf("  Pairs evaluated:  %d\n", n);
    printf("  Encode speed:     %.0f pairs/sec\n\n", n / t_encode);

    printf("  CORRELATION (gold vs engine similarity)\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Pearson  r:       %+.4f\n", r_pearson);
    printf("  Spearman rho:     %+.4f\n", r_spearman);
    printf("  MAE (normalized): %.4f\n\n", mae);

    printf("  GOLD DISTRIBUTION\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Low    [0, 1.67):  %d  (%.1f%%)\n", q_low,  100.0 * q_low  / n);
    printf("  Mid   [1.67, 3.33): %d  (%.1f%%)\n", q_mid,  100.0 * q_mid  / n);
    printf("  High   [3.33, 5]:   %d  (%.1f%%)\n\n", q_high, 100.0 * q_high / n);

    printf("  PER-BIN AVG ENGINE SIMILARITY\n");
    printf("  ─────────────────────────────────────\n");
    for (int b = 0; b <= 5; b++) {
        if (bin_count[b] == 0) continue;
        double avg_gold   = bin_totals[b] / bin_count[b];
        double avg_engine = bin_engine[b] / bin_count[b];
        int bar = (int)(avg_engine * 40.0);
        printf("  gold %d..%d (n=%3d) | gold=%.2f eng=%.2f | ",
               b, b + 1, bin_count[b], avg_gold, avg_engine);
        for (int k = 0; k < bar; k++) printf("#");
        printf("\n");
    }
    printf("\n");

    /* Interpretation */
    printf("  INTERPRETATION\n");
    printf("  ─────────────────────────────────────\n");
    if (r_spearman > 0.5)
        printf("  Strong positive rank correlation.\n");
    else if (r_spearman > 0.3)
        printf("  Moderate positive rank correlation.\n");
    else if (r_spearman > 0.1)
        printf("  Weak positive rank correlation.\n");
    else
        printf("  No meaningful correlation (baseline-level).\n");
    printf("\n");

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    /* Cleanup */
    for (int i = 0; i < n; i++) {
        free(pairs[i].s1);
        free(pairs[i].s2);
    }
    free(pairs);
    free(gold);
    free(engine);

    return 0;
}
