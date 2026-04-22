/*
 * stream_train.c — streaming training for SPATIAL-PATTERN-AI
 *
 * Reads a large text corpus line-by-line (one clause per line, as in
 * wiki-extractor output) and calls ai_store_auto() per clause without
 * ever holding the full file in memory.
 *
 * Usage:
 *   ./build/stream_train --input <path>
 *                        [--max <N>]          (default 50000)
 *                        [--save <path>]      (default build/models/stream_auto.spai)
 *                        [--checkpoint <N>]   (default 5000, 0 disables)
 *                        [--verbose]          (per-clause progress line)
 *                        [--verify]           (unseen-query sanity pass)
 *
 * The binary itself uses ~4 KB of line buffer + whatever the SpatialAI
 * keyframe/delta store accumulates — no full-file buffering.
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <unistd.h>
#  define MKDIR(p) mkdir((p), 0755)
#endif

#define LINE_BUF        4096
#define MIN_CLAUSE_LEN  10
#define DEFAULT_MAX     50000
#define DEFAULT_CKPT    5000
#define VERIFY_PROBE    500

/* ── CLI ──────────────────────────────────────────────────── */

typedef struct {
    const char* input;
    const char* save;
    const char* event_log;       /* --log <path>; NULL = disabled */
    uint32_t    max_clauses;
    uint32_t    checkpoint;
    int         verbose;
    int         verify;
    float       threshold;       /* <0 means "leave engine default" */
    float       target_delta;    /* <0 means "disabled" (explicit threshold wins) */
    uint32_t    calibrate_samples;
} StreamArgs;

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s --input <path> [--max N] [--save path] [--checkpoint N]\n"
        "       [--log path] [--threshold F | --target-delta R] [--verbose] [--verify]\n"
        "\n"
        "  --input <path>        input text file, one clause per line (required)\n"
        "  --max <N>             max clauses to ingest (default %d)\n"
        "  --save <path>         final model path (default build/models/stream_auto.spai)\n"
        "  --checkpoint <N>      save every N clauses (default %d, 0 disables)\n"
        "  --log <path>          emit a binary training-event log consumed by\n"
        "                        tools/animate_training.py (clause-level cell\n"
        "                        events for the twinkling-grid visualization)\n"
        "  --threshold <F>       delta decision threshold in [0, 1]\n"
        "                        (default 0.30; try 0.15 on wiki-style corpora)\n"
        "  --target-delta <R>    auto-calibrate threshold so about R (0..1) of the\n"
        "                        first --calibrate-samples clauses would become\n"
        "                        deltas. Ignored when --threshold is given explicitly.\n"
        "  --calibrate-samples <N>\n"
        "                        clauses used for --target-delta calibration\n"
        "                        (default 500; capped at --max)\n"
        "  --verbose             per-clause progress line\n"
        "  --verify              after training, run an unseen-query sanity pass\n",
        prog, DEFAULT_MAX, DEFAULT_CKPT);
}

static int parse_args(int argc, char** argv, StreamArgs* a) {
    a->input             = NULL;
    a->save              = "build/models/stream_auto.spai";
    a->event_log         = NULL;
    a->max_clauses       = DEFAULT_MAX;
    a->checkpoint        = DEFAULT_CKPT;
    a->verbose           = 0;
    a->verify            = 0;
    a->threshold         = -1.0f;
    a->target_delta      = -1.0f;
    a->calibrate_samples = 500;

    for (int i = 1; i < argc; i++) {
        const char* k = argv[i];
        if (strcmp(k, "--input") == 0 && i + 1 < argc) {
            a->input = argv[++i];
        } else if (strcmp(k, "--max") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 0) v = 0;
            a->max_clauses = (uint32_t)v;
        } else if (strcmp(k, "--save") == 0 && i + 1 < argc) {
            a->save = argv[++i];
        } else if (strcmp(k, "--checkpoint") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 0) v = 0;
            a->checkpoint = (uint32_t)v;
        } else if (strcmp(k, "--log") == 0 && i + 1 < argc) {
            a->event_log = argv[++i];
        } else if (strcmp(k, "--threshold") == 0 && i + 1 < argc) {
            a->threshold = strtof(argv[++i], NULL);
        } else if (strcmp(k, "--target-delta") == 0 && i + 1 < argc) {
            a->target_delta = strtof(argv[++i], NULL);
            if (a->target_delta < 0.0f) a->target_delta = 0.0f;
            if (a->target_delta > 1.0f) a->target_delta = 1.0f;
        } else if (strcmp(k, "--calibrate-samples") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 10) v = 10;
            a->calibrate_samples = (uint32_t)v;
        } else if (strcmp(k, "--verbose") == 0) {
            a->verbose = 1;
        } else if (strcmp(k, "--verify") == 0) {
            a->verify = 1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", k);
            return -1;
        }
    }
    if (!a->input) return -1;
    return 0;
}

/* ── helpers ─────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Strip trailing whitespace in place. */
static void strip_trailing(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' ||
                     s[n-1] == ' '  || s[n-1] == '\t')) {
        s[--n] = '\0';
    }
}

/* Skip obvious non-content lines (empty, XML/HTML-ish). */
static int is_skippable(const char* s) {
    if (s[0] == '\0') return 1;
    if (s[0] == '<')  return 1;            /* <doc>, </doc>, ... */
    if (s[0] == '#')  return 1;            /* comment / markdown */
    return 0;
}

/* Ensure the parent directory of `path` exists. Silently tolerates
 * existing directories. Path must be writable. */
static void ensure_parent_dir(const char* path) {
    if (!path) return;
    char buf[1024];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);

    /* Walk each separator and mkdir on the way. */
    for (size_t i = 1; i < n; i++) {
        if (buf[i] == '/' || buf[i] == '\\') {
            char c = buf[i];
            buf[i] = '\0';
            MKDIR(buf);
            buf[i] = c;
        }
    }
}

static long file_size_or_zero(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fclose(fp);
    return n < 0 ? 0 : n;
}

/* ── verify: unseen-query pass over the last batch from disk ── */

typedef struct {
    uint32_t clauses_scanned;
    uint32_t clauses_matched;
    double   sum_sim;
    double   min_sim;
    double   max_sim;
    uint32_t hits_90;
    uint32_t hits_50;
    uint32_t hits_10;
} VerifyStats;

/* Re-scan the corpus, match each clause against the trained engine,
 * report similarity distribution. Uses ai_predict which also runs the
 * full encode + RGB diffusion path. */
static void verify_run(SpatialAI* ai, const char* input_path,
                       uint32_t probe_limit, VerifyStats* out) {
    memset(out, 0, sizeof(*out));
    out->min_sim = 1.0;
    out->max_sim = 0.0;

    FILE* fp = fopen(input_path, "r");
    if (!fp) {
        printf("[verify] cannot reopen %s\n", input_path);
        return;
    }

    char line[LINE_BUF];
    uint32_t seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        strip_trailing(line);
        if (is_skippable(line)) continue;
        if (strlen(line) < MIN_CLAUSE_LEN) continue;
        seen++;
    }

    /* Probe the last `probe_limit` clauses (treated as "unseen tail"). */
    uint32_t start = (seen > probe_limit) ? (seen - probe_limit) : 0;
    rewind(fp);

    uint32_t idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        strip_trailing(line);
        if (is_skippable(line)) continue;
        if (strlen(line) < MIN_CLAUSE_LEN) continue;
        if (idx++ < start) continue;

        float sim = 0.0f;
        uint32_t kf = ai_predict(ai, line, &sim);
        (void)kf;

        out->clauses_scanned++;
        if (sim > 0.0f) out->clauses_matched++;
        out->sum_sim += sim;
        if (sim < out->min_sim) out->min_sim = sim;
        if (sim > out->max_sim) out->max_sim = sim;
        if (sim >= 0.90f) out->hits_90++;
        if (sim >= 0.50f) out->hits_50++;
        if (sim >= 0.10f) out->hits_10++;

        if (out->clauses_scanned >= probe_limit) break;
    }

    fclose(fp);
}

/* Print summary stats for a trained engine: KF/Delta counts,
 * R/G range over active cells, avg A. */
static void report_engine_stats(const SpatialAI* ai) {
    uint32_t active_cells_total = 0;
    uint64_t a_sum = 0;
    uint8_t  r_min = 255, r_max = 0;
    uint8_t  g_min = 255, g_max = 0;
    uint8_t  b_min = 255, b_max = 0;

    for (uint32_t k = 0; k < ai->kf_count; k++) {
        const SpatialGrid* g = &ai->keyframes[k].grid;
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            if (g->A[i] == 0) continue;
            active_cells_total++;
            a_sum += g->A[i];
            if (g->R[i] < r_min) r_min = g->R[i];
            if (g->R[i] > r_max) r_max = g->R[i];
            if (g->G[i] < g_min) g_min = g->G[i];
            if (g->G[i] > g_max) g_max = g->G[i];
            if (g->B[i] < b_min) b_min = g->B[i];
            if (g->B[i] > b_max) b_max = g->B[i];
        }
    }
    double avg_a = active_cells_total ? (double)a_sum / active_cells_total : 0.0;

    printf("  KF count:        %u\n", ai->kf_count);
    printf("  Delta count:     %u\n", ai->df_count);
    printf("  Active cells:    %u (across all KFs)\n", active_cells_total);
    printf("  Avg A (active):  %.2f\n", avg_a);
    printf("  R range:         %u..%u\n", r_min, r_max);
    printf("  G range:         %u..%u\n", g_min, g_max);
    printf("  B range:         %u..%u\n", b_min, b_max);
}

/* ── event log ──
 *
 * Binary stream consumed by tools/animate_training.py to render the
 * training trajectory as a twinkling 256x256 grid animation.
 *
 * Layout:
 *   header (12 B):
 *     char     magic[4] = "CEVT"
 *     uint32   version  = 1
 *     uint32   reserved = 0
 *
 *   per clause record (variable):
 *     uint32   clause_idx
 *     uint8    decision    ( 0 = new keyframe,
 *                            1 = delta,
 *                            2 = identical / skipped )
 *     uint16   byte_count  ( clauses longer than this are truncated
 *                            to 65535 which is also stream_train's
 *                            per-line cap )
 *     uint8[byte_count * 2] = (y, x) pairs, one per byte of the
 *                             clause: y = byte_index % 256,
 *                             x = clause_byte_value.  Order matches
 *                             the write order in layers_encode_clause
 *                             so animators can replay bytes in stream
 *                             order for extra motion.
 */

static FILE* event_log_open(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[stream] cannot open log %s\n", path);
        return NULL;
    }
    const char magic[4] = { 'C', 'E', 'V', 'T' };
    uint32_t version  = 1;
    uint32_t reserved = 0;
    fwrite(magic, 1, 4, f);
    fwrite(&version,  sizeof(uint32_t), 1, f);
    fwrite(&reserved, sizeof(uint32_t), 1, f);
    return f;
}

static void event_log_write(FILE* f, uint32_t clause_idx,
                            uint8_t decision, const char* text, uint32_t tlen) {
    if (!f) return;
    if (tlen > 65535) tlen = 65535;
    uint16_t n = (uint16_t)tlen;

    fwrite(&clause_idx, sizeof(uint32_t), 1, f);
    fwrite(&decision,   sizeof(uint8_t),  1, f);
    fwrite(&n,          sizeof(uint16_t), 1, f);

    /* Flatten (y, x) pairs. 256 byte per-row limit matches GRID_SIZE. */
    uint8_t buf[512];
    uint32_t written = 0;
    for (uint32_t i = 0; i < tlen; i++) {
        buf[written++] = (uint8_t)(i % 256);
        buf[written++] = (uint8_t)(unsigned char)text[i];
        if (written >= sizeof(buf)) {
            fwrite(buf, 1, written, f);
            written = 0;
        }
    }
    if (written) fwrite(buf, 1, written, f);
}

static uint8_t decode_decision(uint32_t ret) {
    if (ret == UINT32_MAX) return 2;              /* skip / error */
    if (ret & 0x80000000u) return 1;              /* delta */
    return 0;                                     /* new keyframe */
}

/* ── Threshold auto-calibration ──
 *
 * Reads the first `sample_cap` valid clauses from the input, encodes
 * each one, and records the best cosine_a_only against prior same-
 * topic clauses. The observed best_sim distribution is what the real
 * ai_store_auto path would compare against threshold, so the
 * threshold sits at the target_ratio percentile of those values.
 *
 * Returns a threshold in [0, 1], or -1 if calibration produced no
 * usable samples (e.g. every clause had a unique topic).
 *
 * Pre-pass cost: O(N × max_topic_bucket × GRID_TOTAL). For 500
 * samples with a handful of topics, that's a few seconds. */

static int cmp_float_desc(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    if (fa > fb) return -1;
    if (fa < fb) return  1;
    return 0;
}

static float calibrate_threshold(const char* input_path,
                                  uint32_t sample_cap,
                                  float target_ratio) {
    FILE* fp = fopen(input_path, "r");
    if (!fp) return -1.0f;

    SpatialAI* tmp = spatial_ai_create();
    if (!tmp) { fclose(fp); return -1.0f; }

    float*   sims   = (float*)malloc(sample_cap * sizeof(float));
    uint32_t n_sims = 0;
    uint32_t seen   = 0;

    char line[LINE_BUF];
    while (seen < sample_cap && fgets(line, sizeof(line), fp)) {
        strip_trailing(line);
        if (is_skippable(line)) continue;
        if (strlen(line) < MIN_CLAUSE_LEN) continue;

        uint32_t topic = ai_resolve_topic(line, NULL);

        SpatialGrid* grid = grid_create();
        if (!grid) break;
        layers_encode_clause(line, NULL, grid);
        update_rgb_directional(grid);

        /* Best same-topic cosine against what we've already stored */
        if (topic != 0 && n_sims < sample_cap) {
            float best = 0.0f;
            int   any  = 0;
            for (uint32_t i = 0; i < tmp->kf_count; i++) {
                if (tmp->keyframes[i].topic_hash != topic) continue;
                float s = cosine_a_only(grid, &tmp->keyframes[i].grid);
                if (s > best) best = s;
                any = 1;
            }
            if (any) sims[n_sims++] = best;
        }

        /* Accumulate all calibration clauses as keyframes so subsequent
         * same-topic clauses have something to compare against. */
        ai_force_keyframe(tmp, line, NULL);
        grid_destroy(grid);
        seen++;
    }
    fclose(fp);

    float threshold = -1.0f;
    if (n_sims >= 10) {
        qsort(sims, n_sims, sizeof(float), cmp_float_desc);
        /* target_ratio fraction of clauses should be at or above
         * threshold. With the array sorted descending, the value at
         * position ⌊target × N⌋ is that threshold. */
        uint32_t idx = (uint32_t)(target_ratio * (float)n_sims);
        if (idx >= n_sims) idx = n_sims - 1;
        threshold = sims[idx];
        printf("[calibrate] sampled=%u  same-topic_pairs=%u\n"
               "[calibrate] sim distribution: min=%.3f  p%02d=%.3f  p50=%.3f  p10=%.3f  max=%.3f\n",
               seen, n_sims, sims[n_sims - 1],
               (int)(target_ratio * 100), threshold,
               sims[n_sims / 2], sims[n_sims / 10], sims[0]);
        printf("[calibrate] picked threshold=%.3f for target delta ratio %.1f%%\n",
               threshold, target_ratio * 100.0f);
    } else {
        printf("[calibrate] only %u same-topic pairs in %u clauses — "
               "not enough to tune a threshold\n",
               n_sims, seen);
    }

    free(sims);
    spatial_ai_destroy(tmp);
    return threshold;
}

/* ── main ────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    StreamArgs args;
    if (parse_args(argc, argv, &args) != 0) {
        usage(argv[0]);
        return 2;
    }

    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    if (!ai) {
        fprintf(stderr, "[stream] spatial_ai_create failed\n");
        return 1;
    }

    /* Threshold precedence: explicit --threshold > --target-delta
     * auto-calibration > engine default. */
    if (args.threshold >= 0.0f) {
        ai_set_store_threshold(args.threshold);
    } else if (args.target_delta >= 0.0f) {
        uint32_t sample_cap = args.calibrate_samples;
        if (sample_cap > args.max_clauses) sample_cap = args.max_clauses;
        printf("[calibrate] target_delta=%.1f%%  samples=%u\n",
               args.target_delta * 100.0f, sample_cap);
        float t = calibrate_threshold(args.input, sample_cap, args.target_delta);
        if (t >= 0.0f) {
            ai_set_store_threshold(t);
        } else {
            printf("[calibrate] falling back to engine default %.2f\n",
                   ai_get_store_threshold());
        }
    }

    printf("[stream] reading: %s\n", args.input);
    printf("[stream] max=%u  checkpoint=%u  threshold=%.3f  save=%s\n",
           args.max_clauses, args.checkpoint,
           ai_get_store_threshold(), args.save);

    FILE* fp = fopen(args.input, "r");
    if (!fp) {
        fprintf(stderr, "[stream] cannot open %s\n", args.input);
        spatial_ai_destroy(ai);
        return 1;
    }

    ensure_parent_dir(args.save);
    if (args.event_log) ensure_parent_dir(args.event_log);
    FILE* log = event_log_open(args.event_log);
    if (log) printf("[stream] event log: %s\n", args.event_log);

    char line[LINE_BUF];
    uint32_t count = 0;
    uint32_t skipped = 0;
    double   t0 = now_sec();

    while (count < args.max_clauses && fgets(line, sizeof(line), fp)) {
        strip_trailing(line);
        if (is_skippable(line)) { skipped++; continue; }
        uint32_t line_len = (uint32_t)strlen(line);
        if (line_len < MIN_CLAUSE_LEN) { skipped++; continue; }

        /* Full training step: layers_encode_clause → update_rgb_directional
         * → cosine vs existing KFs → delta (≥threshold) or new KF (<threshold).
         * All of that lives inside ai_store_auto. */
        uint32_t rid = ai_store_auto(ai, line, NULL);
        count++;

        if (log) {
            event_log_write(log, count - 1, decode_decision(rid), line, line_len);
        }

        if (args.verbose) {
            printf("[stream] %u: kf=%u df=%u  %.40s%s\n",
                   count, ai->kf_count, ai->df_count, line,
                   strlen(line) > 40 ? "..." : "");
        } else if (count % 5000 == 0) {
            double dt = now_sec() - t0;
            printf("[stream] %u clauses, KF=%u, Delta=%u, elapsed=%.1fs (%.0f c/s)\n",
                   count, ai->kf_count, ai->df_count, dt,
                   dt > 0 ? (double)count / dt : 0.0);
        }

        if (args.checkpoint > 0 && count % args.checkpoint == 0) {
            /* Derive a checkpoint path from the save path: save=foo.spai →
             * checkpoint=foo.ckpt_000005000.spai (same dir). */
            char ckpt[1280];
            size_t base_len = strlen(args.save);
            const char* dot = strrchr(args.save, '.');
            size_t stem_len = dot ? (size_t)(dot - args.save) : base_len;
            if (stem_len > 1100) stem_len = 1100;
            memcpy(ckpt, args.save, stem_len);
            snprintf(ckpt + stem_len, sizeof(ckpt) - stem_len,
                     ".ckpt_%06u.spai", count);

            SpaiStatus s = ai_save(ai, ckpt);
            if (s == SPAI_OK) {
                printf("[checkpoint] saved: %s (%.2f MB)\n",
                       ckpt, file_size_or_zero(ckpt) / 1e6);
            } else {
                printf("[checkpoint] FAILED: %s (%s)\n",
                       ckpt, spai_status_str(s));
            }
        }
    }

    fclose(fp);
    if (log) { fclose(log); printf("[stream] event log closed\n"); }

    double elapsed = now_sec() - t0;
    printf("[stream] ingest done: clauses=%u skipped=%u KF=%u Delta=%u elapsed=%.2fs\n",
           count, skipped, ai->kf_count, ai->df_count, elapsed);

    /* Final save */
    SpaiStatus s = ai_save(ai, args.save);
    if (s != SPAI_OK) {
        fprintf(stderr, "[stream] final save FAILED (%s)\n", spai_status_str(s));
        spatial_ai_destroy(ai);
        return 1;
    }
    printf("[done] saved: %s (%.2f MB)\n",
           args.save, file_size_or_zero(args.save) / 1e6);

    /* Engine stats */
    printf("\n[stats] engine summary\n");
    report_engine_stats(ai);

    /* Optional verify */
    if (args.verify && count > 0) {
        uint32_t probe = count > VERIFY_PROBE ? VERIFY_PROBE : count;
        printf("\n[verify] unseen-query pass on last %u clauses\n", probe);
        VerifyStats vs;
        verify_run(ai, args.input, probe, &vs);

        double avg = vs.clauses_scanned ? vs.sum_sim / vs.clauses_scanned : 0.0;
        printf("  scanned:         %u\n", vs.clauses_scanned);
        printf("  matched (>0):    %u\n", vs.clauses_matched);
        printf("  avg similarity:  %.4f\n", avg);
        printf("  min / max:       %.4f / %.4f\n", vs.min_sim, vs.max_sim);
        printf("  hits >= 0.90:    %u\n", vs.hits_90);
        printf("  hits >= 0.50:    %u\n", vs.hits_50);
        printf("  hits >= 0.10:    %u\n", vs.hits_10);
    }

    spatial_ai_destroy(ai);
    return 0;
}
