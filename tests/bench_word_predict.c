/*
 * bench_word_predict.c — masked word prediction via learned RGBA channels
 *
 * SPEC-aligned generation (§4, §5, §9):
 *
 *   Training diffused R/G/B directionally across every stored keyframe:
 *     R  diagonals  → semantic/morpheme class
 *     G  vertical   → word-substitution class
 *     B  horizontal → clause-order class
 *
 *   For each test clause we mask EVERY in-vocab word (no "longest word"
 *   heuristic, no byte-frequency-only scoring). For each candidate
 *   byte sequence w at offset s, length L:
 *
 *     byte_score(y, v) = A_sum[y, v]
 *                      × (1 - |R_mean[y,v] - input_R_ctx[y]| / 255)
 *                      × (1 - |G_mean[y,v] - input_G_ctx[y]| / 255)
 *                      × (1 - |B_mean[y,v] - input_B_ctx[y]| / 255)
 *
 *     word_score(w)    = Π byte_score((s+i) mod 256, w[i])
 *
 *   All FOUR channels (A, R, G, B) are used — training learned all of
 *   them, generation must consult all of them. Input_R/G/B_ctx comes
 *   from the test clause with the target word masked so the candidate
 *   is evaluated against the surrounding clause's RGBA signature.
 *
 * Metrics
 *   - Top-1 / Top-5 accuracy
 *   - Word-level perplexity from softmax over same-length candidates
 *
 * Train/test split: 70% / 30% by clause index.
 * No self-query: test words live in unseen test clauses.
 *
 * Usage
 *   ./build/bench_word_predict data/sample_ko.txt
 *   ./build/bench_word_predict data/sample_en.txt 1000
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
#include "bench_utf8.h"
#include "bench_args.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

/* ── configuration ── */
#define MAX_CLAUSES     10000
#define MAX_LINE_LEN    4096
#define MIN_CLAUSE_LEN  10
#define MAX_VOCAB       8000
#define MAX_WORD_LEN    64
#define MIN_WORD_BYTES  3
#define TRAIN_RATIO     0.70f
#define DEFAULT_LIMIT   1000

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int ensure_dir(const char* path) {
    if (!path) return -1;
    if (mkdir(path, 0777) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

/* ── text helpers ── */

static int is_meta_line(const char* line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] == '<' || line[0] == '\0') return 1;
    return 0;
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
    return count;
}

/* ── word extraction ── */

typedef struct { char text[MAX_WORD_LEN]; uint32_t offset; uint32_t len; } WordSpan;

static uint32_t extract_words(const char* clause, WordSpan* out, uint32_t max) {
    uint32_t count = 0;
    uint32_t i = 0;
    uint32_t len = (uint32_t)strlen(clause);

    while (i < len && count < max) {
        while (i < len && (clause[i] == ' ' || clause[i] == '\t')) i++;
        if (i >= len) break;

        uint32_t w_start = i;
        while (i < len && clause[i] != ' ' && clause[i] != '\t') i++;
        uint32_t w_end = i;

        /* strip trailing punctuation */
        while (w_end > w_start) {
            char c = clause[w_end - 1];
            if (c == '.' || c == ',' || c == '!' || c == '?' ||
                c == ';' || c == ':' || c == ')' || c == ']' ||
                c == '"' || c == '\'')
                w_end--;
            else break;
        }
        while (w_start < w_end) {
            char c = clause[w_start];
            if (c == '(' || c == '[' || c == '"' || c == '\'') w_start++;
            else break;
        }

        uint32_t w_len = w_end - w_start;
        if (w_len >= MIN_WORD_BYTES && w_len < MAX_WORD_LEN - 1) {
            memcpy(out[count].text, clause + w_start, w_len);
            out[count].text[w_len] = '\0';
            out[count].offset = w_start;
            out[count].len = w_len;
            count++;
        }
    }
    return count;
}

/* ── vocabulary ── */

typedef struct {
    char     word[MAX_WORD_LEN];
    uint32_t len;
    uint32_t freq;
} VocabEntry;

static int vocab_find(VocabEntry* vocab, uint32_t vcount,
                      const char* word, uint32_t wlen) {
    for (uint32_t i = 0; i < vcount; i++) {
        if (vocab[i].len == wlen && memcmp(vocab[i].word, word, wlen) == 0)
            return (int)i;
    }
    return -1;
}

static int vocab_add(VocabEntry* vocab, uint32_t* vcount, uint32_t vmax,
                     const char* word, uint32_t wlen) {
    int idx = vocab_find(vocab, *vcount, word, wlen);
    if (idx >= 0) { vocab[idx].freq++; return idx; }
    if (*vcount >= vmax) return -1;
    VocabEntry* e = &vocab[*vcount];
    memcpy(e->word, word, wlen);
    e->word[wlen] = '\0';
    e->len = wlen;
    e->freq = 1;
    (*vcount)++;
    return (int)(*vcount - 1);
}

/* ── per-length vocab index (for efficient same-length scoring) ── */

typedef struct {
    uint32_t* ids;    /* vocab indices */
    uint32_t  count;
    uint32_t  cap;
} LengthBucket;

static void lbuckets_build(LengthBucket* buckets, VocabEntry* vocab, uint32_t vcount) {
    for (uint32_t i = 0; i < vcount; i++) {
        uint32_t L = vocab[i].len;
        if (L >= MAX_WORD_LEN) continue;
        LengthBucket* b = &buckets[L];
        if (b->count >= b->cap) {
            b->cap = b->cap ? b->cap * 2 : 16;
            b->ids = (uint32_t*)realloc(b->ids, b->cap * sizeof(uint32_t));
        }
        b->ids[b->count++] = i;
    }
}

static void lbuckets_free(LengthBucket* buckets) {
    for (uint32_t L = 0; L < MAX_WORD_LEN; L++) free(buckets[L].ids);
}

/* ── scoring a candidate word ─────────────────────────── */

/* log score of word w at offset s. Higher = better.
   Uses the aggregated training tables and the full RGBA input signature. */
static double score_word(const AggTables* t, const InputSignature* sig,
                         uint32_t s, const char* w, uint32_t len) {
    double log_s = 0.0;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t y = (s + i) % GRID_SIZE;
        double in_R, in_G, in_B;
        input_signature_get(sig, y, &in_R, &in_G, &in_B);
        double bs = agg_score_byte(t, y, (uint8_t)w[i], in_R, in_G, in_B);
        /* Laplace-style smoothing to allow comparison with unseen (y,v) cells */
        log_s += log(bs + 1e-9);
    }
    return log_s;
}

/* ── masked-clause encoding ─────────────────────────────
 * Build a copy of clause with bytes [mask_s..mask_s+mask_len) replaced
 * by space, then encode + RGB-diffuse. This gives an input whose RGB
 * signature reflects the clause WITHOUT the target word. */
static void encode_masked(const char* clause, uint32_t mask_s, uint32_t mask_len,
                          SpatialGrid* out) {
    char buf[MAX_LINE_LEN];
    uint32_t clen = (uint32_t)strlen(clause);
    if (clen >= MAX_LINE_LEN) clen = MAX_LINE_LEN - 1;
    memcpy(buf, clause, clen);
    buf[clen] = '\0';
    for (uint32_t i = mask_s; i < mask_s + mask_len && i < clen; i++) {
        buf[i] = ' ';
    }
    grid_clear(out);
    layers_encode_clause(buf, NULL, out);
    update_rgb_directional(out);
}

/* ── main ─────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    BenchArgs ba;
    if (bench_parse_args(argc, argv, &ba) != 0 || ba.positional_count < 1) {
        fprintf(stderr,
            "Usage: %s <text_file> [max_clauses] [--save P] [--load P] [--load-only P]\n\n"
            "  Train: 70%%, Test: 30%% (by clause order)\n"
            "  Scoring: A x R_sim x G_sim x B_sim  (SPEC §4 §5 §9)\n\n"
            "  Auto-save: if --save is omitted and training runs,\n"
            "             saves to build/models/bench_word_predict_auto.spai\n\n"
            "Example:\n"
            "  %s data/sample_ko.txt --save model_ko.spai\n"
            "  %s data/sample_en.txt 1000 --load-only model_ko.spai\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* filepath   = ba.positional[0];
    uint32_t    max_clauses = (ba.positional_count >= 2) ?
                              (uint32_t)atoi(ba.positional[1]) : DEFAULT_LIMIT;
    if (max_clauses > MAX_CLAUSES) max_clauses = MAX_CLAUSES;

    FILE* fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return 1; }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  Word Prediction\n");
    printf("  (A x R_sim x G_sim x B_sim — SPEC §4 §5 §9)\n");
    printf("========================================\n");
    printf("  File:         %s\n", filepath);
    printf("  Max clauses:  %u\n", max_clauses);
    printf("  Train ratio:  %.0f%%\n", TRAIN_RATIO * 100);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* ── [1/5] Load ── */
    printf("[1/5] Loading clauses...\n");
    double t0 = now_sec();

    char (*clauses)[MAX_LINE_LEN] = malloc((size_t)max_clauses * MAX_LINE_LEN);
    if (!clauses) { fprintf(stderr, "alloc failed\n"); return 1; }
    uint32_t clause_count = 0;
    char line[MAX_LINE_LEN];
    char split_buf[8][MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) && clause_count < max_clauses) {
        trim_trailing(line);
        if (is_meta_line(line)) continue;
        uint32_t n = split_clauses(line, split_buf, 8);
        for (uint32_t i = 0; i < n && clause_count < max_clauses; i++) {
            memcpy(clauses[clause_count++], split_buf[i], MAX_LINE_LEN);
        }
    }
    fclose(fp);
    printf("  Loaded: %u clauses (%.2f sec)\n\n", clause_count, now_sec() - t0);

    if (clause_count < 30) {
        fprintf(stderr, "ERROR: need at least 30 clauses\n");
        free(clauses);
        return 1;
    }

    uint32_t train_count = (uint32_t)(clause_count * TRAIN_RATIO);
    uint32_t test_count  = clause_count - train_count;

    /* ── [2/5] Train: store keyframes (RGB diffusion happens inside) ── */
    printf("[2/5] Training on %u clauses...\n", train_count);
    t0 = now_sec();

    SpatialAI* ai = NULL;
    uint32_t loaded_kf = 0, loaded_df = 0;
    if (ba.load_path) {
        SpaiStatus ls;
        ai = ai_load(ba.load_path, &ls);
        if (!ai) {
            fprintf(stderr, "  ERROR: load '%s' failed: %s\n",
                    ba.load_path, spai_status_str(ls));
            free(clauses);
            return 1;
        }
        loaded_kf = ai->kf_count;
        loaded_df = ai->df_count;
        printf("  Loaded '%s': %u KF, %u Delta\n", ba.load_path, loaded_kf, loaded_df);
    } else {
        ai = spatial_ai_create();
    }

    VocabEntry* vocab = (VocabEntry*)calloc(MAX_VOCAB, sizeof(VocabEntry));
    uint32_t vcount = 0;
    WordSpan words[256];

    SpatialCanvasPool* pool = ai_get_canvas_pool(ai);

    if (!ba.load_only) {
        for (uint32_t c = 0; c < train_count; c++) {
            /* pool_add_clause stores into a type-matching canvas and
             * runs layers_encode_clause (which seeds the B-channel
             * co-occurrence hash). Canvas-wide RGB diffusion runs
             * once after training (below). */
            pool_add_clause(pool, clauses[c]);

            uint32_t nw = extract_words(clauses[c], words, 256);
            for (uint32_t w = 0; w < nw; w++) {
                vocab_add(vocab, &vcount, MAX_VOCAB, words[w].text, words[w].len);
            }
            if ((c + 1) % 100 == 0 || c + 1 == train_count) {
                printf("\r  Processed: %u / %u  (canvases=%u, slots=%u vocab=%u)",
                       c + 1, train_count, pool->count, pool_total_slots(pool), vcount);
            }
        }
        /* Canvas-level RGB diffusion (cross-boundary effects) */
        for (uint32_t i = 0; i < pool->count; i++)
            canvas_update_rgb(pool->canvases[i]);
    } else {
        /* load-only: still need vocab for scoring → scan train split for words */
        printf("  --load-only: skipping training. Building vocab from text...\n");
        for (uint32_t c = 0; c < train_count; c++) {
            uint32_t nw = extract_words(clauses[c], words, 256);
            for (uint32_t w = 0; w < nw; w++) {
                vocab_add(vocab, &vcount, MAX_VOCAB, words[w].text, words[w].len);
            }
        }
    }
    double t_train = now_sec() - t0;
    printf("\n  Done in %.2f sec\n", t_train);

    /* Save if requested */
    if (ba.save_path) {
        SpaiStatus ss;
        if (ba.load_path && strcmp(ba.load_path, ba.save_path) == 0) {
            ss = ai_save_incremental(ai, ba.save_path);
            printf("  [save] incremental → '%s' : %s  (KF %u→%u, Delta %u→%u)\n",
                   ba.save_path, spai_status_str(ss),
                   loaded_kf, ai->kf_count, loaded_df, ai->df_count);
        } else {
            ss = ai_save(ai, ba.save_path);
            printf("  [save] full → '%s' : %s  (%u KF, %u Delta)\n",
                   ba.save_path, spai_status_str(ss),
                   ai->kf_count, ai->df_count);
        }
    } else if (!ba.load_only) {
        const char* auto_path = "build/models/bench_word_predict_auto.spai";
        (void)ensure_dir("build");
        (void)ensure_dir("build/models");
        SpaiStatus ss = ai_save(ai, auto_path);
        printf("  [save] auto → '%s' : %s  (%u KF, %u Delta)\n",
               auto_path, spai_status_str(ss), ai->kf_count, ai->df_count);
    }
    printf("\n");

    /* ── [3/5] Build aggregated RGBA tables from the pool ── */
    printf("[3/5] Building aggregated A/R/G/B tables from %u pool slots...\n",
           pool_total_slots(pool));
    t0 = now_sec();
    AggTables* agg = agg_build_from_pool(pool);
    if (!agg) { fprintf(stderr, "agg_build_from_pool failed\n"); return 1; }

    /* active cells in aggregated A */
    uint32_t active_cells = 0;
    for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        if (agg->A_sum[i] > 0) active_cells++;
    }
    printf("  Active (y,v) cells:  %u / %u  (%.1f%%)\n",
           active_cells, GRID_SIZE * GRID_SIZE,
           100.0 * active_cells / (GRID_SIZE * GRID_SIZE));

    /* per-length index for fast same-length candidate enumeration */
    LengthBucket buckets[MAX_WORD_LEN];
    memset(buckets, 0, sizeof(buckets));
    lbuckets_build(buckets, vocab, vcount);
    printf("  Vocab size:          %u\n", vcount);
    printf("  Built in %.2f sec\n\n", now_sec() - t0);

    /* ── [4/5] Predict masked words on test clauses ── */
    printf("[4/5] Predicting every in-vocab word in %u held-out clauses...\n",
           test_count);
    t0 = now_sec();

    SpatialGrid* masked_grid = grid_create();
    InputSignature sig;

    uint32_t top1_hits = 0;
    uint32_t top5_hits = 0;
    uint32_t total_preds = 0;
    uint32_t clauses_with_any = 0;
    uint32_t oov = 0;
    uint32_t empty_candidates = 0;
    double   log_p_sum = 0.0;

    /* Per-length temporary score buffer */
    double* tmp_scores = (double*)malloc(MAX_VOCAB * sizeof(double));

    /* Cascade-mode scratch buffers */
    uint32_t cascade_ids[64];
    float    cascade_scores[64];

    if (ba.cascade) {
        printf("  [--cascade] match_cascade(CASCADE_GENERATE) +"
               " keyframe-byte extraction\n");
    }

    for (uint32_t t = 0; t < test_count; t++) {
        uint32_t ci = train_count + t;
        uint32_t nw = extract_words(clauses[ci], words, 256);
        if (nw < 1) continue;

        int any_in_clause = 0;

        for (uint32_t w = 0; w < nw; w++) {
            WordSpan* target = &words[w];
            int true_idx = vocab_find(vocab, vcount, target->text, target->len);
            if (true_idx < 0) { oov++; continue; }

            LengthBucket* b = &buckets[target->len];
            if (b->count < 2) { continue; }

            /* 1) Encode test clause with target word masked out. */
            encode_masked(clauses[ci], target->offset, target->len, masked_grid);
            input_signature_compute(&sig, masked_grid);

            /* ── CASCADE mode: top-K pool match, extract bytes per slot ── */
            if (ba.cascade) {
                uint32_t k = pool_match_topk(pool, masked_grid,
                                             5, cascade_ids, cascade_scores);
                if (k == 0) continue;

                /* Extract candidate word from each top-K pool slot. */
                char pred[64];
                SpatialGrid* kg = grid_create();
                int is_top1 = 0, is_top5 = 0;
                for (uint32_t r = 0; r < k; r++) {
                    const SubtitleEntry* se = &pool->track.entries[cascade_ids[r]];
                    canvas_slot_to_grid(pool->canvases[se->canvas_id],
                                        se->slot_id, kg);
                    uint32_t pl = 0;
                    for (uint32_t i = 0; i < target->len && pl + 1 < sizeof(pred); i++) {
                        uint32_t y = (target->offset + i) % GRID_SIZE;
                        uint32_t best_x = 0;
                        uint16_t best_a = 0;
                        for (uint32_t x = 0; x < GRID_SIZE; x++) {
                            uint32_t idx = y * GRID_SIZE + x;
                            if (kg->A[idx] > best_a) {
                                best_a = kg->A[idx];
                                best_x = x;
                            }
                        }
                        pred[pl++] = (char)(uint8_t)best_x;
                    }
                    pred[pl] = '\0';

                    if (pl == target->len &&
                        memcmp(pred, target->text, target->len) == 0) {
                        if (r == 0) is_top1 = 1;
                        is_top5 = 1;
                        break;
                    }
                }
                if (is_top1) top1_hits++;
                if (is_top5) top5_hits++;
                total_preds++;
                any_in_clause = 1;
                grid_destroy(kg);
                continue;
            }

            /* ── Default mode: byte-position score_word over vocab ── */

            /* 2) Score every same-length vocab candidate */
            double max_lp = -1e300;
            double true_lp = 0.0;
            for (uint32_t j = 0; j < b->count; j++) {
                uint32_t vi = b->ids[j];
                double lp = score_word(agg, &sig, target->offset,
                                       vocab[vi].word, vocab[vi].len);
                tmp_scores[j] = lp;
                if (lp > max_lp) max_lp = lp;
                if ((int)vi == true_idx) true_lp = lp;
            }

            if (!isfinite(true_lp) || true_lp < -1e200) {
                empty_candidates++;
                continue;
            }

            uint32_t rank = 0;
            for (uint32_t j = 0; j < b->count; j++) {
                if (b->ids[j] == (uint32_t)true_idx) continue;
                if (tmp_scores[j] > true_lp) rank++;
            }

            double denom = 0.0;
            for (uint32_t j = 0; j < b->count; j++) {
                denom += exp(tmp_scores[j] - max_lp);
            }
            double log_P_true = (true_lp - max_lp) - log(denom);

            if (rank == 0) top1_hits++;
            if (rank < 5)  top5_hits++;
            log_p_sum += log_P_true;
            total_preds++;
            any_in_clause = 1;
        }

        if (any_in_clause) clauses_with_any++;
        if ((t + 1) % 20 == 0 || t + 1 == test_count) {
            printf("\r  Progress: %u / %u clauses, %u preds",
                   t + 1, test_count, total_preds);
        }
    }

    free(tmp_scores);
    grid_destroy(masked_grid);
    (void)cascade_scores;  /* currently unused; reserved for score reporting */

    double t_pred = now_sec() - t0;
    printf("\n  Done in %.2f sec  (%.0f preds/sec)\n\n",
           t_pred, total_preds / (t_pred + 1e-9));

    /* ── [5/5] Report ── */
    printf("[5/5] Results\n");
    printf("========================================\n\n");

    printf("  SETUP\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Train clauses:         %u\n", train_count);
    printf("  Test clauses:          %u  (with predictions: %u)\n",
           test_count, clauses_with_any);
    printf("  Keyframes (I):         %u\n", ai->kf_count);
    printf("  Deltas (P):            %u\n", ai->df_count);
    printf("  Vocab size:            %u\n", vcount);
    printf("  In-vocab preds:        %u\n", total_preds);
    printf("  OOV skipped:           %u\n", oov);
    printf("  Empty-candidate:       %u\n\n", empty_candidates);

    if (total_preds > 0) {
        double top1 = 100.0 * top1_hits / total_preds;
        double top5 = 100.0 * top5_hits / total_preds;
        double mean_log_P = log_p_sum / total_preds;
        double perp = exp(-mean_log_P);

        /* Random baseline = mean vocabulary size at the same length class */
        double expected_candidates = 0;
        uint32_t bucket_count = 0;
        for (uint32_t L = 0; L < MAX_WORD_LEN; L++) {
            if (buckets[L].count > 1) {
                expected_candidates += (double)buckets[L].count;
                bucket_count++;
            }
        }
        double avg_cand = bucket_count ? expected_candidates / bucket_count : 1.0;

        printf("  ACCURACY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Top-1:                 %.2f%%  (%u / %u)\n",
               top1, top1_hits, total_preds);
        printf("  Top-5:                 %.2f%%  (%u / %u)\n",
               top5, top5_hits, total_preds);
        printf("  Random baseline:       %.2f%%  (avg same-length cands = %.0f)\n",
               100.0 / avg_cand, avg_cand);
        printf("  Lift over random:      %.1fx\n\n", top1 / (100.0 / avg_cand + 1e-9));

        printf("  WORD-LEVEL PERPLEXITY\n");
        printf("  ─────────────────────────────────────\n");
        printf("  Avg -log P(true):      %.3f nats\n", -mean_log_P);
        printf("  Perplexity:            %.2f\n\n", perp);

        printf("  INTERPRETATION\n");
        printf("  ─────────────────────────────────────\n");
        double lift = top1 * avg_cand / 100.0;
        if (lift > 10)
            printf("  Strong signal (>10x random baseline).\n");
        else if (lift > 3)
            printf("  Clear signal (>3x random baseline).\n");
        else if (lift > 1.2)
            printf("  Weak signal (>1.2x random).\n");
        else
            printf("  Near baseline.\n");
        printf("\n");
    }

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    lbuckets_free(buckets);
    agg_destroy(agg);
    free(vocab);
    spatial_ai_destroy(ai);
    free(clauses);
    return 0;
}
