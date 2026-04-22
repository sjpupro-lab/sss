/*
 * bench_qa.c — SQuAD-style QA via context frames + B×G×R extraction
 *
 * SPEC alignment:
 *
 *   §7  Context frames are sequential "clause frames". Same-source
 *       clauses share contiguous frame_ids. Questions are matched
 *       against the frame pool, not directly against answers.
 *
 *   §9  Matching uses the full pipeline (3-layer encode → RGB
 *       directional update → overlap coarse → RGB-weighted cosine).
 *
 *   §4, §5  Generation consults all four channels. R (diagonal) gives
 *       semantic linkage, G (vertical) gives substitution candidates,
 *       B (horizontal) gives clause-order direction between frames.
 *
 * Pipeline:
 *
 *   Training
 *     Parse TSV  (question \t context \t answer).
 *     Group by unique context string.
 *     For each unique context:
 *       split into clauses, store each as a consecutive ContextFrame
 *       via context_add_frame → frame_ids are contiguous.
 *     Record each context's (clause_start, clause_count).
 *     Each QA pair points to its context index.
 *
 *   Testing (30% held-out QAs)
 *     Step 1. Encode the question: 3-layer + RGB directional update.
 *     Step 2. Match over the context-frame pool:
 *               best_frame_id = argmax_i cosine_rgb_weighted(q, frame[i])
 *     Step 3. Directional extraction over window = best ±N frames:
 *               For every byte position (y, x) in any window frame:
 *                 novelty = frame.A[y,x] - question.A[y,x]   (≥0 only)
 *                 R_sim   = 1 - |frame.R[y,x] - q_sig.R_row[y]| / 255
 *                 G_sim   = 1 - |frame.G[y,x] - q_sig.G_row[y]| / 255
 *                 B_sim   = 1 - |frame.B[y,x] - q_sig.B_row[y]| / 255
 *                 score   = novelty × R_sim × G_sim × B_sim
 *               Per-frame density → pick frame with highest avg score.
 *               Within that frame, argmax x per row produces bytes in
 *               the score-active Y range.
 *     Step 4. Assemble: sort by (frame_id, y) ascending (= B-channel /
 *             horizontal clause order). Emit the byte stream.
 *
 * Metrics
 *   Context Retrieval   — best_frame_id lands inside gold context range
 *   Answer EM            — exact match after normalization
 *   Answer F1            — word-level precision/recall harmonic mean
 *
 * TSV  (one row per QA pair):
 *     question \t context \t answer
 *
 * Usage
 *   ./build/bench_qa data/qa_en.txt
 *   ./build/bench_qa data/qa_en.txt 500
 *   ./build/bench_qa data/qa_en.txt --save model.spai
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_context.h"
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
#include <ctype.h>
#include <time.h>

/* ── configuration ── */
#define MAX_LINE_LEN    16384
#define MAX_QA_PAIRS    8000
#define MAX_CONTEXTS    4000
#define MAX_CLAUSES     32
#define DEFAULT_LIMIT   500
#define TRAIN_RATIO     0.70f
#define WINDOW_RADIUS   2      /* ±N frames around matched frame */
#define MIN_CLAUSE_LEN  10

/* ── timing ── */
static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── clause splitter (shared style with other benches) ── */
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
                /* trim trailing whitespace */
                int sl = (int)strlen(out[count]);
                while (sl > 0 && (out[count][sl-1] == '\n' ||
                                  out[count][sl-1] == '\r' ||
                                  out[count][sl-1] == ' '))
                    out[count][--sl] = '\0';
                if ((int)strlen(out[count]) >= MIN_CLAUSE_LEN) count++;
            }
            start = p + 1;
            while (*start == ' ') start++;
            p = start;
            continue;
        }
        p++;
    }
    /* tail */
    if (start < p && count < max_out) {
        uint32_t len = (uint32_t)(p - start);
        if (len >= MIN_CLAUSE_LEN && len < MAX_LINE_LEN) {
            memcpy(out[count], start, len);
            out[count][len] = '\0';
            count++;
        }
    }
    return count;
}

/* ── QA data ── */

typedef struct {
    char*    text;
    uint32_t clause_start;   /* first frame_id in ContextManager */
    uint32_t clause_count;
    int      trained;
} ContextEntry;

typedef struct {
    int    ctx_idx;          /* into contexts[] */
    char*  question;
    char*  answer;
} QAItem;

/* Parse one TSV line: question \t context \t answer */
static int parse_line(const char* line, char** q, char** c, char** a) {
    const char* t1 = strchr(line, '\t');
    if (!t1) return 0;
    const char* t2 = strchr(t1 + 1, '\t');
    if (!t2) return 0;

    size_t q_len = (size_t)(t1 - line);
    size_t c_len = (size_t)(t2 - (t1 + 1));
    size_t a_len = strlen(t2 + 1);
    /* strip trailing whitespace on answer */
    while (a_len > 0 && (t2[1 + a_len - 1] == '\n' ||
                         t2[1 + a_len - 1] == '\r' ||
                         t2[1 + a_len - 1] == ' '))
        a_len--;

    if (q_len < 2 || c_len < 10 || a_len < 1) return 0;

    *q = (char*)malloc(q_len + 1);
    *c = (char*)malloc(c_len + 1);
    *a = (char*)malloc(a_len + 1);
    if (!*q || !*c || !*a) return 0;
    memcpy(*q, line,   q_len); (*q)[q_len] = '\0';
    memcpy(*c, t1 + 1, c_len); (*c)[c_len] = '\0';
    memcpy(*a, t2 + 1, a_len); (*a)[a_len] = '\0';
    return 1;
}

/* Find or add a context by string. Returns index or -1. */
static int find_or_add_context(ContextEntry* ctxs, int* n, const char* text) {
    for (int i = 0; i < *n; i++) {
        if (strcmp(ctxs[i].text, text) == 0) return i;
    }
    if (*n >= MAX_CONTEXTS) return -1;
    ctxs[*n].text         = strdup(text);
    ctxs[*n].clause_start = UINT32_MAX;
    ctxs[*n].clause_count = 0;
    ctxs[*n].trained      = 0;
    return (*n)++;
}

/* Matching is now handled by match_cascade(ai, q, CASCADE_QA) via the
   AI's keyframes. Each context clause is stored with ai_force_keyframe
   so clause ids map 1-1 to keyframe ids (no delta collapse). */

/* ── Answer extraction: B × G × R over window ── */

/* Score a single (y, x) cell per SPEC §4 §5 §9:
     score = A × R_sim × G_sim × B_sim
   using the question's per-row and global RGBA signatures. */
static inline double cell_score(const SpatialGrid* frame, uint32_t y, uint32_t x,
                                const InputSignature* q_sig) {
    uint32_t i = y * GRID_SIZE + x;
    if (frame->A[i] == 0) return 0.0;
    double q_R, q_G, q_B;
    input_signature_get(q_sig, y, &q_R, &q_G, &q_B);
    double R_sim = 1.0 - fabs((double)frame->R[i] - q_R) / 255.0;
    double G_sim = 1.0 - fabs((double)frame->G[i] - q_G) / 255.0;
    double B_sim = 1.0 - fabs((double)frame->B[i] - q_B) / 255.0;
    if (R_sim < 0) R_sim = 0;
    if (G_sim < 0) G_sim = 0;
    if (B_sim < 0) B_sim = 0;
    return (double)frame->A[i] * R_sim * G_sim * B_sim;
}

/* Per-frame Y-row aggregate scores, length 256 */
static void frame_y_scores(const SpatialGrid* frame,
                           const InputSignature* q_sig,
                           double* per_y) {
    memset(per_y, 0, GRID_SIZE * sizeof(double));
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            per_y[y] += cell_score(frame, y, x, q_sig);
        }
    }
}

/* For each score-active row y in [y_start, y_end], emit argmax-score byte x. */
static size_t decode_frame_range(const SpatialGrid* frame,
                                 const InputSignature* q_sig,
                                 uint32_t y_start, uint32_t y_end,
                                 char* out, size_t cap) {
    size_t ol = 0;
    for (uint32_t y = y_start; y <= y_end && ol + 1 < cap; y++) {
        int    best_x = -1;
        double best_score = 0.0;

        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            double s = cell_score(frame, y, x, q_sig);
            if (s > best_score) {
                best_score = s;
                best_x = (int)x;
            }
        }
        if (best_x >= 0) out[ol++] = (char)(uint8_t)best_x;
    }
    out[ol] = '\0';
    return ol;
}

/* Extraction over a pool window — operate on tile grids extracted from
 * the SubtitleTrack entries ±radius around the matched entry.
 * Preserves the SPEC §4 §5 §7 §9 scoring pipeline. */
static void extract_answer_pool(SpatialCanvasPool* pool,
                                uint32_t match_entry_id,
                                int radius,
                                const InputSignature* q_sig,
                                char* out, size_t cap) {
    out[0] = '\0';
    if (!pool || pool->track.count == 0) return;

    int32_t w_start = (int32_t)match_entry_id - radius;
    if (w_start < 0) w_start = 0;
    int32_t w_end   = (int32_t)match_entry_id + radius;
    if (w_end >= (int32_t)pool->track.count) w_end = (int32_t)pool->track.count - 1;

    double   best_density = -1.0;
    uint32_t best_eid = match_entry_id;
    uint32_t best_ys = 0, best_ye = 0;

    /* Allocate reusable grid buffer for tile extraction */
    SpatialGrid* tile = grid_create();
    if (!tile) return;

    /* Pass 1: find global max density threshold to trim low-score tails */
    double global_max = 0.0;
    for (int32_t eid = w_start; eid <= w_end; eid++) {
        const SubtitleEntry* e = &pool->track.entries[eid];
        canvas_slot_to_grid(pool->canvases[e->canvas_id], e->slot_id, tile);
        double per_y[GRID_SIZE];
        frame_y_scores(tile, q_sig, per_y);
        for (uint32_t y = 0; y < GRID_SIZE; y++)
            if (per_y[y] > global_max) global_max = per_y[y];
    }
    double threshold = global_max * 0.25;

    for (int32_t eid = w_start; eid <= w_end; eid++) {
        const SubtitleEntry* e = &pool->track.entries[eid];
        canvas_slot_to_grid(pool->canvases[e->canvas_id], e->slot_id, tile);
        double per_y[GRID_SIZE];
        frame_y_scores(tile, q_sig, per_y);

        int32_t first = -1, last = -1;
        double total = 0;
        int32_t active_rows = 0;
        for (uint32_t y = 0; y < GRID_SIZE; y++) {
            if (per_y[y] >= threshold) {
                if (first < 0) first = (int32_t)y;
                last = (int32_t)y;
                total += per_y[y];
                active_rows++;
            }
        }
        if (first < 0 || active_rows < 2) continue;
        double density = total / (double)active_rows;
        if (density > best_density) {
            best_density = density;
            best_eid = (uint32_t)eid;
            best_ys = (uint32_t)first;
            best_ye = (uint32_t)last;
        }
    }

    const SubtitleEntry* be = &pool->track.entries[best_eid];
    canvas_slot_to_grid(pool->canvases[be->canvas_id], be->slot_id, tile);
    if (best_density <= 0.0) {
        grid_decode_text(tile, out, (uint32_t)cap);
    } else {
        decode_frame_range(tile, q_sig, best_ys, best_ye, out, cap);
    }
    grid_destroy(tile);
}

/* ── Evaluation: normalization, EM, F1 ── */

/* Normalize: lowercase ASCII, keep UTF-8 bytes (≥0x80) as-is,
   collapse non-alphanumeric into single spaces, trim. */
static void normalize_text(const char* in, char* out, size_t cap) {
    size_t oi = 0;
    int in_word = 0;
    for (size_t i = 0; in[i] && oi + 1 < cap; i++) {
        unsigned char c = (unsigned char)in[i];
        if ((c >= '0' && c <= '9') ||
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            c >= 0x80) {
            if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            out[oi++] = (char)c;
            in_word = 1;
        } else {
            if (in_word) {
                out[oi++] = ' ';
                in_word = 0;
            }
        }
    }
    while (oi > 0 && out[oi-1] == ' ') oi--;
    out[oi] = '\0';
}

static int compute_em(const char* pred, const char* gold) {
    char np[2048], ng[2048];
    normalize_text(pred, np, sizeof(np));
    normalize_text(gold, ng, sizeof(ng));
    if (np[0] == '\0' || ng[0] == '\0') return 0;
    return strcmp(np, ng) == 0;
}

/* Simple split by space into up to 64 tokens */
static int tokenize(char* s, char* tokens[], int max_tokens) {
    int n = 0;
    char* p = s;
    while (*p && n < max_tokens) {
        while (*p == ' ') p++;
        if (!*p) break;
        tokens[n++] = p;
        while (*p && *p != ' ') p++;
        if (*p) { *p = '\0'; p++; }
    }
    return n;
}

/* F1 over word-level bag multiset intersection */
static double compute_f1(const char* pred, const char* gold) {
    char np[2048], ng[2048];
    normalize_text(pred, np, sizeof(np));
    normalize_text(gold, ng, sizeof(ng));
    if (np[0] == '\0' || ng[0] == '\0') return 0.0;

    char np_c[2048], ng_c[2048];
    strncpy(np_c, np, sizeof(np_c)); np_c[sizeof(np_c) - 1] = '\0';
    strncpy(ng_c, ng, sizeof(ng_c)); ng_c[sizeof(ng_c) - 1] = '\0';

    char* pt[128]; int pn = tokenize(np_c, pt, 128);
    char* gt[128]; int gn = tokenize(ng_c, gt, 128);
    if (pn == 0 || gn == 0) return 0.0;

    int used_g[128]; memset(used_g, 0, sizeof(used_g));
    int overlap = 0;
    for (int i = 0; i < pn; i++) {
        for (int j = 0; j < gn; j++) {
            if (!used_g[j] && strcmp(pt[i], gt[j]) == 0) {
                used_g[j] = 1;
                overlap++;
                break;
            }
        }
    }
    if (overlap == 0) return 0.0;
    double P = (double)overlap / pn;
    double R = (double)overlap / gn;
    return 2.0 * P * R / (P + R);
}

/* ── main ── */

int main(int argc, char* argv[]) {
    utf8_console_init();

    BenchArgs ba;
    if (bench_parse_args(argc, argv, &ba) != 0 || ba.positional_count < 1) {
        fprintf(stderr,
            "Usage: %s <qa.tsv> [max_pairs] [--save P] [--load P] [--load-only P]\n\n"
            "  TSV format:  <question>\\t<context>\\t<answer>\n"
            "  Pipeline:    context frames + B*G*R extraction (SPEC §4 §5 §7 §9)\n\n"
            "Example:\n"
            "  %s data/qa_en.txt\n"
            "  %s data/qa_en.txt 500 --save model_qa.spai\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* filepath = ba.positional[0];
    int max_pairs = (ba.positional_count >= 2) ? atoi(ba.positional[1]) : DEFAULT_LIMIT;
    if (max_pairs > MAX_QA_PAIRS) max_pairs = MAX_QA_PAIRS;

    FILE* fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open '%s'\n", filepath); return 1; }

    printf("========================================\n");
    printf("  SPATIAL-PATTERN-AI  QA Benchmark\n");
    printf("  (context frames + B*G*R extraction)\n");
    printf("========================================\n");
    printf("  File:            %s\n", filepath);
    printf("  Max pairs:       %d\n", max_pairs);
    printf("  Train ratio:     %.0f%%\n", TRAIN_RATIO * 100);
    printf("  Window radius:   +- %d frames\n", WINDOW_RADIUS);
    printf("----------------------------------------\n\n");

    morpheme_init();

    /* ── [1/4] Load TSV ── */
    printf("[1/4] Loading QA pairs and deduplicating contexts...\n");
    double t0 = now_sec();

    ContextEntry* contexts = (ContextEntry*)calloc(MAX_CONTEXTS, sizeof(ContextEntry));
    QAItem*       qa       = (QAItem*)      calloc((size_t)max_pairs, sizeof(QAItem));
    int n_contexts = 0;
    int n_qa       = 0;

    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fp) && n_qa < max_pairs) {
        char *q = NULL, *c = NULL, *a = NULL;
        if (!parse_line(line, &q, &c, &a)) continue;

        int ci = find_or_add_context(contexts, &n_contexts, c);
        free(c);  /* find_or_add copied via strdup */
        if (ci < 0) { free(q); free(a); continue; }

        qa[n_qa].ctx_idx  = ci;
        qa[n_qa].question = q;
        qa[n_qa].answer   = a;
        n_qa++;
    }
    fclose(fp);

    printf("  QA pairs:        %d\n",   n_qa);
    printf("  Unique contexts: %d\n\n", n_contexts);
    printf("  (loaded in %.2f sec)\n\n", now_sec() - t0);

    if (n_qa < 10 || n_contexts < 1) {
        fprintf(stderr, "ERROR: need at least 10 QA pairs with valid contexts\n");
        return 1;
    }

    /* Sample print */
    printf("  Example:\n");
    {
        int i = 0;
        char qp[80], cp[80], ap[80];
        strncpy(qp, qa[i].question, 76); qp[76] = '\0';
        if (strlen(qa[i].question) > 76) strcat(qp, "...");
        strncpy(cp, contexts[qa[i].ctx_idx].text, 76); cp[76] = '\0';
        if (strlen(contexts[qa[i].ctx_idx].text) > 76) strcat(cp, "...");
        strncpy(ap, qa[i].answer, 76); ap[76] = '\0';
        if (strlen(qa[i].answer) > 76) strcat(ap, "...");
        printf("    Q: %s\n", qp);
        printf("    C: %s\n", cp);
        printf("    A: %s\n\n", ap);
    }

    int train_n = (int)(n_qa * TRAIN_RATIO);
    int test_n  = n_qa - train_n;

    /* ── [2/4] Train: store each unique context as consecutive frames ── */
    printf("[2/4] Storing unique contexts as consecutive frames via context_add_frame...\n");
    t0 = now_sec();

    SpatialAI*      ai = NULL;
    uint32_t loaded_kf = 0, loaded_df = 0;

    if (ba.load_path) {
        SpaiStatus ls;
        ai = ai_load(ba.load_path, &ls);
        if (!ai) {
            fprintf(stderr, "  ERROR: load '%s' failed: %s\n",
                    ba.load_path, spai_status_str(ls));
            return 1;
        }
        loaded_kf = ai->kf_count;
        loaded_df = ai->df_count;
        printf("  Loaded '%s': %u KF, %u Delta\n", ba.load_path, loaded_kf, loaded_df);
    } else {
        ai = spatial_ai_create();
    }

    SpatialCanvasPool* pool = ai_get_canvas_pool(ai);

    if (!ba.load_only) {
        /* Store every unique context via pool_add_clause. Same-type
         * clauses cluster on the same canvas; clause_start/count now
         * indexes into the subtitle track (not SpatialAI keyframes). */
        uint32_t total_clauses = 0;
        char split_buf[MAX_CLAUSES][MAX_LINE_LEN];

        for (int ci = 0; ci < n_contexts; ci++) {
            if (contexts[ci].trained) continue;
            uint32_t n_cl = split_clauses(contexts[ci].text, split_buf, MAX_CLAUSES);
            if (n_cl == 0) continue;
            contexts[ci].clause_start = pool->track.count;
            contexts[ci].clause_count = n_cl;
            for (uint32_t k = 0; k < n_cl; k++) {
                pool_add_clause(pool, split_buf[k]);
                total_clauses++;
            }
            contexts[ci].trained = 1;
            if ((ci + 1) % 50 == 0 || ci + 1 == n_contexts) {
                printf("\r  Contexts: %d / %d, canvases: %u, slots: %u",
                       ci + 1, n_contexts, pool->count, pool_total_slots(pool));
            }
        }
        /* Run canvas-level RGB diffusion once all slots are placed */
        for (uint32_t i = 0; i < pool->count; i++)
            canvas_update_rgb(pool->canvases[i]);
        uint32_t n_kf = 0, n_df = 0;
        for (uint32_t i = 0; i < pool->count; i++) {
            if (!pool->canvases[i]->classified) continue;
            if (pool->canvases[i]->frame_type == CANVAS_IFRAME) n_kf++;
            else n_df++;
        }
        printf("\n  Stored %u clauses on %u canvases (KF=%u, Delta=%u)\n",
               total_clauses, pool->count, n_kf, n_df);
    }

    double t_store = now_sec() - t0;
    printf("  (in %.2f sec)\n", t_store);

    /* Save if requested */
    if (ba.save_path) {
        SpaiStatus ss;
        if (ba.load_path && strcmp(ba.load_path, ba.save_path) == 0) {
            ss = ai_save_incremental(ai, ba.save_path);
            printf("  [save] incremental -> '%s' : %s (KF %u->%u, Delta %u->%u)\n",
                   ba.save_path, spai_status_str(ss),
                   loaded_kf, ai->kf_count, loaded_df, ai->df_count);
        } else {
            ss = ai_save(ai, ba.save_path);
            printf("  [save] full -> '%s' : %s (%u KF, %u Delta)\n",
                   ba.save_path, spai_status_str(ss),
                   ai->kf_count, ai->df_count);
        }
    }
    printf("\n");

    /* ── [3/4] Evaluate on held-out QAs ── */
    printf("[3/4] Evaluating %d held-out QAs (context match + B*G*R answer extraction)...\n", test_n);
    t0 = now_sec();

    SpatialGrid*  q_grid = grid_create();
    InputSignature q_sig;

    int ctx_hits   = 0;
    int em_hits    = 0;
    double f1_sum  = 0.0;
    int answered   = 0;

    /* Show a few detailed examples */
    const int DETAIL_N = 5;
    int detail_printed = 0;

    for (int t = 0; t < test_n; t++) {
        int idx = train_n + t;
        ContextEntry* gold_ctx = &contexts[qa[idx].ctx_idx];
        if (!gold_ctx->trained || gold_ctx->clause_count == 0) continue;

        /* 1. Encode question */
        grid_clear(q_grid);
        layers_encode_clause(qa[idx].question, NULL, q_grid);
        update_rgb_directional(q_grid);
        input_signature_compute(&q_sig, q_grid);

        /* 2. Pool match: type jump → A → RG → BA → other */
        PoolMatchResult pr = pool_match(pool, q_grid, qa[idx].question);
        uint32_t best_entry = pr.subtitle_entry_id;
        float match_sim = pr.similarity;

        /* 3. Context retrieval hit: best_entry in [clause_start, clause_start+count) */
        int ctx_hit = (best_entry >= gold_ctx->clause_start &&
                       best_entry <  gold_ctx->clause_start + gold_ctx->clause_count);
        if (ctx_hit) ctx_hits++;

        /* 4. Extract answer over the pool window */
        char pred_ans[1024];
        extract_answer_pool(pool, best_entry, WINDOW_RADIUS, &q_sig,
                            pred_ans, sizeof(pred_ans));

        /* 5. EM / F1 */
        int em = compute_em(pred_ans, qa[idx].answer);
        double f1 = compute_f1(pred_ans, qa[idx].answer);
        if (em) em_hits++;
        f1_sum += f1;
        answered++;

        if (detail_printed < DETAIL_N) {
            char pp[120], qp[120], ap[120];
            strncpy(qp, qa[idx].question, 116); qp[116] = '\0';
            if (strlen(qa[idx].question) > 116) strcat(qp, "...");
            strncpy(pp, pred_ans, 116); pp[116] = '\0';
            if (strlen(pred_ans) > 116) strcat(pp, "...");
            strncpy(ap, qa[idx].answer, 116); ap[116] = '\0';
            if (strlen(qa[idx].answer) > 116) strcat(ap, "...");
            printf("    [%d] %s\n", t, qp);
            printf("        ctx_hit=%s  sim=%.1f%%  em=%d  f1=%.2f\n",
                   ctx_hit ? "Y" : "N", match_sim * 100, em, f1);
            printf("        pred: %s\n", pp);
            printf("        gold: %s\n\n", ap);
            detail_printed++;
        } else if ((t + 1) % 20 == 0 || t + 1 == test_n) {
            printf("\r  Evaluated: %d / %d", t + 1, test_n);
        }
    }
    printf("\n");

    grid_destroy(q_grid);

    double t_eval = now_sec() - t0;
    printf("  Done in %.2f sec (%.1f queries/sec)\n\n",
           t_eval, (double)answered / (t_eval + 1e-9));

    /* ── [4/4] Report ── */
    printf("[4/4] Results\n");
    printf("========================================\n\n");

    printf("  SETUP\n");
    printf("  -------------------------------------\n");
    printf("  Total QA pairs:          %d\n", n_qa);
    printf("  Train pairs:             %d\n", train_n);
    printf("  Test pairs:              %d  (answered: %d)\n", test_n, answered);
    printf("  Unique contexts:         %d\n", n_contexts);
    {
        uint32_t n_kf = 0, n_df = 0;
        for (uint32_t i = 0; i < pool->count; i++) {
            if (!pool->canvases[i]->classified) continue;
            if (pool->canvases[i]->frame_type == CANVAS_IFRAME) n_kf++;
            else n_df++;
        }
        printf("  Canvases (KF/Delta):     %u (KF=%u, Delta=%u)\n",
               pool->count, n_kf, n_df);
        printf("  Pool slots:              %u\n\n", pool_total_slots(pool));
    }

    if (answered > 0) {
        double ctx_acc = 100.0 * ctx_hits / answered;
        double em_acc  = 100.0 * em_hits  / answered;
        double f1_avg  = 100.0 * f1_sum   / answered;

        printf("  METRICS\n");
        printf("  -------------------------------------\n");
        printf("  Context Retrieval:  %.2f%%  (%d / %d)\n",
               ctx_acc, ctx_hits, answered);
        printf("  Answer EM:          %.2f%%  (%d / %d)\n",
               em_acc,  em_hits,  answered);
        printf("  Answer F1:          %.2f\n\n", f1_avg);

        printf("  INTERPRETATION\n");
        printf("  -------------------------------------\n");
        if (ctx_acc > 80) printf("  Strong context retrieval.\n");
        else if (ctx_acc > 40) printf("  Moderate context retrieval.\n");
        else if (ctx_acc > 10) printf("  Weak context retrieval.\n");
        else printf("  Poor context retrieval.\n");

        if (f1_avg > 30) printf("  Useful answer extraction signal.\n");
        else if (f1_avg > 10) printf("  Measurable F1 signal.\n");
        else printf("  Low F1 (extraction still B*G*R, not keyword match).\n");
        printf("\n");
    }

    printf("========================================\n");
    printf("  PASS\n");
    printf("========================================\n");

    /* cleanup */
    for (int i = 0; i < n_contexts; i++) free(contexts[i].text);
    free(contexts);
    for (int i = 0; i < n_qa; i++) { free(qa[i].question); free(qa[i].answer); }
    free(qa);
    spatial_ai_destroy(ai);
    return 0;
}
