/*
 * chat.c — interactive chat REPL for a trained SPATIAL-PATTERN-AI model.
 *
 * Capabilities:
 *   - load a .spai model (or train from text)
 *   - per-turn retrieval (ai_predict)  → closest keyframe + similarity
 *   - per-turn generation (ai_generate_next) → next-frame UTF-8 decode
 *   - top-K candidate list (spatial_match MATCH_PREDICT.topk)
 *
 * Usage:
 *   ./build/chat --load build/models/wiki5k.spai
 *   ./build/chat --train data/wiki5k.txt --max 5000
 *
 * In-session commands:
 *   :q            quit
 *   :topk [N]     show top-N matches for the last input (default 5)
 *   :gen          generate next-frame response (default mode)
 *   :both         show retrieval + generation
 *   :retr         show retrieval only
 *   :help         show commands
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

#define LINE_BUF    8192
#define OUT_BUF     4096
#define MIN_INPUT   1

enum Mode { MODE_GEN = 0, MODE_RETR = 1, MODE_BOTH = 2 };

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [--load <path>] [--train <path> [--max N]]\n"
        "  --load  <path>    load a .spai model\n"
        "  --train <path>    train from a line-per-clause text file\n"
        "  --max   <N>       cap training clauses (default 5000)\n",
        prog);
}

static SpatialAI* load_model(const char* path) {
    SpaiStatus st = SPAI_OK;
    fprintf(stderr, "[chat] loading model: %s\n", path);
    double t0 = now_sec();
    SpatialAI* ai = ai_load(path, &st);
    if (!ai) {
        fprintf(stderr, "[chat] ai_load failed: %s\n", spai_status_str(st));
        return NULL;
    }
    fprintf(stderr, "[chat] loaded KF=%u Delta=%u in %.2fs\n",
            ai->kf_count, ai->df_count, now_sec() - t0);
    return ai;
}

static SpatialAI* train_model(const char* path, uint32_t max_clauses) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    SpatialAI* ai = spatial_ai_create();
    char line[LINE_BUF];
    uint32_t n = 0;
    double t0 = now_sec();
    fprintf(stderr, "[chat] training from %s (max %u)...\n", path, max_clauses);
    while (n < max_clauses && fgets(line, sizeof line, f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r' ||
                     line[L-1] == ' '  || line[L-1] == '\t')) line[--L] = 0;
        if (L < 10) continue;
        if (line[0] == '<') continue;
        ai_store_auto(ai, line, NULL);
        if (++n % 500 == 0) {
            fprintf(stderr, "  %u / %u  KF=%u Delta=%u\n",
                    n, max_clauses, ai->kf_count, ai->df_count);
        }
    }
    fclose(f);
    fprintf(stderr, "[chat] trained %u clauses in %.2fs (KF=%u, Delta=%u)\n",
            n, now_sec() - t0, ai->kf_count, ai->df_count);
    return ai;
}

static void print_topk(SpatialAI* ai, const char* text, uint32_t k) {
    SpatialGrid* g = grid_create();
    layers_encode_clause(text, NULL, g);
    update_rgb_directional(g);
    apply_ema_to_grid(ai, g);

    MatchContext ctx; memset(&ctx, 0, sizeof ctx);
    ctx.bucket_idx = &ai->bucket_idx;
    MatchResult r = spatial_match(ai, g, MATCH_PREDICT, &ctx);
    grid_destroy(g);

    if (k > r.topk_count) k = r.topk_count;
    printf("  top-%u matches:\n", k);
    for (uint32_t i = 0; i < k; i++) {
        uint32_t id = r.topk[i].id;
        if (id >= ai->kf_count) continue;
        const char* label = ai->keyframes[id].label;
        char decoded[256];
        uint32_t nb = grid_decode_text_utf8(&ai->keyframes[id].grid,
                                            decoded, sizeof decoded - 1);
        decoded[nb] = 0;
        printf("   %u. sim=%.4f  id=%u  label=\"%.40s\"\n       decode=\"%.80s\"\n",
               i + 1, r.topk[i].score, id,
               (label && label[0]) ? label : "(none)", decoded);
    }
}

static void do_generate(SpatialAI* ai, const char* text) {
    char out[OUT_BUF];
    float sim = 0.0f;
    double t0 = now_sec();
    uint32_t n = ai_generate_next(ai, text, out, sizeof out - 1, &sim);
    out[n < sizeof out ? n : sizeof out - 1] = 0;
    double dt = now_sec() - t0;
    if (n == 0) {
        printf("  [no response — empty generation]\n");
    } else {
        printf("  bot: %s\n", out);
    }
    printf("  (sim=%.4f, %u bytes, %.1f ms)\n", sim, n, dt * 1000.0);
}

static void do_retrieve(SpatialAI* ai, const char* text) {
    float sim = 0.0f;
    double t0 = now_sec();
    uint32_t id = ai_predict(ai, text, &sim);
    double dt = now_sec() - t0;
    if (id >= ai->kf_count) {
        printf("  [no retrieval hit]\n");
        return;
    }
    char decoded[256];
    uint32_t n = grid_decode_text_utf8(&ai->keyframes[id].grid, decoded,
                                       sizeof decoded - 1);
    decoded[n] = 0;
    printf("  retrieved kf=%u sim=%.4f (%.1f ms)\n    decode=\"%s\"\n",
           id, sim, dt * 1000.0, decoded);
}

static void print_help(void) {
    printf(
        "  :q            quit\n"
        "  :help         this message\n"
        "  :gen          switch to generate-only mode (default)\n"
        "  :retr         switch to retrieve-only mode\n"
        "  :both         show both retrieval and generation\n"
        "  :topk [N]     print top-N matches for the NEXT input (default 5)\n"
        "\n"
        "  Otherwise, type any text to query the model.\n");
}

int main(int argc, char** argv) {
#ifdef _WIN32
    setvbuf(stdout, NULL, _IONBF, 0);
#endif

    const char* load_path  = NULL;
    const char* train_path = NULL;
    uint32_t    max_clauses = 5000;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--load")  && i + 1 < argc) load_path  = argv[++i];
        else if (!strcmp(argv[i], "--train") && i + 1 < argc) train_path = argv[++i];
        else if (!strcmp(argv[i], "--max")   && i + 1 < argc) max_clauses = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!load_path && !train_path) { usage(argv[0]); return 2; }

    SpatialAI* ai = load_path ? load_model(load_path) : train_model(train_path, max_clauses);
    if (!ai) return 1;
    if (ai->kf_count == 0) {
        fprintf(stderr, "[chat] model has no keyframes; nothing to chat about\n");
        spatial_ai_destroy(ai);
        return 1;
    }

    int mode = MODE_GEN;
    int pending_topk = 0;
    int topk_n = 5;

    printf("\nCANVAS chat — KF=%u Delta=%u. Type :help for commands, :q to quit.\n\n",
           ai->kf_count, ai->df_count);
    fflush(stdout);

    char line[LINE_BUF];
    while (1) {
        printf("you> "); fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r' ||
                     line[L-1] == ' '  || line[L-1] == '\t')) line[--L] = 0;
        if (L == 0) continue;

        if (line[0] == ':') {
            if (!strcmp(line, ":q") || !strcmp(line, ":quit")) break;
            else if (!strcmp(line, ":help")) print_help();
            else if (!strcmp(line, ":gen"))  { mode = MODE_GEN;  printf("  [mode: gen]\n"); }
            else if (!strcmp(line, ":retr")) { mode = MODE_RETR; printf("  [mode: retr]\n"); }
            else if (!strcmp(line, ":both")) { mode = MODE_BOTH; printf("  [mode: both]\n"); }
            else if (!strncmp(line, ":topk", 5)) {
                int n = 5;
                if (line[5] == ' ') n = atoi(line + 6);
                if (n < 1) n = 1; if (n > 8) n = 8;
                topk_n = n; pending_topk = 1;
                printf("  [topk=%d armed — next input will print %d candidates]\n", n, n);
            }
            else printf("  unknown command: %s (try :help)\n", line);
            continue;
        }

        if (pending_topk) {
            print_topk(ai, line, (uint32_t)topk_n);
            pending_topk = 0;
        }

        if (mode == MODE_GEN)      do_generate(ai, line);
        else if (mode == MODE_RETR) do_retrieve(ai, line);
        else /* BOTH */            { do_retrieve(ai, line); do_generate(ai, line); }
    }

    spatial_ai_destroy(ai);
    printf("bye.\n");
    return 0;
}
