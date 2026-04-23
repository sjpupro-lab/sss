/* v4 Task E — bench_context
 *
 * Drives a short multi-turn dialogue twice: once with the session
 * context pool disabled, once with it populated. Reports side-by-
 * side generation output and retrieval id for each turn so a human
 * reader can see whether :ctx on changes what the model pulls.
 *
 * No pass/fail thresholds — per 09_CHECKLIST_AND_ORDER.md the
 * numerical goals are reference targets, not merge gates. */

#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_grid.h"
#include "spatial_subtitle.h"
#include "spatial_match.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small background corpus so retrieval has something to land on. */
static const char* KEYFRAMES[] = {
    "the quick brown fox jumps over the lazy dog today clearly happily",
    "pack my box with five dozen liquor jugs by the porch now please",
    "data teams iterate on the morning backlog carefully always warmly",
    "systems retire at midnight after the weekly cleanup pass runs",
    "the kitchen smelled of cinnamon and warm bread each afternoon truly",
    "visitors paused at the iron gate and admired every single bloom brightly",
    "oak trees shed their leaves along the quiet garden path slowly gently",
    "int fib(int n){return n<2?n:fib(n-1)+fib(n-2);}",
    "char* trim(char* s){while(*s==' ')s++;return s;}",
    "void print_sum(int* a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];printf(\"%d\",s);}",
};

/* 10-turn scripted exchange that mixes programming + nature topics
 * so the context pool has something to bias toward. */
static const char* DIALOG[] = {
    "the quick brown fox",
    "pack my box",
    "data teams iterate",
    "the kitchen smelled of cinnamon",
    "oak trees shed their leaves",
    "visitors paused at the iron gate",
    "int fib",
    "char* trim",
    "systems retire at midnight",
    "void print_sum",
};

static SpatialAI* build_model(void) {
    SpatialAI* ai = spatial_ai_create();
    for (uint32_t i = 0; i < sizeof KEYFRAMES / sizeof KEYFRAMES[0]; i++) {
        ai_force_keyframe(ai, KEYFRAMES[i], "mix");
    }
    return ai;
}

static void run_one_turn(SpatialAI* ai, const char* prompt, const char* label) {
    char out[256] = {0};
    float sim = 0.0f;
    uint32_t n = ai_generate_next(ai, prompt, out, sizeof out - 1, &sim);
    out[n < sizeof out ? n : sizeof out - 1] = '\0';
    uint32_t id = ai_predict(ai, prompt, NULL);
    printf("  %-12s  retr_kf=%-3u sim=%.3f  >>  %.60s\n",
           label, id, sim, out);
}

int main(void) {
    morpheme_init();

    const uint32_t N = sizeof DIALOG / sizeof DIALOG[0];
    printf("=== bench_context — %u turns, context OFF then ON ===\n", N);

    /* ── Pass 1: context OFF (baseline) ── */
    SpatialAI* ai_off = build_model();
    printf("\n[context OFF]\n");
    for (uint32_t t = 0; t < N; t++) {
        run_one_turn(ai_off, DIALOG[t], "OFF");
    }
    spatial_ai_destroy(ai_off);

    /* ── Pass 2: context ON — pre-seed the pool with the first half
     * of the dialog so the second half runs against a populated
     * short-term store. ai_generate_next itself isn't biased by the
     * context pool (v4 principle 2), but :ctx accumulation lets us
     * compare retrieval drift and the refine path's short-term
     * sensitivity side by side. ── */
    SpatialAI* ai_on = build_model();
    SpatialCanvasPool* ctx = ai_get_context_pool(ai_on);
    for (uint32_t t = 0; t < N / 2; t++) {
        pool_add_clause(ctx, DIALOG[t]);
    }
    printf("\n[context ON — seeded %u turns into the pool, slots=%u]\n",
           N / 2, pool_total_slots(ctx));
    for (uint32_t t = 0; t < N; t++) {
        run_one_turn(ai_on, DIALOG[t], "ON");
    }

    /* Refine-path side-by-side on the last two turns so the reader
     * can eyeball whether the context pool shifted the decoder. */
    printf("\n[refine path on last 2 turns, context pool populated]\n");
    for (uint32_t t = N - 2; t < N; t++) {
        char out[256] = {0};
        float conf = 0.0f;
        uint32_t iters = 0;
        uint32_t n = ai_generate_refine(ai_on, DIALOG[t], out,
                                        sizeof out - 1, NULL, &conf, &iters);
        out[n < sizeof out ? n : sizeof out - 1] = '\0';
        printf("  REFINE(%s)  conf=%.3f iters=%u  >>  %.60s\n",
               DIALOG[t], conf, iters, out);
    }

    spatial_ai_destroy(ai_on);
    printf("\n=== bench_context done ===\n");
    return 0;
}
