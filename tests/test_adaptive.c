#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)

/* ── 1. weight_init + invariants ── */
static void test_init_and_normalize(void) {
    TEST("weight_init / weight_normalize: sum == 4");
    ChannelWeight w;
    weight_init(&w);
    assert(fabsf(w.w_A - 1.0f) < 1e-6);
    assert(fabsf(w.w_R - 1.0f) < 1e-6);
    assert(fabsf(w.w_G - 1.0f) < 1e-6);
    assert(fabsf(w.w_B - 1.0f) < 1e-6);

    /* Perturb and renormalise */
    w.w_A = 3.0f; w.w_R = 1.0f; w.w_G = 1.0f; w.w_B = 1.0f;
    weight_normalize(&w);
    float s = w.w_A + w.w_R + w.w_G + w.w_B;
    printf("\n    after normalise: A=%.2f R=%.2f G=%.2f B=%.2f (sum=%.3f)\n",
           w.w_A, w.w_R, w.w_G, w.w_B, s);
    assert(fabsf(s - 4.0f) < 1e-4);
    assert(w.w_A > w.w_R);  /* ratios preserved */
    PASS();
}

/* ── 2. weight_update: winner gains, sum conserved ── */
static void test_weight_update_convergence(void) {
    TEST("weight_update: R dominates → w_R grows over steps");
    ChannelWeight w;
    weight_init(&w);

    /* Repeatedly reward R as the winning channel */
    for (int step = 0; step < 50; step++) {
        /* sim_R highest, others lower */
        weight_update(&w, 0.1f, 0.9f, 0.2f, 0.3f);
    }
    float s = w.w_A + w.w_R + w.w_G + w.w_B;
    printf("\n    after 50 R-wins: A=%.2f R=%.2f G=%.2f B=%.2f (sum=%.3f)\n",
           w.w_A, w.w_R, w.w_G, w.w_B, s);

    assert(fabsf(s - 4.0f) < 1e-3);
    assert(w.w_R > w.w_A);
    assert(w.w_R > w.w_G);
    assert(w.w_R > w.w_B);
    PASS();
}

/* ── 3. per-channel similarity helpers ── */
static void test_channel_sim_helpers(void) {
    TEST("channel_sim_A/R/G/B between encoded grids");
    morpheme_init();

    SpatialGrid* a = grid_create();
    SpatialGrid* b = grid_create();
    layers_encode_clause("the cat sits on the mat.",   NULL, a);
    layers_encode_clause("the dog sits on the mat.",   NULL, b);

    float sA = channel_sim_A(a, b);
    float sR = channel_sim_R(a, b);
    float sG = channel_sim_G(a, b);
    float sB = channel_sim_B(a, b);

    printf("\n    sim_A=%.3f  sim_R=%.3f  sim_G=%.3f  sim_B=%.3f\n",
           sA, sR, sG, sB);
    /* All should be in [0, 1]; similar clauses → not zero */
    assert(sA >= 0 && sA <= 1);
    assert(sR >= 0 && sR <= 1);
    assert(sG >= 0 && sG <= 1);
    assert(sB >= 0 && sB <= 1);
    assert(sA > 0.5f);  /* structurally similar */

    grid_destroy(a);
    grid_destroy(b);
    PASS();
}

/* ── 4. adaptive_score combines channels ── */
static void test_adaptive_score(void) {
    TEST("adaptive_score: default (1,1,1,1) matches averaged sims");
    morpheme_init();

    SpatialGrid* a = grid_create();
    SpatialGrid* b = grid_create();
    layers_encode_clause("alpha beta gamma delta.", NULL, a);
    layers_encode_clause("alpha beta gamma delta.", NULL, b);  /* identical */

    float s = adaptive_score(a, b, NULL);  /* NULL → default weights */
    printf("\n    identical pair adaptive_score=%.3f\n", s);
    assert(s > 0.9f);

    /* Now a totally different pair */
    SpatialGrid* c = grid_create();
    layers_encode_clause("totally different tokens elsewhere.", NULL, c);
    float s2 = adaptive_score(a, c, NULL);
    printf("    different pair  adaptive_score=%.3f\n", s2);
    assert(s2 < s);

    grid_destroy(a);
    grid_destroy(b);
    grid_destroy(c);
    PASS();
}

/* ── 5. SpatialAI.global_weights initialised to uniform ── */
static void test_spatial_ai_default_weights(void) {
    TEST("SpatialAI.global_weights starts uniform (1,1,1,1)");
    SpatialAI* ai = spatial_ai_create();
    assert(fabsf(ai->global_weights.w_A - 1.0f) < 1e-6);
    assert(fabsf(ai->global_weights.w_R - 1.0f) < 1e-6);
    assert(fabsf(ai->global_weights.w_G - 1.0f) < 1e-6);
    assert(fabsf(ai->global_weights.w_B - 1.0f) < 1e-6);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── 6. ai_store_auto triggers weight update ── */
static void test_ai_store_auto_updates_weights(void) {
    TEST("ai_store_auto adjusts weights after each store");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();
    ChannelWeight before = ai->global_weights;

    /* Store several related clauses so ai_store_auto produces deltas
       (triggering weight feedback). */
    const char* clauses[] = {
        "the quick brown fox jumps over the lazy dog.",
        "the quick brown fox leaps over the lazy dog.",
        "the quick grey fox jumps over the lazy dog.",
        "the quick brown fox jumps over the sleepy dog.",
        "the slow brown fox jumps over the lazy dog.",
        "a quick brown fox jumps over the lazy cat.",
        "the quick red fox jumps over the lazy pig.",
        NULL
    };
    for (int i = 0; clauses[i]; i++) ai_store_auto(ai, clauses[i], NULL);

    ChannelWeight after = ai->global_weights;
    float diff = fabsf(after.w_A - before.w_A) + fabsf(after.w_R - before.w_R)
               + fabsf(after.w_G - before.w_G) + fabsf(after.w_B - before.w_B);

    printf("\n    before:  A=%.2f R=%.2f G=%.2f B=%.2f\n",
           before.w_A, before.w_R, before.w_G, before.w_B);
    printf("    after:   A=%.2f R=%.2f G=%.2f B=%.2f\n",
           after.w_A, after.w_R, after.w_G, after.w_B);
    printf("    L1 change: %.3f\n", diff);
    assert(diff > 0.01f);  /* weights moved */

    /* Sum invariant preserved */
    float sum = after.w_A + after.w_R + after.w_G + after.w_B;
    assert(fabsf(sum - 4.0f) < 1e-3);

    spatial_ai_destroy(ai);
    PASS();
}

/* ── 7. weights round-trip through save/load (v2 format) ── */
static void test_weights_io_roundtrip(void) {
    TEST("ai_save / ai_load preserves global_weights");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    /* Train so weights drift */
    const char* clauses[] = {
        "first clause here.", "second clause there.", "third clause anywhere.",
        "first clause nearby.", "second clause far.", "third clause home.",
        NULL
    };
    for (int i = 0; clauses[i]; i++) ai_store_auto(ai, clauses[i], NULL);

    ChannelWeight saved = ai->global_weights;

    #ifdef _WIN32
    system("if not exist build mkdir build");
    #else
    system("mkdir -p build");
    #endif
    const char* path = "build/test_adaptive_weights.spai";
    SpaiStatus s = ai_save(ai, path);
    assert(s == SPAI_OK);

    spatial_ai_destroy(ai);

    SpaiStatus ls;
    SpatialAI* ai2 = ai_load(path, &ls);
    assert(ls == SPAI_OK);

    printf("\n    saved:  A=%.3f R=%.3f G=%.3f B=%.3f\n",
           saved.w_A, saved.w_R, saved.w_G, saved.w_B);
    printf("    loaded: A=%.3f R=%.3f G=%.3f B=%.3f\n",
           ai2->global_weights.w_A, ai2->global_weights.w_R,
           ai2->global_weights.w_G, ai2->global_weights.w_B);
    assert(fabsf(ai2->global_weights.w_A - saved.w_A) < 1e-4);
    assert(fabsf(ai2->global_weights.w_R - saved.w_R) < 1e-4);
    assert(fabsf(ai2->global_weights.w_G - saved.w_G) < 1e-4);
    assert(fabsf(ai2->global_weights.w_B - saved.w_B) < 1e-4);

    spatial_ai_destroy(ai2);
    remove(path);
    PASS();
}

/* ── 8. match_cascade_weighted uses weights ── */
static void test_match_cascade_weighted(void) {
    TEST("match_cascade_weighted returns valid ID with custom weights");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    for (int i = 0; i < 10; i++) {
        char buf[96];
        snprintf(buf, sizeof(buf), "clause number %d about various topics here.", i);
        ai_force_keyframe(ai, buf, NULL);
    }

    SpatialGrid* q = grid_create();
    layers_encode_clause("clause number 3 about various topics here.", NULL, q);

    /* Weight heavily on R — semantic channel */
    ChannelWeight w = { .w_A = 0.5f, .w_R = 2.5f, .w_G = 0.5f, .w_B = 0.5f };

    float sim;
    uint32_t id = match_cascade_weighted(ai, q, CASCADE_QA, &w, &sim);
    printf("\n    weighted CASCADE_QA → id=%u sim=%.3f\n", id, sim);
    assert(id < ai->kf_count);

    /* And with NULL (baseline) */
    uint32_t id0 = match_cascade_weighted(ai, q, CASCADE_QA, NULL, &sim);
    printf("    baseline CASCADE_QA → id=%u sim=%.3f\n", id0, sim);
    assert(id0 < ai->kf_count);

    grid_destroy(q);
    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_adaptive ===\n");

    test_init_and_normalize();
    test_weight_update_convergence();
    test_channel_sim_helpers();
    test_adaptive_score();
    test_spatial_ai_default_weights();
    test_ai_store_auto_updates_weights();
    test_weights_io_roundtrip();
    test_match_cascade_weighted();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
