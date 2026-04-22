#include "spatial_context.h"
#include "spatial_layers.h"
#include <assert.h>
#include <stdio.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

static void test_context_create(void) {
    TEST("context create/destroy");
    ContextManager* ctx = context_create();
    assert(ctx != NULL);
    assert(ctx->frame_count == 0);
    context_destroy(ctx);
    PASS();
}

static void test_add_frames(void) {
    TEST("add multiple frames");
    morpheme_init();
    ContextManager* ctx = context_create();
    SpatialAI* ai = spatial_ai_create();

    uint32_t f0 = context_add_frame(ctx, ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "animal_meal");

    uint32_t f1 = context_add_frame(ctx, ai,
        "\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 \xed\x95\xa8\xea\xbb\x98 "
        "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94\xec\x9d\x84 "
        "\xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
        "nostalgia");

    uint32_t f2 = context_add_frame(ctx, ai,
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
        "sky");

    printf("\n    frames: f0=%u, f1=%u, f2=%u, total=%u\n",
           f0, f1, f2, ctx->frame_count);

    assert(ctx->frame_count == 3);
    assert(f0 == 0);
    assert(f1 == 1);
    assert(f2 == 2);

    /* Check frame metadata */
    assert(ctx->frames[0].frame_id == 0);
    assert(ctx->frames[2].frame_id == 2);

    context_destroy(ctx);
    spatial_ai_destroy(ai);
    PASS();
}

static void test_lru_cache(void) {
    TEST("LRU cache: hit/miss/evict");
    FrameCache fc;
    cache_init(&fc);

    /* Create some grids */
    SpatialGrid* g0 = grid_create();
    SpatialGrid* g1 = grid_create();
    SpatialGrid* g2 = grid_create();

    cache_put(&fc, 0, g0);
    cache_put(&fc, 1, g1);
    cache_put(&fc, 2, g2);

    /* Should hit */
    assert(cache_get(&fc, 0) == g0);
    assert(cache_get(&fc, 1) == g1);
    assert(cache_get(&fc, 2) == g2);

    /* Miss */
    assert(cache_get(&fc, 99) == NULL);

    printf("\n    cache count=%u\n", fc.count);
    assert(fc.count == 3);

    grid_destroy(g0);
    grid_destroy(g1);
    grid_destroy(g2);
    PASS();
}

static void test_lru_eviction(void) {
    TEST("LRU cache: eviction with 4 slots");

    /* Simulate a small cache by filling CACHE_SIZE slots, then check eviction */
    FrameCache fc;
    cache_init(&fc);

    /* Fill 4 grids and access in pattern [0,1,3,0,1,0] */
    SpatialGrid* grids[4];
    for (int i = 0; i < 4; i++) {
        grids[i] = grid_create();
        cache_put(&fc, (uint32_t)i, grids[i]);
    }

    /* Access pattern */
    cache_get(&fc, 0);
    cache_get(&fc, 1);
    cache_get(&fc, 3);
    cache_get(&fc, 0);
    cache_get(&fc, 1);
    cache_get(&fc, 0);

    /* All 4 should still be cached since CACHE_SIZE=256 > 4 */
    int hits = 0;
    for (int i = 0; i < 4; i++) {
        if (cache_get(&fc, (uint32_t)i) != NULL) hits++;
    }
    printf("\n    hits after pattern: %d/4\n", hits);
    assert(hits == 4);

    for (int i = 0; i < 4; i++) grid_destroy(grids[i]);
    PASS();
}

static void test_match_engine(void) {
    TEST("integrated match_engine");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();
    FrameCache fc;
    cache_init(&fc);

    /* Store keyframes */
    ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "KF0");

    ai_store_auto(ai,
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
        "KF1");

    /* Input */
    SpatialGrid* input = grid_create();
    layers_encode_clause(
        "\xec\x98\xa4\xeb\x8a\x98 \xec\xa0\x80\xeb\x85\x81 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 "
        "\xec\x95\x84\xeb\xa6\x84\xeb\x8b\xa4\xec\x9a\xb4 "
        "\xeb\xb3\x84\xeb\xa1\x9c \xea\xb0\x80\xeb\x93\x9d\xed\x95\x98\xeb\x8b\xa4.",
        NULL, input);
    update_rgb_directional(input);

    float sim;
    uint32_t best = match_engine(ai, input, NULL, NULL, &fc, &sim);
    printf("\n    match_engine: KF%u (sim=%.1f%%)\n", best, sim * 100.0f);

    assert(best < ai->kf_count);

    grid_destroy(input);
    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_context ===\n");

    test_context_create();
    test_add_frames();
    test_lru_cache();
    test_lru_eviction();
    test_match_engine();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
