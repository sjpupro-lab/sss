#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_context.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

/* Full pipeline test: encode → store → predict → match */
static void test_full_pipeline(void) {
    TEST("full pipeline: 8 keyframes + 5 queries");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    ContextManager* ctx = context_create();

    /* Store 8 keyframes via context manager */
    const char* clauses[] = {
        /* KF0 */ "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
                  "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        /* KF1 */ "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
                  "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        /* KF2 */ "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
                  "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
        /* KF3 */ "\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 \xed\x95\xa8\xea\xbb\x98 "
                  "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94\xec\x9d\x84 "
                  "\xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
        /* KF4 */ "\xec\xb0\xa9\xed\x95\x9c \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
                  "\xec\x9a\xb0\xec\x9c\xa0\xeb\xa5\xbc "
                  "\xeb\xa7\x88\xec\x8b\x9c\xeb\x8a\x94\xeb\x8b\xa4.",
        /* KF5 */ "\xec\x95\x84\xec\x9d\xb4\xea\xb0\x80 "
                  "\xec\xb1\x85\xec\x9d\x84 \xec\x9d\xbd\xeb\x8a\x94\xeb\x8b\xa4.",
        /* KF6 */ "\xeb\xb0\x94\xeb\x8b\xa4\xec\x97\x90\xec\x84\x9c "
                  "\xeb\xb0\x94\xeb\x9e\x8c\xec\x9d\xb4 \xeb\xb6\x88\xeb\x8b\xa4.",
        /* KF7 */ "\xec\xb9\x9c\xea\xb5\xac\xec\x99\x80 \xed\x95\xa8\xea\xbb\x98 "
                  "\xeb\x86\x80\xec\x95\x98\xeb\x8b\xa4.",
    };
    const char* topics[] = {
        "animal_meal", "animal_meal", "sky", "nostalgia",
        "animal_drink", "reading", "sea", "friend"
    };

    for (int i = 0; i < 8; i++) {
        context_add_frame(ctx, ai, clauses[i], topics[i]);
    }

    printf("\n    stored: kf=%u, delta=%u, frames=%u\n",
           ai->kf_count, ai->df_count, ctx->frame_count);

    /* 5 test queries */
    struct { const char* text; const char* desc; } queries[] = {
        {"\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
         "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
         "exact match KF0"},

        {"\xec\x98\xa4\xeb\x8a\x98 \xec\xa0\x80\xeb\x85\x81 "
         "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 "
         "\xec\x95\x84\xeb\xa6\x84\xeb\x8b\xa4\xec\x9a\xb4 "
         "\xeb\xb3\x84\xeb\xa1\x9c \xea\xb0\x80\xeb\x93\x9d\xed\x95\x98\xeb\x8b\xa4.",
         "should match KF2 (sky)"},

        {"\xec\xb0\xa9\xed\x95\x9c \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
         "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
         "should match animal meal"},

        {"\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 "
         "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94 "
         "\xed\x95\xa8\xea\xbb\x98 \xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
         "should match KF3 (nostalgia)"},

        {"\xeb\xb0\x94\xeb\x8b\xa4\xec\x97\x90 "
         "\xed\x81\xb0 \xed\x8c\x8c\xeb\x8f\x84\xea\xb0\x80 \xec\xb9\x9c\xeb\x8b\xa4.",
         "should match KF6 (sea)"},
    };

    for (int i = 0; i < 5; i++) {
        float sim;
        uint32_t match = ai_predict(ai, queries[i].text, &sim);
        printf("    Q%d [%s] -> KF%u (%.1f%%)\n",
               i, queries[i].desc, match, sim * 100.0f);
        assert(match < ai->kf_count);
    }

    context_destroy(ctx);
    spatial_ai_destroy(ai);
    PASS();
}

static void test_summation_conservation(void) {
    TEST("summation conservation law");

    SpatialGrid* combined = grid_create();
    LayerBitmaps* lb = layers_create();
    morpheme_init();

    layers_encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        lb, combined);

    uint32_t combined_total = grid_total_brightness(combined);
    uint32_t layer_sum = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        layer_sum += lb->base[i] + lb->word[i] + lb->morpheme[i];
    }

    printf("\n    combined=%u, layer_sum=%u\n", combined_total, layer_sum);
    assert(combined_total == layer_sum);

    grid_destroy(combined);
    layers_destroy(lb);
    PASS();
}

static void test_block_skip_accuracy(void) {
    TEST("block skip vs full cosine: 0% accuracy loss");

    SpatialGrid* grids[4];
    const char* texts[] = {
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
        "\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 \xed\x95\xa8\xea\xbb\x98 "
        "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94\xec\x9d\x84 "
        "\xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
    };

    morpheme_init();
    BlockSummary bs[4];

    for (int i = 0; i < 4; i++) {
        grids[i] = grid_create();
        layers_encode_clause(texts[i], NULL, grids[i]);
        compute_block_sums(grids[i], &bs[i]);
    }

    /* Compare all pairs */
    int all_match = 1;
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            float full = cosine_a_only(grids[i], grids[j]);
            float skip = cosine_block_skip(grids[i], grids[j], &bs[i], &bs[j]);
            float diff = fabsf(full - skip);
            printf("\n    KF%d vs KF%d: full=%.1f%% skip=%.1f%% diff=%.3f%%",
                   i, j, full * 100, skip * 100, diff * 100);
            if (diff > 0.001f) all_match = 0;
        }
    }
    printf("\n");
    assert(all_match);

    for (int i = 0; i < 4; i++) grid_destroy(grids[i]);
    PASS();
}

static void test_frame_similarity_matrix(void) {
    TEST("frame similarity matrix");
    morpheme_init();

    SpatialGrid* grids[3];
    const char* texts[] = {
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 \xed\x95\xa8\xea\xbb\x98 "
        "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94\xec\x9d\x84 "
        "\xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
    };

    for (int i = 0; i < 3; i++) {
        grids[i] = grid_create();
        layers_encode_clause(texts[i], NULL, grids[i]);
    }

    printf("\n    Similarity matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sim = cosine_a_only(grids[i], grids[j]);
            printf("    F%d vs F%d: %.1f%%\n", i, j, sim * 100.0f);
        }
    }

    /* F0 vs F1 should be low (different content) */
    float f0_f1 = cosine_a_only(grids[0], grids[1]);
    assert(f0_f1 < 0.3f);

    for (int i = 0; i < 3; i++) grid_destroy(grids[i]);
    PASS();
}

int main(void) {
    printf("=== test_integration ===\n");

    test_full_pipeline();
    test_summation_conservation();
    test_block_skip_accuracy();
    test_frame_similarity_matrix();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
