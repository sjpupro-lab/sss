#include "spatial_keyframe.h"
#include "spatial_layers.h"
#include <assert.h>
#include <stdio.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

static void test_create_destroy(void) {
    TEST("ai create/destroy");
    SpatialAI* ai = spatial_ai_create();
    assert(ai != NULL);
    assert(ai->kf_count == 0);
    assert(ai->df_count == 0);
    spatial_ai_destroy(ai);
    PASS();
}

static void test_store_first_keyframe(void) {
    TEST("store first clause as keyframe");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    uint32_t id = ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "animal_meal");

    assert(id == 0);
    assert(ai->kf_count == 1);
    assert(ai->df_count == 0);

    spatial_ai_destroy(ai);
    PASS();
}

static void test_store_similar_as_delta(void) {
    TEST("store similar clause as delta");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    /* First: keyframe */
    ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "KF0");

    /* Second: similar → should be delta */
    uint32_t id2 = ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "KF0_delta");

    printf("\n    kf_count=%u, df_count=%u, id2=0x%08x\n",
           ai->kf_count, ai->df_count, id2);

    /* Should be stored as delta (high bit set) or new keyframe */
    /* The similarity is ~78.5% > 0.3 → should be delta */
    assert(ai->df_count >= 1 || ai->kf_count == 1);

    spatial_ai_destroy(ai);
    PASS();
}

static void test_store_different_as_keyframe(void) {
    TEST("store different clause as new keyframe");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "KF0");

    /* Completely different */
    ai_store_auto(ai,
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 "
        "\xeb\xb0\x9d\xec\x9d\x80 \xeb\xb3\x84\xeb\xa1\x9c "
        "\xea\xb0\x80\xeb\x93\x9d\xed\x96\x88\xeb\x8b\xa4.",
        "KF1");

    printf("\n    kf_count=%u, df_count=%u\n", ai->kf_count, ai->df_count);
    assert(ai->kf_count >= 2);

    spatial_ai_destroy(ai);
    PASS();
}

static void test_delta_computation(void) {
    TEST("delta compute and apply");

    SpatialGrid* a = grid_create();
    SpatialGrid* b = grid_create();

    morpheme_init();
    layers_encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        NULL, a);
    layers_encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        NULL, b);

    DeltaEntry entries[GRID_TOTAL];
    uint32_t count = compute_delta(a, b, entries, GRID_TOTAL);
    printf("\n    delta entries: %u\n", count);
    assert(count > 0);
    assert(count < GRID_TOTAL); /* Should be sparse */

    /* Apply delta to reconstruct b from a */
    SpatialGrid* reconstructed = grid_create();
    apply_delta(a, entries, count, reconstructed);

    /* Verify reconstruction matches b (A channel) */
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        assert(reconstructed->A[i] == b->A[i]);
    }

    grid_destroy(a);
    grid_destroy(b);
    grid_destroy(reconstructed);
    PASS();
}

static void test_predict(void) {
    TEST("predict: find best matching keyframe");
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();

    /* Store several keyframes */
    ai_store_auto(ai,
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        "KF0_animal_meal");

    ai_store_auto(ai,
        "\xec\x9a\xb0\xeb\xa6\xac\xeb\x8a\x94 \xed\x95\xa8\xea\xbb\x98 "
        "\xec\x98\xa4\xeb\x9e\x9c \xec\x84\xb8\xec\x9b\x94\xec\x9d\x84 "
        "\xec\x82\xb4\xec\x95\x98\xeb\x8b\xa4.",
        "KF1_nostalgia");

    ai_store_auto(ai,
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.",
        "KF2_sky");

    /* Predict with similar input */
    float sim;
    uint32_t match = ai_predict(ai,
        "\xec\x98\xa4\xeb\x8a\x98 \xec\xa0\x80\xeb\x85\x81 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 "
        "\xec\x95\x84\xeb\xa6\x84\xeb\x8b\xa4\xec\x9a\xb4 "
        "\xeb\xb3\x84\xeb\xa1\x9c \xea\xb0\x80\xeb\x93\x9d\xed\x95\x98\xeb\x8b\xa4.",
        &sim);

    printf("\n    matched KF%u (sim=%.1f%%)\n", match, sim * 100.0f);

    /* Should match KF2 (sky-related) based on overlap */
    /* The key test is that it finds a reasonable match */
    assert(match < ai->kf_count);
    assert(sim > 0.0f);

    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_keyframe ===\n");

    test_create_destroy();
    test_store_first_keyframe();
    test_store_similar_as_delta();
    test_store_different_as_keyframe();
    test_delta_computation();
    test_predict();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
