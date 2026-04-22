#include "spatial_match.h"
#include "spatial_layers.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

static SpatialGrid* encode_clause(const char* text) {
    SpatialGrid* g = grid_create();
    morpheme_init();
    layers_encode_clause(text, NULL, g);
    update_rgb_directional(g);
    return g;
}

static void test_overlap(void) {
    TEST("overlap: similar vs different");

    /* Similar: 고양이가 밥을 먹는다 vs 강아지가 물을 먹는다 */
    SpatialGrid* a = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");
    SpatialGrid* b = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");
    /* Completely different */
    SpatialGrid* c = encode_clause(
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 \xeb\xb0\x9d\xeb\x8b\xa4.");

    uint32_t ov_ab = overlap_score(a, b);
    uint32_t ov_ac = overlap_score(a, c);

    printf("\n    overlap(similar)=%u, overlap(different)=%u\n", ov_ab, ov_ac);
    assert(ov_ab > ov_ac);

    grid_destroy(a);
    grid_destroy(b);
    grid_destroy(c);
    PASS();
}

static void test_cosine_similar(void) {
    TEST("cosine: similar clauses ~78%");

    SpatialGrid* a = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");
    SpatialGrid* b = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");

    float sim = cosine_a_only(a, b);
    printf("\n    cosine(similar) = %.1f%%\n", sim * 100.0f);
    assert(sim > 0.5f);  /* Should be around 78.5% */
    assert(sim < 1.0f);

    grid_destroy(a);
    grid_destroy(b);
    PASS();
}

static void test_cosine_different(void) {
    TEST("cosine: completely different ~0%");

    SpatialGrid* a = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");
    SpatialGrid* c = encode_clause(
        "\xec\x98\xa4\xeb\x8a\x98 \xec\x95\x84\xec\xb9\xa8 "
        "\xed\x95\x98\xeb\x8a\x98\xec\x9d\xb4 "
        "\xeb\xb0\x9d\xec\x9d\x80 \xeb\xb3\x84\xeb\xa1\x9c "
        "\xea\xb0\x80\xeb\x93\x9d\xed\x96\x88\xeb\x8b\xa4.");

    float sim = cosine_a_only(a, c);
    printf("\n    cosine(different) = %.1f%%\n", sim * 100.0f);
    assert(sim < 0.3f);

    grid_destroy(a);
    grid_destroy(c);
    PASS();
}

static void test_block_skip(void) {
    TEST("block skip: same result as full cosine");

    SpatialGrid* a = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");
    SpatialGrid* b = encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 \xea\xb0\x95\xec\x95\x84\xec\xa7\x80\xea\xb0\x80 "
        "\xeb\xac\xbc\xec\x9d\x84 \xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.");

    BlockSummary bs_a, bs_b;
    compute_block_sums(a, &bs_a);
    compute_block_sums(b, &bs_b);

    float full = cosine_a_only(a, b);
    float skip = cosine_block_skip(a, b, &bs_a, &bs_b);

    printf("\n    full=%.4f, block_skip=%.4f, diff=%.6f\n",
           full, skip, fabsf(full - skip));

    /* Should be equal or very close */
    assert(fabsf(full - skip) < 0.001f);

    /* Count empty blocks */
    int empty = 0;
    for (int by = 0; by < BLOCKS; by++)
        for (int bx = 0; bx < BLOCKS; bx++)
            if (bs_a.sum[by][bx] == 0 && bs_b.sum[by][bx] == 0)
                empty++;
    printf("    empty blocks: %d/256 (%.0f%% skip)\n", empty, empty * 100.0f / 256.0f);

    grid_destroy(a);
    grid_destroy(b);
    PASS();
}

static void test_topk(void) {
    TEST("topk selection");

    Candidate pool[5] = {
        {0, 1.0f}, {1, 5.0f}, {2, 3.0f}, {3, 2.0f}, {4, 4.0f}
    };
    topk_select(pool, 5, 3);

    assert(pool[0].score >= pool[1].score);
    assert(pool[1].score >= pool[2].score);
    assert(pool[0].id == 1); /* score 5 */

    PASS();
}

int main(void) {
    printf("=== test_match ===\n");

    test_overlap();
    test_cosine_similar();
    test_cosine_different();
    test_block_skip();
    test_topk();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
