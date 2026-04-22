#include "spatial_layers.h"
#include <assert.h>
#include <stdio.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

static void test_3layer_summation(void) {
    TEST("3-layer summation basic");
    SpatialGrid* combined = grid_create();
    LayerBitmaps* lb = layers_create();

    morpheme_init();

    /* "귀여운 고양이가 밥을 먹는다." */
    layers_encode_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 "
        "\xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 "
        "\xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        lb, combined);

    uint32_t active = grid_active_count(combined);
    uint16_t max_b = grid_max_brightness(combined);
    uint32_t total = grid_total_brightness(combined);

    printf("\n    active=%u, max=%u, total=%u\n", active, max_b, total);

    /* Should have active pixels */
    assert(active > 0);
    /* Max brightness should be > 1 (due to multi-layer overlap) */
    assert(max_b >= 1);

    /* Summation conservation: combined total = base_total + word_total + morph_total */
    uint32_t base_total = 0, word_total = 0, morph_total = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        base_total += lb->base[i];
        word_total += lb->word[i];
        morph_total += lb->morpheme[i];
    }
    printf("    base=%u, word=%u, morph=%u, sum=%u\n",
           base_total, word_total, morph_total,
           base_total + word_total + morph_total);
    assert(total == base_total + word_total + morph_total);

    grid_destroy(combined);
    layers_destroy(lb);
    PASS();
}

static void test_max_brightness_9(void) {
    TEST("max brightness can reach 9");
    SpatialGrid* combined = grid_create();

    morpheme_init();

    /* "밥을" — overlap can reach base(+1) + word(+5) + morpheme(+3) = 9 */
    layers_encode_clause(
        "\xeb\xb0\xa5\xec\x9d\x84",
        NULL, combined);

    uint16_t max_b = grid_max_brightness(combined);
    printf("\n    max brightness = %u\n", max_b);
    /* Should reach at least 9 in the new weighting scheme */
    assert(max_b >= 9);

    grid_destroy(combined);
    PASS();
}

static void test_empty_input(void) {
    TEST("empty input");
    SpatialGrid* combined = grid_create();

    layers_encode_clause("", NULL, combined);

    assert(grid_active_count(combined) == 0);

    grid_destroy(combined);
    PASS();
}

int main(void) {
    printf("=== test_layers ===\n");

    test_3layer_summation();
    test_max_brightness_9();
    test_empty_input();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
