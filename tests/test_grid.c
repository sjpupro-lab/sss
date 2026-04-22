#include "spatial_grid.h"
#include <assert.h>
#include <stdio.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    printf("  [TEST] %s ... ", name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

static void test_create_destroy(void) {
    TEST("grid create/destroy");
    SpatialGrid* g = grid_create();
    assert(g != NULL);
    assert(g->A != NULL);
    assert(g->R != NULL);
    assert(g->G != NULL);
    assert(g->B != NULL);

    /* Should be zeroed */
    assert(grid_active_count(g) == 0);
    assert(grid_total_brightness(g) == 0);
    assert(grid_max_brightness(g) == 0);

    grid_destroy(g);
    PASS();
}

static void test_encode_basic(void) {
    TEST("grid encode basic ASCII");
    SpatialGrid* g = grid_create();

    /* Encode "AB" */
    grid_encode(g, "AB", 1);

    /* 'A'=65, pos 0 → grid[0][65] = 1 */
    assert(g->A[grid_idx(0, 65)] == 1);
    /* 'B'=66, pos 1 → grid[1][66] = 1 */
    assert(g->A[grid_idx(1, 66)] == 1);
    assert(grid_active_count(g) == 2);

    grid_destroy(g);
    PASS();
}

static void test_encode_weight(void) {
    TEST("grid encode with weight");
    SpatialGrid* g = grid_create();

    grid_encode(g, "A", 3);
    assert(g->A[grid_idx(0, 65)] == 3);

    /* Encode again, should add */
    grid_encode(g, "A", 2);
    assert(g->A[grid_idx(0, 65)] == 5);

    grid_destroy(g);
    PASS();
}

static void test_encode_korean(void) {
    TEST("grid encode Korean UTF-8");
    SpatialGrid* g = grid_create();

    /* Korean char "가" = 0xEA 0xB0 0x80 (3 bytes) */
    grid_encode(g, "\xea\xb0\x80", 1);

    assert(grid_active_count(g) >= 2); /* at least 2 distinct byte positions */
    assert(grid_total_brightness(g) == 3); /* 3 bytes, each weight 1 */

    grid_destroy(g);
    PASS();
}

static void test_copy(void) {
    TEST("grid copy");
    SpatialGrid* a = grid_create();
    SpatialGrid* b = grid_create();

    grid_encode(a, "test", 1);
    grid_copy(b, a);

    assert(grid_active_count(b) == grid_active_count(a));
    assert(grid_total_brightness(b) == grid_total_brightness(a));

    grid_destroy(a);
    grid_destroy(b);
    PASS();
}

static void test_clear(void) {
    TEST("grid clear");
    SpatialGrid* g = grid_create();

    grid_encode(g, "hello", 5);
    assert(grid_active_count(g) > 0);

    grid_clear(g);
    assert(grid_active_count(g) == 0);

    grid_destroy(g);
    PASS();
}

int main(void) {
    printf("=== test_grid ===\n");

    test_create_destroy();
    test_encode_basic();
    test_encode_weight();
    test_encode_korean();
    test_copy();
    test_clear();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
