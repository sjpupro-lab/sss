#include "spatial_grid.h"

#ifdef _WIN32
#include <malloc.h>
static void* aligned_alloc_portable(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void aligned_free_portable(void* ptr) {
    _aligned_free(ptr);
}
#else
static void* aligned_alloc_portable(size_t alignment, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}
static void aligned_free_portable(void* ptr) {
    free(ptr);
}
#endif

SpatialGrid* grid_create(void) {
    SpatialGrid* g = (SpatialGrid*)malloc(sizeof(SpatialGrid));
    if (!g) return NULL;

    g->A = (uint16_t*)aligned_alloc_portable(32, GRID_TOTAL * sizeof(uint16_t));
    g->R = (uint8_t*)aligned_alloc_portable(32, GRID_TOTAL);
    g->G = (uint8_t*)aligned_alloc_portable(32, GRID_TOTAL);
    g->B = (uint8_t*)aligned_alloc_portable(32, GRID_TOTAL);

    if (!g->A || !g->R || !g->G || !g->B) {
        grid_destroy(g);
        return NULL;
    }

    memset(g->A, 0, GRID_TOTAL * sizeof(uint16_t));
    memset(g->R, 0, GRID_TOTAL);
    memset(g->G, 0, GRID_TOTAL);
    memset(g->B, 0, GRID_TOTAL);

    return g;
}

void grid_destroy(SpatialGrid* g) {
    if (!g) return;
    if (g->A) aligned_free_portable(g->A);
    if (g->R) aligned_free_portable(g->R);
    if (g->G) aligned_free_portable(g->G);
    if (g->B) aligned_free_portable(g->B);
    free(g);
}

void grid_clear(SpatialGrid* g) {
    if (!g) return;
    memset(g->A, 0, GRID_TOTAL * sizeof(uint16_t));
    memset(g->R, 0, GRID_TOTAL);
    memset(g->G, 0, GRID_TOTAL);
    memset(g->B, 0, GRID_TOTAL);
}

void grid_copy(SpatialGrid* dst, const SpatialGrid* src) {
    if (!dst || !src) return;
    memcpy(dst->A, src->A, GRID_TOTAL * sizeof(uint16_t));
    memcpy(dst->R, src->R, GRID_TOTAL);
    memcpy(dst->G, src->G, GRID_TOTAL);
    memcpy(dst->B, src->B, GRID_TOTAL);
}

void grid_encode(SpatialGrid* g, const char* text, uint16_t weight) {
    if (!g || !text) return;

    const uint8_t* bytes = (const uint8_t*)text;
    uint32_t i = 0;

    while (bytes[i] != 0) {
        uint8_t v = bytes[i];
        uint32_t x = v;
        uint32_t y = i % GRID_SIZE;
        uint32_t idx = grid_idx(y, x);

        /* Saturating add for uint16 */
        uint32_t new_val = (uint32_t)g->A[idx] + weight;
        g->A[idx] = (new_val > 65535) ? 65535 : (uint16_t)new_val;

        i++;
    }
}

uint32_t grid_active_count(const SpatialGrid* g) {
    if (!g) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (g->A[i] > 0) count++;
    }
    return count;
}

uint32_t grid_total_brightness(const SpatialGrid* g) {
    if (!g) return 0;
    uint32_t total = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        total += g->A[i];
    }
    return total;
}

uint16_t grid_max_brightness(const SpatialGrid* g) {
    if (!g) return 0;
    uint16_t max_val = 0;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        if (g->A[i] > max_val) max_val = g->A[i];
    }
    return max_val;
}

void grid_print_stats(const SpatialGrid* g, const char* label) {
    if (!g) return;
    uint32_t active = grid_active_count(g);
    uint32_t total = grid_total_brightness(g);
    uint16_t max_b = grid_max_brightness(g);
    printf("  %s: %u active pixels, max brightness %u, total brightness %u\n",
           label ? label : "Grid", active, max_b, total);
}
