#ifndef SPATIAL_GRID_H
#define SPATIAL_GRID_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define GRID_SIZE 256
#define GRID_TOTAL (GRID_SIZE * GRID_SIZE)

/* 1D aligned memory layout (SPEC-ENGINE Phase A) */
typedef struct {
    uint16_t* A;  /* 32-byte aligned, brightness/frequency */
    uint8_t*  R;  /* 32-byte aligned, semantic */
    uint8_t*  G;  /* 32-byte aligned, function */
    uint8_t*  B;  /* 32-byte aligned, context */
} SpatialGrid;

/* Create a new grid with aligned memory */
SpatialGrid* grid_create(void);

/* Destroy a grid and free memory */
void grid_destroy(SpatialGrid* g);

/* Reset all channels to zero */
void grid_clear(SpatialGrid* g);

/* Copy grid src into dst */
void grid_copy(SpatialGrid* dst, const SpatialGrid* src);

/* Encode UTF-8 text into grid A-channel with given weight */
void grid_encode(SpatialGrid* g, const char* text, uint16_t weight);

/* Get active pixel count (A > 0) */
uint32_t grid_active_count(const SpatialGrid* g);

/* Get total brightness sum of A channel */
uint32_t grid_total_brightness(const SpatialGrid* g);

/* Get max brightness value in A channel */
uint16_t grid_max_brightness(const SpatialGrid* g);

/* Print grid stats to stdout */
void grid_print_stats(const SpatialGrid* g, const char* label);

/* Coordinate helpers: row-major 1D indexing */
static inline uint32_t grid_idx(uint32_t y, uint32_t x) {
    return y * GRID_SIZE + x;
}

#endif /* SPATIAL_GRID_H */
