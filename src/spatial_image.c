/* spatial_image — PPM P6 reader/writer for the SLIG image bridge.
 *
 * Hosts image_to_grid / grid_to_image used by the higher-level
 * SpatialAI engine. The legacy refine-based prototype generator was
 * removed; ai_generate_image_v2_guided (spatial_image_gen.c) is the
 * current entry point. */

#include "spatial_image.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── PPM P6 parsing ──
 *
 * Header form:
 *   P6\n
 *   [# comments\n]*
 *   <width> <height>\n
 *   <maxval>\n
 *   <width*height*3 raw bytes>
 *
 * Tokens are whitespace-separated; comments (# ... \n) may appear
 * between tokens. maxval must be 255 for our fixed-8-bit reader. */

/* Skip whitespace and '# ... \n' comments in the PPM header. */
static int skip_ws_and_comments(FILE* fp) {
    int c;
    for (;;) {
        c = fgetc(fp);
        if (c == EOF) return EOF;
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') { /* nop */ }
            if (c == EOF) return EOF;
            continue;
        }
        ungetc(c, fp);
        return 0;
    }
}

static int read_ppm_int(FILE* fp, int* out) {
    if (skip_ws_and_comments(fp) == EOF) return 0;
    int v = 0, digits = 0, c;
    while ((c = fgetc(fp)) != EOF && isdigit(c)) {
        v = v * 10 + (c - '0');
        digits++;
    }
    if (c != EOF) ungetc(c, fp);
    if (!digits) return 0;
    *out = v;
    return 1;
}

SpatialGrid* image_to_grid(const char* path) {
    if (!path) return NULL;
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Magic */
    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || strcmp(magic, "P6") != 0) {
        fclose(fp); return NULL;
    }

    int w = 0, h = 0, maxv = 0;
    if (!read_ppm_int(fp, &w) || !read_ppm_int(fp, &h) ||
        !read_ppm_int(fp, &maxv)) {
        fclose(fp); return NULL;
    }
    /* Exactly one whitespace byte separates header from pixel data. */
    int c = fgetc(fp);
    if (c == EOF || !isspace(c)) { fclose(fp); return NULL; }

    if (w != GRID_SIZE || h != GRID_SIZE || maxv != 255) {
        /* Prototype is fixed-resolution. Callers should pre-resize. */
        fclose(fp);
        return NULL;
    }

    SpatialGrid* g = grid_create();
    if (!g) { fclose(fp); return NULL; }

    uint8_t rgb[3];
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            if (fread(rgb, 1, 3, fp) != 3) {
                grid_destroy(g);
                fclose(fp);
                return NULL;
            }
            uint32_t i = y * GRID_SIZE + x;
            g->R[i] = rgb[0];
            g->G[i] = rgb[1];
            g->B[i] = rgb[2];
            /* A ← luminance (BT.601 weights), scaled to grid's u16 A range.
             * Keeping A in 8-bit byte-range (0..255) lets downstream code
             * that reads A like "A > 0 means presence" still work. */
            uint32_t lum = (uint32_t)(0.299 * rgb[0] +
                                      0.587 * rgb[1] +
                                      0.114 * rgb[2] + 0.5);
            if (lum > 255) lum = 255;
            g->A[i] = (uint16_t)lum;
        }
    }

    fclose(fp);
    return g;
}

int grid_to_image(const SpatialGrid* g, const char* out_path) {
    if (!g || !out_path) return 0;
    FILE* fp = fopen(out_path, "wb");
    if (!fp) return 0;

    if (fprintf(fp, "P6\n%u %u\n255\n", GRID_SIZE, GRID_SIZE) < 0) {
        fclose(fp); return 0;
    }

    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            uint8_t rgb[3] = { g->R[i], g->G[i], g->B[i] };
            if (fwrite(rgb, 1, 3, fp) != 3) {
                fclose(fp); return 0;
            }
        }
    }
    if (fclose(fp) != 0) return 0;
    return 1;
}

