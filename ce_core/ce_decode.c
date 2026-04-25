#include "ce_decode.h"
#include <string.h>
#include <math.h>

static uint8_t clamp_u8(int v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

/* Map signed int64_t channel value into [0,255] using a tanh-like squash
 * so that very large magnitudes saturate gracefully. */
static uint8_t squash(int64_t v) {
    /* simple linear squash with clamp; division avoids float libs in hot path */
    int64_t scaled = v / 64; /* heuristic — keeps typical CE outputs visible */
    if (scaled > 127)  scaled = 127;
    if (scaled < -128) scaled = -128;
    return (uint8_t)((int)scaled + 128);
}

void ce_decode_image(CEImage *out, const CELatentGrid *z) {
    out->width = CE_IMAGE_W;
    out->height = CE_IMAGE_H;

    /* Per-cell base colour. */
    uint8_t base_r[CE_GRID_N], base_g[CE_GRID_N], base_b[CE_GRID_N], base_a[CE_GRID_N];
    for (int i = 0; i < CE_GRID_N; ++i) {
        base_r[i] = squash(ce_read(&z->cells[i], CE_CH_R));
        base_g[i] = squash(ce_read(&z->cells[i], CE_CH_G));
        base_b[i] = squash(ce_read(&z->cells[i], CE_CH_B));
        base_a[i] = squash(ce_read(&z->cells[i], CE_CH_A));
        if (base_a[i] < 16) base_a[i] = 255; /* avoid fully transparent default */
    }

    /* Bilinear interpolation between cell centres. Each cell occupies an
     * 8x8 block; the centre of cell (cx, cy) maps to pixel (cx*8+4, cy*8+4). */
    for (int py = 0; py < CE_IMAGE_H; ++py) {
        float fy = ((float)py - 4.0f) / (float)CE_BLOCK;
        int   y0 = (int)floorf(fy);
        float ty = fy - (float)y0;
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        if (y0 >= CE_GRID_H - 1) { y0 = CE_GRID_H - 2; ty = 1.0f; }

        for (int px = 0; px < CE_IMAGE_W; ++px) {
            float fx = ((float)px - 4.0f) / (float)CE_BLOCK;
            int   x0 = (int)floorf(fx);
            float tx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            if (x0 >= CE_GRID_W - 1) { x0 = CE_GRID_W - 2; tx = 1.0f; }

            int i00 = y0 * CE_GRID_W + x0;
            int i10 = y0 * CE_GRID_W + (x0 + 1);
            int i01 = (y0 + 1) * CE_GRID_W + x0;
            int i11 = (y0 + 1) * CE_GRID_W + (x0 + 1);

            float w00 = (1.0f - tx) * (1.0f - ty);
            float w10 = tx * (1.0f - ty);
            float w01 = (1.0f - tx) * ty;
            float w11 = tx * ty;

            float r = base_r[i00] * w00 + base_r[i10] * w10
                    + base_r[i01] * w01 + base_r[i11] * w11;
            float g = base_g[i00] * w00 + base_g[i10] * w10
                    + base_g[i01] * w01 + base_g[i11] * w11;
            float bb = base_b[i00] * w00 + base_b[i10] * w10
                     + base_b[i01] * w01 + base_b[i11] * w11;
            float a = base_a[i00] * w00 + base_a[i10] * w10
                    + base_a[i01] * w01 + base_a[i11] * w11;

            CEPixel *p = &out->pixels[py * CE_IMAGE_W + px];
            p->r = clamp_u8((int)(r + 0.5f));
            p->g = clamp_u8((int)(g + 0.5f));
            p->b = clamp_u8((int)(bb + 0.5f));
            p->a = clamp_u8((int)(a + 0.5f));
        }
    }
}

void ce_decode_text(uint8_t *out, uint32_t *out_len,
                    const CELatentGrid *z) {
    /* Map each cell to one printable ASCII char (32..126). */
    static const char palette[] =
        " .,:;-+*?!#$%&/\\()[]{}<>|=_'\"`~^@"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";
    int psize = (int)(sizeof(palette) - 1);

    int n = CE_GRID_N;
    for (int i = 0; i < n; ++i) {
        int64_t v = ce_read(&z->cells[i], CE_CH_R)
                  ^ ce_read(&z->cells[i], CE_CH_G);
        if (v < 0) v = -v;
        out[i] = (uint8_t)palette[(uint32_t)v % (uint32_t)psize];
    }
    *out_len = (uint32_t)n;
}
