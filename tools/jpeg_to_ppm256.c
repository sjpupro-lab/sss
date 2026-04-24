/* jpeg_to_ppm256 — decode a baseline JPEG and downsample to
 * GRID_SIZE × GRID_SIZE (256×256) PPM P6. Depends on libjpeg.
 *
 * Kept separate from the core engine: the image module itself is
 * PPM-only (Task F prototype scope). This tool is purely an ingestion
 * helper, matching tools/png_to_ppm256.py for PNG inputs.
 *
 * Usage: jpeg_to_ppm256 <in.jpg> <out.ppm> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

#include "spatial_grid.h"

static int decode_jpeg(const char* path, int* out_w, int* out_h, unsigned char** out_rgb) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "[jpeg] cannot open %s\n", path); return 0; }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    int w = (int)cinfo.output_width;
    int h = (int)cinfo.output_height;
    int stride = w * 3;
    unsigned char* rgb = (unsigned char*)malloc((size_t)stride * h);
    if (!rgb) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return 0;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = rgb + (size_t)cinfo.output_scanline * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    *out_w = w;
    *out_h = h;
    *out_rgb = rgb;
    return 1;
}

/* Box-average resize to GRID_SIZE×GRID_SIZE. Upsample regions collapse
 * to a single source pixel; downsample regions average their block. */
static void resize_to_grid(int w, int h, const unsigned char* src, unsigned char* dst) {
    for (uint32_t oy = 0; oy < GRID_SIZE; oy++) {
        uint32_t sy0 = (oy * (uint32_t)h) / GRID_SIZE;
        uint32_t sy1 = ((oy + 1) * (uint32_t)h) / GRID_SIZE;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (uint32_t ox = 0; ox < GRID_SIZE; ox++) {
            uint32_t sx0 = (ox * (uint32_t)w) / GRID_SIZE;
            uint32_t sx1 = ((ox + 1) * (uint32_t)w) / GRID_SIZE;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            uint32_t r = 0, g = 0, b = 0, n = 0;
            for (uint32_t sy = sy0; sy < sy1; sy++) {
                const unsigned char* row = src + (size_t)sy * w * 3;
                for (uint32_t sx = sx0; sx < sx1; sx++) {
                    r += row[sx * 3 + 0];
                    g += row[sx * 3 + 1];
                    b += row[sx * 3 + 2];
                    n++;
                }
            }
            unsigned char* o = dst + (oy * GRID_SIZE + ox) * 3;
            o[0] = (unsigned char)(r / n);
            o[1] = (unsigned char)(g / n);
            o[2] = (unsigned char)(b / n);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in.jpg> <out.ppm>\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0;
    unsigned char* rgb = NULL;
    if (!decode_jpeg(argv[1], &w, &h, &rgb)) return 1;

    fprintf(stderr, "[jpeg_to_ppm256] %s: %dx%d -> %ux%u\n",
            argv[1], w, h, GRID_SIZE, GRID_SIZE);

    unsigned char* out = (unsigned char*)malloc((size_t)GRID_SIZE * GRID_SIZE * 3);
    if (!out) { free(rgb); return 1; }
    resize_to_grid(w, h, rgb, out);
    free(rgb);

    FILE* fp = fopen(argv[2], "wb");
    if (!fp) { free(out); fprintf(stderr, "[jpeg] cannot write %s\n", argv[2]); return 1; }
    fprintf(fp, "P6\n%u %u\n255\n", GRID_SIZE, GRID_SIZE);
    fwrite(out, 1, (size_t)GRID_SIZE * GRID_SIZE * 3, fp);
    fclose(fp);
    free(out);
    fprintf(stderr, "[jpeg_to_ppm256] wrote %s\n", argv[2]);
    return 0;
}
