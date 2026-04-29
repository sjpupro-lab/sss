/* verify_hybrid.c — hybrid_vae_roundtrip PSNR over a list of PPM images.
 *
 * Each input is a 256×256 PPM P6 (RGB, 8 bits per channel). For every
 * image:
 *   1. read PPM into 256×256×4 RGBA (alpha = 255)
 *   2. hybrid_vae_roundtrip(rgba) — encode + decode + PSNR
 *   3. print "<path> PSNR=<dB>"
 *
 * After processing every input the average PSNR across all files is
 * printed.
 *
 * Usage:
 *   verify_hybrid <img1.ppm> [<img2.ppm> ...]
 *
 * Returns 0 if every file produced a valid PSNR (finite, > 0), 1 if
 * any file failed to load or the encoder produced a degenerate result.
 */
#include "ce_core.h"
#include "ce_storage.h"
#include "ce_hybrid_vae.h"
#include "slig_codebook.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_DIM 256
#define IMG_PX  (IMG_DIM * IMG_DIM)

/* --- PPM P6 reader (256×256, maxval 255, no comments after maxval) --- */

static int skip_ws_and_comments(FILE *fp) {
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

static int read_ppm_int(FILE *fp, int *out) {
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

/* Load `path` into a fresh 256×256 RGBA buffer (alpha=255). Returns
 * NULL on parse error or if the image is not 256×256 / maxval 255.
 * Caller frees the returned buffer. */
static uint8_t *load_ppm_rgba_256(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "[verify_hybrid] %s: not a P6 PPM\n", path);
        fclose(fp);
        return NULL;
    }

    int w = 0, h = 0, maxv = 0;
    if (!read_ppm_int(fp, &w) || !read_ppm_int(fp, &h) ||
        !read_ppm_int(fp, &maxv)) {
        fprintf(stderr, "[verify_hybrid] %s: header parse error\n", path);
        fclose(fp);
        return NULL;
    }
    int sep = fgetc(fp);
    if (sep == EOF || !isspace(sep)) {
        fprintf(stderr, "[verify_hybrid] %s: missing header terminator\n", path);
        fclose(fp);
        return NULL;
    }
    if (w != IMG_DIM || h != IMG_DIM || maxv != 255) {
        fprintf(stderr,
                "[verify_hybrid] %s: must be %dx%d maxval 255 (got %dx%d maxval %d)\n",
                path, IMG_DIM, IMG_DIM, w, h, maxv);
        fclose(fp);
        return NULL;
    }

    uint8_t *rgba = (uint8_t *)malloc((size_t)IMG_PX * 4u);
    if (!rgba) { fclose(fp); return NULL; }
    uint8_t rgb[3];
    for (int i = 0; i < IMG_PX; ++i) {
        if (fread(rgb, 1, 3, fp) != 3) {
            fprintf(stderr,
                    "[verify_hybrid] %s: short read at pixel %d\n", path, i);
            free(rgba);
            fclose(fp);
            return NULL;
        }
        rgba[i * 4 + 0] = rgb[0];
        rgba[i * 4 + 1] = rgb[1];
        rgba[i * 4 + 2] = rgb[2];
        rgba[i * 4 + 3] = 255;
    }
    fclose(fp);
    return rgba;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <img1.ppm> [<img2.ppm> ...]\n", argv[0]);
        return 2;
    }

    /* One CEStorage and one codebook are reused across the batch so the
     * roundtrip mirrors the train-time path where every image lands in
     * the same store. */
    CEStorage storage;
    ce_storage_init(&storage, 256);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    HybridVAEConfig cfg;
    hybrid_vae_config_default(&cfg);

    double psnr_sum = 0.0;
    int    valid    = 0;
    int    bad      = 0;

    for (int i = 1; i < argc; ++i) {
        const char *path = argv[i];
        uint8_t *rgba = load_ppm_rgba_256(path);
        if (!rgba) { bad++; continue; }

        uint8_t *out_rgb = (uint8_t *)malloc((size_t)IMG_PX * 3u);
        if (!out_rgb) { free(rgba); bad++; continue; }

        uint32_t cid = (uint32_t)(0x10000u + (uint32_t)i);
        float psnr = hybrid_vae_roundtrip(out_rgb, &storage, &codebook,
                                          rgba, IMG_DIM, IMG_DIM,
                                          cid, &cfg);
        printf("%s\tPSNR=%.2f dB\n", path, psnr);

        if (isfinite(psnr) && psnr > 0.0f) {
            psnr_sum += psnr;
            ++valid;
        } else {
            ++bad;
        }

        free(out_rgb);
        free(rgba);
    }

    if (valid > 0) {
        printf("--\navg PSNR over %d image(s): %.2f dB\n",
               valid, psnr_sum / valid);
    }
    if (bad > 0) {
        fprintf(stderr, "[verify_hybrid] %d file(s) failed\n", bad);
    }

    ce_storage_free(&storage);
    return (bad == 0) ? 0 : 1;
}
