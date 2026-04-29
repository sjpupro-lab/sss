/* wave_debug.c — single-cell wave residual visualizer.
 *
 * Drops one synthetic CE Cell on an empty 256×256 canvas and writes a
 * grayscale PPM, so you can eyeball the wave shape before trusting it
 * inside the masked-train loop. The cell is built directly via
 * slig_pack so all SligSignal fields (dir, sigma, frequency, origin,
 * audio_amps[2], cond_threshold) are individually controllable from
 * the CLI.
 *
 * Usage:
 *   wave_debug <out.ppm> [--dir N] [--sigma U16] [--freq U8]
 *                        [--ox U8] [--oy U8] [--amp2 U8]
 *                        [--cond U8] [--scale 0|1|2]
 *                        [--decay U8] [--speed U8] [--tick U8]
 *
 * Defaults render a centered ripple. Pass --dir 0..5 for global wave
 * patterns, 6..9 for event-style (ripple/beam/cross/cascade).
 */

#include "slig_signal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <out.ppm> [options]\n"
            "  --dir N        SligDir 0..9 (default 6 = RIPPLE)\n"
            "  --sigma U16    amplitude 0..65535 (default 5000)\n"
            "  --freq U8      frequency 1..255 (default 16)\n"
            "  --ox U8        origin x 0..255 (default 128)\n"
            "  --oy U8        origin y 0..255 (default 128)\n"
            "  --amp2 U8      audio_amps[2] energy weight (default 255)\n"
            "  --cond U8      cond_threshold cumulative (default 0)\n"
            "  --scale N      0=COARSE, 1=MID, 2=FINE (default 1)\n"
            "  --decay U8     event decay 0..255 (default 230)\n"
            "  --speed U8     event radial speed 1..255 (default 4)\n"
            "  --tick U8      render tick for event (default 30)\n",
            prog);
}

static int save_ppm_gray(const char *path, const uint8_t *gray, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; ++i) {
        uint8_t rgb[3] = { gray[i], gray[i], gray[i] };
        if (fwrite(rgb, 1, 3, f) != 3) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        }

    const char *out_path = argv[1];

    SligSignal sig;
    memset(&sig, 0, sizeof(sig));
    sig.dir         = SLIG_DIR_RIPPLE;
    sig.scale       = SLIG_SCALE_MID;
    sig.sigma       = 5000;
    sig.frequency   = 16;
    sig.origin_x    = 128;
    sig.origin_y    = 128;
    sig.decay       = 230;
    sig.speed       = 4;
    sig.start_tick  = 0;
    sig.audio_amps[2] = 255;          /* full amp by default */
    sig.cond_threshold = 0;
    uint8_t render_tick = 30;

    for (int i = 2; i < argc; ++i) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "[wave_debug] missing value for %s\n", k); return 2; }
        if      (strcmp(k, "--dir"  ) == 0) { sig.dir         = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--sigma") == 0) { sig.sigma       = (uint16_t)atoi(v); ++i; }
        else if (strcmp(k, "--freq" ) == 0) { sig.frequency   = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--ox"   ) == 0) { sig.origin_x    = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--oy"   ) == 0) { sig.origin_y    = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--amp2" ) == 0) { sig.audio_amps[2] = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--cond" ) == 0) { sig.cond_threshold = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--scale") == 0) { sig.scale       = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--decay") == 0) { sig.decay       = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--speed") == 0) { sig.speed       = (uint8_t)atoi(v); ++i; }
        else if (strcmp(k, "--tick" ) == 0) { render_tick     = (uint8_t)atoi(v); ++i; }
        else {
            fprintf(stderr, "[wave_debug] unknown option: %s\n", k);
            usage(argv[0]);
            return 2;
        }
    }

    /* For global directions we leave u/v at zero — apply_global's
     * outer-product term then collapses to zero, which isn't useful for
     * a visual check. Stamp a deterministic sin/cos profile so global
     * cells produce a non-trivial canvas. */
    if (sig.dir < SLIG_DIR_RIPPLE) {
        for (int k = 0; k < SLIG_SIG_LEN; ++k) {
            sig.u[k] = (int16_t)(sinf((float)k * 0.10f) * 1000.0f);
            sig.v[k] = (int16_t)(cosf((float)k * 0.10f) * 1000.0f);
        }
    }

    CEUnit cell;
    slig_pack(&cell, &sig);

    SligCanvas canvas;
    slig_canvas_clear(&canvas);
    slig_apply_cell_at_tick(&canvas, &cell, render_tick);

    /* int32 canvas → uint8 PPM via min-max normalization so the wave
     * shape is visible regardless of sigma. */
    int32_t mn = canvas.pixels[0][0], mx = mn;
    for (int y = 0; y < SLIG_CANVAS_DIM; ++y)
        for (int x = 0; x < SLIG_CANVAS_DIM; ++x) {
            int32_t v = canvas.pixels[y][x];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    if (mx == mn) { mx = mn + 1; }   /* avoid zero range */

    uint8_t gray[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];
    for (int y = 0; y < SLIG_CANVAS_DIM; ++y)
        for (int x = 0; x < SLIG_CANVAS_DIM; ++x) {
            int64_t scaled = ((int64_t)(canvas.pixels[y][x] - mn) * 255)
                           / (int64_t)(mx - mn);
            if (scaled < 0)   scaled = 0;
            if (scaled > 255) scaled = 255;
            gray[y * SLIG_CANVAS_DIM + x] = (uint8_t)scaled;
        }

    if (!save_ppm_gray(out_path, gray, SLIG_CANVAS_DIM, SLIG_CANVAS_DIM)) {
        fprintf(stderr, "[wave_debug] failed to write %s\n", out_path);
        return 1;
    }

    fprintf(stderr,
            "[wave_debug] %s  dir=%u sigma=%u freq=%u origin=(%u,%u) "
            "amp[2]=%u cond=%u  canvas range=[%d..%d]\n",
            out_path, sig.dir, sig.sigma, sig.frequency,
            sig.origin_x, sig.origin_y,
            sig.audio_amps[2], sig.cond_threshold, mn, mx);
    return 0;
}
