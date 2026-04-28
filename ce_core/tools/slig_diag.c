/* slig_diag.c — diagnostic measurements for the 3 SLIG-side TODOs.
 *
 * Runs slig_decompose_v2 on a small set of synthetic 64x64 images with
 * KNOWN ground-truth properties and reports the unpacked SligSignal
 * fields, so we can answer:
 *
 *   1. SVD direction:
 *        Is sig.dir always SLIG_DIR_HORIZONTAL today, regardless of the
 *        actual stripe orientation? If yes, this is the bug to fix.
 *        Also dump variance(u), variance(v) for the dominant component
 *        so we can pick a derivation rule.
 *
 *   2. cond_threshold:
 *        What value(s) does slig_decompose_v2 actually produce today
 *        (it currently goes out as 0 — the field is unset). Print the
 *        per-channel residual energy ratio after structure removal so we
 *        can derive a sensible default tied to real measurements.
 *
 *   3. dec.A audio_amps[2]:
 *        After Mat-1 / Mat-2 etc. attach, what energy_ratio shows up in
 *        audio_amps[2]? Per-image min/mean/max so the renderer-side amp
 *        modulation has a realistic input range to plan against.
 *
 * Output is human-readable; meant to be eyeballed once and then baked
 * into thresholds. There is NO assertion — this tool measures, does
 * not gate.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../slig_pipeline.h"
#include "../slig_signal.h"

static void fill_solid(SligImage *im, float v) {
    for (int i = 0; i < im->width * im->height; ++i) im->data[i] = v;
}
static void fill_horizontal_stripes(SligImage *im) {
    /* luma depends on Y only -> rows are uniform, columns vary */
    for (int y = 0; y < im->height; ++y) {
        float v = ((y / 8) & 1) ? 0.9f : 0.1f;
        for (int x = 0; x < im->width; ++x)
            im->data[y * im->width + x] = v;
    }
}
static void fill_vertical_stripes(SligImage *im) {
    for (int y = 0; y < im->height; ++y)
        for (int x = 0; x < im->width; ++x)
            im->data[y * im->width + x] = ((x / 8) & 1) ? 0.9f : 0.1f;
}
static void fill_diagonal_stripes(SligImage *im) {
    for (int y = 0; y < im->height; ++y)
        for (int x = 0; x < im->width; ++x)
            im->data[y * im->width + x] = (((x + y) / 8) & 1) ? 0.9f : 0.1f;
}
static void fill_radial(SligImage *im) {
    /* Centred bright disc, dark border. */
    int cx = im->width / 2, cy = im->height / 2;
    int r2 = (im->width / 4) * (im->width / 4);
    for (int y = 0; y < im->height; ++y)
        for (int x = 0; x < im->width; ++x) {
            int dx = x - cx, dy = y - cy;
            im->data[y * im->width + x] =
                (dx * dx + dy * dy < r2) ? 0.9f : 0.1f;
        }
}

static void uv_stats(const int16_t *q, int n, double *mean, double *var) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)q[i] / 10000.0;
    double m = s / n;
    double v = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)q[i] / 10000.0 - m;
        v += d * d;
    }
    *mean = m;
    *var  = v / n;
}

static const char *dir_name(uint8_t d) {
    switch (d) {
        case SLIG_DIR_HORIZONTAL: return "HORIZONTAL";
        case SLIG_DIR_VERTICAL:   return "VERTICAL";
        case SLIG_DIR_DIAG_DOWN:  return "DIAG_DOWN";
        case SLIG_DIR_DIAG_UP:    return "DIAG_UP";
        default: return "OTHER";
    }
}

static void run_one(const char *label, void (*fill)(SligImage *)) {
    int dim = 64;
    SligImage *im = slig_image_alloc(dim, dim, 1);
    fill(im);

    SligDecomposeConfig cfg;
    slig_decompose_config_default(&cfg);

    SligDecomposed d;
    memset(&d, 0, sizeof(d));
    slig_decompose_v2(&d, im, &cfg);

    printf("== %s (%u cells, structure_end=%u) ==\n",
           label, d.num_cells, d.structure_end);

    if (d.structure_end > 0) {
        SligSignal s;
        slig_unpack(&s, &d.cells[0]);

        double um, uv, vm, vv;
        uv_stats(s.u, dim, &um, &uv);
        uv_stats(s.v, dim, &vm, &vv);

        printf("  cell[0]: dir=%-10s sigma=%u  freq=%u\n",
               dir_name(s.dir), s.sigma, s.frequency);
        printf("           u: mean=%+.4f var=%.6f   (uniform => low var)\n",
               um, uv);
        printf("           v: mean=%+.4f var=%.6f\n", vm, vv);
        printf("           cond_threshold=%u  cond_type=%u\n",
               s.cond_threshold, s.cond_type);
        printf("           audio_amps=[%u %u %u %u]\n",
               s.audio_amps[0], s.audio_amps[1],
               s.audio_amps[2], s.audio_amps[3]);

        /* Proposed derivation rule:
         *   var(u) >> var(v)  =>  varies along Y => HORIZONTAL stripes
         *   var(v) >> var(u)  =>  varies along X => VERTICAL stripes
         *   ratio near 1      =>  DIAGONAL */
        double r = (vv > 1e-12) ? (uv / vv) : 1e9;
        const char *proposed;
        if (r > 4.0)        proposed = "HORIZONTAL";
        else if (r < 0.25)  proposed = "VERTICAL";
        else                proposed = "DIAG (or unstructured)";
        printf("           proposed dir from var ratio %.3g -> %s\n",
               r, proposed);
    }

    if (d.num_cells > 0) {
        uint32_t threshold_min = 255, threshold_max = 0;
        uint32_t amp2_min = 255, amp2_max = 0, amp2_sum = 0;
        for (uint32_t i = 0; i < d.num_cells; ++i) {
            SligSignal s;
            slig_unpack(&s, &d.cells[i]);
            if (s.cond_threshold < threshold_min) threshold_min = s.cond_threshold;
            if (s.cond_threshold > threshold_max) threshold_max = s.cond_threshold;
            if (s.audio_amps[2] < amp2_min) amp2_min = s.audio_amps[2];
            if (s.audio_amps[2] > amp2_max) amp2_max = s.audio_amps[2];
            amp2_sum += s.audio_amps[2];
        }
        printf("  all cells: cond_threshold range [%u..%u]\n",
               threshold_min, threshold_max);
        printf("             audio_amps[2] range [%u..%u]  mean=%.1f\n",
               amp2_min, amp2_max, (double)amp2_sum / d.num_cells);
    }

    printf("\n");
    slig_image_free(im);
}

static void wrap_solid(SligImage *im) { fill_solid(im, 0.5f); }

int main(void) {
    printf("=== slig_diag: SVD direction / cond_threshold / energy_ratio ===\n\n");
    run_one("solid 0.5",          wrap_solid);
    run_one("horizontal stripes", fill_horizontal_stripes);
    run_one("vertical stripes",   fill_vertical_stripes);
    run_one("diagonal stripes",   fill_diagonal_stripes);
    run_one("radial disc",        fill_radial);
    return 0;
}
