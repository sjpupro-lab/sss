/* test_slig_energy.c — regression for the 3 slig_decompose_v2 changes:
 *
 *   1. infer_dir_from_uv: SVD direction is no longer hard-coded to
 *      HORIZONTAL — vertical input produces VERTICAL, diagonal produces
 *      DIAG_DOWN/UP.
 *
 *   2. cond_threshold: now stores cumulative SVD energy coverage
 *      (0..255) instead of leaving the field zero.
 *
 *   3. audio_amps[2]: now stores per-component energy ratio
 *      (sigma^2 / total_energy) instead of zero.
 *
 *   4. apply_global respects audio_amps[2] as an amplitude scale
 *      (and stays backward-compatible when the field is unmeasured/0).
 *
 * The first three are checked by re-running slig_decompose_v2 and
 * inspecting the unpacked SligSignal. The fourth runs apply_global
 * directly on a tiny crafted SligSignal: amp[2]=0 must keep the
 * legacy magnitude, amp[2]=128 must produce ~half the legacy
 * magnitude, amp[2]=255 must match the legacy magnitude.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../slig_pipeline.h"
#include "../slig_signal.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static void fill_horizontal(SligImage *im) {
    for (int y = 0; y < im->height; ++y) {
        float v = ((y / 8) & 1) ? 0.9f : 0.1f;
        for (int x = 0; x < im->width; ++x) im->data[y * im->width + x] = v;
    }
}
static void fill_vertical(SligImage *im) {
    for (int y = 0; y < im->height; ++y)
        for (int x = 0; x < im->width; ++x)
            im->data[y * im->width + x] = ((x / 8) & 1) ? 0.9f : 0.1f;
}
static void fill_diagonal(SligImage *im) {
    for (int y = 0; y < im->height; ++y)
        for (int x = 0; x < im->width; ++x)
            im->data[y * im->width + x] = (((x + y) / 8) & 1) ? 0.9f : 0.1f;
}

static uint8_t dominant_dir(const SligImage *im) {
    SligImage *s = slig_image_alloc(im->width, im->height, 1);
    memcpy(s->data, im->data, sizeof(float) * im->width * im->height);

    SligDecomposeConfig cfg;
    slig_decompose_config_default(&cfg);

    SligDecomposed d;
    memset(&d, 0, sizeof(d));
    slig_decompose_v2(&d, s, &cfg);

    SligSignal sig;
    slig_unpack(&sig, &d.cells[0]);
    slig_image_free(s);
    return sig.dir;
}

static void inspect_dominant(const SligImage *im, SligSignal *out) {
    SligImage *s = slig_image_alloc(im->width, im->height, 1);
    memcpy(s->data, im->data, sizeof(float) * im->width * im->height);

    SligDecomposeConfig cfg;
    slig_decompose_config_default(&cfg);

    SligDecomposed d;
    memset(&d, 0, sizeof(d));
    slig_decompose_v2(&d, s, &cfg);
    slig_unpack(out, &d.cells[0]);
    slig_image_free(s);
}

/* Crafted SligSignal that produces a non-trivial canvas via apply_global. */
static void build_crafted_sig(SligSignal *sig, uint8_t amp2) {
    memset(sig, 0, sizeof(*sig));
    sig->dir = SLIG_DIR_HORIZONTAL;
    sig->scale = SLIG_SCALE_COARSE;
    sig->sigma = 1000;
    sig->frequency = 4;
    sig->origin_x = 128;
    sig->origin_y = 128;
    sig->cond_type = 0;        /* no gating */
    sig->cond_threshold = 0;
    sig->audio_amps[2] = amp2;
    /* Simple bipolar wave — guarantees non-zero accumulation. */
    for (int i = 0; i < SLIG_SIG_LEN; ++i) {
        sig->u[i] = (int16_t)(sinf((float)i * 0.10f) * 1000.0f);
        sig->v[i] = (int16_t)(cosf((float)i * 0.10f) * 1000.0f);
    }
}

static int64_t canvas_abs_sum(const SligCanvas *c) {
    int64_t s = 0;
    for (int y = 0; y < SLIG_CANVAS_DIM; ++y)
        for (int x = 0; x < SLIG_CANVAS_DIM; ++x) {
            int32_t v = c->pixels[y][x];
            s += (v < 0) ? -v : v;
        }
    return s;
}

static void test_directions(void) {
    printf("--- 1. SVD direction inference ---\n");
    int dim = 64;
    SligImage *h = slig_image_alloc(dim, dim, 1); fill_horizontal(h);
    SligImage *v = slig_image_alloc(dim, dim, 1); fill_vertical(v);
    SligImage *dg = slig_image_alloc(dim, dim, 1); fill_diagonal(dg);

    uint8_t hd = dominant_dir(h);
    uint8_t vd = dominant_dir(v);
    uint8_t dd = dominant_dir(dg);

    CHECK(hd == SLIG_DIR_HORIZONTAL, "horizontal stripes -> SLIG_DIR_HORIZONTAL");
    CHECK(vd == SLIG_DIR_VERTICAL,   "vertical stripes -> SLIG_DIR_VERTICAL");
    CHECK(dd == SLIG_DIR_DIAG_DOWN || dd == SLIG_DIR_DIAG_UP,
          "diagonal stripes -> SLIG_DIR_DIAG_*");

    slig_image_free(h); slig_image_free(v); slig_image_free(dg);
}

static void test_cond_and_amp(void) {
    printf("\n--- 2. cond_threshold + audio_amps[2] populated ---\n");
    int dim = 64;
    SligImage *h = slig_image_alloc(dim, dim, 1); fill_horizontal(h);

    SligSignal sig;
    inspect_dominant(h, &sig);

    /* Pure horizontal stripes -> rank-1 captures (almost) all energy ->
     * cumulative coverage saturates at 255 and component ratio is also
     * near 255 for the dominant cell. */
    CHECK(sig.cond_threshold > 200,
          "rank-1 cell cond_threshold near full coverage (>200)");
    CHECK(sig.audio_amps[2] > 200,
          "rank-1 cell audio_amps[2] near full ratio (>200)");

    slig_image_free(h);

    /* Diagonal: rank-1 captures only ~40%, so cond_threshold is mid-range. */
    SligImage *dg = slig_image_alloc(dim, dim, 1); fill_diagonal(dg);
    inspect_dominant(dg, &sig);
    CHECK(sig.cond_threshold > 50 && sig.cond_threshold < 200,
          "diagonal rank-1 cond_threshold mid-range (50..200)");
    CHECK(sig.audio_amps[2] > 50 && sig.audio_amps[2] < 200,
          "diagonal rank-1 audio_amps[2] mid-range (50..200)");
    slig_image_free(dg);
}

static void test_apply_global_modulation(void) {
    printf("\n--- 3. apply_global amp scaling by audio_amps[2] ---\n");

    SligSignal s_full, s_half, s_zero;
    build_crafted_sig(&s_full, 255);
    build_crafted_sig(&s_half, 128);
    build_crafted_sig(&s_zero,   0);

    SligCanvas c_full, c_half, c_zero;
    memset(&c_full, 0, sizeof(c_full));
    memset(&c_half, 0, sizeof(c_half));
    memset(&c_zero, 0, sizeof(c_zero));

    slig_apply(&c_full, &s_full);
    slig_apply(&c_half, &s_half);
    slig_apply(&c_zero, &s_zero);

    int64_t mag_full = canvas_abs_sum(&c_full);
    int64_t mag_half = canvas_abs_sum(&c_half);
    int64_t mag_zero = canvas_abs_sum(&c_zero);

    printf("  abs-sum: amp[2]=0 -> %lld  amp[2]=128 -> %lld  amp[2]=255 -> %lld\n",
           (long long)mag_zero, (long long)mag_half, (long long)mag_full);

    /* Backward compat: amp[2]=0 must produce the SAME canvas as
     * amp[2]=255 (legacy unmodulated path). */
    CHECK(mag_zero == mag_full,
          "amp[2]=0 (legacy) matches amp[2]=255 (full) magnitude");

    /* amp[2]=128 -> ~half magnitude. Allow 5% tolerance for integer
     * rounding inside the per-pixel >>20 reduction. */
    double ratio = (mag_full > 0) ? (double)mag_half / (double)mag_full : 0.0;
    printf("  half/full magnitude ratio = %.3f (expected ~0.5)\n", ratio);
    CHECK(ratio > 0.45 && ratio < 0.55,
          "amp[2]=128 produces ~50% of full magnitude (within 5%)");
}

/* Step-1 pipeline-completeness check: edges, texture, and color cells
 * must also populate audio_amps[2] (max-normalized weight) and
 * cond_threshold (cumulative coverage), with the strongest cell in each
 * stage hitting amp[2] = 255. Without this, render-time amplitude is
 * uniform across stages and the energy story stops at the SVD basis. */
static void test_non_structure_stages_carry_energy(void) {
    printf("\n--- 4. edges/texture/color carry audio_amps[2] + cond_threshold ---\n");

    /* Use a 3-channel image with strong horizontal stripes + color
     * variation across quadrants, so all 5 decompose stages produce
     * cells. */
    int dim = 64;
    SligImage *im = slig_image_alloc(dim, dim, 3);
    for (int y = 0; y < dim; ++y) {
        float lum = ((y / 8) & 1) ? 0.85f : 0.10f;
        for (int x = 0; x < dim; ++x) {
            float r = lum, g = lum, b = lum;
            /* Quadrant tinting -> color stage gets non-zero deviations. */
            if (x >= dim/2 && y <  dim/2) r = fminf(1.0f, lum + 0.4f);
            if (x <  dim/2 && y >= dim/2) g = fminf(1.0f, lum + 0.4f);
            if (x >= dim/2 && y >= dim/2) b = fminf(1.0f, lum + 0.4f);
            im->data[(y * dim + x) * 3 + 0] = r;
            im->data[(y * dim + x) * 3 + 1] = g;
            im->data[(y * dim + x) * 3 + 2] = b;
        }
    }

    SligDecomposeConfig cfg;
    slig_decompose_config_default(&cfg);
    SligDecomposed d;
    memset(&d, 0, sizeof(d));
    slig_decompose_v2(&d, im, &cfg);

    int edge_max_amp2 = 0, edge_count = 0;
    int tex_max_amp2  = 0, tex_count  = 0;
    int col_max_amp2  = 0, col_count  = 0;

    for (uint32_t i = d.structure_end; i < d.edge_end; ++i, ++edge_count) {
        SligSignal s; slig_unpack(&s, &d.cells[i]);
        if (s.audio_amps[2] > edge_max_amp2) edge_max_amp2 = s.audio_amps[2];
    }
    for (uint32_t i = d.edge_end; i < d.texture_end; ++i, ++tex_count) {
        SligSignal s; slig_unpack(&s, &d.cells[i]);
        if (s.audio_amps[2] > tex_max_amp2) tex_max_amp2 = s.audio_amps[2];
    }
    for (uint32_t i = d.texture_end; i < d.color_end; ++i, ++col_count) {
        SligSignal s; slig_unpack(&s, &d.cells[i]);
        if (s.audio_amps[2] > col_max_amp2) col_max_amp2 = s.audio_amps[2];
    }

    printf("  edges:   %d cells, max amp[2] = %d\n", edge_count, edge_max_amp2);
    printf("  texture: %d cells, max amp[2] = %d\n", tex_count,  tex_max_amp2);
    printf("  color:   %d cells, max amp[2] = %d\n", col_count,  col_max_amp2);

    /* The strongest cell in each non-empty stage must hit amp[2] = 255
     * (max-normalization preserves the dominant cell at full sigma). */
    if (edge_count > 0)
        CHECK(edge_max_amp2 >= 250, "strongest edge cell amp[2] >= 250");
    if (tex_count > 0)
        CHECK(tex_max_amp2  >= 250, "strongest texture cell amp[2] >= 250");
    if (col_count > 0)
        CHECK(col_max_amp2  >= 250, "strongest color cell amp[2] >= 250");

    /* Cumulative coverage must end at 255 for the last cell of each
     * non-empty stage (cells are pushed in descending-strength order,
     * so the last one closes the budget). */
    if (d.edge_end > d.structure_end) {
        SligSignal last; slig_unpack(&last, &d.cells[d.edge_end - 1]);
        CHECK(last.cond_threshold >= 250,
              "edge stage closes with cond_threshold >= 250");
    }
    if (d.texture_end > d.edge_end) {
        SligSignal last; slig_unpack(&last, &d.cells[d.texture_end - 1]);
        CHECK(last.cond_threshold >= 250,
              "texture stage closes with cond_threshold >= 250");
    }
    if (d.color_end > d.texture_end) {
        SligSignal last; slig_unpack(&last, &d.cells[d.color_end - 1]);
        CHECK(last.cond_threshold >= 250,
              "color stage closes with cond_threshold >= 250");
    }

    slig_image_free(im);
}

int main(void) {
    printf("=== test_slig_energy: SVD dir + cond_threshold + amp[2] ===\n\n");
    test_directions();
    test_cond_and_amp();
    test_apply_global_modulation();
    test_non_structure_stages_carry_energy();
    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
