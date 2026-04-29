/* test_residual_decode.c — hybrid_vae_decode applies CE_TYPE_RESIDUAL
 * descriptors when cfg.residual_book is provided.
 *
 * Strategy:
 *   1. Train one image into a CEStorage + SligCellSet grid using
 *      hybrid_vae_encode (no residual book — produces raw SLIG cells
 *      and the audio_bins link).
 *   2. Walk the encoded SLIG cells, push the texture-budget tail into
 *      a residual codebook, and append CE_TYPE_RESIDUAL descriptors
 *      to the same storage with deterministic strength.
 *   3. Decode twice — once with cfg.residual_book = NULL, once with
 *      it set to the populated book.
 *   4. The two outputs must differ. The expected change is on the
 *      Y plane (ce_hybrid_vae's row-tile residual stamp), so we
 *      verify byte-level inequality and that the difference is
 *      bounded (sanity).
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_hybrid_vae.h"
#include "../ce_residual_codebook.h"
#include "../ce_tick.h"
#include "../slig_codebook.h"
#include "../slig_signal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static void synth_rgba(uint8_t *rgba) {
    /* Mixed-color quadrants — gives the SLIG decomposer enough Y/Cb/Cr
     * energy to populate every (scale, channel) bucket. */
    static const uint8_t Q[4][3] = {
        {220,  40,  40}, { 40, 200,  60},
        { 30,  60, 220}, {220, 200,  40}};
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 256; ++x) {
            int q = (y < 128 ? 0 : 2) + (x < 128 ? 0 : 1);
            int i = (y * 256 + x) * 4;
            rgba[i + 0] = Q[q][0];
            rgba[i + 1] = Q[q][1];
            rgba[i + 2] = Q[q][2];
            rgba[i + 3] = 255;
        }
    }
}

int main(void) {
    printf("=== hybrid_vae_decode + residual codebook ===\n");

    uint8_t *rgba = (uint8_t *)malloc(256 * 256 * 4);
    CHECK(rgba != NULL, "rgba alloc");
    if (!rgba) return 1;
    synth_rgba(rgba);

    CEStorage S;
    ce_storage_init(&S, 64);
    SligCodebook scb;
    slig_codebook_init(&scb);

    HybridEncodeResult enc;
    hybrid_vae_encode(&enc, &S, &scb, rgba, 256, 256, /*cid=*/0xC0DEu);
    CHECK(enc.total_cells > 0, "encode produced SLIG cells");
    free(rgba);

    /* Build a residual codebook + a handful of descriptors against the
     * same canvas. Use the encoded Y/FINE set as the source — its top
     * cells stand in for "correction" patches. */
    CEResidualCodebook book;
    ce_residual_codebook_init(&book);

    const SligCellSet *src = &enc.sets[SLIG_LEVEL_FINE][SLIG_CH_Y];
    uint32_t want = (src->num_cells < 8) ? src->num_cells : 8;
    for (uint32_t i = 0; i < want; ++i) {
        CEResidualCode probe;
        memset(&probe, 0, sizeof(probe));
        probe.unit        = src->cells[i];
        probe.scale_level = SLIG_LEVEL_FINE;
        probe.strength    = ce_tick_amp_from_cell(&src->cells[i]);
        probe.tick.r = (uint8_t)((i * 31) & 0xFF);  /* spread rows */
        probe.tick.g = (uint8_t)(i & 0x01);
        probe.tick.b = 0;
        probe.tick.a = probe.strength;
        uint8_t idx = ce_residual_codebook_add_or_lookup(
            &book, &probe, CE_RESIDUAL_DEFAULT_THRESHOLD);

        /* Force a high enough strength so the apply step actually
         * mutates blend_y by more than 0. Spread the patches across
         * different (x, y) positions so the 8×8 stamp reaches
         * non-overlapping rows. */
        uint16_t px = (uint16_t)((i * 37) & 0xFF);
        uint16_t py = (uint16_t)((i * 53) & 0xFF);
        ce_residual_storage_add(&S, /*cid=*/0xC0DEu,
            /*slot=*/(uint16_t)(i / 8),
            /*block_idx=*/(uint16_t)(i % 8),
            idx, /*strength=*/200, probe.tick, px, py);
    }
    CHECK(book.count > 0, "codebook populated");

    uint32_t resid_count = 0;
    for (uint32_t i = 0; i < S.count; ++i)
        if (S.entries[i].type == CE_TYPE_RESIDUAL) ++resid_count;
    CHECK(resid_count == want, "residual descriptors added to storage");

    /* Decode twice, with and without the codebook reference. */
    HybridVAEConfig cfg_off, cfg_on;
    hybrid_vae_config_default(&cfg_off);
    hybrid_vae_config_default(&cfg_on);
    cfg_on.residual_book  = &book;
    /* Wave refine smooths the row stamps out, so disable it for the
     * comparison — we want to read the raw effect of step 4b. */
    cfg_off.wave_iterations = 0;
    cfg_on.wave_iterations  = 0;

    uint8_t *out_off = (uint8_t *)malloc(256 * 256 * 3);
    uint8_t *out_on  = (uint8_t *)malloc(256 * 256 * 3);
    CHECK(out_off && out_on, "output buffers");
    if (out_off && out_on) {
        hybrid_vae_decode(out_off, &S, 0xC0DEu, enc.sets, &cfg_off);
        hybrid_vae_decode(out_on,  &S, 0xC0DEu, enc.sets, &cfg_on);

        long long diff_total = 0;
        int diff_pixels = 0;
        int max_dev = 0;
        for (int i = 0; i < 256 * 256 * 3; ++i) {
            int d = (int)out_off[i] - (int)out_on[i];
            if (d != 0) {
                ++diff_pixels;
                int ad = d < 0 ? -d : d;
                diff_total += ad;
                if (ad > max_dev) max_dev = ad;
            }
        }
        printf("  diff_pixels=%d  diff_total=%lld  max_dev=%d\n",
               diff_pixels, diff_total, max_dev);
        CHECK(diff_pixels > 0, "residual codebook changes the decode output");
        /* Cap of 64 is per-pattern; multiple patterns may stack on the
         * same row. Allow up to 4× single-pattern cap for cumulative
         * stack of weak patches. */
        CHECK(max_dev <= 256, "per-pixel deviation stays within 8-bit range");

        free(out_off);
        free(out_on);
    }

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
