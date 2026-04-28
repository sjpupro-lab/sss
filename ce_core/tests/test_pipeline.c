/* test_pipeline.c — end-to-end pipeline integration tests.
 *
 *  1. ce_feed_image_16: 16x16 -> 4 quadrant CEUnits, deterministic, distinct.
 *  2. ce_gen_config_hq: 50-step preset is non-default and within bounds.
 *  3. ce_generate_image_typed: round-trips a deterministic synthetic
 *     CEStorage entry through search_by_type + denoise + decode +
 *     wave_refine, and the result depends on `wave_refine_iters > 0`.
 */
#include "../ce_core.h"
#include "../ce_decode.h"
#include "../ce_feed_image.h"
#include "../ce_storage.h"
#include "../ce_search.h"
#include "../ce_denoise.h"
#include "../ce_gen.h"
#include "../ce_type.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static void fill_block_16(uint8_t *blk16,
                          uint8_t r_tl, uint8_t r_tr,
                          uint8_t r_bl, uint8_t r_br) {
    /* Per-quadrant constant red, full alpha; G/B = 0. Lets us verify
     * each output CEUnit isolates the right quadrant. */
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            uint8_t r = (y < 8)
                ? ((x < 8) ? r_tl : r_tr)
                : ((x < 8) ? r_bl : r_br);
            uint8_t *p = blk16 + (size_t)(y * 16 + x) * 4u;
            p[0] = r; p[1] = 0; p[2] = 0; p[3] = 255;
        }
    }
}

static int unit_eq(const CEUnit *a, const CEUnit *b) {
    return memcmp(a, b, sizeof(CEUnit)) == 0;
}

static void test_feed_image_16(void) {
    printf("--- 1. ce_feed_image_16 ---\n");

    uint8_t blk[CE_IMAGE_BLOCK16_BYTES];
    fill_block_16(blk, 10, 90, 170, 250); /* TL=10, TR=90, BL=170, BR=250 */

    CEUnit q1[4]; ce_feed_image_16(q1, blk);
    CEUnit q2[4]; ce_feed_image_16(q2, blk);

    int all_eq = 1;
    for (int i = 0; i < 4; ++i)
        if (!unit_eq(&q1[i], &q2[i])) all_eq = 0;
    CHECK(all_eq, "ce_feed_image_16 deterministic on identical input");

    /* Quadrants must differ since each holds a distinct color. */
    int distinct = !unit_eq(&q1[0], &q1[1])
                 && !unit_eq(&q1[1], &q1[2])
                 && !unit_eq(&q1[2], &q1[3]);
    CHECK(distinct, "four quadrants produce four distinct CEUnits");

    /* Each quadrant must round-trip to its constant red value via
     * ce_decode_image_block (allowing 1 LSB rounding tolerance). */
    uint8_t out[64 * 4];
    int max_err = 0;
    uint8_t expected[4] = { 10, 90, 170, 250 };
    for (int q = 0; q < 4; ++q) {
        ce_decode_image_block(out, &q1[q]);
        int e = (int)out[0] - (int)expected[q];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
    }
    CHECK(max_err <= 2, "16x16 quadrant decode round-trips R within 2 LSB");

    /* NULL input must return four zero-init cells, not crash. */
    CEUnit qz[4];
    memset(qz, 0xAA, sizeof(qz)); /* poison */
    ce_feed_image_16(qz, NULL);
    CEUnit zero; ce_init(&zero);
    int all_zero = 1;
    for (int i = 0; i < 4; ++i)
        if (!unit_eq(&qz[i], &zero)) all_zero = 0;
    CHECK(all_zero, "NULL input -> four zeroed CEUnits");
}

static void test_gen_config_hq(void) {
    printf("\n--- 2. ce_gen_config_hq ---\n");

    CEGenConfig def = ce_gen_config_default();
    CEGenConfig hq  = ce_gen_config_hq();

    CHECK(hq.total_steps == 50, "HQ preset uses 50 denoise steps");
    CHECK(hq.total_steps > def.total_steps, "HQ has more steps than default");
    CHECK(hq.cfg_scale > def.cfg_scale, "HQ cfg_scale > default");
    CHECK(hq.attn_topk >= def.attn_topk, "HQ attn_topk >= default");
    CHECK(hq.conv_radius >= def.conv_radius, "HQ conv_radius >= default");
    CHECK(hq.residual_weight >= 0.0f && hq.residual_weight <= 1.0f,
          "HQ residual_weight within [0,1]");
    CHECK(hq.skip_weight >= 0.0f && hq.skip_weight <= 1.0f,
          "HQ skip_weight within [0,1]");
}

static void seed_storage_with_red(CEStorage *s, uint32_t canvas, int blocks) {
    /* Insert image-tagged entries; first entry = pure red, others = drift. */
    for (int i = 0; i < blocks; ++i) {
        uint8_t blk[64 * 4];
        for (int p = 0; p < 64; ++p) {
            blk[p * 4 + 0] = (uint8_t)(120 + i * 5);
            blk[p * 4 + 1] = (uint8_t)(20 + i);
            blk[p * 4 + 2] = (uint8_t)(20 + i);
            blk[p * 4 + 3] = 255;
        }
        CEUnit kf;     ce_feed_image(&kf, blk);
        CEUnit zero;   ce_init(&zero);
        CEUnit delta;  ce_delta(&delta, &zero, &kf);
        ce_storage_add_typed(s, canvas, 0, (uint16_t)i,
                             CE_TYPE_IMAGE, &kf, &delta);
    }
    /* Add a TEXT-tagged decoy that wave_refine must NOT pull in. */
    CEUnit txt; ce_init(&txt);
    ce_feed(&txt, (const uint8_t *)"distractor", 10);
    CEUnit txd; ce_init(&txd);
    ce_storage_add_typed(s, canvas, 0, (uint16_t)blocks,
                         CE_TYPE_TEXT, &txt, &txd);
}

static double mean_red(const CEImage *img) {
    double sum = 0.0;
    int n = CE_IMAGE_W * CE_IMAGE_H;
    for (int i = 0; i < n; ++i) sum += img->pixels[i].r;
    return sum / (double)n;
}

static int images_differ(const CEImage *a, const CEImage *b) {
    int n = CE_IMAGE_W * CE_IMAGE_H;
    for (int i = 0; i < n; ++i) {
        if (a->pixels[i].r != b->pixels[i].r) return 1;
        if (a->pixels[i].g != b->pixels[i].g) return 1;
        if (a->pixels[i].b != b->pixels[i].b) return 1;
    }
    return 0;
}

static void test_generate_image_typed(void) {
    printf("\n--- 3. ce_generate_image_typed (E2E) ---\n");

    CEStorage S;
    ce_storage_init(&S, 16);
    seed_storage_with_red(&S, 7777, 4);

    /* Use a low-step config so the test is fast. wave_refine = 0. */
    CEGenConfig fast = ce_gen_config_default();
    fast.total_steps = 4;

    CEImage img_a, img_b;
    memset(&img_a, 0, sizeof(img_a));
    memset(&img_b, 0, sizeof(img_b));

    ce_generate_image_typed(&img_a, &S, CE_TYPE_IMAGE,
                            "red apple", 0xC0DE, &fast, 0);
    ce_generate_image_typed(&img_b, &S, CE_TYPE_IMAGE,
                            "red apple", 0xC0DE, &fast, 0);

    CHECK(!images_differ(&img_a, &img_b),
          "ce_generate_image_typed deterministic on identical input");
    CHECK(img_a.width == CE_IMAGE_W && img_a.height == CE_IMAGE_H,
          "output image dimensions match CE_IMAGE_W/H");

    double mr = mean_red(&img_a);
    CHECK(mr > 30.0,
          "image biased toward stored red entries (mean R > 30)");

    /* wave_refine > 0 must shift the canvas. */
    CEImage img_wave;
    memset(&img_wave, 0, sizeof(img_wave));
    ce_generate_image_typed(&img_wave, &S, CE_TYPE_IMAGE,
                            "red apple", 0xC0DE, &fast, 50);
    CHECK(images_differ(&img_a, &img_wave),
          "wave_refine_iters > 0 changes the output canvas");

    /* AUDIO-typed retrieval on an IMAGE+TEXT storage must still produce
     * a valid (zero-init-anchor) image without crashing. */
    CEImage img_audio;
    memset(&img_audio, 0, sizeof(img_audio));
    ce_generate_image_typed(&img_audio, &S, CE_TYPE_AUDIO,
                            "red apple", 0xC0DE, &fast, 0);
    CHECK(img_audio.width == CE_IMAGE_W,
          "AUDIO retrieval on non-AUDIO storage degrades gracefully");

    ce_storage_free(&S);
}

int main(void) {
    printf("=== ce_pipeline integration tests ===\n");
    test_feed_image_16();
    test_gen_config_hq();
    test_generate_image_typed();
    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
