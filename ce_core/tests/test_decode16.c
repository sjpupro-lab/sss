/* test_decode16.c — round-trip ce_feed_image_16 -> ce_decode_image_block_16.
 *
 * Each 16x16 block has 4 quadrants of distinct uniform color. Encode via
 * ce_feed_image_16 to 4 CEUnits, then decode back to a 16x16 RGBA buffer
 * and verify each quadrant's color matches the input within 2 LSB
 * (matching the existing 8x8 round-trip tolerance).
 */
#include "../ce_core.h"
#include "../ce_feed_image.h"
#include "../ce_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static void fill_quadrants(uint8_t *blk16,
                           uint8_t r_tl, uint8_t r_tr,
                           uint8_t r_bl, uint8_t r_br) {
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

static int abs_int(int v) { return v < 0 ? -v : v; }

static int max_quadrant_err(const uint8_t *out16,
                            int qx, int qy,
                            uint8_t expected_r) {
    int max_err = 0;
    for (int dy = 0; dy < 8; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            const uint8_t *p = out16
                + (size_t)((qy + dy) * 16 + qx + dx) * 4u;
            int err = abs_int((int)p[0] - (int)expected_r);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

int main(void) {
    printf("=== test_decode16: ce_feed_image_16 -> ce_decode_image_block_16 round-trip ===\n");

    uint8_t blk[CE_IMAGE_BLOCK16_BYTES];
    fill_quadrants(blk, 30, 90, 150, 220);

    CEUnit q[4];
    ce_feed_image_16(q, blk);

    uint8_t out16[CE_IMAGE_BLOCK16_BYTES];
    ce_decode_image_block_16(out16, q);

    /* TL=0, TR=1, BL=2, BR=3 with quadrant origin (qx, qy). */
    int err_tl = max_quadrant_err(out16, 0, 0,  30);
    int err_tr = max_quadrant_err(out16, 8, 0,  90);
    int err_bl = max_quadrant_err(out16, 0, 8,  150);
    int err_br = max_quadrant_err(out16, 8, 8,  220);

    printf("  per-quadrant max R error: TL=%d TR=%d BL=%d BR=%d\n",
           err_tl, err_tr, err_bl, err_br);

    CHECK(err_tl <= 2, "TL quadrant round-trips R within 2 LSB");
    CHECK(err_tr <= 2, "TR quadrant round-trips R within 2 LSB");
    CHECK(err_bl <= 2, "BL quadrant round-trips R within 2 LSB");
    CHECK(err_br <= 2, "BR quadrant round-trips R within 2 LSB");

    /* Alpha must be set (nonzero default from decode). */
    int all_alpha_ok = 1;
    for (int i = 0; i < 16 * 16; ++i) {
        if (out16[i * 4 + 3] == 0) { all_alpha_ok = 0; break; }
    }
    CHECK(all_alpha_ok, "decoded alpha non-zero everywhere");

    /* Determinism: encode/decode twice and compare bytes. */
    CEUnit q2[4];
    ce_feed_image_16(q2, blk);
    uint8_t out16b[CE_IMAGE_BLOCK16_BYTES];
    ce_decode_image_block_16(out16b, q2);
    CHECK(memcmp(out16, out16b, CE_IMAGE_BLOCK16_BYTES) == 0,
          "two encode+decode passes are byte-identical");

    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
