/* test_decode_image_block.c — unit tests for ce_decode_image_block.
 *
 * The function is the per-block decode side of the image pipeline.
 * It reads inc.plus carry chains (one per RGBA channel), rescales them
 * into [0, 255] against the per-block maximum (64 * 255 = 16320), and
 * fills an 8x8 RGBA block with the per-channel value.
 */
#include "../ce_core.h"
#include "../ce_decode.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

/* Write a 32-bit value into a 4-byte little-endian carry chain. */
static void set_chain(uint8_t chain[4], uint32_t v) {
    chain[0] = (uint8_t)(v & 0xFF);
    chain[1] = (uint8_t)((v >> 8) & 0xFF);
    chain[2] = (uint8_t)((v >> 16) & 0xFF);
    chain[3] = (uint8_t)((v >> 24) & 0xFF);
}

int main(void) {
    printf("=== ce_decode_image_block tests ===\n");

    uint8_t block[64 * 4];

    /* 1. Zero unit -> RGB all zero, A defaulted to 255 (avoid invisible). */
    CEUnit u0;
    ce_init(&u0);
    memset(block, 0xAB, sizeof(block));
    ce_decode_image_block(block, &u0);
    int rgb_zero = 1;
    for (int i = 0; i < 64; ++i) {
        if (block[i*4 + 0] != 0 || block[i*4 + 1] != 0 || block[i*4 + 2] != 0) {
            rgb_zero = 0; break;
        }
    }
    CHECK(rgb_zero, "zero unit -> RGB == 0 across the block");
    int alpha_default = 1;
    for (int i = 0; i < 64; ++i) if (block[i*4 + 3] != 255) { alpha_default = 0; break; }
    CHECK(alpha_default, "zero unit -> A defaulted to 255");

    /* 2. Channel separation: R chain at max, others 0 -> red-only block. */
    CEUnit ur;
    ce_init(&ur);
    set_chain(ur.inc.R.plus, 16320u);
    ce_decode_image_block(block, &ur);
    int red_ok = 1;
    for (int i = 0; i < 64; ++i) {
        if (block[i*4 + 0] != 255 || block[i*4 + 1] != 0 || block[i*4 + 2] != 0) {
            red_ok = 0; break;
        }
    }
    CHECK(red_ok, "R=max yields R==255, G==0, B==0");

    /* 3. Channel separation: B chain at max, others 0. */
    CEUnit ub;
    ce_init(&ub);
    set_chain(ub.inc.B.plus, 16320u);
    ce_decode_image_block(block, &ub);
    int blue_ok = block[0] == 0 && block[1] == 0 && block[2] == 255;
    CHECK(blue_ok, "B=max yields R==0, G==0, B==255");

    /* 4. No cross-channel mixing: setting one channel must not bleed. */
    CEUnit ug;
    ce_init(&ug);
    set_chain(ug.inc.G.plus, 8160u); /* 50% of max */
    ce_decode_image_block(block, &ug);
    CHECK(block[0] == 0,  "G-only: R untouched (no cross-channel mix)");
    CHECK(block[1] == 127 || block[1] == 128, "G-only: G ~= 127/128 at 50%");
    CHECK(block[2] == 0,  "G-only: B untouched (no cross-channel mix)");

    /* 5. Half independence: dec.plus must NOT affect output (image-only
     * normalize reads inc.plus). This guards against accidental ce_read. */
    CEUnit ud;
    ce_init(&ud);
    set_chain(ud.dec.R.plus, 16320u);
    set_chain(ud.dec.R.minus, 16320u);
    ce_decode_image_block(block, &ud);
    CHECK(block[0] == 0, "dec half does not contribute to decoded R");

    /* 6. Saturation: totals above max clamp to 255 instead of wrapping. */
    CEUnit usat;
    ce_init(&usat);
    set_chain(usat.inc.R.plus, 0xFFFFFFFFu);
    ce_decode_image_block(block, &usat);
    CHECK(block[0] == 255, "carry total above max saturates at 255");

    /* 7. Determinism: same input -> same output. */
    CEUnit ud1, ud2;
    ce_init(&ud1); ce_init(&ud2);
    const char *seed = "deterministic image block";
    ce_feed(&ud1, (const uint8_t *)seed, (uint32_t)strlen(seed));
    ce_feed(&ud2, (const uint8_t *)seed, (uint32_t)strlen(seed));
    uint8_t b1[256], b2[256];
    ce_decode_image_block(b1, &ud1);
    ce_decode_image_block(b2, &ud2);
    CHECK(memcmp(b1, b2, sizeof(b1)) == 0, "deterministic across identical units");

    /* 8. Block uniformity: a single CEUnit decodes to 64 identical pixels. */
    int uniform = 1;
    for (int i = 1; i < 64; ++i) {
        if (memcmp(&b1[0], &b1[i*4], 4) != 0) { uniform = 0; break; }
    }
    CHECK(uniform, "single CEUnit yields uniform 8x8 block");

    /* 9. Linearity: total at 25% / 50% / 75% of max maps to ~64 / 127 / 191. */
    CEUnit u25, u50, u75;
    ce_init(&u25); ce_init(&u50); ce_init(&u75);
    set_chain(u25.inc.R.plus, 4080u);   /* 25% */
    set_chain(u50.inc.R.plus, 8160u);   /* 50% */
    set_chain(u75.inc.R.plus, 12240u);  /* 75% */
    ce_decode_image_block(block, &u25); int v25 = block[0];
    ce_decode_image_block(block, &u50); int v50 = block[0];
    ce_decode_image_block(block, &u75); int v75 = block[0];
    CHECK(v25 >= 60  && v25 <= 68,  "25% carry -> ~63");
    CHECK(v50 >= 124 && v50 <= 132, "50% carry -> ~127");
    CHECK(v75 >= 187 && v75 <= 195, "75% carry -> ~191");

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
