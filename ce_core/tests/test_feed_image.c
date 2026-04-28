/* test_feed_image.c — unit tests for ce_feed_image + ce_search_by_type.
 *
 * The encoder must:
 *   - keep RGBA channels separate in the inc half
 *   - leave inc.<ch>.minus untouched (zero) so ce_decode_image_block
 *     remains correct
 *   - capture LR / UD / D1 / D2 gradients in dec.<ch>.{plus,minus}
 *   - be deterministic
 *
 * The search must filter by modality so image queries never retrieve
 * text keyframes.
 */
#include "../ce_core.h"
#include "../ce_decode.h"
#include "../ce_feed_image.h"
#include "../ce_storage.h"
#include "../ce_search.h"
#include "../ce_type.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

/* Fill an 8x8 RGBA block with a constant color. */
static void fill_const(uint8_t *blk, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    for (int i = 0; i < 64; ++i) {
        blk[i*4 + 0] = r; blk[i*4 + 1] = g;
        blk[i*4 + 2] = b; blk[i*4 + 3] = a;
    }
}

/* Fill with a left -> right ramp on the R channel. Other channels constant. */
static void fill_lr_ramp(uint8_t *blk) {
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t *p = blk + (y*8 + x) * 4;
            p[0] = (uint8_t)(x * 32); /* 0, 32, 64, ..., 224 */
            p[1] = 0;
            p[2] = 0;
            p[3] = 255;
        }
    }
}

/* Fill with a top -> bottom ramp on the G channel. */
static void fill_ud_ramp(uint8_t *blk) {
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t *p = blk + (y*8 + x) * 4;
            p[0] = 0;
            p[1] = (uint8_t)(y * 32);
            p[2] = 0;
            p[3] = 255;
        }
    }
}

/* Fill with a TL->BR diagonal ramp on the B channel. */
static void fill_d1_ramp(uint8_t *blk) {
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t *p = blk + (y*8 + x) * 4;
            p[0] = 0;
            p[1] = 0;
            p[2] = (uint8_t)((x + y) * 16);
            p[3] = 255;
        }
    }
}

/* Fill with an anti-diagonal ramp (TR brighter than BL) on the B channel. */
static void fill_d2_ramp(uint8_t *blk) {
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t *p = blk + (y*8 + x) * 4;
            p[0] = 0;
            p[1] = 0;
            p[2] = (uint8_t)((x + (7 - y)) * 16);
            p[3] = 255;
        }
    }
}

int main(void) {
    printf("=== ce_feed_image / ce_search_by_type tests ===\n");

    uint8_t blk[256];
    CEUnit u;

    /* ---------- 1. Constant red block ---------- */
    fill_const(blk, 255, 0, 0, 255);
    ce_feed_image(&u, blk);
    /* inc.R.plus chain == 64*255 = 16320. */
    uint32_t inc_R = (uint32_t)u.inc.R.plus[0]
                   | ((uint32_t)u.inc.R.plus[1] << 8)
                   | ((uint32_t)u.inc.R.plus[2] << 16)
                   | ((uint32_t)u.inc.R.plus[3] << 24);
    CHECK(inc_R == 64u * 255u, "constant red: inc.R.plus == 64*255");
    /* G/B inc chains untouched. */
    int gb_zero = 1;
    for (int i = 0; i < 4; ++i) {
        if (u.inc.G.plus[i] != 0 || u.inc.B.plus[i] != 0) { gb_zero = 0; break; }
    }
    CHECK(gb_zero, "constant red: inc.G/B chains zero (no channel mix)");
    /* inc.<ch>.minus must remain zero (decode reads only inc.plus). */
    int inc_minus_zero = 1;
    if (memcmp(u.inc.R.minus, "\0\0\0\0", 4) != 0) inc_minus_zero = 0;
    if (memcmp(u.inc.G.minus, "\0\0\0\0", 4) != 0) inc_minus_zero = 0;
    if (memcmp(u.inc.B.minus, "\0\0\0\0", 4) != 0) inc_minus_zero = 0;
    if (memcmp(u.inc.A.minus, "\0\0\0\0", 4) != 0) inc_minus_zero = 0;
    CHECK(inc_minus_zero, "inc.<ch>.minus stays zero (decode invariance)");
    /* No gradients on a flat block. */
    int flat = 1;
    for (int c = 0; c < 4; ++c) {
        const CETier *t = (c == 0) ? &u.dec.R : (c == 1) ? &u.dec.G
                          : (c == 2) ? &u.dec.B : &u.dec.A;
        for (int i = 0; i < 4; ++i) if (t->plus[i] != 0) { flat = 0; }
    }
    CHECK(flat, "constant block: all 4-direction gradients == 0");

    /* ---------- 2. Roundtrip with merged ce_decode_image_block ---------- */
    uint8_t out[256];
    ce_decode_image_block(out, &u);
    int decoded_red = (out[0] == 255 && out[1] == 0 && out[2] == 0 && out[3] == 255);
    CHECK(decoded_red, "decode after ce_feed_image (constant red) yields R==255");

    /* Constant 4-corner-mid block: (128,128,128,255). */
    fill_const(blk, 128, 128, 128, 255);
    ce_feed_image(&u, blk);
    ce_decode_image_block(out, &u);
    /* 64 * 128 / 64 = 128 (rounded by 16320 rescale; allow +/- 1). */
    CHECK(out[0] >= 127 && out[0] <= 129, "decode constant 128 -> ~128 R");
    CHECK(out[1] >= 127 && out[1] <= 129, "decode constant 128 -> ~128 G");
    CHECK(out[2] >= 127 && out[2] <= 129, "decode constant 128 -> ~128 B");

    /* ---------- 3. LR ramp on R ---------- */
    fill_lr_ramp(blk);
    ce_feed_image(&u, blk);
    /* LR magnitude on R must be the dominant gradient. */
    CHECK(u.dec.R.plus[0] > 0,  "LR ramp on R: |LR| > 0");
    CHECK(u.dec.R.plus[0] >= u.dec.R.plus[1], "LR ramp on R: |LR| >= |UD|");
    CHECK(u.dec.R.plus[1] == 0, "LR ramp on R: |UD| == 0");
    CHECK(u.dec.R.minus[0] == 0xFF, "LR ramp on R: LR sign positive (right brighter)");
    /* Other channels should have all-zero gradients. */
    int other_grad_zero = 1;
    for (int i = 0; i < 4; ++i) {
        if (u.dec.G.plus[i] != 0) other_grad_zero = 0;
        if (u.dec.B.plus[i] != 0) other_grad_zero = 0;
    }
    CHECK(other_grad_zero, "LR ramp on R: G/B gradients all zero");

    /* ---------- 4. UD ramp on G ---------- */
    fill_ud_ramp(blk);
    ce_feed_image(&u, blk);
    CHECK(u.dec.G.plus[1] > 0,  "UD ramp on G: |UD| > 0");
    CHECK(u.dec.G.plus[1] >= u.dec.G.plus[0], "UD ramp on G: |UD| >= |LR|");
    CHECK(u.dec.G.plus[0] == 0, "UD ramp on G: |LR| == 0");
    CHECK(u.dec.G.minus[1] == 0xFF, "UD ramp on G: UD sign positive (bottom brighter)");

    /* ---------- 5. Main diagonal ramp on B (TL dim, BR bright) ---------- */
    fill_d1_ramp(blk);
    ce_feed_image(&u, blk);
    CHECK(u.dec.B.plus[2] > 0, "D1 ramp on B: |D1| > 0");
    CHECK(u.dec.B.minus[2] == 0xFF, "D1 ramp on B: D1 sign positive (BR brighter)");

    /* ---------- 6. Anti-diagonal ramp on B (TR bright, BL dim) ---------- */
    fill_d2_ramp(blk);
    ce_feed_image(&u, blk);
    CHECK(u.dec.B.plus[3] > 0, "D2 ramp on B: |D2| > 0");
    /* (BL - TR): TR brighter -> BL - TR is negative. */
    CHECK(u.dec.B.minus[3] == 0x00, "D2 ramp on B: D2 sign negative (TR brighter)");

    /* ---------- 7. Determinism ---------- */
    CEUnit u1, u2;
    fill_lr_ramp(blk);
    ce_feed_image(&u1, blk);
    ce_feed_image(&u2, blk);
    CHECK(memcmp(&u1, &u2, sizeof(CEUnit)) == 0, "ce_feed_image deterministic");

    /* Different input -> different output. */
    fill_const(blk, 10, 20, 30, 255);
    ce_feed_image(&u2, blk);
    CHECK(memcmp(&u1, &u2, sizeof(CEUnit)) != 0, "different inputs -> different units");

    /* ---------- 8. ce_search_by_type ---------- */
    CEStorage S; ce_storage_init(&S, 8);

    /* Add 3 text entries. */
    CEUnit txt; ce_init(&txt);
    const char *t = "alpha"; ce_feed(&txt, (const uint8_t *)t, 5);
    ce_storage_add_typed(&S, 1, 0, 0, CE_TYPE_TEXT, &txt, NULL);
    t = "bravo";   ce_init(&txt); ce_feed(&txt, (const uint8_t *)t, 5);
    ce_storage_add_typed(&S, 1, 0, 1, CE_TYPE_TEXT, &txt, NULL);
    t = "charlie"; ce_init(&txt); ce_feed(&txt, (const uint8_t *)t, 7);
    ce_storage_add_typed(&S, 1, 0, 2, CE_TYPE_TEXT, &txt, NULL);

    /* Add 2 image entries. */
    CEUnit img;
    fill_const(blk, 200, 50, 50, 255); ce_feed_image(&img, blk);
    ce_storage_add_typed(&S, 2, 0, 0, CE_TYPE_IMAGE, &img, NULL);
    fill_const(blk, 50, 200, 50, 255); ce_feed_image(&img, blk);
    ce_storage_add_typed(&S, 2, 0, 1, CE_TYPE_IMAGE, &img, NULL);

    /* Image query -> only image entries should be returned. */
    CEUnit query;
    fill_const(blk, 220, 40, 40, 255); ce_feed_image(&query, blk);

    CESearchResult res[4];
    int nimg = ce_search_by_type(&S, &query, CE_TYPE_IMAGE, 4, res);
    CHECK(nimg == 2, "ce_search_by_type(IMAGE) returns only image entries");
    int both_image = (S.entries[res[0].entry_idx].type == CE_TYPE_IMAGE)
                  && (S.entries[res[1].entry_idx].type == CE_TYPE_IMAGE);
    CHECK(both_image, "all returned entries are tagged IMAGE");

    int ntxt = ce_search_by_type(&S, &query, CE_TYPE_TEXT, 4, res);
    CHECK(ntxt == 3, "ce_search_by_type(TEXT) returns only text entries");
    int all_text = 1;
    for (int i = 0; i < ntxt; ++i) {
        if (S.entries[res[i].entry_idx].type != CE_TYPE_TEXT) { all_text = 0; break; }
    }
    CHECK(all_text, "all returned entries are tagged TEXT");

    int naud = ce_search_by_type(&S, &query, CE_TYPE_AUDIO, 4, res);
    CHECK(naud == 0, "ce_search_by_type(AUDIO) on a TEXT+IMAGE storage returns 0");

    /* Unfiltered topk still sees everything. */
    int nall = ce_search_topk(&S, &query, 5, res);
    CHECK(nall == 5, "ce_search_topk (unfiltered) returns all 5 entries");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
