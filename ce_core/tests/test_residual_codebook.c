/* test_residual_codebook.c — CEResidualCodebook + storage descriptor.
 *
 * Verifies:
 *   1. init produces an empty book.
 *   2. add_or_lookup returns 0 for the first novel patch and bumps
 *      used_count on subsequent matches.
 *   3. distance respects scale_level bucketing (mismatch = UINT32_MAX).
 *   4. ce_residual_storage_add writes a descriptor entry with the
 *      expected (idx, strength, tick) bytes; unpack roundtrips them.
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_residual_codebook.h"
#include "../ce_tick.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static void make_patch(CEResidualCode *p, uint8_t scale, uint8_t b0) {
    memset(p, 0, sizeof(*p));
    p->scale_level = scale;
    p->direction   = 0;
    p->strength    = 200;
    p->tick        = (TickRGBA){0};
    ce_init(&p->unit);
    /* Feed `b0` 8 times so two patches with adjacent b0 land within
     * threshold but distinct b0 values are distinguishable. */
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = (uint8_t)(b0 + i);
    ce_feed(&p->unit, buf, 8);
}

int main(void) {
    printf("=== ce_residual_codebook tests ===\n");

    /* --- init --- */
    CEResidualCodebook book;
    ce_residual_codebook_init(&book);
    CHECK(book.count == 0, "init zero count");

    /* --- add a novel patch --- */
    CEResidualCode p1;
    make_patch(&p1, /*scale=*/1, /*b0=*/0x10);
    uint8_t i1 = ce_residual_codebook_add_or_lookup(
        &book, &p1, CE_RESIDUAL_DEFAULT_THRESHOLD);
    CHECK(i1 == 0, "first add returns idx 0");
    CHECK(book.count == 1, "count grew to 1");
    CHECK(book.codes[0].used_count == 1, "used_count = 1 on first add");

    /* --- nearly identical patch (within threshold) reuses --- */
    CEResidualCode p1b;
    make_patch(&p1b, /*scale=*/1, /*b0=*/0x11);
    uint8_t i1b = ce_residual_codebook_add_or_lookup(
        &book, &p1b, CE_RESIDUAL_DEFAULT_THRESHOLD);
    CHECK(i1b == 0, "near-identical patch reuses idx 0");
    CHECK(book.count == 1, "count stays at 1");
    CHECK(book.codes[0].used_count == 2, "used_count incremented");

    /* --- distinct scale level — must NOT collide --- */
    CEResidualCode p2;
    make_patch(&p2, /*scale=*/2, /*b0=*/0x10);
    uint8_t i2 = ce_residual_codebook_add_or_lookup(
        &book, &p2, CE_RESIDUAL_DEFAULT_THRESHOLD);
    CHECK(i2 == 1, "distinct scale gets new slot");
    CHECK(book.count == 2, "count grew to 2");

    /* --- distance: same scale → finite, different scale → MAX --- */
    uint32_t d_same = ce_residual_distance(&book.codes[0], &p1b);
    CHECK(d_same < UINT32_MAX, "same-scale distance is finite");
    uint32_t d_diff = ce_residual_distance(&book.codes[0], &p2);
    CHECK(d_diff == UINT32_MAX, "cross-scale distance is sentinel UINT32_MAX");

    /* --- get accessor --- */
    const CEResidualCode *got = ce_residual_codebook_get(&book, 0);
    CHECK(got != NULL, "get(0) returns a code");
    CHECK(got->scale_level == 1, "get(0).scale_level matches");
    const CEResidualCode *miss = ce_residual_codebook_get(&book, 99);
    CHECK(miss == NULL, "get(out-of-range) returns NULL");

    /* --- storage descriptor add + unpack --- */
    CEStorage S;
    ce_storage_init(&S, 4);
    TickRGBA tick = (TickRGBA){10, 1, 2, 200};
    ce_residual_storage_add(&S, /*cid=*/7u,
        /*slot=*/4, /*block_idx=*/3,
        /*idx=*/0, /*strength=*/180, tick,
        /*x=*/123, /*y=*/200);
    CHECK(S.count == 1, "storage gained one row");
    CHECK(S.entries[0].type == CE_TYPE_RESIDUAL, "row tagged RESIDUAL");
    CHECK(S.entries[0].canvas_id == 7u, "canvas_id stored");
    CHECK(S.entries[0].slot == 4, "slot stored");
    CHECK(S.entries[0].block_idx == 3, "block_idx stored");

    uint8_t got_idx = 0xFF, got_strength = 0;
    TickRGBA got_tick = (TickRGBA){0};
    uint16_t got_x = 0, got_y = 0;
    int rc = ce_residual_storage_unpack(&S.entries[0],
        &got_idx, &got_strength, &got_tick, &got_x, &got_y);
    CHECK(rc == 0, "unpack returns 0");
    CHECK(got_idx == 0, "descriptor idx = 0");
    CHECK(got_strength == 180, "descriptor strength = 180");
    CHECK(got_tick.r == 10 && got_tick.g == 1
       && got_tick.b == 2  && got_tick.a == 200,
       "descriptor tick roundtrips");
    CHECK(got_x == 123 && got_y == 200, "descriptor x/y roundtrip");

    /* High position values use both bytes (16-bit roundtrip). */
    ce_residual_storage_add(&S, /*cid=*/7u,
        /*slot=*/0, /*block_idx=*/0,
        /*idx=*/0, /*strength=*/100, (TickRGBA){0,0,0,0},
        /*x=*/65530, /*y=*/300);
    int rc_hi = ce_residual_storage_unpack(&S.entries[1],
        NULL, NULL, NULL, &got_x, &got_y);
    CHECK(rc_hi == 0, "unpack high-byte returns 0");
    CHECK(got_x == 65530 && got_y == 300,
          "16-bit position survives high-byte roundtrip");

    /* unpack on wrong type returns -1 */
    CEStorageEntry fake = {0};
    fake.type = CE_TYPE_IMAGE;
    int rc2 = ce_residual_storage_unpack(&fake, NULL, NULL, NULL, NULL, NULL);
    CHECK(rc2 != 0, "unpack rejects non-RESIDUAL entries");

    /* --- v3 .ces trailer roundtrip --- */
    {
        const char *tmp_path = "/tmp/ce_storage_residual_roundtrip.ces";
        CEStorage save_s;
        ce_storage_init(&save_s, 4);
        ce_residual_storage_add(&save_s, /*cid=*/9u, 1, 2, 0, 200,
                                (TickRGBA){5,6,7,8}, 100, 100);

        int wok = ce_storage_save_with_codebook(&save_s, &book, tmp_path);
        CHECK(wok, "save_with_codebook returns ok");

        CEStorage load_s;
        CEResidualCodebook load_book;
        int rok = ce_storage_load_with_codebook(&load_s, &load_book, tmp_path);
        CHECK(rok, "load_with_codebook returns ok");
        CHECK(load_s.count == save_s.count, "storage count roundtrips");
        CHECK(load_book.count == book.count, "codebook count roundtrips");
        if (load_book.count == book.count) {
            int byte_match = (memcmp(&load_book.codes[0].unit,
                                     &book.codes[0].unit,
                                     sizeof(CEUnit)) == 0);
            CHECK(byte_match, "first code unit byte-for-byte");
            CHECK(load_book.codes[0].scale_level == book.codes[0].scale_level,
                  "scale_level roundtrips");
            CHECK(load_book.codes[0].used_count == book.codes[0].used_count,
                  "used_count roundtrips");
        }
        ce_storage_free(&load_s);
        ce_storage_free(&save_s);
        remove(tmp_path);
    }

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
