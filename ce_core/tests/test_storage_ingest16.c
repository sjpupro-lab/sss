/* test_storage_ingest16.c — exercise ce_storage_ingest_rgba_16.
 *
 * Confirms:
 *   - 32x32 RGBA -> 4 blocks (2x2 16x16 grid) -> 16 entries (4 quadrants each)
 *   - all entries CE_TYPE_IMAGE with the same canvas_id
 *   - slot = block-row, (block_idx >> 2) = block-col, (block_idx & 3) = quadrant
 *   - 17x17 image (edge padding) -> still exactly 4 blocks -> 16 entries
 *   - deterministic: two ingests of the same buffer produce identical entries
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_feed_image.h"
#include "../ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static uint8_t *make_solid_rgba(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *p = (uint8_t *)malloc((size_t)w * h * 4u);
    for (int i = 0; i < w * h; ++i) {
        p[i * 4 + 0] = r;
        p[i * 4 + 1] = g;
        p[i * 4 + 2] = b;
        p[i * 4 + 3] = 255;
    }
    return p;
}

static int unit_eq(const CEUnit *a, const CEUnit *b) {
    return memcmp(a, b, sizeof(CEUnit)) == 0;
}

int main(void) {
    printf("=== test_storage_ingest16 ===\n");

    /* ---- 32x32 case: 2x2 16x16 grid = 4 blocks * 4 quadrants = 16 ---- */
    {
        CEStorage s; ce_storage_init(&s, 64);
        uint8_t *rgba = make_solid_rgba(32, 32, 200, 50, 50);
        uint32_t added = ce_storage_ingest_rgba_16(&s, 0xDEADBEEF, rgba, 32, 32);

        CHECK(added == 16, "32x32 -> 16 entries (2x2 blocks * 4 quadrants)");
        CHECK(s.count == 16, "storage count matches");

        int all_image = 1, all_canvas = 1;
        for (uint32_t i = 0; i < s.count; ++i) {
            if (s.entries[i].type != CE_TYPE_IMAGE) all_image = 0;
            if (s.entries[i].canvas_id != 0xDEADBEEF) all_canvas = 0;
        }
        CHECK(all_image,  "all entries tagged CE_TYPE_IMAGE");
        CHECK(all_canvas, "all entries share canvas_id");

        /* slot = block-row (0 or 1), block_idx >> 2 = block-col, & 3 = quadrant */
        int layout_ok = 1;
        int counts[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < s.count; ++i) {
            uint16_t slot = s.entries[i].slot;
            uint16_t bcol = s.entries[i].block_idx >> 2;
            uint16_t quad = s.entries[i].block_idx & 0x3;
            if (slot > 1 || bcol > 1 || quad > 3) layout_ok = 0;
            ++counts[quad];
        }
        CHECK(layout_ok, "slot/block_idx layout within bounds");
        CHECK(counts[0] == 4 && counts[1] == 4 && counts[2] == 4 && counts[3] == 4,
              "each quadrant 0..3 appears exactly 4 times");

        free(rgba);
        ce_storage_free(&s);
    }

    /* ---- 17x17 edge padding: still 2x2 blocks = 16 entries ---- */
    {
        CEStorage s; ce_storage_init(&s, 64);
        uint8_t *rgba = make_solid_rgba(17, 17, 50, 200, 50);
        uint32_t added = ce_storage_ingest_rgba_16(&s, 1u, rgba, 17, 17);
        CHECK(added == 16, "17x17 (edge padding) -> 16 entries");
        free(rgba);
        ce_storage_free(&s);
    }

    /* ---- determinism ---- */
    {
        CEStorage a, b;
        ce_storage_init(&a, 64);
        ce_storage_init(&b, 64);
        uint8_t *rgba = make_solid_rgba(32, 32, 80, 80, 200);
        ce_storage_ingest_rgba_16(&a, 0x42u, rgba, 32, 32);
        ce_storage_ingest_rgba_16(&b, 0x42u, rgba, 32, 32);

        int identical = (a.count == b.count);
        for (uint32_t i = 0; identical && i < a.count; ++i) {
            if (a.entries[i].slot      != b.entries[i].slot)      identical = 0;
            if (a.entries[i].block_idx != b.entries[i].block_idx) identical = 0;
            if (!unit_eq(&a.entries[i].keyframe, &b.entries[i].keyframe)) identical = 0;
            if (!unit_eq(&a.entries[i].delta,    &b.entries[i].delta))    identical = 0;
        }
        CHECK(identical, "two ingests of same buffer produce identical entries");

        free(rgba);
        ce_storage_free(&a);
        ce_storage_free(&b);
    }

    /* ---- NULL / zero-size guard ---- */
    {
        CEStorage s; ce_storage_init(&s, 16);
        uint32_t a1 = ce_storage_ingest_rgba_16(&s, 0u, NULL, 32, 32);
        uint32_t a2 = ce_storage_ingest_rgba_16(&s, 0u, (const uint8_t *)"x", 0, 32);
        CHECK(a1 == 0 && a2 == 0 && s.count == 0,
              "NULL or zero-dim input adds zero entries");
        ce_storage_free(&s);
    }

    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
