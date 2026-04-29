/* test_slig_persist.c — SLIG sets persistence + tick-sorted iteration.
 *
 * Verifies:
 *   1. ce_storage_persist_slig_set writes one CE_TYPE_SLIG entry per
 *      cell, with the (slot, block_idx) layout the loader expects.
 *   2. ce_storage_load_slig_sets reassembles the 3×3 grid.
 *   3. hybrid_vae_encode persistence end-to-end: load matches encode.
 *   4. ce_tick_sorted_indices walks SLIG entries in (B, G, R, A) order.
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_tick.h"
#include "../ce_hybrid_vae.h"
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

static void make_synth_rgba(uint8_t *rgba, int seed) {
    /* 2×2 quadrant fill so the SLIG decomposer produces non-trivial
     * structure cells without every pixel being identical. */
    static const uint8_t Q[4][3] = {
        {220,  40,  40}, {40,  200,  60},
        { 30,  60, 220}, {220, 200,  40}};
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 256; ++x) {
            int q = (y < 128 ? 0 : 2) + (x < 128 ? 0 : 1);
            int i = (y * 256 + x) * 4;
            rgba[i + 0] = (uint8_t)(Q[q][0] ^ (seed & 0x07));
            rgba[i + 1] = (uint8_t)(Q[q][1] ^ ((seed >> 1) & 0x07));
            rgba[i + 2] = (uint8_t)(Q[q][2] ^ ((seed >> 2) & 0x07));
            rgba[i + 3] = 255;
        }
    }
}

int main(void) {
    printf("=== ce_storage SLIG persistence + tick sort ===\n");

    /* --- Direct persist/load API --- */
    SligCellSet  set;
    memset(&set, 0, sizeof(set));
    set.scale_level = 1;     /* MID */
    set.channel     = 0;     /* Y */
    set.num_cells   = 5;
    for (uint32_t i = 0; i < set.num_cells; ++i) {
        ce_init(&set.cells[i]);
        uint8_t b = (uint8_t)(0x10 + i);
        ce_feed(&set.cells[i], &b, 1);
    }

    CEStorage S;
    ce_storage_init(&S, 16);
    uint32_t added = ce_storage_persist_slig_set(&S, /*cid=*/42u,
        (const struct SligCellSet *)&set);
    CHECK(added == 5, "persist appends 5 entries");
    CHECK(S.count == 5, "storage.count == 5");
    CHECK(S.entries[0].type == CE_TYPE_SLIG, "type tag is CE_TYPE_SLIG");
    CHECK(S.entries[0].slot == 1 * SLIG_NUM_CHANNELS + 0,
          "slot encodes (scale=1, channel=0)");
    CHECK(S.entries[0].block_idx == 0, "first cell has block_idx=0");
    CHECK(S.entries[4].block_idx == 4, "fifth cell has block_idx=4");

    SligCellSet roundtrip[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS];
    uint32_t loaded = ce_storage_load_slig_sets(&S, /*cid=*/42u,
        (struct SligCellSet *)roundtrip);
    CHECK(loaded == 5, "load returns 5 cells");
    CHECK(roundtrip[1][0].num_cells == 5, "MID×Y bucket has 5 cells");
    CHECK(roundtrip[0][0].num_cells == 0, "COARSE×Y bucket empty");
    int byte_match = 1;
    for (uint32_t i = 0; i < 5; ++i) {
        if (memcmp(&roundtrip[1][0].cells[i], &set.cells[i],
                   sizeof(CEUnit)) != 0) byte_match = 0;
    }
    CHECK(byte_match, "loaded cells match original byte-for-byte");

    /* Wrong canvas_id returns nothing. */
    SligCellSet empty[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS];
    uint32_t z = ce_storage_load_slig_sets(&S, /*cid=*/999u,
        (struct SligCellSet *)empty);
    CHECK(z == 0, "wrong canvas_id loads nothing");

    ce_storage_free(&S);

    /* --- End-to-end hybrid_vae encode + load --- */
    CEStorage T;
    ce_storage_init(&T, 64);
    SligCodebook book;
    slig_codebook_init(&book);
    uint8_t *rgba = (uint8_t *)malloc(256 * 256 * 4);
    CHECK(rgba != NULL, "rgba alloc");
    if (rgba) {
        make_synth_rgba(rgba, /*seed=*/1);
        HybridEncodeResult res;
        hybrid_vae_encode(&res, &T, &book, rgba, 256, 256, /*cid=*/77u);
        CHECK(res.total_cells > 0, "encode produces SLIG cells");
        CHECK(res.block_entries == 1024, "encode keeps 1024 block stamps");

        uint32_t hyb_count = 0;
        for (uint32_t i = 0; i < T.count; ++i) {
            if (T.entries[i].type == CE_TYPE_SLIG) ++hyb_count;
        }
        CHECK(hyb_count == res.total_cells,
              "storage CE_TYPE_SLIG count matches HybridEncodeResult.total_cells");

        SligCellSet rt[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS];
        uint32_t r = ce_storage_load_slig_sets(&T, 77u,
            (struct SligCellSet *)rt);
        CHECK(r == res.total_cells,
              "load_slig_sets recovers all cells");
        free(rgba);
    }

    /* --- Tick-sorted iteration --- */
    /* Walk only the SLIG entries — they have B (channel) ascending,
     * then G (scale) ascending, then R (block_idx) ascending. Confirm
     * the sort produces a non-decreasing sequence. */
    uint32_t idx[1024];
    uint32_t n = ce_tick_sorted_indices(&T, 77u,
        (1u << CE_TYPE_SLIG), idx, 1024);
    CHECK(n > 0, "tick sort returns SLIG entries");
    int monotonic = 1;
    TickRGBA prev_t = ce_tick_from_entry(&T.entries[idx[0]]);
    for (uint32_t i = 1; i < n; ++i) {
        TickRGBA cur = ce_tick_from_entry(&T.entries[idx[i]]);
        if (ce_tick_compare(prev_t, cur) > 0) { monotonic = 0; break; }
        prev_t = cur;
    }
    CHECK(monotonic, "tick-sorted indices are non-decreasing");

    /* B-channel is slowest: across the sorted sequence, channel index
     * (slot % 3) must never decrease. */
    int b_monotonic = 1;
    int prev_b = -1;
    for (uint32_t i = 0; i < n; ++i) {
        int b = T.entries[idx[i]].slot % SLIG_NUM_CHANNELS;
        if (b < prev_b) { b_monotonic = 0; break; }
        prev_b = b;
    }
    CHECK(b_monotonic, "channel (B-tick) never decreases under tick sort");

    ce_storage_free(&T);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
