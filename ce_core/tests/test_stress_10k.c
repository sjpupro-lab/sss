/* test_stress_10k.c — verify CEStorage + search + save/load scale to >=10000
 * entries within reasonable time/memory.
 *
 * Synthesizes N=12000 8x8 RGBA blocks (random per-block color) and pushes
 * them through the full CE_TYPE_IMAGE training path:
 *
 *   ce_feed_image -> ce_delta -> ce_storage_add_typed
 *
 * Then verifies:
 *   - storage.count == N
 *   - ce_search_by_type(IMAGE) returns k entries < 1s
 *   - ce_storage_save / load round-trips identical content
 *   - ce_generate_image_typed completes on the loaded storage
 *
 * Threshold is 1s search and 30s end-to-end so the test stays useful as
 * a perf canary without being flaky on slow CI.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_search.h"
#include "../ce_feed_image.h"
#include "../ce_decode.h"
#include "../ce_gen.h"
#include "../ce_denoise.h"
#include "../ce_type.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* 12000 = above the 10000 training minimum. Override at compile time
 * with -DN_ENTRIES=N to stress further. */
#ifndef N_ENTRIES
#define N_ENTRIES 12000
#endif

static void fill_block_seeded(uint8_t *blk, uint32_t seed) {
    /* Deterministic per-block color so save/load round-trip is checkable. */
    uint32_t r = (seed * 0x9E3779B1u) ^ 0xA5A5A5A5u;
    uint8_t cR = (uint8_t)(r >> 24);
    uint8_t cG = (uint8_t)(r >> 16);
    uint8_t cB = (uint8_t)(r >> 8);
    for (int i = 0; i < 64; ++i) {
        blk[i * 4 + 0] = cR;
        blk[i * 4 + 1] = cG;
        blk[i * 4 + 2] = cB;
        blk[i * 4 + 3] = 255;
    }
}

int main(void) {
    printf("=== ce_stress_10k (N=%d) ===\n", N_ENTRIES);

    CEStorage S;
    ce_storage_init(&S, 256);

    /* 1. Bulk ingest — measure throughput. */
    double t0 = now_s();
    uint8_t blk[64 * 4];
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < N_ENTRIES; ++i) {
        fill_block_seeded(blk, (uint32_t)i + 1u);
        CEUnit fresh;
        ce_feed_image(&fresh, blk);
        CEUnit zero, delta;
        ce_init(&zero);
        ce_delta(&delta, (i == 0) ? &zero : &prev, &fresh);
        ce_storage_add_typed(&S, 1, (uint16_t)(i / 1024), (uint16_t)(i % 1024),
                             CE_TYPE_IMAGE, &fresh, &delta);
        prev = fresh;
    }
    double t_ingest = now_s() - t0;
    printf("  ingest: %.2fs  (%.0f blocks/s)\n",
           t_ingest, N_ENTRIES / t_ingest);

    CHECK(S.count == N_ENTRIES, "storage holds N entries");
    CHECK(t_ingest < 30.0, "ingest under 30s");

    /* 2. Search latency. */
    CEUnit q; ce_noise_init(&q, 0xCAFEBABE);
    CESearchResult res[8];
    double t1 = now_s();
    int found = ce_search_by_type(&S, &q, CE_TYPE_IMAGE, 8, res);
    double t_search = now_s() - t1;
    printf("  search: %.4fs  found=%d\n", t_search, found);

    CHECK(found == 8, "ce_search_by_type returns 8");
    CHECK(t_search < 1.0, "single search under 1s");

    /* 3. Save / load round-trip. PID-suffixed path keeps parallel test
     * runs from clobbering each other's scratch file and avoids spaces
     * in the filename so downstream tooling stays simple. */
    char path[64];
    snprintf(path, sizeof(path), "/tmp/ce_stress_%d_%d.ces",
             (int)getpid(), N_ENTRIES);
    double t2 = now_s();
    int saved = ce_storage_save(&S, path);
    double t_save = now_s() - t2;
    printf("  save:   %.2fs\n", t_save);
    CHECK(saved == 1, "ce_storage_save writes successfully");
    CHECK(t_save < 10.0, "save under 10s");

    CEStorage L;
    double t3 = now_s();
    int loaded = ce_storage_load(&L, path);
    double t_load = now_s() - t3;
    printf("  load:   %.2fs  count=%u\n", t_load, L.count);
    CHECK(loaded == 1, "ce_storage_load reads successfully");
    CHECK(L.count == S.count, "loaded count matches saved count");

    /* Spot check a few entries. */
    int spots_ok = 1;
    for (int i = 0; i < N_ENTRIES; i += 1500) {
        if (memcmp(&S.entries[i].keyframe, &L.entries[i].keyframe,
                   sizeof(CEUnit)) != 0) spots_ok = 0;
    }
    CHECK(spots_ok, "round-trip preserves keyframe bytes (spot checks)");

    /* 4. End-to-end generate from the loaded storage. */
    CEImage *img = (CEImage *)calloc(1, sizeof(CEImage));
    CEGenConfig fast;
    fast = ce_gen_config_default();
    fast.total_steps = 4;
    double t4 = now_s();
    ce_generate_image_typed(img, &L, CE_TYPE_IMAGE,
                            "stress prompt", 0xC0DE, &fast, 30);
    double t_gen = now_s() - t4;
    printf("  gen:    %.2fs  (steps=4 wave=30 over %u entries)\n",
           t_gen, L.count);
    CHECK(img->width == CE_IMAGE_W,
          "generate produces an image of expected size");
    CHECK(t_gen < 30.0, "end-to-end generate under 30s");

    free(img);
    ce_storage_free(&L);
    ce_storage_free(&S);
    remove(path);

    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
