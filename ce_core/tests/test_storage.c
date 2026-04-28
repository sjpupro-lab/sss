/* test_storage.c — Phase 1: storage + noise + query + search + context. */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_search.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

int main(void) {
    printf("=== ce_storage / ce_search tests ===\n");

    CEStorage S;
    ce_storage_init(&S, 4); /* tiny capacity to exercise grow */

    /* Ingest a few text "blocks" into canvas 1, slot 0. */
    const char *blocks[] = {
        "the cat sits on the mat",
        "a dog runs in the park",
        "the cat eats the fish",
        "the fish swims in water",
        "the dog barks at the cat",
        "rain falls on the roof"
    };
    int nb = sizeof(blocks) / sizeof(blocks[0]);
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < nb; ++i) {
        ce_storage_ingest(&S, /*canvas*/1, /*slot*/0, (uint16_t)i,
                          (i == 0) ? NULL : &prev,
                          (const uint8_t *)blocks[i],
                          (uint32_t)strlen(blocks[i]));
        prev = S.entries[S.count - 1].keyframe;
    }
    CHECK((int)S.count == nb, "storage growth + count");
    CHECK(S.capacity >= (uint32_t)nb, "storage capacity grew past initial");

    /* find */
    int32_t f = ce_storage_find(&S, 1, 0, 2);
    CHECK(f == 2, "ce_storage_find returns correct index");
    CHECK(ce_storage_find(&S, 9, 9, 9) == -1, "ce_storage_find missing -> -1");

    /* noise determinism */
    CEUnit n1, n2, n3;
    ce_noise_init(&n1, 42ULL);
    ce_noise_init(&n2, 42ULL);
    ce_noise_init(&n3, 43ULL);
    CHECK(ce_equal(&n1, &n2), "ce_noise_init same seed -> same unit");
    CHECK(!ce_equal(&n1, &n3), "ce_noise_init different seed -> different unit");

    /* query grid */
    CEQueryGrid G1, G2, G3;
    ce_query_from_noise(&G1, 100ULL);
    ce_query_from_noise(&G2, 100ULL);
    ce_query_from_noise(&G3, 101ULL);
    int grid_eq = 1, grid_neq = 0;
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (!ce_equal(&G1.cells[i], &G2.cells[i])) grid_eq = 0;
        if (!ce_equal(&G1.cells[i], &G3.cells[i])) grid_neq = 1;
    }
    CHECK(grid_eq, "query grid determinism");
    CHECK(grid_neq, "query grid changes with seed");
    CHECK(G1.width == CE_GRID_W && G1.height == CE_GRID_H, "query grid dimensions");
    /* No two cells in the same grid should be identical (entropy check). */
    int distinct = 1;
    for (int i = 0; i < 32 && distinct; ++i) {
        for (int j = i + 1; j < 32; ++j) {
            if (ce_equal(&G1.cells[i], &G1.cells[j])) { distinct = 0; break; }
        }
    }
    CHECK(distinct, "query grid cells are pairwise distinct (sampled)");

    /* topk search */
    CESearchResult res[3];
    int n = ce_search_topk(&S, &S.entries[2].keyframe, 3, res);
    CHECK(n == 3, "topk returns k results");
    CHECK(res[0].entry_idx == 2, "topk closest match is the query keyframe");
    CHECK(res[0].distance == 0, "topk identity distance == 0");
    CHECK(res[0].distance <= res[1].distance, "topk sorted ascending #1");
    CHECK(res[1].distance <= res[2].distance, "topk sorted ascending #2");

    /* topk on empty storage */
    CEStorage E; ce_storage_init(&E, 4);
    int n0 = ce_search_topk(&E, &S.entries[0].keyframe, 3, res);
    CHECK(n0 == 0, "topk on empty storage returns 0");
    ce_storage_free(&E);

    /* k > storage_count clamps */
    int n2c = ce_search_topk(&S, &S.entries[0].keyframe, 100, res);
    CHECK(n2c <= (int)S.count, "topk clamps k <= count");

    /* extract_context */
    CECellContext ctx;
    ce_extract_context(&S, 1, 0, 2, /*row_stride=*/3, &ctx);
    CHECK(ce_equal(&ctx.center, &S.entries[2].keyframe), "context center matches stored keyframe");
    CHECK(ctx.neighbor_count >= 1, "context has at least one neighbour");

    /* extract_context on missing canvas */
    CECellContext ctx2;
    ce_extract_context(&S, 999, 0, 0, 32, &ctx2);
    CHECK(ctx2.neighbor_count == 0, "context on missing canvas has 0 neighbours");

    /* select_kd_pair */
    CERetrievedPair pairs[3];
    ce_search_topk(&S, &S.entries[2].keyframe, 3, res);
    ce_select_kd_pair(&S, res, 3, pairs);
    CHECK(pairs[0].relevance == 1.0f, "best match has relevance 1.0");
    CHECK(pairs[1].relevance <= pairs[0].relevance, "relevance non-increasing");
    CHECK(pairs[2].relevance <= pairs[1].relevance, "relevance non-increasing 2");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
