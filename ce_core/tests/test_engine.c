/* test_engine.c — Phase 2: latent grid + UNet-equivalent ops. */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_search.h"
#include "../ce_engine.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static int grids_equal(const CELatentGrid *a, const CELatentGrid *b) {
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&a->cells[i], &b->cells[i])) return 0;
    return 1;
}

static int grids_diff_count(const CELatentGrid *a, const CELatentGrid *b) {
    int n = 0;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&a->cells[i], &b->cells[i])) ++n;
    return n;
}

int main(void) {
    printf("=== ce_engine tests ===\n");

    /* Build a small storage + retrieved pair list for input. */
    CEStorage S; ce_storage_init(&S, 8);
    const char *blocks[] = {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta"
    };
    int nb = 6;
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < nb; ++i) {
        ce_storage_ingest(&S, 1, 0, (uint16_t)i, (i == 0) ? NULL : &prev,
                          (const uint8_t *)blocks[i], (uint32_t)strlen(blocks[i]));
        prev = S.entries[S.count - 1].keyframe;
    }

    CEUnit noise; ce_noise_init(&noise, 0xC001CAFE);
    CEUnit prompt; ce_init(&prompt);
    ce_feed(&prompt, (const uint8_t *)"prompt-text", 11);
    float w[5] = { 0.3f, 0.4f, 0.3f, 0.5f, 0.0f };

    CESearchResult res[3];
    ce_search_topk(&S, &noise, 3, res);
    CERetrievedPair pairs[3];
    ce_select_kd_pair(&S, res, 3, pairs);

    /* latent_init determinism */
    CELatentGrid z1, z2, z3;
    ce_latent_init(&z1, &noise, &pairs[0].keyframe_latent, &pairs[0].delta_latent,
                   &prompt, NULL, w);
    ce_latent_init(&z2, &noise, &pairs[0].keyframe_latent, &pairs[0].delta_latent,
                   &prompt, NULL, w);
    CHECK(grids_equal(&z1, &z2), "latent_init deterministic");
    CHECK(z1.width == CE_GRID_W && z1.height == CE_GRID_H, "latent dims");

    /* different seed -> different grid */
    CEUnit noise_b; ce_noise_init(&noise_b, 0xDEADBEEF);
    ce_latent_init(&z3, &noise_b, &pairs[0].keyframe_latent, &pairs[0].delta_latent,
                   &prompt, NULL, w);
    CHECK(!grids_equal(&z1, &z3), "latent_init differs on different noise");

    /* Cells in the grid are not all equal (variety guard). */
    int distinct = 0;
    for (int i = 0; i < 16; ++i) {
        if (!ce_equal(&z1.cells[0], &z1.cells[i + 1])) { distinct = 1; break; }
    }
    CHECK(distinct, "latent cells diverge across grid");

    /* down: z changes when moving toward keyframes */
    z1.total_steps = 10; z1.current_step = 0;
    static CEUnit skip[CE_GRID_N];
    CELatentGrid before = z1;
    ce_down(&z1, pairs, 3, skip);
    CHECK(grids_diff_count(&before, &z1) > 0, "ce_down moves the grid");
    /* skip captured pre-Down state */
    int skip_match = 1;
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (!ce_equal(&skip[i], &before.cells[i])) { skip_match = 0; break; }
    }
    CHECK(skip_match, "ce_down records pre-Down state into skip_buffer");

    /* down deterministic */
    CELatentGrid zr1, zr2;
    ce_latent_init(&zr1, &noise, &pairs[0].keyframe_latent, &pairs[0].delta_latent, &prompt, NULL, w);
    ce_latent_init(&zr2, &noise, &pairs[0].keyframe_latent, &pairs[0].delta_latent, &prompt, NULL, w);
    zr1.total_steps = zr2.total_steps = 10;
    static CEUnit s1[CE_GRID_N], s2[CE_GRID_N];
    ce_down(&zr1, pairs, 3, s1);
    ce_down(&zr2, pairs, 3, s2);
    CHECK(grids_equal(&zr1, &zr2), "ce_down deterministic");

    /* conv changes the grid */
    CELatentGrid zc = zr1;
    ce_conv(&zc, NULL, 1);
    CHECK(grids_diff_count(&zr1, &zc) > 0, "ce_conv perturbs the grid");

    /* self-attention deterministic */
    CELatentGrid sa1 = zc, sa2 = zc;
    static CEAttentionLink linkbuf[8];
    ce_self_attention(&sa1, 4, linkbuf);
    ce_self_attention(&sa2, 4, linkbuf);
    CHECK(grids_equal(&sa1, &sa2), "self_attention deterministic");

    /* cross-attention with cfg=0 is identity */
    CELatentGrid cx0 = sa1;
    ce_cross_attention(&cx0, &prompt, 1, 0.0f);
    CHECK(grids_equal(&cx0, &sa1), "cross_attention(cfg=0) is identity");

    /* cross-attention with cfg!=0 moves cells */
    CELatentGrid cx1 = sa1;
    ce_cross_attention(&cx1, &prompt, 1, 1.0f);
    CHECK(grids_diff_count(&sa1, &cx1) > 0, "cross_attention(cfg=1) moves cells");

    /* residual: weight 0 -> before, weight 1 -> after */
    CEUnit before_u = sa1.cells[0];
    CEUnit after_u  = cx1.cells[0];
    CEUnit r0, r1;
    ce_residual(&r0, &before_u, &after_u, 0.0f);
    ce_residual(&r1, &before_u, &after_u, 1.0f);
    CHECK(ce_equal(&r0, &before_u), "residual(w=0) == before");
    CHECK(ce_equal(&r1, &after_u),  "residual(w=1) == after");

    /* skip_connect with weight 0 is identity */
    CELatentGrid sk0 = cx1;
    ce_skip_connect(&sk0, skip, 0.0f);
    CHECK(grids_equal(&sk0, &cx1), "skip_connect(w=0) is identity");

    /* skip_connect pulls toward skip */
    CELatentGrid sk1 = cx1;
    ce_skip_connect(&sk1, skip, 1.0f);
    int closer = 0;
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (ce_distance(&sk1.cells[i], &skip[i]) <= ce_distance(&cx1.cells[i], &skip[i])) {
            ++closer;
        }
    }
    CHECK(closer >= CE_GRID_N - 8, "skip_connect(w=1) pulls cells toward skip_buffer");

    /* up changes things */
    CELatentGrid up = sk1;
    ce_up(&up, skip, pairs, 3);
    CHECK(grids_diff_count(&sk1, &up) > 0, "ce_up modifies the grid");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
