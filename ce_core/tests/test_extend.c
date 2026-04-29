/* test_extend.c — sampler + inpaint helpers. */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_search.h"
#include "../ce_engine.h"
#include "../ce_denoise.h"
#include "../ce_decode.h"
#include "../ce_extend.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

int main(void) {
    printf("=== ce_extend tests ===\n");

    /* --- Sampler --- */
    CEUnit cands[5];
    for (int i = 0; i < 5; ++i) {
        ce_init(&cands[i]);
        uint8_t b = (uint8_t)(i * 37 + 1);
        ce_feed(&cands[i], &b, 1);
    }
    CEUnit out_c, out_d;
    ce_sample(&out_c, cands, 5, CE_SAMPLE_CONFIDENCE, NULL);
    ce_sample(&out_d, cands, 5, CE_SAMPLE_DIVERSITY, NULL);
    CHECK(1, "sampler runs without crashing");
    /* sampler is deterministic */
    CEUnit out_c2; ce_sample(&out_c2, cands, 5, CE_SAMPLE_CONFIDENCE, NULL);
    CHECK(ce_equal(&out_c, &out_c2), "sampler deterministic (confidence)");

    /* count == 0 */
    CEUnit empty; ce_sample(&empty, cands, 0, CE_SAMPLE_CONFIDENCE, NULL);
    CEUnit zero; ce_init(&zero);
    CHECK(ce_equal(&empty, &zero), "sampler with 0 candidates returns zero unit");

    /* count == 1 */
    CEUnit single; ce_sample(&single, cands, 1, CE_SAMPLE_CONFIDENCE, NULL);
    CHECK(ce_equal(&single, &cands[0]), "sampler with 1 candidate returns that one");

    /* --- Inpaint --- */
    CEStorage S; ce_storage_init(&S, 8);
    const char *blocks[] = { "alpha", "beta", "gamma", "delta" };
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < 4; ++i) {
        ce_storage_ingest(&S, 1, 0, (uint16_t)i, (i == 0) ? NULL : &prev,
                          (const uint8_t *)blocks[i], (uint32_t)strlen(blocks[i]));
        prev = S.entries[S.count - 1].keyframe;
    }
    CEInpaintMask m;
    /* mask first half (regenerate), keep second half as anchor */
    for (int i = 0; i < CE_GRID_N; ++i) m.mask[i] = (i < CE_GRID_N / 2) ? 1 : 0;

    CELatentGrid zi;
    zi.width = CE_GRID_W; zi.height = CE_GRID_H; zi.current_step = 0; zi.total_steps = 0;
    for (int i = 0; i < CE_GRID_N; ++i) {
        ce_init(&zi.cells[i]);
        uint8_t b = (uint8_t)i;
        ce_feed(&zi.cells[i], &b, 1);
    }
    CELatentGrid anchor_state = zi;
    CEUnit prompt; ce_init(&prompt); ce_feed(&prompt, (const uint8_t *)"alpha", 5);
    CEGenConfig cfg = ce_gen_config_default(); cfg.total_steps = 2;
    ce_inpaint(&zi, &m, &S, &prompt, 1, &cfg);
    /* anchor cells (mask 0) must be unchanged */
    int anchors_ok = 1;
    for (int i = CE_GRID_N / 2; i < CE_GRID_N; ++i) {
        if (!ce_equal(&zi.cells[i], &anchor_state.cells[i])) { anchors_ok = 0; break; }
    }
    CHECK(anchors_ok, "inpaint preserves anchor cells");
    /* masked cells should have changed */
    int regen = 0;
    for (int i = 0; i < CE_GRID_N / 2; ++i) {
        if (!ce_equal(&zi.cells[i], &anchor_state.cells[i])) ++regen;
    }
    CHECK(regen > 0, "inpaint regenerates masked cells");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
