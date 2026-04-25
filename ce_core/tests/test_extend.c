/* test_extend.c — Phase 4: sampler / memo / hint / inpaint / upscale / audio. */
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

    /* --- Memo --- */
    CELatentGrid z;
    z.width = CE_GRID_W; z.height = CE_GRID_H; z.current_step = 0; z.total_steps = 0;
    for (int i = 0; i < CE_GRID_N; ++i) ce_init(&z.cells[i]);
    CEUnit adj[2];
    ce_init(&adj[0]); ce_feed(&adj[0], (const uint8_t *)"X", 1);
    ce_init(&adj[1]); ce_feed(&adj[1], (const uint8_t *)"Y", 1);
    CEMemoLayer memo;
    memcpy(memo.name, "test", 5);
    memo.adjustments = adj;
    memo.count = 2;

    CELatentGrid before = z;
    ce_memo_apply(&z, &memo);
    int diff = 0;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&before.cells[i], &z.cells[i])) ++diff;
    CHECK(diff > 0, "memo apply changes grid");

    ce_memo_remove(&z, &memo);
    int restored = 1;
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (!ce_equal(&before.cells[i], &z.cells[i])) { restored = 0; break; }
    }
    CHECK(restored, "memo apply+remove is identity");

    /* --- Hint --- */
    CEHintLayer hint;
    hint.type = CE_HINT_EDGE;
    hint.strength = 1.0f;
    for (int i = 0; i < CE_GRID_N; ++i) {
        ce_init(&hint.cells[i]);
        uint8_t b = (uint8_t)(i & 0xFF);
        ce_feed(&hint.cells[i], &b, 1);
    }
    CELatentGrid zh = before;
    ce_hint_apply(&zh, &hint);
    int hint_diff = 0;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&zh.cells[i], &before.cells[i])) ++hint_diff;
    CHECK(hint_diff >= CE_GRID_N - 4, "hint at strength 1.0 affects most cells");

    /* hint strength 0 is identity */
    CEHintLayer hint0 = hint; hint0.strength = 0.0f;
    CELatentGrid zh0 = before;
    ce_hint_apply(&zh0, &hint0);
    int hint_eq = 1;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&zh0.cells[i], &before.cells[i])) { hint_eq = 0; break; }
    CHECK(hint_eq, "hint strength 0 is identity");

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

    /* --- Upscale --- */
    CELatentGrid lo;
    lo.width = CE_GRID_W; lo.height = CE_GRID_H; lo.current_step = 0; lo.total_steps = 0;
    for (int i = 0; i < CE_GRID_N; ++i) {
        ce_init(&lo.cells[i]);
        uint8_t b = (uint8_t)(i ^ 0x55);
        ce_feed(&lo.cells[i], &b, 1);
    }
    static CEHiresGrid hi;
    ce_upscale(&hi, &lo, &S, &cfg);
    CHECK(hi.width == 128 && hi.height == 128, "upscale grid dims");
    /* hires should not be all zeroes */
    int nonzero = 0;
    for (int i = 0; i < 128; ++i) {
        for (int b = 0; b < 64; ++b) {
            if (ce_cbytes(&hi.cells[i])[b] != 0) { nonzero = 1; break; }
        }
        if (nonzero) break;
    }
    CHECK(nonzero, "upscale produces non-zero cells");
    /* hires sub-cells of same lo-cell should differ (perturbation) */
    /* lo cell (0,0) maps to hi cells (0..3, 0..3) */
    int sub_diff = 0;
    for (int sy = 0; sy < 4; ++sy) {
        for (int sx = 0; sx < 4; ++sx) {
            if (sx == 0 && sy == 0) continue;
            int hi_idx = sy * CE_HIRES_GRID_W + sx;
            int hi_seed_idx = 0; /* (0,0) is the seed */
            if (!ce_equal(&hi.cells[hi_idx], &hi.cells[hi_seed_idx])) ++sub_diff;
        }
    }
    CHECK(sub_diff > 0, "upscale perturbs sub-cells");

    /* --- Audio --- */
    static uint8_t audio_data[1024];
    for (int i = 0; i < 1024; ++i) audio_data[i] = (uint8_t)((i * 13 + 7) & 0xFF);
    CEAudioTrack track;
    ce_audio_load(&track, audio_data, 1024, 1);
    CHECK(track.count > 0, "audio_load produces segments");
    float e0 = ce_audio_energy_at(&track, 0, 10);
    float e1 = ce_audio_energy_at(&track, 5, 10);
    CHECK(e0 >= 0.5f && e0 <= 2.0f, "audio energy clamped");
    CHECK(e1 >= 0.5f && e1 <= 2.0f, "audio energy clamped 2");
    /* No-track call returns 1.0 */
    float ne = ce_audio_energy_at(NULL, 0, 10);
    CHECK(ne == 1.0f, "audio_energy_at(NULL) == 1.0");

    ce_audio_free(&track);
    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
