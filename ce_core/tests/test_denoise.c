/* test_denoise.c — Phase 3: denoise loop, decode, loss. */
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
    printf("=== ce_denoise / ce_decode tests ===\n");

    /* Build storage. */
    CEStorage S; ce_storage_init(&S, 16);
    const char *blocks[] = {
        "the cat sits on the mat",
        "the dog sleeps in the sun",
        "the bird flies above the lake",
        "the fish swims under the bridge",
        "rain falls on the window pane",
        "snow covers the silent street"
    };
    int nb = 6;
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < nb; ++i) {
        ce_storage_ingest(&S, 1, 0, (uint16_t)i, (i == 0) ? NULL : &prev,
                          (const uint8_t *)blocks[i], (uint32_t)strlen(blocks[i]));
        prev = S.entries[S.count - 1].keyframe;
    }

    /* Latent init. */
    CEUnit noise; ce_noise_init(&noise, 0xABCD1234);
    CEUnit prompt_unit; ce_init(&prompt_unit);
    ce_feed(&prompt_unit, (const uint8_t *)"cat on mat", 10);

    CELatentGrid z1, z2;
    float w[5] = { 0.3f, 0.4f, 0.2f, 0.4f, 0.0f };
    ce_latent_init(&z1, &noise, &S.entries[0].keyframe, &S.entries[0].delta,
                   &prompt_unit, NULL, w);
    ce_latent_init(&z2, &noise, &S.entries[0].keyframe, &S.entries[0].delta,
                   &prompt_unit, NULL, w);

    CEGenConfig cfg = ce_gen_config_default();
    cfg.total_steps = 4;

    /* Determinism */
    ce_denoise_loop(&z1, &S, &prompt_unit, 1, &cfg);
    ce_denoise_loop(&z2, &S, &prompt_unit, 1, &cfg);
    int eq = 1;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&z1.cells[i], &z2.cells[i])) { eq = 0; break; }
    CHECK(eq, "denoise_loop is deterministic for fixed inputs");
    CHECK(z1.current_step == cfg.total_steps, "loop sets current_step to total_steps");

    /* Different seed -> different output */
    CEUnit noise_b; ce_noise_init(&noise_b, 0x1111ABCD);
    CELatentGrid z3;
    ce_latent_init(&z3, &noise_b, &S.entries[0].keyframe, &S.entries[0].delta,
                   &prompt_unit, NULL, w);
    ce_denoise_loop(&z3, &S, &prompt_unit, 1, &cfg);
    int diff_cells = 0;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&z1.cells[i], &z3.cells[i])) ++diff_cells;
    CHECK(diff_cells > CE_GRID_N / 4, "different seed -> diverse output (>25% cells differ)");

    /* Different prompt -> different output */
    CEUnit prompt_b; ce_init(&prompt_b);
    ce_feed(&prompt_b, (const uint8_t *)"snow on street", 14);
    CELatentGrid zp;
    ce_latent_init(&zp, &noise, &S.entries[0].keyframe, &S.entries[0].delta,
                   &prompt_b, NULL, w);
    ce_denoise_loop(&zp, &S, &prompt_b, 1, &cfg);
    int diff_prompt = 0;
    for (int i = 0; i < CE_GRID_N; ++i)
        if (!ce_equal(&z1.cells[i], &zp.cells[i])) ++diff_prompt;
    CHECK(diff_prompt > 0, "different prompt -> different output");

    /* image decode */
    static CEImage img1, img2;
    ce_decode_image(&img1, &z1);
    ce_decode_image(&img2, &z1);
    CHECK(memcmp(&img1, &img2, sizeof(CEImage)) == 0, "decode_image deterministic");
    CHECK(img1.width == 256 && img1.height == 256, "decoded image dims");
    /* Image should not be all zeroes. */
    int nonzero_px = 0;
    for (int i = 0; i < 1024; ++i) {
        const CEPixel *p = &img1.pixels[i];
        if (p->r || p->g || p->b) { nonzero_px = 1; break; }
    }
    CHECK(nonzero_px, "decoded image has non-zero pixels");

    /* text decode */
    static uint8_t txt1[CE_GRID_N + 1];
    static uint8_t txt2[CE_GRID_N + 1];
    uint32_t tl1 = 0, tl2 = 0;
    ce_decode_text(txt1, &tl1, &z1);
    ce_decode_text(txt2, &tl2, &z1);
    CHECK(tl1 == tl2 && tl1 == CE_GRID_N, "decode_text length == grid count");
    CHECK(memcmp(txt1, txt2, tl1) == 0, "decode_text deterministic");
    /* Text characters should be printable. */
    int printable = 1;
    for (uint32_t i = 0; i < tl1; ++i) {
        if (txt1[i] < 32 || txt1[i] > 126) { printable = 0; break; }
    }
    CHECK(printable, "decode_text emits printable ASCII");

    /* loss */
    CELoss loss;
    ce_compute_loss(&loss, &z1.cells[0], &z1.cells[0]);
    CHECK(loss.total_loss == 0.0f, "loss(x,x) == 0");
    ce_compute_loss(&loss, &z1.cells[0], &z1.cells[5]);
    CHECK(loss.total_loss > 0.0f, "loss(x,y) > 0 for distinct cells");

    /* update_params bounds */
    CEGenConfig c = ce_gen_config_default();
    CELoss high = { {0,0,0,0}, 200.0f };
    CELoss low  = { {0,0,0,0}, 5.0f };
    int prev_steps = c.total_steps;
    ce_update_params(&c, &high);
    CHECK(c.total_steps >= prev_steps, "high loss does not reduce steps");
    int after_high = c.total_steps;
    ce_update_params(&c, &low);
    CHECK(c.total_steps <= after_high, "low loss does not increase steps");
    CHECK(c.cfg_scale <= 4.0f, "cfg_scale capped");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
