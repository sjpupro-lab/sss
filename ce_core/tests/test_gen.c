/* test_gen.c — top-level integration test: prompt -> image / text. */
#include "../ce_gen.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

int main(void) {
    printf("=== ce_gen integration ===\n");

    /* Build storage from a small "training set". */
    CEStorage S; ce_storage_init(&S, 32);
    const char *corpus[] = {
        "the orange cat eats fish",
        "the gray cat plays with yarn",
        "the white dog runs in grass",
        "the black bird flies over the lake",
        "rain falls on the city street",
        "snow covers the mountain top",
        "the moon rises over the sea",
        "the sun sets behind the trees"
    };
    int n = 8;
    CEUnit prev; ce_init(&prev);
    for (int i = 0; i < n; ++i) {
        ce_storage_ingest(&S, 1, 0, (uint16_t)i, (i == 0) ? NULL : &prev,
                          (const uint8_t *)corpus[i], (uint32_t)strlen(corpus[i]));
        prev = S.entries[S.count - 1].keyframe;
    }

    CEGenConfig cfg = ce_gen_config_default();
    cfg.total_steps = 4; /* keep test fast */

    /* --- Image generation --- */
    static CEImage img1, img2;

    clock_t t0 = clock();
    ce_generate_image(&img1, &S, "orange cat eating fish", 0xC0FFEE,
                      &cfg, NULL, NULL, NULL);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("  ce_generate_image (256x256, %d steps): %.3f s\n", cfg.total_steps, secs);

    ce_generate_image(&img2, &S, "orange cat eating fish", 0xC0FFEE,
                      &cfg, NULL, NULL, NULL);
    CHECK(memcmp(&img1, &img2, sizeof(CEImage)) == 0, "image generation deterministic on (prompt, seed)");

    /* Different seed -> visibly different image. */
    static CEImage img3;
    ce_generate_image(&img3, &S, "orange cat eating fish", 0xDEADBEEF,
                      &cfg, NULL, NULL, NULL);
    int diff_px = 0;
    for (int i = 0; i < CE_IMAGE_W * CE_IMAGE_H; ++i) {
        if (img1.pixels[i].r != img3.pixels[i].r ||
            img1.pixels[i].g != img3.pixels[i].g ||
            img1.pixels[i].b != img3.pixels[i].b) ++diff_px;
    }
    CHECK(diff_px > (CE_IMAGE_W * CE_IMAGE_H) / 4, "different seed -> diverse image (>25% px differ)");

    /* Different prompt -> different image. */
    static CEImage img4;
    ce_generate_image(&img4, &S, "snow on the mountain top", 0xC0FFEE,
                      &cfg, NULL, NULL, NULL);
    int diff_prompt_px = 0;
    for (int i = 0; i < CE_IMAGE_W * CE_IMAGE_H; ++i) {
        if (img1.pixels[i].r != img4.pixels[i].r ||
            img1.pixels[i].g != img4.pixels[i].g ||
            img1.pixels[i].b != img4.pixels[i].b) ++diff_prompt_px;
    }
    CHECK(diff_prompt_px > 0, "different prompt -> different image");

    /* --- Text generation --- */
    static uint8_t txt[CE_GRID_N + 1];
    uint32_t txt_len = 0;
    ce_generate_text(txt, &txt_len, &S, "orange cat eating fish", 0xC0FFEE, &cfg);
    CHECK(txt_len == CE_GRID_N, "text length matches grid count");
    txt[txt_len] = 0;
    int printable = 1;
    for (uint32_t i = 0; i < txt_len; ++i) {
        if (txt[i] < 32 || txt[i] > 126) { printable = 0; break; }
    }
    CHECK(printable, "text output is printable");

    /* --- Inpaint --- */
    CEInpaintMask mask;
    for (int i = 0; i < CE_GRID_N; ++i) mask.mask[i] = (i < 256) ? 1 : 0;
    static CEImage img_inp;
    ce_generate_inpaint(&img_inp, &img1, &mask, &S,
                        "snow on the mountain top", 0xC0FFEE, &cfg);
    /* Outside the masked region (cells with mask=0), pixel block ought to
     * be similar to img1 (the original) — anchor preservation. */
    int anchor_match = 0;
    for (int cy = 0; cy < CE_GRID_H; ++cy) {
        for (int cx = 0; cx < CE_GRID_W; ++cx) {
            int ci = cy * CE_GRID_W + cx;
            if (mask.mask[ci] == 0) {
                /* check the central pixel of the cell block matches input */
                int px = cx * CE_BLOCK + 3;
                int py = cy * CE_BLOCK + 3;
                int orig = img1.pixels[py * CE_IMAGE_W + px].r;
                int inp  = img_inp.pixels[py * CE_IMAGE_W + px].r;
                if (orig == inp) ++anchor_match;
            }
        }
    }
    /* Most anchor cells should match exactly; allow some interpolation slack. */
    CHECK(anchor_match > 0, "inpaint preserves at least some anchor pixels exactly");

    /* --- Upscale --- */
    CELatentGrid lo;
    lo.width = CE_GRID_W; lo.height = CE_GRID_H; lo.current_step = 0; lo.total_steps = 0;
    CEUnit noise; ce_noise_init(&noise, 0xAAAA);
    for (int i = 0; i < CE_GRID_N; ++i) lo.cells[i] = noise;
    static CEHiresGrid hi;
    ce_generate_upscale(&hi, &lo, &S, &cfg);
    CHECK(hi.width == 128 && hi.height == 128, "upscale dims");

    ce_storage_free(&S);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
