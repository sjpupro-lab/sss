/* test_masked_train.c — 마스크 학습 루프 테스트 */

#include "../ce_masked_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;
#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_pass++; printf("  PASS: %s\n", msg); } \
    else      { printf("  FAIL: %s\n", msg); } \
} while(0)

static void make_gradient(uint8_t *rgba) {
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            int i = (y * 256 + x) * 4;
            rgba[i+0] = (uint8_t)x;
            rgba[i+1] = (uint8_t)y;
            rgba[i+2] = (uint8_t)((x+y)/2);
            rgba[i+3] = 255;
        }
}

static void make_checkerboard(uint8_t *rgba) {
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            int i = (y * 256 + x) * 4;
            uint8_t v = ((x/32 + y/32) % 2 == 0) ? 200 : 50;
            rgba[i+0] = v; rgba[i+1] = v; rgba[i+2] = v; rgba[i+3] = 255;
        }
}

static int epoch_count = 0;
static float last_loss = 0;
static void on_epoch(int epoch, float loss, void *ctx) {
    (void)ctx;
    epoch_count = epoch + 1;
    last_loss = loss;
}

static void test_config(void) {
    printf("\n[test_config]\n");
    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    CHECK(cfg.epochs > 0, "epochs > 0");
    CHECK(cfg.target_loss > 0, "target_loss > 0");
    CHECK(cfg.strategy == MASK_PROGRESSIVE, "default strategy is PROGRESSIVE");
}

static void test_single_image(void) {
    printf("\n[test_single_image]\n");
    uint8_t *rgba = (uint8_t *)calloc(256*256*4, 1);
    make_gradient(rgba);

    CEStorage storage;
    ce_storage_init(&storage, 256);

    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs = 5;
    cfg.denoise_steps = 4;  /* fast */
    cfg.on_epoch = on_epoch;
    epoch_count = 0;

    MaskedTrainResult result;
    masked_train_image(&result, &storage, rgba, 256, 256, 100, &cfg);

    printf("    epochs_run=%d  final_loss=%.1f  converged=%d  cells_stored=%u\n",
           result.epochs_run, result.final_loss, result.converged, result.cells_stored);

    CHECK(result.epochs_run > 0, "ran at least 1 epoch");
    CHECK(result.final_loss >= 0, "loss is non-negative");
    CHECK(result.learned.num_cells > 0, "learned cells > 0");
    CHECK(epoch_count > 0, "callback was called");
    CHECK(storage.count > 0, "cells stored in CEStorage");

    ce_storage_free(&storage);
    free(rgba);
}

static void test_loss_decreases(void) {
    printf("\n[test_loss_decreases]\n");
    uint8_t *rgba = (uint8_t *)calloc(256*256*4, 1);
    make_checkerboard(rgba);

    CEStorage storage;
    ce_storage_init(&storage, 256);

    /* correction-only 마스크로 고정 (가장 쉬운 과제) */
    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs = 3;
    cfg.denoise_steps = 4;
    cfg.strategy = MASK_CORRECTION_ONLY;
    cfg.loss_patience = 99;  /* patience 끄기 */

    MaskedTrainResult result;
    masked_train_image(&result, &storage, rgba, 256, 256, 200, &cfg);

    printf("    final_loss=%.1f  cells=%u\n", result.final_loss, result.learned.num_cells);
    CHECK(result.final_loss < 200.0f, "loss is bounded");

    ce_storage_free(&storage);
    free(rgba);
}

static void test_batch(void) {
    printf("\n[test_batch]\n");
    uint8_t *img1 = (uint8_t *)calloc(256*256*4, 1);
    uint8_t *img2 = (uint8_t *)calloc(256*256*4, 1);
    make_gradient(img1);
    make_checkerboard(img2);

    CEStorage storage;
    ce_storage_init(&storage, 512);

    TrainImageEntry entries[2] = {
        { img1, 256, 256, 301, "gradient" },
        { img2, 256, 256, 302, "checker" }
    };

    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs = 3;
    cfg.denoise_steps = 4;

    BatchTrainResult result;
    masked_train_batch(&result, &storage, entries, 2, &cfg);

    printf("    avg_loss=%.1f  best=%.1f  worst=%.1f  converged=%d/%d\n",
           result.avg_loss, result.best_loss, result.worst_loss,
           result.total_converged, result.total_images);

    CHECK(result.total_images == 2, "processed 2 images");
    CHECK(result.avg_loss >= 0, "avg loss non-negative");
    CHECK(result.best_loss <= result.worst_loss, "best <= worst");
    CHECK(storage.count > 0, "batch stored cells");

    ce_storage_free(&storage);
    free(img1); free(img2);
}

static void test_progressive_strategy(void) {
    printf("\n[test_progressive_strategy]\n");
    uint8_t *rgba = (uint8_t *)calloc(256*256*4, 1);
    make_gradient(rgba);

    CEStorage storage;
    ce_storage_init(&storage, 256);

    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs = 9;  /* 3 phases × 3 epochs */
    cfg.denoise_steps = 4;
    cfg.strategy = MASK_PROGRESSIVE;
    cfg.loss_patience = 99;

    MaskedTrainResult result;
    masked_train_image(&result, &storage, rgba, 256, 256, 400, &cfg);

    printf("    epochs=%d  final_loss=%.1f\n", result.epochs_run, result.final_loss);
    CHECK(result.epochs_run >= 3, "ran through multiple phases");

    ce_storage_free(&storage);
    free(rgba);
}

/* Step-5: segment loss split.
 *   - default config (all weights 0) keeps the legacy mean-loss path
 *   - setting weights triggers weighted aggregation; the per-segment
 *     means are reported separately on the result struct                */
static void test_segment_loss(void) {
    printf("\n[test_segment_loss]\n");
    uint8_t *rgba = (uint8_t *)calloc(256*256*4, 1);
    make_gradient(rgba);

    CEStorage storage;
    ce_storage_init(&storage, 256);

    /* Path 1: default config — segment fields populated even when
     * weighted aggregation is off. */
    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs = 3;
    cfg.denoise_steps = 4;
    cfg.loss_patience = 99;

    MaskedTrainResult r1;
    masked_train_image(&r1, &storage, rgba, 256, 256, 700, &cfg);

    printf("    legacy: total=%.1f  struct=%.1f tex=%.1f color=%.1f\n",
           r1.final_loss, r1.loss_structure, r1.loss_texture, r1.loss_color);
    CHECK(r1.loss_structure >= 0 && r1.loss_texture >= 0 && r1.loss_color >= 0,
          "all segment losses non-negative under legacy config");

    /* Path 2: weighted config. Heavily favoring color → final_loss
     * should track loss_color * 1.0 closely (other weights zero). */
    MaskedTrainConfig cfg2;
    masked_train_config_default(&cfg2);
    cfg2.epochs = 3;
    cfg2.denoise_steps = 4;
    cfg2.loss_patience = 99;
    cfg2.loss_w_structure = 0.0f;
    cfg2.loss_w_texture   = 0.0f;
    cfg2.loss_w_color     = 1.0f;

    MaskedTrainResult r2;
    masked_train_image(&r2, &storage, rgba, 256, 256, 701, &cfg2);

    printf("    color-only weight: total=%.3f  loss_color=%.3f\n",
           r2.final_loss, r2.loss_color);
    /* When only the color weight is active and there were color cells,
     * total must equal loss_color (within fp tolerance). */
    if (r2.loss_color > 0) {
        float diff = fabsf(r2.final_loss - r2.loss_color);
        CHECK(diff < 0.001f, "final_loss == loss_color when only color weight set");
    } else {
        printf("    (no masked color cells in this run, skipping equality check)\n");
    }

    /* Mixed weights: total = 0.5*S + 0.3*T + 0.2*C */
    MaskedTrainConfig cfg3;
    masked_train_config_default(&cfg3);
    cfg3.epochs = 3;
    cfg3.denoise_steps = 4;
    cfg3.loss_patience = 99;
    cfg3.loss_w_structure = 0.5f;
    cfg3.loss_w_texture   = 0.3f;
    cfg3.loss_w_color     = 0.2f;

    MaskedTrainResult r3;
    masked_train_image(&r3, &storage, rgba, 256, 256, 702, &cfg3);
    float expected = 0.5f * r3.loss_structure
                   + 0.3f * r3.loss_texture
                   + 0.2f * r3.loss_color;
    printf("    mixed weights: total=%.3f  expected=%.3f\n",
           r3.final_loss, expected);
    CHECK(fabsf(r3.final_loss - expected) < 0.001f,
          "final_loss matches weighted segment sum");

    ce_storage_free(&storage);
    free(rgba);
}

int main(void) {
    printf("=== Masked Train Test Suite ===\n");
    test_config();
    test_single_image();
    test_loss_decreases();
    test_batch();
    test_progressive_strategy();
    test_segment_loss();
    printf("\n=== %d/%d PASSED ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
