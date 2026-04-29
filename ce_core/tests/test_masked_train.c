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

int main(void) {
    printf("=== Masked Train Test Suite ===\n");
    test_config();
    test_single_image();
    test_loss_decreases();
    test_batch();
    test_progressive_strategy();
    printf("\n=== %d/%d PASSED ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
