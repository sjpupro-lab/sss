/* log_train.c — verbose masked-train runner.
 *
 * Wraps masked_train_image with an on_epoch callback that prints
 * per-epoch loss, plus a final summary including the structure /
 * texture / color segment breakdown. This is the harness used to
 * collect the training-log column when reporting Step-5 results;
 * the production train_demo only emits the final +MSK line per
 * image, which isn't enough to plot a loss curve.
 *
 * Usage:
 *   log_train <img.ppm> <canvas_id> [options]
 *
 * Options forward straight to MaskedTrainConfig (same names as
 * train_demo). One image at a time keeps the log readable.
 */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "ce_storage.h"
#include "ce_masked_train.h"
#include "ce_residual_codebook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_epoch(int epoch, float loss, void *ctx) {
    (void)ctx;
    fprintf(stdout, "  epoch %2d  loss=%.4f\n", epoch + 1, loss);
    fflush(stdout);
}

static int parse_strategy(const char *s, MaskStrategy *out) {
    if (strcmp(s, "correction"  ) == 0) { *out = MASK_CORRECTION_ONLY; return 1; }
    if (strcmp(s, "residual"    ) == 0) { *out = MASK_RESIDUAL_UP;     return 1; }
    if (strcmp(s, "random"      ) == 0) { *out = MASK_RANDOM_HALF;     return 1; }
    if (strcmp(s, "progressive" ) == 0) { *out = MASK_PROGRESSIVE;     return 1; }
    return 0;
}

static uint8_t *grid_to_rgba(const SpatialGrid *g, int *w, int *h) {
    int W = (int)GRID_SIZE, H = (int)GRID_SIZE;
    uint8_t *rgba = (uint8_t *)malloc((size_t)W * H * 4u);
    if (!rgba) return NULL;
    for (int i = 0; i < W * H; ++i) {
        rgba[i * 4 + 0] = g->R[i];
        rgba[i * 4 + 1] = g->G[i];
        rgba[i * 4 + 2] = g->B[i];
        rgba[i * 4 + 3] = 255;
    }
    *w = W; *h = H;
    return rgba;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <img.ppm> <canvas_id> [options]\n"
                "  --masked-epochs N\n"
                "  --target-loss F\n"
                "  --loss-patience N\n"
                "  --denoise-steps N\n"
                "  --mask-strategy correction|residual|random|progressive\n"
                "  --mask-seed U64\n"
                "  --loss-w-struct F  --loss-w-texture F  --loss-w-color F\n",
                argv[0]);
        return 2;
    }

    const char *img_path = argv[1];
    uint32_t canvas_id   = (uint32_t)strtoul(argv[2], NULL, 0);

    MaskedTrainConfig cfg;
    masked_train_config_default(&cfg);
    cfg.epochs        = 10;
    cfg.denoise_steps = 8;
    cfg.on_epoch      = on_epoch;

    for (int i = 3; i < argc; ++i) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "[log_train] missing value for %s\n", k); return 2; }
        if      (strcmp(k, "--masked-epochs") == 0) { cfg.epochs        = atoi(v); ++i; }
        else if (strcmp(k, "--target-loss"  ) == 0) { cfg.target_loss   = strtof(v, NULL); ++i; }
        else if (strcmp(k, "--loss-patience") == 0) { cfg.loss_patience = (float)atoi(v); ++i; }
        else if (strcmp(k, "--denoise-steps") == 0) { cfg.denoise_steps = atoi(v); ++i; }
        else if (strcmp(k, "--mask-strategy") == 0) {
            if (!parse_strategy(v, &cfg.strategy)) {
                fprintf(stderr, "[log_train] unknown strategy: %s\n", v); return 2;
            }
            ++i;
        }
        else if (strcmp(k, "--mask-seed"     ) == 0) { cfg.mask_seed         = strtoull(v, NULL, 0); ++i; }
        else if (strcmp(k, "--loss-w-struct" ) == 0) { cfg.loss_w_structure  = strtof(v, NULL); ++i; }
        else if (strcmp(k, "--loss-w-texture") == 0) { cfg.loss_w_texture    = strtof(v, NULL); ++i; }
        else if (strcmp(k, "--loss-w-color"  ) == 0) { cfg.loss_w_color      = strtof(v, NULL); ++i; }
        else { fprintf(stderr, "[log_train] unknown option: %s\n", k); return 2; }
    }

    SpatialGrid *g = image_to_grid(img_path);
    if (!g) {
        fprintf(stderr, "[log_train] image_to_grid failed for %s\n", img_path);
        return 1;
    }

    int w = 0, h = 0;
    uint8_t *rgba = grid_to_rgba(g, &w, &h);
    if (!rgba) { grid_destroy(g); return 1; }

    CEStorage storage;
    ce_storage_init(&storage, 256);

    fprintf(stdout, "[log_train] %s  cid=0x%08x  epochs=%d patience=%d denoise=%d\n",
            img_path, canvas_id, cfg.epochs, (int)cfg.loss_patience, cfg.denoise_steps);
    fprintf(stdout, "[log_train] strategy=%d seed=%llu  weights: struct=%.2f tex=%.2f color=%.2f\n",
            (int)cfg.strategy, (unsigned long long)cfg.mask_seed,
            cfg.loss_w_structure, cfg.loss_w_texture, cfg.loss_w_color);
    fflush(stdout);

    MaskedTrainResult r;
    masked_train_image(&r, &storage, rgba, w, h, canvas_id, &cfg);

    fprintf(stdout, "\n[log_train] === final ===\n");
    fprintf(stdout, "  epochs_run    = %d\n",  r.epochs_run);
    fprintf(stdout, "  final_loss    = %.4f\n", r.final_loss);
    fprintf(stdout, "  loss_struct   = %.4f\n", r.loss_structure);
    fprintf(stdout, "  loss_texture  = %.4f\n", r.loss_texture);
    fprintf(stdout, "  loss_color    = %.4f\n", r.loss_color);
    fprintf(stdout, "  converged     = %d\n",  r.converged);
    fprintf(stdout, "  cells_stored  = %u\n",  r.cells_stored);

    free(rgba);
    grid_destroy(g);
    ce_storage_free(&storage);
    return 0;
}
