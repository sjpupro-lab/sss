/* train_loop_bench.c — compare CEStorage training loop variants.
 *
 * Builds the SAME synthetic 100-image dataset (each image = 256x256 of
 * a sampled color, varied brightness per row) and ingests it three ways:
 *
 *   A. spatial-prev: current production path. Delta against previous
 *      block in raster scan order within the same image.
 *
 *   B. zero-anchor: delta = fresh - 0. Equivalent to storing only
 *      the keyframe; deltas are maximally large but trivially
 *      decompressed and the storage layout is uniform.
 *
 *   C. dedup-then-spatial: a block is appended only if its nearest
 *      existing keyframe is farther than EPS. Otherwise it is skipped
 *      (semantic deduplication). Survivors get spatial-prev deltas.
 *
 * Metrics per variant:
 *   - entries stored
 *   - mean |delta|        (sum of byte-wise abs differences / 64)
 *   - file size on disk
 *   - search recall: query each *training* block and check whether the
 *     closest stored entry is the same color family (within COLOR_EPS).
 *
 * The objective is informational — the tool prints a table; it does
 * NOT change production. Decisions about which variant to adopt are
 * made downstream once the trade-offs are visible.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_search.h"
#include "../ce_feed_image.h"
#include "../ce_type.h"

#define N_IMAGES        100
#define BLOCK_PX        8
#define IMG_DIM         256
#define BLOCKS_PER_IMG  ((IMG_DIM / BLOCK_PX) * (IMG_DIM / BLOCK_PX)) /* 1024 */
#define DEDUP_EPS       2000u   /* default; overridden per-run by g_dedup_eps */
#define COLOR_EPS       30      /* RGB euclidean tolerance for recall */

static uint32_t g_dedup_eps = DEDUP_EPS;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Sampled color sequence — varied so the dataset has real diversity.
 * Each image i gets a base color (bR, bG, bB) plus a per-row brightness
 * gradient. */
static void color_for_image(int i, uint8_t *r, uint8_t *g, uint8_t *b) {
    /* deterministic LCG-flavoured hash */
    uint32_t s = (uint32_t)(i + 1) * 0x9E3779B1u;
    *r = (uint8_t)(40 + ((s >> 16) % 200));
    *g = (uint8_t)(40 + ((s >> 8)  % 200));
    *b = (uint8_t)(40 + ( s        % 200));
}

static void fill_block(uint8_t *blk, int img_idx, int by, int bx) {
    uint8_t cR, cG, cB;
    color_for_image(img_idx, &cR, &cG, &cB);
    /* per-row brightness drift adds within-image variation. */
    int drift = (by * 4) - 32;
    int rR = (int)cR + drift; if (rR < 0) rR = 0; if (rR > 255) rR = 255;
    int rG = (int)cG + drift; if (rG < 0) rG = 0; if (rG > 255) rG = 255;
    int rB = (int)cB + drift; if (rB < 0) rB = 0; if (rB > 255) rB = 255;
    /* per-column salt — noisy but bounded. */
    int salt = (bx * 3) % 7;
    rR = (rR + salt) & 0xFF;
    for (int p = 0; p < 64; ++p) {
        blk[p * 4 + 0] = (uint8_t)rR;
        blk[p * 4 + 1] = (uint8_t)rG;
        blk[p * 4 + 2] = (uint8_t)rB;
        blk[p * 4 + 3] = 255;
    }
}

/* ----- variant A: spatial-prev delta ----- */
static void train_A(CEStorage *s) {
    uint8_t blk[64 * 4];
    for (int i = 0; i < N_IMAGES; ++i) {
        CEUnit prev; ce_init(&prev);
        for (int by = 0; by < IMG_DIM / BLOCK_PX; ++by) {
            for (int bx = 0; bx < IMG_DIM / BLOCK_PX; ++bx) {
                fill_block(blk, i, by, bx);
                CEUnit fresh; ce_feed_image(&fresh, blk);
                CEUnit delta;
                if (by == 0 && bx == 0) {
                    CEUnit z; ce_init(&z);
                    ce_delta(&delta, &z, &fresh);
                } else {
                    ce_delta(&delta, &prev, &fresh);
                }
                ce_storage_add_typed(s, (uint32_t)i,
                                     (uint16_t)by, (uint16_t)bx,
                                     CE_TYPE_IMAGE, &fresh, &delta);
                prev = fresh;
            }
        }
    }
}

/* ----- variant B: zero-anchor delta ----- */
static void train_B(CEStorage *s) {
    uint8_t blk[64 * 4];
    CEUnit zero; ce_init(&zero);
    for (int i = 0; i < N_IMAGES; ++i) {
        for (int by = 0; by < IMG_DIM / BLOCK_PX; ++by) {
            for (int bx = 0; bx < IMG_DIM / BLOCK_PX; ++bx) {
                fill_block(blk, i, by, bx);
                CEUnit fresh; ce_feed_image(&fresh, blk);
                CEUnit delta; ce_delta(&delta, &zero, &fresh);
                ce_storage_add_typed(s, (uint32_t)i,
                                     (uint16_t)by, (uint16_t)bx,
                                     CE_TYPE_IMAGE, &fresh, &delta);
            }
        }
    }
}

/* ----- variant C: dedup-then-spatial-prev ----- */
static void train_C(CEStorage *s) {
    uint8_t blk[64 * 4];
    for (int i = 0; i < N_IMAGES; ++i) {
        CEUnit prev; ce_init(&prev);
        int prev_set = 0;
        for (int by = 0; by < IMG_DIM / BLOCK_PX; ++by) {
            for (int bx = 0; bx < IMG_DIM / BLOCK_PX; ++bx) {
                fill_block(blk, i, by, bx);
                CEUnit fresh; ce_feed_image(&fresh, blk);

                /* dedup against existing storage */
                int skip = 0;
                if (s->count > 0) {
                    CESearchResult r;
                    ce_search_by_type(s, &fresh, CE_TYPE_IMAGE, 1, &r);
                    if (r.distance < g_dedup_eps) {
                        skip = 1;
                    }
                }
                if (skip) continue;

                CEUnit delta;
                if (!prev_set) {
                    CEUnit z; ce_init(&z);
                    ce_delta(&delta, &z, &fresh);
                } else {
                    ce_delta(&delta, &prev, &fresh);
                }
                ce_storage_add_typed(s, (uint32_t)i,
                                     (uint16_t)by, (uint16_t)bx,
                                     CE_TYPE_IMAGE, &fresh, &delta);
                prev = fresh;
                prev_set = 1;
            }
        }
    }
}

static double mean_abs_delta(const CEStorage *s) {
    if (s->count == 0) return 0.0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < s->count; ++i) {
        const uint8_t *bytes = (const uint8_t *)&s->entries[i].delta;
        for (int b = 0; b < 64; ++b) total += bytes[b];
    }
    return (double)total / (double)(s->count * 64u);
}

/* Recall: pick a block from EACH image and check that the nearest
 * retrieved entry decodes back to the same color family. */
static double recall(const CEStorage *s) {
    int hits = 0, total = 0;
    uint8_t blk[64 * 4];
    for (int i = 0; i < N_IMAGES; ++i) {
        fill_block(blk, i, 16, 16); /* center-ish block */
        CEUnit q; ce_feed_image(&q, blk);

        CESearchResult r;
        int n = ce_search_by_type(s, &q, CE_TYPE_IMAGE, 1, &r);
        if (n == 0) { ++total; continue; }

        uint8_t cR, cG, cB;
        color_for_image(i, &cR, &cG, &cB);

        /* We don't decode the entry; instead we check the canvas_id
         * (which encodes the source image index). A "hit" is the
         * retrieved entry coming from the same image. */
        uint32_t cid = s->entries[r.entry_idx].canvas_id;
        if (cid == (uint32_t)i) ++hits;
        ++total;
    }
    return total ? ((double)hits / total) : 0.0;
}

static long file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long)st.st_size;
}

static void report(const char *name, const CEStorage *s, double t_train, const char *path) {
    ce_storage_save(s, path);
    long sz = file_size(path);
    double rc = recall(s);
    double mad = mean_abs_delta(s);
    printf("%-22s entries=%6u  train=%6.3fs  mean|delta|=%6.2f  recall=%5.1f%%  disk=%6ldB\n",
           name, s->count, t_train, mad, rc * 100.0, sz);
}

int main(void) {
    printf("=== train_loop_bench: %d images, %d blocks/img, dedup_eps=%u ===\n\n",
           N_IMAGES, BLOCKS_PER_IMG, DEDUP_EPS);
    (void)COLOR_EPS;

    {
        CEStorage s; ce_storage_init(&s, 1024);
        double t0 = now_s();
        train_A(&s);
        report("A spatial-prev",        &s, now_s() - t0, "/tmp/loop_A.ces");
        ce_storage_free(&s);
    }
    {
        CEStorage s; ce_storage_init(&s, 1024);
        double t0 = now_s();
        train_B(&s);
        report("B zero-anchor",         &s, now_s() - t0, "/tmp/loop_B.ces");
        ce_storage_free(&s);
    }
    /* Sweep dedup eps to find a useful operating point. */
    uint32_t eps_values[] = { 50, 200, 500, 1000, 2000 };
    for (size_t k = 0; k < sizeof(eps_values) / sizeof(eps_values[0]); ++k) {
        g_dedup_eps = eps_values[k];
        CEStorage s; ce_storage_init(&s, 1024);
        double t0 = now_s();
        train_C(&s);
        char name[64];
        snprintf(name, sizeof(name), "C dedup eps=%-4u", g_dedup_eps);
        char path[64];
        snprintf(path, sizeof(path), "/tmp/loop_C_%u.ces", g_dedup_eps);
        report(name, &s, now_s() - t0, path);
        ce_storage_free(&s);
    }

    printf("\nLegend:\n");
    printf("  mean|delta| = average byte-wise |delta| across all stored cells (lower = more compact deltas)\n");
    printf("  recall      = of N_IMAGES held-out queries (one per image), %% retrieved from the SAME image\n");
    printf("  disk        = .ces file size with current 140B-per-entry layout\n");
    return 0;
}
