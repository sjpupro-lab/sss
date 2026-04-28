/* train_demo.c — joint text+image trainer.
 *
 * Reads a TSV produced by make_demo_dataset:
 *   <label>\t<clause>\t<image_path_relative_to_dataset_dir>
 *
 * For each row:
 *   1. image_to_grid(<dataset>/<image>) -> SpatialGrid
 *   2. ai_store_auto_with_image(ai, clause, grid, label)
 *        — encodes the clause text, attaches the grid as a keyframe
 *          image, and threads in Mat-1..4 material harmonics.
 *   3. ce_storage_ingest_rgba(ces, hash(image_path), rgba, 256, 256)
 *        — appends 32x32=1024 CE_TYPE_IMAGE entries per image to the
 *          CEStorage so ce_search_by_type / ce_generate_image_typed
 *          can retrieve them.
 *
 * Outputs:
 *   <out_base>.spai  (SpatialAI model: text + image keyframes)
 *   <out_base>.ces   (CEStorage:      8x8 image blocks)
 *
 * Usage:
 *   train_demo <dataset_dir> <out_base>
 *     dataset_dir must contain labels.tsv (paths inside it are resolved
 *                 relative to dataset_dir).
 */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_layers.h"
#include "spatial_io.h"

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811C9DC5u;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= *p;
        h *= 0x01000193u;
    }
    return h;
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

/* strip leading/trailing whitespace in-place */
static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') ++s;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) --e;
    *e = '\0';
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <dataset_dir> <out_base>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    const char *out = argv[2];

    char tsv_path[1024];
    snprintf(tsv_path, sizeof(tsv_path), "%s/labels.tsv", dir);
    FILE *tsv = fopen(tsv_path, "r");
    if (!tsv) { perror(tsv_path); return 1; }

    SpatialAI *ai = spatial_ai_create();
    if (!ai) { fclose(tsv); return 1; }

    CEStorage ces;
    ce_storage_init(&ces, 256);

    int rows = 0, ai_ok = 0;
    uint32_t total_blocks = 0;

    char line[1024];
    while (fgets(line, sizeof(line), tsv)) {
        char *p = trim(line);
        if (!*p || *p == '#') continue;

        char *t1 = strchr(p, '\t');
        if (!t1) continue;
        *t1 = '\0';
        char *label = p;
        char *t2 = strchr(t1 + 1, '\t');
        if (!t2) continue;
        *t2 = '\0';
        char *clause = t1 + 1;
        char *img_rel = t2 + 1;

        char img_path[1024];
        snprintf(img_path, sizeof(img_path), "%s/%s", dir, img_rel);

        SpatialGrid *g = image_to_grid(img_path);
        if (!g) {
            fprintf(stderr, "[train_demo] skip %s (image_to_grid failed)\n", img_path);
            continue;
        }

        ++rows;

        /* 1. SpatialAI joint text+image store. */
        uint32_t kf = ai_store_auto_with_image(ai, clause, g, label);
        if (kf != UINT32_MAX) {
            ++ai_ok;
            fprintf(stderr, "[train_demo] +KF id=%u  label=%-10s  clause=\"%s\"\n",
                    kf, label, clause);
        } else {
            fprintf(stderr, "[train_demo] !! ai_store_auto_with_image failed for %s\n",
                    label);
        }

        /* 2. CEStorage image-block ingest. */
        uint32_t cid = fnv1a(img_path);
        int w = 0, h = 0;
        uint8_t *rgba = grid_to_rgba(g, &w, &h);
        if (rgba) {
            uint32_t added = ce_storage_ingest_rgba(&ces, cid, rgba, w, h);
            total_blocks += added;
            free(rgba);
            fprintf(stderr, "[train_demo]  +CES blocks=%u  (cumulative=%u)\n",
                    added, total_blocks);
        }

        /* 3. Label / clause bridge — proper SSS text encoding.
         *
         * The byte-level ce_feed used previously put text CEUnits in a
         * different cell space than image CEUnits (FNV-style channel
         * mixing on raw bytes vs per-channel sums on RGBA pixels), so
         * a prompt CEUnit was never directly comparable to an image
         * CEUnit by ce_distance.
         *
         * Correct path: encode the clause through layers_encode_clause
         * into a 256x256 SpatialGrid (X=byte, Y=position, A=3-layer
         * sum), then ingest that grid into CEStorage exactly like an
         * image — 8x8 blocks, ce_feed_image, CE_TYPE_TEXT tag, same
         * canvas_id as the image entries. After this, text and image
         * cells live in the same CE space and ce_distance is
         * meaningful across the two modalities. */
        SpatialGrid *text_grid = grid_create();
        if (text_grid) {
            layers_encode_clause(clause, NULL, text_grid);
            int tw = 0, th = 0;
            uint8_t *trgba = grid_to_rgba(text_grid, &tw, &th);
            if (trgba) {
                /* Use a slightly different canvas_id so the TEXT
                 * entries don't collide with IMAGE entries on
                 * (canvas_id, slot, block_idx) lookups. The router
                 * normalises this by bit-clearing the high bit. */
                uint32_t text_cid = cid ^ 0x80000000u;
                uint32_t added_t = ce_storage_ingest_rgba(
                    &ces, text_cid, trgba, tw, th);
                /* Re-tag the just-inserted TEXT-grid entries from
                 * IMAGE to TEXT so ce_search_by_type(CE_TYPE_TEXT)
                 * picks them up. */
                for (uint32_t i = ces.count - added_t; i < ces.count; ++i) {
                    ces.entries[i].type = CE_TYPE_TEXT;
                }
                fprintf(stderr, "[train_demo]  +TXT blocks=%u  cid=0x%08x\n",
                        added_t, text_cid);
                free(trgba);
            }
            grid_destroy(text_grid);
        }

        grid_destroy(g);
    }
    fclose(tsv);

    if (rows == 0) {
        fprintf(stderr, "[train_demo] no valid rows in %s\n", tsv_path);
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }

    char spai_path[1024];
    snprintf(spai_path, sizeof(spai_path), "%s.spai", out);
    SpaiStatus st = ai_save(ai, spai_path);
    if (st != SPAI_OK) {
        fprintf(stderr, "[train_demo] ai_save failed (status=%d)\n", (int)st);
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_demo] saved %s (%d/%d rows)\n",
            spai_path, ai_ok, rows);

    char ces_path[1024];
    snprintf(ces_path, sizeof(ces_path), "%s.ces", out);
    if (!ce_storage_save(&ces, ces_path)) {
        fprintf(stderr, "[train_demo] ce_storage_save failed\n");
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_demo] saved %s (%u blocks across %d images)\n",
            ces_path, total_blocks, rows);

    ce_storage_free(&ces);
    spatial_ai_destroy(ai);
    return 0;
}
