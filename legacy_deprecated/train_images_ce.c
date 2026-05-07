/* train_images_ce.c — co-train SpatialAI + CEStorage from a list of PPMs.
 *
 * For each input PPM the tool:
 *   1. image_to_grid(path)              → SpatialGrid (256x256 R/G/B planes)
 *   2. ai_store_grid(ai, ...)           → SpatialAI keyframe (Mat-1..4)
 *   3. ce_storage_ingest_rgba_16        → 16x16 atomic blocks (default)
 *      OR ce_storage_ingest_rgba        → 8x8 blocks (--legacy-8)
 *      (RGBA buffer reconstructed from the SpatialGrid, A=255)
 *
 * The default 16x16 path matches train_demo so models trained by either
 * tool can be consumed by the same hybrid_vae_decode pipeline. The 8x8
 * path is preserved behind --legacy-8 so existing .ces files keyed to
 * the old block layout can still be regenerated bit-identical.
 *
 * On exit it writes both `<out>.spai` (SpatialAI model) and `<out>.ces`
 * (CEStorage), so downstream generation can use the typed retrieval
 * path (ce_search_by_type / ce_generate_image_typed) AND keep the
 * existing SpatialAI render path working without changes.
 *
 * Usage:
 *   train_images_ce <out_basename> [--legacy-8] <img1.ppm> [<img2.ppm> ...]
 */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

#include "ce_storage.h"
#include "ce_storage_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* basename_no_ext(const char* path, char* buf, size_t cap) {
    const char* slash = strrchr(path, '/');
    const char* name  = slash ? slash + 1 : path;
    size_t n = strlen(name);
    const char* dot = strrchr(name, '.');
    if (dot) n = (size_t)(dot - name);
    if (n >= cap) n = cap - 1;
    memcpy(buf, name, n);
    buf[n] = '\0';
    return buf;
}

/* FNV-1a — same constants as ce_path_hash so paths hash consistently
 * across the SpatialAI/CEStorage bridge. Local copy avoids pulling in
 * ce_ingest.c (which depends on stb_image). */
static uint32_t path_hash(const char* path) {
    uint32_t h = 0x811C9DC5u;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        h ^= *p;
        h *= 0x01000193u;
    }
    return h;
}

/* SpatialGrid stores R/G/B as separate planes (uint8) and A as luminance
 * (uint16, but kept in 0..255). Reassemble into an interleaved RGBA
 * buffer for ce_storage_ingest_rgba. Caller frees the buffer. */
static uint8_t* grid_to_rgba(const SpatialGrid* g, int* out_w, int* out_h) {
    if (!g) return NULL;
    int w = (int)GRID_SIZE;
    int h = (int)GRID_SIZE;
    uint8_t* rgba = (uint8_t*)malloc((size_t)w * (size_t)h * 4u);
    if (!rgba) return NULL;
    for (int i = 0; i < w * h; ++i) {
        rgba[i * 4 + 0] = g->R[i];
        rgba[i * 4 + 1] = g->G[i];
        rgba[i * 4 + 2] = g->B[i];
        rgba[i * 4 + 3] = 255;
    }
    *out_w = w;
    *out_h = h;
    return rgba;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <out_basename> [--legacy-8] <img1.ppm> [<img2.ppm> ...]\n"
                "  --legacy-8   use the 8x8 block ingest path (default: 16x16)\n",
                argv[0]);
        return 2;
    }
    const char* out_base = argv[1];

    /* Default to 16x16 atomic blocks so .ces files match train_demo's
     * hybrid_vae layout. --legacy-8 reverts to the original 8x8 path. */
    int legacy_8 = 0;
    int first_img = 2;
    if (argc >= 3 && strcmp(argv[2], "--legacy-8") == 0) {
        legacy_8 = 1;
        first_img = 3;
    }
    if (first_img >= argc) {
        fprintf(stderr, "[train_images_ce] no input images\n");
        return 2;
    }

    SpatialAI* ai = spatial_ai_create();
    if (!ai) {
        fprintf(stderr, "[train_images_ce] spatial_ai_create failed\n");
        return 1;
    }

    CEStorage ces;
    ce_storage_init(&ces, 256);

    int trained_spai = 0;
    int trained_ces  = 0;
    uint32_t total_blocks = 0;

    for (int i = first_img; i < argc; i++) {
        const char* in = argv[i];
        SpatialGrid* g = image_to_grid(in);
        if (!g) {
            fprintf(stderr, "[train_images_ce] skip %s (load failed)\n", in);
            continue;
        }
        char label[64];
        basename_no_ext(in, label, sizeof label);

        /* SpatialAI side: existing keyframe ingestion (Mat-1..4 etc.). */
        uint32_t kf_id = ai_store_grid(ai, g, label);
        if (kf_id != UINT32_MAX) {
            ++trained_spai;
            fprintf(stderr, "[train_images_ce] +KF  id=%u  label=%s\n", kf_id, label);
        } else {
            fprintf(stderr, "[train_images_ce] !! ai_store_grid failed for %s\n", in);
        }

        /* CEStorage side: 16x16 atomic blocks by default (matches
         * train_demo / hybrid_vae_decode), or 8x8 under --legacy-8. */
        int w = 0, h = 0;
        uint8_t* rgba = grid_to_rgba(g, &w, &h);
        if (rgba) {
            uint32_t cid = path_hash(in);
            uint32_t added = legacy_8
                ? ce_storage_ingest_rgba   (&ces, cid, rgba, w, h)
                : ce_storage_ingest_rgba_16(&ces, cid, rgba, w, h);
            free(rgba);
            if (added > 0) {
                ++trained_ces;
                total_blocks += added;
                fprintf(stderr, "[train_images_ce] +CES blocks=%u (%s)  src=%s\n",
                        added, legacy_8 ? "8x8" : "16x16", in);
            }
        }

        grid_destroy(g);
    }

    if (trained_spai == 0 && trained_ces == 0) {
        fprintf(stderr, "[train_images_ce] no images trained, aborting\n");
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }

    /* Save .spai */
    char spai_path[1024];
    snprintf(spai_path, sizeof(spai_path), "%s.spai", out_base);
    SpaiStatus st = ai_save(ai, spai_path);
    if (st != SPAI_OK) {
        fprintf(stderr, "[train_images_ce] ai_save failed (status=%d)\n", (int)st);
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_images_ce] saved %s (%d images)\n",
            spai_path, trained_spai);

    /* Save .ces */
    char ces_path[1024];
    snprintf(ces_path, sizeof(ces_path), "%s.ces", out_base);
    if (!ce_storage_save(&ces, ces_path)) {
        fprintf(stderr, "[train_images_ce] ce_storage_save failed\n");
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_images_ce] saved %s (%u blocks across %d images)\n",
            ces_path, total_blocks, trained_ces);

    ce_storage_free(&ces);
    spatial_ai_destroy(ai);
    return 0;
}
