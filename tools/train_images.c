/* Train SpatialAI from a list of PPM P6 256x256 images.
 *
 * For each input image:
 *   image_to_grid(path) -> ai_store_grid(ai, grid, basename)
 * ai_store_grid commits the grid as a keyframe AND updates the EMA
 * store, so the model gains both image-level retrieval and per-cell
 * color statistics. Then ai_save the resulting model.
 *
 * Usage: train_images <out.spai> <img1.ppm> [<img2.ppm> ...] */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <out.spai> <img1.ppm> [<img2.ppm> ...]\n", argv[0]);
        return 2;
    }
    const char* out_path = argv[1];

    SpatialAI* ai = spatial_ai_create();
    if (!ai) { fprintf(stderr, "[train_images] spatial_ai_create failed\n"); return 1; }

    int trained = 0;
    for (int i = 2; i < argc; i++) {
        const char* in = argv[i];
        SpatialGrid* g = image_to_grid(in);
        if (!g) {
            fprintf(stderr, "[train_images] skip %s (load failed)\n", in);
            continue;
        }
        char label[64];
        basename_no_ext(in, label, sizeof label);
        uint32_t kf_id = ai_store_grid(ai, g, label);
        grid_destroy(g);
        if (kf_id == UINT32_MAX) {
            fprintf(stderr, "[train_images] skip %s (store failed)\n", in);
            continue;
        }
        fprintf(stderr, "[train_images] +KF  id=%u  label=%s  src=%s\n", kf_id, label, in);
        trained++;
    }

    if (trained == 0) {
        fprintf(stderr, "[train_images] no images trained, aborting\n");
        spatial_ai_destroy(ai);
        return 1;
    }

    SpaiStatus st = ai_save(ai, out_path);
    if (st != SPAI_OK) {
        fprintf(stderr, "[train_images] ai_save failed (status=%d)\n", (int)st);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_images] saved %s (%d images trained)\n", out_path, trained);
    spatial_ai_destroy(ai);
    return 0;
}
