/* v4 Task F — grid2img
 *
 * Loads a .spai model, picks the grid of keyframe #<kf_index>, and
 * emits it as a PPM image. Lets the user eyeball whether the R/G/B
 * planes carry anything image-like for a keyframe that came from a
 * text clause (answer: not really) versus one populated via
 * image_to_grid (answer: yes, modulo color-space quirks).
 *
 * Usage: grid2img <model.spai> <kf_index> <out.ppm>
 */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <model.spai> <kf_index> <out.ppm>\n"
        "  emits keyframes[kf_index].grid as a %ux%u PPM image\n",
        prog, GRID_SIZE, GRID_SIZE);
}

int main(int argc, char** argv) {
    if (argc != 4) { usage(argv[0]); return 2; }
    const char* model_path = argv[1];
    uint32_t    kf_index   = (uint32_t)strtoul(argv[2], NULL, 10);
    const char* out_path   = argv[3];

    SpaiStatus st = SPAI_OK;
    SpatialAI* ai = ai_load(model_path, &st);
    if (!ai) {
        fprintf(stderr, "[grid2img] ai_load failed: %s\n", spai_status_str(st));
        return 1;
    }

    if (kf_index >= ai->kf_count) {
        fprintf(stderr, "[grid2img] kf_index %u out of range (have %u)\n",
                kf_index, ai->kf_count);
        spatial_ai_destroy(ai);
        return 1;
    }

    if (!grid_to_image(&ai->keyframes[kf_index].grid, out_path)) {
        fprintf(stderr, "[grid2img] failed to write %s\n", out_path);
        spatial_ai_destroy(ai);
        return 1;
    }

    fprintf(stderr, "[grid2img] %s kf=%u -> %s\n",
            model_path, kf_index, out_path);
    spatial_ai_destroy(ai);
    return 0;
}
