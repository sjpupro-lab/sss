/* v4 Task F — img2grid
 *
 * Reads a PPM P6 image at <in.ppm>, maps it onto a SpatialGrid, and
 * writes the grid back as a PPM image at <out.ppm>. This is the
 * "load → dump" direction of the roundtrip; grid2img does the same
 * cycle starting from a .spai-stored grid.
 *
 * Usage: img2grid <in.ppm> <out.ppm>
 *
 * Prototype-scope: fixed resolution (GRID_SIZE × GRID_SIZE, currently
 * 256 × 256). Resize your input beforehand. */

#include "spatial_grid.h"
#include "spatial_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <in.ppm> <out.ppm>\n"
        "  input must be PPM P6, %ux%u, 8-bit RGB\n",
        prog, GRID_SIZE, GRID_SIZE);
}

int main(int argc, char** argv) {
    if (argc != 3) { usage(argv[0]); return 2; }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    SpatialGrid* g = image_to_grid(in_path);
    if (!g) {
        fprintf(stderr, "[img2grid] failed to load %s (PPM P6 %ux%u only)\n",
                in_path, GRID_SIZE, GRID_SIZE);
        return 1;
    }

    if (!grid_to_image(g, out_path)) {
        fprintf(stderr, "[img2grid] failed to write %s\n", out_path);
        grid_destroy(g);
        return 1;
    }

    fprintf(stderr, "[img2grid] %s -> grid (%ux%u) -> %s\n",
            in_path, GRID_SIZE, GRID_SIZE, out_path);
    grid_destroy(g);
    return 0;
}
