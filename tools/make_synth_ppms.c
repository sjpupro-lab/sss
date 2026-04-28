/* make_synth_ppms.c — emit a small set of synthetic 256x256 PPM P6 images
 * for end-to-end pipeline testing. Each image is a solid-color tile so
 * generation can be verified by mean-channel statistics, not by visual
 * inspection.
 *
 * Usage: make_synth_ppms <out_dir>
 *
 * Output:
 *   <out_dir>/red.ppm     (R=200, G=30,  B=30)
 *   <out_dir>/green.ppm   (R=30,  G=200, B=30)
 *   <out_dir>/blue.ppm    (R=30,  G=30,  B=200)
 *   <out_dir>/yellow.ppm  (R=210, G=200, B=40)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DIM 256

static int write_solid(const char *path, unsigned char r, unsigned char g, unsigned char b) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", DIM, DIM);
    unsigned char rgb[3] = { r, g, b };
    for (int i = 0; i < DIM * DIM; ++i) {
        if (fwrite(rgb, 1, 3, f) != 3) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    mkdir(dir, 0755); /* ignore failure if it exists */

    char path[1024];
    struct { const char *name; unsigned char r, g, b; } imgs[] = {
        { "red",    200, 30,  30  },
        { "green",  30,  200, 30  },
        { "blue",   30,  30,  200 },
        { "yellow", 210, 200, 40  },
    };
    int n = (int)(sizeof(imgs) / sizeof(imgs[0]));
    for (int i = 0; i < n; ++i) {
        snprintf(path, sizeof(path), "%s/%s.ppm", dir, imgs[i].name);
        if (!write_solid(path, imgs[i].r, imgs[i].g, imgs[i].b)) {
            fprintf(stderr, "failed to write %s\n", path);
            return 1;
        }
        fprintf(stderr, "wrote %s\n", path);
    }
    return 0;
}
