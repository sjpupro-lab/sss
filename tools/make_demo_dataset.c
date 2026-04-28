/* make_demo_dataset.c — synthesize the SSS demo training set.
 *
 * Produces, in <out_dir>/:
 *
 *   colors/
 *     red.ppm      green.ppm      blue.ppm      yellow.ppm
 *
 *   fruits/
 *     apple.ppm     banana.ppm     grape.ppm
 *     lime.ppm      blueberry.ppm  orange.ppm
 *
 *   labels.tsv  <label>\t<clause>\t<relative_image_path>  (one line per image)
 *
 * Each PPM is 256x256 P6. Color images are uniform fills. Fruit images
 * draw a simple ellipse on a near-white background so the dataset has
 * some spatial structure (not just a single colour) for the CE block
 * encoder to capture meaningful gradients.
 *
 * Usage: make_demo_dataset <out_dir>
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define DIM 256

typedef struct {
    unsigned char r, g, b;
} RGB;

static int write_ppm(const char *path, const unsigned char *rgb) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", DIM, DIM);
    if (fwrite(rgb, 1, (size_t)DIM * DIM * 3, f) != (size_t)DIM * DIM * 3) {
        fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

static void fill_solid(unsigned char *rgb, RGB c) {
    for (int i = 0; i < DIM * DIM; ++i) {
        rgb[i * 3 + 0] = c.r;
        rgb[i * 3 + 1] = c.g;
        rgb[i * 3 + 2] = c.b;
    }
}

/* Draw a filled ellipse with center (cx,cy), radii (rx,ry), color `c`
 * over an existing buffer. */
static void draw_ellipse(unsigned char *rgb,
                         int cx, int cy, int rx, int ry, RGB c) {
    for (int y = 0; y < DIM; ++y) {
        for (int x = 0; x < DIM; ++x) {
            float nx = (float)(x - cx) / (float)rx;
            float ny = (float)(y - cy) / (float)ry;
            if (nx * nx + ny * ny <= 1.0f) {
                rgb[(y * DIM + x) * 3 + 0] = c.r;
                rgb[(y * DIM + x) * 3 + 1] = c.g;
                rgb[(y * DIM + x) * 3 + 2] = c.b;
            }
        }
    }
}

/* Add a small darker shadow underneath the fruit so the encoder sees
 * a real luminance gradient instead of just a flat ellipse. */
static void darken_lower_half(unsigned char *rgb,
                              int cx, int cy, int rx, int ry, float scale) {
    for (int y = cy; y < DIM; ++y) {
        float dy = (float)(y - cy) / (float)ry;
        if (dy > 1.0f) continue;
        for (int x = 0; x < DIM; ++x) {
            float dx = (float)(x - cx) / (float)rx;
            if (dx * dx + dy * dy > 1.0f) continue;
            int ax = (y * DIM + x) * 3;
            float t = scale + (1.0f - scale) * (1.0f - dy);
            rgb[ax + 0] = (unsigned char)((float)rgb[ax + 0] * t);
            rgb[ax + 1] = (unsigned char)((float)rgb[ax + 1] * t);
            rgb[ax + 2] = (unsigned char)((float)rgb[ax + 2] * t);
        }
    }
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0755) == 0) return 1;
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 2;
    }
    const char *out = argv[1];
    char dir[512];

    if (!ensure_dir(out)) { perror("out_dir"); return 1; }
    snprintf(dir, sizeof(dir), "%s/colors", out);
    if (!ensure_dir(dir)) { perror("colors"); return 1; }
    snprintf(dir, sizeof(dir), "%s/fruits", out);
    if (!ensure_dir(dir)) { perror("fruits"); return 1; }

    unsigned char *buf = (unsigned char *)malloc((size_t)DIM * DIM * 3);
    if (!buf) return 1;

    /* ---- colours ---- */
    struct { const char *name; RGB c; const char *clause; } colors[] = {
        { "red",    { 200,  30,  30 }, "the colour red"    },
        { "green",  {  30, 200,  30 }, "the colour green"  },
        { "blue",   {  30,  30, 200 }, "the colour blue"   },
        { "yellow", { 220, 200,  40 }, "the colour yellow" },
    };
    /* ---- fruits ---- */
    struct {
        const char *name;
        RGB skin;
        RGB bg;
        int cx, cy, rx, ry;
        const char *clause;
    } fruits[] = {
        { "apple",     { 200,  30,  30 }, { 245, 245, 240 }, 128, 132,  88,  92, "apple is red and round"      },
        { "banana",    { 230, 200,  40 }, { 245, 245, 240 }, 128, 128, 110,  40, "banana is yellow and curved" },
        { "grape",     { 130,  40, 160 }, { 245, 245, 240 }, 128, 132,  60,  80, "grape is purple"             },
        { "lime",      { 140, 200,  30 }, { 245, 245, 240 }, 128, 128,  88,  88, "lime is green and round"     },
        { "blueberry", {  40,  40, 180 }, { 245, 245, 240 }, 128, 128,  72,  72, "blueberry is blue"           },
        { "orange",    { 230, 140,  30 }, { 245, 245, 240 }, 128, 132,  90,  90, "orange is orange and round"  },
    };

    char tsv_path[512];
    snprintf(tsv_path, sizeof(tsv_path), "%s/labels.tsv", out);
    FILE *tsv = fopen(tsv_path, "wb");
    if (!tsv) { perror(tsv_path); free(buf); return 1; }

    /* Write colours */
    for (size_t i = 0; i < sizeof(colors) / sizeof(colors[0]); ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/colors/%s.ppm", out, colors[i].name);
        fill_solid(buf, colors[i].c);
        if (!write_ppm(path, buf)) { perror(path); fclose(tsv); free(buf); return 1; }

        char rel[256];
        snprintf(rel, sizeof(rel), "colors/%s.ppm", colors[i].name);
        fprintf(tsv, "%s\t%s\t%s\n", colors[i].name, colors[i].clause, rel);
        fprintf(stderr, "wrote %s\n", path);
    }

    /* Write fruits */
    for (size_t i = 0; i < sizeof(fruits) / sizeof(fruits[0]); ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/fruits/%s.ppm", out, fruits[i].name);
        fill_solid(buf, fruits[i].bg);
        draw_ellipse(buf, fruits[i].cx, fruits[i].cy,
                     fruits[i].rx, fruits[i].ry, fruits[i].skin);
        darken_lower_half(buf, fruits[i].cx, fruits[i].cy,
                          fruits[i].rx, fruits[i].ry, 0.65f);
        if (!write_ppm(path, buf)) { perror(path); fclose(tsv); free(buf); return 1; }

        char rel[256];
        snprintf(rel, sizeof(rel), "fruits/%s.ppm", fruits[i].name);
        fprintf(tsv, "%s\t%s\t%s\n", fruits[i].name, fruits[i].clause, rel);
        fprintf(stderr, "wrote %s\n", path);
    }

    fclose(tsv);
    free(buf);
    fprintf(stderr, "wrote %s\n", tsv_path);
    return 0;
}
