/* sss_animate — Atmos motion frame generator.
 *
 * Pipeline:
 *   1. Load a PPM (P6) source image.
 *   2. ce_scene_build_from_rgba: decompose into Atmos scene-objects
 *      (STATIC / FLOW / RIGID / ORGANIC / PARTICLE auto-tagged via
 *      ce_infer_motion_profile from the cluster signature).
 *   3. For each tick in [0, n_frames): ce_scene_tick_with_profiles
 *      then ce_scene_render → write PPM.
 *
 * Bytes everywhere. No JSON, no float (PPM I/O excepted). The output
 * directory is created if it does not exist.
 *
 *   ./sss_animate <in.ppm> <out_dir> [n_frames] [width] [height]
 *
 * Defaults: n_frames=60, width = source width, height = source height.
 * Frames land at out_dir/frame_NNNN.ppm.
 *
 * This is a production entry point for the Atmos engine (replacing the
 * 0-prod-callers status of ce_scene_build_from_rgba / tick_with_profiles
 * / scene_render that the integration audit flagged).
 */
#include "ce_scene_object.h"
#include "ce_scene_bridge.h"
#include "ce_move_profile.h"

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s INPUT.ppm OUT_DIR [N_FRAMES] [WIDTH] [HEIGHT]\n"
        "  N_FRAMES default 60   (positive int)\n"
        "  WIDTH    default = source width\n"
        "  HEIGHT   default = source height\n",
        argv0);
}

/* Strict positive-int parser shared with sss_gen. */
static int parse_pos_int_strict(const char *s, int *out, int max_val) {
    if (!s || !*s) return -1;
    errno = 0;
    char *endp = NULL;
    long v = strtol(s, &endp, 10);
    if (errno != 0 || !endp || *endp != '\0' || v <= 0 || v > max_val) return -1;
    *out = (int)v;
    return 0;
}

/* Read PPM (P6 only) into a uint8 RGBA buffer (alpha=255). Width/height
 * are returned through out-params. Caller free()s the buffer. */
static uint8_t *read_ppm_rgba(const char *path, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "not a P6 PPM: %s\n", path);
        fclose(f);
        return NULL;
    }
    int w = 0, h = 0, maxval = 0;
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3
        || w <= 0 || h <= 0 || maxval != 255) {
        fprintf(stderr, "bad PPM header in %s\n", path);
        fclose(f);
        return NULL;
    }
    /* PPM spec: exactly one whitespace byte after maxval. */
    int c = fgetc(f);
    if (c == EOF) { fclose(f); return NULL; }

    size_t n = (size_t)w * (size_t)h;
    uint8_t *rgb = (uint8_t *)malloc(n * 3);
    uint8_t *rgba = (uint8_t *)malloc(n * 4);
    if (!rgb || !rgba) {
        free(rgb); free(rgba); fclose(f);
        fprintf(stderr, "OOM\n");
        return NULL;
    }
    if (fread(rgb, 1, n * 3, f) != n * 3) {
        free(rgb); free(rgba); fclose(f);
        fprintf(stderr, "short read on %s\n", path);
        return NULL;
    }
    fclose(f);
    for (size_t i = 0; i < n; ++i) {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    free(rgb);
    *out_w = w;
    *out_h = h;
    return rgba;
}

static int write_ppm(const char *path, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h * 3u;
    int ok = (fwrite(rgb, 1, n, f) == n);
    fclose(f);
    return ok;
}

/* Best-effort recursive mkdir -p (single directory level — out_dir is
 * a flat path here). */
static void ensure_dir(const char *path) {
    if (!path || !*path) return;
    mkdir(path, 0755);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) { usage(argv[0]); return 1; }

    const char *in_path  = argv[1];
    const char *out_dir  = argv[2];
    int n_frames = 60;
    int out_w    = 0;
    int out_h    = 0;

    if (argc >= 4 && parse_pos_int_strict(argv[3], &n_frames, 100000) != 0) {
        fprintf(stderr, "invalid n_frames: '%s'\n", argv[3]); return 1;
    }
    if (argc >= 5 && parse_pos_int_strict(argv[4], &out_w, 16384) != 0) {
        fprintf(stderr, "invalid width: '%s'\n", argv[4]); return 1;
    }
    if (argc >= 6 && parse_pos_int_strict(argv[5], &out_h, 16384) != 0) {
        fprintf(stderr, "invalid height: '%s'\n", argv[5]); return 1;
    }

    int src_w = 0, src_h = 0;
    uint8_t *rgba = read_ppm_rgba(in_path, &src_w, &src_h);
    if (!rgba) return 2;

    if (out_w == 0) out_w = src_w;
    if (out_h == 0) out_h = src_h;

    CEScene scene;
    CESceneProfiles profs;
    uint16_t built = ce_scene_build_from_rgba(&scene, &profs,
                                              rgba, src_w, src_h, 0);
    free(rgba);
    if (built == 0) {
        fprintf(stderr, "ce_scene_build_from_rgba returned 0 objects "
                        "(image too small or low-contrast)\n");
        return 3;
    }

    /* Summarise the decomposition so the caller can see what motion
     * types the bridge tagged each cluster with. STATIC stays put,
     * FLOW slides, ORGANIC sways, RIGID jitters, PARTICLE drifts. */
    static const char *type_names[] = {
        "STATIC", "FLOW", "RIGID", "ORGANIC", "PARTICLE"
    };
    fprintf(stderr, "decomposed %u objects from %s (%dx%d):\n",
            built, in_path, src_w, src_h);
    for (uint16_t i = 0; i < built; ++i) {
        unsigned t = profs.profiles[i].type;
        const char *tn = (t < 5) ? type_names[t] : "UNKNOWN";
        fprintf(stderr, "  obj=%u motion=%s cx=%u cy=%u color=(%u,%u,%u)\n",
                scene.objects[i].id, tn,
                profs.base_cx[i], profs.base_cy[i],
                scene.objects[i].color_r,
                scene.objects[i].color_g,
                scene.objects[i].color_b);
    }

    ensure_dir(out_dir);
    uint8_t *frame = (uint8_t *)malloc((size_t)out_w * (size_t)out_h * 3u);
    if (!frame) {
        fprintf(stderr, "OOM frame buffer\n");
        return 4;
    }

    char path[1024];
    for (int f = 0; f < n_frames; ++f) {
        ce_scene_tick_with_profiles(&scene, &profs);
        memset(frame, 0, (size_t)out_w * (size_t)out_h * 3u);
        ce_scene_render(&scene, frame, out_w, out_h);
        snprintf(path, sizeof(path), "%s/frame_%04d.ppm", out_dir, f);
        if (!write_ppm(path, frame, out_w, out_h)) {
            fprintf(stderr, "write_ppm failed: %s\n", path);
            free(frame);
            return 5;
        }
    }

    free(frame);
    fprintf(stderr, "wrote %d frames to %s/frame_NNNN.ppm (%dx%d)\n",
            n_frames, out_dir, out_w, out_h);
    return 0;
}
