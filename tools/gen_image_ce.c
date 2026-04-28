/* gen_image_ce.c — E2E image generation from a CEStorage (.ces) file.
 *
 * Usage:
 *   gen_image_ce <model.ces> <prompt> <out.ppm> [seed] [steps] [wave_iters]
 *
 * Pipeline:
 *   1. ce_storage_load(.ces)
 *   2. ce_generate_image_typed(CE_TYPE_IMAGE, prompt, seed, cfg, wave_iters)
 *   3. write 256x256 PPM P6 (RGB only).
 *
 * Defaults: seed=0, steps=8 (fast preset; pass >= 50 for HQ), wave_iters=200.
 */
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_gen.h"
#include "ce_decode.h"
#include "ce_denoise.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_ppm(const char *path, const CEImage *img) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    for (int i = 0; i < img->width * img->height; ++i) {
        unsigned char rgb[3] = {
            img->pixels[i].r, img->pixels[i].g, img->pixels[i].b
        };
        if (fwrite(rgb, 1, 3, f) != 3) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s [--routed] <model.ces> <prompt> <out.ppm> [seed] [steps] [wave_iters]\n",
                argv[0]);
        return 2;
    }
    int routed = 0;
    int argi = 1;
    if (strcmp(argv[argi], "--routed") == 0) {
        routed = 1;
        ++argi;
    }
    if (argc - argi < 3) {
        fprintf(stderr,
                "usage: %s [--routed] <model.ces> <prompt> <out.ppm> [seed] [steps] [wave_iters]\n",
                argv[0]);
        return 2;
    }
    const char *ces_path  = argv[argi];
    const char *prompt    = argv[argi + 1];
    const char *out_path  = argv[argi + 2];
    uint64_t seed         = (argc - argi > 3) ? (uint64_t)strtoull(argv[argi + 3], NULL, 0) : 0u;
    int steps             = (argc - argi > 4) ? atoi(argv[argi + 4]) : 8;
    uint32_t wave_iters   = (argc - argi > 5) ? (uint32_t)strtoul(argv[argi + 5], NULL, 0) : 200u;

    CEStorage S;
    if (!ce_storage_load(&S, ces_path)) {
        fprintf(stderr, "[gen_image_ce] load %s failed\n", ces_path);
        return 1;
    }
    fprintf(stderr, "[gen_image_ce] loaded %s (%u entries)\n", ces_path, S.count);

    CEGenConfig cfg = (steps >= 50) ? ce_gen_config_hq() : ce_gen_config_default();
    if (steps > 0) cfg.total_steps = steps;

    CEImage *img = (CEImage *)calloc(1, sizeof(CEImage));
    if (!img) { ce_storage_free(&S); return 1; }

    fprintf(stderr,
            "[gen_image_ce] prompt=\"%s\" seed=0x%llx steps=%d wave_iters=%u\n",
            prompt, (unsigned long long)seed, cfg.total_steps, wave_iters);

    if (routed) {
        ce_generate_image_label_routed(img, &S, prompt, seed, &cfg, wave_iters);
    } else {
        ce_generate_image_typed(img, &S, CE_TYPE_IMAGE,
                                prompt, seed, &cfg, wave_iters);
    }

    /* Print summary stats so an automated pipeline test can verify the
     * generated image without parsing the PPM. */
    double sr = 0, sg = 0, sb = 0;
    int n = img->width * img->height;
    for (int i = 0; i < n; ++i) {
        sr += img->pixels[i].r;
        sg += img->pixels[i].g;
        sb += img->pixels[i].b;
    }
    fprintf(stderr,
            "[gen_image_ce] mean RGB = (%.1f, %.1f, %.1f)\n",
            sr / n, sg / n, sb / n);

    if (!write_ppm(out_path, img)) {
        fprintf(stderr, "[gen_image_ce] write %s failed\n", out_path);
        free(img);
        ce_storage_free(&S);
        return 1;
    }
    fprintf(stderr, "[gen_image_ce] wrote %s\n", out_path);

    free(img);
    ce_storage_free(&S);
    return 0;
}
