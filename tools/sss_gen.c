/* sss_gen — CLI for the spectrogram image engine.
 *
 *   ./sss_gen MODEL.sss "빨간 원 그려줘" out.ppm [seed] [detail]
 *
 * Loads a v8 .sss model (produced by scripts/sss_train.py),
 * generates an image from the prompt, and writes a PPM. seed
 * defaults to 1 (any non-zero value); detail defaults to 1.0.
 */
#include "sss_rowvae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s MODEL.sss PROMPT OUT.ppm [seed] [detail]\n"
        "  seed   default 1     (any unsigned int; controls noise variation)\n"
        "  detail default 1.0   (>1 = sharper high-frequency contribution)\n",
        argv0);
}

int main(int argc, char **argv)
{
    if (argc < 4) { usage(argv[0]); return 1; }

    const char *model_path = argv[1];
    const char *prompt     = argv[2];
    const char *out_path   = argv[3];
    uint32_t seed   = (argc >= 5) ? (uint32_t)strtoul(argv[4], NULL, 0) : 1u;
    float    detail = (argc >= 6) ? (float)atof(argv[5]) : 1.0f;

    SSSModel m;
    int rc = sss_model_load(model_path, &m);
    if (rc != 0) {
        fprintf(stderr, "sss_model_load(%s) failed: rc=%d\n", model_path, rc);
        return 2;
    }
    fprintf(stderr,
        "loaded %s: H=%u W=%u NF=%u NF_LOW=%u cells=%u\n",
        model_path, m.height, m.width, m.nf, m.nf_low, m.num_cells);

    SSSImage img;
    memset(&img, 0, sizeof(img));
    rc = sss_generate(&m, prompt, seed, detail, &img);
    if (rc != 0) {
        fprintf(stderr, "sss_generate failed: rc=%d\n", rc);
        sss_model_free(&m);
        return 3;
    }

    rc = sss_image_save_ppm(&img, out_path);
    if (rc != 0) {
        fprintf(stderr, "sss_image_save_ppm(%s) failed: rc=%d\n", out_path, rc);
        sss_image_free(&img);
        sss_model_free(&m);
        return 4;
    }

    fprintf(stderr,
        "wrote %s  (%dx%d, prompt=\"%s\", seed=%u, detail=%.3f)\n",
        out_path, img.width, img.height, prompt, seed, (double)detail);

    sss_image_free(&img);
    sss_model_free(&m);
    return 0;
}
