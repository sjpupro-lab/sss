/* sss_gen — CLI for the spectrogram image engine.
 *
 *   ./sss_gen MODEL.sss "빨간 원 그려줘" out.ppm [seed] [detail] [steps]
 *
 * Loads a v8 .sss model (produced by scripts/sss_train.py) and runs
 * the iterative noise → measure-error → fix → repeat loop until the
 * image matches the cell spectrograms picked by the prompt. Writes a
 * PPM. seed defaults to 1, detail to 1.0, steps to 24.
 *
 * Numeric arguments are parsed strictly (no `atoi`/`atof` silent-zero
 * fallback): a typo or empty value fails fast so reproducibility is
 * never quietly compromised.
 */
#include "sss_rowvae.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s MODEL.sss PROMPT OUT.ppm [seed] [detail] [steps]\n"
        "  seed    default 1     (any unsigned int; controls noise variation)\n"
        "  detail  default 1.0   (>0; >1 = sharper high-frequency contribution)\n"
        "  steps   default 24    (positive int; refinement iterations)\n",
        argv0);
}

/* All three return 0 on success. The CLI fails loudly on bad input
 * instead of silently falling back to defaults. */

static int parse_uint_strict(const char *s, uint32_t *out)
{
    if (!s || !*s) return -1;
    /* strtoul accepts an optional leading '+' or '-'; without this
     * guard "-1" would silently wrap to 0xFFFFFFFF on most platforms. */
    if (*s == '-' || *s == '+') return -1;
    errno = 0;
    char *endp = NULL;
    unsigned long v = strtoul(s, &endp, 0);
    if (errno != 0 || !endp || *endp != '\0' || v > 0xFFFFFFFFul) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int parse_pos_int_strict(const char *s, int *out)
{
    if (!s || !*s) return -1;
    errno = 0;
    char *endp = NULL;
    long v = strtol(s, &endp, 10);
    if (errno != 0 || !endp || *endp != '\0' || v <= 0 || v > 100000) return -1;
    *out = (int)v;
    return 0;
}

static int parse_pos_float_strict(const char *s, float *out)
{
    if (!s || !*s) return -1;
    errno = 0;
    char *endp = NULL;
    double v = strtod(s, &endp);
    /* Reject non-finite ("inf", "nan") and anything that would overflow
     * the float cast — letting them through would poison the FFT math
     * with infinities or NaNs. */
    if (errno != 0 || !endp || *endp != '\0' || !isfinite(v) || !(v > 0.0))
        return -1;
    float f = (float)v;
    if (!isfinite(f) || !(f > 0.0f)) return -1;
    *out = f;
    return 0;
}

int main(int argc, char **argv)
{
    /* Strict positional contract: 3 required + up to 3 optional. Any
     * trailing arguments are typos that would otherwise be silently
     * ignored — reject them so reproducibility is never quietly
     * compromised. */
    if (argc < 4 || argc > 7) { usage(argv[0]); return 1; }

    const char *model_path = argv[1];
    const char *prompt     = argv[2];
    const char *out_path   = argv[3];
    uint32_t seed   = 1u;
    float    detail = 1.0f;
    int      steps  = 24;

    if (argc >= 5 && parse_uint_strict(argv[4], &seed) != 0) {
        fprintf(stderr, "invalid seed: '%s' (expected unsigned int)\n", argv[4]);
        return 1;
    }
    if (argc >= 6 && parse_pos_float_strict(argv[5], &detail) != 0) {
        fprintf(stderr, "invalid detail: '%s' (expected positive number)\n", argv[5]);
        return 1;
    }
    if (argc >= 7 && parse_pos_int_strict(argv[6], &steps) != 0) {
        fprintf(stderr, "invalid steps: '%s' (expected positive int)\n", argv[6]);
        return 1;
    }

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
    rc = sss_generate(&m, prompt, seed, detail, steps, &img);
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
        "wrote %s  (%dx%d, prompt=\"%s\", seed=%u, detail=%.3f, steps=%d)\n",
        out_path, img.width, img.height, prompt, seed, (double)detail, steps);

    sss_image_free(&img);
    sss_model_free(&m);
    return 0;
}
