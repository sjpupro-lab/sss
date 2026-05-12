/* sss_gen — single CLI entry point for the sss image / video engine.
 *
 *   ./sss_gen MODEL PROMPT OUT [seed] [detail] [steps] \
 *       [--out-w W] [--out-h H] [--atmos] \
 *       [--frames N] [--out-dir DIR] [--feature-bank PATH] \
 *       [--condition LABEL INTENSITY] [--atmos-from PPM] \
 *       [--temporal-mode static|video]
 *
 * Phase 1–3 path (frames = 1):
 *   Loads a v8/v9 .sss model (produced by scripts/sss_train.py) and
 *   runs the iterative noise → measure-error → fix → repeat loop
 *   until the image matches the cell spectrograms picked by the
 *   prompt. Writes a single PPM.
 *
 * Phase 4 path (frames > 1):
 *   Conditions, .sfb feature banks, Atmos warm-starts, and the
 *   resonance-based per-frame warp pipeline live in Python — the
 *   binary refuses those flags and points callers to
 *   `scripts/sss_gen.py`. The C binary's `--frames N > 1` itself
 *   still works: it loops `sss_generate` with seed ^= t per frame,
 *   producing N PPMs that share the prompt-resolved motif content
 *   but carry Phase 1's per-seed random-phase diversity at the
 *   frame level. This is a useful "static motif, per-frame seed
 *   variation" baseline that matches the Phase 4 design
 *   philosophy: frames = 1 is a special case of frames = N, both
 *   running through the same sss_generate core.
 *
 * Defaults: seed=1, detail=1.5, steps=120 (the integration pass
 * raised these from the legacy 1.0/24 because the radio loop's
 * cooling schedule converges much cleaner past 80 steps).
 *
 * Flag reference:
 *   --out-w W / --out-h H   resize the FFT result to (W,H) via
 *                           nearest-neighbour mapping that hits
 *                           both endpoints.
 *   --atmos                 after the FFT pass, additively
 *                           composite ce_scene_render on top —
 *                           production hook for the Atmos engine.
 *   --frames N              N >= 1. N = 1 writes OUT verbatim;
 *                           N > 1 requires --out-dir and writes
 *                           DIR/frame_NNNN.ppm.
 *   --out-dir DIR           required with --frames > 1 (created
 *                           if missing).
 *   --feature-bank PATH     Reserved for the Python entry. The C
 *                           binary refuses .sfb input — call
 *                           `python3 scripts/sss_gen.py` instead.
 *   --condition LBL INT     Reserved for the Python entry; see
 *                           --feature-bank.
 *   --atmos-from PPM        Reserved for the Python entry; see
 *                           --feature-bank.
 *   --temporal-mode MODE    static|video. `static` is a no-op;
 *                           `video` is implied by --frames > 1 and
 *                           accepted only for CLI completeness.
 *
 * Numeric arguments are parsed strictly (no atoi/atof silent-zero
 * fallback): a typo or empty value fails fast so reproducibility
 * is never quietly compromised.
 */
#include "sss_rowvae.h"
#include "ce_scene_object.h"
#include "ce_scene_bridge.h"
#include "ce_move_profile.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s MODEL.sss PROMPT OUT.ppm [seed] [detail] [steps]\n"
        "       [--out-w W] [--out-h H] [--atmos]\n"
        "       [--frames N] [--out-dir DIR]\n"
        "       [--feature-bank PATH] [--condition LBL INT]\n"
        "       [--atmos-from PPM] [--temporal-mode static|video]\n"
        "\n"
        "  seed         default 1   (any unsigned int)\n"
        "  detail       default 1.5 (>0; >1 = sharper high-frequency)\n"
        "  steps        default 120 (positive int)\n"
        "  --out-w/--out-h  output width/height; defaults to model size.\n"
        "  --atmos          composite ce_scene_render on top.\n"
        "  --frames N       N=1 writes OUT verbatim; N>1 requires --out-dir.\n"
        "  --out-dir DIR    output directory for per-frame PPMs.\n"
        "\n"
        "Phase 4 features (.sfb feature banks, condition signals, Atmos\n"
        "warm-starts) live in scripts/sss_gen.py — this binary refuses\n"
        "--feature-bank / --condition / --atmos-from and points there.\n",
        argv0);
}

/* mkdir -p semantics for the --out-dir target. */
static int ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    if (mkdir(path, 0755) == 0) return 0;
    return -1;
}

static int parse_uint_strict(const char *s, uint32_t *out)
{
    if (!s || !*s) return -1;
    if (*s == '-' || *s == '+') return -1;
    errno = 0;
    char *endp = NULL;
    unsigned long v = strtoul(s, &endp, 0);
    if (errno != 0 || !endp || *endp != '\0' || v > 0xFFFFFFFFul) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int parse_pos_int_strict(const char *s, int *out, int max_val)
{
    if (!s || !*s) return -1;
    errno = 0;
    char *endp = NULL;
    long v = strtol(s, &endp, 10);
    if (errno != 0 || !endp || *endp != '\0' || v <= 0 || v > max_val) return -1;
    *out = (int)v;
    return 0;
}

static int parse_pos_float_strict(const char *s, float *out)
{
    if (!s || !*s) return -1;
    errno = 0;
    char *endp = NULL;
    double v = strtod(s, &endp);
    if (errno != 0 || !endp || *endp != '\0' || !isfinite(v) || !(v > 0.0))
        return -1;
    float f = (float)v;
    if (!isfinite(f) || !(f > 0.0f)) return -1;
    *out = f;
    return 0;
}

static unsigned char clamp_u8_pf(float v)
{
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (unsigned char)(v * 255.0f + 0.5f);
}

/* Quantise the float SSSImage into an RGBA8 buffer at (out_w, out_h)
 * using nearest-neighbour endpoint mapping (corners of source → corners
 * of output). Caller free()s the returned buffer. Returns NULL on OOM. */
static uint8_t *image_to_rgba(const SSSImage *img, int out_w, int out_h)
{
    if (!img || out_w <= 0 || out_h <= 0) return NULL;
    uint8_t *rgba = (uint8_t *)malloc((size_t)out_w * (size_t)out_h * 4u);
    if (!rgba) return NULL;
    int sH = img->height, sW = img->width;
    for (int y = 0; y < out_h; ++y) {
        int sy = (out_h <= 1) ? 0
            : (int)((double)y * (sH - 1) / (out_h - 1) + 0.5);
        if (sy >= sH) sy = sH - 1;
        for (int x = 0; x < out_w; ++x) {
            int sx = (out_w <= 1) ? 0
                : (int)((double)x * (sW - 1) / (out_w - 1) + 0.5);
            if (sx >= sW) sx = sW - 1;
            size_t spi = (size_t)sy * sW + sx;
            size_t dpi = (size_t)y  * out_w + x;
            rgba[dpi * 4 + 0] = clamp_u8_pf(img->data[spi * 3 + 0]);
            rgba[dpi * 4 + 1] = clamp_u8_pf(img->data[spi * 3 + 1]);
            rgba[dpi * 4 + 2] = clamp_u8_pf(img->data[spi * 3 + 2]);
            rgba[dpi * 4 + 3] = 255;
        }
    }
    return rgba;
}

static int write_ppm(const char *path, const uint8_t *rgb, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h * 3u;
    int ok = (fwrite(rgb, 1, n, f) == n);
    fclose(f);
    return ok;
}

/* Composite the Atmos additive render on top of the FFT image. Takes
 * an RGBA8 base (from image_to_rgba) and overlays the int32-accumulated
 * scene render. Output is uint8 RGB packed at (w, h). */
static void compose_atmos(const uint8_t *base_rgba, int w, int h,
                          const CEScene *scene, uint8_t *out_rgb)
{
    /* Render the scene into a transient RGB buffer, then add per-pixel
     * with saturation. Cheap and matches the "additive only" rule. */
    uint8_t *atm = (uint8_t *)calloc((size_t)w * (size_t)h * 3u, 1u);
    if (!atm) {
        /* Fall back to plain copy of the base — better than crashing. */
        for (int i = 0; i < w * h; ++i) {
            out_rgb[i * 3 + 0] = base_rgba[i * 4 + 0];
            out_rgb[i * 3 + 1] = base_rgba[i * 4 + 1];
            out_rgb[i * 3 + 2] = base_rgba[i * 4 + 2];
        }
        return;
    }
    ce_scene_render(scene, atm, w, h);
    for (int i = 0; i < w * h; ++i) {
        int r = (int)base_rgba[i * 4 + 0] + (int)atm[i * 3 + 0];
        int g = (int)base_rgba[i * 4 + 1] + (int)atm[i * 3 + 1];
        int b = (int)base_rgba[i * 4 + 2] + (int)atm[i * 3 + 2];
        out_rgb[i * 3 + 0] = (uint8_t)(r > 255 ? 255 : r);
        out_rgb[i * 3 + 1] = (uint8_t)(g > 255 ? 255 : g);
        out_rgb[i * 3 + 2] = (uint8_t)(b > 255 ? 255 : b);
    }
    free(atm);
}

int main(int argc, char **argv)
{
    if (argc < 4) { usage(argv[0]); return 1; }

    const char *model_path = argv[1];
    const char *prompt     = argv[2];
    const char *out_path   = argv[3];
    uint32_t seed   = 1u;
    float    detail = 1.5f;     /* was 1.0 — see header comment. */
    int      steps  = 120;      /* was 24 — see header comment. */
    int      out_w  = 0;        /* 0 = use model size */
    int      out_h  = 0;
    int      use_atmos = 0;
    int      frames    = 1;
    const char *out_dir = NULL;

    /* Walk argv past the 3 required positionals. Numeric positionals
     * (seed, detail, steps) are consumed in order until a flag arrives;
     * after that flags can appear in any order. */
    int positional_slot = 0;
    for (int i = 4; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--out-w") == 0) {
            if (i + 1 >= argc || parse_pos_int_strict(argv[++i], &out_w, 16384) != 0) {
                fprintf(stderr, "invalid --out-w\n"); return 1;
            }
        } else if (strcmp(a, "--out-h") == 0) {
            if (i + 1 >= argc || parse_pos_int_strict(argv[++i], &out_h, 16384) != 0) {
                fprintf(stderr, "invalid --out-h\n"); return 1;
            }
        } else if (strcmp(a, "--atmos") == 0) {
            use_atmos = 1;
        } else if (strcmp(a, "--frames") == 0) {
            if (i + 1 >= argc
                    || parse_pos_int_strict(argv[++i], &frames, 4096) != 0) {
                fprintf(stderr, "invalid --frames\n"); return 1;
            }
        } else if (strcmp(a, "--out-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--out-dir requires a path\n"); return 1;
            }
            out_dir = argv[++i];
        } else if (strcmp(a, "--temporal-mode") == 0) {
            /* `static` is a no-op; `video` is implied by --frames > 1.
             * Accepted only so the flag-set matches the documented
             * unified CLI surface. */
            if (i + 1 >= argc) {
                fprintf(stderr, "--temporal-mode requires static|video\n");
                return 1;
            }
            const char *mode = argv[++i];
            if (strcmp(mode, "static") != 0 && strcmp(mode, "video") != 0) {
                fprintf(stderr, "--temporal-mode: must be static or video\n");
                return 1;
            }
        } else if (strcmp(a, "--feature-bank") == 0
                || strcmp(a, "--condition") == 0
                || strcmp(a, "--atmos-from") == 0) {
            /* These flags drive the Phase 4 resonance synthesiser,
             * which lives in tools/sss_synthesizer.py. The C binary
             * doesn't ship that path — direct the caller cleanly
             * rather than silently dropping the flag. */
            fprintf(stderr,
                "%s requires Phase 4 features that live in "
                "scripts/sss_gen.py.\nInvoke:\n  python3 scripts/sss_gen.py "
                "MODEL PROMPT OUT %s ... \n", a, a);
            return 7;
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown flag: %s\n", a); usage(argv[0]); return 1;
        } else {
            switch (positional_slot++) {
            case 0:
                if (parse_uint_strict(a, &seed) != 0) {
                    fprintf(stderr, "invalid seed: '%s'\n", a); return 1;
                }
                break;
            case 1:
                if (parse_pos_float_strict(a, &detail) != 0) {
                    fprintf(stderr, "invalid detail: '%s'\n", a); return 1;
                }
                break;
            case 2:
                if (parse_pos_int_strict(a, &steps, 100000) != 0) {
                    fprintf(stderr, "invalid steps: '%s'\n", a); return 1;
                }
                break;
            default:
                fprintf(stderr, "unexpected extra argument: '%s'\n", a);
                usage(argv[0]); return 1;
            }
        }
    }

    if (frames > 1 && !out_dir) {
        fprintf(stderr, "--frames %d requires --out-dir DIR\n", frames);
        return 1;
    }
    if (out_dir && ensure_dir(out_dir) != 0) {
        fprintf(stderr, "cannot create --out-dir %s\n", out_dir);
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

    /* Pre-resolve the output resolution against the model so the
     * per-frame loop doesn't have to re-derive it each iteration. */
    int resolved_w = (out_w > 0) ? out_w : (int)m.width;
    int resolved_h = (out_h > 0) ? out_h : (int)m.height;

    int exit_rc = 0;
    for (int t = 0; t < frames; ++t) {
        /* Phase 1 per-seed diversity rule, extended frame-by-frame.
         * seed ^ t gives every frame an independent phase init while
         * keeping the result reproducible from (seed, t). For
         * frames = 1 this is a no-op — the call still uses `seed`
         * verbatim. */
        uint32_t frame_seed = seed ^ (uint32_t)t;

        SSSImage img;
        memset(&img, 0, sizeof(img));
        rc = sss_generate(&m, prompt, frame_seed, detail, steps, &img);
        if (rc != 0) {
            fprintf(stderr, "sss_generate(frame %d) failed: rc=%d\n",
                    t, rc);
            sss_model_free(&m);
            return 3;
        }

        uint8_t *rgba = image_to_rgba(&img, resolved_w, resolved_h);
        uint8_t *rgb_out = (uint8_t *)malloc(
            (size_t)resolved_w * (size_t)resolved_h * 3u);
        if (!rgba || !rgb_out) {
            fprintf(stderr, "OOM at frame %d\n", t);
            free(rgba); free(rgb_out);
            sss_image_free(&img);
            sss_model_free(&m);
            return 4;
        }

        if (use_atmos) {
            CEScene scene;
            CESceneProfiles profs;
            uint16_t built = ce_scene_build_from_rgba(
                &scene, &profs, rgba, resolved_w, resolved_h, 0);
            if (t == 0) {
                fprintf(stderr,
                    "atmos: %u objects from FFT result\n", built);
            }
            compose_atmos(rgba, resolved_w, resolved_h, &scene, rgb_out);
        } else {
            for (int i = 0; i < resolved_w * resolved_h; ++i) {
                rgb_out[i * 3 + 0] = rgba[i * 4 + 0];
                rgb_out[i * 3 + 1] = rgba[i * 4 + 1];
                rgb_out[i * 3 + 2] = rgba[i * 4 + 2];
            }
        }

        /* Pick the destination path: single-frame writes verbatim to
         * `out_path`; multi-frame fills <out_dir>/frame_NNNN.ppm. */
        char frame_path[1024];
        const char *dest = out_path;
        if (frames > 1) {
            snprintf(frame_path, sizeof(frame_path),
                     "%s/frame_%04d.ppm", out_dir, t);
            dest = frame_path;
        }
        if (!write_ppm(dest, rgb_out, resolved_w, resolved_h)) {
            fprintf(stderr, "write_ppm(%s) failed\n", dest);
            free(rgba); free(rgb_out);
            sss_image_free(&img);
            sss_model_free(&m);
            return 5;
        }

        free(rgba); free(rgb_out);
        sss_image_free(&img);
    }

    if (frames == 1) {
        fprintf(stderr,
            "wrote %s  (%dx%d, prompt=\"%s\", seed=%u, detail=%.3f, "
            "steps=%d, atmos=%d)\n",
            out_path, resolved_w, resolved_h, prompt, seed,
            (double)detail, steps, use_atmos);
    } else {
        fprintf(stderr,
            "wrote %d frames to %s  (%dx%d, prompt=\"%s\", "
            "base_seed=%u, detail=%.3f, steps=%d, atmos=%d)\n",
            frames, out_dir, resolved_w, resolved_h, prompt, seed,
            (double)detail, steps, use_atmos);
    }
    sss_model_free(&m);
    return exit_rc;
}
