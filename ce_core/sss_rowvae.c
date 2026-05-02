/* sss_rowvae.c — Spectrogram-based image generation engine.
 *
 * The model stores spectrograms (row-FFT amplitudes for color/face
 * cells, column-FFT amplitudes for shape cells), not pixels. The
 * generator does NOT assemble those amplitudes directly into an
 * image — instead it runs an iterative noise → measure-error →
 * fix → repeat loop:
 *
 *   image = noise (RGB float)
 *   for step in range(steps):
 *     for each row, each channel:
 *       (measured_amp, phase) = rfft(row)
 *       target_amp            = COLOR (low band) ∪ FACE (high band)
 *       new_amp               = α * target + (1 − α) * measured
 *       row                   = irfft(new_amp, phase)
 *     for each column:
 *       (measured_amp, phase) = rfft(grayscale_col)
 *       target_amp            = SHAPE
 *       new_amp               = α * target + (1 − α) * measured
 *     apply grayscale delta to every channel (preserves chroma)
 *     clamp to [0, 1]
 *
 * α decays linearly from ~0.95 down to ~0.30 across the schedule, so
 * early passes pull hard toward the target spectrogram and later
 * passes only nudge — the same Gerchberg–Saxton structure that
 * recovers images from amplitude-only spectra. Phases evolve
 * naturally from the seeded noise: same prompt + different seed
 * yields different images, but each one ends up matching the cell
 * amplitudes.
 *
 * No external FFT library is used — sss_rfft / sss_irfft are direct
 * DFT pairs, fast enough for the 64×64 / 128×128 sizes this engine
 * targets.
 */
#include "sss_rowvae.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_TOKENS 32

/* ── 256-grid fingerprint ─────────────────────────────────────
 * Byte-level histogram with a position-mixed neighbour bump,
 * then L2-normalised. The Python trainer must implement the
 * exact same algorithm (see scripts/sss_train.py:fingerprint).
 */
void sss_fingerprint(const char *text, float *fp)
{
    memset(fp, 0, SSS_FP_LEN * sizeof(float));
    if (!text) return;
    size_t len = strlen(text);
    for (size_t i = 0; i < len; ++i) {
        unsigned char b = (unsigned char)text[i];
        fp[b] += 1.0f;
        unsigned char n = (unsigned char)((b + 1u + (unsigned)i) & 0xffu);
        fp[n] += 0.3f;
    }
    float s = 0.0f;
    for (int i = 0; i < SSS_FP_LEN; ++i) s += fp[i] * fp[i];
    s = sqrtf(s);
    if (s > 1e-9f) {
        float inv = 1.0f / s;
        for (int i = 0; i < SSS_FP_LEN; ++i) fp[i] *= inv;
    }
}

static float fp_distance(const float *a, const float *b)
{
    float s = 0.0f;
    for (int i = 0; i < SSS_FP_LEN; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

int sss_search(const SSSModel *m, uint32_t type, const float *fp)
{
    int best = -1;
    float best_d = 1e30f;
    for (uint32_t i = 0; i < m->num_cells; ++i) {
        if (m->cells[i].type != type) continue;
        float d = fp_distance(fp, m->cells[i].fp);
        if (d < best_d) { best_d = d; best = (int)i; }
    }
    if (best >= 0 && best_d > 1.2f) return -1;
    return best;
}

/* ── Forward real FFT ─────────────────────────────────────────
 * Matches numpy.fft.rfft: X[k] = Σ_n x[n] * exp(-j*2π*k*n/N), no
 * scaling on the forward pass. Out-arrays are nf = N/2+1 long. */
void sss_rfft(const float *x, int N, float *amp, float *phase)
{
    int nf = N / 2 + 1;
    float two_pi_over_N = 2.0f * (float)M_PI / (float)N;
    for (int k = 0; k < nf; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; ++n) {
            float ang = two_pi_over_N * (float)k * (float)n;
            re += x[n] * cosf(ang);
            im -= x[n] * sinf(ang);
        }
        amp[k]   = sqrtf(re * re + im * im);
        phase[k] = atan2f(im, re);
    }
}

/* ── Inverse real FFT ─────────────────────────────────────────
 * Reverses numpy.fft.rfft: given amp[0..nf-1] and phase[0..nf-1],
 * rebuild the N-length real signal via
 *   x[n] = (1/N) * ( amp[0]*cos(phase[0])
 *                  + 2 * sum_{k=1..nf-2} amp[k]*cos(2*pi*k*n/N + phase[k])
 *                  + amp[nf-1]*cos(pi*n + phase[nf-1])    if N even )
 * The Nyquist bin (k = nf-1, present only when N is even) does not
 * get the factor of 2 because it has no symmetric partner.
 */
void sss_irfft(const float *amp, const float *phase, int nf, int N, float *out)
{
    float two_pi_over_N = 2.0f * (float)M_PI / (float)N;
    for (int n = 0; n < N; ++n) {
        float sum = amp[0] * cosf(phase[0]);
        for (int k = 1; k < nf - 1; ++k) {
            float ang = two_pi_over_N * (float)k * (float)n + phase[k];
            sum += 2.0f * amp[k] * cosf(ang);
        }
        if ((N & 1) == 0 && nf > 1) {
            float ang = (float)M_PI * (float)n + phase[nf - 1];
            sum += amp[nf - 1] * cosf(ang);
        } else if (nf > 1) {
            int k = nf - 1;
            float ang = two_pi_over_N * (float)k * (float)n + phase[k];
            sum += 2.0f * amp[k] * cosf(ang);
        }
        out[n] = sum / (float)N;
    }
}

/* ── Image helpers ──────────────────────────────────────────── */

int sss_image_alloc(SSSImage *img, int h, int w)
{
    img->height = h;
    img->width  = w;
    img->data   = (float *)calloc((size_t)h * (size_t)w * 3, sizeof(float));
    return img->data ? 0 : -1;
}

void sss_image_free(SSSImage *img)
{
    if (!img) return;
    free(img->data);
    img->data = NULL;
    img->height = img->width = 0;
}

static unsigned char clamp_u8(float v)
{
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (unsigned char)(v * 255.0f + 0.5f);
}

int sss_image_save_ppm(const SSSImage *img, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->height * (size_t)img->width;
    unsigned char *row = (unsigned char *)malloc(n * 3);
    if (!row) { fclose(f); return -1; }
    for (size_t i = 0; i < n; ++i) {
        row[i * 3 + 0] = clamp_u8(img->data[i * 3 + 0]);
        row[i * 3 + 1] = clamp_u8(img->data[i * 3 + 1]);
        row[i * 3 + 2] = clamp_u8(img->data[i * 3 + 2]);
    }
    size_t wrote = fwrite(row, 1, n * 3, f);
    free(row);
    fclose(f);
    return (wrote == n * 3) ? 0 : -1;
}

/* ── Tiny deterministic PRNG (xorshift32) ─────────────────── */
static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s ? *s : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}
static float rng_uniform(uint32_t *s)        /* [-1, 1) */
{
    return ((float)(rng_next(s) >> 8) / 8388608.0f) - 1.0f;
}

/* ── Tokeniser ─────────────────────────────────────────────── */
static int is_command_word(const char *t)
{
    static const char *cmds[] = {
        "그려줘", "그려", "그리기", "그림",
        "draw", "generate", "make", "render",
        NULL
    };
    for (int i = 0; cmds[i]; ++i)
        if (strcmp(t, cmds[i]) == 0) return 1;
    return 0;
}

static int tokenise(const char *prompt, char **tokens, int *out_count)
{
    int n = 0;
    const char *p = prompt;
    while (*p && n < MAX_TOKENS) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
        size_t len = (size_t)(p - start);
        char *t = (char *)malloc(len + 1);
        if (!t) return -1;
        memcpy(t, start, len);
        t[len] = '\0';
        if (!is_command_word(t)) tokens[n++] = t;
        else                     free(t);
    }
    *out_count = n;
    return 0;
}

static int best_cell_for_type(const SSSModel *m,
                              uint32_t type,
                              char **tokens, int n)
{
    int best = -1;
    float best_d = 1e30f;
    for (int i = 0; i < n; ++i) {
        float fp[SSS_FP_LEN];
        sss_fingerprint(tokens[i], fp);
        for (uint32_t j = 0; j < m->num_cells; ++j) {
            if (m->cells[j].type != type) continue;
            float d = fp_distance(fp, m->cells[j].fp);
            if (d < best_d) { best_d = d; best = (int)j; }
        }
    }
    if (best >= 0 && best_d > 1.2f) return -1;
    return best;
}

/* ── Iterative generation ─────────────────────────────────────
 * The key change vs. a single-pass spectrogram → image bake: we
 * never construct a "target spectrogram" to inverse-FFT. Instead
 * we measure the current image's actual spectrogram on every
 * iteration, see where it disagrees with the cell targets
 * (the "error"), and blend toward the target by α(step). Phases
 * are not touched directly — they evolve through the projections,
 * so different starting noise → different final images.
 */
int sss_generate(const SSSModel *m,
                 const char     *prompt,
                 uint32_t        seed,
                 float           detail,
                 int             steps,
                 SSSImage       *out)
{
    if (!m || !prompt || !out) return -1;
    if (m->height != m->width) return -2;

    int H = (int)m->height;
    int W = (int)m->width;
    int NF = (int)m->nf;
    int NF_LOW = (int)m->nf_low;
    int NF_HIGH = NF - NF_LOW;
    if (detail <= 0.0f) detail = 1.0f;
    if (steps   <= 0)   steps  = 24;

    char *tokens[MAX_TOKENS];
    int ntok = 0;
    if (tokenise(prompt, tokens, &ntok) != 0) return -3;

    int ic  = best_cell_for_type(m, SSS_CE_COLOR, tokens, ntok);
    int is  = best_cell_for_type(m, SSS_CE_SHAPE, tokens, ntok);
    int ifc = best_cell_for_type(m, SSS_CE_FACE,  tokens, ntok);
    const SSSCell *cc = (ic  >= 0) ? &m->cells[ic]  : NULL;
    const SSSCell *cs = (is  >= 0) ? &m->cells[is]  : NULL;
    const SSSCell *cf = (ifc >= 0) ? &m->cells[ifc] : NULL;

    if (sss_image_alloc(out, H, W) != 0) {
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -4;
    }
    size_t HW = (size_t)H * (size_t)W;

    /* Initial noise around mid-gray. */
    uint32_t rng = seed ? seed : 0xC0FFEE11u;
    for (size_t i = 0; i < HW * 3; ++i) {
        out->data[i] = 0.5f + 0.20f * rng_uniform(&rng);
    }

    /* Workspaces. */
    float *row_in    = (float *)malloc((size_t)W  * sizeof(float));
    float *row_amp   = (float *)malloc((size_t)NF * sizeof(float));
    float *row_phase = (float *)malloc((size_t)NF * sizeof(float));
    float *row_out   = (float *)malloc((size_t)W  * sizeof(float));
    float *col_in    = (float *)malloc((size_t)H  * sizeof(float));
    float *col_amp   = (float *)malloc((size_t)NF * sizeof(float));
    float *col_phase = (float *)malloc((size_t)NF * sizeof(float));
    float *col_out   = (float *)malloc((size_t)H  * sizeof(float));
    float *gray      = (float *)malloc(HW * sizeof(float));
    float *new_gray  = (float *)malloc(HW * sizeof(float));
    if (!row_in || !row_amp || !row_phase || !row_out
     || !col_in || !col_amp || !col_phase || !col_out
     || !gray   || !new_gray) {
        free(row_in); free(row_amp); free(row_phase); free(row_out);
        free(col_in); free(col_amp); free(col_phase); free(col_out);
        free(gray);   free(new_gray);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        sss_image_free(out);
        return -5;
    }

    for (int step = 0; step < steps; ++step) {
        /* Cooling schedule: pull hard at the start, just nudge later. */
        float t = (steps > 1) ? (float)step / (float)(steps - 1) : 0.0f;
        float alpha = 0.95f - 0.65f * t;

        /* ── Y projection: row FFT, blend amps toward COLOR/FACE ── */
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < 3; ++c) {
                for (int x = 0; x < W; ++x) {
                    row_in[x] = out->data[((size_t)r * W + x) * 3 + c];
                }
                sss_rfft(row_in, W, row_amp, row_phase);

                for (int k = 0; k < NF; ++k) {
                    float target;
                    if (k < NF_LOW) {
                        target = cc
                            ? cc->amp[((size_t)r * NF_LOW + k) * 3 + c]
                            : row_amp[k];
                    } else {
                        int kh = k - NF_LOW;
                        target = cf
                            ? cf->amp[((size_t)r * NF_HIGH + kh) * 3 + c] * detail
                            : row_amp[k] * 0.5f;
                    }
                    row_amp[k] = alpha * target + (1.0f - alpha) * row_amp[k];
                }
                sss_irfft(row_amp, row_phase, NF, W, row_out);

                for (int x = 0; x < W; ++x) {
                    out->data[((size_t)r * W + x) * 3 + c] = row_out[x];
                }
            }
        }
        for (size_t i = 0; i < HW * 3; ++i) {
            if (out->data[i] < 0.0f) out->data[i] = 0.0f;
            else if (out->data[i] > 1.0f) out->data[i] = 1.0f;
        }

        /* ── X projection: column FFT of grayscale, blend toward SHAPE,
         *    then transfer the structural delta back to RGB ── */
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t pi = (size_t)y * W + x;
                gray[pi] = (out->data[pi * 3 + 0]
                          + out->data[pi * 3 + 1]
                          + out->data[pi * 3 + 2]) * (1.0f / 3.0f);
            }
        }
        memcpy(new_gray, gray, HW * sizeof(float));

        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                col_in[y] = gray[(size_t)y * W + x];
            }
            sss_rfft(col_in, H, col_amp, col_phase);

            for (int k = 0; k < NF; ++k) {
                float target = cs
                    ? cs->amp[(size_t)x * NF + k]
                    : col_amp[k];
                col_amp[k] = alpha * target + (1.0f - alpha) * col_amp[k];
            }
            sss_irfft(col_amp, col_phase, NF, H, col_out);

            for (int y = 0; y < H; ++y) {
                new_gray[(size_t)y * W + x] = col_out[y];
            }
        }
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t pi = (size_t)y * W + x;
                float delta = new_gray[pi] - gray[pi];
                /* 0.7 — partial luminance transfer keeps chroma alive. */
                out->data[pi * 3 + 0] += delta * 0.7f;
                out->data[pi * 3 + 1] += delta * 0.7f;
                out->data[pi * 3 + 2] += delta * 0.7f;
            }
        }
        for (size_t i = 0; i < HW * 3; ++i) {
            if (out->data[i] < 0.0f) out->data[i] = 0.0f;
            else if (out->data[i] > 1.0f) out->data[i] = 1.0f;
        }
    }

    free(row_in); free(row_amp); free(row_phase); free(row_out);
    free(col_in); free(col_amp); free(col_phase); free(col_out);
    free(gray);   free(new_gray);
    for (int i = 0; i < ntok; ++i) free(tokens[i]);
    return 0;
}
