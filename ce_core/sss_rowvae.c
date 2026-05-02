/* sss_rowvae.c — Spectrogram-based image generation engine.
 *
 * The model stores spectrograms (row-FFT amplitudes for color/face
 * cells, column-FFT amplitudes for shape cells), not pixels. To draw
 * an image we look up cells by 256-grid fingerprint, assemble Y and
 * X half-spectra with a touch of seeded noise, inverse-FFT them,
 * cross-multiply (color × shading × edge), and run a few passes of
 * row/column smoothing to suppress FFT ringing.
 *
 * No external FFT library is used — sss_irfft is a direct DFT-inverse
 * with the real-symmetry shortcut, which is fast enough for the
 * 64×64 / 128×128 sizes this engine targets.
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
    /* Unit-vector distance is in [0, sqrt(2)]; treat anything past
     * 1.2 as "no real match" so noise can take over for that slot. */
    if (best >= 0 && best_d > 1.2f) return -1;
    return best;
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

/* ── Generation ──────────────────────────────────────────────
 * The pipeline:
 *   1. Tokenise the prompt on whitespace, strip command words.
 *   2. For each token compute its fingerprint and try to match
 *      it against COLOR / SHAPE / FACE cells. Per type, keep the
 *      single best hit.
 *   3. Build the Y half-spectrum (rows × nf × 3): low-band copy
 *      from COLOR, high-band copy (scaled by `detail`) from FACE,
 *      both blended with seeded noise. Phases pulled from the
 *      stored reference plus a small jitter (otherwise the same
 *      seed/labels always yield the same image — the task requires
 *      seed-driven variation).
 *   4. Build the X half-spectrum (cols × nf): from SHAPE.
 *   5. Inverse-FFT row-wise → y_image (RGB), col-wise → x_struct.
 *   6. Combine: shade × color, with an extracted edge term darkening
 *      structural boundaries.
 *   7. Two-axis median-style smoothing to kill ringing artefacts.
 */

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

/* Splits prompt on whitespace into up to MAX_TOKENS owned strings.
 * Caller must free each tokens[i] and the array slots are valid up
 * to *out_count. */
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
        if (!is_command_word(t)) {
            tokens[n++] = t;
        } else {
            free(t);
        }
    }
    *out_count = n;
    return 0;
}

/* Best cell of `type` across all tokens (lower distance wins). */
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

int sss_generate(const SSSModel *m,
                 const char     *prompt,
                 uint32_t        seed,
                 float           detail,
                 SSSImage       *out)
{
    if (!m || !prompt || !out) return -1;
    if (m->height != m->width) return -2;   /* engine assumes square */

    int H = (int)m->height;
    int W = (int)m->width;
    int NF = (int)m->nf;
    int NF_LOW = (int)m->nf_low;
    int NF_HIGH = NF - NF_LOW;
    if (detail <= 0.0f) detail = 1.0f;

    char *tokens[MAX_TOKENS];
    int ntok = 0;
    if (tokenise(prompt, tokens, &ntok) != 0) return -3;

    int ic = best_cell_for_type(m, SSS_CE_COLOR, tokens, ntok);
    int is = best_cell_for_type(m, SSS_CE_SHAPE, tokens, ntok);
    int ifc = best_cell_for_type(m, SSS_CE_FACE,  tokens, ntok);

    const SSSCell *cc = (ic  >= 0) ? &m->cells[ic]  : NULL;
    const SSSCell *cs = (is  >= 0) ? &m->cells[is]  : NULL;
    const SSSCell *cf = (ifc >= 0) ? &m->cells[ifc] : NULL;

    /* Y half-spectrum: H rows × NF freqs × 3 channels. */
    size_t y_n = (size_t)H * (size_t)NF * 3;
    float *y_amp   = (float *)calloc(y_n, sizeof(float));
    float *y_phase = (float *)calloc(y_n, sizeof(float));
    /* X half-spectrum: W cols × NF freqs (column FFT of grayscale). */
    size_t x_n = (size_t)W * (size_t)NF;
    float *x_amp   = (float *)calloc(x_n, sizeof(float));
    float *x_phase = (float *)calloc(x_n, sizeof(float));
    if (!y_amp || !y_phase || !x_amp || !x_phase) {
        free(y_amp); free(y_phase); free(x_amp); free(x_phase);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -4;
    }

    uint32_t rng = seed ? seed : 0xC0FFEE11u;

    /* ─ Y assembly ─ */
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < NF; ++k) {
                size_t idx = ((size_t)r * NF + k) * 3 + c;
                float amp = 0.0f, ph = 0.0f;
                if (k < NF_LOW) {
                    if (cc) {
                        size_t cidx = ((size_t)r * NF_LOW + k) * 3 + c;
                        amp = cc->amp[cidx];
                        ph  = cc->phase[cidx];
                    } else {
                        amp = 0.05f * fabsf(rng_uniform(&rng));
                        ph  = (float)M_PI * rng_uniform(&rng);
                    }
                    amp += 0.02f * rng_uniform(&rng);
                    ph  += 0.05f * rng_uniform(&rng);
                } else {
                    int kh = k - NF_LOW;
                    if (cf) {
                        size_t fidx = ((size_t)r * NF_HIGH + kh) * 3 + c;
                        amp = cf->amp[fidx] * detail;
                        ph  = cf->phase[fidx];
                    } else {
                        amp = 0.01f * fabsf(rng_uniform(&rng)) * detail;
                        ph  = (float)M_PI * rng_uniform(&rng);
                    }
                    amp += 0.005f * rng_uniform(&rng);
                    ph  += 0.10f * rng_uniform(&rng);
                }
                if (amp < 0.0f) amp = 0.0f;
                y_amp[idx]   = amp;
                y_phase[idx] = ph;
            }
        }
    }

    /* ─ X assembly ─ */
    for (int c = 0; c < W; ++c) {
        for (int k = 0; k < NF; ++k) {
            size_t idx = (size_t)c * NF + k;
            float amp = 0.0f, ph = 0.0f;
            if (cs) {
                amp = cs->amp[idx];
                ph  = cs->phase[idx];
            } else {
                amp = 0.05f * fabsf(rng_uniform(&rng));
                ph  = (float)M_PI * rng_uniform(&rng);
            }
            amp += 0.02f * rng_uniform(&rng);
            ph  += 0.05f * rng_uniform(&rng);
            if (amp < 0.0f) amp = 0.0f;
            x_amp[idx]   = amp;
            x_phase[idx] = ph;
        }
    }

    /* ─ Inverse FFT: rows for Y → RGB image ─ */
    float *y_image = (float *)calloc((size_t)H * W * 3, sizeof(float));
    float *row_amp   = (float *)malloc(NF * sizeof(float));
    float *row_phase = (float *)malloc(NF * sizeof(float));
    float *row_out   = (float *)malloc(W  * sizeof(float));
    if (!y_image || !row_amp || !row_phase || !row_out) {
        free(y_image); free(row_amp); free(row_phase); free(row_out);
        free(y_amp); free(y_phase); free(x_amp); free(x_phase);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -5;
    }
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < NF; ++k) {
                size_t idx = ((size_t)r * NF + k) * 3 + c;
                row_amp[k]   = y_amp[idx];
                row_phase[k] = y_phase[idx];
            }
            sss_irfft(row_amp, row_phase, NF, W, row_out);
            for (int x = 0; x < W; ++x) {
                y_image[((size_t)r * W + x) * 3 + c] = row_out[x];
            }
        }
    }

    /* ─ Inverse FFT: cols for X → grayscale structure ─ */
    float *x_struct  = (float *)calloc((size_t)H * W, sizeof(float));
    float *col_amp   = (float *)malloc(NF * sizeof(float));
    float *col_phase = (float *)malloc(NF * sizeof(float));
    float *col_out   = (float *)malloc(H  * sizeof(float));
    if (!x_struct || !col_amp || !col_phase || !col_out) {
        free(x_struct); free(col_amp); free(col_phase); free(col_out);
        free(y_image); free(row_amp); free(row_phase); free(row_out);
        free(y_amp); free(y_phase); free(x_amp); free(x_phase);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -6;
    }
    for (int c = 0; c < W; ++c) {
        for (int k = 0; k < NF; ++k) {
            size_t idx = (size_t)c * NF + k;
            col_amp[k]   = x_amp[idx];
            col_phase[k] = x_phase[idx];
        }
        sss_irfft(col_amp, col_phase, NF, H, col_out);
        for (int y = 0; y < H; ++y) {
            x_struct[(size_t)y * W + c] = col_out[y];
        }
    }
    free(row_amp); free(row_phase); free(row_out);
    free(col_amp); free(col_phase); free(col_out);
    free(y_amp); free(y_phase); free(x_amp); free(x_phase);

    /* Normalise x_struct to roughly [-0.5, 0.5] for use as shading. */
    float xmin = 1e30f, xmax = -1e30f;
    size_t HW = (size_t)H * W;
    for (size_t i = 0; i < HW; ++i) {
        if (x_struct[i] < xmin) xmin = x_struct[i];
        if (x_struct[i] > xmax) xmax = x_struct[i];
    }
    float xrange = xmax - xmin;
    if (xrange < 1e-6f) xrange = 1.0f;
    for (size_t i = 0; i < HW; ++i) {
        x_struct[i] = (x_struct[i] - xmin) / xrange - 0.5f;
    }

    /* Edge map: high-pass via 3-point Laplacian on x_struct. */
    float *edge = (float *)calloc(HW, sizeof(float));
    if (!edge) {
        free(x_struct); free(y_image);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -7;
    }
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float c  = x_struct[(size_t)y * W + x];
            float u  = x_struct[(size_t)(y - 1) * W + x];
            float d  = x_struct[(size_t)(y + 1) * W + x];
            float l  = x_struct[(size_t)y * W + (x - 1)];
            float rr = x_struct[(size_t)y * W + (x + 1)];
            float e = fabsf(4.0f * c - u - d - l - rr);
            edge[(size_t)y * W + x] = e;
        }
    }
    /* Normalise edge to [0, 1]. */
    float emax = 0.0f;
    for (size_t i = 0; i < HW; ++i) if (edge[i] > emax) emax = edge[i];
    if (emax < 1e-6f) emax = 1.0f;
    for (size_t i = 0; i < HW; ++i) edge[i] /= emax;

    /* Cross-compose: final = y_image * shade * (1 - 0.5 * edge). */
    if (sss_image_alloc(out, H, W) != 0) {
        free(edge); free(x_struct); free(y_image);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -8;
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t pi = (size_t)y * W + x;
            float shade = 0.65f + 0.45f * (x_struct[pi] + 0.5f); /* roughly [0.65, 1.10] */
            float em = 1.0f - 0.5f * edge[pi];
            for (int c = 0; c < 3; ++c) {
                float v = y_image[pi * 3 + c] * shade * em;
                /* y_image carries an arbitrary FFT scale — recentre to [0,1]. */
                v = 0.5f + 0.5f * v;
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                out->data[pi * 3 + c] = v;
            }
        }
    }
    free(edge); free(x_struct); free(y_image);
    for (int i = 0; i < ntok; ++i) free(tokens[i]);

    /* ─ Latent correction: 4 passes, both axes. Damps any single
     *   row/column that disagrees with both of its neighbours. ─ */
    float thresh = 0.18f;
    float *tmp = (float *)malloc(HW * 3 * sizeof(float));
    if (!tmp) return 0;            /* image already produced; skip smoothing */
    for (int pass = 0; pass < 4; ++pass) {
        memcpy(tmp, out->data, HW * 3 * sizeof(float));
        /* Row direction. */
        for (int y = 1; y < H - 1; ++y) {
            float du = 0.0f, dd = 0.0f;
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    du += fabsf(tmp[((size_t)y * W + x) * 3 + c]
                              - tmp[((size_t)(y - 1) * W + x) * 3 + c]);
                    dd += fabsf(tmp[((size_t)y * W + x) * 3 + c]
                              - tmp[((size_t)(y + 1) * W + x) * 3 + c]);
                }
            }
            du /= (float)(W * 3);
            dd /= (float)(W * 3);
            if (du > thresh && dd > thresh) {
                for (int x = 0; x < W; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        size_t idx = ((size_t)y * W + x) * 3 + c;
                        out->data[idx] =
                            0.6f * tmp[idx]
                          + 0.2f * tmp[((size_t)(y - 1) * W + x) * 3 + c]
                          + 0.2f * tmp[((size_t)(y + 1) * W + x) * 3 + c];
                    }
                }
            }
        }
        memcpy(tmp, out->data, HW * 3 * sizeof(float));
        /* Column direction. */
        for (int x = 1; x < W - 1; ++x) {
            float dl = 0.0f, dr = 0.0f;
            for (int y = 0; y < H; ++y) {
                for (int c = 0; c < 3; ++c) {
                    dl += fabsf(tmp[((size_t)y * W + x) * 3 + c]
                              - tmp[((size_t)y * W + (x - 1)) * 3 + c]);
                    dr += fabsf(tmp[((size_t)y * W + x) * 3 + c]
                              - tmp[((size_t)y * W + (x + 1)) * 3 + c]);
                }
            }
            dl /= (float)(H * 3);
            dr /= (float)(H * 3);
            if (dl > thresh && dr > thresh) {
                for (int y = 0; y < H; ++y) {
                    for (int c = 0; c < 3; ++c) {
                        size_t idx = ((size_t)y * W + x) * 3 + c;
                        out->data[idx] =
                            0.6f * tmp[idx]
                          + 0.2f * tmp[((size_t)y * W + (x - 1)) * 3 + c]
                          + 0.2f * tmp[((size_t)y * W + (x + 1)) * 3 + c];
                    }
                }
            }
        }
    }
    free(tmp);
    return 0;
}
