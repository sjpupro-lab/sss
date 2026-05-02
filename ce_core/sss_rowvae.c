/* sss_rowvae.c — Sculpt-based image generation engine.
 *
 * The model stores a *pool of training images* per attribute cell
 * (color / shape / face). To draw, the engine walks the canvas one
 * row at a time and at each row:
 *
 *   1. proposes every (color_candidate, shape_candidate) pair —
 *      "if it were this image, this row would look like that";
 *   2. erases the bad pairs by scoring agreement (low MSE between
 *      the two candidates on this row) plus continuity with the
 *      previous row plus a cumulative bonus for candidates that
 *      have already been picked;
 *   3. composes the surviving pair into the output row by tinting
 *      the color row with the shape row's brightness pattern, then
 *      adding the face row's high-frequency detail;
 *   4. writes the row to the canvas and bumps the winner's score.
 *
 * Row 0 is uncertain; by the time row H-1 is drawn the scores have
 * usually converged on a single (color, shape) pair, so the image
 * is one specific drawing rather than a blur of the pool. Two
 * 4-pass row/column smoothings at the end remove any single rows
 * that disagree with both neighbours.
 *
 * No noise base, no FFT — the canvas starts at a flat (0.94, 0.94,
 * 0.94) gray. The 256-grid fingerprint is only used to look cells
 * up by name.
 */
#include "sss_rowvae.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_TOKENS    32
#define MAX_CANDS     32     /* per cell, sampled if the pool is larger */

/* ── 256-grid fingerprint ─────────────────────────────────────
 * Byte-level histogram with a position-mixed neighbour bump,
 * then L2-normalised. Mirror in scripts/sss_train.py:fingerprint.
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

/* ── Generation ─────────────────────────────────────────────── */

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

/* Mean-square distance between two RGB rows of length W. */
static float row_mse_rgb(const float *a, const float *b, int W)
{
    float s = 0.0f;
    int n = W * 3;
    for (int i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s / (float)n;
}

/* Pointer to row r of image idx inside cell c (RGB, length W*3). */
static inline const float *cell_row(const SSSCell *c, int H, int W,
                                    int img_idx, int r)
{
    (void)H;
    return c->imgs + ((size_t)img_idx * (size_t)H + (size_t)r)
                   * (size_t)W * 3u;
}

/* face_detail = high-frequency residual of a face row, mean-zero,
 * computed from the average of all face candidates at this row.
 * Stored in `out` (length W*3). `scratch` is a caller-supplied
 * buffer of the same length, reused across rows so we don't
 * malloc/free inside the per-row generation loop. */
static void build_face_detail(const SSSCell *cf, int H, int W, int r,
                              float *out, float *scratch)
{
    if (!cf || cf->num_imgs == 0) {
        memset(out, 0, (size_t)W * 3 * sizeof(float));
        return;
    }
    float inv_n = 1.0f / (float)cf->num_imgs;
    /* Average row. */
    float *avg = out;                /* reuse */
    memset(avg, 0, (size_t)W * 3 * sizeof(float));
    for (uint32_t k = 0; k < cf->num_imgs; ++k) {
        const float *row = cell_row(cf, H, W, (int)k, r);
        for (int x = 0; x < W * 3; ++x) avg[x] += row[x];
    }
    for (int x = 0; x < W * 3; ++x) avg[x] *= inv_n;

    /* High-pass: subtract a 1-D box blur (radius 2) along the row,
     * channel-wise, so the residual is mean-zero locally. */
    float buf[3];
    for (int x = 0; x < W; ++x) {
        buf[0] = buf[1] = buf[2] = 0.0f;
        int cnt = 0;
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            buf[0] += avg[xx * 3 + 0];
            buf[1] += avg[xx * 3 + 1];
            buf[2] += avg[xx * 3 + 2];
            ++cnt;
        }
        float inv_c = cnt > 0 ? 1.0f / (float)cnt : 0.0f;
        scratch[x * 3 + 0] = buf[0] * inv_c;
        scratch[x * 3 + 1] = buf[1] * inv_c;
        scratch[x * 3 + 2] = buf[2] * inv_c;
    }
    for (int x = 0; x < W * 3; ++x) avg[x] -= scratch[x];   /* residual */
}

int sss_generate(const SSSModel *m,
                 const char     *prompt,
                 uint32_t        seed,
                 float           detail,
                 SSSImage       *out)
{
    if (!m || !prompt || !out) return -1;
    if (m->height != m->width) return -2;       /* engine assumes square */

    int H = (int)m->height;
    int W = (int)m->width;
    if (detail <= 0.0f) detail = 1.0f;

    char *tokens[MAX_TOKENS];
    int ntok = 0;
    if (tokenise(prompt, tokens, &ntok) != 0) return -3;

    int ic  = best_cell_for_type(m, SSS_CE_COLOR, tokens, ntok);
    int is_ = best_cell_for_type(m, SSS_CE_SHAPE, tokens, ntok);
    int ifc = best_cell_for_type(m, SSS_CE_FACE,  tokens, ntok);

    const SSSCell *cc = (ic  >= 0) ? &m->cells[ic]  : NULL;
    const SSSCell *cs = (is_ >= 0) ? &m->cells[is_] : NULL;
    const SSSCell *cf = (ifc >= 0) ? &m->cells[ifc] : NULL;

    int total_c = (cc && cc->num_imgs > 0) ? (int)cc->num_imgs : 0;
    int total_s = (cs && cs->num_imgs > 0) ? (int)cs->num_imgs : 0;
    int nc = total_c > MAX_CANDS ? MAX_CANDS : total_c;
    int ns = total_s > MAX_CANDS ? MAX_CANDS : total_s;

    uint32_t rng = seed ? seed : 0xC0FFEE11u;

    /* Seed-driven candidate ordering: pick a uniform random sample of
     * up to MAX_CANDS indices from each pool. The full index list
     * lives on the heap (so a malformed model with thousands of
     * images per cell can't blow the stack), and a partial Fisher–
     * Yates writes only nc/ns positions before we copy them into the
     * fixed-size c_idx/s_idx working sets. */
    int c_idx[MAX_CANDS], s_idx[MAX_CANDS];
    int *full_c = NULL, *full_s = NULL;
    if (total_c > 0) {
        full_c = (int *)malloc((size_t)total_c * sizeof(int));
        if (!full_c) {
            for (int i = 0; i < ntok; ++i) free(tokens[i]);
            return -7;
        }
        for (int i = 0; i < total_c; ++i) full_c[i] = i;
        for (int i = 0; i < nc; ++i) {
            int j = i + (int)(rng_next(&rng) % (uint32_t)(total_c - i));
            int tmp = full_c[i]; full_c[i] = full_c[j]; full_c[j] = tmp;
            c_idx[i] = full_c[i];
        }
    }
    if (total_s > 0) {
        full_s = (int *)malloc((size_t)total_s * sizeof(int));
        if (!full_s) {
            free(full_c);
            for (int i = 0; i < ntok; ++i) free(tokens[i]);
            return -7;
        }
        for (int i = 0; i < total_s; ++i) full_s[i] = i;
        for (int i = 0; i < ns; ++i) {
            int j = i + (int)(rng_next(&rng) % (uint32_t)(total_s - i));
            int tmp = full_s[i]; full_s[i] = full_s[j]; full_s[j] = tmp;
            s_idx[i] = full_s[i];
        }
    }
    free(full_c);
    free(full_s);

    /* Cumulative scores (one per candidate). */
    float c_scores[MAX_CANDS];
    float s_scores[MAX_CANDS];
    for (int i = 0; i < MAX_CANDS; ++i) c_scores[i] = s_scores[i] = 0.0f;

    if (sss_image_alloc(out, H, W) != 0) {
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -4;
    }
    /* Blank canvas: flat neutral gray. */
    for (size_t i = 0; i < (size_t)H * (size_t)W * 3; ++i) {
        out->data[i] = 0.94f;
    }

    float *prev_row    = (float *)malloc((size_t)W * 3 * sizeof(float));
    float *face_detail = (float *)malloc((size_t)W * 3 * sizeof(float));
    float *face_scratch = (float *)malloc((size_t)W * 3 * sizeof(float));
    float *out_row     = (float *)malloc((size_t)W * 3 * sizeof(float));
    if (!prev_row || !face_detail || !face_scratch || !out_row) {
        free(prev_row); free(face_detail); free(face_scratch); free(out_row);
        sss_image_free(out);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -5;
    }
    /* prev_row starts as the gray canvas. */
    for (int x = 0; x < W * 3; ++x) prev_row[x] = 0.94f;

    /* If we have no color cell, fall back to a single neutral gray
     * "candidate" row so the loop still produces sensible output. */
    float *fallback = (float *)malloc((size_t)W * 3 * sizeof(float));
    if (!fallback) {
        free(prev_row); free(face_detail); free(face_scratch); free(out_row);
        sss_image_free(out);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -6;
    }
    for (int x = 0; x < W * 3; ++x) fallback[x] = 0.94f;

    for (int r = 0; r < H; ++r) {
        /* Build face detail row (zeros if no face cell). */
        build_face_detail(cf, H, W, r, face_detail, face_scratch);

        /* ── Pick the best (color, shape) pair for this row ── */
        const float *best_color_row = NULL;
        const float *best_shape_row = NULL;
        int best_ci = -1, best_si = -1;
        float best_total = -1e30f;

        if (nc == 0 && ns == 0) {
            best_color_row = fallback;
            best_shape_row = fallback;
        } else if (nc == 0) {
            best_color_row = fallback;
            for (int j = 0; j < ns; ++j) {
                const float *sr = cell_row(cs, H, W, s_idx[j], r);
                float cont = -row_mse_rgb(sr, prev_row, W);
                float bonus = logf(s_scores[j] + 1.0f);
                float total = cont + bonus;
                if (total > best_total) {
                    best_total = total; best_si = j; best_shape_row = sr;
                }
            }
        } else if (ns == 0) {
            best_shape_row = fallback;
            for (int i = 0; i < nc; ++i) {
                const float *cr = cell_row(cc, H, W, c_idx[i], r);
                float cont = -row_mse_rgb(cr, prev_row, W);
                float bonus = logf(c_scores[i] + 1.0f);
                float total = cont + bonus;
                if (total > best_total) {
                    best_total = total; best_ci = i; best_color_row = cr;
                }
            }
        } else {
            for (int i = 0; i < nc; ++i) {
                const float *cr = cell_row(cc, H, W, c_idx[i], r);
                float cont = -row_mse_rgb(cr, prev_row, W);
                float c_bonus = logf(c_scores[i] + 1.0f);
                for (int j = 0; j < ns; ++j) {
                    const float *sr = cell_row(cs, H, W, s_idx[j], r);
                    float agree = -row_mse_rgb(cr, sr, W);
                    float s_bonus = logf(s_scores[j] + 1.0f);
                    float total = agree + cont + c_bonus + s_bonus;
                    if (total > best_total) {
                        best_total = total;
                        best_ci = i; best_si = j;
                        best_color_row = cr; best_shape_row = sr;
                    }
                }
            }
        }

        /* ── Compose: tint color row by shape brightness, add face detail. ── */
        for (int x = 0; x < W; ++x) {
            float cr0 = best_color_row[x * 3 + 0];
            float cr1 = best_color_row[x * 3 + 1];
            float cr2 = best_color_row[x * 3 + 2];
            float sr0 = best_shape_row[x * 3 + 0];
            float sr1 = best_shape_row[x * 3 + 1];
            float sr2 = best_shape_row[x * 3 + 2];

            float c_gray = (cr0 + cr1 + cr2) * (1.0f / 3.0f);
            float s_gray = (sr0 + sr1 + sr2) * (1.0f / 3.0f);

            float structure = s_gray / (c_gray + 1e-6f);
            if (structure < 0.7f) structure = 0.7f;
            if (structure > 1.3f) structure = 1.3f;

            float v0 = cr0 * structure + face_detail[x * 3 + 0] * 0.3f * detail;
            float v1 = cr1 * structure + face_detail[x * 3 + 1] * 0.3f * detail;
            float v2 = cr2 * structure + face_detail[x * 3 + 2] * 0.3f * detail;
            if (v0 < 0.0f) v0 = 0.0f;
            if (v0 > 1.0f) v0 = 1.0f;
            if (v1 < 0.0f) v1 = 0.0f;
            if (v1 > 1.0f) v1 = 1.0f;
            if (v2 < 0.0f) v2 = 0.0f;
            if (v2 > 1.0f) v2 = 1.0f;
            out_row[x * 3 + 0] = v0;
            out_row[x * 3 + 1] = v1;
            out_row[x * 3 + 2] = v2;
        }

        /* ── Accumulate: write to canvas, bump scores, update prev_row. ── */
        memcpy(out->data + (size_t)r * W * 3, out_row,
               (size_t)W * 3 * sizeof(float));
        memcpy(prev_row, out_row, (size_t)W * 3 * sizeof(float));
        if (best_ci >= 0) c_scores[best_ci] += 1.0f;
        if (best_si >= 0) s_scores[best_si] += 1.0f;
    }

    free(prev_row);
    free(face_detail);
    free(face_scratch);
    free(out_row);
    free(fallback);
    for (int i = 0; i < ntok; ++i) free(tokens[i]);

    /* ── Latent correction: 4 passes, both axes. Damps any single
     *   row/column that disagrees sharply with both neighbours. ── */
    size_t HW = (size_t)H * (size_t)W;
    float thresh = 0.12f;
    float *tmp = (float *)malloc(HW * 3 * sizeof(float));
    if (!tmp) return 0;
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
