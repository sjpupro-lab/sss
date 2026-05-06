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

/* Single-best matcher across the whole token set. Kept for backward
 * compatibility — the new path uses find_cells_per_token below, which
 * returns one cell per token so a multi-word prompt like "red blue"
 * can place both colours instead of collapsing to one.
 *
 * Now uses CEUnit/ce_distance against each cell's ce_key. The legacy
 * sss_fingerprint / fp_distance helpers are still defined above for
 * callers that need them. */
__attribute__((unused))
static int best_cell_for_type(const SSSModel *m,
                              uint32_t type,
                              char **tokens, int n)
{
    int best = -1;
    uint32_t best_d = 0xFFFFFFFFu;
    for (int i = 0; i < n; ++i) {
        CEUnit query;
        ce_init(&query);
        ce_feed(&query, (const uint8_t *)tokens[i],
                (uint32_t)strlen(tokens[i]));
        for (uint32_t j = 0; j < m->num_cells; ++j) {
            if (m->cells[j].type != type) continue;
            uint32_t d = ce_distance(&query, &m->cells[j].ce_key);
            if (d < best_d) { best_d = d; best = (int)j; }
        }
    }
    /* ce_distance ranges 0 (identical) up to ~16320 (orthogonal).
     * 4000 keeps "close enough" matches and rejects clear misses. */
    if (best >= 0 && best_d > 4000u) return -1;
    return best;
}

/* For each token, find the closest cell of `type`. Threshold 4000 in
 * ce_distance space (0 = identical, ~16320 = orthogonal). When
 * multiple cells share the minimum distance (the trainer's 2-column
 * format keeps one cell per (image, word) with identical ce_key), one
 * is picked deterministically from the seed — passing a different
 * seed shuffles the choice across the tied set. Duplicates across
 * tokens are dropped. Returns count written into out_indices. */
static int find_cells_per_token(const SSSModel *m, uint32_t type,
                                char **tokens, int ntok,
                                int *out_indices, int max_out,
                                uint32_t seed)
{
    enum { TIE_BUF = 64 };
    const uint32_t TIE_EPS = 4u;       /* tolerate tiny ce_distance drift */
    int n_out = 0;
    for (int ti = 0; ti < ntok && n_out < max_out; ++ti) {
        CEUnit query;
        ce_init(&query);
        ce_feed(&query, (const uint8_t *)tokens[ti],
                (uint32_t)strlen(tokens[ti]));
        uint32_t best_d = 0xFFFFFFFFu;
        for (uint32_t j = 0; j < m->num_cells; ++j) {
            if (m->cells[j].type != type) continue;
            uint32_t d = ce_distance(&query, &m->cells[j].ce_key);
            if (d < best_d) best_d = d;
        }
        if (best_d > 4000u) continue;
        int tied[TIE_BUF]; int n_tied = 0;
        for (uint32_t j = 0; j < m->num_cells && n_tied < TIE_BUF; ++j) {
            if (m->cells[j].type != type) continue;
            uint32_t d = ce_distance(&query, &m->cells[j].ce_key);
            if (d <= best_d + TIE_EPS) tied[n_tied++] = (int)j;
        }
        if (n_tied <= 0) continue;
        uint32_t pick = (n_tied > 1)
            ? ((seed + (uint32_t)ti * 0x9E3779B9u) % (uint32_t)n_tied)
            : 0u;
        int chosen = tied[pick];
        int dup = 0;
        for (int k = 0; k < n_out; ++k) {
            if (out_indices[k] == chosen) { dup = 1; break; }
        }
        if (!dup) out_indices[n_out++] = chosen;
    }
    return n_out;
}

/* Pick the region cell for a given axis position. When `n` ≥ 2 and
 * `axis_pos` is within ±fade_pixels of a region boundary, also returns
 * the next region's cell and the fade weight `t ∈ [0, 1]`. t=0 means
 * fully `out_a`, t=1 fully `out_b`. */
static void pick_region_cell(const SSSModel *m, const int *indices, int n,
                             int axis_pos, int axis_extent, int fade_pixels,
                             const SSSCell **out_a,
                             const SSSCell **out_b,
                             float *out_t)
{
    *out_a = NULL; *out_b = NULL; *out_t = 0.0f;
    if (n <= 0 || axis_extent <= 0) return;
    int region = (axis_pos * n) / axis_extent;
    if (region < 0) region = 0;
    if (region >= n) region = n - 1;
    *out_a = &m->cells[indices[region]];
    if (region + 1 < n && fade_pixels > 0) {
        int boundary = ((region + 1) * axis_extent) / n;
        if (axis_pos >= boundary - fade_pixels &&
            axis_pos <= boundary + fade_pixels) {
            *out_b = &m->cells[indices[region + 1]];
            float t = (float)(axis_pos - (boundary - fade_pixels))
                    / (float)(2 * fade_pixels);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            *out_t = t;
        }
    }
}

/* Shortest-arc phase blend. Converts each phase to (cos, sin), linearly
 * mixes, then atan2 back — handles the ±π wrap automatically. */
static float blend_phase(float pa, float pb, float t)
{
    float ca = cosf(pa), sa = sinf(pa);
    float cb = cosf(pb), sb = sinf(pb);
    float cm = ca * (1.0f - t) + cb * t;
    float sm = sa * (1.0f - t) + sb * t;
    return atan2f(sm, cm);
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

    /* One cell per token. "red blue" → both red AND blue cells; the
     * sculpt loop below partitions rows/columns between them and
     * fades across the band boundaries. */
    int color_indices[MAX_TOKENS];
    int shape_indices[MAX_TOKENS];
    int face_indices[MAX_TOKENS];
    /* Seed feeds into tie-breaking when same-fp cells exist (multiple
     * training images per word in the 2-column labels.tsv format). */
    uint32_t pick_seed = seed ? seed : 0xC0FFEE11u;
    int n_colors = find_cells_per_token(m, SSS_CE_COLOR, tokens, ntok,
                                        color_indices, MAX_TOKENS,
                                        pick_seed);
    int n_shapes = find_cells_per_token(m, SSS_CE_SHAPE, tokens, ntok,
                                        shape_indices, MAX_TOKENS,
                                        pick_seed ^ 0xA5A5A5A5u);
    int n_faces  = find_cells_per_token(m, SSS_CE_FACE,  tokens, ntok,
                                        face_indices,  MAX_TOKENS,
                                        pick_seed ^ 0x5A5A5A5Au);
    /* Fade band across region boundaries (rows / columns). 2 means a
     * 5-row transition centred on the boundary. */
    const int FADE_PIXELS = 2;

    if (sss_image_alloc(out, H, W) != 0) {
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -4;
    }
    size_t HW = (size_t)H * (size_t)W;

    /* Workspaces. Allocated up-front because the spectral-noise
     * initialiser below also needs row_amp / row_phase / row_out. */
    uint32_t rng = seed ? seed : 0xC0FFEE11u;
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

    /* Spectral noise init — "radio static". Build each row of each
     * channel as the inverse-FFT of random amp + random phase, instead
     * of dropping uniform noise on top of mid-gray. The sculpt loop
     * below then tunes that noise toward the cell targets the way you
     * tune a radio dial — small amp/phase nudges per pass.
     *
     * rng_uniform returns [-1, 1); we map it into [0, 0.3) for amp and
     * [-π, π) for phase to match the natural FFT ranges. */
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < NF; ++k) {
                row_amp[k]   = 0.3f * 0.5f * (rng_uniform(&rng) + 1.0f);
                row_phase[k] = (float)M_PI * rng_uniform(&rng);
            }
            sss_irfft(row_amp, row_phase, NF, W, row_out);
            for (int x = 0; x < W; ++x)
                out->data[((size_t)r * W + x) * 3 + c] = row_out[x];
        }
    }
    /* Bring it back into the displayable [0, 1] band before sculpting. */
    for (size_t i = 0; i < HW * 3; ++i) {
        if (out->data[i] < 0.0f) out->data[i] = 0.0f;
        else if (out->data[i] > 1.0f) out->data[i] = 1.0f;
    }

    /* Phase is only blended when the cell carries one matching its
     * amp shape. Trainers that drop phase leave row_phase / col_phase
     * to evolve naturally (matches the pre-tuning behaviour). The
     * inline cell_has_phase macro below is reused for every region
     * cell picked per row / column. */
    #define CELL_HAS_PHASE(c) ((c) && (c)->phase \
                            && (c)->phase_len == (c)->amp_len)

    for (int step = 0; step < steps; ++step) {
        /* Cooling schedule: pull hard at the start, just nudge later. */
        float t = (steps > 1) ? (float)step / (float)(steps - 1) : 0.0f;
        float alpha       = 0.95f - 0.65f * t;
        float phase_alpha = alpha * 0.5f;     /* phase tunes more slowly */

        /* ── Y projection: row FFT, blend amp + phase toward COLOR/FACE ──
         * Each row picks its own COLOR / FACE cell from the per-token
         * list (via row index → region). Within FADE_PIXELS of a region
         * boundary, the row is a weighted blend of the two adjacent
         * cells so transitions don't show as a hard seam. */
        for (int r = 0; r < H; ++r) {
            const SSSCell *cc_a, *cc_b; float cc_t;
            const SSSCell *cf_a, *cf_b; float cf_t;
            pick_region_cell(m, color_indices, n_colors,
                             r, H, FADE_PIXELS, &cc_a, &cc_b, &cc_t);
            pick_region_cell(m, face_indices,  n_faces,
                             r, H, FADE_PIXELS, &cf_a, &cf_b, &cf_t);
            int cc_a_phase = CELL_HAS_PHASE(cc_a);
            int cc_b_phase = CELL_HAS_PHASE(cc_b);
            int cf_a_phase = CELL_HAS_PHASE(cf_a);
            int cf_b_phase = CELL_HAS_PHASE(cf_b);

            for (int c = 0; c < 3; ++c) {
                for (int x = 0; x < W; ++x) {
                    row_in[x] = out->data[((size_t)r * W + x) * 3 + c];
                }
                sss_rfft(row_in, W, row_amp, row_phase);

                for (int k = 0; k < NF; ++k) {
                    float target_amp   = row_amp[k];
                    float target_phase = row_phase[k];
                    int   has_target_phase = 0;
                    if (k < NF_LOW) {
                        if (cc_a) {
                            size_t idx = ((size_t)r * NF_LOW + k) * 3 + c;
                            float amp_a = cc_a->amp[idx];
                            if (cc_b) {
                                float amp_b = cc_b->amp[idx];
                                target_amp = amp_a * (1.0f - cc_t)
                                           + amp_b * cc_t;
                            } else {
                                target_amp = amp_a;
                            }
                            if (cc_a_phase) {
                                if (cc_b && cc_b_phase) {
                                    target_phase = blend_phase(
                                        cc_a->phase[idx],
                                        cc_b->phase[idx], cc_t);
                                } else {
                                    target_phase = cc_a->phase[idx];
                                }
                                has_target_phase = 1;
                            }
                        }
                    } else {
                        int kh = k - NF_LOW;
                        /* No FACE match → leave the high band where it is.
                         * (A `* 0.5f` factor here would compound across the
                         * schedule and drive every iteration's high-freq
                         * energy toward zero, making FACE de-facto required
                         * for sharp output.) */
                        if (cf_a) {
                            size_t idx = ((size_t)r * NF_HIGH + kh) * 3 + c;
                            float amp_a = cf_a->amp[idx] * detail;
                            if (cf_b) {
                                float amp_b = cf_b->amp[idx] * detail;
                                target_amp = amp_a * (1.0f - cf_t)
                                           + amp_b * cf_t;
                            } else {
                                target_amp = amp_a;
                            }
                            if (cf_a_phase) {
                                if (cf_b && cf_b_phase) {
                                    target_phase = blend_phase(
                                        cf_a->phase[idx],
                                        cf_b->phase[idx], cf_t);
                                } else {
                                    target_phase = cf_a->phase[idx];
                                }
                                has_target_phase = 1;
                            }
                        }
                    }
                    row_amp[k] = alpha * target_amp
                               + (1.0f - alpha) * row_amp[k];

                    if (has_target_phase) {
                        float dph = target_phase - row_phase[k];
                        while (dph >  (float)M_PI) dph -= 2.0f * (float)M_PI;
                        while (dph < -(float)M_PI) dph += 2.0f * (float)M_PI;
                        row_phase[k] += phase_alpha * dph;
                    }
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

        /* ── X projection: column FFT of grayscale, blend amp + phase
         *    toward SHAPE, then transfer the structural delta back to RGB
         *    Note: m->height == m->width is required, so the column FFT
         *    uses the same NF as the row FFT. */
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
            const SSSCell *cs_a, *cs_b; float cs_t;
            pick_region_cell(m, shape_indices, n_shapes,
                             x, W, FADE_PIXELS, &cs_a, &cs_b, &cs_t);
            int cs_a_phase = CELL_HAS_PHASE(cs_a);
            int cs_b_phase = CELL_HAS_PHASE(cs_b);

            for (int y = 0; y < H; ++y) {
                col_in[y] = gray[(size_t)y * W + x];
            }
            sss_rfft(col_in, H, col_amp, col_phase);

            for (int k = 0; k < NF; ++k) {
                float target_amp   = col_amp[k];
                float target_phase = col_phase[k];
                int   has_target_phase = 0;
                if (cs_a) {
                    size_t idx = (size_t)x * NF + k;
                    float amp_a = cs_a->amp[idx];
                    if (cs_b) {
                        float amp_b = cs_b->amp[idx];
                        target_amp = amp_a * (1.0f - cs_t) + amp_b * cs_t;
                    } else {
                        target_amp = amp_a;
                    }
                    if (cs_a_phase) {
                        if (cs_b && cs_b_phase) {
                            target_phase = blend_phase(
                                cs_a->phase[idx], cs_b->phase[idx], cs_t);
                        } else {
                            target_phase = cs_a->phase[idx];
                        }
                        has_target_phase = 1;
                    }
                }
                col_amp[k] = alpha * target_amp
                           + (1.0f - alpha) * col_amp[k];

                if (has_target_phase) {
                    float dph = target_phase - col_phase[k];
                    while (dph >  (float)M_PI) dph -= 2.0f * (float)M_PI;
                    while (dph < -(float)M_PI) dph += 2.0f * (float)M_PI;
                    col_phase[k] += phase_alpha * dph;
                }
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
