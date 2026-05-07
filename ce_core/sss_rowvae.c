/* sss_rowvae.c — Sculpt-then-Radio image generation engine.
 *
 * Two stages, sharing the same per-cell amplitude+phase storage:
 *
 *   ── Sculpt (PR #19) ──────────────────────────────────────────
 *   Decides *what* to draw. Collects every candidate cell whose
 *   ce_distance to a prompt token is at the minimum (per-image
 *   cells from the 2-column trainer share an identical ce_key, so
 *   one prompt token like "red" pulls in every red-image's COLOR
 *   cell), reconstructs each candidate to a full (H, W, 3) image
 *   via irfft, then walks the canvas row-by-row scoring every
 *   (color, shape) pair by
 *
 *      score = agree(color_row, shape_row)        // -MSE
 *            + continuity(color_row, prev_row)    // -MSE
 *            + log(color_score + 1)               // bandit bonus
 *            + log(shape_score + 1)
 *
 *   The best pair wins this row, its scores are bumped, and prev_row
 *   advances. After H rows the cells with the highest cumulative
 *   scores are the winners. Seed shuffles the candidate order so
 *   different seeds break ties differently → genuinely different
 *   outputs from the same prompt.
 *
 *   ── Radio (existing spectrogram tuner) ───────────────────────
 *   Decides *how* to draw. The winners' reconstructed images are
 *   row-rfft'd and column-rfft'd to build a target spectrogram
 *   (low band = COLOR, mid band = 0.6·SHAPE + 0.4·FACE, high band
 *   = SHAPE; column tune = SHAPE). The image starts as spectral
 *   noise (random amp+phase per row), then a steps-iteration loop
 *   blends amp + phase toward the target with a cooling α schedule
 *   (steps≤30 → α=0.95→0.30; steps≤80 → 0.50→0.16; else 0.30→0.10).
 *   Phase tunes at half rate (phase_alpha = alpha/2) and uses
 *   shortest-arc wrapping so it never spins the wrong way around π.
 *
 * Net effect: Sculpt converges on the right cells (the *content*
 * decision), Radio converges on a clean rendering of those cells'
 * spectra (the *execution*). Same .sss file, no format change.
 *
 * No external FFT library — sss_rfft / sss_irfft are direct DFT
 * pairs, fast enough for the 64×64 / 128×128 sizes this engine
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

/* Candidate pool size for the sculpt competition. The 2-column
 * trainer produces one cell per (image, word) pair, all sharing the
 * same ce_key; with a 310-image corpus a single prompt token can
 * easily pull in 16+ tied cells. 32 keeps the inner (nc × ns) loop
 * tractable (~1000 pair scores per row at the cap). */
#define MAX_CANDS  32

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

/* Fisher–Yates shuffle of `n` indices, deterministic per seed.
 * Used by the sculpt loop to break ties in candidate ordering — same
 * pool, but a different visit order so each seed lands on a different
 * winner when scores are close. */
static void rng_shuffle(int *idx, int n, uint32_t *s)
{
    for (int i = n - 1; i > 0; --i) {
        uint32_t r = rng_next(s);
        int j = (int)(r % (uint32_t)(i + 1));
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
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
 * tokens are dropped. Returns count written into out_indices.
 *
 * Kept for backward-compat only — sss_generate now uses
 * find_candidate_cells (below) which returns the FULL tied set instead
 * of one-per-token, so the sculpt loop can score every candidate. */
__attribute__((unused))
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
 * fully `out_a`, t=1 fully `out_b`.
 *
 * Kept for backward-compat only — the new sculpt path doesn't carve
 * the canvas into per-token regions, it picks a single winner cell
 * per type from the candidate pool. */
__attribute__((unused))
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
 * mixes, then atan2 back — handles the ±π wrap automatically.
 *
 * Kept for backward-compat only — the new sculpt path resolves a single
 * winner per type, so cross-cell phase blends are no longer needed. */
__attribute__((unused))
static float blend_phase(float pa, float pb, float t)
{
    float ca = cosf(pa), sa = sinf(pa);
    float cb = cosf(pb), sb = sinf(pb);
    float cm = ca * (1.0f - t) + cb * t;
    float sm = sa * (1.0f - t) + sb * t;
    return atan2f(sm, cm);
}

/* ── Sculpt-stage helpers ──────────────────────────────────────── */

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

/* Collect every cell of `type` whose ce_distance to ANY token is
 * essentially zero — i.e. the cell's `ce_key` was built from the
 * exact bytes of one of the prompt tokens. The 2-column trainer
 * keeps one cell per (image, word) with identical ce_key (because
 * `ce_key_bytes("red")` is deterministic), so a single token like
 * "red" brings in every red-image's COLOR cell. The 4-column legacy
 * trainer puts one cell per (label, type), so "red" matches just
 * one cell. Either way the pool is the right one for the sculpt
 * loop to compete over.
 *
 * The absolute threshold (`MATCH_TIGHT`) replaces the older 4000-cap
 * behaviour: at 4000 a semantically unrelated token like "keroppi"
 * would still loosely match COLOR "pink" (d ≈ 600) and contaminate
 * the pool with the wrong cell type. Same-label cells produce
 * d == 0 exactly, off-label cells start at d ≥ ~400, so 100 is far
 * inside the safe band on both sides. TIE_EPS stays small because
 * ce_feed is deterministic per byte input — it's only there as
 * insurance against a future codebook drift in slig_codebook. */
static int find_candidate_cells(const SSSModel *m, uint32_t type,
                                char **tokens, int ntok,
                                int *out_indices, int max_out)
{
    const uint32_t MATCH_TIGHT = 100u;
    const uint32_t TIE_EPS     = 4u;
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
        if (best_d > MATCH_TIGHT) continue;
        for (uint32_t j = 0; j < m->num_cells && n_out < max_out; ++j) {
            if (m->cells[j].type != type) continue;
            uint32_t d = ce_distance(&query, &m->cells[j].ce_key);
            if (d > best_d + TIE_EPS) continue;
            int dup = 0;
            for (int k = 0; k < n_out; ++k) {
                if (out_indices[k] == (int)j) { dup = 1; break; }
            }
            if (!dup) out_indices[n_out++] = (int)j;
        }
    }
    return n_out;
}

/* Reconstruct a COLOR cell to a full (H, W, 3) image by zero-padding
 * the high band and inverse-RFFTing per row, per channel. Cell amp
 * layout is (H, NF_LOW, 3); phase is optional (zero phase is used
 * when the trainer dropped it, so the sculpt scorer still gets a
 * deterministic row to compare against). Workspace buffers are
 * caller-allocated so this can run in tight loops without malloc. */
static int reconstruct_color_cell(const SSSCell *cell,
                                  int H, int W, int NF, int NF_LOW,
                                  float *out_img,
                                  float *amp_pad, float *phase_pad,
                                  float *row_out)
{
    if (!cell || cell->amp_len != (uint32_t)(H * NF_LOW * 3)) return -1;
    int has_phase = (cell->phase
                  && cell->phase_len == cell->amp_len);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < NF; ++k) {
                amp_pad[k]   = 0.0f;
                phase_pad[k] = 0.0f;
            }
            for (int k = 0; k < NF_LOW; ++k) {
                size_t idx = ((size_t)r * NF_LOW + k) * 3 + c;
                amp_pad[k] = cell->amp[idx];
                if (has_phase) phase_pad[k] = cell->phase[idx];
            }
            sss_irfft(amp_pad, phase_pad, NF, W, row_out);
            for (int x = 0; x < W; ++x) {
                out_img[((size_t)r * W + x) * 3 + c] = row_out[x];
            }
        }
    }
    return 0;
}

/* Reconstruct a FACE cell. Same shape as COLOR but the amp covers the
 * HIGH band — pad bins 0..NF_LOW-1 with zero, fill NF_LOW..NF-1 from
 * the cell. */
static int reconstruct_face_cell(const SSSCell *cell,
                                 int H, int W, int NF, int NF_LOW,
                                 float *out_img,
                                 float *amp_pad, float *phase_pad,
                                 float *row_out)
{
    int NF_HIGH = NF - NF_LOW;
    if (!cell || cell->amp_len != (uint32_t)(H * NF_HIGH * 3)) return -1;
    int has_phase = (cell->phase
                  && cell->phase_len == cell->amp_len);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < NF; ++k) {
                amp_pad[k]   = 0.0f;
                phase_pad[k] = 0.0f;
            }
            for (int kh = 0; kh < NF_HIGH; ++kh) {
                size_t idx = ((size_t)r * NF_HIGH + kh) * 3 + c;
                amp_pad[NF_LOW + kh] = cell->amp[idx];
                if (has_phase) phase_pad[NF_LOW + kh] = cell->phase[idx];
            }
            sss_irfft(amp_pad, phase_pad, NF, W, row_out);
            for (int x = 0; x < W; ++x) {
                out_img[((size_t)r * W + x) * 3 + c] = row_out[x];
            }
        }
    }
    return 0;
}

/* Reconstruct a SHAPE cell. The trainer stores SHAPE as a per-column
 * gray RFFT — amp/phase shape (W, NF). Inverse-RFFT each column to
 * get a gray (H, W) field, then broadcast to all 3 channels so the
 * sculpt scorer can MSE it against the (RGB) color rows. */
static int reconstruct_shape_cell(const SSSCell *cell,
                                  int H, int W, int NF,
                                  float *out_img,
                                  float *col_amp, float *col_phase,
                                  float *col_out)
{
    if (!cell || cell->amp_len != (uint32_t)(W * NF)) return -1;
    int has_phase = (cell->phase
                  && cell->phase_len == cell->amp_len);
    for (int x = 0; x < W; ++x) {
        for (int k = 0; k < NF; ++k) {
            col_amp[k]   = cell->amp[(size_t)x * NF + k];
            col_phase[k] = has_phase ? cell->phase[(size_t)x * NF + k]
                                     : 0.0f;
        }
        sss_irfft(col_amp, col_phase, NF, H, col_out);
        for (int y = 0; y < H; ++y) {
            float v = col_out[y];
            out_img[((size_t)y * W + x) * 3 + 0] = v;
            out_img[((size_t)y * W + x) * 3 + 1] = v;
            out_img[((size_t)y * W + x) * 3 + 2] = v;
        }
    }
    return 0;
}

/* argmax over n floats. Returns 0 when n == 0. */
static int argmax_f(const float *a, int n)
{
    int best = 0;
    float best_v = (n > 0) ? a[0] : 0.0f;
    for (int i = 1; i < n; ++i) {
        if (a[i] > best_v) { best_v = a[i]; best = i; }
    }
    return best;
}

/* ── Sculpt-then-Radio generation ─────────────────────────────
 * Stage 1 (Sculpt): collect candidate pools per type, reconstruct
 * each candidate to a full image, walk row-by-row scoring every
 * (color, shape) pair with -MSE(color, shape) -MSE(color, prev)
 * + log(c_score+1) + log(s_score+1), bump the winner's score, advance
 * prev_row. After H rows the highest-scoring candidate per type is
 * the winner.
 *
 * Stage 2 (Radio): row-RFFT and column-RFFT the winner reconstructions
 * to build target spectrograms (low band = COLOR, mid = 0.6·SHAPE +
 * 0.4·FACE, high = SHAPE; column = SHAPE). Initialise the canvas with
 * spectral noise (amp ∈ [0, 0.3), phase ∈ [-π, π) per row, per
 * channel), then iterate `steps` times blending row+column FFT amp
 * and phase toward the targets with a cooling α schedule. Larger
 * `steps` means a smaller base α — fewer, gentler nudges per pass,
 * trading speed for sharpness (AM → FM analogy: same target, finer
 * tuning grain).
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
    /* Output resolution is currently locked to model resolution; the
     * hooks below (out_w / out_h locals) are kept as the obvious place
     * to plug a future resolution-override parameter without touching
     * the sculpt or radio paths. */
    int out_w = W;
    int out_h = H;
    if (detail <= 0.0f) detail = 1.0f;
    if (steps   <= 0)   steps  = 24;

    char *tokens[MAX_TOKENS];
    int ntok = 0;
    if (tokenise(prompt, tokens, &ntok) != 0) return -3;

    /* ── Stage 1: Sculpt — collect candidate pools, reconstruct, compete ── */

    int color_idx[MAX_CANDS];
    int shape_idx[MAX_CANDS];
    int face_idx [MAX_CANDS];
    int nc = find_candidate_cells(m, SSS_CE_COLOR, tokens, ntok,
                                  color_idx, MAX_CANDS);
    int ns = find_candidate_cells(m, SSS_CE_SHAPE, tokens, ntok,
                                  shape_idx, MAX_CANDS);
    int nf = find_candidate_cells(m, SSS_CE_FACE,  tokens, ntok,
                                  face_idx,  MAX_CANDS);

    /* Per-pool reconstruction scratch — reused across all candidates of
     * each type. NF amp + NF phase + max(W, H) row/col output. */
    int max_axis = (W > H) ? W : H;
    float *amp_pad   = (float *)malloc((size_t)NF * sizeof(float));
    float *phase_pad = (float *)malloc((size_t)NF * sizeof(float));
    float *axis_out  = (float *)malloc((size_t)max_axis * sizeof(float));

    /* Reconstructed candidate images — laid out as
     *   pool_imgs[pool_idx * (H * W * 3) ...]
     * One block per pool (color / shape / face). */
    size_t per_img = (size_t)H * (size_t)W * 3u;
    float *pool_color = (nc > 0) ? (float *)malloc(per_img * (size_t)nc * sizeof(float)) : NULL;
    float *pool_shape = (ns > 0) ? (float *)malloc(per_img * (size_t)ns * sizeof(float)) : NULL;
    float *pool_face  = (nf > 0) ? (float *)malloc(per_img * (size_t)nf * sizeof(float)) : NULL;
    if (!amp_pad || !phase_pad || !axis_out
     || (nc > 0 && !pool_color)
     || (ns > 0 && !pool_shape)
     || (nf > 0 && !pool_face)) {
        free(amp_pad); free(phase_pad); free(axis_out);
        free(pool_color); free(pool_shape); free(pool_face);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -4;
    }

    for (int i = 0; i < nc; ++i) {
        if (reconstruct_color_cell(&m->cells[color_idx[i]],
                                   H, W, NF, NF_LOW,
                                   pool_color + (size_t)i * per_img,
                                   amp_pad, phase_pad, axis_out) != 0) {
            /* Wrong-shape cell: mark it neutral so the scorer ignores it. */
            float *p = pool_color + (size_t)i * per_img;
            for (size_t k = 0; k < per_img; ++k) p[k] = 0.94f;
        }
    }
    for (int i = 0; i < ns; ++i) {
        if (reconstruct_shape_cell(&m->cells[shape_idx[i]],
                                   H, W, NF,
                                   pool_shape + (size_t)i * per_img,
                                   amp_pad, phase_pad, axis_out) != 0) {
            float *p = pool_shape + (size_t)i * per_img;
            for (size_t k = 0; k < per_img; ++k) p[k] = 0.94f;
        }
    }
    for (int i = 0; i < nf; ++i) {
        if (reconstruct_face_cell(&m->cells[face_idx[i]],
                                  H, W, NF, NF_LOW,
                                  pool_face + (size_t)i * per_img,
                                  amp_pad, phase_pad, axis_out) != 0) {
            float *p = pool_face + (size_t)i * per_img;
            for (size_t k = 0; k < per_img; ++k) p[k] = 0.0f;
        }
    }

    /* Seed-shuffled visit order for the candidate pools. The pool
     * itself stays the same; only tie-breaking depends on order. */
    uint32_t rng = seed ? seed : 0xC0FFEE11u;
    int c_order[MAX_CANDS], s_order[MAX_CANDS];
    for (int i = 0; i < nc; ++i) c_order[i] = i;
    for (int i = 0; i < ns; ++i) s_order[i] = i;
    if (nc > 1) rng_shuffle(c_order, nc, &rng);
    if (ns > 1) rng_shuffle(s_order, ns, &rng);

    /* Cumulative bandit-style scores. */
    float c_scores[MAX_CANDS];
    float s_scores[MAX_CANDS];
    for (int i = 0; i < MAX_CANDS; ++i) c_scores[i] = s_scores[i] = 0.0f;

    /* prev_row carries the previous winning composition into the next
     * row's continuity term. Starts at the flat 0.94 gray of the blank
     * canvas so row 0 doesn't have to compete against zero. */
    float *prev_row = (float *)malloc((size_t)W * 3 * sizeof(float));
    if (!prev_row) {
        free(amp_pad); free(phase_pad); free(axis_out);
        free(pool_color); free(pool_shape); free(pool_face);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -5;
    }
    for (int x = 0; x < W * 3; ++x) prev_row[x] = 0.94f;

    /* Macro: pointer to row r of the i-th candidate inside pool `P`. */
    #define POOL_ROW(P, i, r) \
        ((P) + ((size_t)(i) * per_img + (size_t)(r) * (size_t)W * 3u))

    for (int r = 0; r < H; ++r) {
        int   best_ci = -1, best_si = -1;
        float best_total = -1e30f;
        const float *win_color = NULL, *win_shape = NULL;

        if (nc == 0 && ns == 0) {
            /* No competitors at all — leave win_color / win_shape NULL.
             * The radio stage runs on a pure spectral-noise target. */
        } else if (nc == 0) {
            for (int j = 0; j < ns; ++j) {
                const float *sr = POOL_ROW(pool_shape, s_order[j], r);
                float cont  = -row_mse_rgb(sr, prev_row, W);
                float bonus = logf(s_scores[s_order[j]] + 1.0f);
                float total = cont + bonus;
                if (total > best_total) {
                    best_total = total; best_si = s_order[j];
                    win_shape = sr;
                }
            }
        } else if (ns == 0) {
            for (int i = 0; i < nc; ++i) {
                const float *cr = POOL_ROW(pool_color, c_order[i], r);
                float cont  = -row_mse_rgb(cr, prev_row, W);
                float bonus = logf(c_scores[c_order[i]] + 1.0f);
                float total = cont + bonus;
                if (total > best_total) {
                    best_total = total; best_ci = c_order[i];
                    win_color = cr;
                }
            }
        } else {
            for (int i = 0; i < nc; ++i) {
                const float *cr = POOL_ROW(pool_color, c_order[i], r);
                float cont    = -row_mse_rgb(cr, prev_row, W);
                float c_bonus = logf(c_scores[c_order[i]] + 1.0f);
                for (int j = 0; j < ns; ++j) {
                    const float *sr = POOL_ROW(pool_shape, s_order[j], r);
                    float agree   = -row_mse_rgb(cr, sr, W);
                    float s_bonus = logf(s_scores[s_order[j]] + 1.0f);
                    float total   = agree + cont + c_bonus + s_bonus;
                    if (total > best_total) {
                        best_total = total;
                        best_ci = c_order[i]; best_si = s_order[j];
                        win_color = cr; win_shape = sr;
                    }
                }
            }
        }

        /* Bump winner scores. */
        if (best_ci >= 0) c_scores[best_ci] += 1.0f;
        if (best_si >= 0) s_scores[best_si] += 1.0f;

        /* Update prev_row from the composed winning row (color tinted
         * by shape brightness). This is what feeds the next row's
         * continuity term — the radio stage rebuilds the actual pixels
         * later from the winners' spectra. */
        for (int x = 0; x < W; ++x) {
            float c0 = win_color ? win_color[x * 3 + 0] : 0.94f;
            float c1 = win_color ? win_color[x * 3 + 1] : 0.94f;
            float c2 = win_color ? win_color[x * 3 + 2] : 0.94f;
            float s0 = win_shape ? win_shape[x * 3 + 0] : c0;
            float s1 = win_shape ? win_shape[x * 3 + 1] : c1;
            float s2 = win_shape ? win_shape[x * 3 + 2] : c2;
            float c_g = (c0 + c1 + c2) * (1.0f / 3.0f);
            float s_g = (s0 + s1 + s2) * (1.0f / 3.0f);
            float k = s_g / (c_g + 1e-6f);
            if (k < 0.7f) k = 0.7f;
            if (k > 1.3f) k = 1.3f;
            float v0 = c0 * k, v1 = c1 * k, v2 = c2 * k;
            if (v0 < 0.0f) v0 = 0.0f; else if (v0 > 1.0f) v0 = 1.0f;
            if (v1 < 0.0f) v1 = 0.0f; else if (v1 > 1.0f) v1 = 1.0f;
            if (v2 < 0.0f) v2 = 0.0f; else if (v2 > 1.0f) v2 = 1.0f;
            prev_row[x * 3 + 0] = v0;
            prev_row[x * 3 + 1] = v1;
            prev_row[x * 3 + 2] = v2;
        }
    }
    free(prev_row);

    /* Final winners (in pool space). FACE doesn't get its own sculpt
     * round — it acts as the high-frequency residual the radio mixes
     * into the mid band — so we just take the best-matching FACE cell
     * (index 0 of the pool, which is the closest tie-class member). */
    int winner_c = (nc > 0) ? argmax_f(c_scores, nc) : -1;
    int winner_s = (ns > 0) ? argmax_f(s_scores, ns) : -1;
    int winner_f = (nf > 0) ? 0 : -1;

    free(amp_pad); free(phase_pad); free(axis_out);
    /* pool_color / pool_shape / pool_face are still needed by Stage 2
     * (target assembly RFFTs the winners), so keep them alive. */

    /* ── Stage 2: Radio — assemble FFT targets, init noise, tune ── */

    if (sss_image_alloc(out, out_h, out_w) != 0) {
        free(pool_color); free(pool_shape); free(pool_face);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -6;
    }
    size_t HW = (size_t)H * (size_t)W;

    rng = seed ? seed : 0xC0FFEE11u;
    float *row_in    = (float *)malloc((size_t)W  * sizeof(float));
    float *row_amp   = (float *)malloc((size_t)NF * sizeof(float));
    float *row_phase = (float *)malloc((size_t)NF * sizeof(float));
    float *row_out   = (float *)malloc((size_t)W  * sizeof(float));
    float *col_in    = (float *)malloc((size_t)H  * sizeof(float));
    float *col_amp   = (float *)malloc((size_t)NF * sizeof(float));
    float *col_phase = (float *)malloc((size_t)NF * sizeof(float));
    float *col_out   = (float *)malloc((size_t)H  * sizeof(float));

    /* Target spectrograms.
     *   tgt_row_amp / phase  layout: [r * 3 * NF + c * NF + k]
     *   tgt_col_amp / phase  layout: [x * 3 * NF + c * NF + k]
     */
    float *tgt_row_amp   = (float *)malloc((size_t)H * 3 * NF * sizeof(float));
    float *tgt_row_phase = (float *)malloc((size_t)H * 3 * NF * sizeof(float));
    float *tgt_col_amp   = (float *)malloc((size_t)W * 3 * NF * sizeof(float));
    float *tgt_col_phase = (float *)malloc((size_t)W * 3 * NF * sizeof(float));

    if (!row_in || !row_amp || !row_phase || !row_out
     || !col_in || !col_amp || !col_phase || !col_out
     || !tgt_row_amp || !tgt_row_phase || !tgt_col_amp || !tgt_col_phase) {
        free(row_in); free(row_amp); free(row_phase); free(row_out);
        free(col_in); free(col_amp); free(col_phase); free(col_out);
        free(tgt_row_amp); free(tgt_row_phase);
        free(tgt_col_amp); free(tgt_col_phase);
        free(pool_color); free(pool_shape); free(pool_face);
        sss_image_free(out);
        for (int i = 0; i < ntok; ++i) free(tokens[i]);
        return -7;
    }

    /* Build row targets per (r, c, k) from winner reconstructions:
     *   k < NF_LOW                      → COLOR amp + phase
     *   NF_LOW ≤ k < 2·NF_LOW & FACE    → 0.6·SHAPE + 0.4·FACE, FACE phase
     *   else                            → SHAPE amp + phase
     * SHAPE is gray-broadcast so its three channels carry identical
     * data — the per-channel split is preserved for layout symmetry
     * with the column-tune target. */
    {
        float ca[64], cp[64], sa[64], sp[64], fa[64], fp[64];
        /* NF caps at W/2 + 1; sss_train.py uses W ≤ 128 so NF ≤ 65.
         * The 64-slot stack arrays cover the typical 64×64 / 128×128
         * cases. Fall through to malloc if a future caller crosses 64. */
        int use_heap = (NF > 64);
        float *ha = NULL, *hp = NULL, *hsa = NULL, *hsp = NULL, *hfa = NULL, *hfp = NULL;
        if (use_heap) {
            ha  = (float *)malloc((size_t)NF * sizeof(float));
            hp  = (float *)malloc((size_t)NF * sizeof(float));
            hsa = (float *)malloc((size_t)NF * sizeof(float));
            hsp = (float *)malloc((size_t)NF * sizeof(float));
            hfa = (float *)malloc((size_t)NF * sizeof(float));
            hfp = (float *)malloc((size_t)NF * sizeof(float));
            if (!ha || !hp || !hsa || !hsp || !hfa || !hfp) {
                free(ha); free(hp); free(hsa); free(hsp); free(hfa); free(hfp);
                free(row_in); free(row_amp); free(row_phase); free(row_out);
                free(col_in); free(col_amp); free(col_phase); free(col_out);
                free(tgt_row_amp); free(tgt_row_phase);
                free(tgt_col_amp); free(tgt_col_phase);
                free(pool_color); free(pool_shape); free(pool_face);
                sss_image_free(out);
                for (int i = 0; i < ntok; ++i) free(tokens[i]);
                return -8;
            }
        }
        float *bca  = use_heap ? ha  : ca;
        float *bcp  = use_heap ? hp  : cp;
        float *bsa  = use_heap ? hsa : sa;
        float *bsp  = use_heap ? hsp : sp;
        float *bfa  = use_heap ? hfa : fa;
        float *bfp  = use_heap ? hfp : fp;
        for (int k = 0; k < NF; ++k) {
            bca[k] = bcp[k] = 0.0f;
            bsa[k] = bsp[k] = 0.0f;
            bfa[k] = bfp[k] = 0.0f;
        }

        const float *cimg = (winner_c >= 0) ? pool_color + (size_t)winner_c * per_img : NULL;
        const float *simg = (winner_s >= 0) ? pool_shape + (size_t)winner_s * per_img : NULL;
        const float *fimg = (winner_f >= 0) ? pool_face  + (size_t)winner_f * per_img : NULL;

        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (cimg) {
                    for (int x = 0; x < W; ++x)
                        row_in[x] = cimg[((size_t)r * W + x) * 3 + c];
                    sss_rfft(row_in, W, bca, bcp);
                }
                if (simg) {
                    for (int x = 0; x < W; ++x)
                        row_in[x] = simg[((size_t)r * W + x) * 3 + c];
                    sss_rfft(row_in, W, bsa, bsp);
                }
                if (fimg) {
                    for (int x = 0; x < W; ++x)
                        row_in[x] = fimg[((size_t)r * W + x) * 3 + c];
                    sss_rfft(row_in, W, bfa, bfp);
                }
                for (int k = 0; k < NF; ++k) {
                    float ta, tp;
                    if (k < NF_LOW && cimg) {
                        ta = bca[k]; tp = bcp[k];
                    } else if (k < NF_LOW * 2 && fimg && simg) {
                        ta = bsa[k] * 0.6f + bfa[k] * 0.4f * detail;
                        tp = bfp[k];
                    } else if (simg) {
                        ta = bsa[k]; tp = bsp[k];
                    } else if (fimg && k >= NF_LOW) {
                        ta = bfa[k] * detail; tp = bfp[k];
                    } else if (cimg) {
                        ta = bca[k]; tp = bcp[k];
                    } else {
                        ta = 0.0f; tp = 0.0f;
                    }
                    size_t idx = ((size_t)r * 3 + (size_t)c) * (size_t)NF + (size_t)k;
                    tgt_row_amp[idx]   = ta;
                    tgt_row_phase[idx] = tp;
                }
            }
        }

        /* Column targets — pure SHAPE, per channel (identical across
         * channels because shape was gray-broadcast at reconstruction). */
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                if (simg) {
                    for (int y = 0; y < H; ++y)
                        col_in[y] = simg[((size_t)y * W + x) * 3 + c];
                    sss_rfft(col_in, H, bsa, bsp);
                } else {
                    for (int k = 0; k < NF; ++k) { bsa[k] = 0.0f; bsp[k] = 0.0f; }
                }
                for (int k = 0; k < NF; ++k) {
                    size_t idx = ((size_t)x * 3 + (size_t)c) * (size_t)NF + (size_t)k;
                    tgt_col_amp[idx]   = bsa[k];
                    tgt_col_phase[idx] = bsp[k];
                }
            }
        }

        if (use_heap) {
            free(ha); free(hp); free(hsa); free(hsp); free(hfa); free(hfp);
        }
    }

    /* Spectral noise init — radio static. Build each row of each
     * channel as the inverse-FFT of random amp + random phase. The
     * tuning loop then nudges this toward the target spectra one
     * α-step at a time, the way you tune a radio dial.
     *
     * rng_uniform returns [-1, 1); map into [0, 0.3) for amp and
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
    for (size_t i = 0; i < HW * 3; ++i) {
        if (out->data[i] < 0.0f) out->data[i] = 0.0f;
        else if (out->data[i] > 1.0f) out->data[i] = 1.0f;
    }

    /* Steps-aware cooling: more steps → smaller per-step nudges, so a
     * 60 / 120 step run converges into a cleaner image than the 24-step
     * preview. AM (coarse) → FM (fine) → digital (snap). */
    float base_alpha = (steps <= 30) ? 0.95f
                     : (steps <= 80) ? 0.50f
                     :                 0.30f;
    float decay = base_alpha * 0.68f;

    for (int step = 0; step < steps; ++step) {
        float t = (steps > 1) ? (float)step / (float)(steps - 1) : 0.0f;
        float alpha       = base_alpha - decay * t;
        if (alpha < 0.05f) alpha = 0.05f;
        float phase_alpha = alpha * 0.5f;

        /* ── Row tune: pull each row toward the assembled row target. ── */
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < 3; ++c) {
                for (int x = 0; x < W; ++x) {
                    row_in[x] = out->data[((size_t)r * W + x) * 3 + c];
                }
                sss_rfft(row_in, W, row_amp, row_phase);
                for (int k = 0; k < NF; ++k) {
                    size_t idx = ((size_t)r * 3 + (size_t)c) * (size_t)NF + (size_t)k;
                    float ta = tgt_row_amp[idx];
                    float tp = tgt_row_phase[idx];
                    row_amp[k] = alpha * ta + (1.0f - alpha) * row_amp[k];
                    float dph = tp - row_phase[k];
                    while (dph >  (float)M_PI) dph -= 2.0f * (float)M_PI;
                    while (dph < -(float)M_PI) dph += 2.0f * (float)M_PI;
                    row_phase[k] += phase_alpha * dph;
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

        /* ── Column tune: per-channel column FFT toward SHAPE target.
         *   No more gray-detour — shape was gray-broadcast at
         *   reconstruction time, so per-channel tuning preserves chroma
         *   and keeps the structural pull aligned with the row stage. */
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < H; ++y) {
                    col_in[y] = out->data[((size_t)y * W + x) * 3 + c];
                }
                sss_rfft(col_in, H, col_amp, col_phase);
                for (int k = 0; k < NF; ++k) {
                    size_t idx = ((size_t)x * 3 + (size_t)c) * (size_t)NF + (size_t)k;
                    float ta = tgt_col_amp[idx];
                    float tp = tgt_col_phase[idx];
                    col_amp[k] = alpha * ta + (1.0f - alpha) * col_amp[k];
                    float dph = tp - col_phase[k];
                    while (dph >  (float)M_PI) dph -= 2.0f * (float)M_PI;
                    while (dph < -(float)M_PI) dph += 2.0f * (float)M_PI;
                    col_phase[k] += phase_alpha * dph;
                }
                sss_irfft(col_amp, col_phase, NF, H, col_out);
                for (int y = 0; y < H; ++y) {
                    out->data[((size_t)y * W + x) * 3 + c] = col_out[y];
                }
            }
        }
        for (size_t i = 0; i < HW * 3; ++i) {
            if (out->data[i] < 0.0f) out->data[i] = 0.0f;
            else if (out->data[i] > 1.0f) out->data[i] = 1.0f;
        }
    }

    free(row_in); free(row_amp); free(row_phase); free(row_out);
    free(col_in); free(col_amp); free(col_phase); free(col_out);
    free(tgt_row_amp); free(tgt_row_phase);
    free(tgt_col_amp); free(tgt_col_phase);
    free(pool_color); free(pool_shape); free(pool_face);
    for (int i = 0; i < ntok; ++i) free(tokens[i]);

    #undef POOL_ROW
    return 0;
}
