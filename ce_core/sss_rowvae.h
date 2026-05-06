/* sss_rowvae.h — Spectrogram-based image generation engine.
 *
 * Stores "how to draw" as row/column FFT amplitudes (a spectrogram),
 * not pixels. Generation starts from noise and is shaped by the
 * spectrogram patterns associated with each morpheme.
 *
 *   COLOR cell  → low-frequency Y spectrogram per RGB channel
 *   SHAPE cell  → X spectrogram (column FFT) of grayscale structure
 *   FACE  cell  → high-frequency Y spectrogram per RGB channel
 *
 * The 256-grid fingerprint is a search key only (morpheme bytes →
 * 256 floats). Generation reads cells out of the model and
 * inverse-FFTs them into an image.
 */
#ifndef SSS_ROWVAE_H
#define SSS_ROWVAE_H

#include <stdint.h>
#include <stddef.h>

/* CEUnit is the 256-grid morpheme cell. Pulled in here so SSSCell
 * can carry a `ce_key` — the label's ce_feed-fingerprint, which is
 * what find_cells_per_token compares against at generate time.
 * ce_core must NOT include this header back (would cycle). */
#include "ce_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SSS_MAGIC      0x53535839u   /* "SSX9" little-endian (v9 ce_key) */
#define SSS_VERSION    9u
#define SSS_FP_LEN     256
#define SSS_LABEL_MAX  64

typedef enum {
    SSS_CE_COLOR = 1,
    SSS_CE_SHAPE = 2,
    SSS_CE_FACE  = 3,
} SSSCellType;

typedef struct {
    uint32_t type;                 /* SSSCellType */
    char     label[SSS_LABEL_MAX];
    CEUnit   ce_key;               /* 256-grid morpheme link — search key */
    float    fp[SSS_FP_LEN];       /* legacy byte-histogram fp; kept for
                                    * backward-compat callers (sss_search,
                                    * fp_distance). New search path uses
                                    * ce_key + ce_distance. */
    uint32_t amp_len;
    float   *amp;
    uint32_t phase_len;
    float   *phase;
} SSSCell;

typedef struct {
    uint32_t height, width;        /* must be equal (square) */
    uint32_t nf;                   /* width/2 + 1 */
    uint32_t nf_low;               /* nf/3, low-frequency cutoff */
    uint32_t num_cells;
    SSSCell *cells;
} SSSModel;

typedef struct {
    int    height, width;
    float *data;                   /* H * W * 3, RGB float [0,1] */
} SSSImage;

/* ── Fingerprint (must match the Python trainer) ─────────────── */
void sss_fingerprint(const char *text, float *fp);

/* ── Model I/O ──────────────────────────────────────────────── */
int  sss_model_load(const char *path, SSSModel *out);
void sss_model_free(SSSModel *m);

/* Search the model for the best matching cell of the given type.
 * Returns cell index, or -1 if no cell of that type is below the
 * match threshold. */
int  sss_search(const SSSModel *m, uint32_t type, const float *fp);

/* Forward real FFT: rebuilds the (amp[0..nf-1], phase[0..nf-1])
 * half-spectrum of an N-length real signal x. Matches numpy.fft.rfft
 * convention (no scaling on the forward pass). nf must equal N/2+1. */
void sss_rfft(const float *x, int N, float *amp, float *phase);

/* Real inverse FFT: rebuilds an N-length real signal from the
 * (amp[0..nf-1], phase[0..nf-1]) half-spectrum produced by a real
 * FFT (numpy.fft.rfft). Implemented from scratch — no external
 * FFT library. */
void sss_irfft(const float *amp, const float *phase, int nf, int N, float *out);

/* Image helpers. */
int  sss_image_alloc(SSSImage *img, int h, int w);
void sss_image_free(SSSImage *img);
int  sss_image_save_ppm(const SSSImage *img, const char *path);

/* End-to-end iterative generation:
 *
 *     image = noise
 *     for step in range(steps):
 *         measure  = rfft(image)
 *         error    = target - measure
 *         image    = irfft(measure + α(step) * error)
 *
 * The model's COLOR/SHAPE/FACE cells supply the targets, the prompt
 * picks the cells via 256-grid fingerprint, `seed` initialises the
 * noise PRNG (xorshift32), `detail` scales the high-frequency target
 * (must be > 0), and `steps` is a positive integer iteration count
 * (24 is the typical schedule; the library defensively falls back to
 * 24 if a non-positive value slips through, but new callers should
 * pass an explicit positive value). */
int  sss_generate(const SSSModel *m,
                  const char     *prompt,
                  uint32_t        seed,
                  float           detail,
                  int             steps,
                  SSSImage       *out);

#ifdef __cplusplus
}
#endif

#endif /* SSS_ROWVAE_H */
