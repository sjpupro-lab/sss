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

#ifdef __cplusplus
extern "C" {
#endif

#define SSS_MAGIC      0x53535838u   /* "SSX8" little-endian */
#define SSS_VERSION    8u
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
    float    fp[SSS_FP_LEN];       /* fingerprint = search key */
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

/* Real inverse FFT: rebuilds an N-length real signal from the
 * (amp[0..nf-1], phase[0..nf-1]) half-spectrum produced by a real
 * FFT (numpy.fft.rfft). Implemented from scratch — no external
 * FFT library. */
void sss_irfft(const float *amp, const float *phase, int nf, int N, float *out);

/* Image helpers. */
int  sss_image_alloc(SSSImage *img, int h, int w);
void sss_image_free(SSSImage *img);
int  sss_image_save_ppm(const SSSImage *img, const char *path);

/* End-to-end generation. `prompt` is split on whitespace into
 * morphemes; each morpheme is matched against COLOR/SHAPE/FACE
 * cells. `seed` controls the noise PRNG. `detail` scales the
 * high-frequency contribution (1.0 = neutral, >1 = sharper). */
int  sss_generate(const SSSModel *m,
                  const char     *prompt,
                  uint32_t        seed,
                  float           detail,
                  SSSImage       *out);

#ifdef __cplusplus
}
#endif

#endif /* SSS_ROWVAE_H */
