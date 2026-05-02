/* sss_rowvae.h — Sculpt-based image generation engine.
 *
 * Each CE cell stores a *pool of training images* for one attribute
 * (color / shape / face). Generation is row-by-row: at each row the
 * engine looks at every candidate's row pattern, picks the pair that
 * best agrees with itself and the previous row, composes one output
 * row from the chosen color × shape pair, and accumulates a score
 * that biases later rows toward the same candidate. There is no
 * spectrogram, no FFT, no noise base — the canvas starts as a flat
 * neutral gray and is sculpted row by row.
 *
 *   COLOR cell  → pool of training images whose color label matches
 *   SHAPE cell  → pool of training images whose shape label matches
 *   FACE  cell  → pool of training images whose face label matches
 *
 * The 256-grid fingerprint is a search key only (morpheme bytes →
 * 256 floats). It indexes which cell to load, never the pixels.
 */
#ifndef SSS_ROWVAE_H
#define SSS_ROWVAE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SSS_MAGIC      0x53535839u   /* "SSX9" little-endian */
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
    float    fp[SSS_FP_LEN];       /* fingerprint = search key */
    uint32_t num_imgs;             /* number of training images in pool */
    float   *imgs;                 /* num_imgs × H × W × C floats, [0,1] */
} SSSCell;

typedef struct {
    uint32_t height, width;        /* must be equal (square) */
    uint32_t channels;             /* always 3 */
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

/* Image helpers. */
int  sss_image_alloc(SSSImage *img, int h, int w);
void sss_image_free(SSSImage *img);
int  sss_image_save_ppm(const SSSImage *img, const char *path);

/* End-to-end generation. `prompt` is split on whitespace into
 * morphemes; each morpheme is matched against COLOR/SHAPE/FACE
 * cells. `seed` permutes the candidate iteration order so the same
 * prompt can produce slight variations. `detail` scales the
 * face-row contribution (1.0 = neutral, >1 = sharper). */
int  sss_generate(const SSSModel *m,
                  const char     *prompt,
                  uint32_t        seed,
                  float           detail,
                  SSSImage       *out);

#ifdef __cplusplus
}
#endif

#endif /* SSS_ROWVAE_H */
