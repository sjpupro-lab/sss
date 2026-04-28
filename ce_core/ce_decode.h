/* ce_decode.h — Step 15: latent grid -> image / text. */
#ifndef CE_DECODE_H
#define CE_DECODE_H

#include "ce_core.h"
#include "ce_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_IMAGE_W 256
#define CE_IMAGE_H 256
#define CE_BLOCK   8   /* CE_GRID_W * CE_BLOCK = 256 */

typedef struct { uint8_t r, g, b, a; } CEPixel;

typedef struct {
    CEPixel pixels[CE_IMAGE_W * CE_IMAGE_H];
    int width, height;
} CEImage;

/* Map ce_read(R/G/B/A) into a 256x256 RGBA buffer. Bilinear interpolation
 * across the 8x8 block of each cell to smooth boundaries. */
void ce_decode_image(CEImage *out, const CELatentGrid *z);

/* Decode latent into an ASCII text stream. Each cell yields one byte
 * derived from its R channel signed projection -> printable ASCII. */
void ce_decode_text(uint8_t *out, uint32_t *out_len,
                    const CELatentGrid *z);

/* Decode a single CEUnit into an 8x8 RGBA block (256 bytes, row-major).
 *
 * Image-specific normalization: read each channel from the inc.plus carry
 * chain in base 256, rescale into [0, 255] against the per-block maximum
 * 64 * 255 = 16320, and clamp. Distinct from ce_read (which mixes
 * inc/dec/plus/minus into a signed projection) — this routine does NOT
 * mix channels and does NOT mix halves. The 8x8 output is filled with
 * the per-channel value; per-pixel detail must come from the surrounding
 * grid (handled by callers), not from a single CEUnit.
 */
void ce_decode_image_block(uint8_t *out_rgba_8x8, const CEUnit *u);

/* Decode 4 quadrant CEUnits (TL=0, TR=1, BL=2, BR=3) into a 16x16 RGBA
 * patch (1024 bytes, row-major). Each quadrant is decoded via
 * ce_decode_image_block (8x8) and pasted into its sub-region of the
 * 16x16 output. Inverse of ce_feed_image_16 for uniform-colour input. */
void ce_decode_image_block_16(uint8_t *out_rgba_16x16,
                              const CEUnit q[4]);

#ifdef __cplusplus
}
#endif
#endif /* CE_DECODE_H */
