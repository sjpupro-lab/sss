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

#ifdef __cplusplus
}
#endif
#endif /* CE_DECODE_H */
