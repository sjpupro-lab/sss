/* ce_feed_image.h — image-modality encoder.
 *
 * Encodes a single 8x8 RGBA block (256 bytes, row-major) into one CEUnit
 * (64 bytes). Distinct from ce_feed (text path), which mixes channels and
 * applies position-mixing salt — that path is meant for byte streams of
 * arbitrary length and is unsuitable for image semantics.
 *
 * Layout produced by ce_feed_image:
 *
 *   inc.<R|G|B|A>.plus  : per-channel raw sum across the 8x8 block.
 *                         max value = 64 * 255 = 16320 per channel,
 *                         which is the constant ce_decode_image_block
 *                         already rescales against.
 *
 *   inc.<R|G|B|A>.minus : reserved (zero). Kept clean so that ce_distance
 *                         on the inc half compares "absolute color".
 *
 *   dec.<R|G|B|A>.plus  : per-channel gradient/edge magnitudes:
 *                            byte 0 = |LR|   left-to-right
 *                            byte 1 = |UD|   top-to-bottom
 *                            byte 2 = |D1|   main diagonal (TL -> BR)
 *                            byte 3 = |D2|   anti-diagonal  (TR -> BL)
 *                         Magnitudes are in [0, 255] (clipped).
 *
 *   dec.<R|G|B|A>.minus : per-channel direction signs:
 *                            byte k = 0xFF when the corresponding
 *                                     directional difference is >= 0,
 *                                     0x00 when negative.
 *
 * Constraints honoured:
 *   - No cross-channel mixing in the inc half (R never bleeds into G).
 *   - No mean-based smoothing — the inc half stores raw sums; gradients
 *     are computed from quadrant *sums* and converted to per-pixel-equivalent
 *     differences without any low-pass step.
 *   - Deterministic: identical input bytes always produce the identical
 *     CEUnit, regardless of build flags.
 *   - The natural 4-byte chain carry (R chain byte 0 -> R chain byte 1
 *     when crossing 256) is preserved by carry_add. Cross-channel cascade
 *     (R chain overflow into G chain) is not triggered for a single 8x8
 *     block (max sum 16320 << 2^32) but remains available for higher-level
 *     aggregation steps that repeatedly feed the same CEUnit.
 *
 * The companion decoder ce_decode_image_block reads only inc.<ch>.plus
 * and rescales by 64*255 — so this encoder is round-trip-compatible for
 * uniform-color blocks, and the dec half carries the per-block detail
 * the decoder will need once gradient-aware decode lands.
 */
#ifndef CE_FEED_IMAGE_H
#define CE_FEED_IMAGE_H

#include "ce_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_IMAGE_BLOCK_PX    8
#define CE_IMAGE_BLOCK_BYTES (CE_IMAGE_BLOCK_PX * CE_IMAGE_BLOCK_PX * 4) /* 256 */

/* Encode an 8x8 RGBA block (256 bytes, row-major) into a CEUnit.
 * The input is read via [r,g,b,a, r,g,b,a, ...] across 64 pixels.
 * The CEUnit is reset before encoding. */
void ce_feed_image(CEUnit *u, const uint8_t *rgba_block_8x8);

#ifdef __cplusplus
}
#endif
#endif /* CE_FEED_IMAGE_H */
