/* ce_feed_image.c — image-modality encoder.
 *
 * See ce_feed_image.h for the byte layout produced.
 *
 * The block is treated as four 4x4 quadrants:
 *
 *      x=0..3   x=4..7
 *    +--------+--------+
 *  y |   TL   |   TR   |   y=0..3
 * 0..3+--------+--------+
 *    |   BL   |   BR   |   y=4..7
 *  y +--------+--------+
 * 4..7
 *
 * Quadrant sums (per channel) drive both:
 *   - the absolute color stored in inc.<ch>.plus (TL+TR+BL+BR == full sum)
 *   - the four directional gradients stored in dec.<ch>.{plus,minus}
 *
 * The "rgba carry tick" semantic is preserved by reusing the same 4-byte
 * carry chain primitive used by ce_feed: each pixel's channel value is
 * added to its channel's plus chain, and chain byte 0 overflowing 255
 * naturally cascades into chain byte 1, etc. For a single 8x8 block the
 * total never reaches the 32-bit chain capacity, so cross-channel cascade
 * (R chain wrap -> G chain tick) is not triggered here — but the chain
 * mechanic is the same as in ce_feed, so callers that aggregate many
 * blocks into a single CEUnit (e.g. a coarser slot keyframe) still
 * inherit the odometer behaviour described in the design.
 */
#include "ce_feed_image.h"
#include "ce_core.h"
#include <string.h>

/* Add `v` to a 4-byte little-endian carry chain (mirrors carry_add in
 * ce_core.c — kept private here to avoid exporting it). */
static void chain_add(uint8_t chain[4], uint32_t v) {
    uint32_t acc = (uint32_t)chain[0]
                 | ((uint32_t)chain[1] << 8)
                 | ((uint32_t)chain[2] << 16)
                 | ((uint32_t)chain[3] << 24);
    acc += v;
    chain[0] = (uint8_t)(acc & 0xFF);
    chain[1] = (uint8_t)((acc >> 8) & 0xFF);
    chain[2] = (uint8_t)((acc >> 16) & 0xFF);
    chain[3] = (uint8_t)((acc >> 24) & 0xFF);
}

static uint8_t clip_u8_abs(int32_t v) {
    if (v < 0) v = -v;
    if (v > 255) v = 255;
    return (uint8_t)v;
}

void ce_feed_image(CEUnit *u, const uint8_t *rgba_block_8x8) {
    ce_init(u);
    if (!rgba_block_8x8) return;

    /* Per-channel quadrant sums (TL, TR, BL, BR), and total sums. */
    uint32_t qTL[4] = {0,0,0,0};
    uint32_t qTR[4] = {0,0,0,0};
    uint32_t qBL[4] = {0,0,0,0};
    uint32_t qBR[4] = {0,0,0,0};

    for (int y = 0; y < CE_IMAGE_BLOCK_PX; ++y) {
        for (int x = 0; x < CE_IMAGE_BLOCK_PX; ++x) {
            const uint8_t *p = rgba_block_8x8 + (size_t)(y * CE_IMAGE_BLOCK_PX + x) * 4u;
            uint32_t *q = (y < 4)
                ? ((x < 4) ? qTL : qTR)
                : ((x < 4) ? qBL : qBR);
            q[0] += p[0];
            q[1] += p[1];
            q[2] += p[2];
            q[3] += p[3];
        }
    }

    /* Absolute color: store full per-channel sum in inc.<ch>.plus.
     * Channels stay independent — no cross-channel writes here. */
    uint8_t *inc_plus[4] = {
        u->inc.R.plus, u->inc.G.plus, u->inc.B.plus, u->inc.A.plus
    };
    for (int c = 0; c < 4; ++c) {
        uint32_t total = qTL[c] + qTR[c] + qBL[c] + qBR[c];
        chain_add(inc_plus[c], total);
    }

    /* Directional gradients (per channel), normalised to per-pixel-equivalent
     * magnitudes by dividing each quadrant sum by 16 (its pixel count) before
     * differencing. This keeps the |gradient| in the same [0, 255] units as
     * raw pixel values, so the dec.plus byte is directly readable as an
     * 8-bit magnitude.
     *
     * The four directions:
     *   LR = (TR + BR)/2 - (TL + BL)/2   --  positive => brighter on the right
     *   UD = (BL + BR)/2 - (TL + TR)/2   --  positive => brighter at the bottom
     *   D1 = BR - TL                     --  positive => brighter at bottom-right
     *   D2 = BL - TR                     --  positive => brighter at bottom-left
     *
     * Quadrant means are sum/16. To halve the per-pair denominator we divide
     * the relevant sums by 32 for LR/UD and by 16 for D1/D2 before subtracting
     * (all integer arithmetic; no mean smoothing of pixel data). */
    uint8_t *dec_plus[4] = {
        u->dec.R.plus, u->dec.G.plus, u->dec.B.plus, u->dec.A.plus
    };
    uint8_t *dec_minus[4] = {
        u->dec.R.minus, u->dec.G.minus, u->dec.B.minus, u->dec.A.minus
    };

    for (int c = 0; c < 4; ++c) {
        int32_t TL = (int32_t)qTL[c];
        int32_t TR = (int32_t)qTR[c];
        int32_t BL = (int32_t)qBL[c];
        int32_t BR = (int32_t)qBR[c];

        /* (TR+BR)/32 - (TL+BL)/32  ==  (right_pair_mean - left_pair_mean) */
        int32_t lr = (TR + BR) / 32 - (TL + BL) / 32;
        int32_t ud = (BL + BR) / 32 - (TL + TR) / 32;
        int32_t d1 = BR / 16 - TL / 16;
        int32_t d2 = BL / 16 - TR / 16;

        dec_plus[c][0] = clip_u8_abs(lr);
        dec_plus[c][1] = clip_u8_abs(ud);
        dec_plus[c][2] = clip_u8_abs(d1);
        dec_plus[c][3] = clip_u8_abs(d2);

        dec_minus[c][0] = (lr >= 0) ? 0xFF : 0x00;
        dec_minus[c][1] = (ud >= 0) ? 0xFF : 0x00;
        dec_minus[c][2] = (d1 >= 0) ? 0xFF : 0x00;
        dec_minus[c][3] = (d2 >= 0) ? 0xFF : 0x00;
    }
}
