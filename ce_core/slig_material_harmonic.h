/* slig_material_harmonic.h — auto-learned per-cell material descriptor.
 *
 *  Each CE Cell carries 8 bytes of material parameters (in dec.A) that
 *  describe the texture/surface of the image region the cell came from.
 *  Extraction is pure integer (subtraction + comparison) on an 8×8
 *  pixel block — no float, no presets needed. The block is analysed
 *  once during decompose; the cell carries the result for the rest of
 *  its life.
 *
 *  Storage layout in dec.A (8 bytes total):
 *    plus[0..3]:  roughness, anisotropy, dir_angle, h_strength
 *    minus[0..3]: h_decay,   pore,        specular,  grain
 *
 *  Flag: inc.R.minus[3] |= SLIG_FLAG_HAS_MATERIAL when material is
 *  written. Render / spectrum-weight code checks this before
 *  interpreting dec.A as audio_bins/audio_amps.
 *
 *  Skipped on Cb/Cr cells — chroma marker (audio_bins[0]=1/2) and
 *  block-falloff value (audio_amps[0]) live in dec.A and must stay
 *  intact for Phase 4 dispatch.
 */
#ifndef SLIG_MATERIAL_HARMONIC_H
#define SLIG_MATERIAL_HARMONIC_H

#include <stdint.h>
#include "ce_core.h"
#include "slig_signal.h"   /* SligCellSet (used by overlay API) */

#ifdef __cplusplus
extern "C" {
#endif

/* Bit 4 of sig.flags / cell->inc.R.minus[3]. Disjoint from existing
 * SLIG_FLAG_NEGATIVE (0x01), SLIG_FLAG_HAS_TRIGGER (0x02),
 * SLIG_FLAG_AUDIO_LINK (0x04) defined in slig_signal.h. */
#define SLIG_FLAG_HAS_MATERIAL 0x08

typedef enum {
    SLIG_MAT_GENERIC = 0,
    SLIG_MAT_SKIN,
    SLIG_MAT_FABRIC,
    SLIG_MAT_METAL,
    SLIG_MAT_WOOD,
    SLIG_MAT_COUNT
} SligMatPreset;

typedef struct {
    uint8_t roughness;     /* sum of neighbour |Δ| / 56               */
    uint8_t anisotropy;    /* (max_dir − min_dir) / (max_dir + 1)     */
    uint8_t dir_angle;     /* 0 / 32 / 64 / 96 (H / 45 / V / 135)     */
    uint8_t h_strength;    /* high-freq energy / low-freq energy      */
    uint8_t h_decay;       /* (255 − rough·150/255) + 50              */
    uint8_t pore;          /* outlier pixel count × 7 (clamped)        */
    uint8_t specular;      /* peak − mean                              */
    uint8_t grain;         /* (anisotropy × roughness) >> 8           */
} SligMaterialTick;

/* Hand-tuned defaults for known materials. Useful as a fallback when
 * no analysis source is available, and as a sanity check for the
 * analyzer (preset values bracket the analyzer output range). */
void slig_mat_preset(SligMaterialTick *out, SligMatPreset preset);

/* Extract material from an 8×8 block. `block` points at the top-left
 * pixel; `stride` is the row stride in bytes (= image width for
 * planar input). */
void slig_mat_analyze_block(SligMaterialTick *out,
                            const uint8_t    *block,
                            int               stride);

/* Write material to cell->dec.A and OR the HAS_MATERIAL flag into
 * inc.R.minus[3]. Must be called AFTER slig_pack (which writes dec.A
 * with audio_bins/audio_amps). */
void slig_mat_to_cell(CEUnit *cell, const SligMaterialTick *mat);

/* Read material out of cell->dec.A. Caller should check slig_mat_has
 * first; otherwise the bytes are interpreted regardless. */
void slig_mat_from_cell(SligMaterialTick *out, const CEUnit *cell);

/* Returns 1 iff cell->inc.R.minus[3] carries SLIG_FLAG_HAS_MATERIAL. */
int  slig_mat_has(const CEUnit *cell);

/* Mat-S4 — render-time material overlay.
 *
 * Apply a deterministic per-pixel texture on top of a rendered
 * grayscale luminance panel, driven by the average material descriptor
 * across `cells` that carry HAS_MATERIAL. Cells without material data
 * are ignored. No-op if no cell carries material info.
 *
 * The overlay is a 3-component pixel hash:
 *   - roughness drives a band-limited high-frequency micro-texture
 *   - pore drives sparse dark spots (occasional outliers)
 *   - grain drives a low-amplitude directional shimmer
 *
 * The overlay is fully deterministic: same input cells + same panel →
 * same output panel. No RNG state, no time dependence — different
 * materials (different signature) produce different patterns, but each
 * material's pattern is reproducible.
 *
 * `panel` is in/out, `dim`×`dim`, row-major uint8 luminance. */
void slig_apply_material_overlay(uint8_t *panel, int dim,
                                 const SligCellSet *cells);

#ifdef __cplusplus
}
#endif
#endif /* SLIG_MATERIAL_HARMONIC_H */
