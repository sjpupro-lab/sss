/* ce_tick.h — TickRGBA channel-split clock for CEStorage entries.
 *
 * The user-specified clock model: every CE generation event happens
 * at a 4-channel "time stamp" packed into a single uint32 with carry
 * chain.
 *
 *   R  = fastest tick — cell index within its (scale, channel) set
 *   G  = mid-rate tick — scale_level (0=COARSE, 1=MID, 2=FINE)
 *   B  = slow tick — render layer / channel (0=Y, 1=Cb, 2=Cr; or
 *         CE_TYPE_IMAGE / CE_TYPE_SLIG / CE_TYPE_RESIDUAL when sorting
 *         across modalities)
 *   A  = amplitude / residual strength (carries energy of the cell)
 *
 * Carry semantics (uint8 wrap, never float):
 *   ++R; if (R == 0) { ++G; if (G == 0) { ++B; if (B == 0) ++A; } }
 *
 * The helper mirrors slig_tick_math.h's `tick_add` exactly: R is the
 * LSB, A the MSB, packed little-endian. So `tick_add(plus, 1)` from
 * slig_tick_math is the same primitive — `ce_tick_step` is just the
 * `+1` specialisation that exposes the named channels.
 *
 * What this header provides:
 *   - TickRGBA struct
 *   - ce_tick_step      — carry-chain +1
 *   - ce_tick_compare   — strict ordering (B > G > R > A)
 *   - ce_tick_from_slig — storage entry of CE_TYPE_SLIG → TickRGBA
 *   - ce_tick_from_image — storage entry of CE_TYPE_IMAGE → TickRGBA
 *
 * No CEUnit byte layout is touched. This is a *derived* clock — the
 * R/G/B/A values are read from (slot, block_idx) of each entry and
 * its associated SligSignal energy, not stored in a new header byte.
 * Existing SLIG cells therefore keep their full DCT payload in
 * inc.{R,G,B,A}.plus[0..1] without conflict.
 */
#ifndef CE_TICK_H
#define CE_TICK_H

#include <stdint.h>
#include "ce_storage.h"
#include "ce_type.h"
#include "slig_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t r;  /* cell index within set (fastest)       */
    uint8_t g;  /* scale_level                           */
    uint8_t b;  /* channel / render layer                */
    uint8_t a;  /* amplitude / residual strength         */
} TickRGBA;

/* +1 carry step. Mirrors slig_tick_math `tick_add(plus, 1)`. */
static inline void ce_tick_step(TickRGBA *t) {
    if (++t->r == 0) {
        if (++t->g == 0) {
            if (++t->b == 0) {
                ++t->a;
            }
        }
    }
}

/* Pack TickRGBA into a uint32 (R = LSB byte). */
static inline uint32_t ce_tick_pack(TickRGBA t) {
    return  (uint32_t)t.r
         | ((uint32_t)t.g <<  8)
         | ((uint32_t)t.b << 16)
         | ((uint32_t)t.a << 24);
}

static inline TickRGBA ce_tick_unpack(uint32_t v) {
    TickRGBA t;
    t.r = (uint8_t)( v        & 0xFF);
    t.g = (uint8_t)((v >>  8) & 0xFF);
    t.b = (uint8_t)((v >> 16) & 0xFF);
    t.a = (uint8_t)((v >> 24) & 0xFF);
    return t;
}

/* Strict ordering: returns < 0 if a < b, > 0 if a > b, 0 if equal.
 * Most-significant channel first (B → G → R → A). The amplitude byte
 * is *last* in priority because it represents how strongly to apply
 * the cell, not when to fire it. */
static inline int ce_tick_compare(TickRGBA a, TickRGBA b) {
    if (a.b != b.b) return (a.b < b.b) ? -1 : 1;
    if (a.g != b.g) return (a.g < b.g) ? -1 : 1;
    if (a.r != b.r) return (a.r < b.r) ? -1 : 1;
    if (a.a != b.a) return (a.a < b.a) ? -1 : 1;
    return 0;
}

/* Read the audio_amps amplitude byte from a SLIG cell's signal. The
 * hybrid_vae encode path stores per-bin amplitudes in audio_amps[];
 * cell strength = max amp seen, which is what the renderer cares
 * about when choosing how heavily to mix this cell. */
static inline uint8_t ce_tick_amp_from_cell(const CEUnit *cell) {
    SligSignal sig;
    slig_unpack(&sig, cell);
    uint8_t amp = 0;
    for (int i = 0; i < 4; ++i) {
        if (sig.audio_amps[i] > amp) amp = sig.audio_amps[i];
    }
    /* Sigma scales 0..65535; scale to 0..255 for tick.A. */
    uint16_t sigma_byte = (uint16_t)(sig.sigma >> 8);
    if (sigma_byte > amp) amp = (uint8_t)sigma_byte;
    return amp;
}

/* CE_TYPE_SLIG entry → TickRGBA.
 *   slot      = scale_level * SLIG_NUM_CHANNELS + channel
 *   block_idx = cell index within set
 * so:
 *   tick.r = block_idx              (cell index inside set)
 *   tick.g = slot / SLIG_NUM_CHANNELS  (scale level 0..2)
 *   tick.b = slot % SLIG_NUM_CHANNELS  (channel 0..2)
 *   tick.a = audio_amps max from the cell
 */
static inline TickRGBA ce_tick_from_slig(const CEStorageEntry *e) {
    TickRGBA t = (TickRGBA){0, 0, 0, 0};
    if (!e || e->type != CE_TYPE_SLIG) return t;
    t.r = (uint8_t)(e->block_idx & 0xFF);
    t.g = (uint8_t)((e->slot / SLIG_NUM_CHANNELS) & 0xFF);
    t.b = (uint8_t)((e->slot % SLIG_NUM_CHANNELS) & 0xFF);
    t.a = ce_tick_amp_from_cell(&e->keyframe);
    return t;
}

/* CE_TYPE_IMAGE 16×16 atomic entry → TickRGBA.
 *   slot      = block-row
 *   block_idx = (block-col << 2) | quadrant
 * so:
 *   tick.r = quadrant (0..3 — TL/TR/BL/BR)
 *   tick.g = block-col (0..15)
 *   tick.b = block-row (0..15)
 *   tick.a = 0 (block stamps carry colour, not amplitude — leave A free)
 */
static inline TickRGBA ce_tick_from_image(const CEStorageEntry *e) {
    TickRGBA t = (TickRGBA){0, 0, 0, 0};
    if (!e || e->type != CE_TYPE_IMAGE) return t;
    t.r = (uint8_t)(e->block_idx & 0x03);
    t.g = (uint8_t)((e->block_idx >> 2) & 0xFF);
    t.b = (uint8_t)(e->slot & 0xFF);
    t.a = 0;
    return t;
}

/* Generic dispatch — picks the right helper by entry type. Returns
 * a zero TickRGBA for types that have no defined mapping yet. */
static inline TickRGBA ce_tick_from_entry(const CEStorageEntry *e) {
    if (!e) return (TickRGBA){0, 0, 0, 0};
    switch (e->type) {
        case CE_TYPE_SLIG:  return ce_tick_from_slig(e);
        case CE_TYPE_IMAGE: return ce_tick_from_image(e);
        default:            return (TickRGBA){0, 0, 0, 0};
    }
}

/* ─── Tick-sorted iteration over CEStorage ─────────────────────────
 *
 * Build a sorted index of CEStorage entry indices for one canvas_id
 * such that walking the index reproduces the user-specified render
 * order:
 *
 *   B (channel) ascending  →  Y first, then Cb, then Cr
 *   G (scale)   ascending  →  COARSE first, then MID, then FINE
 *   R (cell idx) ascending →  cell 0, 1, 2, ...
 *   A (amp)     ascending  →  weakest first (within ties)
 *
 * The caller passes a `uint32_t out_idx[]` of length `cap`; the
 * function fills in indices into `s->entries[]` for matching
 * (canvas_id, allowed_types) and returns how many it wrote.
 *
 * `allowed_type_mask` is a bitmask: 1 << CE_TYPE_X for each accepted
 * type. Pass `(1u << CE_TYPE_SLIG)` to walk only SLIG cells; pass
 * `(1u << CE_TYPE_SLIG) | (1u << CE_TYPE_IMAGE)` for both.
 */
uint32_t ce_tick_sorted_indices(const CEStorage *s,
                                uint32_t         canvas_id,
                                uint32_t         allowed_type_mask,
                                uint32_t        *out_idx,
                                uint32_t         cap);

#ifdef __cplusplus
}
#endif
#endif /* CE_TICK_H */
