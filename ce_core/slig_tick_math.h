/* slig_tick_math.h — RGBA tick calculator helpers + waveform tables.
 *
 * The CE Cell layout (slig_signal.h) is 8 × CETier × { minus[4], plus[4] }.
 * Treat each 4-byte half (`plus` or `minus`) as a packed 32-bit integer
 * with little-endian carry: plus[0] is the LSB, plus[3] is the MSB.
 * This gives ~10⁻¹⁰ relative precision (256⁴ = ~4.29e9 levels) with
 * pure-integer add/sub/mul, no float involved.
 *
 * Section 33-letter ASCII summary (precision step):
 *   plus[0]  R-tier  ← finest
 *   plus[1]  G-tier  carry from plus[0]
 *   plus[2]  B-tier  carry from plus[1]
 *   plus[3]  A-tier  most-significant byte
 *
 * Provides:
 *   tick_add / tick_sub / tick_read / tick_write     32-bit unsigned packed
 *   tick_mul8                                         u8 × u8 → u16
 *   tick_sqrt                                         √x via 16-step bisection
 *   tick_div                                          integer division
 *   TICK_SIN_TABLE / TICK_COS_TABLE / TICK_GAUSS_TABLE  256-entry lookup
 *
 * All math integer. All buffers uint8. Use this header from
 * slig_pipeline.c to replace float operations end-to-end.
 */
#ifndef SLIG_TICK_MATH_H
#define SLIG_TICK_MATH_H

#include <stdint.h>
#include "ce_core.h"   /* CEUnit (for the §5 frequency-search helpers below) */

#ifdef __cplusplus
extern "C" {
#endif

/* --- 32-bit packed tick helpers ---------------------------------- */

static inline uint32_t tick_read(const uint8_t plus[4]) {
    return  (uint32_t)plus[0]
         | ((uint32_t)plus[1] << 8)
         | ((uint32_t)plus[2] << 16)
         | ((uint32_t)plus[3] << 24);
}

static inline void tick_write(uint8_t plus[4], uint32_t value) {
    plus[0] = (uint8_t)( value        & 0xFF);
    plus[1] = (uint8_t)((value >>  8) & 0xFF);
    plus[2] = (uint8_t)((value >> 16) & 0xFF);
    plus[3] = (uint8_t)((value >> 24) & 0xFF);
}

static inline void tick_add(uint8_t plus[4], uint32_t value) {
    tick_write(plus, tick_read(plus) + value);
}

static inline void tick_sub(uint8_t plus[4], uint32_t value) {
    tick_write(plus, tick_read(plus) - value);
}

/* --- Small operands ---------------------------------------------- */

static inline uint16_t tick_mul8(uint8_t a, uint8_t b) {
    return (uint16_t)a * (uint16_t)b;
}

/* Integer division. Returns 0 on b == 0 (caller's responsibility). */
static inline uint32_t tick_div(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    return a / b;
}

/* sqrt via 16-step binary search. Range: [0, 2^32) → uint16_t. */
static inline uint16_t tick_sqrt(uint32_t x) {
    uint32_t lo = 0, hi = 65535;
    for (int i = 0; i < 16; i++) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (mid * mid <= x) lo = mid + 1;
        else                hi = mid;
    }
    return (uint16_t)(lo - 1);
}

/* --- Mixed-precision reads (4-byte CETier reinterpret) ----------- */

/* Same 4 bytes, read as two 16-bit values (low/high half). Useful when
 * you want medium-precision storage for two related values (e.g. a
 * Q8 magnitude and a Q8 phase) packed into one CETier. */
static inline void tick_read16x2(const uint8_t p[4],
                                 uint16_t *out_low, uint16_t *out_high) {
    if (out_low)  *out_low  = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    if (out_high) *out_high = (uint16_t)p[2] | ((uint16_t)p[3] << 8);
}

static inline void tick_write16x2(uint8_t p[4], uint16_t low, uint16_t high) {
    p[0] = (uint8_t)( low        & 0xFF);
    p[1] = (uint8_t)((low  >> 8) & 0xFF);
    p[2] = (uint8_t)( high       & 0xFF);
    p[3] = (uint8_t)((high >> 8) & 0xFF);
}

/* --- Frequency-space search / distance (CEUnit primitives) ------- */

/* 32-bit-precision SAD between two CEUnits. Reads each CETier's plus
 * and minus halves as packed uint32 and accumulates |a−b| over all 16
 * tier-halves (8 inc + 8 dec). Returns higher-precision distance than
 * ce_distance's per-byte SAD — preserves direction within a tier. */
uint64_t tick_distance_precise(const CEUnit *a, const CEUnit *b);

/* Integer cosine similarity over the 64 raw bytes of two CEUnits.
 * Returns dot / (|a| × |b|) scaled to ~[0, 256]. 0 = orthogonal /
 * one is zero, ~256 = identical magnitude+direction. Uses tick_sqrt
 * for the norm. */
uint32_t tick_cosine_similarity(const CEUnit *a, const CEUnit *b);

/* DCT-coefficient-only distance — compares inc.G (u DCT[0..7]) and
 * inc.B (v DCT[0..7]) only. Patterns with similar frequency content
 * but different metadata / events return small values. Useful for
 * "find a similar texture" queries. */
uint32_t tick_frequency_distance(const CEUnit *a, const CEUnit *b);

/* --- Waveform tables (defined in slig_tick_math.c) --------------- */

/* sin(2π × i/256) → [0, 255] (centred at 128). */
extern const uint8_t TICK_SIN_TABLE[256];
/* cos(2π × i/256) → [0, 255] (centred at 128). */
extern const uint8_t TICK_COS_TABLE[256];
/* exp(−x²/2) at x = i/64 → [0, 255]. Decays from 255 at i=0 to 0 at i=255. */
extern const uint8_t TICK_GAUSS_TABLE[256];

/* WAVE_STEPS — see ce_wave_steps.h. Re-exported here so callers that
 * already pull in slig_tick_math.h (slig_signal.c, slig_pipeline.c)
 * don't need a second include. */
#include "ce_wave_steps.h"

#ifdef __cplusplus
}
#endif
#endif /* SLIG_TICK_MATH_H */
