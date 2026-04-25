/* ce_core.h — CE Cell base unit (64-byte deterministic signal carrier).
 *
 * A CEUnit is the atomic signal cell of the CANVAS engine. It carries 64
 * bytes organised as two halves (inc/dec), each containing four channels
 * (R/G/B/A) of a 4-tier carry chain (8 bytes per channel).
 *
 * All operations are deterministic and byte-exact. ce_apply / ce_delta form
 * a perfectly reversible pair under modular arithmetic (mod 256).
 */
#ifndef CE_CORE_H
#define CE_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 4-tier carry chain: 256^4 = ~4.29B states per direction per channel. */
typedef struct {
    uint8_t minus[4];
    uint8_t plus[4];
} CETier; /* 8 bytes */

typedef struct { CETier R, G, B, A; } CEHalf; /* 32 bytes */
typedef struct { CEHalf inc; CEHalf dec; } CEUnit; /* 64 bytes */

enum CEChannel { CE_CH_R = 0, CE_CH_G = 1, CE_CH_B = 2, CE_CH_A = 3 };

/* Initialise the unit to the zero signal. */
void ce_init(CEUnit *u);

/* Feed a byte stream into the unit. Deterministic, length-independent
 * (different lengths yield different states). Throughput target: >=8 MB/s. */
void ce_feed(CEUnit *u, const uint8_t *data, uint32_t len);

/* 64-byte sum-of-absolute-differences distance.
 * Range: [0, 64*255] = [0, 16320]. */
uint32_t ce_distance(const CEUnit *a, const CEUnit *b);

/* out = current - anchor (byte-wise mod 256). */
void ce_delta(CEUnit *out, const CEUnit *anchor, const CEUnit *current);

/* out = anchor + delta (byte-wise mod 256).
 * Reverse of ce_delta: ce_apply(anchor, ce_delta(anchor, c)) == c. */
void ce_apply(CEUnit *out, const CEUnit *anchor, const CEUnit *delta);

/* Read a signed 64-bit projection of the requested channel.
 * Combines plus/minus tiers across inc/dec halves. */
int64_t ce_read(const CEUnit *u, int channel);

/* --- Helpers used by higher layers ---------------------------------- */

/* Scale a delta by weight (clamped to [-8.0, 8.0]). out_byte = round(d * w)
 * mod 256. Used for residual / cross-attn weighted application. */
void ce_delta_scale(CEUnit *out, const CEUnit *delta, float weight);

/* Equality check (byte-exact). */
int ce_equal(const CEUnit *a, const CEUnit *b);

/* Raw view: treat CEUnit as 64 bytes. */
static inline uint8_t *ce_bytes(CEUnit *u) { return (uint8_t *)u; }
static inline const uint8_t *ce_cbytes(const CEUnit *u) { return (const uint8_t *)u; }

#ifdef __cplusplus
}
#endif

#endif /* CE_CORE_H */
