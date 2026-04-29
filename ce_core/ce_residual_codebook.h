/* ce_residual_codebook.h — correction-patch dictionary.
 *
 * The masked-train and hybrid_vae paths produce two kinds of cells:
 *   basis     — large structural shape (slig_decompose_v2 structure)
 *   residual  — edge / texture detail (slig_decompose_v2 edge+texture)
 *   correction — color + event tweaks (slig_decompose_v2 color+events)
 *
 * Once the basis+residual is good, the correction layer is mostly
 * "small fix-ups" that repeat across different images — a red apple
 * and a red ball share most of the saturation-bumping correction
 * cells. Storing them as full CEUnits 64 bytes each is wasteful.
 *
 * This codebook keeps a small set of *correction patches* (CEResidualCode)
 * and lets the storage layer reference them by 1-byte index instead.
 *
 *   CEResidualCode {
 *       CEUnit   unit;            -- the actual correction CEUnit
 *       uint8_t  scale_level;     -- which SLIG scale this matches
 *       uint8_t  direction;       -- SligDir (HORIZ/VERT/DIAG/RIPPLE...)
 *       uint8_t  strength;        -- relative strength 0..255
 *       TickRGBA tick;            -- when in the render order it fires
 *       uint32_t used_count;      -- ref count for cleanup / pruning
 *   }
 *
 * On the storage side a CE_TYPE_RESIDUAL entry stores **only**:
 *   keyframe[0]   = codebook index
 *   keyframe[1]   = strength
 *   keyframe[2..5]= TickRGBA (r, g, b, a)
 *   keyframe[6..7]= x position    (uint16 little-endian, 0..65535;
 *                                  decoder maps to canvas coords)
 *   keyframe[8..9]= y position    (uint16 little-endian)
 *   keyframe[10..]= reserved (zero)
 *
 * That keeps the entry size identical to other CEStorage rows but
 * the 64 bytes are interpreted as a *descriptor*, not as a CEUnit
 * payload. Decode rebuilds the CEUnit by:
 *   idx       = entry.keyframe.bytes[0]
 *   patch     = ce_residual_codebook_get(book, idx)->unit
 * and applies it scaled by entry.keyframe.bytes[1] (strength).
 *
 * Why store the patch by reference instead of as the raw CEUnit:
 *   - if the same correction repeats N times, we save (N-1) × 64 B
 *   - the codebook index (1 byte) is searchable / shareable across
 *     canvases, which is exactly what "사진첩 복사가 아니라 패턴
 *     생성" needs.
 */
#ifndef CE_RESIDUAL_CODEBOOK_H
#define CE_RESIDUAL_CODEBOOK_H

#include <stdint.h>
#include "ce_core.h"
#include "ce_tick.h"
#include "ce_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_RESIDUAL_CODEBOOK_MAX 256

/* Default acceptance threshold (sum of |a-b| over 64 bytes). Pairs
 * within this distance share a codebook slot. ce_distance is L1 over
 * 64 bytes (peak 16320), so a "nearly identical correction" lands
 * around 1500–4000; this threshold (3000) merges those without
 * collapsing distinct corrections. */
#define CE_RESIDUAL_DEFAULT_THRESHOLD 3000u

typedef struct {
    CEUnit   unit;            /* the patch itself */
    uint8_t  scale_level;     /* SligScaleLevel — 0..2 or 0xFF for any */
    uint8_t  direction;       /* SligDir */
    uint8_t  strength;        /* relative strength 0..255 */
    uint8_t  reserved0;
    TickRGBA tick;            /* derived position in render order */
    uint32_t used_count;      /* increments on every add_or_lookup hit */
} CEResidualCode;

typedef struct {
    CEResidualCode codes[CE_RESIDUAL_CODEBOOK_MAX];
    uint32_t       count;
} CEResidualCodebook;

void ce_residual_codebook_init(CEResidualCodebook *book);

/* L1 distance over the 64 raw bytes of two patch CEUnits. Returns
 * UINT32_MAX if scale_level tags don't match — coarse residuals must
 * not collide with fine residuals. */
uint32_t ce_residual_distance(const CEResidualCode *a,
                              const CEResidualCode *b);

/* Best matching index, or -1 if no entry is within `threshold`. */
int ce_residual_codebook_lookup(const CEResidualCodebook *book,
                                const CEResidualCode     *probe,
                                uint32_t                  threshold);

/* Lookup + add. Returns the chosen index 0..count-1.
 *   - probe matches existing → that index, ++used_count
 *   - probe is novel & codebook has room  → append
 *   - probe is novel & codebook is full   → nearest (lossy)
 */
uint8_t ce_residual_codebook_add_or_lookup(CEResidualCodebook   *book,
                                           const CEResidualCode *probe,
                                           uint32_t              threshold);

/* Read accessor. Returns NULL when idx >= count. */
const CEResidualCode *ce_residual_codebook_get(const CEResidualCodebook *book,
                                               uint8_t idx);

/* ─── CEStorage bridge for CE_TYPE_RESIDUAL ─────────────────────────
 *
 * Encode one correction cell as a CE_TYPE_RESIDUAL CEStorage entry.
 * The keyframe bytes are reused as a descriptor:
 *
 *   bytes[0]    = codebook_idx
 *   bytes[1]    = strength
 *   bytes[2..5] = tick.r / tick.g / tick.b / tick.a
 *   bytes[6..7] = x position (uint16 little-endian)
 *   bytes[8..9] = y position (uint16 little-endian)
 *   bytes[10..] = zero (no payload — the patch lives in the codebook)
 *
 * delta is left zero (the descriptor doesn't chain). slot/block_idx
 * carry the (scale_level * 3 + channel, cell index) layout used by
 * SLIG entries so retrieval/sort code stays uniform.
 */
void ce_residual_storage_add(CEStorage *storage,
                             uint32_t   canvas_id,
                             uint16_t   slot,
                             uint16_t   block_idx,
                             uint8_t    codebook_idx,
                             uint8_t    strength,
                             TickRGBA   tick,
                             uint16_t   x,
                             uint16_t   y);

/* Decode the descriptor portion of a CE_TYPE_RESIDUAL entry.
 * Returns 0 on success. The `out_*` fields are optional (pass NULL
 * to skip). */
int ce_residual_storage_unpack(const CEStorageEntry *entry,
                               uint8_t              *out_codebook_idx,
                               uint8_t              *out_strength,
                               TickRGBA             *out_tick,
                               uint16_t             *out_x,
                               uint16_t             *out_y);

#ifdef __cplusplus
}
#endif
#endif /* CE_RESIDUAL_CODEBOOK_H */
