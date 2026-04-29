/* ce_residual_codebook.c — implementation of the correction-patch
 * dictionary.
 *
 * Design notes:
 *   - Distance: L1 over the 64 raw bytes of unit, only when both
 *     scale_level fields match (or one is 0xFF, meaning "any").
 *   - Add-or-lookup: linear scan over `count`. count is bounded at
 *     CE_RESIDUAL_CODEBOOK_MAX (256), so the scan is < a microsecond
 *     even at full capacity.
 *   - Storage descriptor: see header — keyframe bytes reused as a
 *     compact (idx, strength, tick) tuple; the patch payload lives in
 *     the codebook, not in the storage row.
 */
#include "ce_residual_codebook.h"

#include <string.h>

void ce_residual_codebook_init(CEResidualCodebook *book) {
    if (!book) return;
    memset(book, 0, sizeof(*book));
}

uint32_t ce_residual_distance(const CEResidualCode *a,
                              const CEResidualCode *b) {
    if (!a || !b) return UINT32_MAX;
    /* Scale-tag bucketing — patches with mismatched scale levels are
     * effectively distinct dictionaries. 0xFF acts as a wildcard so
     * caller code that hasn't yet learned a scale can still match. */
    if (a->scale_level != 0xFF && b->scale_level != 0xFF
        && a->scale_level != b->scale_level) {
        return UINT32_MAX;
    }

    const uint8_t *pa = ce_cbytes(&a->unit);
    const uint8_t *pb = ce_cbytes(&b->unit);
    uint32_t d = 0;
    for (int i = 0; i < 64; ++i) {
        int diff = (int)pa[i] - (int)pb[i];
        d += (uint32_t)(diff < 0 ? -diff : diff);
    }
    return d;
}

int ce_residual_codebook_lookup(const CEResidualCodebook *book,
                                const CEResidualCode     *probe,
                                uint32_t                  threshold) {
    if (!book || !probe || book->count == 0) return -1;
    int best = -1;
    uint32_t best_d = UINT32_MAX;
    for (uint32_t i = 0; i < book->count; ++i) {
        uint32_t d = ce_residual_distance(probe, &book->codes[i]);
        if (d < best_d) {
            best_d = d;
            best = (int)i;
        }
    }
    if (best_d > threshold) return -1;
    return best;
}

uint8_t ce_residual_codebook_add_or_lookup(CEResidualCodebook   *book,
                                           const CEResidualCode *probe,
                                           uint32_t              threshold) {
    if (!book || !probe) return 0;

    int hit = ce_residual_codebook_lookup(book, probe, threshold);
    if (hit >= 0) {
        ++book->codes[hit].used_count;
        return (uint8_t)hit;
    }

    if (book->count < CE_RESIDUAL_CODEBOOK_MAX) {
        uint8_t idx = (uint8_t)book->count;
        book->codes[idx] = *probe;
        book->codes[idx].used_count = 1;
        ++book->count;
        return idx;
    }

    /* Codebook is full — fall back to nearest entry, lossy but
     * graceful. Mirrors slig_codebook_add_or_lookup's policy. */
    int nearest = -1;
    uint32_t nearest_d = UINT32_MAX;
    for (uint32_t i = 0; i < book->count; ++i) {
        uint32_t d = ce_residual_distance(probe, &book->codes[i]);
        if (d < nearest_d) {
            nearest_d = d;
            nearest = (int)i;
        }
    }
    if (nearest < 0) nearest = 0;
    ++book->codes[nearest].used_count;
    return (uint8_t)nearest;
}

const CEResidualCode *ce_residual_codebook_get(const CEResidualCodebook *book,
                                               uint8_t idx) {
    if (!book || (uint32_t)idx >= book->count) return NULL;
    return &book->codes[idx];
}

/* ─── CEStorage bridge ──────────────────────────────────────────── */

void ce_residual_storage_add(CEStorage *storage,
                             uint32_t   canvas_id,
                             uint16_t   slot,
                             uint16_t   block_idx,
                             uint8_t    codebook_idx,
                             uint8_t    strength,
                             TickRGBA   tick,
                             uint16_t   x,
                             uint16_t   y) {
    if (!storage) return;

    CEUnit descriptor;
    ce_init(&descriptor);
    /* Pack via direct byte access. ce_cbytes is the read-side helper;
     * we write through a pointer cast since CEUnit's byte layout is
     * exposed by ce_core.h. The first 10 bytes carry the descriptor;
     * the rest stay zero so distance comparisons against true CEUnits
     * never falsely tie. */
    uint8_t *b = (uint8_t *)&descriptor;
    b[0] = codebook_idx;
    b[1] = strength;
    b[2] = tick.r;
    b[3] = tick.g;
    b[4] = tick.b;
    b[5] = tick.a;
    b[6] = (uint8_t)( x        & 0xFF);
    b[7] = (uint8_t)((x >> 8)  & 0xFF);
    b[8] = (uint8_t)( y        & 0xFF);
    b[9] = (uint8_t)((y >> 8)  & 0xFF);
    /* b[10..] reserved — zero from ce_init */

    CEUnit zero;
    ce_init(&zero);
    ce_storage_add_typed(storage, canvas_id, slot, block_idx,
                         CE_TYPE_RESIDUAL, &descriptor, &zero);
}

int ce_residual_storage_unpack(const CEStorageEntry *entry,
                               uint8_t              *out_codebook_idx,
                               uint8_t              *out_strength,
                               TickRGBA             *out_tick,
                               uint16_t             *out_x,
                               uint16_t             *out_y) {
    if (!entry || entry->type != CE_TYPE_RESIDUAL) return -1;
    const uint8_t *b = ce_cbytes(&entry->keyframe);
    if (out_codebook_idx) *out_codebook_idx = b[0];
    if (out_strength)     *out_strength     = b[1];
    if (out_tick) {
        out_tick->r = b[2];
        out_tick->g = b[3];
        out_tick->b = b[4];
        out_tick->a = b[5];
    }
    if (out_x) *out_x = (uint16_t)b[6] | ((uint16_t)b[7] << 8);
    if (out_y) *out_y = (uint16_t)b[8] | ((uint16_t)b[9] << 8);
    return 0;
}
