/* ce_storage.h — keyframe/delta storage (Step 0).
 *
 * The storage holds the engine's "training state": every chunk of source
 * material (text block, 8x8 image patch, audio segment) is reduced to a
 * (keyframe, delta) pair via ce_feed/ce_delta and indexed by canvas/slot.
 */
#ifndef CE_STORAGE_H
#define CE_STORAGE_H

#include "ce_core.h"
#include "ce_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t canvas_id;   /* logical document/image id */
    uint16_t slot;        /* coarse position in canvas */
    uint16_t block_idx;   /* fine position within slot */
    uint8_t  type;        /* CEType — modality tag (TEXT/IMAGE/AUDIO) */
    uint8_t  reserved[3]; /* pad / reserved for future use, MUST be zero */
    CEUnit   keyframe;    /* anchor signal */
    CEUnit   delta;       /* anchor -> observed */
} CEStorageEntry;

typedef struct {
    CEStorageEntry *entries;
    uint32_t count;
    uint32_t capacity;
} CEStorage;

void ce_storage_init(CEStorage *s, uint32_t capacity);
void ce_storage_free(CEStorage *s);

/* Append a (canvas_id, slot, block_idx, keyframe, delta) record with
 * modality CE_TYPE_TEXT. Kept for backwards compatibility — new image
 * paths must use ce_storage_add_typed. Grows automatically when full. */
void ce_storage_add(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                    uint16_t block_idx,
                    const CEUnit *keyframe, const CEUnit *delta);

/* Type-tagged append. Use this for image / audio ingestion so that
 * ce_search_by_type can later filter retrievals to a single modality. */
void ce_storage_add_typed(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                          uint16_t block_idx, CEType type,
                          const CEUnit *keyframe, const CEUnit *delta);

/* Convenience: feed `data` of `len` bytes into a fresh CEUnit, compute
 * its delta against `prev_keyframe` (or zero if NULL), and append. The
 * new keyframe stored is the freshly-fed unit (so the caller can chain).
 * Tagged as CE_TYPE_TEXT (this entrypoint is the text-stream path). */
void ce_storage_ingest(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                       uint16_t block_idx,
                       const CEUnit *prev_keyframe,
                       const uint8_t *data, uint32_t len);

/* Find first entry matching (canvas_id, slot, block_idx). Returns index
 * or -1. */
int32_t ce_storage_find(const CEStorage *s, uint32_t canvas_id,
                        uint16_t slot, uint16_t block_idx);

#ifdef __cplusplus
}
#endif
#endif /* CE_STORAGE_H */
