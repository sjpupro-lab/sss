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

/* Ingest a raw RGBA pixel buffer into CEStorage using ce_feed_image's
 * 8x8-block encoder. The image is split into ceil(w/8) x ceil(h/8)
 * blocks; right/bottom edges are zero-padded. Each block is stored as
 * a (keyframe, delta) entry tagged CE_TYPE_IMAGE, with `slot` = block-
 * row and `block_idx` = block-column.
 *
 * Unlike ce_ingest_file (which uses stb_image and pulls in the
 * third_party header), this entry point operates on an in-memory
 * buffer the caller already decoded — keeping the SSS integration
 * path free of stb_image. Returns the number of blocks added. */
uint32_t ce_storage_ingest_rgba(CEStorage *s,
                                uint32_t canvas_id,
                                const uint8_t *rgba, int width, int height);

/* 16x16 group ingest. Splits the image into ceil(w/16) x ceil(h/16)
 * 16x16 blocks (right/bottom edges zero-padded), then encodes each
 * block via ce_feed_image_16 into 4 quadrant CEUnits (TL=0, TR=1,
 * BL=2, BR=3). All four quadrants of a block share the same `slot`
 * (block-row index) and use `block_idx = (block_col << 2) | quadrant`,
 * so retrieval can recover the 4 quadrants of a logical 16x16 patch
 * by looking up `slot` and matching block_idx >> 2. Tagged
 * CE_TYPE_IMAGE. delta of the very first quadrant is taken against the
 * zero CEUnit; subsequent quadrants chain the previous quadrant's
 * keyframe (raster order). Returns total entries added (= 4 * blocks). */
uint32_t ce_storage_ingest_rgba_16(CEStorage *s,
                                   uint32_t canvas_id,
                                   const uint8_t *rgba, int width, int height);

/* ─── SLIG cell persistence (CE_TYPE_SLIG) ─────────────────────────
 *
 *   `set` is one (scale_level, channel) bucket — the same struct
 *   produced by slig_decompose_channel. Each cell becomes one
 *   CEStorage entry with:
 *       canvas_id  = caller-supplied
 *       slot       = scale_level * 3 + channel        (0..8)
 *       block_idx  = cell index inside the set        (0..num_cells-1)
 *       type       = CE_TYPE_SLIG
 *       keyframe   = the SLIG CEUnit
 *       delta      = delta against the previous cell of the same set
 *                    (zero CEUnit for cell 0)
 *
 *   The 9-bucket layout matches what hybrid_vae_encode produces, so
 *   reassembly only needs to walk entries whose canvas_id matches and
 *   bucket them by (slot / 3, slot % 3, block_idx).
 *
 *   Returns number of entries appended. The set's channel /
 *   scale_level fields are read from the struct itself — the function
 *   does not require the caller to recompute the slot number.
 */
struct SligCellSet; /* forward; full definition in slig_signal.h */
uint32_t ce_storage_persist_slig_set(CEStorage *s,
                                     uint32_t canvas_id,
                                     const struct SligCellSet *set);

/* Count how many CE_TYPE_SLIG cells are already persisted under
 * (canvas_id, slot = scale*SLIG_NUM_CHANNELS + channel). Returns the
 * smallest available block_idx for an append operation. Useful when a
 * second pass (e.g. masked-train) wants to add cells behind the
 * encoder's set without overwriting them. Returns 0 when the bucket
 * is empty. */
uint32_t ce_storage_slig_bucket_count(const CEStorage *s,
                                      uint32_t canvas_id,
                                      uint8_t  scale_level,
                                      uint8_t  channel);

/* Like ce_storage_persist_slig_set but writes cells starting at
 * block_idx = base_idx, base_idx+1, ... so additional sets stack
 * behind any pre-existing entries in the same bucket. */
uint32_t ce_storage_append_slig_set(CEStorage *s,
                                    uint32_t canvas_id,
                                    const struct SligCellSet *set,
                                    uint32_t base_idx);

/* Reassemble the [SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS] set grid for one
 * canvas_id. `out` is a 3×3 grid of SligCellSets the caller owns; the
 * function zero-clears it before populating. Sets that have no entries
 * in storage are left zero (num_cells = 0 and the matching channel /
 * scale_level tags pre-set so downstream `slig_reconstruct_at_dim`
 * treats them as silent residuals).
 *
 * Returns the total number of cells loaded across all sets.
 */
uint32_t ce_storage_load_slig_sets(const CEStorage *s,
                                   uint32_t canvas_id,
                                   struct SligCellSet *out);

/* ─── Atmos scene-object persistence (CE_TYPE_SCENE) ─────────
 *
 *   Each CESceneObject is packed into one CEUnit signature and appended
 *   as a CE_TYPE_SCENE entry:
 *       canvas_id  = caller-supplied (one per logical scene/canvas)
 *       slot       = scene_id                      (0..65535)
 *       block_idx  = obj->id                       (0..65535)
 *       keyframe   = packed scene-object signature
 *       delta      = delta against the previous object stored under the
 *                    same (canvas_id, slot)        (zero CEUnit for the
 *                    first object in a scene)
 *
 *   The signature layout is byte-deterministic and lives entirely inside
 *   the 64-byte CEUnit — no JSON, no external sidecar. See
 *   ce_scene_bridge.c for the exact byte map (matches what the matcher
 *   below reads).
 *
 *   Returns 1 on success, 0 if either pointer is NULL.
 */
struct CESceneObject_s; /* forward; full definition in ce_scene_object.h */
uint32_t ce_storage_persist_scene_object(CEStorage *s,
                                         uint32_t canvas_id,
                                         uint16_t scene_id,
                                         const struct CESceneObject_s *obj);

/* Find the closest CE_TYPE_SCENE signature in `s` to the query object's
 * packed signature. Distance is ce_distance() on the keyframe CEUnit
 * (sum of absolute byte differences across 64 bytes, range [0, 16320]).
 *
 *   scene_id_filter — if non-zero, restrict the search to entries whose
 *                     slot == scene_id_filter; if zero, search all
 *                     CE_TYPE_SCENE entries.
 *   out_object_id   — receives the matched entry's block_idx (object id).
 *   out_distance    — receives the SAD distance (smaller = better).
 *
 * Returns 1 if a match was found, 0 if no CE_TYPE_SCENE entries exist
 * (with the optional scene_id_filter applied).
 */
int ce_storage_match_scene_signature(const CEStorage *s,
                                     const struct CESceneObject_s *query,
                                     uint16_t scene_id_filter,
                                     uint16_t *out_object_id,
                                     uint32_t *out_distance);

#ifdef __cplusplus
}
#endif
#endif /* CE_STORAGE_H */
