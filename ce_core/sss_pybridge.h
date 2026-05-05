/* sss_pybridge.h — minimal C ABI for Python ctypes integration.
 *
 * Wraps CEStorage so the Python SSS upgrade loop's CEMemory.add_cell
 * routes through ce_storage_add_typed (every add lands in real CE
 * storage, with type-tagged keyframe + delta against the previous
 * cell in the same canvas/slot).
 *
 * The opaque handle owns its CEStorage and a tiny per-(canvas, slot)
 * tail map used to chain delta computation. Python keeps its own
 * sidecar (quality, source, use_count) — those are out-of-band and do
 * not belong inside CEStorage entries.
 */
#ifndef SSS_PYBRIDGE_H
#define SSS_PYBRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sss_memory sss_memory;

sss_memory *sss_memory_create(uint32_t initial_capacity);
void        sss_memory_destroy(sss_memory *m);

/* Encode `bytes[len]` into a fresh CEUnit via ce_feed, compute the
 * delta against the previous tail keyframe for (canvas_id, slot), and
 * append via ce_storage_add_typed. Returns the block_idx assigned
 * (monotonic per (canvas_id, slot)). Returns UINT32_MAX on error. */
uint32_t    sss_memory_add_typed(sss_memory   *m,
                                 uint32_t      canvas_id,
                                 uint16_t      slot,
                                 uint8_t       type,
                                 const uint8_t *bytes,
                                 uint32_t      len);

/* Total entries across all canvases / slots. */
uint32_t    sss_memory_count(const sss_memory *m);

/* Number of entries with the given (canvas_id, slot). */
uint32_t    sss_memory_count_by_slot(const sss_memory *m,
                                     uint32_t          canvas_id,
                                     uint16_t          slot);

/* Persist as .ces (storage_io v3, no codebook). 1 = ok. */
int         sss_memory_save(const sss_memory *m, const char *path);

/* Replace contents with a previously-saved .ces. 1 = ok. */
int         sss_memory_load(sss_memory *m, const char *path);

/* Copy the 64-byte keyframe for (canvas_id, slot, block_idx) into
 * `out[64]`. Returns 1 on hit, 0 if not found. */
int         sss_memory_get_keyframe(const sss_memory *m,
                                    uint32_t          canvas_id,
                                    uint16_t          slot,
                                    uint16_t          block_idx,
                                    uint8_t           out[64]);

#ifdef __cplusplus
}
#endif
#endif /* SSS_PYBRIDGE_H */
