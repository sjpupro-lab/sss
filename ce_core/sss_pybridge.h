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

/* sss_rowvae thin wrapper: load .sss model, run sss_generate, write
 * uint8 RGB into out_rgb (size out_w * out_h * 3). If the model size
 * differs from out_w/out_h, the result is nearest-neighbour resampled.
 * Returns 0 on success, -1 on any failure (model load, allocation,
 * generate). The caller owns out_rgb. */
int         sss_pybridge_generate(const char *model_path,
                                  const char *prompt,
                                  uint32_t    seed,
                                  float       detail,
                                  int         steps,
                                  uint8_t    *out_rgb,
                                  uint32_t    out_w,
                                  uint32_t    out_h);

/* Run ce_feed over `text[len]` and copy the resulting CEUnit into the
 * caller's 64-byte buffer. Used by the SSS trainer to derive each
 * cell's `ce_key` (the 256-grid morpheme link) without re-implementing
 * ce_feed in Python. */
void        sss_pybridge_ce_feed(const char *text,
                                 uint32_t    len,
                                 uint8_t     out_64[64]);

/* ce_distance over the two 64-byte CEUnits. 0 means identical;
 * upper-bounded around 16320. */
uint32_t    sss_pybridge_ce_distance(const uint8_t a_64[64],
                                     const uint8_t b_64[64]);

/* ── Atmos scene-object production entry points ─────────────────────
 *
 * These wrappers expose ce_scene_build_from_rgba / ce_scene_render /
 * ce_scene_tick_with_profiles / ce_scene_persist_to_storage to Python
 * (and any other ctypes consumer) so the Atmos engine can drive
 * production code paths — restore-time object detection
 * (tools/sss_unified.py), motion frame generation
 * (scripts/make_anim_frames.py), and direct CE storage round-trips.
 *
 * The handle is opaque (`sss_atmos_scene`); it carries a CEScene plus
 * the matching CESceneProfiles bag in lock-step. Build → tick → render
 * is the standard cycle. Bytes everywhere — no float crosses the ABI. */

typedef struct sss_atmos_scene sss_atmos_scene;

/* Allocate an empty scene + profiles bag. Returns NULL on OOM.
 * Caller must call sss_atmos_scene_destroy. */
sss_atmos_scene *sss_atmos_scene_create(void);
void             sss_atmos_scene_destroy(sss_atmos_scene *s);

/* Decompose `rgba` (row-major, width*height*4 bytes) into Atmos scene
 * objects via ce_scene_build_from_rgba. Returns the number of objects
 * produced; 0 on bad input. Subsequent calls overwrite the scene. */
uint16_t sss_atmos_scene_build_from_rgba(sss_atmos_scene *s,
                                         const uint8_t *rgba,
                                         int width, int height,
                                         uint16_t base_id);

/* Number of scene objects currently in the scene. */
uint16_t sss_atmos_scene_object_count(const sss_atmos_scene *s);

/* Copy per-object summary (id, motion-type, base_cx, base_cy, color)
 * into the caller's array — one row per object, 8 bytes per row:
 *   [0..1] = id           (uint16, little-endian)
 *   [2]    = motion_type  (CEMotionType: 0=STATIC, 1=FLOW, 2=RIGID,
 *                          3=ORGANIC, 4=PARTICLE)
 *   [3]    = base_cx
 *   [4]    = base_cy
 *   [5]    = color_r
 *   [6]    = color_g
 *   [7]    = color_b
 * out_rows must be at least object_count * 8 bytes. Returns objects
 * written. */
uint16_t sss_atmos_scene_dump_objects(const sss_atmos_scene *s,
                                      uint8_t *out_rows,
                                      uint16_t max_rows);

/* Advance the scene by one tick, layering profile-driven displacement
 * on top of the keyframe interpolation (ce_scene_tick_with_profiles). */
void sss_atmos_scene_tick(sss_atmos_scene *s);

/* Seek to a specific tick (ce_scene_seek). */
void sss_atmos_scene_seek(sss_atmos_scene *s, uint32_t target_tick);

/* Render the current scene state into out_rgb (width*height*3 bytes).
 * Caller owns the buffer. Buffer is zeroed before rendering so callers
 * never see stale pixels from a previous frame. */
void sss_atmos_scene_render(const sss_atmos_scene *s,
                            uint8_t *out_rgb, int width, int height);

/* Persist every object in the scene through ce_scene_persist_to_storage
 * into the given sss_memory's CEStorage under (canvas_id, scene_id).
 * Returns objects written; 0 on bad input. */
uint16_t sss_atmos_scene_persist(sss_memory *mem,
                                 const sss_atmos_scene *s,
                                 uint32_t canvas_id,
                                 uint16_t scene_id);

/* ── Phase 2 feature bank pybridge ──────────────────────────────────
 *
 * Thin ctypes-friendly wrappers around sss_feature_bank_save / load.
 * The caller hands us tightly-packed motif/relation/identity buffers
 * (one record-size block per record, no padding) — exactly the bytes
 * tools/sss_feature_bank.py already produces. We slot those into an
 * SSSFeatureBank and delegate to the C save / load.
 *
 * Both functions return SFB_OK (0) on success or a negative SFB_ERR_*. */
int sss_pybridge_feature_bank_save(const char    *path,
                                   const uint8_t *motifs_buf,
                                   uint32_t       motif_count,
                                   const uint8_t *relations_buf,
                                   uint32_t       relation_count,
                                   const uint8_t *identities_buf,
                                   uint32_t       identity_count);

/* Load into caller-provided buffers. Caller passes the buffer sizes
 * in *out_*_count (capacity); on success they're overwritten with the
 * actual count read. If the file holds more records than capacity,
 * returns SFB_ERR_ALLOC and writes the required counts. */
int sss_pybridge_feature_bank_load(const char *path,
                                   uint8_t   *motifs_buf,
                                   uint32_t  *out_motif_count,
                                   uint8_t   *relations_buf,
                                   uint32_t  *out_relation_count,
                                   uint8_t   *identities_buf,
                                   uint32_t  *out_identity_count);

/* Header-only probe for callers that need to size the buffers before
 * the full load. Fills *out_*_count from the header and returns
 * SFB_OK; the file is closed before returning. */
int sss_pybridge_feature_bank_probe(const char *path,
                                    uint32_t   *out_motif_count,
                                    uint32_t   *out_relation_count,
                                    uint32_t   *out_identity_count);

#ifdef __cplusplus
}
#endif
#endif /* SSS_PYBRIDGE_H */
