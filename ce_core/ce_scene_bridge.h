/* ce_scene_bridge.h — bridge from SSS image decomposition (16x16
 * RGBA blocks) to Atmos scene-objects.
 *
 * The bridge converts the existing block-decomposition (kept at the
 * SSS-canonical 16x16 block size — see ce_storage_ingest_rgba_16) into
 * coarse "audio objects" suitable for the Atmos scene engine:
 *
 *   1. Each 16x16 block is reduced to a per-block signature
 *      (avg_r, avg_g, avg_b, texture_energy, edge_strength) using
 *      integer-only arithmetic.
 *   2. Adjacent blocks with similar color and texture are clustered
 *      into one CESceneObject (id = base_id + cluster_index).
 *   3. The cluster's mean signature seeds the object's CEWaveParam
 *      array (freq derived from texture_energy, amp from brightness,
 *      phase from cluster index, waveform from edge_strength).
 *   4. ce_infer_motion_profile() is automatically called on the cluster
 *      signature; the resulting CEMoveProfile is stored alongside the
 *      object in CESceneProfiles, and a pair of seed keyframes are
 *      added to the CESceneObject so that ce_scene_tick alone already
 *      produces the right kind of motion (FLOW slides horizontally,
 *      STATIC stays put, ORGANIC sways, etc.).
 *
 * The bridge is integer-only (no float / double), additive-friendly
 * (it does not invent any blending mode the renderer would not use),
 * and never emits JSON: every object that needs to survive across
 * runs is written through ce_storage_persist_scene_object into the
 * existing CEStorage, which is the only persistence layer the SSS
 * pipeline understands.
 */
#ifndef CE_SCENE_BRIDGE_H
#define CE_SCENE_BRIDGE_H

#include <stdint.h>
#include "ce_scene_object.h"
#include "ce_move_profile.h"
#include "ce_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-object motion profile bag. Index `i` corresponds to
 * scene->objects[i] for any scene built by ce_scene_build_from_rgba(). */
typedef struct {
    CEMoveProfile profiles[CE_SCENE_MAX_OBJECTS];
    uint8_t       base_cx [CE_SCENE_MAX_OBJECTS];
    uint8_t       base_cy [CE_SCENE_MAX_OBJECTS];
    uint16_t      count;
} CESceneProfiles;

/* Build a scene from a raw RGBA image.
 *
 *   scene      — initialised CEScene the bridge will populate.
 *   profiles   — companion bag the bridge fills in lock-step with
 *                scene->objects (one CEMoveProfile per object).
 *   rgba       — caller-owned RGBA8 image, row-major, width*height*4 bytes.
 *   width/height — pixel dims; bridge uses CE_IMAGE_BLOCK16_PX (16) as
 *                  the block step (existing block size, must not change).
 *   base_id    — first object id; subsequent objects get base_id+1, +2, …
 *
 * Returns the number of CESceneObjects created (also reflected in
 * scene->object_count and profiles->count). Returns 0 on bad input
 * or when no clusters survive the minimum-block filter. */
uint16_t ce_scene_build_from_rgba(CEScene *scene,
                                  CESceneProfiles *profiles,
                                  const uint8_t *rgba,
                                  int width, int height,
                                  uint16_t base_id);

/* Persist every object currently in `scene` into `storage` under
 * (canvas_id, scene_id) using ce_storage_persist_scene_object. The
 * profiles bag is consulted for the motion type, which is encoded
 * inside the object's wave_count/keyframe pattern — no JSON sidecar.
 * Returns the number of objects written. */
uint16_t ce_scene_persist_to_storage(CEStorage *storage,
                                     uint32_t canvas_id,
                                     uint16_t scene_id,
                                     const CEScene *scene);

/* One-tick advance that combines the keyframe interpolator with the
 * motion-profile-driven displacement. STATIC objects collapse back to
 * their captured base position every tick, FLOW objects gain horizontal
 * speed plus a vertical wobble, etc. ce_scene_tick is invoked first so
 * that any user-supplied keyframes still apply; the profile is then
 * superimposed on top using the additive ce_move_apply_tick. */
void ce_scene_tick_with_profiles(CEScene *scene,
                                 CESceneProfiles *profiles);

#ifdef __cplusplus
}
#endif
#endif /* CE_SCENE_BRIDGE_H */
