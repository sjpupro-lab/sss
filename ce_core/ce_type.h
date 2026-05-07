/* ce_type.h — modality tag for CEStorage entries.
 *
 * Each entry in CEStorage is tagged with the modality it was ingested
 * from. This allows search, generation, and decode to filter by type
 * (TEXT keyframes for text generation, IMAGE keyframes for image, etc.)
 * without mixing inappropriate sources.
 *
 * The numeric values are persisted in .ces files (storage_io v2) and
 * MUST remain stable across versions.
 */
#ifndef CE_TYPE_H
#define CE_TYPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CE_TYPE_TEXT     = 0,
    CE_TYPE_IMAGE    = 1,
    CE_TYPE_AUDIO    = 2,
    /* SLIG cellset entries — one CEStorage row per SLIG cell:
     *   slot      = scale_level * SLIG_NUM_CHANNELS + channel  (0..8)
     *   block_idx = cell index inside the (scale, channel) set
     *   keyframe  = the SLIG CEUnit
     *   delta     = delta against the previous cell of the same set
     *               (zero CEUnit for index 0)
     * Encoded by ce_storage_persist_slig_set, reassembled by
     * ce_storage_load_slig_sets. */
    CE_TYPE_SLIG     = 3,
    /* Residual codebook reference — color/event correction patches that
     * land in ce_residual_codebook. The CEStorage entry stores only the
     * (index, position, tick, strength) descriptor, not the original
     * patch — see ce_residual_codebook.h. */
    CE_TYPE_RESIDUAL = 4,
    /* Atmos scene-object signatures — one CEStorage row per scene
     * object packed by ce_storage_persist_scene_object():
     *   slot      = scene_id          (caller-supplied logical scene)
     *   block_idx = scene-local object id
     *   keyframe  = packed signature (avg color, waves, geometry, motion)
     *   delta     = delta against the previous object in the same scene
     *               (zero CEUnit for index 0)
     * Encoded by ce_storage_persist_scene_object, matched by
     * ce_storage_match_scene_signature. The packed layout never spills
     * to JSON — it lives entirely inside the CEUnit byte stream. */
    CE_TYPE_SCENE    = 5,

    CE_TYPE_UNKNOWN = 0xFF
} CEType;

static inline const char *ce_type_name(CEType t) {
    switch (t) {
        case CE_TYPE_TEXT:     return "text";
        case CE_TYPE_IMAGE:    return "image";
        case CE_TYPE_AUDIO:    return "audio";
        case CE_TYPE_SLIG:     return "slig";
        case CE_TYPE_RESIDUAL: return "residual";
        case CE_TYPE_SCENE:    return "scene";
        default:               return "unknown";
    }
}

#ifdef __cplusplus
}
#endif
#endif /* CE_TYPE_H */
