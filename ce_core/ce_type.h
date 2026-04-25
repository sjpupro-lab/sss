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
    CE_TYPE_TEXT  = 0,
    CE_TYPE_IMAGE = 1,
    CE_TYPE_AUDIO = 2,

    CE_TYPE_UNKNOWN = 0xFF
} CEType;

static inline const char *ce_type_name(CEType t) {
    switch (t) {
        case CE_TYPE_TEXT:  return "text";
        case CE_TYPE_IMAGE: return "image";
        case CE_TYPE_AUDIO: return "audio";
        default:            return "unknown";
    }
}

#ifdef __cplusplus
}
#endif
#endif /* CE_TYPE_H */
