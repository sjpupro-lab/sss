#ifndef SPATIAL_IO_H
#define SPATIAL_IO_H

#include "spatial_keyframe.h"

/*
 * Binary model I/O for SpatialAI.
 *
 * File layout:
 *   [Header - 32 bytes]
 *     magic[4]        = "SPAI"
 *     version         = uint32  (SPAI_VERSION)
 *     kf_count        = uint32  (number of keyframe records on disk)
 *     df_count        = uint32  (number of delta records on disk)
 *     reserved[3]     = uint32 x 3  (zero for now; reserved for future)
 *
 *   [Record]* — tagged record stream, KF and Delta interleaved in the
 *   order they were written. Enables append-only incremental save.
 *
 *     tag = 0x01  Keyframe record
 *       uint32 id
 *       char   label[64]
 *       uint32 text_byte_count
 *       uint16 A[GRID_TOTAL]       (128 KB)
 *       uint8  R[GRID_TOTAL]       ( 64 KB)
 *       uint8  G[GRID_TOTAL]       ( 64 KB)
 *       uint8  B[GRID_TOTAL]       ( 64 KB)
 *       → body = 4 + 64 + 4 + 327680 = 327 752 bytes
 *
 *     tag = 0x02  Delta record
 *       uint32 id
 *       uint32 parent_id
 *       char   label[64]
 *       uint32 count
 *       float  change_ratio
 *       DeltaEntry entries[count]  (8 bytes each)
 *
 *   Native little-endian byte order. Suitable for single-machine
 *   save/load; cross-platform callers should byte-swap on read/write.
 *
 * Incremental save preserves the "disk == in-memory prefix" invariant:
 *   keyframes[i] and deltas[i] that are already on disk are never
 *   rewritten; only new entries past the recorded counts are appended.
 */

#define SPAI_MAGIC    "SPAI"
#define SPAI_VERSION  5u       /* v5 adds Keyframe.topic_hash + seq_in_topic (v3/v4 kept readable) */

#define SPAI_TAG_KEYFRAME  0x01
#define SPAI_TAG_DELTA     0x02
#define SPAI_TAG_WEIGHTS   0x03   /* ChannelWeight block: 4 × float */
#define SPAI_TAG_CANVAS    0x04   /* One SpatialCanvas (v3) */
#define SPAI_TAG_SUBTITLE  0x05   /* SubtitleTrack (v3) */
#define SPAI_TAG_EMA       0x06   /* SpatialAI EMA tables (v4+): 4 × GRID_TOTAL × float */

typedef enum {
    SPAI_OK = 0,
    SPAI_ERR_OPEN,
    SPAI_ERR_MAGIC,
    SPAI_ERR_VERSION,
    SPAI_ERR_READ,
    SPAI_ERR_WRITE,
    SPAI_ERR_CORRUPT,
    SPAI_ERR_ALLOC,
    SPAI_ERR_STATE   /* in-memory state inconsistent with file */
} SpaiStatus;

const char* spai_status_str(SpaiStatus s);

/* Persist the full model to disk. Overwrites any existing file. */
SpaiStatus ai_save(const SpatialAI* ai, const char* path);

/* Load a model from disk into a freshly allocated SpatialAI.
   On success (*out_status == SPAI_OK) the caller owns the returned
   engine and must call spatial_ai_destroy(). On failure returns NULL
   and writes the status code. out_status may be NULL. */
SpatialAI* ai_load(const char* path, SpaiStatus* out_status);

/* Append-only save.
 *   - if path does not exist or is not a valid SPAI file → full save
 *   - if path exists and was produced by a prior ai_save/ai_save_incremental
 *     on this engine, append only entries past the recorded counts
 *     and update the header.
 *   - if the file records more entries than are in memory →
 *     SPAI_ERR_STATE (prevents silent data loss).
 */
SpaiStatus ai_save_incremental(const SpatialAI* ai, const char* path);

/* Report the counts currently recorded in a file's header without
   loading the model. Useful for progress reporting. Returns SPAI_OK on
   success, error code on failure. */
SpaiStatus ai_peek_header(const char* path,
                          uint32_t* out_kf_count,
                          uint32_t* out_df_count,
                          uint32_t* out_version);

#endif /* SPATIAL_IO_H */
