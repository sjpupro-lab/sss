/* ce_ingest.h — directory of PNG/JPG -> CEStorage.
 *
 * Each image is decoded to RGBA, padded to a multiple of 8x8, then split
 * into 8x8 RGBA blocks. Each block becomes one CE cell:
 *
 *   block bytes (256 = 8 * 8 * 4) -> ce_feed -> CEUnit (keyframe)
 *   delta = keyframe - prev_block_in_image (chained per image)
 *
 * Indexing scheme:
 *   canvas_id = stable hash of the file path (FNV-1a 32-bit)
 *   slot      = image's row index in 8x8 blocks (block_y)
 *   block_idx = image's column index in 8x8 blocks (block_x)
 *
 * Supported by stb_image: PNG, JPG, BMP, TGA, PSD, GIF (single frame),
 * HDR, PIC, PNM. The walker only filters by extension (.png, .jpg, .jpeg)
 * to keep behaviour predictable; pass other extensions explicitly via
 * ce_ingest_file() if you need them.
 */
#ifndef CE_INGEST_H
#define CE_INGEST_H

#include "ce_core.h"
#include "ce_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t images_seen;     /* files matched */
    uint32_t images_decoded;  /* successfully decoded */
    uint32_t blocks_added;    /* CE cells appended */
    uint32_t errors;          /* decode / read failures */
} CEIngestStats;

/* Ingest a single image file. Returns 1 if at least one block was added. */
int ce_ingest_file(CEStorage *s, const char *path, CEIngestStats *stats);

/* Walk `dir` non-recursively and ingest every *.png / *.jpg / *.jpeg
 * (case-insensitive). Returns the number of images successfully ingested.
 * `stats` may be NULL. */
int ce_ingest_directory(CEStorage *s, const char *dir, CEIngestStats *stats);

/* Walk `dir` recursively. */
int ce_ingest_directory_recursive(CEStorage *s, const char *dir,
                                  CEIngestStats *stats);

/* FNV-1a 32-bit hash of a NUL-terminated string. Exposed for tests
 * (also used internally as canvas_id). */
uint32_t ce_path_hash(const char *path);

#ifdef __cplusplus
}
#endif
#endif
