/* ce_storage_io.h — CEStorage <-> binary file (.ces).
 *
 * File format v3 (current — little-endian, fixed):
 *   bytes  0.. 3 : magic "CES1" (kept for ID; file format version is in
 *                  the next field, so the magic is just a sanity check)
 *   bytes  4.. 7 : uint32_t version (3)
 *   bytes  8..11 : uint32_t entry count
 *   bytes 12..15 : uint32_t reserved (0)
 *   bytes 16..   : entry[count] each (140 B as in v2, see below)
 *   trailer      : v3 only — appears after the last entry:
 *      uint32_t section_magic = 0x4B424352  ("RCBK" residual codebook)
 *      uint32_t code_count     (0..256)
 *      code[count] each:
 *         uint8_t  scale_level
 *         uint8_t  direction
 *         uint8_t  strength
 *         uint8_t  reserved0
 *         uint8_t  tick[4]      (r, g, b, a)
 *         uint32_t used_count
 *         uint8_t  unit[64]
 *
 * v2 entry layout (still emitted):
 *   uint32_t canvas_id
 *   uint16_t slot
 *   uint16_t block_idx
 *   uint8_t  type             -- CEType (TEXT=0, IMAGE=1, AUDIO=2, ...)
 *   uint8_t  reserved[3]      -- zero
 *   uint8_t  keyframe[64]
 *   uint8_t  delta[64]
 * Total per entry: 4 + 2 + 2 + 1 + 3 + 64 + 64 = 140 bytes.
 *
 * Backward compatibility:
 *   v1 files (no type/reserved bytes, 136 B per entry) still load — every
 *   entry tagged as CE_TYPE_TEXT.
 *   v2 files load as before (no codebook trailer, codebook output left empty).
 *   v3 files always carry a codebook trailer; code_count == 0 is legal.
 */
#ifndef CE_STORAGE_IO_H
#define CE_STORAGE_IO_H

#include "ce_storage.h"
#include "ce_residual_codebook.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_STORAGE_MAGIC      0x31534543u /* "CES1" little-endian */
#define CE_STORAGE_VERSION    3u
#define CE_STORAGE_VERSION_V2 2u
#define CE_STORAGE_VERSION_V1 1u

/* Codebook trailer marker — "RCBK" little-endian. */
#define CE_STORAGE_CODEBOOK_MAGIC 0x4B424352u

/* Save storage to `path` (no codebook trailer — writes v3 with empty
 * codebook section). Returns 1 on success, 0 on I/O error. */
int ce_storage_save(const CEStorage *s, const char *path);

/* Save storage and an associated residual codebook. Pass `book = NULL`
 * to write a v3 file with code_count = 0 (equivalent to ce_storage_save). */
int ce_storage_save_with_codebook(const CEStorage          *s,
                                  const CEResidualCodebook *book,
                                  const char               *path);

/* Load storage from `path` into `out`. Calls ce_storage_init internally.
 * Returns 1 on success, 0 on I/O / format error (out is left empty but
 * valid on failure). v1/v2 files load with no codebook side effects. */
int ce_storage_load(CEStorage *out, const char *path);

/* Load storage and (optionally) the residual codebook trailer. Pass
 * `book = NULL` to discard the trailer (v3 files only). For v1/v2 files
 * the codebook is initialised but left empty. */
int ce_storage_load_with_codebook(CEStorage          *out,
                                  CEResidualCodebook *book,
                                  const char         *path);

#ifdef __cplusplus
}
#endif
#endif
