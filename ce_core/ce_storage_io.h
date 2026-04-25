/* ce_storage_io.h — CEStorage <-> binary file (.ces).
 *
 * File format (little-endian, fixed):
 *   bytes  0.. 3 : magic "CES1"
 *   bytes  4.. 7 : uint32_t version (currently 1)
 *   bytes  8..11 : uint32_t entry count
 *   bytes 12..15 : uint32_t reserved (0)
 *   bytes 16..   : entry[count] each:
 *      uint32_t canvas_id
 *      uint16_t slot
 *      uint16_t block_idx
 *      uint8_t  keyframe[64]
 *      uint8_t  delta[64]
 *
 * Total per entry: 4 + 2 + 2 + 64 + 64 = 136 bytes.
 */
#ifndef CE_STORAGE_IO_H
#define CE_STORAGE_IO_H

#include "ce_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CE_STORAGE_MAGIC   0x31534543u /* "CES1" little-endian */
#define CE_STORAGE_VERSION 1u

/* Save storage to `path`. Returns 1 on success, 0 on I/O error. */
int ce_storage_save(const CEStorage *s, const char *path);

/* Load storage from `path` into `out`. The caller passes an *uninitialised*
 * or already-freed CEStorage; this function calls ce_storage_init internally.
 * Returns 1 on success, 0 on I/O / format error (in which case `out` is
 * left empty but valid). */
int ce_storage_load(CEStorage *out, const char *path);

#ifdef __cplusplus
}
#endif
#endif
