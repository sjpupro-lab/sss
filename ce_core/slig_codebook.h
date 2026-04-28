/* slig_codebook.h — texture / pattern dictionary over SligCellSets.
 *
 * The motivation: after enough training the engine sees the same
 * underlying cell sets for repeated textures (grass, sky, skin, …).
 * Storing them once and referencing by index drops per-keyframe
 * storage from O(cells_per_set) to one byte per (scale × channel)
 * slot.
 *
 *   patterns[0]  = SligCellSet   (e.g. "grass" Y/coarse)
 *   patterns[1]  = SligCellSet   (e.g. "sky"   Cb/fine)
 *   ...
 *   patterns[N]  = (≤ SLIG_CODEBOOK_MAX entries; clamped to 256 for
 *                  the on-disk index byte)
 *
 * A SligCellSet is the codebook's pattern unit (not a single cell)
 * because keyframes already organise cells by (channel, scale_level)
 * and a "texture" naturally manifests as the whole set of SVD layers
 * at one scale/channel.
 *
 * Online clustering: slig_codebook_add_or_lookup walks the existing
 * patterns; if the probe is within `threshold` per-byte distance of
 * any entry, we reuse that index. Otherwise we append a new pattern,
 * or — when the codebook is full — fall back to the nearest entry
 * (lossy but graceful).
 *
 * Distance is computed only between matching (channel, scale_level)
 * tags; mismatched tags return UINT32_MAX so chroma never collides
 * with luminance, fine never collides with coarse.
 */
#ifndef SLIG_CODEBOOK_H
#define SLIG_CODEBOOK_H

#include "slig_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SLIG_CODEBOOK_MAX 256

/* Default similarity threshold — sum of ce_distance over paired
 * cells. ce_distance is L1 over 64 bytes so per-cell distance peaks
 * at 16320. With ≈6-cell sets a "broadly similar" pair lands near
 * 8000–25000; this threshold (60000) merges nearly-identical
 * textures while keeping distinct shapes apart. */
#define SLIG_CODEBOOK_DEFAULT_THRESHOLD 60000u

typedef struct {
    SligCellSet patterns[SLIG_CODEBOOK_MAX];
    uint32_t    pattern_count;
    uint32_t    pattern_usage[SLIG_CODEBOOK_MAX];
} SligCodebook;

void slig_codebook_init(SligCodebook *book);

/* Distance between two cell sets. Returns UINT32_MAX if (channel,
 * scale_level) tags don't match — patterns are bucketed strictly by
 * (channel, scale) so a Y-coarse probe never collides with a Cb-fine
 * pattern. Otherwise returns ∑ ce_distance over paired cells plus a
 * penalty for count mismatch. */
uint32_t slig_cellset_distance(const SligCellSet *a, const SligCellSet *b);

/* Best matching index, or -1 if no entry is within `threshold`. */
int slig_codebook_lookup(const SligCodebook *book,
                         const SligCellSet  *probe,
                         uint32_t            threshold);

/* Lookup + add. Returns the chosen pattern index 0..pattern_count-1.
 *   - probe matches existing pattern within threshold → that idx
 *   - probe is novel & codebook has room  → append, return new idx
 *   - probe is novel & codebook is full   → nearest idx (lossy)
 * Always increments pattern_usage[idx]. */
uint8_t slig_codebook_add_or_lookup(SligCodebook       *book,
                                    const SligCellSet  *probe,
                                    uint32_t            threshold);

/* Read accessor — NULL if idx ≥ pattern_count. */
const SligCellSet *slig_codebook_get(const SligCodebook *book, uint8_t idx);

#ifdef __cplusplus
}
#endif

#endif /* SLIG_CODEBOOK_H */
