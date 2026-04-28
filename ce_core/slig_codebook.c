/* slig_codebook.c — see slig_codebook.h. */

#include "slig_codebook.h"
#include "ce_core.h"
#include <string.h>
#include <stdint.h>

void slig_codebook_init(SligCodebook *book) {
    if (!book) return;
    memset(book, 0, sizeof(*book));
}

uint32_t slig_cellset_distance(const SligCellSet *a, const SligCellSet *b) {
    if (!a || !b) return UINT32_MAX;
    if (a->channel     != b->channel)     return UINT32_MAX;
    if (a->scale_level != b->scale_level) return UINT32_MAX;

    uint32_t common = a->num_cells < b->num_cells
                    ? a->num_cells : b->num_cells;
    uint32_t total = 0;
    for (uint32_t i = 0; i < common; i++) {
        total += ce_distance(&a->cells[i], &b->cells[i]);
    }
    /* Penalty for cell-count difference: each missing/extra cell adds
     * ~1000 to the distance, enough to keep different-rank patterns
     * apart without dominating the sum for small differences. */
    uint32_t diff = (a->num_cells > b->num_cells)
                  ? a->num_cells - b->num_cells
                  : b->num_cells - a->num_cells;
    total += diff * 1000;
    return total;
}

int slig_codebook_lookup(const SligCodebook *book,
                         const SligCellSet  *probe,
                         uint32_t            threshold) {
    if (!book || !probe || book->pattern_count == 0) return -1;
    int      best      = -1;
    uint32_t best_dist = threshold;
    for (uint32_t i = 0; i < book->pattern_count; i++) {
        uint32_t d = slig_cellset_distance(&book->patterns[i], probe);
        if (d < best_dist) {
            best_dist = d;
            best      = (int)i;
        }
    }
    return best;
}

uint8_t slig_codebook_add_or_lookup(SligCodebook      *book,
                                    const SligCellSet *probe,
                                    uint32_t           threshold) {
    if (!book || !probe) return 0;

    int hit = slig_codebook_lookup(book, probe, threshold);
    if (hit >= 0) {
        book->pattern_usage[hit]++;
        return (uint8_t)hit;
    }

    if (book->pattern_count < SLIG_CODEBOOK_MAX) {
        uint8_t idx = (uint8_t)book->pattern_count;
        book->patterns[idx]      = *probe;
        book->pattern_usage[idx] = 1;
        book->pattern_count++;
        return idx;
    }

    /* Codebook is full and no entry was within threshold — fall back
     * to the globally nearest pattern. Lossy but bounded. */
    int nearest = slig_codebook_lookup(book, probe, UINT32_MAX);
    if (nearest < 0) nearest = 0;
    book->pattern_usage[nearest]++;
    return (uint8_t)nearest;
}

const SligCellSet *slig_codebook_get(const SligCodebook *book, uint8_t idx) {
    if (!book) return NULL;
    if ((uint32_t)idx >= book->pattern_count) return NULL;
    return &book->patterns[idx];
}
