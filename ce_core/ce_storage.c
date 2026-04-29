#include "ce_storage.h"
#include "ce_feed_image.h"
#include "ce_tick.h"
#include "slig_signal.h"
#include <stdlib.h>
#include <string.h>

void ce_storage_init(CEStorage *s, uint32_t capacity) {
    s->count = 0;
    s->capacity = capacity ? capacity : 16;
    s->entries = (CEStorageEntry *)calloc(s->capacity, sizeof(CEStorageEntry));
}

void ce_storage_free(CEStorage *s) {
    free(s->entries);
    s->entries = NULL;
    s->count = s->capacity = 0;
}

static void grow(CEStorage *s) {
    uint32_t cap = s->capacity * 2;
    if (cap < 16) cap = 16;
    CEStorageEntry *p = (CEStorageEntry *)realloc(s->entries, cap * sizeof(*p));
    /* If realloc fails, we drop the add silently; tests cover normal capacity. */
    if (!p) return;
    memset(p + s->capacity, 0, (cap - s->capacity) * sizeof(*p));
    s->entries = p;
    s->capacity = cap;
}

void ce_storage_add_typed(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                          uint16_t block_idx, CEType type,
                          const CEUnit *keyframe, const CEUnit *delta) {
    if (s->count >= s->capacity) grow(s);
    if (s->count >= s->capacity) return;
    CEStorageEntry *e = &s->entries[s->count++];
    e->canvas_id = canvas_id;
    e->slot = slot;
    e->block_idx = block_idx;
    e->type = (uint8_t)type;
    e->reserved[0] = e->reserved[1] = e->reserved[2] = 0;
    e->keyframe = *keyframe;
    if (delta) e->delta = *delta;
    else       memset(&e->delta, 0, sizeof(e->delta));
}

void ce_storage_add(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                    uint16_t block_idx,
                    const CEUnit *keyframe, const CEUnit *delta) {
    ce_storage_add_typed(s, canvas_id, slot, block_idx,
                         CE_TYPE_TEXT, keyframe, delta);
}

void ce_storage_ingest(CEStorage *s, uint32_t canvas_id, uint16_t slot,
                       uint16_t block_idx,
                       const CEUnit *prev_keyframe,
                       const uint8_t *data, uint32_t len) {
    CEUnit fresh;
    ce_init(&fresh);
    ce_feed(&fresh, data, len);

    CEUnit anchor;
    if (prev_keyframe) anchor = *prev_keyframe;
    else               ce_init(&anchor);

    CEUnit delta;
    ce_delta(&delta, &anchor, &fresh);
    ce_storage_add(s, canvas_id, slot, block_idx, &fresh, &delta);
}

int32_t ce_storage_find(const CEStorage *s, uint32_t canvas_id,
                        uint16_t slot, uint16_t block_idx) {
    for (uint32_t i = 0; i < s->count; ++i) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->canvas_id == canvas_id && e->slot == slot && e->block_idx == block_idx)
            return (int32_t)i;
    }
    return -1;
}

uint32_t ce_storage_ingest_rgba_16(CEStorage *s,
                                   uint32_t canvas_id,
                                   const uint8_t *rgba, int width, int height) {
    if (!s || !rgba || width <= 0 || height <= 0) return 0;

    int blocks_x = (width  + CE_IMAGE_BLOCK16_PX - 1) / CE_IMAGE_BLOCK16_PX;
    int blocks_y = (height + CE_IMAGE_BLOCK16_PX - 1) / CE_IMAGE_BLOCK16_PX;

    uint8_t block16[CE_IMAGE_BLOCK16_BYTES];
    CEUnit prev; ce_init(&prev);
    int has_prev = 0;

    uint32_t added = 0;
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            memset(block16, 0, sizeof(block16));
            int x0 = bx * CE_IMAGE_BLOCK16_PX;
            int y0 = by * CE_IMAGE_BLOCK16_PX;
            for (int dy = 0; dy < CE_IMAGE_BLOCK16_PX; ++dy) {
                int y = y0 + dy;
                if (y >= height) break;
                for (int dx = 0; dx < CE_IMAGE_BLOCK16_PX; ++dx) {
                    int x = x0 + dx;
                    if (x >= width) break;
                    const uint8_t *src = rgba + ((size_t)y * (size_t)width + (size_t)x) * 4u;
                    uint8_t *dst = block16 + ((size_t)dy * CE_IMAGE_BLOCK16_PX + (size_t)dx) * 4u;
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                }
            }

            CEUnit q[4];
            ce_feed_image_16(q, block16);

            for (int qi = 0; qi < 4; ++qi) {
                CEUnit delta;
                if (!has_prev) {
                    CEUnit zero; ce_init(&zero);
                    ce_delta(&delta, &zero, &q[qi]);
                } else {
                    ce_delta(&delta, &prev, &q[qi]);
                }
                uint16_t slot = (uint16_t)(by & 0xFFFF);
                uint16_t bidx = (uint16_t)(((bx & 0x3FFF) << 2) | (qi & 0x3));
                ce_storage_add_typed(s, canvas_id, slot, bidx,
                                     CE_TYPE_IMAGE, &q[qi], &delta);
                prev = q[qi];
                has_prev = 1;
                ++added;
            }
        }
    }
    return added;
}

uint32_t ce_storage_ingest_rgba(CEStorage *s,
                                uint32_t canvas_id,
                                const uint8_t *rgba, int width, int height) {
    if (!s || !rgba || width <= 0 || height <= 0) return 0;

    int blocks_x = (width  + CE_IMAGE_BLOCK_PX - 1) / CE_IMAGE_BLOCK_PX;
    int blocks_y = (height + CE_IMAGE_BLOCK_PX - 1) / CE_IMAGE_BLOCK_PX;

    uint8_t block[CE_IMAGE_BLOCK_BYTES];
    CEUnit prev; ce_init(&prev);

    uint32_t added = 0;
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            memset(block, 0, sizeof(block));
            int x0 = bx * CE_IMAGE_BLOCK_PX;
            int y0 = by * CE_IMAGE_BLOCK_PX;
            for (int dy = 0; dy < CE_IMAGE_BLOCK_PX; ++dy) {
                int y = y0 + dy;
                if (y >= height) break;
                for (int dx = 0; dx < CE_IMAGE_BLOCK_PX; ++dx) {
                    int x = x0 + dx;
                    if (x >= width) break;
                    const uint8_t *src = rgba + ((size_t)y * (size_t)width + (size_t)x) * 4u;
                    uint8_t *dst = block + ((size_t)dy * CE_IMAGE_BLOCK_PX + (size_t)dx) * 4u;
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                }
            }

            CEUnit fresh;
            ce_feed_image(&fresh, block);

            CEUnit delta;
            if (added == 0) {
                CEUnit zero; ce_init(&zero);
                ce_delta(&delta, &zero, &fresh);
            } else {
                ce_delta(&delta, &prev, &fresh);
            }

            ce_storage_add_typed(s, canvas_id,
                                 (uint16_t)(by & 0xFFFF),
                                 (uint16_t)(bx & 0xFFFF),
                                 CE_TYPE_IMAGE,
                                 &fresh, &delta);
            prev = fresh;
            ++added;
        }
    }
    return added;
}

uint32_t ce_storage_append_slig_set(CEStorage *s,
                                    uint32_t canvas_id,
                                    const struct SligCellSet *set,
                                    uint32_t base_idx) {
    if (!s || !set || set->num_cells == 0) return 0;
    if (set->scale_level >= SLIG_NUM_LEVELS ||
        set->channel >= SLIG_NUM_CHANNELS) return 0;

    uint16_t slot = (uint16_t)(set->scale_level * SLIG_NUM_CHANNELS + set->channel);

    CEUnit prev; ce_init(&prev);
    int has_prev = 0;
    uint32_t added = 0;
    for (uint32_t i = 0; i < set->num_cells; ++i) {
        uint32_t idx = base_idx + i;
        if (idx > 0xFFFFu) break;     /* slot index is uint16_t */
        if (idx >= SLIG_MAX_CELLS) break; /* loader caps reads at this */

        const CEUnit *kf = &set->cells[i];
        CEUnit delta;
        if (!has_prev) {
            CEUnit zero; ce_init(&zero);
            ce_delta(&delta, &zero, kf);
        } else {
            ce_delta(&delta, &prev, kf);
        }
        ce_storage_add_typed(s, canvas_id, slot, (uint16_t)idx,
                             CE_TYPE_SLIG, kf, &delta);
        prev = *kf;
        has_prev = 1;
        ++added;
    }
    return added;
}

uint32_t ce_storage_persist_slig_set(CEStorage *s,
                                     uint32_t canvas_id,
                                     const struct SligCellSet *set) {
    return ce_storage_append_slig_set(s, canvas_id, set, 0);
}

uint32_t ce_storage_slig_bucket_count(const CEStorage *s,
                                      uint32_t canvas_id,
                                      uint8_t  scale_level,
                                      uint8_t  channel) {
    if (!s) return 0;
    if (scale_level >= SLIG_NUM_LEVELS || channel >= SLIG_NUM_CHANNELS) return 0;
    uint16_t slot = (uint16_t)(scale_level * SLIG_NUM_CHANNELS + channel);
    uint32_t max_idx_plus_one = 0;
    for (uint32_t i = 0; i < s->count; ++i) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->type != CE_TYPE_SLIG)   continue;
        if (e->canvas_id != canvas_id) continue;
        if (e->slot != slot)           continue;
        uint32_t idx_p1 = (uint32_t)e->block_idx + 1u;
        if (idx_p1 > max_idx_plus_one) max_idx_plus_one = idx_p1;
    }
    return max_idx_plus_one;
}

uint32_t ce_storage_load_slig_sets(const CEStorage *s,
                                   uint32_t canvas_id,
                                   struct SligCellSet *out_) {
    SligCellSet (*out)[SLIG_NUM_CHANNELS] =
        (SligCellSet (*)[SLIG_NUM_CHANNELS])out_;
    if (!s || !out_) return 0;

    /* Initialise the 3×3 grid: zeroed cells, num_cells = 0, but tag
     * channel/scale_level so renderers can still match buckets. */
    for (uint8_t lvl = 0; lvl < SLIG_NUM_LEVELS; ++lvl) {
        for (uint8_t ch = 0; ch < SLIG_NUM_CHANNELS; ++ch) {
            SligCellSet *set = &out[lvl][ch];
            memset(set, 0, sizeof(*set));
            set->scale_level = lvl;
            set->channel = ch;
        }
    }

    uint32_t loaded = 0;
    for (uint32_t i = 0; i < s->count; ++i) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->type != CE_TYPE_SLIG)        continue;
        if (e->canvas_id != canvas_id)      continue;
        uint16_t slot = e->slot;
        uint8_t lvl = (uint8_t)(slot / SLIG_NUM_CHANNELS);
        uint8_t ch  = (uint8_t)(slot % SLIG_NUM_CHANNELS);
        if (lvl >= SLIG_NUM_LEVELS) continue;
        uint16_t idx = e->block_idx;
        if (idx >= SLIG_MAX_CELLS)  continue;

        SligCellSet *set = &out[lvl][ch];
        set->cells[idx] = e->keyframe;
        if (idx + 1 > set->num_cells) set->num_cells = idx + 1;
        ++loaded;
    }
    return loaded;
}

/* qsort comparator over CEStorage entry indices, sorted by their
 * derived TickRGBA. The comparator takes a context pointer (the
 * storage we are indexing into) but qsort_r isn't portable to MinGW,
 * so we use a thread-local pointer and qsort. ce_tick_sorted_indices
 * is the only call site, so the static is bounded. */
static const CEStorage *g_tick_sort_storage;
static int tick_index_cmp(const void *pa, const void *pb) {
    uint32_t ia = *(const uint32_t *)pa;
    uint32_t ib = *(const uint32_t *)pb;
    const CEStorageEntry *ea = &g_tick_sort_storage->entries[ia];
    const CEStorageEntry *eb = &g_tick_sort_storage->entries[ib];
    TickRGBA ta = ce_tick_from_entry(ea);
    TickRGBA tb = ce_tick_from_entry(eb);
    return ce_tick_compare(ta, tb);
}

uint32_t ce_tick_sorted_indices(const CEStorage *s,
                                uint32_t         canvas_id,
                                uint32_t         allowed_type_mask,
                                uint32_t        *out_idx,
                                uint32_t         cap) {
    if (!s || !out_idx || cap == 0) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->count && n < cap; ++i) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->canvas_id != canvas_id) continue;
        uint32_t bit = 1u << (uint32_t)e->type;
        if ((allowed_type_mask & bit) == 0) continue;
        out_idx[n++] = i;
    }
    if (n > 1) {
        g_tick_sort_storage = s;
        qsort(out_idx, n, sizeof(uint32_t), tick_index_cmp);
        g_tick_sort_storage = NULL;
    }
    return n;
}
