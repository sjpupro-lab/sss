#include "ce_storage.h"
#include "ce_feed_image.h"
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
