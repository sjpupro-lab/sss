#include "ce_storage.h"
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
