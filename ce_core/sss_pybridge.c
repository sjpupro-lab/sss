/* sss_pybridge.c — see sss_pybridge.h. */
#include "sss_pybridge.h"
#include "ce_core.h"
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_type.h"

#include <stdlib.h>
#include <string.h>

/* Per-(canvas_id, slot) tail entry: tracks the last block_idx assigned
 * and the keyframe of the most-recent cell so the next append computes
 * delta against it. Linear scan is fine — Python's CEMemory uses 4
 * slots × 1 canvas in the upgrade loop. */
typedef struct {
    uint32_t canvas_id;
    uint16_t slot;
    uint16_t _pad;
    /* uint32 so we can detect overflow before truncating to uint16 */
    uint32_t next_block_idx;
    int      has_prev;
    CEUnit   prev_kf;
} sss_tail;

struct sss_memory {
    CEStorage storage;
    sss_tail *tails;
    uint32_t  tail_count;
    uint32_t  tail_capacity;
};

static sss_tail *tail_lookup_or_insert(sss_memory *m,
                                       uint32_t canvas_id,
                                       uint16_t slot) {
    for (uint32_t i = 0; i < m->tail_count; ++i) {
        if (m->tails[i].canvas_id == canvas_id && m->tails[i].slot == slot)
            return &m->tails[i];
    }
    if (m->tail_count >= m->tail_capacity) {
        uint32_t cap = m->tail_capacity ? m->tail_capacity * 2 : 8;
        sss_tail *p = (sss_tail *)realloc(m->tails, cap * sizeof(sss_tail));
        if (!p) return NULL;
        memset(p + m->tail_capacity, 0, (cap - m->tail_capacity) * sizeof(sss_tail));
        m->tails = p;
        m->tail_capacity = cap;
    }
    sss_tail *t = &m->tails[m->tail_count++];
    t->canvas_id = canvas_id;
    t->slot = slot;
    t->next_block_idx = 0;
    t->has_prev = 0;
    return t;
}

sss_memory *sss_memory_create(uint32_t initial_capacity) {
    sss_memory *m = (sss_memory *)calloc(1, sizeof(sss_memory));
    if (!m) return NULL;
    ce_storage_init(&m->storage, initial_capacity);
    return m;
}

void sss_memory_destroy(sss_memory *m) {
    if (!m) return;
    ce_storage_free(&m->storage);
    free(m->tails);
    free(m);
}

uint32_t sss_memory_add_typed(sss_memory   *m,
                              uint32_t      canvas_id,
                              uint16_t      slot,
                              uint8_t       type,
                              const uint8_t *bytes,
                              uint32_t      len) {
    if (!m || (!bytes && len > 0)) return UINT32_MAX;

    sss_tail *t = tail_lookup_or_insert(m, canvas_id, slot);
    if (!t) return UINT32_MAX;
    /* CEStorageEntry.block_idx is uint16; refuse to wrap. */
    if (t->next_block_idx > 0xFFFFu) return UINT32_MAX;

    CEUnit fresh; ce_init(&fresh);
    if (len > 0) ce_feed(&fresh, bytes, len);

    CEUnit anchor;
    if (t->has_prev) anchor = t->prev_kf;
    else             ce_init(&anchor);

    CEUnit delta;
    ce_delta(&delta, &anchor, &fresh);

    uint16_t bidx = (uint16_t)t->next_block_idx;
    uint32_t before = m->storage.count;
    ce_storage_add_typed(&m->storage, canvas_id, slot, bidx,
                         (CEType)type, &fresh, &delta);
    if (m->storage.count == before) return UINT32_MAX;  /* OOM */

    t->prev_kf = fresh;
    t->has_prev = 1;
    t->next_block_idx = (uint32_t)bidx + 1u;  /* may reach 0x10000 -> next add errors */
    return (uint32_t)bidx;
}

uint32_t sss_memory_count(const sss_memory *m) {
    if (!m) return 0;
    return m->storage.count;
}

uint32_t sss_memory_count_by_slot(const sss_memory *m,
                                  uint32_t canvas_id,
                                  uint16_t slot) {
    if (!m) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->storage.count; ++i) {
        const CEStorageEntry *e = &m->storage.entries[i];
        if (e->canvas_id == canvas_id && e->slot == slot) ++n;
    }
    return n;
}

int sss_memory_save(const sss_memory *m, const char *path) {
    if (!m || !path) return 0;
    return ce_storage_save(&m->storage, path);
}

int sss_memory_load(sss_memory *m, const char *path) {
    if (!m || !path) return 0;
    ce_storage_free(&m->storage);
    if (!ce_storage_load(&m->storage, path)) {
        ce_storage_init(&m->storage, 16);
        return 0;
    }
    /* Rebuild tail map so subsequent adds chain correctly. .ces files
     * carry no per-(canvas,slot) ordering guarantee, so we pick the
     * entry with the highest block_idx as the chain anchor — that's
     * the cell the next append must compute its delta against. */
    free(m->tails);
    m->tails = NULL;
    m->tail_count = m->tail_capacity = 0;
    for (uint32_t i = 0; i < m->storage.count; ++i) {
        const CEStorageEntry *e = &m->storage.entries[i];
        sss_tail *t = tail_lookup_or_insert(m, e->canvas_id, e->slot);
        if (!t) continue;
        uint32_t entry_idx_p1 = (uint32_t)e->block_idx + 1u;
        if (!t->has_prev || entry_idx_p1 > t->next_block_idx) {
            t->next_block_idx = entry_idx_p1;
            t->prev_kf = e->keyframe;
            t->has_prev = 1;
        }
    }
    return 1;
}

int sss_memory_get_keyframe(const sss_memory *m,
                            uint32_t canvas_id,
                            uint16_t slot,
                            uint16_t block_idx,
                            uint8_t  out[64]) {
    if (!m || !out) return 0;
    int32_t idx = ce_storage_find(&m->storage, canvas_id, slot, block_idx);
    if (idx < 0) return 0;
    memcpy(out, ce_cbytes(&m->storage.entries[idx].keyframe), 64);
    return 1;
}
