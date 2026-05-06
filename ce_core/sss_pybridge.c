/* sss_pybridge.c — see sss_pybridge.h. */
#include "sss_pybridge.h"
#include "ce_core.h"
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_type.h"
#include "sss_rowvae.h"

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

/* ── sss_rowvae thin wrapper for ctypes. Returns 0 on success, -1 on
 * any failure. Output is uint8 RGB packed (out_w * out_h * 3); if the
 * model's size differs from out_w/out_h, the float image is resampled
 * with nearest-neighbour mapping that hits both endpoints. */
static unsigned char clamp_u8_pf(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (unsigned char)(v * 255.0f + 0.5f);
}

int sss_pybridge_generate(const char *model_path,
                          const char *prompt,
                          uint32_t    seed,
                          float       detail,
                          int         steps,
                          uint8_t    *out_rgb,
                          uint32_t    out_w,
                          uint32_t    out_h) {
    if (!model_path || !prompt || !out_rgb || out_w == 0 || out_h == 0)
        return -1;

    SSSModel model;
    memset(&model, 0, sizeof(model));
    if (sss_model_load(model_path, &model) != 0) return -1;

    SSSImage img;
    memset(&img, 0, sizeof(img));
    int rc = sss_generate(&model, prompt, seed, detail, steps, &img);
    if (rc != 0) {
        sss_image_free(&img);
        sss_model_free(&model);
        return -1;
    }

    /* Resample if requested size differs from model size. Using
     * linspace-style endpoint mapping so the corners of the source
     * map to the corners of the output. */
    int sH = img.height, sW = img.width;
    if ((uint32_t)sH == out_h && (uint32_t)sW == out_w) {
        size_t n = (size_t)sH * (size_t)sW;
        for (size_t i = 0; i < n; ++i) {
            out_rgb[i * 3 + 0] = clamp_u8_pf(img.data[i * 3 + 0]);
            out_rgb[i * 3 + 1] = clamp_u8_pf(img.data[i * 3 + 1]);
            out_rgb[i * 3 + 2] = clamp_u8_pf(img.data[i * 3 + 2]);
        }
    } else {
        for (uint32_t y = 0; y < out_h; ++y) {
            uint32_t sy = (out_h <= 1) ? 0
                : (uint32_t)((double)y * (sH - 1) / (out_h - 1) + 0.5);
            if (sy >= (uint32_t)sH) sy = sH - 1;
            for (uint32_t x = 0; x < out_w; ++x) {
                uint32_t sx = (out_w <= 1) ? 0
                    : (uint32_t)((double)x * (sW - 1) / (out_w - 1) + 0.5);
                if (sx >= (uint32_t)sW) sx = sW - 1;
                size_t spi = (size_t)sy * sW + sx;
                size_t dpi = (size_t)y  * out_w + x;
                out_rgb[dpi * 3 + 0] = clamp_u8_pf(img.data[spi * 3 + 0]);
                out_rgb[dpi * 3 + 1] = clamp_u8_pf(img.data[spi * 3 + 1]);
                out_rgb[dpi * 3 + 2] = clamp_u8_pf(img.data[spi * 3 + 2]);
            }
        }
    }

    sss_image_free(&img);
    sss_model_free(&model);
    return 0;
}

void sss_pybridge_ce_feed(const char *text, uint32_t len, uint8_t out_64[64])
{
    if (!out_64) return;
    CEUnit u;
    ce_init(&u);
    if (text && len > 0) {
        ce_feed(&u, (const uint8_t *)text, len);
    }
    memcpy(out_64, ce_cbytes(&u), 64);
}

uint32_t sss_pybridge_ce_distance(const uint8_t a_64[64], const uint8_t b_64[64])
{
    if (!a_64 || !b_64) return 0xFFFFFFFFu;
    CEUnit a, b;
    memcpy(ce_bytes(&a), a_64, 64);
    memcpy(ce_bytes(&b), b_64, 64);
    return ce_distance(&a, &b);
}
