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

/* ─── Atmos scene-object persistence (CE_TYPE_SCENE) ────────────────
 *
 * The signature packing layout is shared with ce_scene_bridge.c and
 * matches the byte map advertised in ce_storage.h. Keeping the packer
 * here (alongside the storage append) guarantees that the bytes that
 * land in CEStorage are exactly what ce_storage_match_scene_signature
 * reads — no per-call drift. */
#include "ce_scene_object.h"

static void scene_object_pack_signature(CEUnit *out,
                                        const struct CESceneObject_s *obj) {
    /* Layout (deterministic, byte-exact). 64 bytes total.
     *
     *   inc.R.plus[0..3]  = color_r, color_g, color_b, brightness
     *   inc.R.minus[0..3] = id_lo, id_hi, group_lo, group_hi
     *   inc.G.plus[0..3]  = waves[0..3].freq
     *   inc.G.minus[0..3] = waves[4..7].freq
     *   inc.B.plus[0..3]  = waves[0..3].amp
     *   inc.B.minus[0..3] = waves[4..7].amp
     *   inc.A.plus[0..3]  = radius, opacity, cx, cy
     *   inc.A.minus[0..3] = wave_count, kf_count, color_r, color_g
     *
     *   dec.R.plus[0..3]  = waves[0..3].phase
     *   dec.R.minus[0..3] = waves[4..7].phase
     *   dec.G.plus[0..3]  = waves[0..3].waveform
     *   dec.G.minus[0..3] = waves[4..7].waveform
     *   dec.B.plus[0..3]  = color_b, opacity, master_marker(0xA7), reserved
     *   dec.B.minus[0..3] = interp_cx_lo, interp_cx_hi, interp_cy_lo, interp_cy_hi
     *   dec.A.plus[0..3]  = current_key_lo, current_key_hi, scene_marker(0x5C), version(0x02)
     *   dec.A.minus[0..3] = 0, 0, 0, 0
     */
    uint8_t *b = ce_bytes(out);
    for (int i = 0; i < 64; i++) b[i] = 0;

    int brightness = ((int)obj->color_r + (int)obj->color_g + (int)obj->color_b) / 3;

    b[ 0] = obj->color_r;
    b[ 1] = obj->color_g;
    b[ 2] = obj->color_b;
    b[ 3] = (uint8_t)brightness;
    b[ 4] = (uint8_t)(obj->id & 0xFF);
    b[ 5] = (uint8_t)((obj->id >> 8) & 0xFF);
    b[ 6] = (uint8_t)(obj->group_id & 0xFF);
    b[ 7] = (uint8_t)((obj->group_id >> 8) & 0xFF);
    for (int i = 0; i < 4; i++) {
        b[ 8 + i] = (i < (int)obj->wave_count) ? obj->waves[i].freq : 0;
        b[12 + i] = (i + 4 < (int)obj->wave_count) ? obj->waves[i + 4].freq : 0;
    }
    for (int i = 0; i < 4; i++) {
        b[16 + i] = (i < (int)obj->wave_count) ? obj->waves[i].amp : 0;
        b[20 + i] = (i + 4 < (int)obj->wave_count) ? obj->waves[i + 4].amp : 0;
    }
    b[24] = obj->radius;
    b[25] = obj->opacity;
    b[26] = obj->cx;
    b[27] = obj->cy;
    b[28] = obj->wave_count;
    b[29] = (uint8_t)(obj->keyframe_count & 0xFF);
    b[30] = obj->color_r;
    b[31] = obj->color_g;

    for (int i = 0; i < 4; i++) {
        b[32 + i] = (i < (int)obj->wave_count) ? obj->waves[i].phase : 0;
        b[36 + i] = (i + 4 < (int)obj->wave_count) ? obj->waves[i + 4].phase : 0;
    }
    for (int i = 0; i < 4; i++) {
        b[40 + i] = (i < (int)obj->wave_count) ? obj->waves[i].waveform : 0;
        b[44 + i] = (i + 4 < (int)obj->wave_count) ? obj->waves[i + 4].waveform : 0;
    }
    b[48] = obj->color_b;
    b[49] = obj->opacity;
    b[50] = 0xA7;
    b[51] = 0;
    b[52] = (uint8_t)(obj->interp_cx_256 & 0xFF);
    b[53] = (uint8_t)((obj->interp_cx_256 >> 8) & 0xFF);
    b[54] = (uint8_t)(obj->interp_cy_256 & 0xFF);
    b[55] = (uint8_t)((obj->interp_cy_256 >> 8) & 0xFF);
    b[56] = (uint8_t)(obj->current_key & 0xFF);
    b[57] = (uint8_t)((obj->current_key >> 8) & 0xFF);
    b[58] = 0x5C;
    b[59] = 0x02;
}

uint32_t ce_storage_persist_scene_object(CEStorage *s,
                                         uint32_t canvas_id,
                                         uint16_t scene_id,
                                         const struct CESceneObject_s *obj) {
    if (!s || !obj) return 0;

    CEUnit signature;
    scene_object_pack_signature(&signature, obj);

    /* Chain delta against the most recent CE_TYPE_SCENE entry under
     * (canvas_id, scene_id). Zero CEUnit if none yet. */
    CEUnit anchor;
    ce_init(&anchor);
    int32_t last_idx = -1;
    for (uint32_t i = 0; i < s->count; i++) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->type != CE_TYPE_SCENE)   continue;
        if (e->canvas_id != canvas_id)  continue;
        if (e->slot != scene_id)        continue;
        last_idx = (int32_t)i;
    }
    if (last_idx >= 0) anchor = s->entries[last_idx].keyframe;

    CEUnit delta;
    ce_delta(&delta, &anchor, &signature);

    ce_storage_add_typed(s, canvas_id, scene_id, obj->id, CE_TYPE_SCENE,
                         &signature, &delta);
    return 1;
}

int ce_storage_match_scene_signature(const CEStorage *s,
                                     const struct CESceneObject_s *query,
                                     uint16_t scene_id_filter,
                                     uint16_t *out_object_id,
                                     uint32_t *out_distance) {
    if (!s || !query) return 0;

    CEUnit q;
    scene_object_pack_signature(&q, query);

    int found = 0;
    uint32_t best = 0xFFFFFFFFu;
    uint16_t best_id = 0;
    for (uint32_t i = 0; i < s->count; i++) {
        const CEStorageEntry *e = &s->entries[i];
        if (e->type != CE_TYPE_SCENE) continue;
        if (scene_id_filter != 0 && e->slot != scene_id_filter) continue;
        uint32_t d = ce_distance(&q, &e->keyframe);
        if (!found || d < best) {
            best = d;
            best_id = e->block_idx;
            found = 1;
        }
    }
    if (found) {
        if (out_object_id) *out_object_id = best_id;
        if (out_distance)  *out_distance  = best;
        return 1;
    }
    return 0;
}
