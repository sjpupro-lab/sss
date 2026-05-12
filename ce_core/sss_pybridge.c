/* sss_pybridge.c — see sss_pybridge.h. */
#include "sss_pybridge.h"
#include "ce_core.h"
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_type.h"
#include "sss_rowvae.h"
#include "ce_scene_object.h"
#include "ce_scene_bridge.h"
#include "ce_move_profile.h"
#include "sss_feature_bank.h"

#include <stdio.h>
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

/* ── Atmos scene-object pybridge wrappers ─────────────────────────── */

struct sss_atmos_scene {
    CEScene         scene;
    CESceneProfiles profiles;
};

sss_atmos_scene *sss_atmos_scene_create(void) {
    sss_atmos_scene *s = (sss_atmos_scene *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    ce_scene_init(&s->scene);
    /* profiles is zero-initialised by calloc — count = 0, all clear. */
    return s;
}

void sss_atmos_scene_destroy(sss_atmos_scene *s) {
    if (!s) return;
    free(s);
}

uint16_t sss_atmos_scene_build_from_rgba(sss_atmos_scene *s,
                                         const uint8_t *rgba,
                                         int width, int height,
                                         uint16_t base_id) {
    if (!s || !rgba || width <= 0 || height <= 0) return 0;
    return ce_scene_build_from_rgba(&s->scene, &s->profiles,
                                    rgba, width, height, base_id);
}

uint16_t sss_atmos_scene_object_count(const sss_atmos_scene *s) {
    if (!s) return 0;
    return s->scene.object_count;
}

uint16_t sss_atmos_scene_dump_objects(const sss_atmos_scene *s,
                                      uint8_t *out_rows,
                                      uint16_t max_rows) {
    if (!s || !out_rows) return 0;
    uint16_t n = s->scene.object_count;
    if (max_rows < n) n = max_rows;
    /* Motion type comes from the profiles bag when the scene was built
     * via build_from_rgba; hand-built scenes default to STATIC so the
     * row layout is always 8 bytes / object. */
    for (uint16_t i = 0; i < n; ++i) {
        const CESceneObject *o = &s->scene.objects[i];
        uint8_t motion = (i < s->profiles.count)
            ? (uint8_t)s->profiles.profiles[i].type : 0u;
        uint8_t base_cx = (i < s->profiles.count)
            ? s->profiles.base_cx[i] : o->cx;
        uint8_t base_cy = (i < s->profiles.count)
            ? s->profiles.base_cy[i] : o->cy;
        uint8_t *row = out_rows + (size_t)i * 8u;
        row[0] = (uint8_t)(o->id & 0xFFu);
        row[1] = (uint8_t)((o->id >> 8) & 0xFFu);
        row[2] = motion;
        row[3] = base_cx;
        row[4] = base_cy;
        row[5] = o->color_r;
        row[6] = o->color_g;
        row[7] = o->color_b;
    }
    return n;
}

void sss_atmos_scene_tick(sss_atmos_scene *s) {
    if (!s) return;
    /* Always run the profile-aware tick: when profiles.count == 0
     * (hand-built scene that never went through build_from_rgba) the
     * inner ce_scene_tick still applies and the per-profile offset
     * loop is a no-op, so this entry point handles both cases. */
    ce_scene_tick_with_profiles(&s->scene, &s->profiles);
}

void sss_atmos_scene_seek(sss_atmos_scene *s, uint32_t target_tick) {
    if (!s) return;
    ce_scene_seek(&s->scene, target_tick);
}

void sss_atmos_scene_render(const sss_atmos_scene *s,
                            uint8_t *out_rgb, int width, int height) {
    if (!s || !out_rgb || width <= 0 || height <= 0) return;
    /* Zero the frame so additive accumulation never leaks state from a
     * previous render the caller's buffer happened to hold. */
    memset(out_rgb, 0, (size_t)width * (size_t)height * 3u);
    ce_scene_render(&s->scene, out_rgb, width, height);
}

uint16_t sss_atmos_scene_persist(sss_memory *mem,
                                 const sss_atmos_scene *s,
                                 uint32_t canvas_id,
                                 uint16_t scene_id) {
    if (!mem || !s) return 0;
    return ce_scene_persist_to_storage(&mem->storage, canvas_id,
                                       scene_id, &s->scene);
}

/* ── Phase 2 feature bank pybridge ─────────────────────────────────
 *
 * Python passes tightly-packed record buffers (one record_size block
 * per record, no padding). Because SSSMotif / SSSRelation / SSSIdentity
 * are pack(1) on the C side and Python lays the same bytes out via
 * struct.pack, we can reinterpret the buffer as an array of structs
 * with a single cast — no per-record copy. */

/* Phase 4 v2 extension to the pybridge save path. The buffers are
 * optional — Phase 2 callers can pass NULL/0 for conditions_buf and
 * responses_buf and behaviour is identical to the old v2-with-no-
 * conditions case (v2 file with condition_count == 0). */
int sss_pybridge_feature_bank_save(const char    *path,
                                   const uint8_t *motifs_buf,
                                   uint32_t       motif_count,
                                   const uint8_t *relations_buf,
                                   uint32_t       relation_count,
                                   const uint8_t *identities_buf,
                                   uint32_t       identity_count) {
    if (!path) return SFB_ERR_IO;
    SSSFeatureBank bank;
    memset(&bank, 0, sizeof(bank));
    bank.motif_count    = motif_count;
    bank.motifs         = (SSSMotif *)(uintptr_t)motifs_buf;
    bank.relation_count = relation_count;
    bank.relations      = (SSSRelation *)(uintptr_t)relations_buf;
    bank.identity_count = identity_count;
    bank.identities     = (SSSIdentity *)(uintptr_t)identities_buf;
    return sss_feature_bank_save(path, &bank);
}

int sss_pybridge_feature_bank_save_v2(
        const char    *path,
        const uint8_t *motifs_buf,     uint32_t motif_count,
        const uint8_t *relations_buf,  uint32_t relation_count,
        const uint8_t *identities_buf, uint32_t identity_count,
        const uint8_t *conditions_buf, uint32_t condition_count,
        const uint8_t *responses_buf,  uint32_t response_count) {
    if (!path) return SFB_ERR_IO;
    SSSFeatureBank bank;
    memset(&bank, 0, sizeof(bank));
    bank.motif_count     = motif_count;
    bank.motifs          = (SSSMotif *)(uintptr_t)motifs_buf;
    bank.relation_count  = relation_count;
    bank.relations       = (SSSRelation *)(uintptr_t)relations_buf;
    bank.identity_count  = identity_count;
    bank.identities      = (SSSIdentity *)(uintptr_t)identities_buf;
    bank.condition_count = condition_count;
    bank.conditions      = (SSSCondition *)(uintptr_t)conditions_buf;
    bank.response_count  = response_count;
    bank.responses       = (SSSInteractionResponse *)(uintptr_t)responses_buf;
    return sss_feature_bank_save(path, &bank);
}

int sss_pybridge_feature_bank_probe(const char *path,
                                    uint32_t   *out_motif_count,
                                    uint32_t   *out_relation_count,
                                    uint32_t   *out_identity_count) {
    if (!path) return SFB_ERR_IO;
    FILE *f = fopen(path, "rb");
    if (!f) return SFB_ERR_IO;
    SSSFeatureBankHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f); return SFB_ERR_IO;
    }
    fclose(f);
    if (memcmp(hdr.magic, SFB_MAGIC, 4) != 0) return SFB_ERR_MAGIC;
    /* Accept both v1 and v2 here so the Python load helper works on
     * legacy files. The loader proper handles the per-record lifting.
     */
    if (hdr.version != 1u && hdr.version != SFB_VERSION) {
        return SFB_ERR_VERSION;
    }
    if (out_motif_count)    *out_motif_count    = hdr.motif_count;
    if (out_relation_count) *out_relation_count = hdr.relation_count;
    if (out_identity_count) *out_identity_count = hdr.identity_count;
    return SFB_OK;
}

int sss_pybridge_feature_bank_load(const char *path,
                                   uint8_t   *motifs_buf,
                                   uint32_t  *out_motif_count,
                                   uint8_t   *relations_buf,
                                   uint32_t  *out_relation_count,
                                   uint8_t   *identities_buf,
                                   uint32_t  *out_identity_count) {
    if (!path || !out_motif_count || !out_relation_count
        || !out_identity_count) {
        return SFB_ERR_IO;
    }
    SSSFeatureBank bank;
    int rc = sss_feature_bank_load(path, &bank);
    if (rc != SFB_OK) return rc;

    uint32_t need_m = bank.motif_count;
    uint32_t need_r = bank.relation_count;
    uint32_t need_i = bank.identity_count;

    int short_buf = (need_m > *out_motif_count)
                 || (need_r > *out_relation_count)
                 || (need_i > *out_identity_count);
    if (short_buf) {
        /* Tell the caller the required capacity, then bail with
         * SFB_ERR_ALLOC — Python can re-allocate and call again. */
        *out_motif_count    = need_m;
        *out_relation_count = need_r;
        *out_identity_count = need_i;
        sss_feature_bank_free(&bank);
        return SFB_ERR_ALLOC;
    }

    if (need_m > 0 && motifs_buf) {
        memcpy(motifs_buf, bank.motifs, (size_t)need_m * sizeof(SSSMotif));
    }
    if (need_r > 0 && relations_buf) {
        memcpy(relations_buf, bank.relations,
               (size_t)need_r * sizeof(SSSRelation));
    }
    if (need_i > 0 && identities_buf) {
        memcpy(identities_buf, bank.identities,
               (size_t)need_i * sizeof(SSSIdentity));
    }
    *out_motif_count    = need_m;
    *out_relation_count = need_r;
    *out_identity_count = need_i;
    sss_feature_bank_free(&bank);
    return SFB_OK;
}

/* Phase 4 v2 probe: includes condition + response counts on top of
 * the Phase 2 motif / relation / identity counts. NULL out_* pointers
 * are tolerated (caller may not need every count). */
int sss_pybridge_feature_bank_probe_v2(const char *path,
                                       uint32_t   *out_motif_count,
                                       uint32_t   *out_relation_count,
                                       uint32_t   *out_identity_count,
                                       uint32_t   *out_condition_count,
                                       uint32_t   *out_response_count) {
    if (!path) return SFB_ERR_IO;
    FILE *f = fopen(path, "rb");
    if (!f) return SFB_ERR_IO;
    SSSFeatureBankHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f); return SFB_ERR_IO;
    }
    fclose(f);
    if (memcmp(hdr.magic, SFB_MAGIC, 4) != 0) return SFB_ERR_MAGIC;
    if (hdr.version != 1u && hdr.version != SFB_VERSION) {
        return SFB_ERR_VERSION;
    }
    if (out_motif_count)     *out_motif_count     = hdr.motif_count;
    if (out_relation_count)  *out_relation_count  = hdr.relation_count;
    if (out_identity_count)  *out_identity_count  = hdr.identity_count;
    if (out_condition_count) *out_condition_count =
        (hdr.version == 1u) ? 0u : (uint32_t)hdr.condition_count;
    if (out_response_count)  *out_response_count  =
        (hdr.version == 1u) ? 0u : hdr.response_count;
    return SFB_OK;
}

/* Phase 4 v2 load: fills the five record arrays and reports counts.
 * Same short-buffer convention as the v1 load (returns SFB_ERR_ALLOC
 * with the required counts written into *out_*_count). Condition /
 * response buffers may be NULL/0 when the caller doesn't care; the
 * relevant arrays just aren't copied. */
int sss_pybridge_feature_bank_load_v2(
        const char *path,
        uint8_t *motifs_buf,     uint32_t *out_motif_count,
        uint8_t *relations_buf,  uint32_t *out_relation_count,
        uint8_t *identities_buf, uint32_t *out_identity_count,
        uint8_t *conditions_buf, uint32_t *out_condition_count,
        uint8_t *responses_buf,  uint32_t *out_response_count) {
    if (!path || !out_motif_count || !out_relation_count
        || !out_identity_count || !out_condition_count
        || !out_response_count) {
        return SFB_ERR_IO;
    }
    SSSFeatureBank bank;
    int rc = sss_feature_bank_load(path, &bank);
    if (rc != SFB_OK) return rc;

    uint32_t need_m = bank.motif_count;
    uint32_t need_r = bank.relation_count;
    uint32_t need_i = bank.identity_count;
    uint32_t need_c = bank.condition_count;
    uint32_t need_p = bank.response_count;

    int short_buf = (need_m > *out_motif_count)
                 || (need_r > *out_relation_count)
                 || (need_i > *out_identity_count)
                 || (need_c > *out_condition_count)
                 || (need_p > *out_response_count);
    if (short_buf) {
        *out_motif_count     = need_m;
        *out_relation_count  = need_r;
        *out_identity_count  = need_i;
        *out_condition_count = need_c;
        *out_response_count  = need_p;
        sss_feature_bank_free(&bank);
        return SFB_ERR_ALLOC;
    }
    if (need_m > 0 && motifs_buf) {
        memcpy(motifs_buf, bank.motifs, (size_t)need_m * sizeof(SSSMotif));
    }
    if (need_r > 0 && relations_buf) {
        memcpy(relations_buf, bank.relations,
               (size_t)need_r * sizeof(SSSRelation));
    }
    if (need_i > 0 && identities_buf) {
        memcpy(identities_buf, bank.identities,
               (size_t)need_i * sizeof(SSSIdentity));
    }
    if (need_c > 0 && conditions_buf) {
        memcpy(conditions_buf, bank.conditions,
               (size_t)need_c * sizeof(SSSCondition));
    }
    if (need_p > 0 && responses_buf) {
        memcpy(responses_buf, bank.responses,
               (size_t)need_p * sizeof(SSSInteractionResponse));
    }
    *out_motif_count     = need_m;
    *out_relation_count  = need_r;
    *out_identity_count  = need_i;
    *out_condition_count = need_c;
    *out_response_count  = need_p;
    sss_feature_bank_free(&bank);
    return SFB_OK;
}
