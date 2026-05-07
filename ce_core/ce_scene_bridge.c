/* ce_scene_bridge.c — image decomposition -> CESceneObject bridge.
 *
 * Integer-only. Honours the existing 16x16 block size from
 * ce_feed_image.h (CE_IMAGE_BLOCK16_PX) and reuses additive blending
 * via ce_scene_render. No JSON: every persisted artefact lives inside
 * CEStorage as a CE_TYPE_SCENE entry produced by
 * ce_storage_persist_scene_object.
 */

#include "ce_scene_bridge.h"
#include "ce_feed_image.h"

#include <stdint.h>
#include <string.h>

/* ── Per-block signature, integer-only ──────────────────────────────
 * For a single 16x16 RGBA block we extract:
 *   avg_r, avg_g, avg_b   — channel means (0..255)
 *   texture_energy        — sum of |Y - mean(Y)| / N, scaled to 0..255
 *   edge_strength         — max abs gradient (Sobel-lite, 4-tap), 0..255
 *
 * All operations stay in 32-bit integers; division is the only
 * non-trivial step and the divisors are powers of two or constants.
 */
typedef struct {
    uint8_t r, g, b;
    uint8_t texture;
    uint8_t edge;
    uint16_t bx;
    uint16_t by;
} BlockSig;

#define BRIDGE_BLOCK CE_IMAGE_BLOCK16_PX  /* 16 — must not change */

static void compute_block_signature(const uint8_t *rgba,
                                    int width, int height,
                                    int bx, int by,
                                    BlockSig *out) {
    int x0 = bx * BRIDGE_BLOCK;
    int y0 = by * BRIDGE_BLOCK;
    uint32_t sumR = 0, sumG = 0, sumB = 0, count = 0;
    uint32_t y_acc[BRIDGE_BLOCK * BRIDGE_BLOCK];
    int valid_w = (x0 + BRIDGE_BLOCK <= width)  ? BRIDGE_BLOCK : (width  - x0);
    int valid_h = (y0 + BRIDGE_BLOCK <= height) ? BRIDGE_BLOCK : (height - y0);
    if (valid_w <= 0 || valid_h <= 0) {
        out->r = out->g = out->b = out->texture = out->edge = 0;
        out->bx = (uint16_t)bx;
        out->by = (uint16_t)by;
        return;
    }

    for (int dy = 0; dy < valid_h; ++dy) {
        for (int dx = 0; dx < valid_w; ++dx) {
            const uint8_t *p = rgba + (((size_t)(y0 + dy) * (size_t)width)
                                       + (size_t)(x0 + dx)) * 4u;
            uint32_t r = p[0], g = p[1], b = p[2];
            sumR += r; sumG += g; sumB += b;
            /* Y channel: integer Rec.601 (66*R + 129*G + 25*B + 128) >> 8 + 16
             * — keep stays in [16, 235], plenty of head-room. */
            uint32_t y = ((66u * r + 129u * g + 25u * b + 128u) >> 8) + 16u;
            y_acc[count++] = y;
        }
    }

    out->bx = (uint16_t)bx;
    out->by = (uint16_t)by;
    if (count == 0) {
        out->r = out->g = out->b = out->texture = out->edge = 0;
        return;
    }
    out->r = (uint8_t)(sumR / count);
    out->g = (uint8_t)(sumG / count);
    out->b = (uint8_t)(sumB / count);

    uint32_t y_mean = 0;
    for (uint32_t i = 0; i < count; ++i) y_mean += y_acc[i];
    y_mean /= count;

    uint32_t mad = 0;
    for (uint32_t i = 0; i < count; ++i) {
        int32_t d = (int32_t)y_acc[i] - (int32_t)y_mean;
        if (d < 0) d = -d;
        mad += (uint32_t)d;
    }
    mad /= count;
    if (mad > 255u) mad = 255u;
    out->texture = (uint8_t)mad;

    /* 4-tap edge: |LR| + |UD| from quadrant means. Keeps the math
     * aligned with ce_feed_image's gradient encoding. */
    if (valid_w >= 2 && valid_h >= 2) {
        uint32_t qTL = 0, qTR = 0, qBL = 0, qBR = 0, n = 0;
        int hw = valid_w / 2, hh = valid_h / 2;
        if (hw < 1) hw = 1;
        if (hh < 1) hh = 1;
        for (int dy = 0; dy < valid_h; ++dy) {
            for (int dx = 0; dx < valid_w; ++dx) {
                const uint8_t *p = rgba + (((size_t)(y0 + dy) * (size_t)width)
                                           + (size_t)(x0 + dx)) * 4u;
                uint32_t y = ((66u * p[0] + 129u * p[1] + 25u * p[2] + 128u) >> 8) + 16u;
                if (dy < hh && dx < hw) qTL += y;
                else if (dy < hh)       qTR += y;
                else if (dx < hw)       qBL += y;
                else                    qBR += y;
                ++n;
            }
        }
        uint32_t qa = qTL / (uint32_t)(hw * hh);
        uint32_t qb = qTR / (uint32_t)((valid_w - hw) * hh);
        uint32_t qc = qBL / (uint32_t)(hw * (valid_h - hh));
        uint32_t qd = qBR / (uint32_t)((valid_w - hw) * (valid_h - hh));
        int32_t lr = ((int32_t)qa + (int32_t)qc) - ((int32_t)qb + (int32_t)qd);
        int32_t ud = ((int32_t)qa + (int32_t)qb) - ((int32_t)qc + (int32_t)qd);
        if (lr < 0) lr = -lr;
        if (ud < 0) ud = -ud;
        uint32_t e = (uint32_t)(lr + ud) / 2u;
        if (e > 255u) e = 255u;
        out->edge = (uint8_t)e;
        (void)n;
    } else {
        out->edge = 0;
    }
}

/* ── Cluster: union-find over neighbouring blocks with similar
 *           color and texture. The threshold uses per-channel SAD plus
 *           a texture delta — both in integer 0..255 space. ───────── */
#define BRIDGE_MAX_BLOCKS 4096   /* up to 64x64 blocks (1024x1024 image) */

typedef struct {
    int parent;
} DSU;

static int dsu_find(DSU *u, int i) {
    while (u[i].parent != i) {
        u[i].parent = u[u[i].parent].parent;
        i = u[i].parent;
    }
    return i;
}

static void dsu_union(DSU *u, int a, int b) {
    int ra = dsu_find(u, a);
    int rb = dsu_find(u, b);
    if (ra != rb) u[ra].parent = rb;
}

static int similar(const BlockSig *a, const BlockSig *b) {
    int dr = (int)a->r - (int)b->r; if (dr < 0) dr = -dr;
    int dg = (int)a->g - (int)b->g; if (dg < 0) dg = -dg;
    int db = (int)a->b - (int)b->b; if (db < 0) db = -db;
    int dt = (int)a->texture - (int)b->texture; if (dt < 0) dt = -dt;
    int color_sad = dr + dg + db;        /* 0..765 */
    if (color_sad > 60) return 0;        /* perceptually different colour */
    if (dt > 35)        return 0;        /* very different texture */
    return 1;
}

/* Map a CEMoveProfile back to a wave + keyframe pattern. The keyframe
 * pattern is what ce_scene_tick (alone, no profile) will play back —
 * STATIC stays put, FLOW slides across, ORGANIC traces a small loop,
 * RIGID jitters around base, PARTICLE drifts upward. The profile is
 * still kept around for ce_scene_tick_with_profiles which adds the
 * fine-grained per-tick offsets on top. */
static void seed_keyframes_from_motion(CESceneObject *obj,
                                       const CEMoveProfile *p,
                                       uint8_t base_cx, uint8_t base_cy) {
    obj->keyframe_count = 0;
    obj->current_key = 0;
    /* Start always at base position. */
    ce_scene_add_keyframe(obj, 0, base_cx, base_cy, CE_MOVE_SMOOTH);

    switch (p->type) {
    case CE_MOTION_STATIC: {
        /* Two identical keyframes far in the future = no motion. */
        ce_scene_add_keyframe(obj, 60, base_cx, base_cy, CE_MOVE_SMOOTH);
        break;
    }
    case CE_MOTION_FLOW: {
        /* Horizontal sweep across 60 ticks. dx=3 typical -> ~+22 px on
         * a 256-wide canvas. We pick a 50-step sweep, then loop. */
        int target = (int)base_cx + 60;
        if (target > 240) target = 240;
        ce_scene_add_keyframe(obj, 30, (uint8_t)target,
                              (uint8_t)((int)base_cy + 4), CE_MOVE_SMOOTH);
        target = (int)base_cx;
        ce_scene_add_keyframe(obj, 60, (uint8_t)target, base_cy, CE_MOVE_SMOOTH);
        break;
    }
    case CE_MOTION_ORGANIC: {
        /* Small circular sway. */
        int dx = 3, dy = 3;
        ce_scene_add_keyframe(obj, 15, (uint8_t)((int)base_cx + dx),
                              (uint8_t)((int)base_cy + dy), CE_MOVE_SMOOTH);
        ce_scene_add_keyframe(obj, 30, base_cx,
                              (uint8_t)((int)base_cy + 2 * dy), CE_MOVE_SMOOTH);
        ce_scene_add_keyframe(obj, 45, (uint8_t)((int)base_cx - dx),
                              (uint8_t)((int)base_cy + dy), CE_MOVE_SMOOTH);
        ce_scene_add_keyframe(obj, 60, base_cx, base_cy, CE_MOVE_SMOOTH);
        break;
    }
    case CE_MOTION_RIGID: {
        ce_scene_add_keyframe(obj, 30, (uint8_t)((int)base_cx + 1),
                              (uint8_t)((int)base_cy - 1), CE_MOVE_SMOOTH);
        ce_scene_add_keyframe(obj, 60, base_cx, base_cy, CE_MOVE_SMOOTH);
        break;
    }
    case CE_MOTION_PARTICLE:
    default: {
        int up = (int)base_cy - 30;
        if (up < 10) up = 10;
        ce_scene_add_keyframe(obj, 60, base_cx, (uint8_t)up, CE_MOVE_EASE_OUT);
        break;
    }
    }
}

/* Map a cluster signature to the object's wave parameters. The mapping
 * is intentionally simple and integer-only:
 *   wave 0 : sin, freq scales with 1+(texture/32), amp = brightness/2
 *   wave 1 : tri, freq derived from edge strength
 *   wave 2 : sin, low-frequency carrier proportional to colour energy
 * Up to CE_SCENE_MAX_WAVES are added; the bridge stops at 3 to keep
 * the additive render budget predictable. */
static void seed_waves_from_signature(CESceneObject *obj,
                                      uint8_t avg_r, uint8_t avg_g, uint8_t avg_b,
                                      uint8_t texture, uint8_t edge,
                                      uint16_t cluster_idx) {
    int brightness = ((int)avg_r + (int)avg_g + (int)avg_b) / 3;
    uint8_t f0 = (uint8_t)(2 + (texture >> 5));
    uint8_t a0 = (uint8_t)(brightness / 2 + 32);
    uint8_t p0 = (uint8_t)((cluster_idx * 37) & 0xFF);
    ce_scene_add_wave(obj, f0 ? f0 : 1, a0, p0, 0);

    uint8_t f1 = (uint8_t)(4 + (edge >> 4));
    uint8_t a1 = (uint8_t)(48 + (texture / 4));
    uint8_t p1 = (uint8_t)((cluster_idx * 113 + 64) & 0xFF);
    ce_scene_add_wave(obj, f1 ? f1 : 1, a1, p1, 1);

    uint8_t f2 = 1;
    uint8_t a2 = (uint8_t)(brightness / 3 + 16);
    uint8_t p2 = (uint8_t)((cluster_idx * 191 + 128) & 0xFF);
    ce_scene_add_wave(obj, f2, a2, p2, 0);
}

uint16_t ce_scene_build_from_rgba(CEScene *scene,
                                  CESceneProfiles *profiles,
                                  const uint8_t *rgba,
                                  int width, int height,
                                  uint16_t base_id) {
    if (!scene || !profiles || !rgba || width <= 0 || height <= 0) return 0;
    ce_scene_init(scene);
    memset(profiles, 0, sizeof(*profiles));

    int bw = (width  + BRIDGE_BLOCK - 1) / BRIDGE_BLOCK;
    int bh = (height + BRIDGE_BLOCK - 1) / BRIDGE_BLOCK;
    if (bw * bh > BRIDGE_MAX_BLOCKS) return 0;

    /* Per-block signature pass. */
    static BlockSig sig[BRIDGE_MAX_BLOCKS];
    static DSU      dsu[BRIDGE_MAX_BLOCKS];
    int total = bw * bh;
    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx)
            compute_block_signature(rgba, width, height, bx, by,
                                    &sig[by * bw + bx]);
    for (int i = 0; i < total; ++i) dsu[i].parent = i;

    /* 4-neighbour clustering. */
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            int i = by * bw + bx;
            if (bx + 1 < bw) {
                int j = i + 1;
                if (similar(&sig[i], &sig[j])) dsu_union(dsu, i, j);
            }
            if (by + 1 < bh) {
                int j = i + bw;
                if (similar(&sig[i], &sig[j])) dsu_union(dsu, i, j);
            }
        }
    }

    /* Aggregate per-cluster stats. */
    typedef struct {
        uint32_t sumR, sumG, sumB, sumT, sumE;
        uint32_t sumX, sumY, count;
        int      root;
    } Cluster;
    static Cluster cl[BRIDGE_MAX_BLOCKS];
    static int     root_to_cl[BRIDGE_MAX_BLOCKS];
    for (int i = 0; i < total; ++i) root_to_cl[i] = -1;

    int cluster_count = 0;
    for (int i = 0; i < total; ++i) {
        int r = dsu_find(dsu, i);
        int ci = root_to_cl[r];
        if (ci < 0) {
            ci = cluster_count++;
            root_to_cl[r] = ci;
            memset(&cl[ci], 0, sizeof(cl[ci]));
            cl[ci].root = r;
        }
        Cluster *c = &cl[ci];
        c->sumR += sig[i].r;
        c->sumG += sig[i].g;
        c->sumB += sig[i].b;
        c->sumT += sig[i].texture;
        c->sumE += sig[i].edge;
        c->sumX += sig[i].bx;
        c->sumY += sig[i].by;
        c->count += 1;
    }

    /* Drop tiny clusters (noise). Threshold 3 blocks = ~48x16 px area. */
    const uint32_t MIN_BLOCKS = 3;

    /* Sort clusters by descending block count so the largest ones land
     * first inside CE_SCENE_MAX_OBJECTS. Selection sort over indices is
     * fine because cluster_count is bounded by total blocks (<= 4096),
     * but we early-out once we've emitted CE_SCENE_MAX_OBJECTS objects. */
    static int order[BRIDGE_MAX_BLOCKS];
    int kept = 0;
    for (int i = 0; i < cluster_count; ++i) order[i] = i;
    for (int i = 0; i < cluster_count; ++i) {
        int best = i;
        for (int j = i + 1; j < cluster_count; ++j) {
            if (cl[order[j]].count > cl[order[best]].count) best = j;
        }
        if (best != i) { int t = order[i]; order[i] = order[best]; order[best] = t; }
    }

    for (int oi = 0; oi < cluster_count && kept < CE_SCENE_MAX_OBJECTS; ++oi) {
        Cluster *c = &cl[order[oi]];
        if (c->count < MIN_BLOCKS) break;  /* sorted desc, so we're done */

        uint8_t avg_r = (uint8_t)(c->sumR / c->count);
        uint8_t avg_g = (uint8_t)(c->sumG / c->count);
        uint8_t avg_b = (uint8_t)(c->sumB / c->count);
        uint8_t tex   = (uint8_t)(c->sumT / c->count);
        uint8_t edge  = (uint8_t)(c->sumE / c->count);

        /* Centroid -> normalised cx/cy in 0..255. */
        uint32_t cx_blk = (c->sumX / c->count); /* in block units */
        uint32_t cy_blk = (c->sumY / c->count);
        int cx_px = (int)(cx_blk * BRIDGE_BLOCK + BRIDGE_BLOCK / 2);
        int cy_px = (int)(cy_blk * BRIDGE_BLOCK + BRIDGE_BLOCK / 2);
        uint8_t cx256 = (uint8_t)((cx_px * 255) / (width  > 1 ? width  - 1 : 1));
        uint8_t cy256 = (uint8_t)((cy_px * 255) / (height > 1 ? height - 1 : 1));

        /* Radius proportional to sqrt(count) — cheap integer approx. */
        uint32_t r_est = 1;
        while (r_est * r_est < c->count) ++r_est;
        if (r_est > 80) r_est = 80;
        uint8_t radius = (uint8_t)(20 + r_est * 4);

        CESceneObject *obj = ce_scene_add_object(scene,
                                                 (uint16_t)(base_id + kept));
        if (!obj) break;
        obj->color_r = avg_r;
        obj->color_g = avg_g;
        obj->color_b = avg_b;
        obj->cx = cx256;
        obj->cy = cy256;
        obj->interp_cx_256 = (int16_t)((int)cx256 << 8);
        obj->interp_cy_256 = (int16_t)((int)cy256 << 8);
        obj->radius = radius;
        obj->opacity = 200;

        seed_waves_from_signature(obj, avg_r, avg_g, avg_b, tex, edge,
                                  (uint16_t)kept);

        /* Auto-call ce_infer_motion_profile per spec. */
        CEMoveProfile prof = ce_infer_motion_profile(
            avg_r, avg_g, avg_b, tex, edge, cy256,
            (uint16_t)c->count);

        seed_keyframes_from_motion(obj, &prof, cx256, cy256);

        profiles->profiles[kept] = prof;
        profiles->base_cx [kept] = cx256;
        profiles->base_cy [kept] = cy256;
        ++kept;
    }
    profiles->count = (uint16_t)kept;
    return (uint16_t)kept;
}

uint16_t ce_scene_persist_to_storage(CEStorage *storage,
                                     uint32_t canvas_id,
                                     uint16_t scene_id,
                                     const CEScene *scene) {
    if (!storage || !scene) return 0;
    uint16_t n = 0;
    for (uint16_t i = 0; i < scene->object_count; ++i) {
        if (ce_storage_persist_scene_object(storage, canvas_id, scene_id,
                                            &scene->objects[i])) ++n;
    }
    return n;
}

void ce_scene_tick_with_profiles(CEScene *scene, CESceneProfiles *profiles) {
    if (!scene) return;
    /* Run the standard keyframe interpolator first. */
    ce_scene_tick(scene);
    if (!profiles) return;
    /* Then layer the per-profile per-tick offset. STATIC pins the
     * object back to base; everything else displaces from base. */
    uint32_t tick = scene->current_tick;
    uint16_t n = profiles->count < scene->object_count
                 ? profiles->count : scene->object_count;
    for (uint16_t i = 0; i < n; ++i) {
        ce_move_apply_tick(&scene->objects[i],
                           &profiles->profiles[i],
                           profiles->base_cx[i],
                           profiles->base_cy[i],
                           tick);
    }
}
