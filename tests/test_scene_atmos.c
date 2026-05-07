/* test_scene_atmos.c — integration test for the Atmos scene-object
 * engine inside SSS.
 *
 * Verifies (each line is a separate PASS/FAIL gate following the SSS
 * test convention used by tests/test_image_roundtrip.c):
 *
 *   1. STATIC objects keep their cx/cy across 60 ticks.
 *   2. FLOW objects move horizontally over time (cx changes).
 *   3. ce_scene_seek(0) restores the initial position after advancing.
 *   4. 60 frames are rendered to PPM (build/atmos_frames/) with no
 *      blank frame.
 *   5. ce_storage_persist_scene_object + ce_storage_match_scene_signature
 *      retrieve the right object for a query that matches one stored
 *      signature, and prefer the closer signature when two are present.
 *   6. ce_scene_build_from_rgba + ce_infer_motion_profile auto-tag
 *      a synthetic blue-water region as CE_MOTION_FLOW and the
 *      uniform sky region as CE_MOTION_STATIC.
 *
 * No JSON output. Frames go straight to PPM (P6, integer-only).
 * No float / double anywhere in this file.
 */
#include "ce_scene_object.h"
#include "ce_move_profile.h"
#include "ce_scene_bridge.h"
#include "ce_storage.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define IMG_W 128
#define IMG_H 128
#define FRAMES 60

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; \
    printf("  [TEST] %s ... ", name); fflush(stdout); } while (0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while (0)
#define FAIL(msg)  do { printf("FAIL — %s\n", msg); } while (0)

static int write_ppm(const char *path, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h * 3u;
    int ok = (fwrite(rgb, 1, n, f) == n);
    fclose(f);
    return ok;
}

static int frame_has_pixel(const uint8_t *rgb, int n_bytes) {
    for (int i = 0; i < n_bytes; ++i) if (rgb[i] != 0) return 1;
    return 0;
}

/* ── Test 1 + 2 + 3 + 4: STATIC fixed, FLOW moving, seek restore,
 *                         60 frames non-blank. ─────────────────── */
static void test_motion_and_render(void) {
    CEScene scene;
    ce_scene_init(&scene);

    /* STATIC object — bright cyan, large radius, parked at centre. */
    CESceneObject *bg = ce_scene_add_object(&scene, 100);
    bg->color_r = 60; bg->color_g = 120; bg->color_b = 200;
    bg->radius  = 90; bg->opacity = 200;
    bg->cx = 64; bg->cy = 64;
    bg->interp_cx_256 = 64 << 8;
    bg->interp_cy_256 = 64 << 8;
    ce_scene_add_wave(bg, 1, 200, 0, 0);
    ce_scene_add_wave(bg, 2, 90,  64, 1);
    /* No keyframes -> ce_scene_tick is a no-op for this object. */
    uint8_t bg_cx0 = bg->cx, bg_cy0 = bg->cy;

    /* FLOW object — small warm spot moving from left to right. */
    CESceneObject *flow = ce_scene_add_object(&scene, 200);
    flow->color_r = 220; flow->color_g = 180; flow->color_b = 120;
    flow->radius  = 30;  flow->opacity = 220;
    flow->cx = 30; flow->cy = 90;
    flow->interp_cx_256 = 30 << 8;
    flow->interp_cy_256 = 90 << 8;
    ce_scene_add_wave(flow, 8, 200, 0, 0);
    ce_scene_add_wave(flow, 14, 80, 100, 1);
    ce_scene_add_keyframe(flow,  0, 30,  90, CE_MOVE_SMOOTH);
    ce_scene_add_keyframe(flow, 60, 220, 90, CE_MOVE_SMOOTH);
    uint8_t flow_cx0 = flow->cx;

    /* Render loop. */
    mkdir("build", 0755);
    mkdir("build/atmos_frames", 0755);
    uint8_t *rgb = (uint8_t *)malloc((size_t)IMG_W * IMG_H * 3u);
    if (!rgb) { FAIL("malloc"); return; }

    int blank_frames = 0;
    int static_drift = 0;
    int flow_cx_seen_max = flow->cx;
    int flow_cx_seen_min = flow->cx;
    char path[256];
    for (int f = 0; f < FRAMES; ++f) {
        ce_scene_tick(&scene);

        if (bg->cx != bg_cx0 || bg->cy != bg_cy0) static_drift++;
        if (flow->cx > flow_cx_seen_max) flow_cx_seen_max = flow->cx;
        if (flow->cx < flow_cx_seen_min) flow_cx_seen_min = flow->cx;

        memset(rgb, 0, (size_t)IMG_W * IMG_H * 3u);
        ce_scene_render(&scene, rgb, IMG_W, IMG_H);
        if (!frame_has_pixel(rgb, IMG_W * IMG_H * 3)) blank_frames++;

        snprintf(path, sizeof(path),
                 "build/atmos_frames/frame_%03d.ppm", f);
        write_ppm(path, rgb, IMG_W, IMG_H);
    }

    TEST("STATIC object keeps position over 60 ticks");
    if (static_drift == 0) PASS(); else FAIL("STATIC object drifted");

    TEST("FLOW object cx advances over 60 ticks");
    if (flow_cx_seen_max - flow_cx_seen_min >= 100 &&
        flow->cx > flow_cx0) PASS();
    else { char m[64]; snprintf(m, sizeof(m),
            "min=%d max=%d final=%d start=%d",
            flow_cx_seen_min, flow_cx_seen_max, flow->cx, flow_cx0);
            FAIL(m); }

    TEST("60 frames rendered with no blank frame");
    if (blank_frames == 0) PASS();
    else { char m[64]; snprintf(m, sizeof(m),
            "%d/%d blank", blank_frames, FRAMES); FAIL(m); }

    TEST("ce_scene_seek(0) restores initial FLOW position");
    ce_scene_seek(&scene, 0);
    if (flow->cx == flow_cx0) PASS();
    else { char m[64]; snprintf(m, sizeof(m),
            "after seek cx=%d (expected %d)", flow->cx, flow_cx0);
            FAIL(m); }

    free(rgb);
}

/* ── Test 5: CE Storage signature persist + match ───────────────── */
static void test_ce_storage_match(void) {
    CEStorage st;
    ce_storage_init(&st, 16);

    CEScene scene;
    ce_scene_init(&scene);

    /* Two distinguishable objects in scene 1. */
    CESceneObject *a = ce_scene_add_object(&scene, 11);
    a->color_r = 200; a->color_g = 50; a->color_b = 50;
    a->cx = 40; a->cy = 60; a->radius = 32; a->opacity = 200;
    ce_scene_add_wave(a, 4, 180, 0,  0);
    ce_scene_add_wave(a, 9, 120, 64, 1);

    CESceneObject *b = ce_scene_add_object(&scene, 22);
    b->color_r = 50; b->color_g = 200; b->color_b = 70;
    b->cx = 180; b->cy = 200; b->radius = 28; b->opacity = 200;
    ce_scene_add_wave(b, 3, 140, 32, 0);
    ce_scene_add_wave(b, 7, 90, 100, 1);

    uint16_t scene_id = 1;
    uint32_t canvas_id = 0xABCD;
    uint16_t persisted = ce_scene_persist_to_storage(&st, canvas_id,
                                                     scene_id, &scene);

    TEST("ce_scene_persist_to_storage writes one entry per object");
    if (persisted == 2 && st.count == 2 &&
        st.entries[0].type == CE_TYPE_SCENE &&
        st.entries[1].type == CE_TYPE_SCENE) PASS();
    else { char m[64]; snprintf(m, sizeof(m),
            "persisted=%u count=%u", persisted, st.count); FAIL(m); }

    /* Query 1: an object identical to `a` should match a's id (11)
     * with distance 0. */
    {
        CESceneObject q;
        memcpy(&q, a, sizeof(q));
        uint16_t mid = 0; uint32_t md = 0;
        int ok = ce_storage_match_scene_signature(&st, &q, scene_id,
                                                  &mid, &md);
        TEST("matcher returns exact object on identical query");
        if (ok && mid == 11 && md == 0) PASS();
        else { char m[64]; snprintf(m, sizeof(m),
                "ok=%d id=%u dist=%u", ok, mid, md); FAIL(m); }
    }

    /* Query 2: a slightly perturbed `b` should still prefer b over a. */
    {
        CESceneObject q;
        memcpy(&q, b, sizeof(q));
        q.color_r += 5; q.color_g -= 4; q.cx += 2;
        uint16_t mid = 0; uint32_t md = 0;
        int ok = ce_storage_match_scene_signature(&st, &q, scene_id,
                                                  &mid, &md);
        TEST("matcher prefers nearer signature under noise");
        if (ok && mid == 22) PASS();
        else { char m[64]; snprintf(m, sizeof(m),
                "ok=%d id=%u dist=%u (expected id=22)", ok, mid, md);
                FAIL(m); }
    }

    /* Query 3: scene-id filter should ignore unrelated scenes. */
    {
        CESceneObject q;
        memcpy(&q, a, sizeof(q));
        uint16_t mid = 0; uint32_t md = 0;
        int ok = ce_storage_match_scene_signature(&st, &q, /*scene*/ 9,
                                                  &mid, &md);
        TEST("matcher returns 0 when scene-id filter excludes all");
        if (!ok) PASS(); else FAIL("matched a foreign scene");
    }

    ce_storage_free(&st);
}

/* ── Test 6: bridge auto-classifies a synthetic 2-region image ──── */
static void test_bridge_motion_inference(void) {
    /* 64x64 RGBA: top half is uniform sky-ish bright cyan-blue
     * (=> STATIC sky), bottom half is teal water with small noise
     * (=> FLOW water). All channels integer. */
    const int W = 64, H = 64;
    uint8_t *rgba = (uint8_t *)calloc((size_t)W * H * 4u, 1);
    if (!rgba) { TEST("bridge alloc"); FAIL("malloc"); return; }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t *p = rgba + ((size_t)y * W + x) * 4u;
            if (y < H / 2) {
                /* Sky: bright, uniform, cy_norm < 60 -> STATIC. */
                p[0] = 180; p[1] = 200; p[2] = 230; p[3] = 255;
            } else {
                /* Water: teal, low texture noise -> FLOW. */
                uint8_t noise = (uint8_t)((x * 3 + y * 5) & 0x07);
                p[0] = 30 + noise;
                p[1] = 140 + noise;
                p[2] = 170 + noise;
                p[3] = 255;
            }
        }
    }

    CEScene    scene;
    CESceneProfiles profs;
    uint16_t built = ce_scene_build_from_rgba(&scene, &profs,
                                              rgba, W, H, 1);
    free(rgba);

    TEST("bridge builds at least 2 scene-objects from 2-region image");
    if (built >= 2) PASS();
    else { char m[64]; snprintf(m, sizeof(m), "built=%u", built); FAIL(m); }

    int saw_static = 0, saw_flow = 0;
    for (uint16_t i = 0; i < profs.count; ++i) {
        if (profs.profiles[i].type == CE_MOTION_STATIC) saw_static = 1;
        if (profs.profiles[i].type == CE_MOTION_FLOW)   saw_flow   = 1;
    }
    TEST("ce_infer_motion_profile auto-tags sky region as STATIC");
    if (saw_static) PASS(); else FAIL("no STATIC profile assigned");

    TEST("ce_infer_motion_profile auto-tags water region as FLOW");
    if (saw_flow)   PASS(); else FAIL("no FLOW profile assigned");

    /* Ensure tick_with_profiles still runs without crashing and the
     * STATIC object keeps its base position. */
    int saw_static_drift = 0;
    uint8_t *rgb = (uint8_t *)calloc((size_t)IMG_W * IMG_H * 3u, 1);
    for (int t = 0; t < 10; ++t) {
        ce_scene_tick_with_profiles(&scene, &profs);
        for (uint16_t i = 0; i < profs.count; ++i) {
            if (profs.profiles[i].type == CE_MOTION_STATIC) {
                if (scene.objects[i].cx != profs.base_cx[i] ||
                    scene.objects[i].cy != profs.base_cy[i])
                    saw_static_drift = 1;
            }
        }
        memset(rgb, 0, (size_t)IMG_W * IMG_H * 3u);
        ce_scene_render(&scene, rgb, IMG_W, IMG_H);
    }
    free(rgb);
    TEST("ce_scene_tick_with_profiles keeps STATIC pinned to base");
    if (!saw_static_drift) PASS();
    else FAIL("STATIC object drifted under tick_with_profiles");
}

int main(void) {
    printf("=== test_scene_atmos ===\n");
    test_motion_and_render();
    test_ce_storage_match();
    test_bridge_motion_inference();

    printf("\n=== test_scene_atmos: %d/%d PASSED ===\n",
           tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
