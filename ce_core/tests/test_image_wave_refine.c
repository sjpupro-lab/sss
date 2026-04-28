/* test_image_wave_refine.c — wave-refine image carving tests.
 *
 * Covers:
 *   - 7-step wave cycle (0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01)
 *   - move_toward semantics: clamp by max_step, snap when within step
 *   - top-3 score-weighted target blending (NULL slots ignored, all-NULL safe)
 *   - refine_once channel-wise clamp into [0, 255]
 *   - full pipeline determinism + convergence to target after enough steps
 */
#include "../ce_image_wave_refine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

static int feq(float a, float b) {
    float d = a - b;
    if (d < 0) d = -d;
    return d < 1e-4f;
}

static void fill_canvas_const(CEImageCanvas *c, float r, float g, float b, float a) {
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        c->px[i].r = r; c->px[i].g = g; c->px[i].b = b; c->px[i].a = a;
    }
}

static void fill_link_const_target(CEImageLink64 *L, float score,
                                   float r, float g, float b, float a) {
    L->score = score;
    memset(L->semantic, 0, sizeof(L->semantic));
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        L->target[i].r = r; L->target[i].g = g;
        L->target[i].b = b; L->target[i].a = a;
    }
}

int main(void) {
    printf("=== ce_image_wave_refine tests ===\n");

    /* ---------- 1. Wave step cycle ---------- */
    const float expected[7] = {0.01f, 0.1f, 1.0f, 3.0f, 1.0f, 0.1f, 0.01f};
    int wave_ok = 1;
    for (uint32_t i = 0; i < 21; ++i) {
        if (!feq(ce_wave_step_value(i), expected[i % 7])) { wave_ok = 0; break; }
    }
    CHECK(wave_ok, "wave_step_value cycles 0.01→0.1→1→3→1→0.1→0.01");
    CHECK(feq(ce_wave_step_value(0), 0.01f),  "iter 0  -> 0.01");
    CHECK(feq(ce_wave_step_value(3), 3.0f),   "iter 3  -> 3.0  (peak)");
    CHECK(feq(ce_wave_step_value(7), 0.01f),  "iter 7  wraps to 0.01");
    CHECK(feq(ce_wave_step_value(10), 3.0f),  "iter 10 -> 3.0  (next peak)");

    /* ---------- 2. move_toward via refine_once ---------- */
    CEImageCanvas *c = (CEImageCanvas *)calloc(1, sizeof(CEImageCanvas));
    CEPixelF *t = (CEPixelF *)calloc(CE_WAVE_IMG_W * CE_WAVE_IMG_H, sizeof(CEPixelF));
    if (!c || !t) { printf("  FAIL: alloc\n"); return 1; }

    /* canvas at 0, target at 100, step 3 -> after one step pixel == 3. */
    fill_canvas_const(c, 0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        t[i].r = 100.0f; t[i].g = 100.0f; t[i].b = 100.0f; t[i].a = 100.0f;
    }
    ce_wave_refine_once(c, t, 3.0f);
    CHECK(feq(c->px[0].r, 3.0f), "step 3.0 from 0 -> 3 (max-step ramp)");

    /* canvas at 99.5, target at 100, step 1 -> snap to 100 (within step). */
    fill_canvas_const(c, 99.5f, 99.5f, 99.5f, 99.5f);
    ce_wave_refine_once(c, t, 1.0f);
    CHECK(feq(c->px[0].r, 100.0f), "within max_step snaps to target");

    /* canvas at 200, target at 100, step 0.1 -> 199.9 (move down). */
    fill_canvas_const(c, 200.0f, 200.0f, 200.0f, 200.0f);
    ce_wave_refine_once(c, t, 0.1f);
    CHECK(feq(c->px[0].r, 199.9f), "moves down toward lower target");

    /* clamp into [0, 255]: target 1000, step 5 from 252 -> 255 capped. */
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        t[i].r = 1000.0f; t[i].g = 1000.0f; t[i].b = 1000.0f; t[i].a = 1000.0f;
    }
    fill_canvas_const(c, 252.0f, 252.0f, 252.0f, 252.0f);
    ce_wave_refine_once(c, t, 5.0f);
    CHECK(feq(c->px[0].r, 255.0f), "post-step clamp at 255 ceiling");

    /* clamp into [0, 255]: target -100, step 5 from 3 -> 0 floor. */
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        t[i].r = -100.0f; t[i].g = -100.0f; t[i].b = -100.0f; t[i].a = -100.0f;
    }
    fill_canvas_const(c, 3.0f, 3.0f, 3.0f, 3.0f);
    ce_wave_refine_once(c, t, 5.0f);
    CHECK(feq(c->px[0].r, 0.0f), "post-step clamp at 0 floor");

    /* ---------- 3. Top-3 score-weighted blend ---------- */
    CEImageLink64 *A = (CEImageLink64 *)calloc(1, sizeof(CEImageLink64));
    CEImageLink64 *B = (CEImageLink64 *)calloc(1, sizeof(CEImageLink64));
    CEImageLink64 *C = (CEImageLink64 *)calloc(1, sizeof(CEImageLink64));
    if (!A || !B || !C) { printf("  FAIL: alloc links\n"); return 1; }

    /* A: target = (255, 0, 0, 255), score 6.
     * B: target = (0, 255, 0, 255), score 3.
     * C: target = (0, 0, 255, 255), score 1.
     * Total 10 -> weights 0.6, 0.3, 0.1.
     * Blended target per pixel = (153, 76.5, 25.5, 255). */
    fill_link_const_target(A, 6.0f, 255.0f,   0.0f,   0.0f, 255.0f);
    fill_link_const_target(B, 3.0f,   0.0f, 255.0f,   0.0f, 255.0f);
    fill_link_const_target(C, 1.0f,   0.0f,   0.0f, 255.0f, 255.0f);
    CEImageLink64 *top3_arr[CE_WAVE_TOPK] = {A, B, C};
    ce_wave_build_target_from_top3(t, top3_arr);
    CHECK(feq(t[0].r, 153.0f), "blend R = 0.6 * 255 = 153");
    CHECK(feq(t[0].g, 76.5f),  "blend G = 0.3 * 255 = 76.5");
    CHECK(feq(t[0].b, 25.5f),  "blend B = 0.1 * 255 = 25.5");
    CHECK(feq(t[0].a, 255.0f), "blend A = (0.6+0.3+0.1) * 255 = 255");

    /* NULL slot is skipped (treat as missing candidate). */
    CEImageLink64 *partial[CE_WAVE_TOPK] = {A, NULL, C};
    /* total = 6 + 1 = 7. weights 6/7 (R), 1/7 (B). */
    ce_wave_build_target_from_top3(t, partial);
    CHECK(feq(t[0].r, 255.0f * 6.0f / 7.0f), "NULL slot ignored: weight redistributes (R)");
    CHECK(feq(t[0].g, 0.0f),                 "NULL slot ignored: G stays 0");
    CHECK(feq(t[0].b, 255.0f * 1.0f / 7.0f), "NULL slot ignored: weight redistributes (B)");

    /* All-NULL: zero target (no div-by-zero). */
    CEImageLink64 *empty[CE_WAVE_TOPK] = {NULL, NULL, NULL};
    /* Pre-fill target with garbage; build_target must overwrite with zero. */
    for (int i = 0; i < 8; ++i) { t[i].r = 99.0f; t[i].g = 99.0f; t[i].b = 99.0f; t[i].a = 99.0f; }
    ce_wave_build_target_from_top3(t, empty);
    CHECK(feq(t[0].r, 0.0f) && feq(t[0].g, 0.0f) && feq(t[0].b, 0.0f) && feq(t[0].a, 0.0f),
          "all-NULL top3 -> zero target (no div-by-zero)");

    /* All-zero-score: same safe fallback. */
    A->score = B->score = C->score = 0.0f;
    for (int i = 0; i < 8; ++i) { t[i].r = 99.0f; }
    ce_wave_build_target_from_top3(t, top3_arr);
    CHECK(feq(t[0].r, 0.0f), "all-zero score -> zero target");

    /* ---------- 4. Full pipeline: determinism + convergence ---------- */
    fill_link_const_target(A, 1.0f, 200.0f, 100.0f, 50.0f, 255.0f);
    CEImageLink64 *single[CE_WAVE_TOPK] = {A, NULL, NULL};

    CEImageCanvas *c1 = (CEImageCanvas *)calloc(1, sizeof(CEImageCanvas));
    CEImageCanvas *c2 = (CEImageCanvas *)calloc(1, sizeof(CEImageCanvas));
    fill_canvas_const(c1, 10.0f, 10.0f, 10.0f, 10.0f);
    fill_canvas_const(c2, 10.0f, 10.0f, 10.0f, 10.0f);
    ce_image_wave_refine(c1, single, 50);
    ce_image_wave_refine(c2, single, 50);
    CHECK(memcmp(c1, c2, sizeof(CEImageCanvas)) == 0,
          "ce_image_wave_refine deterministic on identical input");

    /* After 50 wave steps the per-iter step sequence sums to:
     *     50 = 7*7 + 1 ; total per-channel motion budget
     *     ≈ 7 * (0.01+0.1+1+3+1+0.1+0.01) + 0.01
     *     ≈ 7 * 5.22 + 0.01 = 36.55
     * Starting pixel is 10, target R = 200. Distance = 190 — exceeds budget,
     * so canvas should approach but not yet reach target. R must grow
     * monotonically toward 200 (no overshoot since move_toward clamps). */
    CHECK(c1->px[0].r > 10.0f && c1->px[0].r <= 200.0f,
          "convergence: R moves toward target without overshoot");
    CHECK(c1->px[0].g > 10.0f && c1->px[0].g <= 100.0f,
          "convergence: G moves toward target without overshoot");

    /* Per-cycle (7 iters) motion budget = 0.01+0.1+1+3+1+0.1+0.01 = 5.22.
     * Need >= max distance / 5.22 cycles to converge. R distance = 190 →
     * 190/5.22 ≈ 37 cycles = 259 iters. Use 300 for headroom. */
    fill_canvas_const(c1, 10.0f, 10.0f, 10.0f, 10.0f);
    ce_image_wave_refine(c1, single, 300);
    CHECK(feq(c1->px[0].r, 200.0f), "300 iterations -> R converged to target");
    CHECK(feq(c1->px[0].g, 100.0f), "300 iterations -> G converged to target");
    CHECK(feq(c1->px[0].b, 50.0f),  "300 iterations -> B converged to target");

    /* Different starting canvas -> different mid-iteration state (no smoothing). */
    fill_canvas_const(c2, 250.0f, 250.0f, 250.0f, 250.0f);
    ce_image_wave_refine(c2, single, 5);
    fill_canvas_const(c1, 10.0f, 10.0f, 10.0f, 10.0f);
    ce_image_wave_refine(c1, single, 5);
    CHECK(c1->px[0].r != c2->px[0].r,
          "different starting canvas -> different intermediate (no flattening)");

    free(A); free(B); free(C);
    free(c); free(c1); free(c2); free(t);

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
