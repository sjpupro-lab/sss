/* Manual probe for v4 Task A — ContextPool.
 *
 * Not part of `make test`. Built on demand and run once to verify
 * that the pool plumbing works end-to-end:
 *   1. lazy-create via ai_get_context_pool
 *   2. pool_add_clause populates the pool
 *   3. canvas_pool and context_pool are distinct (no aliasing)
 *   4. agg_build_from_pool reads the context pool
 *   5. agg_score_byte_combined blends long + short priors
 *   6. clear empties, release NULLs
 *
 * Compile:
 *   make -C build probe_task_a   (see Makefile rule added alongside)
 */

#include "spatial_keyframe.h"
#include "spatial_subtitle.h"
#include "spatial_generate.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void check(int cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        exit(1);
    }
    printf("  ok: %s\n", msg);
}

int main(void) {
    printf("=== probe_task_a: v4 ContextPool ===\n");

    SpatialAI* ai = spatial_ai_create();
    check(ai != NULL, "spatial_ai_create");
    check(ai->context_pool == NULL, "context_pool starts NULL");
    check(ai->canvas_pool == NULL, "canvas_pool starts NULL");

    /* Seed some long-term knowledge so agg_build has something. */
    ai_force_keyframe(ai, "the quick brown fox jumps", "animals");
    ai_force_keyframe(ai, "the lazy dog barks loudly", "animals");

    /* 1. lazy-create */
    SpatialCanvasPool* ctx = ai_get_context_pool(ai);
    check(ctx != NULL, "ai_get_context_pool lazy-create returns non-NULL");
    check(ai->context_pool == ctx, "pointer stored on ai");
    check(ai_get_context_pool(ai) == ctx, "second call returns the same pool");

    /* 3. distinct from canvas_pool */
    SpatialCanvasPool* canvas = ai_get_canvas_pool(ai);
    check(canvas != NULL, "ai_get_canvas_pool lazy-create");
    check(canvas != ctx, "canvas_pool and context_pool are distinct objects");

    /* 2. pool_add_clause populates (returns non-negative entry_id) */
    int added = pool_add_clause(ctx, "context clause one");
    check(added >= 0, "pool_add_clause returns >= 0 on first add");
    added = pool_add_clause(ctx, "context clause two with more words");
    check(added >= 0, "pool_add_clause returns >= 0 on second add");
    uint32_t slots = pool_total_slots(ctx);
    check(slots >= 2, "pool_total_slots >= 2 after two adds");
    printf("    context pool total_slots=%u\n", slots);

    /* 4. agg_build_from_pool reads context */
    AggTables* agg_L = agg_build(ai);
    AggTables* agg_S = agg_build_from_pool(ctx);
    check(agg_L != NULL, "agg_build(ai) ok");
    check(agg_S != NULL, "agg_build_from_pool(ctx) ok");

    /* 5. combined scoring — pick an arbitrary (y, v). The exact
     * value depends on byte positions; we just verify that the
     * combined score is finite and >= the zero-weighted calls. */
    uint32_t y = 2;
    uint8_t  v = (uint8_t)'t';
    double sL = agg_score_byte(agg_L, y, v, 128, 128, 128);
    double sS = agg_score_byte(agg_S, y, v, 128, 128, 128);
    double sC = agg_score_byte_combined(agg_L, agg_S,
                                        SPAI_W_LONG_DEFAULT, SPAI_W_SHORT_DEFAULT,
                                        y, v, 128, 128, 128);
    printf("    score long=%.3f short=%.3f combined(0.6,0.4)=%.3f\n", sL, sS, sC);

    double expected = SPAI_W_LONG_DEFAULT * sL + SPAI_W_SHORT_DEFAULT * sS;
    double diff = sC - expected;
    if (diff < 0) diff = -diff;
    check(diff < 1e-9, "combined == 0.6 * long + 0.4 * short");

    /* Long-only collapse */
    double sL_only = agg_score_byte_combined(agg_L, NULL,
                                             1.0, 0.0,
                                             y, v, 128, 128, 128);
    diff = sL_only - sL;
    if (diff < 0) diff = -diff;
    check(diff < 1e-9, "combined(agg_S=NULL, w_short=0) == agg_score_byte(agg_L)");

    /* Short-only collapse */
    double sS_only = agg_score_byte_combined(NULL, agg_S,
                                             0.0, 1.0,
                                             y, v, 128, 128, 128);
    diff = sS_only - sS;
    if (diff < 0) diff = -diff;
    check(diff < 1e-9, "combined(agg_L=NULL, w_long=0) == agg_score_byte(agg_S)");

    agg_destroy(agg_L);
    agg_destroy(agg_S);

    /* 6. clear empties, release NULLs */
    ai_clear_context_pool(ai);
    check(ai->context_pool != NULL, "clear keeps a fresh (empty) pool");
    check(pool_total_slots(ai->context_pool) == 0, "clear: pool is empty");

    ai_release_context_pool(ai);
    check(ai->context_pool == NULL, "release NULLs the pointer");

    /* Verify long-term state was not mutated: keyframes still intact. */
    check(ai->kf_count == 2, "long-term keyframes preserved");

    spatial_ai_destroy(ai);
    printf("=== probe_task_a: OK ===\n");
    return 0;
}
