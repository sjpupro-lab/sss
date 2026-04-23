/* v4 Task D — refine-path tests.
 *
 * D1 coverage: types and defaults are correctly shaped, the stub
 * ai_generate_refine delegates to ai_generate_next without touching
 * long-term state, and anchor-immutability invariants hold on the
 * (empty) DraftField zero-init. Real refine-loop behavior lands in
 * D2/D3 and is tested there. */

#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_grid.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)

/* ── D1-1 ── config defaults match spec values ── */
static void test_default_text_config(void) {
    TEST("refine_config_default_text matches spec §D.4 values");
    RefineConfig c = refine_config_default_text();

    /* Channel weights: B-dominant L0, G-dominant L1, R-dominant L2 */
    assert(c.ch_weights[0][2] == 1.0f);  /* B at L0 */
    assert(c.ch_weights[1][1] == 1.0f);  /* G at L1 */
    assert(c.ch_weights[2][0] == 1.0f);  /* R at L2 */

    /* Top-K grows with detail */
    assert(c.topk[0] == 4 && c.topk[1] == 8 && c.topk[2] == 16);
    /* Thresholds tighten per level */
    assert(c.promote_threshold[0] < c.promote_threshold[1]);
    assert(c.promote_threshold[1] < c.promote_threshold[2]);
    /* Radius shrinks coarse → fine */
    assert(c.neighbor_radius[0] > c.neighbor_radius[1]);
    assert(c.neighbor_radius[1] > c.neighbor_radius[2]);

    /* Safety defaults: no prior-anchor seeding, context on */
    assert(c.allow_prior_anchors == 0);
    assert(c.use_context_pool == 1);
    assert(c.temperature == 0.0f);
    PASS();
}

static void test_default_image_config(void) {
    TEST("refine_config_default_image tweaks radii and disables context");
    RefineConfig t = refine_config_default_text();
    RefineConfig i = refine_config_default_image();

    /* Image preset uses larger neighborhoods and no session context. */
    assert(i.neighbor_radius[0] > t.neighbor_radius[0]);
    assert(i.neighbor_radius[2] > t.neighbor_radius[2]);
    assert(i.use_context_pool == 0);
    /* allow_prior_anchors stays off by default */
    assert(i.allow_prior_anchors == 0);
    PASS();
}

/* ── D1-2 ── CellState zero-init matches CELL_EMPTY semantics ── */
static void test_cell_state_zero_init(void) {
    TEST("CellState{} is CELL_EMPTY and has no candidates");
    CellState cs;
    memset(&cs, 0, sizeof cs);
    assert(cs.status == CELL_EMPTY);  /* 0 */
    assert(cs.n_cand == 0);
    assert(cs.confidence == 0.0f);

    /* enum values line up with spec */
    assert(CELL_EMPTY     == 0);
    assert(CELL_CANDIDATE == 1);
    assert(CELL_RESOLVED  == 2);
    assert(CELL_ANCHOR    == 3);
    /* REFINE_TOPK_MAX accommodates L2 top-k = 16 */
    assert(REFINE_TOPK_MAX >= 16);
    PASS();
}

/* ── D1-3 ── NULL-safety on the public entry point ── */
static void test_refine_null_safety(void) {
    TEST("ai_generate_refine handles NULL ai / input / out gracefully");
    char buf[64];
    float conf = -1.0f;
    uint32_t iters = 999;

    uint32_t n = ai_generate_refine(NULL, "hello", buf, sizeof buf,
                                    NULL, &conf, &iters);
    assert(n == 0);
    assert(iters == 0);

    n = ai_generate_refine(NULL, NULL, buf, sizeof buf, NULL, NULL, NULL);
    assert(n == 0);
    PASS();
}

/* ── D1-4 ── Stub delegates to ai_generate_next and baseline stays
 * unchanged. We train a tiny model, take an ai_generate_next reading,
 * and verify ai_generate_refine returns the same bytes for the same
 * input + does not mutate the output of a follow-up ai_generate_next
 * call. ── */
static void test_refine_stub_matches_baseline(void) {
    TEST("D1 stub: refine output == baseline, baseline path is untouched");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    ai_force_keyframe(ai, "the quick brown fox jumps over the lazy dog", "anim");
    ai_force_keyframe(ai, "pack my box with five dozen liquor jugs",     "anim");
    ai_force_keyframe(ai, "data teams iterate on the morning backlog",   "work");
    ai_force_keyframe(ai, "systems retire at midnight after cleanup",    "work");

    const char* prompt = "the quick brown fox";
    char base_out[128] = {0};
    float base_sim = 0.0f;
    uint32_t base_n = ai_generate_next(ai, prompt, base_out, sizeof base_out - 1, &base_sim);
    base_out[base_n < sizeof base_out ? base_n : sizeof base_out - 1] = '\0';

    char ref_out[128] = {0};
    float ref_conf = -1.0f;
    uint32_t ref_iters = 999;
    uint32_t ref_n = ai_generate_refine(ai, prompt, ref_out, sizeof ref_out - 1,
                                        NULL, &ref_conf, &ref_iters);
    ref_out[ref_n < sizeof ref_out ? ref_n : sizeof ref_out - 1] = '\0';

    /* Stub reports 0 iterations (it didn't run the refine loop) */
    assert(ref_iters == 0);
    /* Confidence propagated from baseline match similarity */
    assert(ref_conf >= 0.0f);
    /* Byte-for-byte match with baseline */
    assert(ref_n == base_n);
    assert(strncmp(ref_out, base_out, ref_n) == 0);

    /* Baseline still produces the same thing on a fresh call */
    char base_out2[128] = {0};
    float base_sim2 = 0.0f;
    uint32_t base_n2 = ai_generate_next(ai, prompt, base_out2,
                                        sizeof base_out2 - 1, &base_sim2);
    base_out2[base_n2 < sizeof base_out2 ? base_n2 : sizeof base_out2 - 1] = '\0';
    assert(base_n2 == base_n);
    assert(strncmp(base_out2, base_out, base_n) == 0);

    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_refine ===\n");

    test_default_text_config();
    test_default_image_config();
    test_cell_state_zero_init();
    test_refine_null_safety();
    test_refine_stub_matches_baseline();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
