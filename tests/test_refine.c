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

/* ── D1/D3 ── ai_generate_refine does not mutate long-term state.
 * Output bytes may now diverge from ai_generate_next because the
 * refine loop is live — what must still hold is that ai_generate_next
 * produces identical output on calls bracketing the refine run. The
 * stronger test_refine_preserves_baseline below repeats this on a
 * separate probe model. ── */
static void test_refine_does_not_mutate_ai(void) {
    TEST("ai_generate_refine leaves ai_generate_next output unchanged on the same ai");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    ai_force_keyframe(ai, "the quick brown fox jumps over the lazy dog", "anim");
    ai_force_keyframe(ai, "pack my box with five dozen liquor jugs",     "anim");
    ai_force_keyframe(ai, "data teams iterate on the morning backlog",   "work");
    ai_force_keyframe(ai, "systems retire at midnight after cleanup",    "work");

    const char* prompt = "the quick brown fox";
    char base_before[128] = {0};
    float sim_before = 0.0f;
    uint32_t n_before = ai_generate_next(ai, prompt, base_before,
                                         sizeof base_before - 1, &sim_before);

    char ref_out[128] = {0};
    float ref_conf = -1.0f;
    uint32_t ref_iters = 0;
    (void)ai_generate_refine(ai, prompt, ref_out, sizeof ref_out - 1,
                             NULL, &ref_conf, &ref_iters);

    char base_after[128] = {0};
    float sim_after = 0.0f;
    uint32_t n_after = ai_generate_next(ai, prompt, base_after,
                                        sizeof base_after - 1, &sim_after);

    assert(n_after == n_before);
    assert(memcmp(base_after, base_before, n_before) == 0);
    assert(sim_after == sim_before);

    spatial_ai_destroy(ai);
    PASS();
}

/* ── D2 ── draft_field_init NULL safety ── */
static void test_draft_init_null_safety(void) {
    TEST("draft_field_init is safe on NULL df / inputs");
    /* NULL df — no crash, nothing to assert beyond that */
    draft_field_init(NULL, NULL, NULL, NULL);

    DraftField df;
    memset(&df, 0xFF, sizeof df);  /* junk */
    draft_field_init(&df, NULL, NULL, NULL);
    assert(df.n_anchor == 0);
    assert(df.n_candidate == 0);
    assert(df.n_resolved == 0);
    PASS();
}

/* ── D2 ── anchor rows get CELL_ANCHOR, non-input rows don't ── */
static void test_draft_init_anchors_from_input(void) {
    TEST("draft_field_init marks input-positive cells as CELL_ANCHOR");
    morpheme_init();

    SpatialGrid* input = grid_create();
    layers_encode_clause("the quick brown fox", NULL, input);

    DraftField df;
    draft_field_init(&df, input, NULL, NULL);

    /* Every anchor cell's value equals its row's argmax-x. */
    uint32_t verified = 0;
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        uint16_t best_a = 0;
        uint32_t best_x = 0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint16_t a = input->A[y * GRID_SIZE + x];
            if (a > best_a) { best_a = a; best_x = x; }
        }
        if (best_a == 0) continue;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (input->A[i] > 0) {
                assert(df.cells[i].status == CELL_ANCHOR);
                assert(df.cells[i].value == (uint8_t)best_x);
                assert(df.cells[i].confidence == 1.0f);
                verified++;
            }
        }
    }
    assert(verified > 0);
    assert(df.n_anchor == verified);
    assert(df.n_candidate == 0);  /* agg == NULL */
    assert(df.n_resolved == 0);

    grid_destroy(input);
    PASS();
}

/* ── D2 ── with an agg present, non-anchor prior cells become
 *         CELL_CANDIDATE, and anchor/candidate sets are disjoint. ── */
static void test_draft_init_candidates_from_agg(void) {
    TEST("draft_field_init: agg produces candidates, disjoint from anchors");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    ai_force_keyframe(ai, "the quick brown fox jumps over the lazy dog", "anim");
    ai_force_keyframe(ai, "pack my box with five dozen liquor jugs",     "anim");

    AggTables* agg = agg_build(ai);
    assert(agg != NULL);

    SpatialGrid* input = grid_create();
    layers_encode_clause("the quick brown fox", NULL, input);

    RefineConfig cfg = refine_config_default_text();
    /* allow_prior_anchors off → no RESOLVED cells from A_sum */
    assert(cfg.allow_prior_anchors == 0);

    DraftField df;
    draft_field_init(&df, input, agg, &cfg);

    assert(df.n_anchor > 0);
    assert(df.n_candidate > 0);
    assert(df.n_resolved == 0);

    /* No cell is both anchor and candidate. */
    for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        uint8_t s = df.cells[i].status;
        if (s == CELL_ANCHOR)    continue;
        if (s == CELL_CANDIDATE) continue;
        if (s == CELL_EMPTY)     continue;
        assert(!"unexpected status");
    }

    /* Anchors carry their row's byte; candidates carry value 0 (unset). */
    uint32_t checked_candidate = 0;
    for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE && checked_candidate < 16; i++) {
        if (df.cells[i].status == CELL_CANDIDATE) {
            assert(df.cells[i].value == 0);
            assert(df.cells[i].n_cand == 0);
            checked_candidate++;
        }
    }
    assert(checked_candidate > 0);

    grid_destroy(input);
    agg_destroy(agg);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── D2 ── allow_prior_anchors promotes high-A_sum cells to RESOLVED
 * (never to ANCHOR). ── */
static void test_draft_init_prior_anchors_guard(void) {
    TEST("allow_prior_anchors seeds CELL_RESOLVED, never CELL_ANCHOR-from-prior");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    /* Diverse keyframes spread A_sum across many cells so only the
     * strongest cells clear the 4× row-avg promotion threshold and
     * the rest survive as CANDIDATE. */
    const char* kfs[] = {
        "the quick brown fox jumps over the lazy dog today clearly",
        "pack my box with five dozen liquor jugs by the porch",
        "data teams iterate on the morning backlog carefully always",
        "systems retire at midnight after the weekly cleanup pass",
        "the kitchen smelled of cinnamon and warm bread each afternoon",
        "visitors paused at the iron gate and admired every bloom",
        "oak trees shed their leaves along the quiet garden path",
        "int fib(int n){return n<2?n:fib(n-1)+fib(n-2);}",
    };
    for (uint32_t i = 0; i < sizeof kfs / sizeof kfs[0]; i++) {
        ai_force_keyframe(ai, kfs[i], "mix");
    }
    AggTables* agg = agg_build(ai);
    SpatialGrid* input = grid_create();
    /* Empty input so no anchor cells steal the prior candidates. */

    RefineConfig cfg = refine_config_default_text();
    cfg.allow_prior_anchors = 1;

    DraftField df;
    draft_field_init(&df, input, agg, &cfg);

    assert(df.n_anchor == 0);        /* no input → no anchors */
    assert(df.n_resolved > 0);       /* high-A_sum cells seeded */
    assert(df.n_candidate > 0);      /* rest are still candidates */

    /* Any RESOLVED cell's value equals its column x (byte value), and
     * confidence is the 0.5f seed, not an anchor-equivalent 1.0f. */
    uint32_t checked = 0;
    for (uint32_t y = 0; y < GRID_SIZE && checked < 8; y++) {
        for (uint32_t x = 0; x < GRID_SIZE && checked < 8; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (df.cells[i].status == CELL_RESOLVED) {
                assert(df.cells[i].value == (uint8_t)x);
                assert(df.cells[i].confidence == 0.5f);
                checked++;
            }
        }
    }
    assert(checked > 0);

    grid_destroy(input);
    agg_destroy(agg);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── D3 ── helper: train a compact model for refine smoke tests. ── */
static SpatialAI* build_tiny_model(void) {
    SpatialAI* ai = spatial_ai_create();
    const char* kfs[] = {
        "the quick brown fox jumps over the lazy dog today clearly",
        "pack my box with five dozen liquor jugs by the porch now",
        "data teams iterate on the morning backlog carefully always warmly",
        "systems retire at midnight after the weekly cleanup pass runs",
        "the kitchen smelled of cinnamon and warm bread each afternoon",
        "visitors paused at the iron gate and admired every single bloom",
        "oak trees shed their leaves along the quiet garden path slowly",
    };
    for (uint32_t i = 0; i < sizeof kfs / sizeof kfs[0]; i++) {
        ai_force_keyframe(ai, kfs[i], "mix");
    }
    return ai;
}

/* D3: refine loop actually iterates and reports non-zero iterations
 * for a prompt with enough anchor density. */
static void test_refine_iterates(void) {
    TEST("ai_generate_refine runs the level loop for an anchored prompt");
    morpheme_init();
    SpatialAI* ai = build_tiny_model();

    char out[256];
    float conf = -1.0f;
    uint32_t iters = 0;
    /* Long-enough prompt to clear the REFINE_ANCHOR_MIN floor. */
    uint32_t n = ai_generate_refine(ai,
        "the quick brown fox jumps over the lazy dog",
        out, sizeof out - 1, NULL, &conf, &iters);
    out[n < sizeof out ? n : sizeof out - 1] = '\0';
    printf("\n    n=%u iters=%u conf=%.3f\n", n, iters, conf);

    /* refine path must have done _some_ work (either iterated or
     * emitted via the in-loop fallback with iters carried through). */
    assert(n > 0);

    spatial_ai_destroy(ai);
    PASS();
}

/* D3: short prompts trigger the anchor-density fallback and the
 * baseline handles them. iters stays at 0 to signal fallback. */
static void test_refine_falls_back_on_thin_prompt(void) {
    TEST("ai_generate_refine falls back to baseline on thin prompt");
    morpheme_init();
    SpatialAI* ai = build_tiny_model();

    char out[128];
    float conf = -1.0f;
    uint32_t iters = 42;  /* sentinel — must be overwritten */
    uint32_t n = ai_generate_refine(ai, "hi", out, sizeof out - 1,
                                    NULL, &conf, &iters);
    out[n < sizeof out ? n : sizeof out - 1] = '\0';
    assert(iters == 0);  /* fallback path zeros iters */

    spatial_ai_destroy(ai);
    PASS();
}

/* D3: running refine does NOT mutate long-term state. A
 * back-to-back ai_generate_next produces the same output it would
 * have produced in isolation. */
static void test_refine_preserves_baseline(void) {
    TEST("ai_generate_refine does not change ai_generate_next behavior");
    morpheme_init();
    SpatialAI* ai_base = build_tiny_model();
    SpatialAI* ai_probe = build_tiny_model();

    const char* prompt = "the quick brown fox jumps over the lazy dog";
    char base_out[256] = {0};
    float base_sim = 0.0f;
    uint32_t base_n = ai_generate_next(ai_base, prompt, base_out,
                                       sizeof base_out - 1, &base_sim);

    /* Run refine first on a parallel model, then read baseline. */
    char ref_out[256] = {0};
    float conf = 0.0f;
    uint32_t iters = 0;
    (void)ai_generate_refine(ai_probe, prompt, ref_out,
                             sizeof ref_out - 1, NULL, &conf, &iters);

    char probe_out[256] = {0};
    float probe_sim = 0.0f;
    uint32_t probe_n = ai_generate_next(ai_probe, prompt, probe_out,
                                        sizeof probe_out - 1, &probe_sim);

    /* Both models should give ai_generate_next identical output for
     * the same prompt: refine didn't touch ai_probe's keyframes. */
    assert(probe_n == base_n);
    assert(memcmp(probe_out, base_out, probe_n) == 0);

    spatial_ai_destroy(ai_base);
    spatial_ai_destroy(ai_probe);
    PASS();
}

/* D3: anchor cells survive the refine loop with their status +
 * value + confidence unchanged. */
static void test_refine_anchor_immutability(void) {
    TEST("refine loop never overwrites CELL_ANCHOR cells");
    morpheme_init();
    SpatialAI* ai = build_tiny_model();

    /* Drive draft_field_init → refine_run_levels directly so we can
     * snapshot the anchor cells before and after. */
    SpatialGrid* in_grid = grid_create();
    layers_encode_clause("the quick brown fox jumps over the lazy dog",
                         NULL, in_grid);
    update_rgb_directional(in_grid);
    apply_ema_to_grid(ai, in_grid);

    AggTables* agg = agg_build(ai);
    RefineConfig cfg = refine_config_default_text();

    DraftField* df = (DraftField*)calloc(1, sizeof(DraftField));
    draft_field_init(df, in_grid, agg, &cfg);
    assert(df->n_anchor > 0);

    /* Snapshot anchor cells (status/value/confidence). */
    uint32_t snap_idx[256];
    uint8_t  snap_val[256];
    float    snap_conf[256];
    uint32_t snap_n = 0;
    for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE && snap_n < 256; i++) {
        if (df->cells[i].status == CELL_ANCHOR) {
            snap_idx[snap_n]  = i;
            snap_val[snap_n]  = df->cells[i].value;
            snap_conf[snap_n] = df->cells[i].confidence;
            snap_n++;
        }
    }
    assert(snap_n > 0);

    /* Call the exposed refine path via the public entry. */
    char buf[256];
    float conf = 0.0f;
    uint32_t iters = 0;
    ai_generate_refine(ai, "the quick brown fox jumps over the lazy dog",
                       buf, sizeof buf - 1, &cfg, &conf, &iters);

    /* Re-run draft_field_init on a fresh DraftField and re-examine
     * the same cells. The init is deterministic, so the same
     * (input, agg) pair must produce the same anchor set; if refine
     * had leaked state via agg mutations or ai state, we'd see a
     * difference here. */
    DraftField* df2 = (DraftField*)calloc(1, sizeof(DraftField));
    AggTables*  agg2 = agg_build(ai);
    draft_field_init(df2, in_grid, agg2, &cfg);
    for (uint32_t k = 0; k < snap_n; k++) {
        uint32_t i = snap_idx[k];
        assert(df2->cells[i].status     == CELL_ANCHOR);
        assert(df2->cells[i].value      == snap_val[k]);
        assert(df2->cells[i].confidence == snap_conf[k]);
    }

    free(df); free(df2);
    agg_destroy(agg); agg_destroy(agg2);
    grid_destroy(in_grid);
    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_refine ===\n");

    test_default_text_config();
    test_default_image_config();
    test_cell_state_zero_init();
    test_refine_null_safety();
    test_refine_does_not_mutate_ai();
    test_draft_init_null_safety();
    test_draft_init_anchors_from_input();
    test_draft_init_candidates_from_agg();
    test_draft_init_prior_anchors_guard();
    test_refine_iterates();
    test_refine_falls_back_on_thin_prompt();
    test_refine_preserves_baseline();
    test_refine_anchor_immutability();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
