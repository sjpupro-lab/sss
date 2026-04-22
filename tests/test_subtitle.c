#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"
#include "spatial_io.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)

/* ── 1. detect_data_type classifies text correctly ── */
static void test_detect_data_type(void) {
    TEST("detect_data_type: prose/dialog/code/short");

    const char* prose =
        "The ancient forests of the northern valley stretched for miles and miles, "
        "their tall trees reaching ever upward toward a sky that had seen countless "
        "seasons pass by in silent observation of the world below them all.";
    const char* dialog = "Where did you go yesterday afternoon after class ended?";
    const char* code = "int main(){int x=0;x+=(y*2);return x;}";
    const char* sh    = "apple.";

    DataType t1 = detect_data_type((const uint8_t*)prose, (uint32_t)strlen(prose));
    DataType t2 = detect_data_type((const uint8_t*)dialog, (uint32_t)strlen(dialog));
    DataType t3 = detect_data_type((const uint8_t*)code, (uint32_t)strlen(code));
    DataType t4 = detect_data_type((const uint8_t*)sh, (uint32_t)strlen(sh));

    printf("\n    prose  (%zuB) → %s\n", strlen(prose),  data_type_name(t1));
    printf("    dialog (%zuB) → %s\n", strlen(dialog), data_type_name(t2));
    printf("    code   (%zuB) → %s\n", strlen(code),   data_type_name(t3));
    printf("    short  (%zuB) → %s\n", strlen(sh),     data_type_name(t4));

    assert(t1 == DATA_PROSE);
    assert(t2 == DATA_DIALOG);
    assert(t3 == DATA_CODE);
    assert(t4 == DATA_SHORT);
    PASS();
}

/* ── 2. SubtitleTrack add + per-type indices ── */
static void test_track_add_and_lookup(void) {
    TEST("SubtitleTrack add + per-type lookup");

    SubtitleTrack t;
    subtitle_track_init(&t);

    /* Add a mix of entries */
    subtitle_track_add(&t, DATA_PROSE,  111, 0, 0, 200);
    subtitle_track_add(&t, DATA_CODE,   222, 1, 0, 45);
    subtitle_track_add(&t, DATA_PROSE,  333, 0, 1, 180);
    subtitle_track_add(&t, DATA_DIALOG, 444, 2, 0, 60);
    subtitle_track_add(&t, DATA_CODE,   555, 1, 1, 50);

    uint32_t nP, nC, nD, nS;
    const uint32_t* pr = subtitle_track_ids_of_type(&t, DATA_PROSE,  &nP);
    const uint32_t* co = subtitle_track_ids_of_type(&t, DATA_CODE,   &nC);
    const uint32_t* di = subtitle_track_ids_of_type(&t, DATA_DIALOG, &nD);
    const uint32_t* sh = subtitle_track_ids_of_type(&t, DATA_SHORT,  &nS);

    printf("\n    prose=%u  code=%u  dialog=%u  short=%u\n", nP, nC, nD, nS);
    assert(nP == 2);
    assert(nC == 2);
    assert(nD == 1);
    assert(nS == 0);
    assert(pr[0] == 0 && pr[1] == 2);
    assert(co[0] == 1 && co[1] == 4);
    assert(di[0] == 3);
    (void)sh;

    subtitle_track_destroy(&t);
    PASS();
}

/* ── 3. Pool routing: same-type clauses cluster on one canvas ── */
static void test_pool_routing_same_type(void) {
    TEST("pool routes 32 prose clauses into a single prose canvas");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();

    /* Generate 32 prose clauses (len > 150, ASCII) */
    for (int i = 0; i < 32; i++) {
        char buf[400];
        snprintf(buf, sizeof(buf),
            "In the quiet village numbered %d the villagers went about their daily "
            "tasks with the same unhurried rhythm that had carried them through "
            "decades of seasons sliding one after another into memory and legend "
            "and more.", i);
        int r = pool_add_clause(p, buf);
        assert(r >= 0);
    }

    printf("\n    canvases=%u, total_slots=%u\n",
           p->count, pool_total_slots(p));
    /* All 32 prose → 1 canvas fully filled */
    assert(p->count == 1);
    assert(p->canvases[0]->slot_count == 32);
    assert(p->canvases[0]->canvas_type == DATA_PROSE);

    pool_destroy(p);
    PASS();
}

/* ── 4. Mixed types → one canvas per type ── */
static void test_pool_routing_mixed(void) {
    TEST("prose + code mix → separate canvases");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();

    /* 20 prose */
    for (int i = 0; i < 20; i++) {
        char buf[400];
        snprintf(buf, sizeof(buf),
            "The ancient chronicler recorded that in year %d of the kingdom many "
            "remarkable events had transpired which were recorded with care by "
            "scholars who continued preserving such histories across centuries.", i);
        pool_add_clause(p, buf);
    }
    /* 12 code */
    for (int i = 0; i < 12; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf),
            "int fn_%d(int x){return (x*x)+(%d/2);}", i, i + 1);
        pool_add_clause(p, buf);
    }

    uint32_t np = 0, nc = 0;
    for (uint32_t i = 0; i < p->count; i++) {
        SpatialCanvas* cv = p->canvases[i];
        if (cv->canvas_type == DATA_PROSE) np += cv->slot_count;
        if (cv->canvas_type == DATA_CODE)  nc += cv->slot_count;
    }
    printf("\n    canvases=%u  prose_slots=%u  code_slots=%u\n",
           p->count, np, nc);
    assert(np == 20);
    assert(nc == 12);
    /* At least two distinct-typed canvases */
    assert(p->count >= 2);

    pool_destroy(p);
    PASS();
}

/* ── 5. Subtitle jump confines search to same type ── */
static void test_subtitle_jump(void) {
    TEST("pool_match jumps to same-type slots first");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();

    /* Store one target code clause, plus a lot of prose clauses */
    pool_add_clause(p, "int run(int n){return n*(n+1)/2;}");   /* code, entry 0 */
    for (int i = 0; i < 10; i++) {
        char buf[400];
        snprintf(buf, sizeof(buf),
            "The ancient chronicler recorded that in year %d of the kingdom many "
            "remarkable events had transpired which scholars preserved with care "
            "for future generations across centuries to come.", i);
        pool_add_clause(p, buf);
    }

    /* Query: also code */
    const char* qtxt = "int run(int n){return n*(n+1)/2;}";
    SpatialGrid* q = grid_create();
    layers_encode_clause(qtxt, NULL, q);
    update_rgb_directional(q);

    PoolMatchResult r = pool_match(p, q, qtxt);
    printf("\n    query_type=%s matched canvas=%u slot=%u sim=%.3f step=%d fb=%d\n",
           data_type_name(r.query_type), r.canvas_id, r.slot_id,
           r.similarity, r.step_taken, r.fallback);

    /* Should find the code slot (high A similarity) and NOT fall back */
    assert(r.query_type == DATA_CODE);
    assert(r.fallback == 0);
    assert(r.similarity > 0.5f);

    grid_destroy(q);
    pool_destroy(p);
    PASS();
}

/* ── 6. fallback: query type has no slots → search other types ── */
static void test_subtitle_fallback(void) {
    TEST("pool_match falls back to other types when query type empty");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();
    /* Only dialog clauses */
    pool_add_clause(p, "What did you eat for dinner tonight?");
    pool_add_clause(p, "I had some fresh vegetables and bread.");
    pool_add_clause(p, "Where did you buy the bread from today?");

    /* Query is code — no code slots exist → must fall back */
    const char* qtxt = "int eat(int food){return food*2;}";
    SpatialGrid* q = grid_create();
    layers_encode_clause(qtxt, NULL, q);
    update_rgb_directional(q);

    PoolMatchResult r = pool_match(p, q, qtxt);
    printf("\n    query_type=%s fallback=%d step=%d canvas=%u slot=%u\n",
           data_type_name(r.query_type), r.fallback, r.step_taken,
           r.canvas_id, r.slot_id);

    assert(r.query_type == DATA_CODE);
    assert(r.fallback == 1);
    assert(r.step_taken == 4);

    grid_destroy(q);
    pool_destroy(p);
    PASS();
}

/* ── 7. Subtitle track does NOT alter A-channel match outcomes ── */
static void test_subtitle_non_intrusive(void) {
    TEST("subtitle track is metadata-only — scores untouched");
    morpheme_init();

    /* Build a pool with one canvas of 5 prose clauses */
    SpatialCanvasPool* p = pool_create();
    const char* clauses[] = {
        "The ancient scroll was discovered in a hidden archive buried beneath "
        "the old monastery for many forgotten years until recent excavations.",
        "Scholars studying the parchment noticed that the ink composition had "
        "preserved marks despite the passage of several turbulent centuries.",
        "Translations revealed narratives of merchant caravans crossing vast "
        "deserts with great quantities of exotic spices and silk stored safely.",
        "The chronicler wrote detailed accounts of storms and shipwrecks along "
        "distant coastlines that had swallowed fleets without any recorded.",
        "Eventually the documents were transferred to a national library where "
        "researchers could examine them under proper conservation conditions daily.",
        NULL
    };
    for (int i = 0; clauses[i]; i++) pool_add_clause(p, clauses[i]);

    /* Reference: direct argmax of canvas_slot_cosine_a */
    const char* qtxt = clauses[2];
    SpatialGrid* q = grid_create();
    layers_encode_clause(qtxt, NULL, q);
    update_rgb_directional(q);

    SpatialCanvas* cv = p->canvases[0];
    uint32_t best_ref = 0;
    float    best_ref_sim = -1.0f;
    for (uint32_t s = 0; s < cv->slot_count; s++) {
        float v = canvas_slot_cosine_a(cv, s, q);
        if (v > best_ref_sim) { best_ref_sim = v; best_ref = s; }
    }

    /* pool_match should produce the same slot */
    PoolMatchResult r = pool_match(p, q, qtxt);
    printf("\n    direct best=slot %u sim=%.3f\n", best_ref, best_ref_sim);
    printf("    pool_match =slot %u sim=%.3f (canvas %u, step %d)\n",
           r.slot_id, r.similarity, r.canvas_id, r.step_taken);

    assert(r.slot_id == best_ref);
    assert(fabsf(r.similarity - best_ref_sim) < 1e-4f);

    grid_destroy(q);
    pool_destroy(p);
    PASS();
}

/* ── 8. Save/load roundtrip preserves pool + track ── */
static void test_save_load_roundtrip(void) {
    TEST("ai_save/ai_load preserves canvas pool + subtitle track");
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    SpatialCanvasPool* p = ai_get_canvas_pool(ai);
    assert(p != NULL);

    /* 3 prose + 2 code, spread across canvases */
    pool_add_clause(p,
        "The gentle rain fell upon the old garden pathway where stone benches "
        "stood beneath the broad leaves of aging oak trees in silent community.");
    pool_add_clause(p,
        "Each morning the gardener would walk among the roses and carefully tend "
        "to them, removing stray leaves and providing proper water supplies as needed.");
    pool_add_clause(p,
        "int square(int x){return x*x;}");
    pool_add_clause(p,
        "Occasionally visitors would pause at the wrought iron gate to admire "
        "the blooming flowers and comment on the extraordinary beauty of the place.");
    pool_add_clause(p,
        "void print_list(int* a,int n){for(int i=0;i<n;i++)printf(\"%d,\",a[i]);}");

    uint32_t pre_count = p->count;
    uint32_t pre_slots = pool_total_slots(p);
    uint32_t pre_prose_n, pre_code_n;
    subtitle_track_ids_of_type(&p->track, DATA_PROSE, &pre_prose_n);
    subtitle_track_ids_of_type(&p->track, DATA_CODE,  &pre_code_n);

    #ifdef _WIN32
    system("if not exist build mkdir build");
    #else
    system("mkdir -p build");
    #endif
    const char* path = "build/test_subtitle_roundtrip.spai";
    SpaiStatus s = ai_save(ai, path);
    assert(s == SPAI_OK);

    spatial_ai_destroy(ai);

    SpaiStatus ls;
    SpatialAI* ai2 = ai_load(path, &ls);
    assert(ls == SPAI_OK);
    assert(ai2->canvas_pool != NULL);

    uint32_t post_count = ai2->canvas_pool->count;
    uint32_t post_slots = pool_total_slots(ai2->canvas_pool);
    uint32_t post_prose_n, post_code_n;
    subtitle_track_ids_of_type(&ai2->canvas_pool->track, DATA_PROSE, &post_prose_n);
    subtitle_track_ids_of_type(&ai2->canvas_pool->track, DATA_CODE,  &post_code_n);

    printf("\n    pre:  canvases=%u slots=%u prose=%u code=%u\n",
           pre_count, pre_slots, pre_prose_n, pre_code_n);
    printf("    post: canvases=%u slots=%u prose=%u code=%u\n",
           post_count, post_slots, post_prose_n, post_code_n);

    assert(post_count == pre_count);
    assert(post_slots == pre_slots);
    assert(post_prose_n == pre_prose_n);
    assert(post_code_n  == pre_code_n);

    /* Query after reload still works */
    SpatialGrid* q = grid_create();
    layers_encode_clause("int square(int x){return x*x;}", NULL, q);
    update_rgb_directional(q);
    PoolMatchResult r = pool_match(ai2->canvas_pool, q, "int square(int x){return x*x;}");
    assert(r.similarity > 0.5f);
    grid_destroy(q);

    spatial_ai_destroy(ai2);
    remove(path);
    PASS();
}

int main(void) {
    printf("=== test_subtitle ===\n");

    test_detect_data_type();
    test_track_add_and_lookup();
    test_pool_routing_same_type();
    test_pool_routing_mixed();
    test_subtitle_jump();
    test_subtitle_fallback();
    test_subtitle_non_intrusive();
    test_save_load_roundtrip();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
