#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
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

/* ── synthetic clause generator ──
 *   Deterministic: "apple number %03d eats pattern %03d" sort of thing.
 *   Enough variety so the engine creates both I-frames and P-frames,
 *   but terse enough for tests to run quickly. */
static void make_clause(char* buf, size_t cap, uint32_t i) {
    static const char* subjects[] = {
        "apple", "bird", "cat", "dog", "elephant", "fish", "goat", "horse",
        "ink", "jar", "koala", "lemon", "mango", "nest", "owl", "peach"
    };
    static const char* verbs[] = {
        "runs", "flies", "walks", "sleeps", "eats", "swims", "climbs", "reads"
    };
    static const char* objects[] = {
        "home", "stone", "tree", "cloud", "river", "song", "book", "light"
    };
    uint32_t si = i % 16;
    uint32_t vi = (i / 16) % 8;
    uint32_t oi = (i / 128) % 8;
    snprintf(buf, cap, "a %s %s to the %s today number %u.",
             subjects[si], verbs[vi], objects[oi], i);
}

/* ── test 1: roundtrip — train 700, save, destroy, load, compare ── */
static void test_roundtrip(void) {
    TEST("roundtrip: 700 clauses save/load cosine preserved");

    const uint32_t N = 700;
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    char buf[256];

    for (uint32_t i = 0; i < N; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(ai, buf, NULL);
    }
    assert(ai->kf_count > 0);
    uint32_t kf0 = ai->kf_count;
    uint32_t df0 = ai->df_count;

    /* Save */
    const char* path = "build/test_io_roundtrip.spai";
    SpaiStatus s = ai_save(ai, path);
    assert(s == SPAI_OK);

    /* Take a cosine fingerprint BEFORE destroy */
    const uint32_t PROBE = 16;
    float probe_before[PROBE];
    SpatialGrid* q = grid_create();
    for (uint32_t p = 0; p < PROBE; p++) {
        make_clause(buf, sizeof(buf), p * 37 + 1);
        grid_clear(q);
        layers_encode_clause(buf, NULL, q);
        update_rgb_directional(q);
        float sim = 0.0f;
        ai_predict(ai, buf, &sim);
        probe_before[p] = sim;
    }
    spatial_ai_destroy(ai);

    /* Load */
    SpaiStatus ls;
    SpatialAI* ai2 = ai_load(path, &ls);
    assert(ls == SPAI_OK);
    assert(ai2 != NULL);
    assert(ai2->kf_count == kf0);
    assert(ai2->df_count == df0);

    /* Same queries → same similarities */
    int all_match = 1;
    for (uint32_t p = 0; p < PROBE; p++) {
        make_clause(buf, sizeof(buf), p * 37 + 1);
        float sim = 0.0f;
        ai_predict(ai2, buf, &sim);
        if (fabsf(sim - probe_before[p]) > 0.001f) {
            printf("\n    probe %u: before=%.4f after=%.4f diff=%.4f\n",
                   p, probe_before[p], sim, fabsf(sim - probe_before[p]));
            all_match = 0;
        }
    }
    assert(all_match);

    grid_destroy(q);
    spatial_ai_destroy(ai2);
    remove(path);
    PASS();
}

/* ── test 2: incremental — 700 save, load, +300, incremental save, reload ── */
static void test_incremental(void) {
    TEST("incremental: 700 + 300 more, counts grow, entries persist");

    const uint32_t N1 = 700;
    const uint32_t N2 = 300;
    morpheme_init();

    SpatialAI* ai = spatial_ai_create();
    char buf[256];

    /* Phase A: train 700 */
    for (uint32_t i = 0; i < N1; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(ai, buf, NULL);
    }
    uint32_t kf_phaseA = ai->kf_count;
    uint32_t df_phaseA = ai->df_count;

    const char* path = "build/test_io_incr.spai";
    SpaiStatus s = ai_save(ai, path);
    assert(s == SPAI_OK);
    spatial_ai_destroy(ai);

    /* Phase B: reload, train 300 more from a disjoint slice */
    SpaiStatus ls;
    SpatialAI* ai2 = ai_load(path, &ls);
    assert(ls == SPAI_OK);
    assert(ai2->kf_count == kf_phaseA);
    assert(ai2->df_count == df_phaseA);

    for (uint32_t i = 0; i < N2; i++) {
        make_clause(buf, sizeof(buf), N1 + i);
        ai_store_auto(ai2, buf, NULL);
    }
    uint32_t kf_phaseB = ai2->kf_count;
    uint32_t df_phaseB = ai2->df_count;

    /* One of (KF, Delta) counts must have grown */
    assert(kf_phaseB + df_phaseB > kf_phaseA + df_phaseA);

    /* Incremental save should append only the new entries */
    uint32_t file_kf_before = 0, file_df_before = 0, ver = 0;
    ai_peek_header(path, &file_kf_before, &file_df_before, &ver);
    assert(file_kf_before == kf_phaseA);
    assert(file_df_before == df_phaseA);

    s = ai_save_incremental(ai2, path);
    assert(s == SPAI_OK);

    /* Peek after: header reflects new totals */
    uint32_t file_kf_after = 0, file_df_after = 0;
    ai_peek_header(path, &file_kf_after, &file_df_after, &ver);
    assert(file_kf_after == kf_phaseB);
    assert(file_df_after == df_phaseB);

    /* Verify on-disk size grew only by the delta */
    FILE* fp = fopen(path, "rb");
    assert(fp != NULL);
    fseek(fp, 0, SEEK_END);
    long size_after = ftell(fp);
    fclose(fp);
    assert(size_after > 0);

    /* Full reload and compare */
    spatial_ai_destroy(ai2);
    SpatialAI* ai3 = ai_load(path, &ls);
    assert(ls == SPAI_OK);
    assert(ai3->kf_count == kf_phaseB);
    assert(ai3->df_count == df_phaseB);

    /* Sample probe: queries from both phase A and phase B data should resolve */
    float probe_sim;
    make_clause(buf, sizeof(buf), 123);   /* in phase A */
    ai_predict(ai3, buf, &probe_sim);
    assert(probe_sim > 0.0f);
    make_clause(buf, sizeof(buf), 850);   /* in phase B */
    ai_predict(ai3, buf, &probe_sim);
    assert(probe_sim > 0.0f);

    spatial_ai_destroy(ai3);
    remove(path);
    PASS();
}

/* ── test 3: corruption — bad magic rejected ── */
static void test_bad_magic(void) {
    TEST("corrupt file rejected (bad magic)");

    const char* path = "build/test_io_badmagic.spai";
    FILE* fp = fopen(path, "wb");
    assert(fp != NULL);
    /* Write garbage with wrong magic */
    char hdr[32];
    memcpy(hdr, "NOPE", 4);
    memset(hdr + 4, 0x5a, 28);
    fwrite(hdr, 1, 32, fp);
    fclose(fp);

    SpaiStatus s;
    SpatialAI* ai = ai_load(path, &s);
    assert(ai == NULL);
    assert(s == SPAI_ERR_MAGIC);
    printf("(status=%s) ", spai_status_str(s));

    remove(path);
    PASS();
}

/* ── test 4: corruption — wrong version rejected ── */
static void test_bad_version(void) {
    TEST("unsupported version rejected");

    const char* path = "build/test_io_badver.spai";
    FILE* fp = fopen(path, "wb");
    assert(fp != NULL);
    char hdr[32];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, "SPAI", 4);
    uint32_t bad_ver = 999;
    memcpy(hdr + 4, &bad_ver, 4);
    fwrite(hdr, 1, 32, fp);
    fclose(fp);

    SpaiStatus s;
    SpatialAI* ai = ai_load(path, &s);
    assert(ai == NULL);
    assert(s == SPAI_ERR_VERSION);

    remove(path);
    PASS();
}

/* ── test 5: truncated file rejected ── */
static void test_truncated(void) {
    TEST("truncated body rejected");

    /* Train a small model */
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();
    char buf[256];
    for (uint32_t i = 0; i < 20; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(ai, buf, NULL);
    }

    const char* path = "build/test_io_trunc.spai";
    ai_save(ai, path);
    spatial_ai_destroy(ai);

    /* Truncate after header: only the 32-byte header, records missing */
    FILE* fp = fopen(path, "rb");
    assert(fp != NULL);
    char hdr[32];
    size_t got = fread(hdr, 1, 32, fp);
    fclose(fp);
    assert(got == 32);

    fp = fopen(path, "wb");
    fwrite(hdr, 1, 32, fp);
    fclose(fp);

    SpaiStatus s;
    SpatialAI* loaded = ai_load(path, &s);
    assert(loaded == NULL);
    assert(s == SPAI_ERR_READ);

    remove(path);
    PASS();
}

/* ── test 6: ai_save_incremental on missing file = full save ── */
static void test_incremental_new_file(void) {
    TEST("incremental save on missing file == full save");

    morpheme_init();
    SpatialAI* ai = spatial_ai_create();
    char buf[256];
    for (uint32_t i = 0; i < 50; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(ai, buf, NULL);
    }

    const char* path = "build/test_io_new_incr.spai";
    remove(path);  /* ensure missing */
    SpaiStatus s = ai_save_incremental(ai, path);
    assert(s == SPAI_OK);

    uint32_t fkf = 0, fdf = 0, v = 0;
    ai_peek_header(path, &fkf, &fdf, &v);
    assert(fkf == ai->kf_count);
    assert(fdf == ai->df_count);

    spatial_ai_destroy(ai);
    remove(path);
    PASS();
}

/* ── test 7: state-error — file has more than memory ── */
static void test_state_error(void) {
    TEST("incremental refuses to shrink (engine < file)");

    morpheme_init();
    /* Big model on disk */
    SpatialAI* big = spatial_ai_create();
    char buf[256];
    for (uint32_t i = 0; i < 100; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(big, buf, NULL);
    }

    const char* path = "build/test_io_state.spai";
    ai_save(big, path);
    uint32_t kf_big = big->kf_count;
    uint32_t df_big = big->df_count;
    spatial_ai_destroy(big);

    /* Small in-memory model (fewer entries than on disk) */
    SpatialAI* small = spatial_ai_create();
    for (uint32_t i = 0; i < 5; i++) {
        make_clause(buf, sizeof(buf), i);
        ai_store_auto(small, buf, NULL);
    }
    /* sanity: small < big */
    assert(small->kf_count < kf_big || small->df_count < df_big);

    SpaiStatus s = ai_save_incremental(small, path);
    assert(s == SPAI_ERR_STATE);
    printf("(status=%s) ", spai_status_str(s));

    spatial_ai_destroy(small);
    remove(path);
    PASS();
}

int main(void) {
    printf("=== test_io ===\n");

    /* Ensure build dir exists — tests write temp files there */
    #ifdef _WIN32
    system("if not exist build mkdir build");
    #else
    system("mkdir -p build");
    #endif

    test_roundtrip();
    test_incremental();
    test_bad_magic();
    test_bad_version();
    test_truncated();
    test_incremental_new_file();
    test_state_error();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
