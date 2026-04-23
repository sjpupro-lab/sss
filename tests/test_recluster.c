/* v4 Task C — re-clustering tests.
 *
 * C1 coverage: pool_iterate_in_sequence_order visits every occupied
 * slot exactly once in ascending sequence_id order, regardless of
 * where the slot ended up physically. Full pool_recluster_by_topic
 * behavior lands in C2. */

#include "spatial_morpheme.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"
#include "spatial_recluster.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)

/* Collector used by the iterate callback. */
typedef struct {
    uint32_t ids[256];
    uint32_t n;
} SeqCollect;

static void collect_seq(const SpatialCanvas* c, uint32_t slot, void* user) {
    (void)c;
    SeqCollect* out = (SeqCollect*)user;
    if (out->n < 256) {
        out->ids[out->n++] = c->meta[slot].sequence_id;
    }
}

/* ── C1-1 ── iterate sees every occupied slot once, in order ── */
static void test_iterate_order(void) {
    TEST("pool_iterate_in_sequence_order walks every occupied slot in order");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();
    assert(p != NULL);

    /* Mix types so the pool spreads placements across multiple canvases.
     * next_sequence_id will stamp 0..N-1 in insertion order. */
    const char* clauses[] = {
        "The soft rain fell upon the old garden walkway at dawn today indeed truly.",
        "int square(int x){return x*x;}",
        "Each morning the gardener tended the roses and removed the stray leaves everywhere.",
        "void print_sum(int* a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];printf(\"%d\",s);}",
        "Visitors paused at the iron gate to admire the blooming wild flowers brightly.",
        "int fib(int n){return n<2?n:fib(n-1)+fib(n-2);}",
        "The afternoon light painted the field in broad soft amber colors gently warmly.",
        "char* trim(char* s){while(*s==' ')s++;return s;}",
    };
    const uint32_t N = (uint32_t)(sizeof(clauses) / sizeof(clauses[0]));
    for (uint32_t i = 0; i < N; i++) {
        int r = pool_add_clause(p, clauses[i]);
        assert(r >= 0);
    }
    assert(p->next_sequence_id == N);

    SeqCollect out; out.n = 0;
    pool_iterate_in_sequence_order(p, collect_seq, &out);
    assert(out.n == N);
    for (uint32_t i = 0; i < N; i++) {
        assert(out.ids[i] == i);  /* sequence_ids come back in 0..N-1 */
    }
    pool_destroy(p);
    PASS();
}

/* ── C1-2 ── NULL pool / NULL visit are no-ops ── */
static void test_iterate_nulls(void) {
    TEST("pool_iterate_in_sequence_order is a no-op on NULL inputs");
    SeqCollect out; out.n = 0;
    pool_iterate_in_sequence_order(NULL, collect_seq, &out);
    assert(out.n == 0);

    SpatialCanvasPool* p = pool_create();
    pool_iterate_in_sequence_order(p, NULL, &out);  /* visit=NULL */
    assert(out.n == 0);
    pool_destroy(p);
    PASS();
}

/* ── C1-3 ── recluster on a small single-topic pool commits nothing
 *           (nothing to reorder). ── */
static void test_recluster_small_pool(void) {
    TEST("pool_recluster_by_topic on a 1-canvas pool leaves it unchanged");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();
    for (uint32_t i = 0; i < 6; i++) {
        char buf[128];
        snprintf(buf, sizeof buf,
                 "the quick brown fox jumps over the lazy dog on day number %u today.", i);
        pool_add_clause(p, buf);
    }
    uint32_t before = p->count;
    uint32_t before_seq = p->next_sequence_id;

    ReclusterReport r = pool_recluster_by_topic(p, 0.10f);
    /* With a single canvas there's nothing to swap. */
    assert(r.committed == 0);
    assert(r.canvases_reordered == 0);
    assert(p->count == before);
    assert(p->next_sequence_id == before_seq);

    pool_destroy(p);
    PASS();
}

/* ── C2 ── 10-topic × N-clause scenario ── fill many same-topic
 * canvases, shuffle them physically, then expect recluster to regroup
 * by topic_hash and keep the subtitle track + sequence_id consistent. */

/* Build clauses that share the same topic_hash (djb2 over identical
 * text); 32+ copies of one clause fill one canvas completely. */
static void fill_canvas_with(SpatialCanvasPool* p, const char* text, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) pool_add_clause(p, text);
}

static void test_recluster_regroups_topics(void) {
    TEST("recluster regroups interleaved same-topic canvases");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();

    /* 3 topics, 64 clauses each → 6 canvases total (2 per topic),
     * fully occupying CV_SLOTS=32 slots each. Interleaving the inserts
     * spreads each topic's canvases across non-adjacent physical
     * indices, giving recluster something to do. */
    const char* topicA = "the quick brown fox jumps over the lazy dog today indeed truly.";
    const char* topicB = "data science teams iterate on the morning backlog carefully always warmly.";
    const char* topicC = "the kitchen smelled of cinnamon and warm bread in the afternoon truly dearly.";

    for (uint32_t round = 0; round < 2; round++) {
        fill_canvas_with(p, topicA, 32);
        fill_canvas_with(p, topicB, 32);
        fill_canvas_with(p, topicC, 32);
    }
    assert(p->count == 6);

    /* Sanity: every canvas is full and dominant-topic is one of A/B/C. */
    for (uint32_t ci = 0; ci < p->count; ci++) {
        assert(p->canvases[ci]->slot_count == CV_SLOTS);
    }

    /* Snapshot original sequence_ids keyed by entry for later order
     * comparison. Before recluster, subtitle entries line up with
     * physical slots. */
    uint32_t pre_next_seq = p->next_sequence_id;

    /* Map subtitle entry -> current (canvas_id, slot_id) so we can
     * check that after recluster the referenced canvas still contains
     * the original sequence_id at that slot. Since we only reorder
     * canvases (not slots within a canvas), slot_id is stable. */
    uint32_t pre_entry_seq[256];
    assert(p->track.count <= 256);
    for (uint32_t e = 0; e < p->track.count; e++) {
        uint32_t cid = p->track.entries[e].canvas_id;
        uint32_t sid = p->track.entries[e].slot_id;
        pre_entry_seq[e] = p->canvases[cid]->meta[sid].sequence_id;
    }

    ReclusterReport r = pool_recluster_by_topic(p, 0.05f);
    printf("\n    groups=%u reordered=%u before=%u after=%u gain=%.3f committed=%d\n",
           r.groups_examined, r.canvases_reordered,
           r.bytes_before, r.bytes_after, r.compression_gain, r.committed);

    /* 3 groups of 2 canvases each are examined. */
    assert(r.groups_examined == 3);
    /* Shouldn't inflate bytes: the new order is at least as good as
     * the interleaved baseline for this scenario (identical canvases
     * placed adjacent → minimal delta). */
    assert(r.bytes_after <= r.bytes_before);

    /* sequence counter and total slot count are invariant regardless
     * of whether commit fired. */
    assert(p->next_sequence_id == pre_next_seq);
    uint32_t post_total = pool_total_slots(p);
    assert(post_total == 6 * CV_SLOTS);

    /* SubtitleTrack consistency: each entry's (canvas_id, slot_id)
     * must still resolve to the sequence_id it originally pointed at. */
    for (uint32_t e = 0; e < p->track.count; e++) {
        uint32_t cid = p->track.entries[e].canvas_id;
        uint32_t sid = p->track.entries[e].slot_id;
        assert(cid < p->count);
        assert(p->canvases[cid]->meta[sid].sequence_id == pre_entry_seq[e]);
    }

    /* If committed, same-topic canvases should now be physically
     * contiguous. */
    if (r.committed) {
        assert(r.canvases_reordered > 0);
        uint32_t prev_topic = 0;
        int seen = 0, transitions = 0;
        for (uint32_t ci = 0; ci < p->count; ci++) {
            /* Use the first occupied slot's topic as a cheap probe. */
            uint32_t t = 0;
            for (uint32_t s = 0; s < CV_SLOTS; s++) {
                if (p->canvases[ci]->meta[s].occupied) {
                    t = p->canvases[ci]->meta[s].topic_hash;
                    break;
                }
            }
            if (seen && t != prev_topic) transitions++;
            prev_topic = t;
            seen = 1;
        }
        /* 3 topics grouped contiguously → 2 transitions (A→B, B→C).
         * Interleaved baseline would show 5. */
        assert(transitions == 2);
    }

    pool_destroy(p);
    PASS();
}

/* ── C2 ── sequence-order iterator still recovers the original
 * utterance order even if recluster scrambled physical indices. ── */
static void test_recluster_preserves_sequence_order(void) {
    TEST("pool_iterate_in_sequence_order survives recluster");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();
    const char* topicA = "the quick brown fox jumps over the lazy dog today indeed truly.";
    const char* topicB = "data science teams iterate on the morning backlog carefully always warmly.";
    for (uint32_t round = 0; round < 2; round++) {
        fill_canvas_with(p, topicA, 32);
        fill_canvas_with(p, topicB, 32);
    }
    uint32_t N = p->next_sequence_id;

    /* Snapshot sequence_id walk pre-recluster. */
    SeqCollect pre; pre.n = 0;
    pool_iterate_in_sequence_order(p, collect_seq, &pre);
    assert(pre.n == N);

    ReclusterReport r = pool_recluster_by_topic(p, 0.05f);
    (void)r;

    SeqCollect post; post.n = 0;
    pool_iterate_in_sequence_order(p, collect_seq, &post);
    assert(post.n == N);
    for (uint32_t i = 0; i < N; i++) assert(pre.ids[i] == post.ids[i]);
    for (uint32_t i = 0; i < N; i++) assert(post.ids[i] == i);

    pool_destroy(p);
    PASS();
}

/* ── C3 ── impossibly high gain threshold: candidate is good but
 * doesn't clear the bar → pool must stay byte-identical. ── */
static void test_recluster_rollback_on_low_gain(void) {
    TEST("recluster rolls back when compression_gain < min_gain_ratio");
    morpheme_init();

    SpatialCanvasPool* p = pool_create();
    const char* topicA = "the quick brown fox jumps over the lazy dog today indeed truly.";
    const char* topicB = "data science teams iterate on the morning backlog carefully always warmly.";
    for (uint32_t round = 0; round < 2; round++) {
        fill_canvas_with(p, topicA, 32);
        fill_canvas_with(p, topicB, 32);
    }
    assert(p->count == 4);

    /* Snapshot the full canvas pointer array + every subtitle entry's
     * (canvas_id, slot_id) tuple. */
    SpatialCanvas* pre_ptrs[8];
    for (uint32_t i = 0; i < p->count; i++) pre_ptrs[i] = p->canvases[i];
    uint32_t pre_cids[256], pre_sids[256];
    assert(p->track.count <= 256);
    for (uint32_t e = 0; e < p->track.count; e++) {
        pre_cids[e] = p->track.entries[e].canvas_id;
        pre_sids[e] = p->track.entries[e].slot_id;
    }

    /* 200% gain is impossible — every canvas would have to collapse
     * to zero bytes. Report should describe the candidate but commit
     * nothing. */
    ReclusterReport r = pool_recluster_by_topic(p, 2.0f);
    assert(r.committed == 0);
    assert(r.canvases_reordered == 0);
    assert(r.groups_examined == 2);  /* two topics, two canvases each */

    /* Pool layout is untouched. */
    for (uint32_t i = 0; i < p->count; i++) assert(p->canvases[i] == pre_ptrs[i]);
    for (uint32_t e = 0; e < p->track.count; e++) {
        assert(p->track.entries[e].canvas_id == pre_cids[e]);
        assert(p->track.entries[e].slot_id   == pre_sids[e]);
    }
    pool_destroy(p);
    PASS();
}

/* ── C3 ── empty / NULL inputs: report committed = 0, no crash ── */
static void test_recluster_empty_and_null(void) {
    TEST("recluster handles empty pool and NULL safely");

    ReclusterReport r0 = pool_recluster_by_topic(NULL, 0.10f);
    assert(r0.committed == 0);
    assert(r0.canvases_reordered == 0);

    SpatialCanvasPool* p = pool_create();
    assert(p->count == 0);
    ReclusterReport r1 = pool_recluster_by_topic(p, 0.10f);
    assert(r1.committed == 0);
    assert(r1.canvases_reordered == 0);
    assert(r1.groups_examined == 0);
    pool_destroy(p);
    PASS();
}

int main(void) {
    printf("=== test_recluster ===\n");

    test_iterate_order();
    test_iterate_nulls();
    test_recluster_small_pool();
    test_recluster_regroups_topics();
    test_recluster_preserves_sequence_order();
    test_recluster_rollback_on_low_gain();
    test_recluster_empty_and_null();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
