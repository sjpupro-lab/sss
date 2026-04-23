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

/* ── C1-3 ── recluster stub is a safe no-op (baseline preserved) ── */
static void test_recluster_stub(void) {
    TEST("pool_recluster_by_topic (C1 stub) leaves the pool unchanged");
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
    assert(r.committed == 0);
    assert(r.canvases_reordered == 0);
    assert(p->count == before);
    assert(p->next_sequence_id == before_seq);

    pool_destroy(p);
    PASS();
}

int main(void) {
    printf("=== test_recluster ===\n");

    test_iterate_order();
    test_iterate_nulls();
    test_recluster_stub();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
