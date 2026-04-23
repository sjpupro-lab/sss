/* v4 Task C — re-clustering implementation.
 *
 * C1 (this file's current content): iteration helper only. The real
 * pool_recluster_by_topic is stubbed out and lands in the C2 commit
 * so the baseline remains byte-identical until the topic-grouping
 * path is fully tested. */

#include "spatial_recluster.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ── pool_iterate_in_sequence_order ────────────────────── */

typedef struct {
    const SpatialCanvas* canvas;
    uint32_t             slot;
    uint32_t             sequence_id;
} SeqRef;

static int cmp_seq_ref(const void* a, const void* b) {
    uint32_t sa = ((const SeqRef*)a)->sequence_id;
    uint32_t sb = ((const SeqRef*)b)->sequence_id;
    if (sa < sb) return -1;
    if (sa > sb) return 1;
    return 0;
}

void pool_iterate_in_sequence_order(
    const SpatialCanvasPool* p,
    void (*visit)(const SpatialCanvas* c, uint32_t slot, void* user),
    void* user) {
    if (!p || !visit) return;
    if (p->count == 0) return;

    /* Collect every occupied slot across every canvas. */
    uint32_t cap = p->count * CV_SLOTS;
    SeqRef* refs = (SeqRef*)malloc(cap * sizeof(SeqRef));
    if (!refs) return;
    uint32_t n = 0;
    for (uint32_t ci = 0; ci < p->count; ci++) {
        const SpatialCanvas* c = p->canvases[ci];
        if (!c) continue;
        for (uint32_t s = 0; s < CV_SLOTS; s++) {
            if (!c->meta[s].occupied) continue;
            refs[n].canvas = c;
            refs[n].slot = s;
            refs[n].sequence_id = c->meta[s].sequence_id;
            n++;
        }
    }
    qsort(refs, n, sizeof(SeqRef), cmp_seq_ref);
    for (uint32_t i = 0; i < n; i++) {
        visit(refs[i].canvas, refs[i].slot, user);
    }
    free(refs);
}

/* ── pool_recluster_by_topic (C1 stub) ──────────────────
 *
 * C1 intentionally leaves the pool untouched and returns a zeroed
 * report with committed = 0. The full greedy / commit-gate logic
 * arrives in C2. A stub keeps the public symbol available so
 * downstream code and tests compile/link now. */

ReclusterReport pool_recluster_by_topic(SpatialCanvasPool* p,
                                        float min_gain_ratio) {
    (void)p;
    (void)min_gain_ratio;
    ReclusterReport r;
    memset(&r, 0, sizeof r);
    return r;
}
