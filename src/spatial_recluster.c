/* v4 Task C — re-clustering implementation.
 *
 * Core pieces:
 *   pool_iterate_in_sequence_order — read-only walk by sequence_id
 *   pool_recluster_by_topic         — offline grouping + greedy reorder
 *
 * The recluster path groups canvases by their dominant topic_hash
 * (exact-match; bit distance is explicitly not used — see SPEC v4),
 * walks each group greedily to minimize adjacent canvas_delta_rle_
 * bytes, measures total RLE cost before/after, and commits the new
 * physical order only if the relative gain meets min_gain_ratio.
 * Slot-level SlotMeta.sequence_id is left untouched; pool_iterate_in_
 * sequence_order remains the authoritative source of logical order. */

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

/* ── pool_recluster_by_topic ─────────────────────────────
 *
 * A canvas typically packs CV_SLOTS=32 clauses. A single canvas may
 * therefore contain slots from several topics; its "topic identity"
 * for grouping purposes is the mode of topic_hash across its occupied
 * slots (exact match only). Canvases with no occupied slots are
 * excluded from grouping and left in-place at the tail. */
static uint32_t canvas_dominant_topic(const SpatialCanvas* c) {
    uint32_t best_h = 0;
    uint32_t best_n = 0;
    for (uint32_t i = 0; i < CV_SLOTS; i++) {
        if (!c->meta[i].occupied) continue;
        uint32_t h = c->meta[i].topic_hash;
        uint32_t n = 0;
        for (uint32_t j = 0; j < CV_SLOTS; j++) {
            if (!c->meta[j].occupied) continue;
            if (c->meta[j].topic_hash == h) n++;
        }
        if (n > best_n) { best_n = n; best_h = h; }
    }
    return best_h;
}

/* Adjacent-pair RLE bytes for canvases in the order they appear in
 * `order[]`. `tmp` is scratch of size CV_TOTAL for the sparse delta
 * buffer. Pairs with a NULL canvas contribute 0. */
static uint32_t total_rle_bytes(SpatialCanvas* const* canvases,
                                const uint32_t* order,
                                uint32_t n,
                                CanvasDeltaEntry* tmp) {
    if (n < 2) return 0;
    uint32_t sum = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        const SpatialCanvas* a = canvases[order[i]];
        const SpatialCanvas* b = canvases[order[i + 1]];
        if (!a || !b) continue;
        uint32_t cnt = canvas_delta_sparse(a, b, tmp, CV_TOTAL);
        sum += canvas_delta_rle_bytes(tmp, cnt);
    }
    return sum;
}

/* Greedy order for `members[]` (indices into canvases[]):
 * start at members[0]; repeatedly pick the unvisited member whose
 * RLE delta vs. the current tail is smallest. Writes the result in
 * place. O(k^2) delta evaluations per group of size k. */
static void greedy_order_group(SpatialCanvas* const* canvases,
                               uint32_t* members, uint32_t k,
                               CanvasDeltaEntry* tmp) {
    if (k < 3) return;  /* 0/1 trivial; size-2 has nothing to choose */
    for (uint32_t pos = 0; pos + 1 < k; pos++) {
        uint32_t cur = members[pos];
        uint32_t best_j = pos + 1;
        uint32_t best_cost = UINT32_MAX;
        for (uint32_t j = pos + 1; j < k; j++) {
            const SpatialCanvas* a = canvases[cur];
            const SpatialCanvas* b = canvases[members[j]];
            if (!a || !b) continue;
            uint32_t cnt = canvas_delta_sparse(a, b, tmp, CV_TOTAL);
            uint32_t cost = canvas_delta_rle_bytes(tmp, cnt);
            if (cost < best_cost) { best_cost = cost; best_j = j; }
        }
        if (best_j != pos + 1) {
            uint32_t swap = members[pos + 1];
            members[pos + 1] = members[best_j];
            members[best_j]  = swap;
        }
    }
}

ReclusterReport pool_recluster_by_topic(SpatialCanvasPool* p,
                                        float min_gain_ratio) {
    ReclusterReport r;
    memset(&r, 0, sizeof r);

    if (!p || p->count < 2) return r;

    const uint32_t N = p->count;

    /* Scratch: sparse delta buffer reused across every pair. */
    CanvasDeltaEntry* tmp =
        (CanvasDeltaEntry*)malloc((size_t)CV_TOTAL * sizeof(CanvasDeltaEntry));
    if (!tmp) return r;

    /* Per-canvas dominant topic_hash. Canvases with no occupied slots
     * get topic = 0 and a "no_topic" flag so they drop to the tail. */
    uint32_t* topics   = (uint32_t*)malloc(N * sizeof(uint32_t));
    int*      no_topic = (int*)     calloc(N,  sizeof(int));
    if (!topics || !no_topic) { free(tmp); free(topics); free(no_topic); return r; }

    for (uint32_t i = 0; i < N; i++) {
        SpatialCanvas* c = p->canvases[i];
        if (!c) { topics[i] = 0; no_topic[i] = 1; continue; }
        int any = 0;
        for (uint32_t s = 0; s < CV_SLOTS; s++) {
            if (c->meta[s].occupied) { any = 1; break; }
        }
        if (!any) { topics[i] = 0; no_topic[i] = 1; }
        else      { topics[i] = canvas_dominant_topic(c); }
    }

    /* Candidate order: group-contiguous by first appearance, greedy
     * inside each group. Empty canvases stay at the tail in their
     * original relative order. */
    uint32_t* cand = (uint32_t*)malloc(N * sizeof(uint32_t));
    int*      used = (int*)     calloc(N,  sizeof(int));
    if (!cand || !used) {
        free(tmp); free(topics); free(no_topic); free(cand); free(used);
        return r;
    }

    uint32_t pos = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (used[i] || no_topic[i]) continue;
        /* Collect this group's members in first-appearance order. */
        uint32_t start = pos;
        uint32_t topic = topics[i];
        for (uint32_t j = i; j < N; j++) {
            if (used[j] || no_topic[j]) continue;
            if (topics[j] == topic) {
                cand[pos++] = j;
                used[j] = 1;
            }
        }
        uint32_t k = pos - start;
        if (k >= 2) r.groups_examined++;
        if (k >= 3) greedy_order_group(p->canvases, &cand[start], k, tmp);
    }
    for (uint32_t i = 0; i < N; i++) {
        if (!used[i]) { cand[pos++] = i; used[i] = 1; }
    }

    /* Baseline order = identity. */
    uint32_t* base = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!base) {
        free(tmp); free(topics); free(no_topic); free(cand); free(used);
        return r;
    }
    for (uint32_t i = 0; i < N; i++) base[i] = i;

    r.bytes_before = total_rle_bytes(p->canvases, base, N, tmp);
    r.bytes_after  = total_rle_bytes(p->canvases, cand, N, tmp);
    r.compression_gain = (r.bytes_before > 0)
        ? ((float)(int64_t)(r.bytes_before - r.bytes_after) /
           (float)r.bytes_before)
        : 0.0f;

    int reorders_exist = 0;
    for (uint32_t i = 0; i < N; i++) if (cand[i] != i) { reorders_exist = 1; break; }

    if (reorders_exist && r.compression_gain >= min_gain_ratio) {
        /* Build permutation map old_idx -> new_idx so SubtitleTrack
         * entries can follow. cand[new] = old, so inv[old] = new. */
        uint32_t* inv = (uint32_t*)malloc(N * sizeof(uint32_t));
        SpatialCanvas** newarr =
            (SpatialCanvas**)malloc(N * sizeof(SpatialCanvas*));
        if (!inv || !newarr) {
            free(inv); free(newarr);
            free(tmp); free(topics); free(no_topic);
            free(cand); free(used); free(base);
            return r;
        }
        for (uint32_t newi = 0; newi < N; newi++) {
            inv[cand[newi]] = newi;
            newarr[newi] = p->canvases[cand[newi]];
        }

        /* Commit: swap pool->canvases and re-point every SubtitleTrack
         * entry whose canvas_id references a moved canvas. */
        memcpy(p->canvases, newarr, N * sizeof(SpatialCanvas*));
        for (uint32_t e = 0; e < p->track.count; e++) {
            uint32_t old_cid = p->track.entries[e].canvas_id;
            if (old_cid < N) p->track.entries[e].canvas_id = inv[old_cid];
        }
        for (uint32_t i = 0; i < N; i++) if (inv[i] != i) r.canvases_reordered++;
        r.committed = 1;

        free(inv); free(newarr);
    }

    free(tmp); free(topics); free(no_topic);
    free(cand); free(used); free(base);
    return r;
}
