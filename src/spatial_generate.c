#include "spatial_generate.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_context.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Aggregated tables ─────────────────────────────────── */

AggTables* agg_build(const SpatialAI* ai) {
    if (!ai) return NULL;
    AggTables* t = (AggTables*)calloc(1, sizeof(AggTables));
    if (!t) return NULL;

    /* Sum A; accumulate A-weighted sums of R, G, B per (y, x) */
    for (uint32_t k = 0; k < ai->kf_count; k++) {
        const SpatialGrid* g = &ai->keyframes[k].grid;
        for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            uint16_t a = g->A[i];
            if (a == 0) continue;
            double da = (double)a;
            t->A_sum [i] += da;
            t->R_mean[i] += da * (double)g->R[i];
            t->G_mean[i] += da * (double)g->G[i];
            t->B_mean[i] += da * (double)g->B[i];
        }
    }

    /* Finalize: divide weighted sums by A_sum to get means;
       compute per-row activation totals. */
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (t->A_sum[i] > 0.0) {
                t->R_mean[i] /= t->A_sum[i];
                t->G_mean[i] /= t->A_sum[i];
                t->B_mean[i] /= t->A_sum[i];
            }
            row += t->A_sum[i];
        }
        t->row_total_A[y] = row;
    }
    return t;
}

AggTables* agg_build_from_pool(const struct SpatialCanvasPool_* pool) {
    if (!pool) return NULL;
    AggTables* t = (AggTables*)calloc(1, sizeof(AggTables));
    if (!t) return NULL;

    /* Iterate every populated slot in every canvas, aggregating into
     * tile-local (y, x) coordinates. This mirrors agg_build but with
     * pool as the source of training patterns. */
    for (uint32_t ei = 0; ei < pool->track.count; ei++) {
        const SubtitleEntry* e = &pool->track.entries[ei];
        const SpatialCanvas* c = pool->canvases[e->canvas_id];
        uint32_t x0, y0;
        canvas_slot_byte_offset(e->slot_id, &x0, &y0);

        for (uint32_t dy = 0; dy < GRID_SIZE; dy++) {
            for (uint32_t dx = 0; dx < GRID_SIZE; dx++) {
                uint32_t ti = dy * GRID_SIZE + dx;
                uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
                uint16_t a = c->A[ci];
                if (a == 0) continue;
                double da = (double)a;
                t->A_sum [ti] += da;
                t->R_mean[ti] += da * (double)c->R[ci];
                t->G_mean[ti] += da * (double)c->G[ci];
                t->B_mean[ti] += da * (double)c->B[ci];
            }
        }
    }

    /* Finalise means */
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (t->A_sum[i] > 0.0) {
                t->R_mean[i] /= t->A_sum[i];
                t->G_mean[i] /= t->A_sum[i];
                t->B_mean[i] /= t->A_sum[i];
            }
            row += t->A_sum[i];
        }
        t->row_total_A[y] = row;
    }
    return t;
}

void agg_destroy(AggTables* t) { free(t); }

/* ── Input signature ────────────────────────────────────── */

void input_signature_compute(InputSignature* sig, const SpatialGrid* input) {
    if (!sig || !input) return;
    memset(sig, 0, sizeof(*sig));

    double global_aw = 0.0, global_rw = 0.0, global_gw = 0.0, global_bw = 0.0;

    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double aw = 0.0, rw = 0.0, gw = 0.0, bw = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (input->A[i] == 0) continue;
            double da = (double)input->A[i];
            aw += da;
            rw += da * (double)input->R[i];
            gw += da * (double)input->G[i];
            bw += da * (double)input->B[i];
        }
        if (aw > 0.0) {
            sig->R_row[y] = rw / aw;
            sig->G_row[y] = gw / aw;
            sig->B_row[y] = bw / aw;
            sig->has_activity[y] = 1;
        }
        global_aw += aw;
        global_rw += rw;
        global_gw += gw;
        global_bw += bw;
    }

    if (global_aw > 0.0) {
        sig->R_global = global_rw / global_aw;
        sig->G_global = global_gw / global_aw;
        sig->B_global = global_bw / global_aw;
    }
}

void input_signature_get(const InputSignature* sig, uint32_t y,
                         double* out_R, double* out_G, double* out_B) {
    if (!sig || !out_R || !out_G || !out_B) return;

    /* Fast path: this row has activity */
    if (sig->has_activity[y]) {
        *out_R = sig->R_row[y];
        *out_G = sig->G_row[y];
        *out_B = sig->B_row[y];
        return;
    }

    /* Fallback: nearest active neighbor row within a window */
    for (int d = 1; d < 32; d++) {
        int yu = (int)y - d;
        int yd = (int)y + d;
        if (yu >= 0 && sig->has_activity[yu]) {
            *out_R = sig->R_row[yu];
            *out_G = sig->G_row[yu];
            *out_B = sig->B_row[yu];
            return;
        }
        if (yd < (int)GRID_SIZE && sig->has_activity[yd]) {
            *out_R = sig->R_row[yd];
            *out_G = sig->G_row[yd];
            *out_B = sig->B_row[yd];
            return;
        }
    }

    /* Last resort: global clause signature */
    *out_R = sig->R_global;
    *out_G = sig->G_global;
    *out_B = sig->B_global;
}

/* ── Byte scoring: A × G_sim × R_sim ──────────────────── */

double agg_score_byte(const AggTables* t, uint32_t y, uint8_t v,
                      double in_R, double in_G, double in_B) {
    if (!t) return 0.0;
    uint32_t i = y * GRID_SIZE + (uint32_t)v;
    double A = t->A_sum[i];
    if (A <= 0.0) return 0.0;

    double R = t->R_mean[i];
    double G = t->G_mean[i];
    double B = t->B_mean[i];

    double R_sim = 1.0 - fabs(R - in_R) / 255.0;
    double G_sim = 1.0 - fabs(G - in_G) / 255.0;
    double B_sim = 1.0 - fabs(B - in_B) / 255.0;
    if (R_sim < 0.0) R_sim = 0.0;
    if (G_sim < 0.0) G_sim = 0.0;
    if (B_sim < 0.0) B_sim = 0.0;

    /* Full A × R × G × B product — SPEC §5.1 §9.4 */
    return A * R_sim * G_sim * B_sim;
}

/* ── Grid → text decoding ───────────────────────────────
 *
 * Two variants live side by side:
 *
 *   grid_decode_text       pure row-argmax (legacy). Fast, byte-level,
 *                          no UTF-8 awareness. Kept for callers that
 *                          feed ASCII or don't care about multi-byte
 *                          integrity (e.g. bench_qa byte snapshots).
 *
 *   grid_decode_text_utf8  UTF-8 aware. Validates lead + continuation
 *                          bytes across consecutive rows so Korean and
 *                          other multi-byte output doesn't clip. Used
 *                          by ai_generate_next.
 *
 * Both read row-by-row and stop at the first empty row.
 */

static int utf8_lead_len(uint8_t b) {
    if ((b & 0x80) == 0x00) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

static int utf8_is_cont(uint8_t b) { return (b & 0xC0) == 0x80; }

/* Fill out_bytes with up to n x-candidates sorted by A descending.
 * Missing slots get A=0 sentinels. */
static void row_top_n(const SpatialGrid* g, uint32_t y,
                      uint8_t* out_bytes, uint16_t* out_scores, int n) {
    for (int i = 0; i < n; i++) { out_bytes[i] = 0; out_scores[i] = 0; }
    for (uint32_t x = 0; x < GRID_SIZE; x++) {
        uint16_t a = g->A[y * GRID_SIZE + x];
        if (a == 0) continue;
        for (int k = 0; k < n; k++) {
            if (a > out_scores[k]) {
                for (int j = n - 1; j > k; j--) {
                    out_bytes[j]  = out_bytes[j - 1];
                    out_scores[j] = out_scores[j - 1];
                }
                out_bytes[k]  = (uint8_t)x;
                out_scores[k] = a;
                break;
            }
        }
    }
}

/* Legacy: row-argmax, one byte per row, no UTF-8 validation. */
uint32_t grid_decode_text(const SpatialGrid* g, char* out, uint32_t max_out) {
    if (!g || !out || max_out == 0) return 0;

    uint32_t written = 0;
    for (uint32_t y = 0; y < GRID_SIZE && written + 1 < max_out; y++) {
        uint32_t best_x = 0;
        uint16_t best_a = 0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (g->A[i] > best_a) {
                best_a = g->A[i];
                best_x = x;
            }
        }
        if (best_a == 0) break;
        out[written++] = (char)(uint8_t)best_x;
    }
    out[written] = '\0';
    return written;
}

/* UTF-8 aware: row-argmax for the lead byte, then consume
 * `utf8_lead_len(lead) - 1` continuation bytes from the following
 * rows. If the required continuations aren't present in the top
 * candidates, fall back to a single-byte emit so ASCII still round-
 * trips and garbled cells don't stall the decoder. */
uint32_t grid_decode_text_utf8(const SpatialGrid* g, char* out, uint32_t max_out) {
    if (!g || !out || max_out == 0) return 0;

    uint32_t written = 0;
    uint32_t y = 0;

    while (y < GRID_SIZE && written + 4 < max_out) {
        uint8_t  cands[4];
        uint16_t scores[4];
        row_top_n(g, y, cands, scores, 4);

        if (scores[0] == 0) break;  /* empty row = clause end */

        uint8_t lead = cands[0];
        int len = utf8_lead_len(lead);

        if (len == 1) {
            out[written++] = (char)lead;
            y++;
            continue;
        }
        if (len == 0) {
            /* stray continuation or invalid lead — keep legacy behavior:
             * emit the raw byte and advance. */
            out[written++] = (char)lead;
            y++;
            continue;
        }

        /* multi-byte: look for continuation bytes on the next rows */
        uint8_t seq[4] = { lead, 0, 0, 0 };
        int ok = 1;
        for (int k = 1; k < len; k++) {
            if (y + (uint32_t)k >= GRID_SIZE) { ok = 0; break; }
            uint8_t  nb[4];
            uint16_t ns[4];
            row_top_n(g, y + (uint32_t)k, nb, ns, 4);
            int found = 0;
            for (int c = 0; c < 4; c++) {
                if (ns[c] == 0) break;
                if (utf8_is_cont(nb[c])) { seq[k] = nb[c]; found = 1; break; }
            }
            if (!found) { ok = 0; break; }
        }

        if (ok && written + (uint32_t)len < max_out) {
            for (int k = 0; k < len; k++) out[written++] = (char)seq[k];
            y += (uint32_t)len;
        } else {
            /* Validation failed: fall back to single-byte emit so ASCII
             * still round-trips. */
            out[written++] = (char)lead;
            y++;
        }
    }

    out[written] = '\0';
    return written;
}

/* ── Full-clause generation ────────────────────────────── */

/* ── Topic-aware next-frame lookup ──
 *
 * For a matched keyframe carrying a non-zero topic_hash, the "next"
 * frame is the same-topic keyframe whose seq_in_topic is the
 * smallest value strictly greater than the matched KF's seq. When
 * the match has no topic_hash (label-less input) or there's nothing
 * ahead of it in the topic, fall back to id+1 — which preserves the
 * original generation behavior on legacy data. */
static uint32_t find_next_in_topic(const SpatialAI* ai, uint32_t matched_id) {
    if (matched_id >= ai->kf_count) return matched_id;
    uint32_t topic = ai->keyframes[matched_id].topic_hash;
    uint32_t seq   = ai->keyframes[matched_id].seq_in_topic;

    if (topic == 0) {
        /* no topic assigned: legacy sequential fallback */
        return (matched_id + 1 < ai->kf_count) ? matched_id + 1 : matched_id;
    }

    uint32_t best_next = UINT32_MAX;
    uint32_t best_diff = UINT32_MAX;
    for (uint32_t i = 0; i < ai->kf_count; i++) {
        if (ai->keyframes[i].topic_hash != topic) continue;
        if (ai->keyframes[i].seq_in_topic <= seq) continue;
        uint32_t diff = ai->keyframes[i].seq_in_topic - seq;
        if (diff < best_diff) { best_diff = diff; best_next = i; }
    }
    if (best_next == UINT32_MAX) {
        return (matched_id + 1 < ai->kf_count) ? matched_id + 1 : matched_id;
    }
    return best_next;
}

uint32_t ai_generate_next(SpatialAI* ai, const char* input_text,
                          char* out, uint32_t max_out,
                          float* out_match_similarity) {
    if (!ai || !input_text || !out || max_out == 0 || ai->kf_count == 0) {
        if (out && max_out > 0) out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 1. Encode input through full pipeline */
    SpatialGrid* in_grid = grid_create();
    layers_encode_clause(input_text, NULL, in_grid);
    update_rgb_directional(in_grid);
    apply_ema_to_grid(ai, in_grid);

    /* 2. Unified match in GENERATE mode (bg_score precision stage).
     *    With the B channel now POS-seeded (spec v2 Mod D), MATCH_GENERATE
     *    favors candidates whose B × G pattern matches the query, which
     *    is a better proxy for "what should come next" than a pure
     *    cosine. The engine's bucket index is passed through so
     *    large-corpus retrieval stays fast. */
    MatchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.bucket_idx = &ai->bucket_idx;
    MatchResult r = spatial_match(ai, in_grid, MATCH_GENERATE, &ctx);
    grid_destroy(in_grid);

    if (out_match_similarity) *out_match_similarity = r.best_score;
    if (r.best_id >= ai->kf_count) {
        out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 3. Next frame: topic-aware if the matched KF has a topic tag,
     *    otherwise sequential (legacy). */
    uint32_t target_id = find_next_in_topic(ai, r.best_id);

    /* 4. Decode target frame's grid → text (UTF-8 aware). */
    return grid_decode_text_utf8(&ai->keyframes[target_id].grid, out, max_out);
}
