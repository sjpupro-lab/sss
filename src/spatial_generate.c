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

/* v4 Task A — combined long + short prior scoring.
 * If either table is NULL, that side contributes 0 (collapses to the
 * other side alone). Negative weights are clamped to 0 so a caller
 * that accidentally passes a signed leftover doesn't produce an
 * "anti-prior" bias — the combined score is strictly non-negative,
 * matching agg_score_byte's own contract. */
double agg_score_byte_combined(const AggTables* agg_long,
                               const AggTables* agg_short,
                               double w_long, double w_short,
                               uint32_t y, uint8_t v,
                               double in_R, double in_G, double in_B) {
    if (w_long  < 0.0) w_long  = 0.0;
    if (w_short < 0.0) w_short = 0.0;

    double s_long  = (agg_long  && w_long  > 0.0)
                   ? agg_score_byte(agg_long,  y, v, in_R, in_G, in_B)
                   : 0.0;
    double s_short = (agg_short && w_short > 0.0)
                   ? agg_score_byte(agg_short, y, v, in_R, in_G, in_B)
                   : 0.0;

    return w_long * s_long + w_short * s_short;
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

/* ══════════════════════════════════════════════════════════════════
 * v4 Task D — hierarchical draft refinement (experimental)
 * ══════════════════════════════════════════════════════════════════
 *
 * D1 scaffolding lands here: RefineConfig presets and an ai_generate_
 * refine stub that transparently routes to ai_generate_next. The real
 * DraftField init + per-level scoring loop arrives in D2/D3, guarded
 * by the same assertions that lock anchor immutability.
 *
 * Baseline invariance: nothing in this block touches ai->keyframes /
 * ai->deltas / ai->ema_* / the RGB layer encoder. All state is
 * thread-local to the single ai_generate_refine call. */

RefineConfig refine_config_default_text(void) {
    RefineConfig c;
    memset(&c, 0, sizeof c);
    /* Spec §D.4 — coarse → fine. B-dominant at L0 (broad co-occurrence),
     * G-dominant at L1 (mid), R-dominant at L2 (tight). */
    c.ch_weights[0][0] = 0.1f; c.ch_weights[0][1] = 0.3f; c.ch_weights[0][2] = 1.0f;
    c.ch_weights[1][0] = 0.3f; c.ch_weights[1][1] = 1.0f; c.ch_weights[1][2] = 0.3f;
    c.ch_weights[2][0] = 1.0f; c.ch_weights[2][1] = 0.3f; c.ch_weights[2][2] = 0.1f;
    c.topk[0] = 4;  c.topk[1] = 8;  c.topk[2] = 16;
    c.promote_threshold[0] = 0.55f;
    c.promote_threshold[1] = 0.65f;
    c.promote_threshold[2] = 0.75f;
    c.max_iter[0]      = 12; c.max_iter[1]      = 20; c.max_iter[2]      = 30;
    c.converge_rate[0] = 0.02f;
    c.converge_rate[1] = 0.02f;
    c.converge_rate[2] = 0.01f;
    c.neighbor_radius[0] = 16;
    c.neighbor_radius[1] = 8;
    c.neighbor_radius[2] = 2;
    c.temperature = 0.0f;
    c.use_context_pool   = 1;
    c.allow_prior_anchors = 0;
    return c;
}

/* Spec §D.5 — anchor/candidate initialization.
 *
 * Refine operates on the 2D grid surface: a cell at (y, x) represents
 * "byte value x appearing at clause position y". The input-encoded
 * grid is sparse — at most one (y, x) pair per input byte carries
 * positive A, so every anchor cell in a row shares the same byte
 * value (the row's argmax-x). Candidate cells are everywhere a
 * long-term prior exists without direct input coverage. */
void draft_field_init(DraftField* df,
                      const SpatialGrid* input_grid,
                      const AggTables* agg,
                      const RefineConfig* cfg) {
    if (!df) return;
    memset(df, 0, sizeof *df);

    /* Row-level argmax from the input grid. A row either has any
     * activity (anchor row) or none. */
    uint8_t row_byte[GRID_SIZE];
    int     row_has[GRID_SIZE];
    memset(row_has, 0, sizeof row_has);

    if (input_grid) {
        for (uint32_t y = 0; y < GRID_SIZE; y++) {
            uint16_t best_a = 0;
            uint32_t best_x = 0;
            for (uint32_t x = 0; x < GRID_SIZE; x++) {
                uint16_t a = input_grid->A[y * GRID_SIZE + x];
                if (a > best_a) { best_a = a; best_x = x; }
            }
            if (best_a > 0) {
                row_has[y]  = 1;
                row_byte[y] = (uint8_t)best_x;
            }
        }

        /* Every input-positive cell becomes an anchor. The cell's value
         * is the row's argmax byte — consistent across all anchor cells
         * in a row, which keeps the neighborhood-signature computation
         * (D.6) seeing a stable target byte when it averages over
         * anchors in a row's vicinity. */
        for (uint32_t y = 0; y < GRID_SIZE; y++) {
            if (!row_has[y]) continue;
            for (uint32_t x = 0; x < GRID_SIZE; x++) {
                uint32_t i = y * GRID_SIZE + x;
                if (input_grid->A[i] > 0) {
                    df->cells[i].status     = CELL_ANCHOR;
                    df->cells[i].value      = row_byte[y];
                    df->cells[i].confidence = 1.0f;
                    df->n_anchor++;
                }
            }
        }
    }

    if (!agg) return;

    /* Candidates cover the prior-support mask, minus anchor cells.
     * We deliberately do NOT populate cand_values here — per spec
     * §D.5, copying a row's argmax byte into every high-A cell as the
     * initial value would collapse learned diversity. D3's per-level
     * scoring loop is responsible for the actual Top-K build. */
    const int allow_prior = cfg && cfg->allow_prior_anchors;

    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row_total = agg->row_total_A[y];
        double row_avg   = row_total / (double)GRID_SIZE;  /* per-cell mean */
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (df->cells[i].status == CELL_ANCHOR) continue;
            double a = agg->A_sum[i];
            if (a <= 0.0) continue;

            if (allow_prior && row_avg > 0.0 && a > row_avg * 4.0) {
                /* High-A_sum seed, RESOLVED only — never ANCHOR. */
                df->cells[i].status     = CELL_RESOLVED;
                df->cells[i].value      = (uint8_t)x;
                df->cells[i].confidence = 0.5f;
                df->n_resolved++;
            } else {
                df->cells[i].status = CELL_CANDIDATE;
                df->n_candidate++;
            }
        }
    }
}

RefineConfig refine_config_default_image(void) {
    /* v4 feasibility prototype — image presets are tuned larger than
     * text since image grids carry denser local structure. These are
     * placeholders the Task F prototype will refine. */
    RefineConfig c = refine_config_default_text();
    c.neighbor_radius[0] = 32;
    c.neighbor_radius[1] = 16;
    c.neighbor_radius[2] = 4;
    c.promote_threshold[0] = 0.50f;
    c.promote_threshold[1] = 0.60f;
    c.promote_threshold[2] = 0.70f;
    c.use_context_pool     = 0;  /* image task isn't session-driven by default */
    return c;
}

/* ── D3: per-level refine loop ──────────────────────────
 *
 * Simplified (text-path) scoring:
 *   For each candidate (y, x), use the long-term prior's per-cell
 *   RGB means as the neighbor signature (radius is an upper-bound
 *   cue, not a hard window — spec §D.1 explicitly allows level-by-
 *   level policy swaps without changing the candidate set). When
 *   short-term prior is available (cfg.use_context_pool + context
 *   pool populated), blend with SPAI_W_LONG/SPAI_W_SHORT defaults.
 *
 * Confidence is the row-normalized score: score divided by row's
 * total activity. Promotion fires when confidence >= threshold[L].
 * Per-level loop stops when either max_iter[L] is reached or the
 * per-iteration promotion rate falls below converge_rate[L].
 *
 * ANCHOR cells are completely untouched — the loop only considers
 * cells with status == CELL_CANDIDATE. This is the enforcement
 * point for spec §D.7's anchor-immutability rule.  */
static double score_candidate_cell(const AggTables* agg_long,
                                   const AggTables* agg_short,
                                   uint32_t y, uint32_t x,
                                   float wR, float wG, float wB) {
    uint32_t i = y * GRID_SIZE + x;
    double R = agg_long->R_mean[i];
    double G = agg_long->G_mean[i];
    double B = agg_long->B_mean[i];
    double s_long  = agg_score_byte(agg_long,  y, (uint8_t)x, R, G, B);
    double s_short = 0.0;
    if (agg_short) {
        s_short = agg_score_byte(agg_short, y, (uint8_t)x, R, G, B);
    }
    double w_avg = ((double)wR + (double)wG + (double)wB) / 3.0;
    if (w_avg <= 0.0) w_avg = 1.0;
    if (agg_short) {
        return (SPAI_W_LONG_DEFAULT * s_long + SPAI_W_SHORT_DEFAULT * s_short) * w_avg;
    }
    return s_long * w_avg;
}

static uint32_t refine_run_levels(DraftField* df,
                                  const AggTables* agg_long,
                                  const AggTables* agg_short,
                                  const RefineConfig* cfg,
                                  RefineTrace* trace) {
    uint32_t iters_total = 0;
    uint32_t last_level  = 0;
    for (int L = 0; L < 3; L++) {
        last_level = (uint32_t)L;
        float wR = cfg->ch_weights[L][0];
        float wG = cfg->ch_weights[L][1];
        float wB = cfg->ch_weights[L][2];
        float thr = cfg->promote_threshold[L];
        float conv = cfg->converge_rate[L];
        uint32_t max_iter = cfg->max_iter[L];

        for (uint32_t iter = 0; iter < max_iter; iter++) {
            uint32_t n_before = df->n_candidate;
            if (n_before == 0) break;
            uint32_t promoted = 0;

            for (uint32_t y = 0; y < GRID_SIZE; y++) {
                double row_total = agg_long->row_total_A[y];
                if (row_total <= 0.0) continue;
                for (uint32_t x = 0; x < GRID_SIZE; x++) {
                    uint32_t i = y * GRID_SIZE + x;
                    CellState* cs = &df->cells[i];
                    if (cs->status != CELL_CANDIDATE) continue;  /* skips ANCHOR */
                    double s = score_candidate_cell(agg_long, agg_short,
                                                    y, x, wR, wG, wB);
                    double conf = s / (row_total + 1e-9);
                    if (conf > 1.0) conf = 1.0;
                    cs->confidence = (float)conf;
                    cs->value = (uint8_t)x;
                    if (conf >= (double)thr) {
                        cs->status = CELL_RESOLVED;
                        df->n_candidate--;
                        df->n_resolved++;
                        promoted++;
                    }
                }
            }
            iters_total++;
            df->n_promoted_this_iter = promoted;

            float rate = (n_before > 0)
                ? (float)promoted / (float)n_before
                : 0.0f;

            if (trace) {
                if (trace->n < REFINE_TRACE_MAX) {
                    RefineTraceEntry* e = &trace->entries[trace->n++];
                    e->level         = (uint8_t)L;
                    e->iter          = (uint16_t)(iter + 1);
                    e->n_candidates  = n_before;
                    e->n_promoted    = promoted;
                    e->promote_rate  = rate;
                } else {
                    trace->dropped++;
                }
            }

            if (promoted == 0) break;
            if (rate < conv) break;
        }
    }
    if (trace) trace->last_level = (uint8_t)last_level;
    return iters_total;
}

/* After refinement, pick one byte per row:
 *   1. ANCHOR wins (input-derived).
 *   2. Otherwise the RESOLVED cell in the row with highest confidence.
 *   3. Otherwise no output — truncate at the first unresolved row. */
static uint32_t refine_decode_rows(const DraftField* df,
                                   char* out, uint32_t max_out) {
    uint32_t written = 0;
    for (uint32_t y = 0; y < GRID_SIZE && written < max_out; y++) {
        uint8_t pick = 0;
        float   pick_conf = -1.0f;
        int     kind = CELL_EMPTY;

        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            const CellState* cs = &df->cells[y * GRID_SIZE + x];
            if (cs->status == CELL_ANCHOR) {
                if (kind != CELL_ANCHOR || cs->confidence > pick_conf) {
                    pick = cs->value;
                    pick_conf = cs->confidence;
                    kind = CELL_ANCHOR;
                }
            } else if (cs->status == CELL_RESOLVED && kind != CELL_ANCHOR) {
                if (kind != CELL_RESOLVED || cs->confidence > pick_conf) {
                    pick = cs->value;
                    pick_conf = cs->confidence;
                    kind = CELL_RESOLVED;
                }
            }
        }
        if (kind == CELL_EMPTY) break;  /* row has no decision → stop output */
        out[written++] = (char)pick;
    }
    if (written < max_out) out[written] = '\0';
    return written;
}

/* Minimum anchor count below which refinement has nothing to lock
 * against. At that point we delegate to the baseline and log the
 * fallback per spec §D.7. Empirically "a few dozen anchor cells per
 * clause" is plenty, so 16 is deliberately conservative. */
#define REFINE_ANCHOR_MIN 16

uint32_t ai_generate_refine(SpatialAI* ai,
                            const char* input_text,
                            char* out, uint32_t max_out,
                            const RefineConfig* cfg,
                            float* out_confidence,
                            uint32_t* out_iterations) {
    return ai_generate_refine_traced(ai, input_text, out, max_out,
                                     cfg, out_confidence, out_iterations,
                                     NULL);
}

uint32_t ai_generate_refine_traced(SpatialAI* ai,
                                   const char* input_text,
                                   char* out, uint32_t max_out,
                                   const RefineConfig* cfg,
                                   float* out_confidence,
                                   uint32_t* out_iterations,
                                   RefineTrace* out_trace) {
    if (out_confidence) *out_confidence = 0.0f;
    if (out_iterations) *out_iterations = 0;
    if (out_trace) memset(out_trace, 0, sizeof *out_trace);
    if (!ai || !input_text || !out || max_out == 0) {
        if (out && max_out > 0) out[0] = '\0';
        return 0;
    }

    RefineConfig local = cfg ? *cfg : refine_config_default_text();

    /* 1. Encode input (same front-end as ai_generate_next, but on a
     *    scratch grid — we never mutate ai state from here). */
    SpatialGrid* in_grid = grid_create();
    if (!in_grid) { out[0] = '\0'; return 0; }
    layers_encode_clause(input_text, NULL, in_grid);
    update_rgb_directional(in_grid);
    apply_ema_to_grid(ai, in_grid);

    /* 2. Build priors. Short-term is optional; we only consult the
     *    pool when the config opts in AND the pool has slots. */
    AggTables* agg_long  = agg_build(ai);
    AggTables* agg_short = NULL;
    if (local.use_context_pool && ai->context_pool &&
        pool_total_slots(ai->context_pool) > 0) {
        agg_short = agg_build_from_pool(ai->context_pool);
    }

    /* 3. Initialize the draft field. */
    DraftField* df = (DraftField*)calloc(1, sizeof(DraftField));
    if (!df) {
        grid_destroy(in_grid);
        if (agg_long)  agg_destroy(agg_long);
        if (agg_short) agg_destroy(agg_short);
        out[0] = '\0';
        return 0;
    }
    draft_field_init(df, in_grid, agg_long, &local);

    /* 4. Low-anchor-density fallback. Even if the input encodes into
     *    a lot of cells, deliberately-short prompts fall through to
     *    the baseline so refine can't invent confidence out of thin
     *    air. Spec §D.7 tags this path "refine_fallback". */
    if (df->n_anchor < REFINE_ANCHOR_MIN) {
        uint32_t n_anchor = df->n_anchor;
        free(df);
        grid_destroy(in_grid);
        if (agg_long)  agg_destroy(agg_long);
        if (agg_short) agg_destroy(agg_short);
        float sim = 0.0f;
        uint32_t n = ai_generate_next(ai, input_text, out, max_out, &sim);
        if (out_confidence) *out_confidence = sim;
        if (out_iterations) *out_iterations = 0;
        if (out_trace)      out_trace->fallback_fired = 1;
        fprintf(stderr, "refine_fallback: anchor density=%u < %d\n",
                n_anchor, REFINE_ANCHOR_MIN);
        return n;
    }

    /* 5. Per-level coarse → fine loop. */
    uint32_t iters = refine_run_levels(df, agg_long, agg_short, &local, out_trace);

    /* 6. Decode. If refine produced no RESOLVED rows (e.g. thresholds
     *    were too strict for this prompt), fall back to the baseline
     *    so we still hand the caller a meaningful reply. */
    uint32_t written = refine_decode_rows(df, out, max_out);

    /* Confidence summary: average of RESOLVED + ANCHOR confidences. */
    double conf_sum = 0.0;
    uint32_t conf_n = 0;
    for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        uint8_t st = df->cells[i].status;
        if (st == CELL_RESOLVED || st == CELL_ANCHOR) {
            conf_sum += df->cells[i].confidence;
            conf_n++;
        }
    }

    free(df);
    grid_destroy(in_grid);
    if (agg_long)  agg_destroy(agg_long);
    if (agg_short) agg_destroy(agg_short);

    if (written == 0) {
        float sim = 0.0f;
        uint32_t n = ai_generate_next(ai, input_text, out, max_out, &sim);
        if (out_confidence) *out_confidence = sim;
        if (out_iterations) *out_iterations = iters;  /* we did iterate, just didn't emit */
        if (out_trace) {
            out_trace->fallback_fired = 1;
            out_trace->total_iters    = iters;
            out_trace->final_confidence = sim;
        }
        fprintf(stderr, "refine_fallback: no resolved rows after %u iterations\n", iters);
        return n;
    }

    float final_conf = (conf_n > 0) ? (float)(conf_sum / conf_n) : 0.0f;
    if (out_confidence) *out_confidence = final_conf;
    if (out_iterations) *out_iterations = iters;
    if (out_trace) {
        out_trace->total_iters      = iters;
        out_trace->final_confidence = final_conf;
    }
    return written;
}
