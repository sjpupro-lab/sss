#ifndef SPATIAL_GENERATE_H
#define SPATIAL_GENERATE_H

#include "spatial_grid.h"
#include "spatial_keyframe.h"

/*
 * Generation module — implements "reading back" from the learned
 * spatial pattern per SPEC.md §4, §5, §9.
 *
 * Principle:
 *   Training ran update_rgb_directional on every stored keyframe, so all
 *   four channels carry learned structure:
 *     A           byte frequency / activation strength
 *     R  diagonal → semantic / morpheme class
 *     G  vertical → word-substitution class
 *     B  horizontal → clause-order class
 *
 *   Generation reads these learned values to produce candidates.
 *   Scoring combines ALL four channels (SPEC §5.1, §9.4):
 *
 *     byte_score(y, v) = A_sum[y,v]                    (activation prior)
 *                        × R_similarity(y, v)         (diagonal semantics)
 *                        × G_similarity(y, v)         (vertical substitution)
 *                        × B_similarity(y, v)         (horizontal ordering)
 */

/* ── Aggregated channel tables ───────────────────────────
 *   A_sum[y,v]   = sum over keyframes of kf.A[y,v]
 *   R_mean[y,v]  = A-weighted mean of kf.R[y,v]
 *   G_mean[y,v]  = A-weighted mean of kf.G[y,v]
 *   B_mean[y,v]  = A-weighted mean of kf.B[y,v]
 */
typedef struct {
    double A_sum [GRID_SIZE * GRID_SIZE];
    double R_mean[GRID_SIZE * GRID_SIZE];
    double G_mean[GRID_SIZE * GRID_SIZE];
    double B_mean[GRID_SIZE * GRID_SIZE];
    /* Per-row activation totals, for normalization */
    double row_total_A[GRID_SIZE];
} AggTables;

/* Allocate + build aggregated tables from all stored keyframes */
AggTables* agg_build(const SpatialAI* ai);
/* Build aggregated tables from all slots in a canvas pool. Tile-local
 * (y, x) coordinates are used so candidates scored afterwards index
 * naturally into the 256×256 space. */
struct SpatialCanvasPool_;
AggTables* agg_build_from_pool(const struct SpatialCanvasPool_* pool);
void       agg_destroy(AggTables* t);

/* ── Context signature from an input grid ────────────────
 * Summarizes the input's RGBA pattern for comparison against
 * candidate (y, v) positions.
 *
 *   R_row[y] = A-weighted mean of input.R on row y
 *   G_row[y] = A-weighted mean of input.G on row y
 *   B_row[y] = A-weighted mean of input.B on row y
 * Falls back to neighbor rows where row-y has no activity.
 */
typedef struct {
    double R_row[GRID_SIZE];
    double G_row[GRID_SIZE];
    double B_row[GRID_SIZE];
    int    has_activity[GRID_SIZE];
    double R_global;
    double G_global;
    double B_global;
} InputSignature;

void input_signature_compute(InputSignature* sig, const SpatialGrid* input);

/* Get context R/G/B to compare against for position (y).
   Uses row-y if it has activity, otherwise nearest active neighbor,
   finally the global clause mean. */
void input_signature_get(const InputSignature* sig, uint32_t y,
                         double* out_R, double* out_G, double* out_B);

/* ── Byte-candidate scoring ──────────────────────────────
 * Full RGBA product per SPEC §5.1, §9.4:
 *
 *   score(y, v) = A_sum[y,v]
 *               × (1 - |R_mean[y,v] - in_R| / 255)   (diagonal, semantic)
 *               × (1 - |G_mean[y,v] - in_G| / 255)   (vertical, substitution)
 *               × (1 - |B_mean[y,v] - in_B| / 255)   (horizontal, clause order)
 *
 * Returns 0 if the (y, v) cell was never active in training. */
double agg_score_byte(const AggTables* t, uint32_t y, uint8_t v,
                      double in_R, double in_G, double in_B);

/* v4 Task A — weighted long/short-term scoring.
 *
 *   score(y, v) = w_long  * agg_score_byte(agg_long,  ...)
 *               + w_short * agg_score_byte(agg_short, ...)
 *
 * agg_long is typically agg_build(ai) — long-term prior from
 * keyframes. agg_short is typically agg_build_from_pool(ctx) over
 * SpatialAI.context_pool (NULL → long-term-only scoring).
 *
 * Callers: the refinement / REPL biasing paths (Task D / Task E).
 * Not wired into ai_generate_next; the existing baseline generation
 * path is preserved unchanged per v4 Principle 2. Defined here so
 * both tasks can pick up the same helper and same default weights.
 *
 * Recommended defaults: w_long = 0.6, w_short = 0.4. */
#define SPAI_W_LONG_DEFAULT   0.6
#define SPAI_W_SHORT_DEFAULT  0.4

double agg_score_byte_combined(const AggTables* agg_long,
                               const AggTables* agg_short,
                               double w_long, double w_short,
                               uint32_t y, uint8_t v,
                               double in_R, double in_G, double in_B);

/* ── Grid → text decoding ────────────────────────────────
 * For each row y in sequence, take the byte x with the highest A
 * value (argmax across the row) as that position's byte.
 * Stops at the first empty row (all A == 0) or when out is full.
 * Returns bytes written. Byte-level; no UTF-8 awareness. */
uint32_t grid_decode_text(const SpatialGrid* g, char* out, uint32_t max_out);

/* UTF-8-aware variant used by ai_generate_next. Reads the row-argmax
 * lead byte, then validates that the next utf8_lead_len(lead) - 1
 * rows carry continuation bytes (10xxxxxx) before emitting the
 * complete codepoint. Falls back to single-byte emit when validation
 * fails, so ASCII-only output matches grid_decode_text exactly. */
uint32_t grid_decode_text_utf8(const SpatialGrid* g, char* out, uint32_t max_out);

/* ── Full-clause generation ──────────────────────────────
 * SPEC §11.3:  "매칭된 키프레임의 다음 프레임이 곧 응답 텍스트의 패턴이다."
 *
 *   1. encode input_text (3-layer + RGB diffusion)
 *   2. match_engine → best_kf_id
 *   3. next_kf = keyframes[best_kf_id + 1]   (if exists)
 *   4. decode next_kf.grid → out via grid_decode_text
 *
 * Returns bytes written. If no next frame exists, decodes the matched
 * keyframe itself. */
uint32_t ai_generate_next(SpatialAI* ai, const char* input_text,
                          char* out, uint32_t max_out,
                          float* out_match_similarity);

/* ── v4 Task D — hierarchical draft refinement (experimental) ─────
 *
 * Parallel generation path. ai_generate_refine() does NOT replace
 * ai_generate_next(); it runs an independent coarse-to-fine loop
 * over a DraftField of per-cell candidate lists. See
 * 06_TASK_D_DRAFT_REFINEMENT.md.
 *
 * Hard invariants:
 *   - CELL_ANCHOR cells are never overwritten by any path.
 *   - High-A_sum seeding to CELL_ANCHOR requires
 *     RefineConfig.allow_prior_anchors == 1; default is 0.
 *   - Levels differ only in scoring policy (channel weights +
 *     neighborhood radius), not in the candidate set itself. */

#define REFINE_TOPK_MAX 16

typedef enum {
    CELL_EMPTY     = 0,  /* untouched; A_sum ~ 0 */
    CELL_CANDIDATE = 1,  /* on the Top-K scoring list */
    CELL_RESOLVED  = 2,  /* promoted above threshold this run */
    CELL_ANCHOR    = 3   /* prompt-derived; must never be overwritten */
} CellStatus;

typedef struct {
    uint8_t  status;                      /* CellStatus */
    uint8_t  value;                       /* valid for RESOLVED / ANCHOR */
    uint8_t  n_cand;
    uint8_t  cand_values[REFINE_TOPK_MAX];
    float    cand_scores[REFINE_TOPK_MAX];
    float    confidence;                  /* score[0] / sum(scores) */
} CellState;

typedef struct {
    CellState cells[GRID_SIZE * GRID_SIZE];
    uint32_t  n_anchor;
    uint32_t  n_candidate;
    uint32_t  n_resolved;
    uint32_t  n_promoted_this_iter;
} DraftField;

typedef struct {
    float    ch_weights[3][3];      /* [level][R,G,B] */
    uint32_t topk[3];
    float    promote_threshold[3];
    uint32_t max_iter[3];
    float    converge_rate[3];      /* promotion rate below which we advance level */
    uint32_t neighbor_radius[3];    /* large / medium / small */
    float    temperature;           /* 0 = argmax */
    int      use_context_pool;      /* blend short-term prior when available */
    int      allow_prior_anchors;   /* off by default — A_sum is a seed, not an anchor */
} RefineConfig;

RefineConfig refine_config_default_text(void);
RefineConfig refine_config_default_image(void);

/* Generation via the refine path. Guaranteed to be independent of
 * ai_generate_next: it does not mutate ai->keyframes / ai->deltas /
 * ai->ema_*. Writes up to max_out bytes to `out` (NUL-terminated if
 * space permits). On low-anchor-density inputs the implementation is
 * allowed to internally fall back to ai_generate_next and report
 * `"refine_fallback"` via stderr; out_confidence / out_iterations
 * still receive meaningful values (0 iterations when falling back).
 * cfg == NULL applies refine_config_default_text(). */
uint32_t ai_generate_refine(SpatialAI* ai,
                            const char* input_text,
                            char* out, uint32_t max_out,
                            const RefineConfig* cfg,
                            float* out_confidence,
                            uint32_t* out_iterations);

#endif /* SPATIAL_GENERATE_H */
