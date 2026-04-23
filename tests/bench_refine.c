/* v4 Task D — bench_refine
 *
 * For a handful of prompts against a small trained model, prints:
 *   - refine output + confidence + iterations + trace
 *   - baseline output for side-by-side comparison
 *   - retrieval-id consistency (did refine land on the same kf?)
 *
 * Report-only. Numerical targets (diversity, Top-1 delta, etc.) are
 * reference values per 09_CHECKLIST_AND_ORDER.md, not merge gates. */

#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* KFS[] = {
    "the quick brown fox jumps over the lazy dog today clearly happily",
    "pack my box with five dozen liquor jugs by the porch now please",
    "data teams iterate on the morning backlog carefully always warmly",
    "systems retire at midnight after the weekly cleanup pass runs",
    "the kitchen smelled of cinnamon and warm bread each afternoon truly",
    "visitors paused at the iron gate and admired every single bloom brightly",
    "oak trees shed their leaves along the quiet garden path slowly gently",
};

static const char* PROMPTS[] = {
    "the quick brown fox jumps over the lazy dog",
    "data teams iterate on the morning backlog",
    "the kitchen smelled of cinnamon and warm bread",
    "oak trees shed their leaves along the quiet garden path",
    "hi",  /* intentionally short — expect fallback */
};

static void print_trace_short(const RefineTrace* t) {
    printf("    trace: total_iters=%u last_level=%u dropped=%u fallback=%s\n",
           t->total_iters, t->last_level, t->dropped,
           t->fallback_fired ? "yes" : "no");
    for (uint32_t i = 0; i < t->n; i++) {
        const RefineTraceEntry* e = &t->entries[i];
        printf("      L%u iter%02u  n_cand=%-5u promoted=%-5u rate=%5.2f%%\n",
               e->level, e->iter, e->n_candidates,
               e->n_promoted, e->promote_rate * 100.0f);
    }
}

int main(void) {
    morpheme_init();
    SpatialAI* ai = spatial_ai_create();
    for (uint32_t i = 0; i < sizeof KFS / sizeof KFS[0]; i++) {
        ai_force_keyframe(ai, KFS[i], "mix");
    }

    RefineConfig cfg = refine_config_default_text();

    const uint32_t N = sizeof PROMPTS / sizeof PROMPTS[0];
    printf("=== bench_refine — %u prompts ===\n\n", N);

    uint32_t same_retrieval = 0;
    uint32_t total = 0;

    for (uint32_t p = 0; p < N; p++) {
        const char* pr = PROMPTS[p];
        printf("[%u] prompt: %s\n", p, pr);

        /* Baseline */
        char base_out[256] = {0};
        float base_sim = 0.0f;
        uint32_t base_n = ai_generate_next(ai, pr, base_out,
                                           sizeof base_out - 1, &base_sim);
        base_out[base_n < sizeof base_out ? base_n : sizeof base_out - 1] = '\0';
        uint32_t base_id = ai_predict(ai, pr, NULL);
        printf("  baseline: kf=%-3u sim=%.3f  >>  %.80s\n",
               base_id, base_sim, base_out);

        /* Refine with trace */
        char ref_out[256] = {0};
        float ref_conf = 0.0f;
        uint32_t ref_iters = 0;
        RefineTrace tr;
        uint32_t ref_n = ai_generate_refine_traced(ai, pr, ref_out,
                                                   sizeof ref_out - 1,
                                                   &cfg, &ref_conf,
                                                   &ref_iters, &tr);
        ref_out[ref_n < sizeof ref_out ? ref_n : sizeof ref_out - 1] = '\0';
        printf("  refine:   conf=%.3f iters=%u  >>  %.80s\n",
               ref_conf, ref_iters, ref_out);
        print_trace_short(&tr);

        /* Retrieval consistency via ai_predict on the same prompt:
         * refine path does not change ai_predict output (it doesn't
         * mutate ai state), but we record for the final summary. */
        uint32_t ref_id = ai_predict(ai, pr, NULL);
        if (ref_id == base_id) same_retrieval++;
        total++;

        printf("\n");
    }

    printf("=== summary ===\n");
    printf("  retrieval consistency: %u / %u prompts matched baseline kf id\n",
           same_retrieval, total);
    printf("  (refine does not mutate ai state — mismatch here would be a bug)\n");

    spatial_ai_destroy(ai);
    return 0;
}
