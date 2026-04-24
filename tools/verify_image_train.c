/* Sanity-check that ai_save -> ai_load roundtrips the EMA store
 * trained from images, and report active-cell coverage. */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_io.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.spai>\n", argv[0]);
        return 2;
    }
    SpaiStatus st = SPAI_OK;
    SpatialAI* ai = ai_load(argv[1], &st);
    if (!ai || st != SPAI_OK) {
        fprintf(stderr, "[verify] ai_load failed: %s (status=%d)\n", argv[1], (int)st);
        if (ai) spatial_ai_destroy(ai);
        return 1;
    }

    uint64_t evidence = 0;
    double total_count = 0.0;
    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
    float max_count = 0.0f;
    for (uint32_t i = 0; i < GRID_TOTAL; i++) {
        float c = ai->ema_count[i];
        if (c > 0.0f) {
            evidence++;
            total_count += c;
            r_sum += ai->ema_R[i];
            g_sum += ai->ema_G[i];
            b_sum += ai->ema_B[i];
            if (c > max_count) max_count = c;
        }
    }
    printf("[verify] model           : %s\n", argv[1]);
    printf("[verify] kf_count        : %u\n", ai->kf_count);
    printf("[verify] EMA cells w/ evi: %llu / %llu (%.1f%%)\n",
           (unsigned long long)evidence, (unsigned long long)GRID_TOTAL,
           100.0 * (double)evidence / (double)GRID_TOTAL);
    if (evidence) {
        printf("[verify] EMA mean count  : %.2f  (max=%.0f)\n",
               total_count / (double)evidence, (double)max_count);
        printf("[verify] EMA mean RGB    : (%.1f, %.1f, %.1f)\n",
               r_sum / (double)evidence,
               g_sum / (double)evidence,
               b_sum / (double)evidence);
    }
    spatial_ai_destroy(ai);
    return 0;
}
