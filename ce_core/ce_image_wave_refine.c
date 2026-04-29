/* ce_image_wave_refine.c — see ce_image_wave_refine.h. */
#include "ce_image_wave_refine.h"
#include "slig_tick_math.h"   /* WAVE_STEPS — single source of truth */

#include <stdlib.h>
#include <string.h>

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float absf2(float v) {
    return v < 0.0f ? -v : v;
}

float ce_wave_step_value(uint32_t iter) {
    /* WAVE_STEPS lives in tick units (×100 of the float-domain values
     * documented in the header: 0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01). */
    return (float)WAVE_STEPS[iter % 7] / 100.0f;
}

void ce_wave_build_target_from_top3(CEPixelF *out_target,
                                    CEImageLink64 *const top3[CE_WAVE_TOPK]) {
    if (!out_target) return;

    float total = 0.0f;
    for (int k = 0; k < CE_WAVE_TOPK; ++k) {
        if (top3[k]) total += top3[k]->score;
    }
    /* All slots NULL or zero score: emit a zero target rather than dividing
     * by zero. Caller can detect this by inspecting (or ignoring) out_target. */
    if (total <= 0.0f) {
        memset(out_target, 0, sizeof(CEPixelF) * CE_WAVE_IMG_W * CE_WAVE_IMG_H);
        return;
    }

    for (uint32_t i = 0; i < (uint32_t)(CE_WAVE_IMG_W * CE_WAVE_IMG_H); ++i) {
        CEPixelF t = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < CE_WAVE_TOPK; ++k) {
            if (!top3[k]) continue;
            float w = top3[k]->score / total;
            t.r += top3[k]->target[i].r * w;
            t.g += top3[k]->target[i].g * w;
            t.b += top3[k]->target[i].b * w;
            t.a += top3[k]->target[i].a * w;
        }
        out_target[i] = t;
    }
}

static float move_toward(float cur, float target, float max_step) {
    float diff = target - cur;
    if (absf2(diff) <= max_step) return target;
    return (diff > 0.0f) ? (cur + max_step) : (cur - max_step);
}

void ce_wave_refine_once(CEImageCanvas *canvas,
                         const CEPixelF *target,
                         float max_step) {
    if (!canvas || !target) return;
    for (uint32_t i = 0; i < (uint32_t)(CE_WAVE_IMG_W * CE_WAVE_IMG_H); ++i) {
        canvas->px[i].r = move_toward(canvas->px[i].r, target[i].r, max_step);
        canvas->px[i].g = move_toward(canvas->px[i].g, target[i].g, max_step);
        canvas->px[i].b = move_toward(canvas->px[i].b, target[i].b, max_step);
        canvas->px[i].a = move_toward(canvas->px[i].a, target[i].a, max_step);
        canvas->px[i].r = clampf(canvas->px[i].r, 0.0f, 255.0f);
        canvas->px[i].g = clampf(canvas->px[i].g, 0.0f, 255.0f);
        canvas->px[i].b = clampf(canvas->px[i].b, 0.0f, 255.0f);
        canvas->px[i].a = clampf(canvas->px[i].a, 0.0f, 255.0f);
    }
}

void ce_image_wave_refine(CEImageCanvas *canvas,
                          CEImageLink64 *const top3[CE_WAVE_TOPK],
                          uint32_t iterations) {
    if (!canvas) return;

    CEPixelF *target = (CEPixelF *)malloc(sizeof(CEPixelF)
                                          * CE_WAVE_IMG_W * CE_WAVE_IMG_H);
    if (!target) return;
    ce_wave_build_target_from_top3(target, top3);

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        float step = ce_wave_step_value(iter);
        ce_wave_refine_once(canvas, target, step);
    }

    free(target);
}
