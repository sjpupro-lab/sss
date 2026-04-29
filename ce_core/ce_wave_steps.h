/* ce_wave_steps.h — single source of truth for the 7-step wave cycle.
 *
 * Split out from slig_tick_math.h so that callers which only need the
 * cycle table (e.g. ce_image_wave_refine.c) don't transitively pull in
 * ce_core.h / CEUnit machinery.
 *
 * The cycle (1, 10, 100, 300, 100, 10, 1) is expressed in tick units —
 * ×100 of the float-domain pattern documented in ce_image_wave_refine.h
 * (0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01).
 *
 *   ce_wave_step_value(iter)       = WAVE_STEPS[iter % 7] / 100.0f
 *   slig_wave_refine step at iter  = WAVE_STEPS[iter % 7] × scale
 *
 * uint16_t because the peak (300) overflows uint8_t. Defined in
 * slig_tick_math.c.
 */
#ifndef CE_WAVE_STEPS_H
#define CE_WAVE_STEPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t WAVE_STEPS[7];

#ifdef __cplusplus
}
#endif
#endif /* CE_WAVE_STEPS_H */
