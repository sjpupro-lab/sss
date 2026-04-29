/* ce_extend.h — sampler + inpaint helpers used by the masked-train path. */
#ifndef CE_EXTEND_H
#define CE_EXTEND_H

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_search.h"
#include "ce_engine.h"
#include "ce_denoise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- Sampler -------------------------------------------------------- */

void ce_sample(CEUnit *out,
               const CEUnit *candidates, int count,
               enum CESamplerMode mode,
               const CECellContext *ctx);

/* --- Inpaint -------------------------------------------------------- */

typedef struct {
    uint8_t mask[CE_GRID_N]; /* 0 = anchor, 1 = regenerate */
} CEInpaintMask;

void ce_inpaint(CELatentGrid *z,
                const CEInpaintMask *mask,
                const CEStorage *storage,
                const CEUnit *prompt, int prompt_count,
                const CEGenConfig *config);

#ifdef __cplusplus
}
#endif
#endif /* CE_EXTEND_H */
