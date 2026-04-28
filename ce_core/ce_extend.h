/* ce_extend.h — Phase 4: sampler / memo / hint / inpaint / upscale / audio. */
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

/* --- Memo (LoRA-equivalent) ----------------------------------------- */

typedef struct {
    char     name[64];
    CEUnit  *adjustments; /* keyframe-indexed offsets; cycled if count < grid */
    uint32_t count;
} CEMemoLayer;

void ce_memo_apply(CELatentGrid *z, const CEMemoLayer *memo);
void ce_memo_remove(CELatentGrid *z, const CEMemoLayer *memo);

/* --- Hint (ControlNet-equivalent) ----------------------------------- */

typedef enum {
    CE_HINT_EDGE  = 0,
    CE_HINT_DEPTH = 1,
    CE_HINT_POSE  = 2,
    CE_HINT_COLOR = 3
} CEHintType;

typedef struct {
    CEHintType type;
    CEUnit     cells[CE_GRID_N];
    float      strength;
} CEHintLayer;

void ce_hint_apply(CELatentGrid *z, const CEHintLayer *hint);

/* --- Inpaint -------------------------------------------------------- */

typedef struct {
    uint8_t mask[CE_GRID_N]; /* 0 = anchor, 1 = regenerate */
} CEInpaintMask;

void ce_inpaint(CELatentGrid *z,
                const CEInpaintMask *mask,
                const CEStorage *storage,
                const CEUnit *prompt, int prompt_count,
                const CEGenConfig *config);

/* --- Upscale -------------------------------------------------------- */

#define CE_HIRES_GRID_W 128
#define CE_HIRES_GRID_H 128
#define CE_HIRES_N (CE_HIRES_GRID_W * CE_HIRES_GRID_H)

typedef struct {
    CEUnit cells[CE_HIRES_N];
    uint16_t width, height;
} CEHiresGrid;

void ce_upscale(CEHiresGrid *hi_res,
                const CELatentGrid *lo_res,
                const CEStorage *storage,
                const CEGenConfig *config);

/* --- Audio ---------------------------------------------------------- */

typedef struct CEAudioTrack {
    CEUnit   *segments;
    uint32_t count;
    float    *energy;
} CEAudioTrack;

void ce_audio_load(CEAudioTrack *track,
                   const uint8_t *audio_data, uint32_t len,
                   int segments_per_step);

void ce_audio_free(CEAudioTrack *track);

float ce_audio_energy_at(const CEAudioTrack *track, int step, int total_steps);

#ifdef __cplusplus
}
#endif
#endif /* CE_EXTEND_H */
