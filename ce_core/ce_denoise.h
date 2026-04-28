/* ce_denoise.h — Step 14 + loss + sampler config.
 *
 * Drives the engine for `total_steps` iterations and records optional loss
 * against a target.
 */
#ifndef CE_DENOISE_H
#define CE_DENOISE_H

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_search.h"
#include "ce_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

enum CESamplerMode {
    CE_SAMPLE_CONFIDENCE = 0,
    CE_SAMPLE_DIVERSITY  = 1,
    CE_SAMPLE_CONTEXT    = 2,
    CE_SAMPLE_ADAPTIVE   = 3
};

typedef struct {
    int    total_steps;     /* >= 1, recommended >= 4 */
    float  cfg_scale;
    int    conv_radius;     /* 1..3 */
    int    attn_topk;       /* 0 disables */
    float  residual_weight; /* 0..1 */
    float  skip_weight;     /* 0..1 */
    int    sampler_mode;    /* enum CESamplerMode */
} CEGenConfig;

typedef struct {
    float channel_loss[4];
    float total_loss;
} CELoss;

/* Default config tuned for 4..50 step inference. */
CEGenConfig ce_gen_config_default(void);

/* Step 14 — full denoising loop. `prompt_cells` may be NULL with
 * prompt_count = 0. `audio_cells` likewise optional. */
struct CEAudioTrack;
void ce_denoise_loop(CELatentGrid *z,
                     const CEStorage *storage,
                     const CEUnit *prompt_cells, int prompt_count,
                     const struct CEAudioTrack *audio,
                     const CEGenConfig *config);

/* Loss / param update. */
void ce_compute_loss(CELoss *loss,
                     const CEUnit *predicted, const CEUnit *target);

void ce_update_params(CEGenConfig *config, const CELoss *loss);

#ifdef __cplusplus
}
#endif
#endif /* CE_DENOISE_H */
