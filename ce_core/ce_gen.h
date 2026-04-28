/* ce_gen.h — top-level generation API. */
#ifndef CE_GEN_H
#define CE_GEN_H

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_search.h"
#include "ce_engine.h"
#include "ce_denoise.h"
#include "ce_decode.h"
#include "ce_extend.h"
#include "ce_type.h"

#ifdef __cplusplus
extern "C" {
#endif

void ce_generate_image(
    CEImage *output,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config,
    const CEMemoLayer *memo,
    const CEHintLayer *hint,
    const CEAudioTrack *audio);

/* HQ image generation: filters retrieval to a single modality and (when
 * `wave_refine_iters > 0`) runs ce_image_wave_refine on the decoded canvas
 * using the top-k retrieved blocks as wave targets. The `type` argument
 * is normally CE_TYPE_IMAGE — but generation paths that want to retrieve
 * across modalities can pass CE_TYPE_TEXT/AUDIO.
 *
 * Pass `config = NULL` to use ce_gen_config_hq() (50-step preset). */
void ce_generate_image_typed(
    CEImage *output,
    const CEStorage *storage,
    CEType type,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config,
    uint32_t wave_refine_iters);

/* Label-routed image generation. WIP / superseded — uses byte-level
 * ce_feed for prompt encoding which puts text CEUnits in a different
 * cell space than image CEUnits, so the cross-modal alignment relies
 * on lucky byte overlap between prompt tokens and stored labels.
 *
 * The architecturally correct path (per project owner's direction)
 * routes through SSS's layers_encode_clause (256x256 SpatialGrid) with
 * 16x16 ce_feed_image_16 blocks so text and image cells share the
 * same CE space. That refactor is the next iteration; this function
 * is kept here as a reference of the byte-level baseline. */
void ce_generate_image_label_routed(
    CEImage *output,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config,
    uint32_t wave_refine_iters);

void ce_generate_text(
    uint8_t *output, uint32_t *output_len,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config);

void ce_generate_inpaint(
    CEImage *output,
    const CEImage *original,
    const CEInpaintMask *mask,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config);

void ce_generate_upscale(
    CEHiresGrid *output_hires,
    const CELatentGrid *input_lores,
    const CEStorage *storage,
    const CEGenConfig *config);

#ifdef __cplusplus
}
#endif
#endif /* CE_GEN_H */
