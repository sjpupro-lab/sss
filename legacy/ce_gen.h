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

/* Canvas-routed image generation. The caller has already picked the
 * canvas_id whose IMAGE entries should drive retrieval (typically by
 * tokenising the prompt with morpheme_tokenize_clause, voting across
 * CE_TYPE_TEXT entries via ce_search_by_type, and taking the winner's
 * canvas_id — see tools/gen_image_ce.c for the worked example).
 *
 * This function:
 *   1. Seeds the initial latent from the first CE_TYPE_IMAGE entry whose
 *      canvas_id matches `routed_canvas`. If none exists, falls back to
 *      ce_generate_image_typed(CE_TYPE_IMAGE, ...).
 *   2. Runs ce_denoise_loop + ce_decode_image to produce a 256x256 RGBA
 *      starting canvas.
 *   3. (When wave_refine_iters > 0) walks `storage` and groups every 4
 *      CE_TYPE_IMAGE entries that share canvas_id/slot/(block_idx>>2)
 *      into a 16x16 atomic patch via ce_decode_image_block_16, ranks
 *      groups by ce_distance against the centre latent cell, and uses
 *      the top-CE_WAVE_TOPK as wave-refine targets. */
void ce_generate_image_canvas_routed(
    CEImage *output,
    const CEStorage *storage,
    uint32_t routed_canvas,
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

#ifdef __cplusplus
}
#endif
#endif /* CE_GEN_H */
