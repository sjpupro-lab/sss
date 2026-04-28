#ifndef SPATIAL_IMAGE_H
#define SPATIAL_IMAGE_H

/* v4 Task F — image modality FEASIBILITY PROTOTYPE.
 *
 * Scope (per 08_TASK_F_IMAGE_PROTOTYPE.md):
 *   - image ↔ SpatialGrid roundtrip, qualitative check
 *   - per-image-level presets (Composition / Contour / Detail) distinct
 *     from the text refine preset
 *   - NOT a diffusion-equivalent generator. NOT production-ready.
 *
 * File format: PPM P6 (binary RGB, 8 bits per channel) to keep the
 * module dependency-free. Spec permits stb_image / stb_image_write
 * but PPM is sufficient for the roundtrip proof-of-concept.
 *
 * The grid used here has shape GRID_SIZE × GRID_SIZE (currently 256).
 * image_to_grid rescales / crops / letterboxes to that fixed size;
 * grid_to_image emits the full GRID_SIZE × GRID_SIZE tile. */

#include <stdint.h>
#include "spatial_grid.h"
#include "spatial_keyframe.h"   /* SpatialAI */
#include "spatial_generate.h"   /* RefineConfig */

/* Load a PPM P6 image from `path` and map it onto a fresh SpatialGrid.
 *   - R/G/B channels ← image R/G/B bytes
 *   - A channel       ← luminance (0.30 R + 0.59 G + 0.11 B)
 * Non-PPM or GRID_SIZE-mismatched images return NULL. Caller owns
 * the returned grid; use grid_destroy to free. */
SpatialGrid* image_to_grid(const char* path);

/* Emit a SpatialGrid's R/G/B planes to `out_path` as PPM P6.
 * Returns 1 on success, 0 on I/O error or NULL inputs.
 * The A channel is not written — grid→image is lossy on luminance. */
int grid_to_image(const SpatialGrid* g, const char* out_path);

/* Feasibility-prototype generator. Encodes the prompt text through
 * the existing refine path with an image-tuned RefineConfig, then
 * emits the resulting grid as a PPM file at `out_path`.
 *
 * cfg == NULL applies refine_config_default_image(). Returns 1 on
 * success, 0 if refine produced no output or the file write failed.
 *
 * This is NOT a diffusion-style generator. It is strictly an
 * iterative refinement over the grid's prior, rendered as an image. */
int ai_generate_image(SpatialAI* ai,
                      const char* prompt_text,
                      const char* out_path,
                      const RefineConfig* cfg);

/* ── SLIG v2 image API ───────────────────────────────────────
 *
 *  v2 keeps ai_generate_image (above) for the legacy refine-based
 *  prototype and adds the integrated pipeline below:
 *
 *    ai_learn_image  — load a PPM, decompose into a CEUnit set
 *                      (multi-direction SVD + ripple), and store it
 *                      as a keyframe with has_image=1 and
 *                      image_cells populated.
 *
 *    ai_generate_image_v2  — text prompt → 256² grid → per-morpheme
 *                            keyframe match → CEUnit pull → grid-
 *                            spectrum weighting → POS-ordered
 *                            sequential render → PPM.
 *
 *  Both return 0 on failure, non-zero on success. ai_learn_image
 *  returns the new keyframe id (or UINT32_MAX). */

uint32_t ai_learn_image(SpatialAI*  ai,
                        const char* image_path,
                        const char* label);

int ai_generate_image_v2(SpatialAI*  ai,
                         const char* prompt_text,
                         const char* out_path);

/* Variant with classifier-free guidance.
 *   guidance_scale = 1.0 → identical to ai_generate_image_v2
 *   guidance_scale > 1.0 → push pixels further from achromatic 128
 *                          (intensifies prompt-driven deviation)
 *   guidance_scale < 1.0 → soften prompt (rarely useful)
 * Same return convention (1 success, 0 failure). */
int ai_generate_image_v2_guided(SpatialAI*  ai,
                                const char* prompt_text,
                                const char* out_path,
                                float       guidance_scale);

/* Generate `num_frames` PPMs into `output_dir/frame_NNN.ppm`. Each
 * frame is a render at event_max_tick = (frame index), so cells with
 * tick-driven evolution (RIPPLE / BEAM / CROSS / CASCADE) build up
 * cumulatively across frames — frame 0 shows the initial state,
 * frame N shows the full evolution after N ticks of accumulation.
 * Returns the number of frames successfully written (0 on failure). */
uint32_t ai_generate_animation(SpatialAI*  ai,
                               const char* prompt_text,
                               const char* output_dir,
                               uint32_t    num_frames);

/* SSS-driven generation through the v3 standalone renderer.
 *
 * Bridges v2.3 codebook patterns (for the matched keyframe) into a
 * SligDecomposed and runs slig_render_adaptive — gives access to v3's
 * progressive callback, depth/sigma/guidance knobs, and Atmos-style
 * 3-canvas Y/Cb/Cr render — without rerunning slig_learn_image's SVD
 * (~233 ms saved per call). guidance_scale follows the same convention
 * as ai_generate_image_v2_guided. Returns 1 on success, 0 on failure
 * (no morpheme match, no has_image keyframe, render error). */
int ai_generate_image_v3(SpatialAI*  ai,
                         const char* prompt_text,
                         const char* out_path,
                         float       guidance_scale);

#endif /* SPATIAL_IMAGE_H */
