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

#endif /* SPATIAL_IMAGE_H */
