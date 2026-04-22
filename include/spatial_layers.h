#ifndef SPATIAL_LAYERS_H
#define SPATIAL_LAYERS_H

#include "spatial_grid.h"
#include "spatial_morpheme.h"

/* 3-layer bitmaps before summation */
typedef struct {
    uint16_t base[GRID_TOTAL];     /* base layer: weight +1 */
    uint16_t word[GRID_TOTAL];     /* word layer: weight +5 */
    uint16_t morpheme[GRID_TOTAL]; /* morpheme layer: weight +3 */
} LayerBitmaps;

/* Create layer bitmaps (zeroed) */
LayerBitmaps* layers_create(void);

/* Destroy layer bitmaps */
void layers_destroy(LayerBitmaps* lb);

/* Encode a clause into 3 layers and produce a combined SpatialGrid.
   clause_text: UTF-8 Korean text (one clause)
   out_layers: optional, receives individual layers (can be NULL)
    out_combined: receives the summed grid (A = base*1 + word*5 + morpheme*3)
*/
void layers_encode_clause(const char* clause_text,
                          LayerBitmaps* out_layers,
                          SpatialGrid* out_combined);

#endif /* SPATIAL_LAYERS_H */
