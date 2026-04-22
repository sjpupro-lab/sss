#ifndef SPATIAL_CONTEXT_H
#define SPATIAL_CONTEXT_H

#include "spatial_grid.h"
#include "spatial_keyframe.h"
#include "spatial_match.h"

/* Context frame */
typedef struct {
    uint32_t    frame_id;
    uint32_t    topic_hash;
    char        topic_label[64];
    uint8_t     frame_type;     /* 0=I-frame, 1=P-frame */
    uint32_t    parent_id;
    SpatialGrid* grid;          /* pointer to grid (owned) */
} ContextFrame;

/* LRU frame cache (SPEC-ENGINE Phase E) */
#define CACHE_SIZE 256

typedef struct {
    uint32_t     frame_id;
    SpatialGrid* grid;
    uint32_t     access_order;
} CacheEntry;

typedef struct {
    CacheEntry entries[CACHE_SIZE];
    uint32_t   count;
    uint32_t   clock;
} FrameCache;

/* Context manager: frame array + cache */
typedef struct {
    ContextFrame* frames;
    uint32_t      frame_count;
    uint32_t      frame_capacity;
    FrameCache    cache;
} ContextManager;

/* Create/destroy context manager */
ContextManager* context_create(void);
void            context_destroy(ContextManager* ctx);

/* Add a frame to the context */
uint32_t context_add_frame(ContextManager* ctx,
                           SpatialAI* ai,
                           const char* clause_text,
                           const char* topic);

/* ── LRU Cache ── */

/* Initialize cache */
void cache_init(FrameCache* fc);

/* Get grid from cache (NULL if miss) */
SpatialGrid* cache_get(FrameCache* fc, uint32_t frame_id);

/* Put grid into cache (evicts LRU if full) */
void cache_put(FrameCache* fc, uint32_t frame_id, SpatialGrid* grid);

/* ── Integrated matching engine (SPEC-ENGINE final) ── */

uint32_t match_engine(SpatialAI* ai, SpatialGrid* input,
                      BucketIndex* bidx, BlockSummary* bs_all,
                      FrameCache* cache, float* out_sim);

#endif /* SPATIAL_CONTEXT_H */
