#include "spatial_canvas.h"
#include "spatial_layers.h"
#include "spatial_match.h"
#include "spatial_morpheme.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── DataType classification ────────────────────────────── */

static const char SPECIAL_CHARS[] = "{}()[];=<>#@$&|\\/";

DataType detect_data_type(const uint8_t* bytes, uint32_t len) {
    if (!bytes || len == 0) return DATA_SHORT;

    uint32_t special = 0;
    for (uint32_t i = 0; i < len; i++) {
        uint8_t b = bytes[i];
        if (b < 128 && b > 0 && strchr(SPECIAL_CHARS, (int)b)) special++;
    }
    float special_ratio = (float)special / (float)len;

    if (special_ratio > 0.15f) return DATA_CODE;
    if (len < 30)               return DATA_SHORT;
    if (len > 150)              return DATA_PROSE;
    return DATA_DIALOG;
}

const char* data_type_name(DataType t) {
    switch (t) {
        case DATA_PROSE:  return "PROSE";
        case DATA_DIALOG: return "DIALOG";
        case DATA_CODE:   return "CODE";
        case DATA_SHORT:  return "SHORT";
        default:          return "?";
    }
}

float data_type_boundary_weight(DataType t) {
    switch (t) {
        case DATA_PROSE:  return 0.5f;
        case DATA_DIALOG: return 0.3f;
        case DATA_CODE:   return 0.1f;
        case DATA_SHORT:  return 0.02f;
        default:          return 0.5f;
    }
}

/* djb2 topic hash used by canvas_add_clause */
static uint32_t topic_hash_djb2(const char* s) {
    uint32_t h = 5381;
    while (*s) h = h * 33u + (uint8_t)(*s++);
    return h;
}

#ifdef _WIN32
#include <malloc.h>
static void* cv_aligned(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void cv_aligned_free(void* p) { _aligned_free(p); }
#else
static void* cv_aligned(size_t alignment, size_t size) {
    void* p = NULL;
    posix_memalign(&p, alignment, size);
    return p;
}
static void cv_aligned_free(void* p) { free(p); }
#endif

/* ── Lifecycle ─────────────────────────────────────────── */

SpatialCanvas* canvas_create(void) {
    SpatialCanvas* c = (SpatialCanvas*)calloc(1, sizeof(SpatialCanvas));
    if (!c) return NULL;

    c->A = (uint16_t*)cv_aligned(32, CV_TOTAL * sizeof(uint16_t));
    c->R = (uint8_t*) cv_aligned(32, CV_TOTAL);
    c->G = (uint8_t*) cv_aligned(32, CV_TOTAL);
    c->B = (uint8_t*) cv_aligned(32, CV_TOTAL);

    if (!c->A || !c->R || !c->G || !c->B) {
        canvas_destroy(c);
        return NULL;
    }
    memset(c->A, 0, CV_TOTAL * sizeof(uint16_t));
    memset(c->R, 0, CV_TOTAL);
    memset(c->G, 0, CV_TOTAL);
    memset(c->B, 0, CV_TOTAL);
    c->width  = CV_WIDTH;
    c->height = CV_HEIGHT;
    c->slot_count = 0;
    c->canvas_type = DATA_PROSE;  /* overwritten by first placement */
    /* Default meta: unoccupied, full-weight boundaries so canvases
     * that never call canvas_add_clause still diffuse uniformly. */
    for (uint32_t s = 0; s < CV_SLOTS; s++) {
        c->meta[s].type = DATA_PROSE;
        c->meta[s].boundary_weight = 1.0f;
        c->meta[s].byte_length = 0;
        c->meta[s].topic_hash = 0;
        c->meta[s].occupied = 0;
    }
    /* I/P defaults — unclassified canvas treated as IFRAME with no parent */
    c->frame_type = CANVAS_IFRAME;
    c->parent_canvas_id = UINT32_MAX;
    c->changed_ratio = 0.0f;
    c->classified = 0;
    return c;
}

void canvas_destroy(SpatialCanvas* c) {
    if (!c) return;
    if (c->A) cv_aligned_free(c->A);
    if (c->R) cv_aligned_free(c->R);
    if (c->G) cv_aligned_free(c->G);
    if (c->B) cv_aligned_free(c->B);
    free(c);
}

void canvas_clear(SpatialCanvas* c) {
    if (!c) return;
    memset(c->A, 0, CV_TOTAL * sizeof(uint16_t));
    memset(c->R, 0, CV_TOTAL);
    memset(c->G, 0, CV_TOTAL);
    memset(c->B, 0, CV_TOTAL);
    c->slot_count = 0;
    for (uint32_t s = 0; s < CV_SLOTS; s++) {
        c->meta[s].type = DATA_PROSE;
        c->meta[s].boundary_weight = 1.0f;
        c->meta[s].byte_length = 0;
        c->meta[s].topic_hash = 0;
        c->meta[s].occupied = 0;
    }
    c->frame_type = CANVAS_IFRAME;
    c->parent_canvas_id = UINT32_MAX;
    c->changed_ratio = 0.0f;
    c->classified = 0;
}

/* ── Slot → canvas-space coordinate helpers ────────────── */

uint32_t canvas_slot_byte_offset(uint32_t slot, uint32_t* out_x0, uint32_t* out_y0) {
    uint32_t col = slot % CV_COLS;
    uint32_t row = slot / CV_COLS;
    uint32_t x0  = col * CV_TILE;
    uint32_t y0  = row * CV_TILE;
    if (out_x0) *out_x0 = x0;
    if (out_y0) *out_y0 = y0;
    return y0 * CV_WIDTH + x0;
}

/* ── Tile placement ────────────────────────────────────── */

int canvas_add_clause(SpatialCanvas* c, const char* text) {
    if (!c || !text || c->slot_count >= CV_SLOTS) return -1;

    /* Encode into a temporary 256×256 tile */
    SpatialGrid* tile = grid_create();
    if (!tile) return -1;
    morpheme_init();
    layers_encode_clause(text, NULL, tile);
    /* Note: we intentionally skip update_rgb_directional on the tile;
       diffusion happens canvas-wide after placement (see
       canvas_update_rgb). B is already seeded with the clause's
       co-occurrence hash by layers_encode_clause. */

    /* Copy tile into the slot */
    uint32_t slot = c->slot_count;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);

    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t ti = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            c->A[ci] = tile->A[ti];
            c->R[ci] = tile->R[ti];
            c->G[ci] = tile->G[ti];
            c->B[ci] = tile->B[ti];
        }
    }

    grid_destroy(tile);

    /* Populate meta for this slot from the clause's auto-detected type */
    uint32_t text_len = (uint32_t)strlen(text);
    DataType t = detect_data_type((const uint8_t*)text, text_len);
    c->meta[slot].type            = t;
    c->meta[slot].boundary_weight = data_type_boundary_weight(t);
    c->meta[slot].byte_length     = text_len;
    c->meta[slot].topic_hash      = topic_hash_djb2(text);
    c->meta[slot].occupied        = 1;

    /* First placement sets the canvas's overall type */
    if (c->slot_count == 0) c->canvas_type = t;

    c->slot_count++;
    return (int)slot;
}

/* ── Canvas-wide RGB directional diffusion ─────────────── */

/* Return the boundary-diffusion multiplier for an update that flows
 *   from (sx, sy) into (dx, dy). Within the same slot the multiplier
 *   is 1.0; crossing a slot boundary uses the min of the two slots'
 *   boundary_weight. */
static inline float boundary_multiplier(const SpatialCanvas* c,
                                        uint32_t sx, uint32_t sy,
                                        uint32_t dx, uint32_t dy) {
    uint32_t s1 = (sy / CV_TILE) * CV_COLS + (sx / CV_TILE);
    uint32_t s2 = (dy / CV_TILE) * CV_COLS + (dx / CV_TILE);
    if (s1 == s2) return 1.0f;
    if (s1 >= CV_SLOTS || s2 >= CV_SLOTS) return 1.0f;
    float w1 = c->meta[s1].boundary_weight;
    float w2 = c->meta[s2].boundary_weight;
    return (w1 < w2) ? w1 : w2;
}

void canvas_update_rgb(SpatialCanvas* c) {
    if (!c) return;
    uint32_t W = c->width;
    uint32_t H = c->height;

    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint32_t i = y * W + x;
            if (c->A[i] == 0) continue;

            /* R: diagonal neighbours (semantic / morpheme relation) */
            static const int dx_off[4] = {1, 1, -1, -1};
            static const int dy_off[4] = {1, -1, 1, -1};
            for (int d = 0; d < 4; d++) {
                int nx = (int)x + dx_off[d], ny = (int)y + dy_off[d];
                if (nx < 0 || nx >= (int)W || ny < 0 || ny >= (int)H) continue;
                uint32_t ni = (uint32_t)ny * W + (uint32_t)nx;
                if (c->A[ni] == 0) continue;
                float bw = boundary_multiplier(c, (uint32_t)nx, (uint32_t)ny, x, y);
                int diff = (int)c->R[ni] - (int)c->R[i];
                int new_v = (int)c->R[i] + (int)(ALPHA_R * diff * bw);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->R[i] = (uint8_t)new_v;
            }
            /* G: vertical (substitution) — crosses tile row boundaries */
            for (int d = -1; d <= 1; d += 2) {
                int ny = (int)y + d;
                if (ny < 0 || ny >= (int)H) continue;
                uint32_t ni = (uint32_t)ny * W + x;
                if (c->A[ni] == 0) continue;
                float bw = boundary_multiplier(c, x, (uint32_t)ny, x, y);
                int diff = (int)c->G[ni] - (int)c->G[i];
                int new_v = (int)c->G[i] + (int)(BETA_G * diff * bw);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->G[i] = (uint8_t)new_v;
            }
            /* B: horizontal (clause order) — crosses tile column boundaries */
            for (int d = -1; d <= 1; d += 2) {
                int nx = (int)x + d;
                if (nx < 0 || nx >= (int)W) continue;
                uint32_t ni = y * W + (uint32_t)nx;
                if (c->A[ni] == 0) continue;
                float bw = boundary_multiplier(c, (uint32_t)nx, y, x, y);
                int diff = (int)c->B[ni] - (int)c->B[i];
                int new_v = (int)c->B[i] + (int)(GAMMA_B * diff * bw);
                if (new_v < 0) new_v = 0;
                if (new_v > 255) new_v = 255;
                c->B[i] = (uint8_t)new_v;
            }
        }
    }
}

/* ── Stats ─────────────────────────────────────────────── */

uint32_t canvas_active_count(const SpatialCanvas* c) {
    if (!c) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < CV_TOTAL; i++) if (c->A[i] > 0) n++;
    return n;
}

void canvas_compute_block_sums(const SpatialCanvas* c, CanvasBlockSummary* out) {
    if (!c || !out) return;
    memset(out->sums, 0, sizeof(out->sums));
    for (uint32_t by = 0; by < CV_BLOCKS_Y; by++) {
        for (uint32_t bx = 0; bx < CV_BLOCKS_X; bx++) {
            uint32_t s = 0;
            for (uint32_t dy = 0; dy < CV_BLOCK; dy++) {
                for (uint32_t dx = 0; dx < CV_BLOCK; dx++) {
                    uint32_t ci = (by * CV_BLOCK + dy) * CV_WIDTH
                                 + (bx * CV_BLOCK + dx);
                    s += c->A[ci];
                }
            }
            out->sums[by * CV_BLOCKS_X + bx] = s;
        }
    }
}

/* ── Slot → grid export ────────────────────────────────── */

void canvas_slot_to_grid(const SpatialCanvas* c, uint32_t slot, SpatialGrid* out) {
    if (!c || !out || slot >= CV_SLOTS) return;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);
    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t ti = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            out->A[ti] = c->A[ci];
            out->R[ti] = c->R[ci];
            out->G[ti] = c->G[ci];
            out->B[ti] = c->B[ci];
        }
    }
}

/* ── Slot matching ─────────────────────────────────────── */

float canvas_match_slot(const SpatialCanvas* c, const SpatialGrid* query, uint32_t slot) {
    if (!c || !query || slot >= CV_SLOTS) return 0.0f;
    uint32_t x0, y0;
    canvas_slot_byte_offset(slot, &x0, &y0);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t dy = 0; dy < CV_TILE; dy++) {
        for (uint32_t dx = 0; dx < CV_TILE; dx++) {
            uint32_t qi = dy * CV_TILE + dx;
            uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
            double qa = (double)query->A[qi];
            double ca = (double)c->A[ci];
            if (qa > 0.0 && ca > 0.0) {
                /* RGB per-cell weight (same formula as rgb_weight) */
                double dr = fabs((double)query->R[qi] - c->R[ci]) / 255.0;
                double dg = fabs((double)query->G[qi] - c->G[ci]) / 255.0;
                double db = fabs((double)query->B[qi] - c->B[ci]) / 255.0;
                double w  = 1.0 - (0.5*dr + 0.3*dg + 0.2*db);
                if (w < 0.0) w = 0.0;
                dot += qa * ca * w;
            }
            na += qa * qa;
            nb += ca * ca;
        }
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

uint32_t canvas_best_slot(const SpatialCanvas* c, const SpatialGrid* query, float* out_sim) {
    if (!c || !query || c->slot_count == 0) {
        if (out_sim) *out_sim = 0.0f;
        return 0;
    }
    uint32_t best_slot = 0;
    float    best_sim  = -1.0f;
    for (uint32_t s = 0; s < c->slot_count; s++) {
        float v = canvas_match_slot(c, query, s);
        if (v > best_sim) { best_sim = v; best_slot = s; }
    }
    if (out_sim) *out_sim = best_sim;
    return best_slot;
}

/* ── Delta (sparse + RLE byte estimate) ────────────────── */

uint32_t canvas_delta_sparse(const SpatialCanvas* a, const SpatialCanvas* b,
                             CanvasDeltaEntry* out, uint32_t max_out) {
    if (!a || !b || !out) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < CV_TOTAL && n < max_out; i++) {
        int16_t diff = (int16_t)b->A[i] - (int16_t)a->A[i];
        if (diff != 0) {
            out[n].index  = i;
            out[n].diff_A = diff;
            n++;
        }
    }
    return n;
}

uint32_t canvas_delta_rle_bytes(const CanvasDeltaEntry* entries, uint32_t count) {
    if (!entries || count == 0) return 0;
    /* RLE record: (start:u32, length:u16, diff:i16) = 8 bytes per run.
       A run = consecutive indices where diff_A is identical. */
    uint32_t runs = 1;
    uint32_t prev_idx  = entries[0].index;
    int16_t  prev_diff = entries[0].diff_A;
    for (uint32_t k = 1; k < count; k++) {
        if (entries[k].index == prev_idx + 1 && entries[k].diff_A == prev_diff) {
            /* extend current run */
        } else {
            runs++;
        }
        prev_idx  = entries[k].index;
        prev_diff = entries[k].diff_A;
    }
    return runs * 8;
}
