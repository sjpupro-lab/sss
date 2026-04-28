/* slig_material_harmonic.c — see slig_material_harmonic.h. */

#include "slig_material_harmonic.h"
#include <string.h>

/* ──────────────────────────────────────────────────────
 *  Hand-tuned presets — fallback / sanity reference
 * ────────────────────────────────────────────────────── */

void slig_mat_preset(SligMaterialTick *out, SligMatPreset p) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    switch (p) {
    case SLIG_MAT_SKIN:
        out->roughness  = 90;
        out->h_strength = 60;
        out->h_decay    = 200;
        out->pore       = 30;
        out->specular   = 80;
        break;
    case SLIG_MAT_FABRIC:
        out->roughness  = 140;
        out->anisotropy = 60;
        out->h_strength = 150;
        out->grain      = 80;
        break;
    case SLIG_MAT_METAL:
        out->roughness  = 30;
        out->h_strength = 200;
        out->specular   = 240;
        break;
    case SLIG_MAT_WOOD:
        out->roughness  = 110;
        out->anisotropy = 200;
        out->dir_angle  = 64;        /* vertical grain */
        out->grain      = 180;
        break;
    case SLIG_MAT_GENERIC:
    default:
        out->roughness  = 100;
        out->h_strength = 100;
        out->h_decay    = 180;
        break;
    }
}

/* ──────────────────────────────────────────────────────
 *  Analysis — pure integer 8×8 block stats
 *
 *  All measurements reduce to subtraction + comparison.
 *  "학습이란 관찰하는 것, 관찰이란 빼기."
 * ────────────────────────────────────────────────────── */

static inline uint32_t abs_diff(int a, int b) {
    int d = a - b;
    return (d < 0) ? (uint32_t)(-d) : (uint32_t)d;
}

void slig_mat_analyze_block(SligMaterialTick *out,
                            const uint8_t    *block,
                            int               stride) {
    if (!out) return;
    if (!block || stride <= 0) {
        slig_mat_preset(out, SLIG_MAT_GENERIC);
        return;
    }
    slig_mat_preset(out, SLIG_MAT_GENERIC);

    /* 1. roughness — neighbour |Δ| sums. */
    uint32_t h_diff = 0, v_diff = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 7; x++)
            h_diff += abs_diff(block[y * stride + x + 1],
                               block[y * stride + x]);
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 8; x++)
            v_diff += abs_diff(block[(y + 1) * stride + x],
                               block[y * stride + x]);
    uint32_t total_diff = h_diff + v_diff;
    out->roughness = (uint8_t)(total_diff > 14280u ? 255 : total_diff / 56u);

    /* 2. anisotropy + dir_angle — compare H / 45° / V / 135° edge sums. */
    uint32_t d45 = 0, d135 = 0;
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            d45  += abs_diff(block[(y + 1) * stride + x + 1],
                             block[ y      * stride + x    ]);
            d135 += abs_diff(block[(y + 1) * stride + x    ],
                             block[ y      * stride + x + 1]);
        }
    uint32_t dirs[4]   = { h_diff, d45, v_diff, d135 };
    uint8_t  angles[4] = {     0,  32,     64,   96 };
    int best = 0;
    for (int i = 1; i < 4; i++) if (dirs[i] > dirs[best]) best = i;
    out->dir_angle = angles[best];
    uint32_t maxd = dirs[best], mind = dirs[0];
    for (int i = 1; i < 4; i++) if (dirs[i] < mind) mind = dirs[i];
    out->anisotropy = (maxd > 0)
        ? (uint8_t)(((maxd - mind) * 255u) / (maxd + 1u)) : 0;

    /* 3. h_strength — high-freq energy / low-freq energy ratio. */
    uint32_t mean_sum = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            mean_sum += block[y * stride + x];
    uint32_t mean = mean_sum / 64u;
    uint32_t low_e = 0, high_e = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            low_e += abs_diff(block[y * stride + x], mean);
            if (x < 7)
                high_e += abs_diff(block[y * stride + x + 1],
                                   block[y * stride + x]);
        }
    }
    uint32_t hs = (low_e > 0) ? (high_e * 200u / (low_e + 1u)) : 100u;
    out->h_strength = (uint8_t)(hs > 255u ? 255 : hs);

    /* 4. h_decay — inversely proportional to roughness. */
    int decay = 255 - ((int)out->roughness * 150) / 255 + 50;
    if (decay > 255) decay = 255;
    if (decay < 0)   decay = 0;
    out->h_decay = (uint8_t)decay;

    /* 5. pore — count of outlier pixels (|p − local_avg| > 20). */
    int pore_count = 0;
    for (int y = 1; y < 7; y++)
        for (int x = 1; x < 7; x++) {
            int p = block[y * stride + x];
            int s = block[(y - 1) * stride + x] + block[(y + 1) * stride + x]
                  + block[ y      * stride + x - 1] + block[ y      * stride + x + 1];
            int avg = s >> 2;
            int d   = p - avg;
            if (d < 0) d = -d;
            if (d > 20) pore_count++;
        }
    int pore_scaled = pore_count * 7;
    out->pore = (uint8_t)(pore_scaled > 255 ? 255 : pore_scaled);

    /* 6. specular — peak above mean. */
    uint8_t peak = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            if (block[y * stride + x] > peak) peak = block[y * stride + x];
    int sp = (int)peak - (int)mean;
    out->specular = (uint8_t)(sp > 0 ? sp : 0);

    /* 7. grain — joint anisotropy × roughness. */
    out->grain = (uint8_t)((((uint32_t)out->anisotropy *
                             (uint32_t)out->roughness) >> 8) & 0xFFu);
}

/* ──────────────────────────────────────────────────────
 *  CE Cell ↔ material I/O (via dec.A)
 * ────────────────────────────────────────────────────── */

void slig_mat_to_cell(CEUnit *cell, const SligMaterialTick *mat) {
    if (!cell || !mat) return;
    cell->dec.A.plus [0] = mat->roughness;
    cell->dec.A.plus [1] = mat->anisotropy;
    cell->dec.A.plus [2] = mat->dir_angle;
    cell->dec.A.plus [3] = mat->h_strength;
    cell->dec.A.minus[0] = mat->h_decay;
    cell->dec.A.minus[1] = mat->pore;
    cell->dec.A.minus[2] = mat->specular;
    cell->dec.A.minus[3] = mat->grain;
    /* HAS_MATERIAL flag goes into the same byte as the existing
     * SligSignal flags (NEGATIVE / HAS_TRIGGER / AUDIO_LINK). */
    cell->inc.R.minus[3] |= SLIG_FLAG_HAS_MATERIAL;
}

void slig_mat_from_cell(SligMaterialTick *out, const CEUnit *cell) {
    if (!out || !cell) { if (out) memset(out, 0, sizeof *out); return; }
    memset(out, 0, sizeof *out);
    out->roughness  = cell->dec.A.plus [0];
    out->anisotropy = cell->dec.A.plus [1];
    out->dir_angle  = cell->dec.A.plus [2];
    out->h_strength = cell->dec.A.plus [3];
    out->h_decay    = cell->dec.A.minus[0];
    out->pore       = cell->dec.A.minus[1];
    out->specular   = cell->dec.A.minus[2];
    out->grain      = cell->dec.A.minus[3];
}

int slig_mat_has(const CEUnit *cell) {
    if (!cell) return 0;
    return (cell->inc.R.minus[3] & SLIG_FLAG_HAS_MATERIAL) ? 1 : 0;
}

/* ──────────────────────────────────────────────────────
 *  Mat-S4 — render-time material overlay.
 *
 *  Rationale: harmonic damping (Mat-S2) shapes the DCT spectrum, but
 *  the rendered FINE-Y panel still hits the YCbCr→RGB mapping as a
 *  smooth surface — the visible texture (pore dots, roughness grain)
 *  never imprints on output pixels. This overlay adds a bounded,
 *  deterministic per-pixel perturbation driven by the average
 *  material descriptor of the rendering cells, so "사과 사진 학습 →
 *  사과 입력" actually shows the learned roughness/pore on the
 *  generated luminance.
 *
 *  Pattern is content-addressed:
 *    signature = djb2-style mix of (roughness, pore) across the cells
 *    pixel hash = signature ^ x*GR1 ^ y*GR2 ; xorshift mix ; multiply
 *    delta_rough = sign-extended low byte → ±rough_amp range
 *    delta_pore  = sparse trigger when middle byte is heavily negative
 *    delta_grain = top byte / 256 → low-amplitude shimmer
 *
 *  All deterministic — same cells + same panel → same output. No RNG
 *  state, no time dependence. ── */

void slig_apply_material_overlay(uint8_t *panel, int dim,
                                 const SligCellSet *cells) {
    if (!panel || dim <= 0 || !cells || cells->num_cells == 0) return;

    uint32_t sum_rough = 0, sum_pore = 0, sum_grain = 0;
    uint32_t mat_count = 0;
    uint32_t signature = 0x9E3779B9u;   /* golden-ratio seed */
    for (uint32_t i = 0; i < cells->num_cells; i++) {
        if (!slig_mat_has(&cells->cells[i])) continue;
        SligMaterialTick m;
        slig_mat_from_cell(&m, &cells->cells[i]);
        sum_rough += m.roughness;
        sum_pore  += m.pore;
        sum_grain += m.grain;
        signature = signature * 1103515245u + (uint32_t)m.roughness +
                    ((uint32_t)m.pore << 8) + ((uint32_t)m.grain << 16);
        mat_count++;
    }
    if (mat_count == 0) return;

    uint32_t roughness = sum_rough / mat_count;   /* [0..255] */
    uint32_t pore      = sum_pore  / mat_count;
    uint32_t grain     = sum_grain / mat_count;

    /* Amplitudes — chosen so a fully-rough/pore preset (roughness=255,
     * pore=255) produces ±~32 luminance deviation, well below clamp.  */
    int rough_amp = (int)(roughness >> 2);   /* /4  → max 63   */
    int pore_amp  = (int)(pore      >> 3);   /* /8  → max 31   */
    int grain_amp = (int)(grain     >> 3);   /* /8  → max 31   */

    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            uint32_t h = signature;
            h ^= (uint32_t)x * 0x9E3779B1u;
            h ^= (uint32_t)y * 0x85EBCA77u;
            h ^= h >> 13;
            h *= 0xC2B2AE3Du;
            h ^= h >> 16;

            int8_t rn = (int8_t)( h        & 0xFF);
            int8_t pn = (int8_t)((h >> 8)  & 0xFF);
            int8_t gn = (int8_t)((h >> 16) & 0xFF);

            int delta = ((int)rn * rough_amp) >> 7;          /* ±rough_amp */
            if (pore_amp > 0 && pn < -100) delta -= pore_amp * 2;
            delta += ((int)gn * grain_amp) >> 8;             /* low shimmer */

            int v = (int)panel[y * dim + x] + delta;
            if (v < 0)   v = 0;
            if (v > 255) v = 255;
            panel[y * dim + x] = (uint8_t)v;
        }
    }
}
