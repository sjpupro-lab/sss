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
