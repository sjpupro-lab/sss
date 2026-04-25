/* ce_core.c — CE Cell base unit implementation. */
#include "ce_core.h"
#include <string.h>
#include <math.h>

void ce_init(CEUnit *u) {
    memset(u, 0, sizeof(*u));
}

/* Add `v` to a 4-byte carry chain `chain[4]`. Wraps mod 2^32 (cheap & deterministic). */
static void carry_add(uint8_t chain[4], uint32_t v) {
    uint32_t acc = (uint32_t)chain[0]
                 | ((uint32_t)chain[1] << 8)
                 | ((uint32_t)chain[2] << 16)
                 | ((uint32_t)chain[3] << 24);
    acc += v;
    chain[0] = (uint8_t)(acc & 0xFF);
    chain[1] = (uint8_t)((acc >> 8) & 0xFF);
    chain[2] = (uint8_t)((acc >> 16) & 0xFF);
    chain[3] = (uint8_t)((acc >> 24) & 0xFF);
}

void ce_feed(CEUnit *u, const uint8_t *data, uint32_t len) {
    if (!data || len == 0) return;
    /* Each input byte is dispatched to a (half, channel, direction) triple
     * based on its position. Distinct byte streams produce distinct states.
     * The ordering uses position-mixing so adjacent bytes hit different
     * tiers, building up entropy across all 64 bytes quickly. */
    for (uint32_t i = 0; i < len; ++i) {
        uint8_t b = data[i];
        /* Stream-position salt — keeps long streams from collapsing. */
        uint32_t pos = i ^ (i >> 3) ^ (i << 5);
        uint8_t ch = (uint8_t)((b ^ (uint8_t)pos) & 0x03);
        uint8_t half = (uint8_t)(((i >> 2) ^ (b >> 6)) & 1);
        uint8_t dir  = (uint8_t)(((i >> 1) ^ (b >> 3)) & 1);

        CEHalf *h = half ? &u->dec : &u->inc;
        CETier *tier;
        switch (ch) {
            case 0: tier = &h->R; break;
            case 1: tier = &h->G; break;
            case 2: tier = &h->B; break;
            default: tier = &h->A; break;
        }
        if (dir) {
            carry_add(tier->plus, (uint32_t)b + 1u);
        } else {
            carry_add(tier->minus, (uint32_t)b + 1u);
        }
        /* Cross-mix nibbles into the opposite half/dir for richer coupling. */
        uint8_t nh = (uint8_t)(b >> 4);
        uint8_t nl = (uint8_t)(b & 0x0F);
        CEHalf *oh = half ? &u->inc : &u->dec;
        CETier *ot;
        switch ((ch + 1) & 3) {
            case 0: ot = &oh->R; break;
            case 1: ot = &oh->G; break;
            case 2: ot = &oh->B; break;
            default: ot = &oh->A; break;
        }
        if (dir) carry_add(ot->minus, nh + 1u);
        else     carry_add(ot->plus,  nl + 1u);
    }
}

uint32_t ce_distance(const CEUnit *a, const CEUnit *b) {
    const uint8_t *pa = ce_cbytes(a);
    const uint8_t *pb = ce_cbytes(b);
    uint32_t sum = 0;
    for (int i = 0; i < 64; ++i) {
        int d = (int)pa[i] - (int)pb[i];
        sum += (d < 0) ? (uint32_t)(-d) : (uint32_t)d;
    }
    return sum;
}

void ce_delta(CEUnit *out, const CEUnit *anchor, const CEUnit *current) {
    uint8_t *po = ce_bytes(out);
    const uint8_t *pa = ce_cbytes(anchor);
    const uint8_t *pc = ce_cbytes(current);
    for (int i = 0; i < 64; ++i) {
        po[i] = (uint8_t)(pc[i] - pa[i]); /* mod 256 */
    }
}

void ce_apply(CEUnit *out, const CEUnit *anchor, const CEUnit *delta) {
    uint8_t *po = ce_bytes(out);
    const uint8_t *pa = ce_cbytes(anchor);
    const uint8_t *pd = ce_cbytes(delta);
    for (int i = 0; i < 64; ++i) {
        po[i] = (uint8_t)(pa[i] + pd[i]); /* mod 256 */
    }
}

int64_t ce_read(const CEUnit *u, int channel) {
    const CETier *ti, *td;
    switch (channel) {
        case CE_CH_R: ti = &u->inc.R; td = &u->dec.R; break;
        case CE_CH_G: ti = &u->inc.G; td = &u->dec.G; break;
        case CE_CH_B: ti = &u->inc.B; td = &u->dec.B; break;
        case CE_CH_A: ti = &u->inc.A; td = &u->dec.A; break;
        default: return 0;
    }
    int64_t inc_plus  = (int64_t)ti->plus[0]  | ((int64_t)ti->plus[1]  << 8)
                      | ((int64_t)ti->plus[2]  << 16) | ((int64_t)ti->plus[3]  << 24);
    int64_t inc_minus = (int64_t)ti->minus[0] | ((int64_t)ti->minus[1] << 8)
                      | ((int64_t)ti->minus[2] << 16) | ((int64_t)ti->minus[3] << 24);
    int64_t dec_plus  = (int64_t)td->plus[0]  | ((int64_t)td->plus[1]  << 8)
                      | ((int64_t)td->plus[2]  << 16) | ((int64_t)td->plus[3]  << 24);
    int64_t dec_minus = (int64_t)td->minus[0] | ((int64_t)td->minus[1] << 8)
                      | ((int64_t)td->minus[2] << 16) | ((int64_t)td->minus[3] << 24);
    return (inc_plus - inc_minus) - (dec_plus - dec_minus);
}

void ce_delta_scale(CEUnit *out, const CEUnit *delta, float weight) {
    if (weight > 8.0f)  weight = 8.0f;
    if (weight < -8.0f) weight = -8.0f;
    uint8_t *po = ce_bytes(out);
    const uint8_t *pd = ce_cbytes(delta);
    for (int i = 0; i < 64; ++i) {
        /* Treat the byte as signed delta in [-128, 127] for scaling. */
        int sb = (int)(int8_t)pd[i];
        float fv = (float)sb * weight;
        if (fv >= 0) fv += 0.5f; else fv -= 0.5f;
        int iv = (int)fv;
        po[i] = (uint8_t)(iv & 0xFF);
    }
}

int ce_equal(const CEUnit *a, const CEUnit *b) {
    return memcmp(a, b, sizeof(CEUnit)) == 0;
}
