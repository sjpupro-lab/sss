/* test_tick_math.c — slig_tick_math helpers + tables. */

#include "slig_tick_math.h"
#include "ce_core.h"
#include "slig_signal.h"
#include <stdio.h>
#include <string.h>

static int pass = 0, fail = 0;
static void check(const char *name, int cond) {
    if (cond) { printf("  [PASS] %s\n", name); pass++; }
    else      { printf("  [FAIL] %s\n", name); fail++; }
}

int main(void) {
    printf("=== slig_tick_math ===\n");

    /* tick_add carry: 255 + 1 → plus[0]=0, plus[1]=1 */
    {
        uint8_t plus[4] = {255, 0, 0, 0};
        tick_add(plus, 1);
        check("carry 255+1 → plus[0]=0", plus[0] == 0);
        check("carry 255+1 → plus[1]=1", plus[1] == 1);
    }

    /* tick_read / tick_write roundtrip */
    {
        uint8_t plus[4] = {0};
        tick_write(plus, 0xDEADBEEFu);
        check("write/read roundtrip", tick_read(plus) == 0xDEADBEEFu);
    }

    /* tick_sub */
    {
        uint8_t plus[4] = {0};
        tick_write(plus, 1000);
        tick_sub(plus, 250);
        check("sub 1000−250", tick_read(plus) == 750);
    }

    /* tick_mul8 */
    check("mul8 200×200", tick_mul8(200, 200) == 40000);

    /* tick_sqrt */
    check("sqrt(65536)≈256",   tick_sqrt(65536u)  == 256);
    check("sqrt(10000)≈100",   tick_sqrt(10000u)  == 100);
    check("sqrt(0)=0",         tick_sqrt(0u)      == 0);

    /* tick_div */
    check("div 1000/3 = 333",  tick_div(1000, 3)  == 333);
    check("div by 0 → 0",      tick_div(1000, 0)  == 0);

    /* Sin table — peak near i=64 (π/2). 128+127·sin(π/2) = 255. */
    check("SIN_TABLE[64] ≈ 255", TICK_SIN_TABLE[64]  >= 254);
    check("SIN_TABLE[0]  ≈ 128", TICK_SIN_TABLE[0]   == 128);
    check("SIN_TABLE[192] ≈ 1",  TICK_SIN_TABLE[192] <= 2);

    /* Cos table — peak at i=0. 128+127·cos(0) = 255. */
    check("COS_TABLE[0]   ≈ 255", TICK_COS_TABLE[0]   >= 254);
    check("COS_TABLE[64]  ≈ 128", TICK_COS_TABLE[64]  >= 127 && TICK_COS_TABLE[64] <= 129);

    /* Gauss table — 255 at 0, decays to ~0 at high indices */
    check("GAUSS_TABLE[0] = 255", TICK_GAUSS_TABLE[0] == 255);
    check("GAUSS_TABLE[255] near 0", TICK_GAUSS_TABLE[255] <= 5);
    check("GAUSS strictly decreasing (0→64)",
          TICK_GAUSS_TABLE[0] > TICK_GAUSS_TABLE[64]);

    /* WAVE_STEPS shape check (1,3,10,30,10,3,1) */
    check("WAVE_STEPS[3] = 30 (peak)", WAVE_STEPS[3] == 30);
    check("WAVE_STEPS symmetric",
          WAVE_STEPS[0] == WAVE_STEPS[6] &&
          WAVE_STEPS[1] == WAVE_STEPS[5] &&
          WAVE_STEPS[2] == WAVE_STEPS[4]);

    /* Mixed-mode read/write 16x2 */
    {
        uint8_t p[4] = {0};
        tick_write16x2(p, 0xCAFE, 0x1234);
        uint16_t lo, hi;
        tick_read16x2(p, &lo, &hi);
        check("read16x2 low",  lo == 0xCAFE);
        check("read16x2 high", hi == 0x1234);
    }

    /* Frequency-space search primitives (v3 §5) */
    {
        SligSignal s;
        memset(&s, 0, sizeof s);
        s.dir       = SLIG_DIR_HORIZONTAL;
        s.scale     = SLIG_SCALE_MID;
        s.sigma     = 5000;
        s.frequency = 8;
        for (int i = 0; i < SLIG_SIG_LEN; i++) {
            s.u[i] = (int16_t)((i * 13) & 0x7FFF);
            s.v[i] = (int16_t)((i *  7) & 0x7FFF);
        }
        CEUnit a, b, c;
        slig_pack(&a, &s);
        slig_pack(&b, &s);            /* identical */
        s.sigma = 100;                /* perturbed */
        slig_pack(&c, &s);

        check("distance_precise(a,a) = 0", tick_distance_precise(&a, &a) == 0);
        check("distance_precise(a,b) = 0 (same content)",
              tick_distance_precise(&a, &b) == 0);
        check("distance_precise(a,c) > 0",
              tick_distance_precise(&a, &c) > 0);

        uint32_t cs_aa = tick_cosine_similarity(&a, &a);
        uint32_t cs_ac = tick_cosine_similarity(&a, &c);
        printf("    cos_sim(a,a)=%u  cos_sim(a,c)=%u\n", cs_aa, cs_ac);
        check("cosine_sim(a,a) ≈ 256 (full)", cs_aa >= 250 && cs_aa <= 260);
        check("cosine_sim(a,c) ≤ cosine_sim(a,a)", cs_ac <= cs_aa);

        uint32_t fd_aa = tick_frequency_distance(&a, &a);
        uint32_t fd_ac = tick_frequency_distance(&a, &c);
        check("frequency_distance(a,a) = 0", fd_aa == 0);
        printf("    freq_dist(a,c)=%u\n", fd_ac);
    }

    printf("=== %d PASS / %d FAIL ===\n", pass, fail);
    return fail;
}
