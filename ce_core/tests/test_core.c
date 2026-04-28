/* test_core.c — base CE Cell unit tests. */
#include "../ce_core.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

int main(void) {
    printf("=== ce_core tests ===\n");

    CEUnit u;
    ce_init(&u);
    int zero = 1;
    for (int i = 0; i < 64; ++i) if (ce_cbytes(&u)[i] != 0) { zero = 0; break; }
    CHECK(zero, "ce_init zeroes 64 bytes");

    /* feed determinism */
    CEUnit a, b;
    ce_init(&a); ce_init(&b);
    const char *s = "the quick brown fox jumps over the lazy dog";
    ce_feed(&a, (const uint8_t *)s, (uint32_t)strlen(s));
    ce_feed(&b, (const uint8_t *)s, (uint32_t)strlen(s));
    CHECK(ce_equal(&a, &b), "ce_feed deterministic on identical input");

    /* feed sensitivity */
    CEUnit c;
    ce_init(&c);
    const char *s2 = "the quick brown fox jumps over the lazy doh"; /* one byte diff */
    ce_feed(&c, (const uint8_t *)s2, (uint32_t)strlen(s2));
    CHECK(!ce_equal(&a, &c), "ce_feed differs on different input");

    /* distance properties */
    CHECK(ce_distance(&a, &a) == 0, "ce_distance(x,x) == 0");
    CHECK(ce_distance(&a, &b) == 0, "ce_distance equal-state == 0");
    CHECK(ce_distance(&a, &c) > 0, "ce_distance non-zero on different");
    CHECK(ce_distance(&a, &c) == ce_distance(&c, &a), "ce_distance symmetric");

    /* delta + apply round trip */
    CEUnit d, restored;
    ce_delta(&d, &a, &c);
    ce_apply(&restored, &a, &d);
    CHECK(ce_equal(&restored, &c), "ce_apply(anchor, ce_delta(anchor, x)) == x");

    /* delta(x,x) is the zero delta */
    CEUnit zero_delta;
    ce_delta(&zero_delta, &a, &a);
    int z = 1;
    for (int i = 0; i < 64; ++i) if (ce_cbytes(&zero_delta)[i] != 0) { z = 0; break; }
    CHECK(z, "ce_delta(x,x) is zero");

    /* ce_read returns 0 for fresh init */
    CEUnit u0; ce_init(&u0);
    CHECK(ce_read(&u0, CE_CH_R) == 0, "ce_read(init, R) == 0");
    CHECK(ce_read(&u0, CE_CH_A) == 0, "ce_read(init, A) == 0");

    /* ce_read changes after feed */
    int nonzero = 0;
    for (int ch = 0; ch < 4; ++ch) {
        if (ce_read(&a, ch) != 0) { nonzero = 1; break; }
    }
    CHECK(nonzero, "ce_read changes after ce_feed");

    /* delta_scale: weight 0 → all zeros; weight 1 → identity */
    CEUnit ds, idd;
    ce_delta_scale(&ds, &d, 0.0f);
    int allz = 1;
    for (int i = 0; i < 64; ++i) if (ce_cbytes(&ds)[i] != 0) { allz = 0; break; }
    CHECK(allz, "ce_delta_scale(d, 0.0) == zero");

    ce_delta_scale(&idd, &d, 1.0f);
    /* For bytes representing signed values, scaling by 1 should round-trip. */
    int matches = 1;
    for (int i = 0; i < 64; ++i) if (ce_cbytes(&idd)[i] != ce_cbytes(&d)[i]) { matches = 0; break; }
    CHECK(matches, "ce_delta_scale(d, 1.0) preserves bytes");

    /* feed throughput sanity (no perf assertion, just runs without overflow) */
    CEUnit big;
    ce_init(&big);
    static uint8_t buf[4096];
    for (int i = 0; i < (int)sizeof(buf); ++i) buf[i] = (uint8_t)(i * 31 + 7);
    ce_feed(&big, buf, (uint32_t)sizeof(buf));
    CHECK(1, "ce_feed handles 4KB buffer without crash");

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
