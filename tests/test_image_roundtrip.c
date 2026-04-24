/* v4 Task F — image roundtrip feasibility test.
 *
 * Creates a synthetic PPM P6 (horizontal gradient with a diagonal
 * stripe), runs it through image_to_grid → grid_to_image, reloads
 * the output, and checks that:
 *   - the output file is a valid PPM of the same size
 *   - the RGB byte stream survives the roundtrip byte-for-byte
 *     (lossless for the prototype's direct copy path)
 *   - PSNR is reported for informational use — NOT a pass/fail gate
 *     per 08_TASK_F policy */

#include "spatial_grid.h"
#include "spatial_image.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS()     do { tests_passed++; printf("PASS\n"); } while(0)

/* Write a synthetic PPM P6 file. Returns 1 on success. */
static int write_synthetic_ppm(const char* path) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp, "P6\n%u %u\n255\n", GRID_SIZE, GRID_SIZE);
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint8_t rgb[3];
            rgb[0] = (uint8_t)x;                        /* horizontal R ramp */
            rgb[1] = (uint8_t)y;                        /* vertical   G ramp */
            rgb[2] = (uint8_t)((x + y) & 0xFF);         /* diagonal   B */
            if ((x + y) % 32 == 0) {                    /* diagonal stripe */
                rgb[0] = 255; rgb[1] = 255; rgb[2] = 255;
            }
            if (fwrite(rgb, 1, 3, fp) != 3) { fclose(fp); return 0; }
        }
    }
    return fclose(fp) == 0;
}

/* Compare two PPM P6 files byte-for-byte AND compute PSNR across
 * the RGB byte streams. Returns 1 on match, 0 otherwise. */
static int compare_ppm_and_psnr(const char* a, const char* b, double* psnr_out) {
    FILE* fa = fopen(a, "rb");
    FILE* fb = fopen(b, "rb");
    if (!fa || !fb) { if (fa) fclose(fa); if (fb) fclose(fb); return 0; }

    /* Skip headers of both (P6\n W H\n 255\n) — we wrote them with a
     * known format, so we can advance each past 3 newlines. */
    for (FILE* f = fa; ; f = fb) {
        int nl = 0, c;
        while (nl < 3 && (c = fgetc(f)) != EOF) {
            if (c == '\n') nl++;
        }
        if (f == fb) break;
    }

    uint64_t sq_err = 0;
    uint64_t n_bytes = 0;
    int match = 1;
    for (;;) {
        int ca = fgetc(fa);
        int cb = fgetc(fb);
        if (ca == EOF && cb == EOF) break;
        if (ca == EOF || cb == EOF) { match = 0; break; }
        int d = ca - cb;
        sq_err += (uint64_t)(d * d);
        n_bytes++;
        if (ca != cb) match = 0;
    }
    fclose(fa); fclose(fb);

    if (psnr_out) {
        if (n_bytes == 0 || sq_err == 0) {
            *psnr_out = 99.0;  /* identical */
        } else {
            double mse = (double)sq_err / (double)n_bytes;
            double psnr = 10.0 * log10((255.0 * 255.0) / mse);
            *psnr_out = psnr;
        }
    }
    return match;
}

/* ── F2-1 ── ppm → grid → ppm preserves every RGB byte ── */
static void test_roundtrip_rgb_lossless(void) {
    TEST("image_to_grid → grid_to_image is lossless on RGB bytes");
    const char* in  = "build/roundtrip_in.ppm";
    const char* out = "build/roundtrip_out.ppm";

    assert(write_synthetic_ppm(in));
    SpatialGrid* g = image_to_grid(in);
    assert(g != NULL);
    assert(grid_to_image(g, out) == 1);
    grid_destroy(g);

    double psnr = 0.0;
    int match = compare_ppm_and_psnr(in, out, &psnr);
    printf("\n    PSNR=%.2f dB  identical=%s\n",
           psnr, match ? "yes" : "no");
    assert(match);      /* byte-for-byte */
    assert(psnr > 50);  /* informational — always near-infinite for identical */

    remove(in); remove(out);
    PASS();
}

/* ── F2-2 ── malformed inputs return NULL / fail gracefully ── */
static void test_malformed_inputs(void) {
    TEST("image_to_grid rejects malformed / wrong-size PPM");
    SpatialGrid* g = image_to_grid("does_not_exist.ppm");
    assert(g == NULL);

    /* Wrong magic. */
    const char* bad = "build/roundtrip_bad.ppm";
    FILE* fp = fopen(bad, "wb");
    assert(fp);
    fprintf(fp, "P3\n%u %u\n255\nnot binary\n", GRID_SIZE, GRID_SIZE);
    fclose(fp);
    assert(image_to_grid(bad) == NULL);

    /* Wrong size. */
    fp = fopen(bad, "wb");
    fprintf(fp, "P6\n%u %u\n255\n", (GRID_SIZE/2), GRID_SIZE);
    for (uint32_t i = 0; i < (GRID_SIZE/2) * GRID_SIZE * 3; i++) {
        fputc(0, fp);
    }
    fclose(fp);
    assert(image_to_grid(bad) == NULL);

    remove(bad);
    /* grid_to_image with NULL inputs */
    assert(grid_to_image(NULL, "x") == 0);
    assert(grid_to_image((const SpatialGrid*)0x1, NULL) == 0);
    PASS();
}

int main(void) {
    printf("=== test_image_roundtrip ===\n");
    /* build/ directory exists once make all has run; tests run after
     * all → OBJS so the directory is present. */
    test_roundtrip_rgb_lossless();
    test_malformed_inputs();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
