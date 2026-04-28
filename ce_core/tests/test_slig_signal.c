/* test_slig_signal.c — SLIG signal v2 integration test.
 *
 * Tests:
 *   1. Pack → unpack CE Cell roundtrip
 *   2. Image decompose → CE Cell set
 *   3. Multi-direction rendering (horizontal + diagonal)
 *   4. Event signals (ripple, beam, cross, cascade)
 *   5. Audio track spectrum → weight
 *   6. Progressive render
 *   7. Timed render (tick accumulation)
 *   8. Interpolation
 *   9. Generation speed
 */

#include "../slig_signal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int pass = 0, fail = 0;
static void check(const char *name, int cond) {
    if (cond) { printf("  [PASS] %s\n", name); pass++; }
    else      { printf("  [FAIL] %s\n", name); fail++; }
}

/* ── Test images ── */
static void make_circle(uint8_t *img, int n) {
    float cx = n/2.0f, cy = n/2.0f, r = n*0.35f;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float d = sqrtf((y-cy)*(y-cy)+(x-cx)*(x-cx));
            img[y*n+x] = (d < r) ? 220 : 25;
        }
}
static void make_star(uint8_t *img, int n) {
    float cx = n/2.0f, cy = n/2.0f;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float dx = x-cx, dy = y-cy;
            float a = atan2f(dy,dx), d = sqrtf(dx*dx+dy*dy);
            float sr = n*0.12f + n*0.22f*fabsf(cosf(2.5f*a));
            img[y*n+x] = (d < sr) ? 240 : 15;
        }
}
static void make_diag(uint8_t *img, int n) {
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            img[y*n+x] = ((x+y) / 8 % 2 == 0) ? 200 : 50;
}

int main(void) {
    printf("=== SLIG Signal v2 — 통합 테스트 ===\n\n");

    int dim = 64;
    uint8_t *img_circle = (uint8_t*)malloc(dim*dim);
    uint8_t *img_star   = (uint8_t*)malloc(dim*dim);
    uint8_t *img_diag   = (uint8_t*)malloc(dim*dim);
    make_circle(img_circle, dim);
    make_star(img_star, dim);
    make_diag(img_diag, dim);

    /* ═══ Test 1: Pack → Unpack roundtrip ═══ */
    printf("--- Test 1: CE Cell pack/unpack roundtrip ---\n");
    {
        SligSignal orig;
        memset(&orig, 0, sizeof(orig));
        orig.dir = SLIG_DIR_DIAG_DOWN;
        orig.scale = SLIG_SCALE_MID;
        orig.sigma = 1234;
        orig.frequency = 16;
        orig.origin_x = 64;
        orig.origin_y = 100;
        orig.flags = SLIG_FLAG_AUDIO_LINK;
        orig.start_tick = 5;
        orig.decay = 200;
        orig.speed = 3;
        orig.beam_dir = 4;
        for (int i = 0; i < SLIG_SIG_LEN; i++) {
            orig.u[i] = (int16_t)(sinf(i * 0.05f) * 5000);
            orig.v[i] = (int16_t)(cosf(i * 0.03f) * 5000);
        }
        orig.audio_bins[0] = 42; orig.audio_amps[0] = 200;

        CEUnit cell;
        slig_pack(&cell, &orig);

        SligSignal restored;
        slig_unpack(&restored, &cell);

        check("dir preserved", restored.dir == orig.dir);
        check("scale preserved", restored.scale == orig.scale);
        check("sigma preserved", restored.sigma == orig.sigma);
        check("frequency preserved", restored.frequency == orig.frequency);
        check("origin_x preserved", restored.origin_x == orig.origin_x);
        check("origin_y preserved", restored.origin_y == orig.origin_y);
        check("flags preserved", restored.flags == orig.flags);
        check("start_tick preserved", restored.start_tick == orig.start_tick);
        check("decay preserved", restored.decay == orig.decay);
        check("speed preserved", restored.speed == orig.speed);
        check("audio_bins[0] preserved", restored.audio_bins[0] == 42);
        check("audio_amps[0] preserved", restored.audio_amps[0] == 200);

        /* u/v 신호: DCT 압축으로 완벽하진 않지만 상관관계 높아야 함 */
        float corr_u = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < SLIG_SIG_LEN; i++) {
            corr_u += orig.u[i] * restored.u[i];
            norm1 += orig.u[i] * orig.u[i];
            norm2 += restored.u[i] * restored.u[i];
        }
        float r_u = corr_u / (sqrtf(norm1) * sqrtf(norm2) + 1e-10f);
        printf("    u-signal correlation: %.3f\n", r_u);
        check("u-signal correlation > 0.5", r_u > 0.5f);
    }

    /* ═══ Test 2: Image decompose ═══ */
    printf("\n--- Test 2: Image → CE Cell set ---\n");
    {
        SligCellSet set_c, set_s, set_d;
        slig_decompose(&set_c, img_circle, dim, dim);
        slig_decompose(&set_s, img_star, dim, dim);
        slig_decompose(&set_d, img_diag, dim, dim);

        printf("    Circle: %d cells\n", set_c.num_cells);
        printf("    Star:   %d cells\n", set_s.num_cells);
        printf("    Diag:   %d cells\n", set_d.num_cells);

        check("Circle has cells", set_c.num_cells > 0);
        check("Star has cells", set_s.num_cells > 0);
        check("Diag has cells", set_d.num_cells > 0);

        /* 방향 다양성 확인 */
        int has_horiz = 0, has_diag = 0, has_event = 0;
        for (uint32_t i = 0; i < set_c.num_cells; i++) {
            SligSignal sig;
            slig_unpack(&sig, &set_c.cells[i]);
            if (sig.dir == SLIG_DIR_HORIZONTAL) has_horiz = 1;
            if (sig.dir == SLIG_DIR_DIAG_DOWN)  has_diag = 1;
            if (sig.dir >= SLIG_DIR_RIPPLE)      has_event = 1;
        }
        check("Has horizontal signals", has_horiz);
        check("Has diagonal signals", has_diag);
        check("Has event signals", has_event);
    }

    /* ═══ Test 3: Multi-direction render ═══ */
    printf("\n--- Test 3: Multi-direction render ---\n");
    {
        SligCellSet set;
        slig_decompose(&set, img_circle, dim, dim);

        uint8_t output[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
        slig_render((uint8_t*)output, set.cells, set.num_cells, NULL);

        /* 이미지가 빈 게 아닌지 확인 */
        int non_zero = 0;
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++)
                if (output[y][x] > 0 && output[y][x] < 255) non_zero++;

        check("Rendered image has content", non_zero > dim * dim / 4);
        printf("    Non-trivial pixels: %d / %d\n", non_zero, dim*dim);
    }

    /* ═══ Test 4: Event signals ═══ */
    printf("\n--- Test 4: Event signals ---\n");
    {
        SligCanvas canvas;
        SligSignal ripple, beam, cross;
        memset(&ripple, 0, sizeof(ripple));
        memset(&beam, 0, sizeof(beam));
        memset(&cross, 0, sizeof(cross));

        /* Ripple */
        ripple.dir = SLIG_DIR_RIPPLE;
        ripple.sigma = 5000;
        ripple.origin_x = 128; ripple.origin_y = 128;
        ripple.frequency = 10; ripple.speed = 3; ripple.decay = 230;

        slig_canvas_clear(&canvas);
        for (uint8_t t = 0; t < 20; t++)
            slig_apply_at_tick(&canvas, &ripple, t);
        int32_t center = canvas.pixels[128][128];
        int32_t edge = canvas.pixels[0][0];
        check("Ripple: center has more energy", abs(center) > abs(edge));

        /* Beam */
        beam.dir = SLIG_DIR_BEAM;
        beam.sigma = 5000;
        beam.origin_x = 10; beam.origin_y = 128;
        beam.beam_dir = 0; /* → */
        beam.beam_width = 15;
        beam.frequency = 8; beam.speed = 4; beam.decay = 240;

        slig_canvas_clear(&canvas);
        for (uint8_t t = 0; t < 20; t++)
            slig_apply_at_tick(&canvas, &beam, t);
        int32_t along = canvas.pixels[128][64]; /* beam 경로 위 */
        int32_t off = canvas.pixels[50][64];    /* beam 경로 밖 */
        check("Beam: along path > off path", abs(along) > abs(off));

        /* Cross */
        cross.dir = SLIG_DIR_CROSS;
        cross.sigma = 5000;
        cross.origin_x = 128; cross.origin_y = 128;
        cross.frequency = 8; cross.speed = 3; cross.decay = 230;

        slig_canvas_clear(&canvas);
        for (uint8_t t = 0; t < 15; t++)
            slig_apply_at_tick(&canvas, &cross, t);
        int32_t on_axis = canvas.pixels[128][100]; /* 십자 위 */
        int32_t off_axis = canvas.pixels[100][100]; /* 대각선 */
        check("Cross: on-axis > off-axis", abs(on_axis) > abs(off_axis));
    }

    /* ═══ Test 5: Audio track ═══ */
    printf("\n--- Test 5: Audio track spectrum ---\n");
    {
        /* 간이 격자 생성 */
        uint16_t grid_A[256][256];
        memset(grid_A, 0, sizeof(grid_A));
        /* "고양이" 비슷한 바이트 패턴 */
        grid_A[0][234] = 3; grid_A[1][179] = 3; grid_A[2][160] = 3;
        grid_A[3][236] = 3; grid_A[4][150] = 3; grid_A[5][145] = 3;

        uint32_t spectrum[256];
        slig_grid_to_spectrum(spectrum, grid_A);

        check("Spectrum at 234 > 0", spectrum[234] > 0);
        check("Spectrum at 0 == 0", spectrum[0] == 0);

        /* CE Cell 가중치 */
        SligCellSet set;
        slig_decompose(&set, img_circle, dim, dim);

        uint8_t weights[SLIG_MAX_CELLS];
        slig_spectrum_weight(weights, set.num_cells, set.cells, spectrum);

        int has_nonzero_weight = 0;
        for (uint32_t i = 0; i < set.num_cells; i++)
            if (weights[i] > 0) has_nonzero_weight = 1;
        check("Spectrum weighting produces non-zero weights", has_nonzero_weight);
    }

    /* ═══ Test 6: Progressive render ═══ */
    printf("\n--- Test 6: Progressive render ---\n");
    {
        SligCellSet set;
        slig_decompose(&set, img_circle, dim, dim);

        uint8_t img1[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];
        uint8_t img5[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];

        slig_render_progressive(img1, set.cells, set.num_cells, 1);
        slig_render_progressive(img5, set.cells, set.num_cells, 5);

        /* rank 5 이미지가 rank 1보다 더 다양해야 함 */
        int unique1 = 0, unique5 = 0;
        uint8_t seen1[256] = {0}, seen5[256] = {0};
        for (int i = 0; i < dim*dim; i++) {
            if (!seen1[img1[i]]) { seen1[img1[i]] = 1; unique1++; }
            if (!seen5[img5[i]]) { seen5[img5[i]] = 1; unique5++; }
        }
        printf("    rank 1: %d unique values\n", unique1);
        printf("    rank 5: %d unique values\n", unique5);
        check("More ranks = more detail", unique5 >= unique1);
    }

    /* ═══ Test 7: Timed render ═══ */
    printf("\n--- Test 7: Timed render ---\n");
    {
        /* 이벤트 CE Cell 수동 구성 */
        CEUnit cells[2];
        SligSignal r1, r2;
        memset(&r1, 0, sizeof(r1));
        memset(&r2, 0, sizeof(r2));

        r1.dir = SLIG_DIR_RIPPLE; r1.sigma = 3000;
        r1.origin_x = 64; r1.origin_y = 64;
        r1.frequency = 10; r1.speed = 3; r1.decay = 220;
        r1.start_tick = 0;

        r2.dir = SLIG_DIR_RIPPLE; r2.sigma = 2000;
        r2.origin_x = 192; r2.origin_y = 192;
        r2.frequency = 8; r2.speed = 3; r2.decay = 220;
        r2.start_tick = 5;

        slig_pack(&cells[0], &r1);
        slig_pack(&cells[1], &r2);

        uint8_t output[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];
        slig_render_timed(output, cells, 2, 20);

        int non_zero = 0;
        for (int i = 0; i < SLIG_CANVAS_DIM * SLIG_CANVAS_DIM; i++)
            if (output[i] > 10 && output[i] < 245) non_zero++;
        check("Timed render produces content", non_zero > 100);
        printf("    Non-trivial pixels: %d\n", non_zero);
    }

    /* ═══ Test 8: Interpolation ═══ */
    printf("\n--- Test 8: CE Cell interpolation ---\n");
    {
        SligCellSet set_c, set_s;
        slig_decompose(&set_c, img_circle, dim, dim);
        slig_decompose(&set_s, img_star, dim, dim);

        CEUnit interp;
        slig_interpolate_cell(&interp, &set_c.cells[0], &set_s.cells[0], 128);

        SligSignal sig;
        slig_unpack(&sig, &interp);
        check("Interpolated cell has valid sigma", sig.sigma > 0);

        /* 연속성: t=0 → A, t=255 → B */
        CEUnit at0, at255;
        slig_interpolate_cell(&at0, &set_c.cells[0], &set_s.cells[0], 0);
        slig_interpolate_cell(&at255, &set_c.cells[0], &set_s.cells[0], 255);
        check("t=0 equals A", ce_distance(&at0, &set_c.cells[0]) < 10);
        check("t=255 equals B", ce_distance(&at255, &set_s.cells[0]) < 10);
    }

    /* ═══ Test 9: Speed ═══ */
    printf("\n--- Test 9: Generation speed ---\n");
    {
        SligCellSet set;
        slig_decompose(&set, img_circle, dim, dim);

        uint8_t output[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];

        int iters = 1000;
        clock_t t0 = clock();
        for (int i = 0; i < iters; i++)
            slig_render(output, set.cells, set.num_cells, NULL);
        double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
        double per_img = elapsed/iters*1000.0;

        printf("    %d iters: %.3f s\n", iters, elapsed);
        printf("    Per image: %.2f ms\n", per_img);
        printf("    Throughput: %.0f img/s\n", iters/elapsed);
        check("Render < 10ms per image", per_img < 10.0);
    }

    /* ═══ Test 10: PPM output ═══ */
    printf("\n--- Test 10: PPM output ---\n");
    {
        SligCellSet set;
        slig_decompose(&set, img_circle, dim, dim);

        uint8_t output[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
        slig_render((uint8_t*)output, set.cells, set.num_cells, NULL);
        int ok = slig_save_ppm("/home/claude/slig_v2_circle.ppm", output);
        check("PPM saved", ok);

        /* Event render */
        slig_decompose(&set, img_star, dim, dim);
        slig_render((uint8_t*)output, set.cells, set.num_cells, NULL);
        ok = slig_save_ppm("/home/claude/slig_v2_star.ppm", output);
        check("Star PPM saved", ok);
    }

    /* ═══ Summary ═══ */
    printf("\n=== Result: %d PASS / %d FAIL ===\n", pass, fail);
    if (fail == 0) {
        printf("\n  CE Cell에 주파수 정보 직접 저장 ✓\n");
        printf("  다방향 (가로+대각+이벤트) 분해/생성 ✓\n");
        printf("  오디오트랙 스펙트럼 가중 ✓\n");
        printf("  시간 틱 누적 ✓\n");
        printf("  보간 ✓\n");
    }

    free(img_circle); free(img_star); free(img_diag);
    return fail;
}
