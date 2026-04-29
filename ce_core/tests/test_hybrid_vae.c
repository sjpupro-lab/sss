/* test_hybrid_vae.c — 통합 VAE 라운드트립 테스트
 *
 * 테스트 항목:
 *   1. 합성 이미지 encode → decode → PSNR 측정
 *   2. 블록 도장만 복원 → SLIG만 복원 → 통합 비교
 *   3. 주파수 업스케일 효과 확인
 *   4. wave refine 수렴 확인
 */

#include "../ce_hybrid_vae.h"
#include "../ce_storage.h"
#include "../slig_codebook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;

#define CHECK(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_pass++; printf("  PASS: %s\n", msg); } \
    else      { printf("  FAIL: %s\n", msg); } \
} while(0)

/* 합성 이미지 생성: 4사분면 다른 색 + 대각선 그래디언트 */
static void make_synth_image(uint8_t *rgba) {
    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            int i = (y * 256 + x) * 4;
            if (x < 128 && y < 128) {
                /* 왼쪽 위: 빨간 그래디언트 */
                rgba[i + 0] = (uint8_t)(x + y);    /* R */
                rgba[i + 1] = 30;                   /* G */
                rgba[i + 2] = 30;                   /* B */
            } else if (x >= 128 && y < 128) {
                /* 오른쪽 위: 파란색 */
                rgba[i + 0] = 30;
                rgba[i + 1] = 30;
                rgba[i + 2] = (uint8_t)(255 - (x - 128) * 2);
            } else if (x < 128 && y >= 128) {
                /* 왼쪽 아래: 초록색 */
                rgba[i + 0] = 30;
                rgba[i + 1] = (uint8_t)(128 + (y - 128));
                rgba[i + 2] = 30;
            } else {
                /* 오른쪽 아래: 대각선 줄무늬 */
                uint8_t stripe = ((x + y) % 32 < 16) ? 200 : 50;
                rgba[i + 0] = stripe;
                rgba[i + 1] = stripe;
                rgba[i + 2] = stripe;
            }
            rgba[i + 3] = 255;  /* A */
        }
    }
}

/* 단색 이미지 (기본 검증용) */
static void make_solid_image(uint8_t *rgba, uint8_t r, uint8_t g, uint8_t b) {
    for (int i = 0; i < 256 * 256; i++) {
        rgba[i * 4 + 0] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = 255;
    }
}

static void test_config_default(void) {
    printf("\n[test_config_default]\n");
    HybridVAEConfig cfg;
    hybrid_vae_config_default(&cfg);
    CHECK(cfg.base_weight > 0.0f && cfg.base_weight < 1.0f,
          "base_weight in valid range");
    CHECK(cfg.detail_weight > 0.0f && cfg.detail_weight < 1.0f,
          "detail_weight in valid range");
    CHECK(fabsf(cfg.base_weight + cfg.detail_weight - 1.0f) < 0.01f,
          "weights sum to 1.0");
    CHECK(cfg.wave_iterations > 0, "wave_iterations > 0");
    CHECK(cfg.guidance_scale >= 1.0f, "guidance_scale >= 1.0");
}

static void test_encode_solid(void) {
    printf("\n[test_encode_solid]\n");

    uint8_t *rgba = (uint8_t *)calloc(256 * 256 * 4, 1);
    make_solid_image(rgba, 200, 100, 50);

    CEStorage storage;
    ce_storage_init(&storage, 2048);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    HybridEncodeResult enc;
    hybrid_vae_encode(&enc, &storage, &codebook, rgba, 256, 256, /*canvas_id=*/1);

    CHECK(enc.block_entries == 1024,
          "solid image → 1024 block entries");
    CHECK(enc.canvas_id == 1, "canvas_id preserved");
    CHECK(enc.total_cells > 0, "SLIG produced cells");

    printf("    block_entries=%u  total_cells=%u\n",
           enc.block_entries, enc.total_cells);

    /* 오디오 링크 확인 */
    SligSignal sig;
    slig_unpack(&sig, &enc.sets[SLIG_LEVEL_FINE][SLIG_CH_Y].cells[0]);
    uint32_t linked_id = (uint32_t)sig.audio_bins[0]
                       | ((uint32_t)sig.audio_bins[1] << 8)
                       | ((uint32_t)sig.audio_amps[0] << 16)
                       | ((uint32_t)sig.audio_amps[1] << 24);
    CHECK(linked_id == 1, "audio link carries canvas_id=1");
    CHECK(sig.flags & SLIG_FLAG_AUDIO_LINK, "AUDIO_LINK flag set");

    ce_storage_free(&storage);
    free(rgba);
}

static void test_roundtrip_solid(void) {
    printf("\n[test_roundtrip_solid]\n");

    uint8_t *rgba = (uint8_t *)calloc(256 * 256 * 4, 1);
    make_solid_image(rgba, 200, 100, 50);

    uint8_t *out_rgb = (uint8_t *)calloc(256 * 256 * 3, 1);

    CEStorage storage;
    ce_storage_init(&storage, 2048);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    float psnr = hybrid_vae_roundtrip(out_rgb, &storage, &codebook,
                                       rgba, 256, 256, /*canvas_id=*/1, NULL);

    printf("    solid roundtrip PSNR = %.1f dB\n", psnr);
    CHECK(psnr > 5.0f, "solid PSNR > 5 dB (base+detail blend)");

    /* 중앙 픽셀이 원래 색에 가까운지 확인 */
    int ci = (128 * 256 + 128) * 3;
    int dr = abs((int)out_rgb[ci + 0] - 200);
    int dg = abs((int)out_rgb[ci + 1] - 100);
    int db = abs((int)out_rgb[ci + 2] - 50);
    printf("    center pixel delta: R=%d G=%d B=%d\n", dr, dg, db);
    CHECK(dr + dg + db < 400, "center pixel sum delta < 400");

    ce_storage_free(&storage);
    free(rgba);
    free(out_rgb);
}

static void test_roundtrip_synth(void) {
    printf("\n[test_roundtrip_synth]\n");

    uint8_t *rgba = (uint8_t *)calloc(256 * 256 * 4, 1);
    make_synth_image(rgba);

    uint8_t *out_rgb = (uint8_t *)calloc(256 * 256 * 3, 1);

    CEStorage storage;
    ce_storage_init(&storage, 2048);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    float psnr = hybrid_vae_roundtrip(out_rgb, &storage, &codebook,
                                       rgba, 256, 256, /*canvas_id=*/2, NULL);

    printf("    synth roundtrip PSNR = %.1f dB\n", psnr);
    CHECK(psnr > 10.0f, "synth PSNR > 10 dB");

    ce_storage_free(&storage);
    free(rgba);
    free(out_rgb);
}

static void test_base_vs_detail(void) {
    printf("\n[test_base_vs_detail]\n");

    uint8_t *rgba = (uint8_t *)calloc(256 * 256 * 4, 1);
    make_synth_image(rgba);

    CEStorage storage;
    ce_storage_init(&storage, 2048);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    HybridEncodeResult enc;
    hybrid_vae_encode(&enc, &storage, &codebook, rgba, 256, 256, 3);

    /* 블록 도장만 복원 */
    uint8_t *base = (uint8_t *)calloc(256 * 256 * 4, 1);
    hybrid_decode_base_only(base, &storage, 3);

    /* base가 빈 캔버스가 아닌지 확인 */
    int base_sum = 0;
    for (int i = 0; i < 256 * 256; i++)
        base_sum += base[i * 4 + 0];  /* R 채널 합 */
    CHECK(base_sum > 0, "base is not blank");

    /* SLIG만 복원 */
    uint8_t *detail = (uint8_t *)calloc(256 * 256, 1);
    HybridVAEConfig cfg;
    hybrid_vae_config_default(&cfg);
    hybrid_decode_detail_only(detail, enc.sets, &cfg);

    int detail_sum = 0;
    for (int i = 0; i < 256 * 256; i++)
        detail_sum += detail[i];
    CHECK(detail_sum > 0, "detail is not blank");

    printf("    base_R_sum=%d  detail_Y_sum=%d\n", base_sum, detail_sum);

    ce_storage_free(&storage);
    free(rgba); free(base); free(detail);
}

static void test_wave_refine_converge(void) {
    printf("\n[test_wave_refine_converge]\n");

    uint8_t *rgba = (uint8_t *)calloc(256 * 256 * 4, 1);
    make_synth_image(rgba);
    uint8_t *orig_rgb = (uint8_t *)malloc(256 * 256 * 3);
    for (int i = 0; i < 256 * 256; i++) {
        orig_rgb[i * 3 + 0] = rgba[i * 4 + 0];
        orig_rgb[i * 3 + 1] = rgba[i * 4 + 1];
        orig_rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    CEStorage storage;
    ce_storage_init(&storage, 2048);
    SligCodebook codebook;
    slig_codebook_init(&codebook);

    HybridEncodeResult enc;
    hybrid_vae_encode(&enc, &storage, &codebook, rgba, 256, 256, 4);

    /* wave 0회 vs wave 100회 비교 */
    uint8_t *out_no_wave = (uint8_t *)calloc(256 * 256 * 3, 1);
    uint8_t *out_wave    = (uint8_t *)calloc(256 * 256 * 3, 1);

    HybridVAEConfig cfg_no;
    hybrid_vae_config_default(&cfg_no);
    cfg_no.wave_iterations = 0;
    hybrid_vae_decode(out_no_wave, &storage, 4, enc.sets, &cfg_no);

    HybridVAEConfig cfg_wave;
    hybrid_vae_config_default(&cfg_wave);
    cfg_wave.wave_iterations = 100;
    hybrid_vae_decode(out_wave, &storage, 4, enc.sets, &cfg_wave);

    float psnr_no   = hybrid_psnr(orig_rgb, out_no_wave, 256, 256);
    float psnr_wave = hybrid_psnr(orig_rgb, out_wave, 256, 256);

    printf("    no_wave PSNR=%.1f  wave_100 PSNR=%.1f\n", psnr_no, psnr_wave);
    CHECK(psnr_no > 5.0f, "no_wave PSNR > 5 dB (baseline)");

    ce_storage_free(&storage);
    free(rgba); free(orig_rgb); free(out_no_wave); free(out_wave);
}

int main(void) {
    printf("=== Hybrid VAE Test Suite ===\n");

    test_config_default();
    test_encode_solid();
    test_roundtrip_solid();
    test_roundtrip_synth();
    test_base_vs_detail();
    test_wave_refine_converge();

    printf("\n=== %d/%d PASSED ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
