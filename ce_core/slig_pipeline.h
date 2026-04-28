/* slig_pipeline.h — SLIG v2 전체 파이프라인 스켈레톤
 *
 * 5개 모듈:
 *   1. 전처리기 (slig_preprocess)
 *   2. 전용 분해기 (slig_decompose_v2) — 5단계 파이프라인
 *   3. 업스케일러 (slig_upscale) — 오디오 복구 기법
 *   4. 렌더러 (slig_render_adaptive) — 애트모스 방식
 *   5. 후처리기 (slig_postprocess)
 *
 * 전체 흐름:
 *   [학습] 이미지 → 전처리 → 분해 → CE Cell 저장
 *   [생성] CE Cell → 업스케일 → 렌더 → 후처리 → 이미지
 */

#ifndef SLIG_PIPELINE_H
#define SLIG_PIPELINE_H

#include "ce_core.h"
#include "slig_signal.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* ════════════════════════════════════════════════════════
 *  공통 구조체
 * ════════════════════════════════════════════════════════ */

/* 이미지 버퍼 (모든 모듈이 공유) */
typedef struct {
    float   *data;       /* 픽셀 데이터 [height × width × channels] */
    int      width;
    int      height;
    int      channels;   /* 1=gray, 3=YCbCr or RGB */
} SligImage;

/* 분해 결과: CE Cell 세트 + 메타데이터 */
typedef struct {
    CEUnit   cells[64];       /* 최대 64개 CE Cell */
    uint32_t num_cells;

    /* 분해 단계별 구간 (어디까지가 구조, 어디부터가 텍스처) */
    uint32_t structure_end;   /* cells[0..structure_end) = 구조 */
    uint32_t edge_end;        /* cells[structure_end..edge_end) = 엣지 */
    uint32_t texture_end;     /* cells[edge_end..texture_end) = 텍스처 */
    uint32_t color_end;       /* cells[texture_end..color_end) = 색상 */
    /* cells[color_end..num_cells) = 이벤트 (특이점) */

    /* 전처리 메타 (후처리에서 역변환 시 필요) */
    float    orig_min;        /* 원본 히스토그램 최소 */
    float    orig_max;        /* 원본 히스토그램 최대 */
    float    orig_mean;       /* 원본 평균 */
    uint8_t  symmetry;        /* 0=없음, 1=좌우, 2=상하, 3=4-fold */
    float    sym_axis_x;      /* 대칭축 위치 */
    float    sym_axis_y;
} SligDecomposed;

/* 렌더 설정 (애트모스 방식) */
typedef struct {
    int      target_width;    /* 출력 해상도 (자유) */
    int      target_height;
    float    guidance_scale;  /* 조건부 강조 (1.0=기본, 2.0=강조) */
    float    sigma_scale;     /* 다이내믹 레인지 조정 (1.0=기본) */
    uint8_t  depth_enable;    /* 깊이 표현 (전경 강하게, 배경 부드럽게) */
    uint8_t  progressive;     /* 1이면 콜백으로 중간 결과 전달 */

    /* 프로그레시브 콜백 (NULL이면 무시) */
    void (*on_progress)(const SligImage *partial, int step, int total, void *ctx);
    void *progress_ctx;
} SligRenderConfig;


/* ════════════════════════════════════════════════════════
 *
 *  1. 전처리기
 *
 *  이미지를 분해하기 전에 CE Cell 효율을 최대화하는 정리.
 *  "페인트칠 전에 벽을 사포질하는 것."
 *
 * ════════════════════════════════════════════════════════ */

typedef struct {
    float    noise_threshold;  /* 이 이하의 변화는 노이즈로 간주 (기본 0.02) */
    float    edge_sigma;       /* 엣지 보존 블러 강도 (기본 1.5) */
    uint8_t  detect_symmetry;  /* 대칭 검출 켜기 (기본 1) */
    uint8_t  detect_repeat;    /* 반복 패턴 검출 켜기 (기본 1) */
} SligPreprocessConfig;

/* 전처리 수행
 *
 * 1. 히스토그램 정규화
 *    왜: 어두운 이미지는 σ 범위의 일부만 사용 → 낭비
 *    방법: 실제 밝기를 0.0~1.0 전체에 펼침
 *    효과: CE Cell의 σ가 전 범위를 활용 → 정밀도 향상
 *
 * 2. 노이즈 제거
 *    왜: 카메라 노이즈는 의미 없는데 CE Cell을 잡아먹음
 *    방법: 위너 필터 (신호/노이즈 비율 추정)
 *          — 오디오 녹음에서 히스 노이즈 제거하는 것과 같은 원리
 *    효과: CE Cell이 진짜 구조에만 집중
 *
 * 3. 엣지 보존 블러
 *    왜: 의미 없는 미세 변화가 고주파 CE Cell을 낭비
 *    방법: 바이래터럴 필터 (가까운 밝기끼리만 블러)
 *          — 엣지는 살리고 평탄한 곳만 부드럽게
 *    효과: 중요한 경계선에 CE Cell 집중
 *
 * 4. 대칭/반복 검출
 *    왜: 얼굴은 좌우 대칭 → 절반만 분해하면 CE Cell 절반
 *    방법: 이미지를 반으로 접어서 차이 측정
 *          차이가 임계값 이하면 대칭으로 판정
 *    효과: CE Cell 수 최대 75% 절약 (4-fold 대칭 시)
 *
 * in/out이 같은 포인터일 수 있음 (in-place 가능).
 * meta에 역변환 정보 저장 → 후처리에서 사용. */
void slig_preprocess(SligImage *out,
                     SligDecomposed *meta,
                     const SligImage *in,
                     const SligPreprocessConfig *cfg);

/* 기본 설정 */
void slig_preprocess_config_default(SligPreprocessConfig *cfg);


/* ════════════════════════════════════════════════════════
 *
 *  2. 전용 분해기 (5단계 파이프라인)
 *
 *  SVD 하나로 다 하던 것을 → 각 단계가 잘하는 것만 담당.
 *  "망치 하나 대신, 공구함 전체를 쓰는 것."
 *
 * ════════════════════════════════════════════════════════ */

typedef struct {
    /* 각 단계 CE Cell 할당량 */
    int max_structure_cells;   /* 구조: SVD (기본 8) */
    int max_edge_cells;        /* 엣지: 가보르/소벨 (기본 8) */
    int max_texture_cells;     /* 텍스처: 웨이블릿 (기본 8) */
    int max_color_cells;       /* 색상: YCbCr 분리 (기본 6) */
    int max_event_cells;       /* 이벤트: 특이점 (기본 4) */

    float energy_target;       /* 에너지 커버리지 목표 (기본 0.95) */
} SligDecomposeConfig;

/* 5단계 분해 수행
 *
 * Stage 1: 구조 분해 (SVD)
 *   입력: 전처리된 이미지
 *   방법: 전체 이미지에 SVD → 에너지 큰 순서로 CE Cell
 *   잡는 것: 큰 윤곽, 밝기 분포, 전체 형태
 *   CE Cell에 저장: dir=HORIZONTAL, scale=COARSE
 *   왜 SVD: 전역 구조를 최소 개수로 잡는 데 최적
 *
 * Stage 2: 엣지 분해 (방향별 가보르 필터)
 *   입력: Stage 1 잔차 (= 원본 - Stage 1 복원)
 *   방법: 4방향(0°, 45°, 90°, 135°) 가보르 필터 적용
 *         각 방향의 반응이 강한 곳 → CE Cell
 *   잡는 것: 경계선, 윤곽선, 방향성 있는 구조
 *   CE Cell에 저장: dir=HORIZONTAL/DIAG_DOWN/VERTICAL/DIAG_UP
 *   왜 가보르: 방향+주파수+위치를 동시에 잡음
 *             인간 시각 피질이 이 방식을 씀
 *
 * Stage 3: 텍스처 분해 (웨이블릿)
 *   입력: Stage 2 잔차
 *   방법: 블록별 웨이블릿 변환
 *         = "이 8×8 영역의 질감이 뭔지"
 *         = 호시가 말한 "4칸→2칸→1칸 진동"
 *   잡는 것: 반복 패턴, 질감, 미세 구조
 *   CE Cell에 저장: scale=COARSE/MID/FINE/PIXEL
 *   왜 웨이블릿: 위치별로 다른 주파수를 잡을 수 있음
 *               (SVD는 전역만, 웨이블릿은 국소도 가능)
 *
 * Stage 4: 색상 분해 (YCbCr 분리)
 *   입력: 원본 컬러 이미지
 *   방법: RGB → YCbCr 변환
 *         Y(밝기)는 Stage 1~3에서 이미 처리됨
 *         Cb(파랑), Cr(빨강)만 여기서 처리
 *         인간 눈이 색에 둔감 → 적은 CE Cell로 충분
 *   잡는 것: 색 분포 (형태 무관)
 *   CE Cell에 저장: 별도 channel 태그
 *   왜 분리: 밝기에 80%, 색에 20% → 효율적 배분
 *
 * Stage 5: 이벤트 분해 (특이점 검출)
 *   입력: 원본 이미지
 *   방법: 밝기 무게중심 → 물결 CE Cell
 *         대칭축 → 십자 CE Cell
 *         극값(가장 밝은/어두운 점) → 빔 CE Cell
 *   잡는 것: "이 이미지의 초점이 어디인지"
 *   CE Cell에 저장: dir=RIPPLE/BEAM/CROSS
 *   왜 이벤트: 좌표 기반 신호가 국소적 강조를 가능하게 함 */
void slig_decompose_v2(SligDecomposed *out,
                       const SligImage *image,
                       const SligDecomposeConfig *cfg);

/* 기본 설정 */
void slig_decompose_config_default(SligDecomposeConfig *cfg);

/* 개별 단계 (디버그/커스텀 파이프라인용) */
void slig_decompose_structure(SligDecomposed *out, const SligImage *img,
                              int max_cells, float energy_target);
void slig_decompose_edges(SligDecomposed *out, const SligImage *img,
                          const SligImage *residual, int max_cells);
void slig_decompose_texture(SligDecomposed *out, const SligImage *img,
                            const SligImage *residual, int max_cells);
void slig_decompose_color(SligDecomposed *out, const SligImage *img,
                          int max_cells);
void slig_decompose_events(SligDecomposed *out, const SligImage *img,
                           int max_cells);


/* ════════════════════════════════════════════════════════
 *
 *  3. 업스케일러 (오디오 복구 기법 응용)
 *
 *  CE Cell의 16개 DCT 계수에서 32~48개를 "예측".
 *  MP3에서 잘린 고음을 복원하는 것과 같은 원리.
 *  추가 데이터 없이, 수학만으로.
 *
 * ════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t  enable_harmonic;   /* 하모닉 확장 (기본 1) */
    uint8_t  enable_sbr;        /* 대역폭 복제 (기본 1) */
    uint8_t  enable_wiener;     /* 위너 필터 (기본 1) */
    uint8_t  enable_cs;         /* 압축 센싱 (기본 0, 느림) */
    int      target_coeffs;     /* 목표 DCT 계수 수 (기본 32) */
    int      cs_iterations;     /* 압축 센싱 반복 수 (기본 50) */
} SligUpscaleConfig;

/* CE Cell 세트의 신호 해상도를 높인다.
 *
 * 1. 하모닉 확장 (Harmonic Extension)
 *    원리: 기본 주파수 k가 있으면, 2k/3k에도 에너지가 있을 확률 높음
 *    방법: DCT 계수 k의 값에서 2k/3k를 추정
 *          추정값 = k의 값 × 감쇠비 (보통 0.3~0.5)
 *    오디오 비유: 기타 줄의 기본음이 있으면 배음이 자동으로 있듯이
 *    효과: 16계수 → 32계수로 확장 (부드러운 고주파 추가)
 *
 * 2. 대역폭 복제 — SBR (Spectral Band Replication)
 *    원리: 저주파 패턴이 고주파에서 반복되는 경향
 *    방법: 저주파 계수 0~15의 "모양"을 고주파 16~31에 복사
 *          스케일만 줄임 (고주파는 에너지가 작으니까)
 *    오디오 비유: MP3의 SBR이 정확히 이것. AAC-HE가 쓰는 기법.
 *    효과: 텍스처 느낌이 살아남 (날카로운 엣지 개선)
 *
 * 3. 위너 필터 (Wiener Filter)
 *    원리: "진짜 신호 대 노이즈" 비율을 추정해서 최적 복원
 *    방법: 현재 계수의 파워 스펙트럼 → 신호+노이즈 모델
 *          모델에서 노이즈를 빼면 "예상 원본 신호"
 *    오디오 비유: 녹음에서 배경 잡음을 제거하는 표준 기법
 *    효과: wave refine과 결합하면 수렴 속도 향상
 *    SSS 연결: ce_image_wave_refine의 파동 스텝과 동일 철학
 *
 * 4. 압축 센싱 (Compressed Sensing) — 선택적
 *    원리: "자연 이미지의 주파수 분포는 sparse하다"는 가정
 *          16개 관측에서 64개를 복원할 수 있음 (수학적 보장)
 *    방법: L1 정규화 최적화 (반복적)
 *    오디오 비유: MRI에서 적은 스캔으로 전체 영상 복원하는 기법
 *    효과: 가장 정밀하지만 계산 비용 높음
 *    주의: 느리므로 기본 OFF. 고품질 필요 시에만 켜기. */
void slig_upscale(SligDecomposed *decomposed,
                  const SligUpscaleConfig *cfg);

/* 기본 설정 */
void slig_upscale_config_default(SligUpscaleConfig *cfg);

/* 개별 기법 (디버그/비교용) */
void slig_upscale_harmonic(CEUnit *cell, int target_coeffs);
void slig_upscale_sbr(CEUnit *cell, int target_coeffs);
void slig_upscale_wiener(CEUnit *cell);
void slig_upscale_cs(CEUnit *cell, int target_coeffs, int iterations);


/* ════════════════════════════════════════════════════════
 *
 *  4. 적응형 렌더러 (돌비 애트모스 방식)
 *
 *  CE Cell을 "신호 오브젝트"로 취급.
 *  해상도에 독립적. 같은 CE Cell로 64×64도 4K도 렌더.
 *  "콘텐츠는 건드리지 않고 렌더러만 바꾸는" 구조.
 *
 * ════════════════════════════════════════════════════════ */

/* 적응형 렌더링 수행
 *
 * 돌비 애트모스와의 대응:
 *
 * 1. 오브젝트 기반 렌더링
 *    애트모스: 소리 오브젝트를 스피커 수에 맞게 배치
 *    SLIG: CE Cell 오브젝트를 타겟 해상도에 맞게 배치
 *
 *    구현:
 *      CE Cell의 좌표를 0.0~1.0 정규화로 저장
 *      렌더 시 target_width/height에 맞게 스케일
 *      origin_x=128 (256 기준) → 0.5 → 타겟의 중앙
 *      → 64×64에서도 1024×1024에서도 같은 위치
 *
 * 2. 채널 베드 + 오브젝트 믹스
 *    애트모스: 기본 7.1 채널(배경) + 개별 소리 오브젝트(효과)
 *    SLIG: 전역 신호(배경) + 이벤트 신호(국소 효과)
 *
 *    구현:
 *      Phase 1: structure + edge CE Cell → 전역 신호로 캔버스 전체에 적용
 *               = 채널 베드 (어디서나 기본으로 깔리는 것)
 *      Phase 2: event CE Cell → 좌표 기반으로 국소 적용
 *               = 오브젝트 (특정 위치에 올리는 것)
 *      Phase 3: texture CE Cell → 영역별 텍스처 적용
 *               = 환경 효과 (공간감, 질감)
 *
 * 3. 다이내믹 레인지 관리
 *    애트모스: 큰 극장에서는 넓은 다이내믹 레인지
 *             작은 스피커에서는 압축해서 작아도 잘 들리게
 *    SLIG: 큰 모니터에서는 σ 레인지 그대로
 *          모바일에서는 σ 압축해서 작아도 잘 보이게
 *
 *    구현:
 *      sigma_scale < 1.0 → 대비 줄임 (모바일)
 *      sigma_scale = 1.0 → 원본 대비 (데스크탑)
 *      sigma_scale > 1.0 → 대비 높임 (프린트/HDR)
 *
 * 4. 깊이 표현 (바이노럴 렌더링 응용)
 *    애트모스: 헤드폰 2채널로 3D 공간감
 *    SLIG: 2D 캔버스에서 깊이감
 *
 *    구현:
 *      depth_enable=1이면:
 *        structure CE Cell (전경) → σ 유지, 신호 날카롭게
 *        texture CE Cell (배경) → σ 줄임, 신호 부드럽게
 *        → 전경은 선명, 배경은 흐림 = 깊이감
 *
 * 5. 메타데이터 기반 적응
 *    애트모스: "이건 대사다" → 대사 볼륨 자동 조절
 *    SLIG: CE Cell에 저장된 메타로 자동 조절
 *
 *    구현:
 *      inc.A의 에너지 프로파일 → 스케일별 중요도
 *      dec.B의 트리거 정보 → 조건부 적용
 *      형태소 POS 태그(외부) → 명사=형태 강조, 형용사=속성 강조
 *
 * 6. 프로그레시브 스트리밍
 *    애트모스: 먼저 기본 채널, 그 위에 오브젝트 추가
 *    SLIG: 먼저 구조, 그 위에 엣지, 텍스처, 이벤트 추가
 *
 *    구현:
 *      progressive=1이면 on_progress 콜백으로 중간 결과 전달
 *      Step 1: structure만 렌더 → 콜백 (큰 형태)
 *      Step 2: + edges → 콜백 (윤곽 선명)
 *      Step 3: + texture → 콜백 (질감 추가)
 *      Step 4: + color → 콜백 (색 입힘)
 *      Step 5: + events → 콜백 (최종)
 *      → 클라이언트는 받는 족족 표시 가능 */
void slig_render_adaptive(SligImage *out,
                          const SligDecomposed *decomposed,
                          const SligRenderConfig *cfg);

/* 기본 설정 */
void slig_render_config_default(SligRenderConfig *cfg);

/* 개별 페이즈 렌더 (디버그용) */
void slig_render_phase_structure(SligCanvas *c, const SligDecomposed *d,
                                 int target_w, int target_h);
void slig_render_phase_edges(SligCanvas *c, const SligDecomposed *d,
                             int target_w, int target_h);
void slig_render_phase_texture(SligCanvas *c, const SligDecomposed *d,
                               int target_w, int target_h);
void slig_render_phase_color(SligCanvas *c, const SligDecomposed *d,
                             int target_w, int target_h);
void slig_render_phase_events(SligCanvas *c, const SligDecomposed *d,
                              int target_w, int target_h, uint8_t max_tick);


/* ════════════════════════════════════════════════════════
 *
 *  5. 후처리기
 *
 *  렌더된 이미지의 최종 품질 보정.
 *  "액자에 넣기 전 마지막 손질."
 *
 * ════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t  enable_sharpen;    /* 언샤프 마스크 (기본 1) */
    float    sharpen_amount;    /* 선명화 강도 (기본 0.5) */
    float    sharpen_radius;    /* 선명화 범위 (기본 1.0) */

    uint8_t  enable_debanding;  /* 밴딩 제거 (기본 1) */

    uint8_t  enable_color_correct; /* 컬러 보정 (기본 1) */
    float    saturation;        /* 채도 조절 (1.0=기본, 1.5=선명) */
    float    brightness;        /* 밝기 조절 (0.0=기본 오프셋) */
    float    contrast;          /* 대비 조절 (1.0=기본) */

    uint8_t  enable_histogram_match; /* 히스토그램 매칭 (기본 0) */
} SligPostprocessConfig;

/* 후처리 수행
 *
 * 1. 언샤프 마스크 (선명화)
 *    왜: DCT 압축으로 엣지가 뭉개졌을 수 있음
 *    방법: 블러 버전을 만들고, (원본 - 블러) × 강도를 원본에 더함
 *    오디오 비유: 이퀄라이저에서 고음을 올리는 것
 *    효과: 경계선이 살아남
 *
 * 2. 밴딩 제거
 *    왜: DCT가 블록 단위라서 블록 경계에 줄이 보일 수 있음
 *       (JPEG 저화질에서 보이는 네모난 자국과 같음)
 *    방법: 블록 경계에만 약한 블러 적용
 *    효과: 부드러운 그라데이션 복원
 *
 * 3. 컬러 보정
 *    왜: YCbCr→RGB 변환 후 채도가 빠질 수 있음
 *    방법: HSV 공간에서 채도/밝기/대비 조절
 *    효과: 생기 있는 색감
 *
 * 4. 히스토그램 매칭 (선택)
 *    왜: 전처리에서 정규화했던 것을 원래 밝기 분포로 되돌림
 *    방법: meta.orig_min/max/mean 사용
 *    효과: 원본과 밝기 느낌이 같아짐 */
void slig_postprocess(SligImage *out,
                      const SligImage *in,
                      const SligDecomposed *meta,
                      const SligPostprocessConfig *cfg);

/* 기본 설정 */
void slig_postprocess_config_default(SligPostprocessConfig *cfg);


/* ════════════════════════════════════════════════════════
 *
 *  6. 통합 API (원샷)
 *
 *  전체 파이프라인을 한 번에 실행.
 *
 * ════════════════════════════════════════════════════════ */

/* [학습] 이미지 → 전처리 → 분해 → CE Cell 세트 반환
 *
 * 호출 후 out->cells를 키프레임에 저장하면 학습 완료. */
void slig_learn_image(SligDecomposed *out,
                      const uint8_t *image_data,
                      int width, int height, int channels);

/* [생성] CE Cell 세트 → 업스케일 → 렌더 → 후처리 → 이미지 반환
 *
 * 형태소별 CE Cell이 이미 모아진 상태에서 호출.
 * MorphemeMatch[] → cells 합산 → 여기에 전달. */
void slig_generate_image(uint8_t *out_image,
                         int target_width, int target_height,
                         const SligDecomposed *decomposed,
                         float guidance_scale);


/* ════════════════════════════════════════════════════════
 *
 *  7. 유틸리티
 *
 * ════════════════════════════════════════════════════════ */

/* SligImage 할당/해제 */
SligImage* slig_image_alloc(int width, int height, int channels);
void       slig_image_free(SligImage *img);

/* 잔차 계산: residual = original - reconstructed */
void slig_compute_residual(SligImage *residual,
                           const SligImage *original,
                           const SligDecomposed *decomposed);

/* PPM 저장 */
int slig_image_save_ppm(const char *path, const SligImage *img);

/* CE Cell 세트의 복원 품질 측정 (PSNR) */
float slig_measure_psnr(const SligImage *original,
                        const SligDecomposed *decomposed);


#ifdef __cplusplus
}
#endif
#endif /* SLIG_PIPELINE_H */
