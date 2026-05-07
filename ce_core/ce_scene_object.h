/* ce_scene_object.h — Atmos-style Scene Object Layer for SSS
 *
 * 핵심 개념:
 *   이미지 = 음향공간 (soundfield)
 *   오브젝트 = 주파수 시그니처 + 2D 위치
 *   이동 = 위치 보간 (smooth pan / instant cut)
 *   전환 = 파형 파라미터 변경
 *
 * Dolby Atmos 대응:
 *   Audio Object  →  CESceneObject
 *   Frequency     →  CEWaveObject.freq/amp/phase
 *   3D Position   →  (cx, cy) 정규화 좌표
 *   Bed Channel   →  배경 오브젝트 (전체 캔버스)
 *   Object Track  →  이동 경로 (keyframe 배열)
 *
 * CEWaveObject (12B)는 개별 파형.
 * CESceneObject는 복수 파형을 묶어 하나의 시각 개체로 만든다.
 * 렌더링은 기존 ce_wave_render를 그대로 사용 — additive blending.
 */
#ifndef CE_SCENE_OBJECT_H
#define CE_SCENE_OBJECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 최대값 상수 ── */
#define CE_SCENE_MAX_WAVES    8   /* 오브젝트당 최대 파형 수 */
#define CE_SCENE_MAX_OBJECTS  32  /* 씬당 최대 오브젝트 수 */
#define CE_SCENE_MAX_KEYFRAMES 64 /* 이동 경로 최대 키프레임 */

/* ── 이동 모드 ── */
typedef enum {
    CE_MOVE_SMOOTH = 0,   /* 자연스러운 보간 (lerp) */
    CE_MOVE_INSTANT = 1,  /* 즉시 전환 (cut) */
    CE_MOVE_EASE_IN = 2,  /* 가속 시작 */
    CE_MOVE_EASE_OUT = 3, /* 감속 종료 */
} CEMoveMode;

/* ── 파형 파라미터 (CEWaveObject 경량 버전) ── */
typedef struct {
    uint8_t freq;       /* 주파수 (1~64) */
    uint8_t amp;        /* 진폭 (0~255) */
    uint8_t phase;      /* 위상 (0~255, 256분할) */
    uint8_t waveform;   /* 0=sin, 1=tri, 2=saw, 3=square */
} CEWaveParam;

/* ── 위치 키프레임 ── */
typedef struct {
    uint32_t tick;      /* 이 키프레임의 틱 시점 */
    uint8_t cx;         /* X 위치 (0~255, 정규화) */
    uint8_t cy;         /* Y 위치 (0~255, 정규화) */
    uint8_t move_mode;  /* CEMoveMode */
    uint8_t reserved;
} CEPositionKey;

/* ── 씬 오브젝트: Atmos 오디오 오브젝트에 대응 ──
 * struct tag `CESceneObject_s` is exposed so other ce_core headers
 * (notably ce_storage.h's CE_TYPE_SCENE API) can forward-declare it
 * without pulling in this whole header. */
typedef struct CESceneObject_s {
    /* 식별 */
    uint16_t id;
    uint16_t group_id;      /* 연관 오브젝트 그룹 (0=없음) */

    /* 주파수 시그니처 — 이 오브젝트의 "소리" */
    CEWaveParam waves[CE_SCENE_MAX_WAVES];
    uint8_t wave_count;

    /* 현재 상태 */
    uint8_t cx;             /* 현재 X (0~255) */
    uint8_t cy;             /* 현재 Y (0~255) */
    uint8_t radius;         /* 영향 반경 (0~255) */

    /* 색상 (렌더 시 파형에 곱해짐) */
    uint8_t color_r;
    uint8_t color_g;
    uint8_t color_b;
    uint8_t opacity;        /* 0~255 */

    /* 이동 경로 */
    CEPositionKey keyframes[CE_SCENE_MAX_KEYFRAMES];
    uint16_t keyframe_count;
    uint16_t current_key;   /* 현재 키프레임 인덱스 */

    /* 이동 보간 상태 (내부용) */
    int16_t interp_cx_256;  /* 고정소수점 현재 X (8.8) */
    int16_t interp_cy_256;  /* 고정소수점 현재 Y (8.8) */
} CESceneObject;

/* ── 씬: 전체 음향공간 ── */
typedef struct {
    CESceneObject objects[CE_SCENE_MAX_OBJECTS];
    uint16_t object_count;
    uint32_t current_tick;

    /* 글로벌 파라미터 */
    uint8_t master_amp;     /* 전체 진폭 스케일 (0~255, 128=1.0x) */
    uint8_t reserved[3];
} CEScene;


/* ═══════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════ */

/* 초기화 */
void ce_scene_init(CEScene *scene);

/* 오브젝트 추가. 성공 시 오브젝트 포인터 반환, 꽉 차면 NULL */
CESceneObject* ce_scene_add_object(CEScene *scene, uint16_t id);

/* 오브젝트에 파형 추가 */
int ce_scene_add_wave(CESceneObject *obj,
                      uint8_t freq, uint8_t amp,
                      uint8_t phase, uint8_t waveform);

/* 키프레임 추가 (이동 경로) */
int ce_scene_add_keyframe(CESceneObject *obj,
                          uint32_t tick, uint8_t cx, uint8_t cy,
                          CEMoveMode mode);

/* ── 틱 진행 ── */

/* 1틱 진행: 모든 오브젝트의 위치를 키프레임에 따라 보간 */
void ce_scene_tick(CEScene *scene);

/* N틱 한번에 진행 */
void ce_scene_advance(CEScene *scene, uint32_t n_ticks);

/* 특정 틱으로 점프 (seek) */
void ce_scene_seek(CEScene *scene, uint32_t target_tick);

/* ── 즉석 이동 (키프레임 없이) ── */

/* 오브젝트를 목표 위치로 부드럽게 이동 시작 */
void ce_scene_move_smooth(CESceneObject *obj,
                          uint8_t target_cx, uint8_t target_cy,
                          uint32_t duration_ticks);

/* 오브젝트를 즉시 이동 */
void ce_scene_move_instant(CESceneObject *obj,
                           uint8_t target_cx, uint8_t target_cy);

/* ── 렌더링 ── */

/* 씬의 현재 상태를 RGB 버퍼에 렌더.
 * rgb: width * height * 3 uint8 배열 (caller가 할당/초기화)
 * 내부적으로 int32 accumulation → clamp → uint8 변환.
 * 모든 오브젝트는 additive blending. */
void ce_scene_render(const CEScene *scene,
                     uint8_t *rgb, int width, int height);

/* 단일 오브젝트만 int32 accumulation 버퍼에 렌더 (내부용이지만 노출) */
void ce_scene_render_object(const CESceneObject *obj,
                            int32_t *acc, int width, int height,
                            uint32_t tick);

/* ── 조회 ── */

/* ID로 오브젝트 찾기. 없으면 NULL */
CESceneObject* ce_scene_find(CEScene *scene, uint16_t id);

/* 그룹 ID로 오브젝트들 찾기. 반환값 = 찾은 개수 */
int ce_scene_find_group(const CEScene *scene, uint16_t group_id,
                        CESceneObject **out, int max_out);

#ifdef __cplusplus
}
#endif
#endif /* CE_SCENE_OBJECT_H */
