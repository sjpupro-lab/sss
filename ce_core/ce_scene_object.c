/* ce_scene_object.c — Atmos-style Scene Object Engine
 *
 * 정수 연산만 사용. float 없음.
 * 좌표는 0~255 정규화 → 렌더 시 캔버스 해상도로 매핑.
 * 보간은 8.8 고정소수점.
 * 파형 평가는 slig_tick_math의 SIN_TABLE 사용.
 */

#include "ce_scene_object.h"
#include <string.h>
#include <stdlib.h>

/* ── 틱 sin 테이블 (slig_tick_math에서 가져오거나 자체 생성) ── */
#ifndef TICK_SIN_TABLE_SIZE
#define TICK_SIN_TABLE_SIZE 256
static int8_t g_sin_lut[TICK_SIN_TABLE_SIZE];
static int g_sin_lut_init = 0;

static void ensure_sin_lut(void) {
    if (g_sin_lut_init) return;
    for (int i = 0; i < 256; i++) {
        /* sin(i/256 * 2pi) * 127, 정수 근사 */
        /* 3차 다항식 근사: 충분히 정확하고 float 불필요 */
        int x = i;
        int quadrant = x >> 6;          /* 0~3 */
        int pos = (x & 63) << 2;        /* 0~252 */
        int half;
        switch (quadrant) {
            case 0: half = pos; break;
            case 1: half = 252 - pos; break;
            case 2: half = -pos; break;
            default: half = -(252 - pos); break;
        }
        /* 스케일: [-252, 252] → [-127, 127] */
        g_sin_lut[i] = (int8_t)((half * 127) / 252);
    }
    g_sin_lut_init = 1;
}
#endif

/* ── 유틸 ── */
static inline int iabs(int v) { return v < 0 ? -v : v; }
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int iclamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* 정수 제곱근 (Babylonian) */
static uint32_t isqrt(uint32_t n) {
    if (n == 0) return 0;
    uint32_t x = n, y = (x + 1) >> 1;
    while (y < x) { x = y; y = (x + n / x) >> 1; }
    return x;
}

/* 고정소수점 8.8 보간 */
static inline int16_t lerp88(int16_t a, int16_t b, uint8_t t) {
    /* t: 0=a, 255=b */
    return (int16_t)(a + (((int32_t)(b - a) * t) >> 8));
}

/* ═══════════════════════════════════════════════════════════════════
 * 초기화
 * ═══════════════════════════════════════════════════════════════════ */

void ce_scene_init(CEScene *scene) {
    memset(scene, 0, sizeof(CEScene));
    scene->master_amp = 128; /* 1.0x */
    ensure_sin_lut();
}

CESceneObject* ce_scene_add_object(CEScene *scene, uint16_t id) {
    if (scene->object_count >= CE_SCENE_MAX_OBJECTS) return NULL;
    CESceneObject *obj = &scene->objects[scene->object_count++];
    memset(obj, 0, sizeof(CESceneObject));
    obj->id = id;
    obj->opacity = 255;
    obj->radius = 30;  /* 기본 반경 */
    return obj;
}

int ce_scene_add_wave(CESceneObject *obj,
                      uint8_t freq, uint8_t amp,
                      uint8_t phase, uint8_t waveform) {
    if (obj->wave_count >= CE_SCENE_MAX_WAVES) return -1;
    CEWaveParam *w = &obj->waves[obj->wave_count++];
    w->freq = freq;
    w->amp = amp;
    w->phase = phase;
    w->waveform = waveform;
    return 0;
}

int ce_scene_add_keyframe(CESceneObject *obj,
                          uint32_t tick, uint8_t cx, uint8_t cy,
                          CEMoveMode mode) {
    if (obj->keyframe_count >= CE_SCENE_MAX_KEYFRAMES) return -1;
    CEPositionKey *k = &obj->keyframes[obj->keyframe_count++];
    k->tick = tick;
    k->cx = cx;
    k->cy = cy;
    k->move_mode = (uint8_t)mode;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * 틱 & 보간
 * ═══════════════════════════════════════════════════════════════════ */

/* 단일 오브젝트의 위치를 현재 틱에 맞게 보간 */
static void interpolate_object(CESceneObject *obj, uint32_t tick) {
    if (obj->keyframe_count == 0) return;

    /* 현재 틱이 속하는 키프레임 구간 찾기 */
    uint16_t ki = 0;
    for (uint16_t i = 0; i < obj->keyframe_count - 1; i++) {
        if (tick >= obj->keyframes[i].tick &&
            tick <  obj->keyframes[i + 1].tick) {
            ki = i;
            break;
        }
        if (i == obj->keyframe_count - 2) {
            ki = i; /* 마지막 구간 */
        }
    }

    const CEPositionKey *k0 = &obj->keyframes[ki];
    const CEPositionKey *k1 = (ki + 1 < obj->keyframe_count)
                              ? &obj->keyframes[ki + 1] : k0;

    if (k0 == k1 || tick >= k1->tick) {
        /* 키프레임 끝을 넘었거나 단일 키프레임 */
        obj->cx = k1->cx;
        obj->cy = k1->cy;
        obj->interp_cx_256 = (int16_t)(k1->cx) << 8;
        obj->interp_cy_256 = (int16_t)(k1->cy) << 8;
        return;
    }

    /* 보간 계수 t (0~255) */
    uint32_t span = k1->tick - k0->tick;
    uint32_t elapsed = tick - k0->tick;
    uint8_t t = (uint8_t)((elapsed * 255) / span);

    /* 이동 모드에 따른 이징 */
    switch ((CEMoveMode)k0->move_mode) {
        case CE_MOVE_INSTANT:
            t = 255; /* 즉시 */
            break;
        case CE_MOVE_EASE_IN:
            t = (uint8_t)(((uint32_t)t * t) >> 8);
            break;
        case CE_MOVE_EASE_OUT:
            t = (uint8_t)(255 - (((uint32_t)(255 - t) * (255 - t)) >> 8));
            break;
        case CE_MOVE_SMOOTH:
        default:
            break; /* 선형 */
    }

    int16_t cx0_88 = (int16_t)(k0->cx) << 8;
    int16_t cy0_88 = (int16_t)(k0->cy) << 8;
    int16_t cx1_88 = (int16_t)(k1->cx) << 8;
    int16_t cy1_88 = (int16_t)(k1->cy) << 8;

    obj->interp_cx_256 = lerp88(cx0_88, cx1_88, t);
    obj->interp_cy_256 = lerp88(cy0_88, cy1_88, t);
    obj->cx = (uint8_t)(obj->interp_cx_256 >> 8);
    obj->cy = (uint8_t)(obj->interp_cy_256 >> 8);
}

void ce_scene_tick(CEScene *scene) {
    scene->current_tick++;
    for (uint16_t i = 0; i < scene->object_count; i++) {
        interpolate_object(&scene->objects[i], scene->current_tick);
    }
}

void ce_scene_advance(CEScene *scene, uint32_t n_ticks) {
    for (uint32_t i = 0; i < n_ticks; i++) {
        ce_scene_tick(scene);
    }
}

void ce_scene_seek(CEScene *scene, uint32_t target_tick) {
    scene->current_tick = target_tick;
    for (uint16_t i = 0; i < scene->object_count; i++) {
        interpolate_object(&scene->objects[i], target_tick);
    }
}

/* ── 즉석 이동 (키프레임을 동적 추가) ── */

void ce_scene_move_smooth(CESceneObject *obj,
                          uint8_t target_cx, uint8_t target_cy,
                          uint32_t duration_ticks) {
    /* 현재 위치를 키프레임 0으로, 목표를 키프레임 1로 설정 */
    obj->keyframe_count = 0;
    obj->current_key = 0;
    /* 시작: 현재 위치 */
    ce_scene_add_keyframe(obj, 0, obj->cx, obj->cy, CE_MOVE_SMOOTH);
    /* 끝: 목표 위치 */
    ce_scene_add_keyframe(obj, duration_ticks, target_cx, target_cy,
                          CE_MOVE_SMOOTH);
}

void ce_scene_move_instant(CESceneObject *obj,
                           uint8_t target_cx, uint8_t target_cy) {
    obj->cx = target_cx;
    obj->cy = target_cy;
    obj->interp_cx_256 = (int16_t)(target_cx) << 8;
    obj->interp_cy_256 = (int16_t)(target_cy) << 8;
    obj->keyframe_count = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * 렌더링 — additive, 정수 전용
 * ═══════════════════════════════════════════════════════════════════ */

/* 파형 평가: 위치 pos(0~255)에서의 파형 값 반환 (signed, ~[-128,127]) */
static inline int eval_wave(const CEWaveParam *w, int pos, uint32_t tick) {
    int p = (pos * w->freq + w->phase + (int)(tick * w->freq / 4)) & 0xFF;

    int val;
    switch (w->waveform) {
        case 1: /* triangle */
            val = (p < 128) ? (p * 2 - 128) : (384 - p * 2 - 128);
            break;
        case 2: /* saw */
            val = p - 128;
            break;
        case 3: /* square */
            val = (p < 128) ? 127 : -128;
            break;
        default: /* sin */
            val = g_sin_lut[p];
            break;
    }
    return (val * w->amp) >> 8;
}

void ce_scene_render_object(const CESceneObject *obj,
                            int32_t *acc, int width, int height,
                            uint32_t tick) {
    if (obj->wave_count == 0) return;

    /* 정규화 좌표 → 픽셀 */
    int cxPx = ((int)obj->cx * width) >> 8;
    int cyPx = ((int)obj->cy * height) >> 8;
    int rPx  = ((int)obj->radius * imin(width, height)) >> 8;
    if (rPx < 1) rPx = 1;

    int x0 = imax(0, cxPx - rPx);
    int x1 = imin(width - 1, cxPx + rPx);
    int y0 = imax(0, cyPx - rPx);
    int y1 = imin(height - 1, cyPx + rPx);

    int rPx2 = rPx * rPx;
    if (rPx2 == 0) rPx2 = 1;

    for (int y = y0; y <= y1; y++) {
        int dy = y - cyPx;
        int dy2 = dy * dy;
        for (int x = x0; x <= x1; x++) {
            int dx = x - cxPx;
            int dist2 = dx * dx + dy2;
            if (dist2 > rPx2) continue;

            /* 거리 기반 감쇠 (정수: 256 = 1.0) */
            int falloff = 256 - ((dist2 * 256) / rPx2);
            if (falloff <= 0) continue;

            /* 거리 (0~255) — 파형 입력 */
            int dist256 = (isqrt((uint32_t)dist2) * 256) / (uint32_t)rPx;
            if (dist256 > 255) dist256 = 255;

            /* 모든 파형 합산 (additive) */
            int val = 0;
            for (int wi = 0; wi < obj->wave_count; wi++) {
                val += eval_wave(&obj->waves[wi], dist256, tick);
            }

            /* 정규화: [-N*128, N*128] → [0, 256] */
            int norm = (val / (int)obj->wave_count) + 128;
            if (norm < 0) norm = 0;
            if (norm > 255) norm = 255;

            /* 최종 강도 = 감쇠 * 정규화값 * opacity / (256*256) */
            int intensity = (falloff * norm * (int)obj->opacity) >> 16;

            /* additive blending (색상 곱) */
            int idx = (y * width + x) * 3;
            acc[idx]     += ((int)obj->color_r * intensity) >> 8;
            acc[idx + 1] += ((int)obj->color_g * intensity) >> 8;
            acc[idx + 2] += ((int)obj->color_b * intensity) >> 8;
        }
    }
}

void ce_scene_render(const CEScene *scene,
                     uint8_t *rgb, int width, int height) {
    int npx = width * height * 3;

    /* accumulation buffer */
    int32_t *acc = (int32_t*)calloc((size_t)npx, sizeof(int32_t));
    if (!acc) return;

    /* 모든 오브젝트 렌더 (additive) */
    for (uint16_t i = 0; i < scene->object_count; i++) {
        ce_scene_render_object(&scene->objects[i], acc,
                               width, height, scene->current_tick);
    }

    /* master amp 적용 + clamp → uint8 */
    for (int i = 0; i < npx; i++) {
        int v = (acc[i] * (int)scene->master_amp) >> 7;
        rgb[i] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
    }

    free(acc);
}

/* ═══════════════════════════════════════════════════════════════════
 * 조회
 * ═══════════════════════════════════════════════════════════════════ */

CESceneObject* ce_scene_find(CEScene *scene, uint16_t id) {
    for (uint16_t i = 0; i < scene->object_count; i++) {
        if (scene->objects[i].id == id) return &scene->objects[i];
    }
    return NULL;
}

int ce_scene_find_group(const CEScene *scene, uint16_t group_id,
                        CESceneObject **out, int max_out) {
    int count = 0;
    for (uint16_t i = 0; i < scene->object_count && count < max_out; i++) {
        if (scene->objects[i].group_id == group_id) {
            out[count++] = (CESceneObject*)&scene->objects[i];
        }
    }
    return count;
}
