/* ce_move_profile.c — 오브젝트 유형별 이동 엔진
 *
 * 물 → 수평 흐름 + 수직 파동 (출렁임)
 * 피부/유기체 → 부드러운 곡선 (호흡, 흔들림)
 * 배경 → 고정
 * 바위/건물 → 미세 진동만
 * 효과 → 자유 확산 + 중력
 *
 * 모든 연산 정수. float 없음.
 */

#include "ce_move_profile.h"
#include <string.h>

/* ── sin LUT (256 엔트리, [-127, 127]) ── */
static int8_t g_sin[256];
static int g_sin_ready = 0;

static void ensure_lut(void) {
    if (g_sin_ready) return;
    for (int i = 0; i < 256; i++) {
        int q = i >> 6;
        int p = (i & 63) << 2;
        int h;
        switch (q) {
            case 0: h = p; break;
            case 1: h = 252 - p; break;
            case 2: h = -p; break;
            default: h = -(252 - p); break;
        }
        g_sin[i] = (int8_t)((h * 127) / 252);
    }
    g_sin_ready = 1;
}

static inline int8_t isin(uint8_t phase) { return g_sin[phase]; }

static inline uint8_t clamp8(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* ═══════════════════════════════════════════════════════════════
 * 시그니처 → 이동 프로파일 추론
 *
 * 판단 기준:
 *   1. 색상 자체 (청록 → 물, 피부톤 → 유기체)
 *   2. 텍스처 에너지 (낮음 → 균일 표면, 높음 → 복잡)
 *   3. 에지 강도 (높음 → 경계가 뚜렷 → 단단한 물체)
 *   4. 위치 (상단 → 하늘, 하단 → 바닥)
 *   5. 크기 (큰 것 → 배경일 확률 높음)
 * ═══════════════════════════════════════════════════════════════ */

CEMoveProfile ce_infer_motion_profile(
    uint8_t avg_r, uint8_t avg_g, uint8_t avg_b,
    uint8_t texture_energy,
    uint8_t edge_strength,
    uint8_t cy_norm,
    uint16_t block_count)
{
    ensure_lut();

    CEMoveProfile p;
    memset(&p, 0, sizeof(p));
    p.damping = 128;

    int brightness = ((int)avg_r + avg_g + avg_b) / 3;

    /* ── 1. 어둠/배경 판별 ── */
    if (brightness < 15) {
        p.type = CE_MOTION_STATIC;
        return p;
    }

    /* ── 2. 하늘 판별: 상단 + 밝음 + 균일 ── */
    if (cy_norm < 60 && brightness > 120 && texture_energy < 40) {
        p.type = CE_MOTION_STATIC;
        return p;
    }

    /* ── 3. 물 판별: 청록/파랑 계열 + 낮은 텍스처 ── */
    int is_water = 0;
    /* 청록: G와 B가 높고, R이 낮음 */
    if (avg_g > 100 && avg_b > 80 && avg_r < avg_g) is_water = 1;
    /* 파랑: B가 dominant */
    if (avg_b > 120 && avg_b > avg_r + 30 && texture_energy < 60) is_water = 1;

    if (is_water) {
        p.type = CE_MOTION_FLOW;
        /* 물 흐름: 수평 방향 느린 이동 + 수직 출렁임 */
        p.flow_dx = 3;              /* 오른쪽으로 느리게 */
        p.flow_dy = 0;
        p.flow_wave_freq = 8;       /* 출렁임 주기 */
        p.flow_wave_amp = 4;        /* 출렁임 폭 (작게) */
        p.damping = 0;              /* 감쇠 없음 (계속 흐름) */
        return p;
    }

    /* ── 4. 녹색/숲 판별: 약간의 흔들림 ── */
    if (avg_g > avg_r && avg_g > avg_b && avg_g > 80) {
        p.type = CE_MOTION_ORGANIC;
        p.sway_freq = 6;           /* 바람에 흔들리는 주기 */
        p.sway_amp = 2;            /* 아주 작은 흔들림 */
        p.phase_offset = 0;
        p.damping = 200;
        return p;
    }

    /* ── 5. 피부톤 판별: 유기체 이동 ── */
    int is_skin = 0;
    if (avg_r > 120 && avg_r > avg_b + 20 &&
        avg_g > 70 && avg_g < avg_r) is_skin = 1;
    /* 어두운 피부톤 */
    if (avg_r > 60 && avg_r > avg_b &&
        avg_g > 30 && avg_g < avg_r &&
        texture_energy > 40) is_skin = 1;

    if (is_skin) {
        p.type = CE_MOTION_ORGANIC;
        p.sway_freq = 12;          /* 호흡 리듬 */
        p.sway_amp = 3;            /* 미세한 움직임 */
        p.phase_offset = 0;
        p.damping = 180;
        return p;
    }

    /* ── 6. 암석/단단한 물체: 높은 에지 + 갈색 계열 ── */
    if (edge_strength > 20 && texture_energy > 80) {
        p.type = CE_MOTION_RIGID;
        p.vibrate_amp = 1;          /* 거의 안 움직임 */
        p.damping = 250;
        return p;
    }

    /* ── 7. 큰 오브젝트 → 배경 성향 ── */
    if (block_count > 40) {
        p.type = CE_MOTION_STATIC;
        return p;
    }

    /* ── 8. 작고 밝은 오브젝트 → 파티클 ── */
    if (block_count < 6 && brightness > 150) {
        p.type = CE_MOTION_PARTICLE;
        p.spread_rate = 5;
        p.gravity = -2;            /* 위로 떠오름 */
        p.damping = 100;
        return p;
    }

    /* 기본: RIGID */
    p.type = CE_MOTION_RIGID;
    p.vibrate_amp = 1;
    p.damping = 200;
    return p;
}

/* ═══════════════════════════════════════════════════════════════
 * 프로파일 기반 틱 이동
 * ═══════════════════════════════════════════════════════════════ */

void ce_move_apply_tick(CESceneObject *obj,
                        const CEMoveProfile *profile,
                        uint8_t base_cx, uint8_t base_cy,
                        uint32_t tick)
{
    /* obj->cx/cy are written below per profile type; current values
     * are not needed because base_cx/base_cy are the anchor reference. */
    switch (profile->type) {
    case CE_MOTION_STATIC:
        /* 완전 고정 */
        obj->cx = base_cx;
        obj->cy = base_cy;
        break;

    case CE_MOTION_FLOW: {
        /* 방향 흐름 + 파동 */
        /* 수평 이동: 매 틱 dx/256 만큼 */
        int new_cx = (int)base_cx + (profile->flow_dx * (int)tick) / 32;
        /* 래핑 (화면 밖 → 반대편) */
        new_cx = ((new_cx % 256) + 256) % 256;

        /* 수직 파동: sin(tick * freq) * amp */
        int wave = (isin((uint8_t)(tick * profile->flow_wave_freq)) *
                    profile->flow_wave_amp) >> 7;
        int new_cy = (int)base_cy + wave;

        obj->cx = clamp8(new_cx);
        obj->cy = clamp8(new_cy);
        break;
    }

    case CE_MOTION_RIGID: {
        /* 미세 진동: 원래 위치 근처에서 떨림 */
        int vx = (isin((uint8_t)(tick * 17)) * profile->vibrate_amp) >> 7;
        int vy = (isin((uint8_t)(tick * 23 + 64)) * profile->vibrate_amp) >> 7;
        obj->cx = clamp8((int)base_cx + vx);
        obj->cy = clamp8((int)base_cy + vy);
        break;
    }

    case CE_MOTION_ORGANIC: {
        /* 부드러운 곡선: sin/cos 조합으로 8자 또는 원 궤도 */
        uint8_t phase = (uint8_t)(tick * profile->sway_freq + profile->phase_offset);
        int dx = (isin(phase) * profile->sway_amp) >> 7;
        int dy = (isin((uint8_t)(phase + 64)) * profile->sway_amp) >> 7;  /* 90도 위상차 → 원 */
        obj->cx = clamp8((int)base_cx + dx);
        obj->cy = clamp8((int)base_cy + dy);
        break;
    }

    case CE_MOTION_PARTICLE: {
        /* 확산 + 중력 */
        int age = (int)tick;
        int dx = (isin((uint8_t)(tick * 31 + obj->id * 97)) * profile->spread_rate) >> 7;
        int dy = (profile->gravity * age) / 16;
        obj->cx = clamp8((int)base_cx + dx);
        obj->cy = clamp8((int)base_cy + dy);
        break;
    }
    }
}

void ce_move_apply_group(CESceneObject *objects, uint16_t count,
                         const CEMoveProfile *profiles,
                         const uint8_t *base_cx, const uint8_t *base_cy,
                         uint32_t tick)
{
    for (uint16_t i = 0; i < count; i++) {
        ce_move_apply_tick(&objects[i], &profiles[i],
                           base_cx[i], base_cy[i], tick);
    }
}
