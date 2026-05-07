/* ce_move_profile.h — 오브젝트 유형별 이동 프로파일
 *
 * 핵심 원칙:
 *   물은 물답게 흐르고,
 *   사람은 사람답게 움직이고,
 *   배경은 배경답게 고정.
 *
 * 시그니처에서 이동 유형을 추론하고,
 * 유형에 따라 다른 이동 함수를 적용.
 *
 * Atmos 비유:
 *   배경음악 = 공간 전체에 깔림 (STATIC)
 *   대사     = 화자 위치에 고정 (ANCHOR)
 *   효과음   = 방향 이동 (FLOW / RIGID)
 *   발소리   = 리듬에 맞춤 (ORGANIC)
 */
#ifndef CE_MOVE_PROFILE_H
#define CE_MOVE_PROFILE_H

#include "ce_scene_object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── 이동 유형 ── */
typedef enum {
    CE_MOTION_STATIC  = 0,  /* 배경, 하늘: 움직이지 않음 */
    CE_MOTION_FLOW    = 1,  /* 물, 구름: 방향성 흐름 + 파동 */
    CE_MOTION_RIGID   = 2,  /* 건물, 바위: 미세 진동만 */
    CE_MOTION_ORGANIC = 3,  /* 얼굴, 신체: 부드러운 곡선 이동 */
    CE_MOTION_PARTICLE= 4,  /* 효과, 불꽃: 자유 확산 */
} CEMotionType;

/* ── 이동 프로파일: 각 오브젝트에 부착 ── */
typedef struct {
    CEMotionType type;

    /* FLOW 전용 */
    int8_t flow_dx;         /* 흐름 방향 X (-128~127, 8.0 고정소수) */
    int8_t flow_dy;         /* 흐름 방향 Y */
    uint8_t flow_wave_freq; /* 흐름 내 파동 주기 */
    uint8_t flow_wave_amp;  /* 흐름 내 파동 진폭 */

    /* ORGANIC 전용 */
    uint8_t sway_freq;      /* 흔들림 주기 */
    uint8_t sway_amp;       /* 흔들림 진폭 (0~255 → 0~1.0 매핑) */
    uint8_t phase_offset;   /* 그룹 내 위상 차이 (팔다리 교차용) */

    /* RIGID 전용 */
    uint8_t vibrate_amp;    /* 미세 진동 진폭 (보통 1~3) */

    /* PARTICLE 전용 */
    uint8_t spread_rate;    /* 확산 속도 */
    int8_t gravity;         /* 중력 방향 (양수=아래, 음수=위) */

    /* 공통 */
    uint8_t damping;        /* 감쇠율 (255=즉시멈춤, 0=영원히) */
    uint8_t group_id;       /* 연동 이동 그룹 (0=독립) */
} CEMoveProfile;

/* ═══════════════════════════════════════════════════════════════
 * 시그니처 → 이동 프로파일 추론
 * ═══════════════════════════════════════════════════════════════ */

/* 시그니처 기반 자동 프로파일 생성.
 * avg_r/g/b: 평균 색상
 * texture_energy: 텍스처 에너지 (분산 합)
 * edge_strength: 에지 강도
 * cy_norm: 수직 위치 (0=상단, 255=하단)
 * block_count: 오브젝트 크기 (블록 수)
 */
CEMoveProfile ce_infer_motion_profile(
    uint8_t avg_r, uint8_t avg_g, uint8_t avg_b,
    uint8_t texture_energy,
    uint8_t edge_strength,
    uint8_t cy_norm,
    uint16_t block_count
);

/* ═══════════════════════════════════════════════════════════════
 * 프로파일 기반 틱 이동
 * ═══════════════════════════════════════════════════════════════ */

/* 프로파일에 따라 1틱 이동 적용.
 * obj의 cx, cy를 직접 수정.
 * base_cx/cy는 원래 위치 (복귀 기준점). */
void ce_move_apply_tick(CESceneObject *obj,
                        const CEMoveProfile *profile,
                        uint8_t base_cx, uint8_t base_cy,
                        uint32_t tick);

/* 그룹 연동 이동: 같은 group_id의 오브젝트들이
 * 위상만 다르게 동일 패턴으로 이동.
 * (걷기: 왼발/오른발 교차) */
void ce_move_apply_group(CESceneObject *objects, uint16_t count,
                         const CEMoveProfile *profiles,
                         const uint8_t *base_cx, const uint8_t *base_cy,
                         uint32_t tick);

#ifdef __cplusplus
}
#endif
#endif /* CE_MOVE_PROFILE_H */
