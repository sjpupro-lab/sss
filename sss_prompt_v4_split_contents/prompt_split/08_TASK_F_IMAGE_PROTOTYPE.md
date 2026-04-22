# 08 — Task F: Image Modality Prototype (experimental only)

**우선순위**: 최후 (A~E 모두 안정화 후)
**의존성**: Task D (`ai_generate_refine`)
**브랜치 권장**: **반드시 별도 실험 브랜치.** production feature로 취급하지 말 것.

---

## 핵심

**텍스트의 절/단어/형태소를 이미지에 그대로 옮기지 않는다.** 이미지용 coarse-to-fine 계층은 **Composition / Contour / Detail**로 재정의한다.

현재 논의 기준으로, 이미지 확장은 **"가능성 있는 prototype"**이지, 바로 stable-diffusion-equivalent generator를 약속하는 단계는 **아니다**.

---

## 이미지 레벨 정의

| Level | Name | Meaning | Dominant channel |
|---|---|---|---|
| 0 | Composition | large layout, foreground/background, coarse blobs | B |
| 1 | Contour | edges, boundaries, part placement | G |
| 2 | Detail | texture, fine shading, local detail | R |

---

## 신규 파일

- `include/spatial_image.h`
- `src/spatial_image.c`
- `tools/img2grid.c`
- `tools/grid2img.c`
- `tests/test_image_roundtrip.c`

---

## 최소 API

```c
SpatialGrid* image_to_grid(const char* path);
int          grid_to_image(const SpatialGrid* g, const char* out_path);

int ai_generate_image(SpatialAI* ai,
                      const char* prompt_text,
                      const char* out_path,
                      const RefineConfig* cfg);
```

---

## 구현 방침

1. **먼저 roundtrip feasibility를 본다.** 이미지 → grid → 이미지 왕복이 질적으로 복원되는지.
2. **qualitative visualization을 우선한다.** 수치 목표보다 "사람 눈으로 봐서 뭔가 복원되는가" 기준.
3. **image refine preset은 text preset과 별도로 둔다.**
4. **external dependency는 최소화** (`stb_image.h`, `stb_image_write.h` 정도만).

---

## 이미지 프리셋 예시

```c
RefineConfig refine_config_default_image(void) {
    RefineConfig c = refine_config_default_text();
    c.ch_weights[0][0] = 0.1f; c.ch_weights[0][1] = 0.2f; c.ch_weights[0][2] = 1.0f;
    c.ch_weights[1][0] = 0.3f; c.ch_weights[1][1] = 1.0f; c.ch_weights[1][2] = 0.3f;
    c.ch_weights[2][0] = 1.0f; c.ch_weights[2][1] = 0.3f; c.ch_weights[2][2] = 0.1f;
    c.promote_threshold[0] = 0.40f;
    c.promote_threshold[1] = 0.50f;
    c.promote_threshold[2] = 0.60f;
    c.max_iter[0] = 20;
    c.max_iter[1] = 40;
    c.max_iter[2] = 80;
    c.converge_rate[0] = 0.01f;
    c.converge_rate[1] = 0.01f;
    c.converge_rate[2] = 0.005f;
    return c;
}
```

### 왜 텍스트보다 임계치가 낮은가

이미지 셀 값은 연속적이고 앵커 밀도가 텍스트보다 훨씬 낮다. 엄격한 임계치를 쓰면 수렴이 실패한다. 승격을 너그럽게, 대신 반복 상한을 길게.

---

## 이미지 → grid 매핑 지침

- **A 채널**: 휘도 또는 엣지 강도 (둘 중 실험 결과 나은 쪽)
- **R 채널**: 이미지 원본 R 또는 처리된 R
- **G 채널**: 이미지 원본 G 또는 처리된 G
- **B 채널**: 이미지 원본 B 또는 처리된 B

주의: 여기서 R/G/B는 **이미지 색상 채널**로 재사용한다. 텍스트 채널 해석(의미/기능/문맥)과 **병렬적으로 공존**할 수 있도록, 이미지 모드 여부를 캔버스 단위 플래그로 표시.

---

## 검증

`tests/test_image_roundtrip.c`:

- 몇 장의 샘플 이미지 (256×256) 로드
- `image_to_grid()` → `grid_to_image()` 왕복
- 시각적 유사성 확인 (PSNR 등의 정량 지표는 **참고용**)

```
[ ] make all              경고 없이 빌드
[ ] make test             기존 69 tests PASS
[ ] test_image_roundtrip  왕복 후 이미지 파일 생성
[ ] 수동 육안 검사        원본과 복원본이 질적으로 유사
```

---

## 절대 주장하지 말 것

- "already equivalent to diffusion model"
- "stable-diffusion 대체"
- "production-ready image generator"
- "reverse noise process learned"

대신:

- "feasibility prototype"
- "image roundtrip proof of concept"
- "iterative refinement on image grids"

---

## 주의

- quantitative PSNR target은 **참고용**으로만 둘 것. merge gate로 두지 말 것.
- image task는 A~E 안정화 후 **마지막**에 착수.
- 실패해도 baseline 전체가 보존되어야 함.

---

## 완료 후

이미지 프로토타입이 feasibility를 보여주면, 다음 단계로:

- 오디오 트랙 같은 추가 side-car 채널
- 텍스트/이미지 cross-modal 검색
- 이미지 입력 + 텍스트 출력 (캡션 등)

등을 **별도 프로젝트로** 검토할 수 있다. 단, 현재 범위는 여기까지.
