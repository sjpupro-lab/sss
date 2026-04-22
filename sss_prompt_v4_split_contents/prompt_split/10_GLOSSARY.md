# 10 — 용어 가이드 (부록)

> 이 문서는 **참조용**입니다. 구현 중 용어가 애매해질 때 돌아와서 확인하세요.

---

## 핵심 대조표

| 피해야 할 표현 | 권장 표현 | 이유 |
|---|---|---|
| full diffusion model | iterative refinement / denoising-like refinement | 현재 구조는 classical diffusion과 다름 |
| random start | structured initial field / learned draft | 출발점이 pure random이라고 단정하면 오해 |
| replace baseline | add experimental path | baseline 유지가 원칙 |
| exact token-axis mapping for image | composition / contour / detail hierarchy | 이미지와 텍스트 계층은 다름 |
| topic-hash distance = semantic distance | exact hash grouping + actual delta/RLE cost | 해시 bit 차이는 의미 거리 근거가 약함 |
| perplexity | avg confidence / retrieval consistency | 이 엔진은 standard LM이 아님 |
| reverse noise process | iterative scoring + promotion | 학습 자체가 classical diffusion과 다름 |
| hard anchor (AggTables 기반) | strong prior seed (옵션 시 hard anchor) | 고-A 셀을 무조건 고정하면 다양성 소실 |

---

## 개념 용어

### `structured initial field` (구조화된 초기장)

생성 루프의 출발 상태. pure random noise가 아니라, 학습된 `AggTables`와 입력 프롬프트 앵커가 섞인 **부분 정답 상태**.

### `draft refinement` (초안 정제)

출발 상태를 단계별로 다듬어 최종 출력에 수렴시키는 과정. "노이즈 제거"가 아니라 "흐릿한 밑그림을 진하게 하기"에 더 가깝다.

### `hard anchor`

생성 루프 전반에서 **절대 변경되지 않는** 셀. 원칙적으로 **입력 프롬프트에서 직접 유도된 셀**에만 부여.

### `strong prior seed`

`AggTables.A_sum`이 높아 scoring 단계에서 강한 영향력을 갖는 셀. 단, **hard anchor는 아니다**. 기본적으로 `CELL_CANDIDATE` 상태로 시작하며, scoring 결과에 따라 `CELL_RESOLVED`로 승격될 수 있다.

### `candidate`

아직 확정되지 않은 셀. Top-K 후보군을 보유하고, 매 반복에서 재점수화된다.

### `resolved`

이번 반복에서 confidence 임계치를 넘어 확정된 셀. 다음 레벨에서는 사실상 anchor-like source로 작용.

### `convergence` (수렴)

한 레벨의 promotion rate가 `converge_rate` 아래로 떨어져 더 이상 승격이 일어나지 않는 상태. 이때 다음 레벨로 이동.

### `coarse-to-fine`

해상도를 낮은 쪽에서 높은 쪽으로 올려가는 전략. 이 엔진에서는 **B → G → R** 순서 (텍스트 기준).

---

## 채널 용어

| 채널 | 일반 해석 | 이미지 해석 |
|---|---|---|
| A | 활성도 / 중요도 | 휘도 or 엣지 강도 |
| R | 의미·형태소 성격 | 이미지 R 채널 |
| G | 기능·대체 관계 | 이미지 G 채널 |
| B | 문맥·순서 흐름 | 이미지 B 채널 |

주의: 동일 채널이 텍스트 모드와 이미지 모드에서 **다른 의미**로 해석된다. 혼동 방지를 위해 모드 플래그로 구분.

---

## 레벨 이름 (절대 혼용 금지)

### 텍스트 레벨

- L0: clause (절, 큰 구조)
- L1: word (단어, 중간 구조)
- L2: morpheme (형태소, 세부 구조)

### 이미지 레벨

- L0: **Composition** (구도)
- L1: **Contour** (윤곽)
- L2: **Detail** (디테일)

**텍스트의 절/단어/형태소를 이미지에 그대로 옮기지 말 것.**

---

## 자료 구조 용어

- `SpatialAI` — 엔진 메인 구조체. `keyframes`, `deltas`, `context_pool`, `canvas_pool` 보유.
- `SpatialGrid` — 256×256 grid, clause 1개 단위.
- `SpatialCanvas` — 2048×1024, 32 slot.
- `SpatialCanvasPool` — canvas 시퀀스.
- `AggTables` — keyframe 또는 pool 집계 prior field.
- `DraftField` — Task D의 refine 루프 상태.
- `CellState` — `DraftField` 내 단일 셀 상태.
- `RefineConfig` — refine 루프 설정 (텍스트/이미지 프리셋 별도).
- `ReclusterReport` — Task C 재배치 결과 리포트.

---

## API 네이밍 규칙

- `ai_*` — SpatialAI 대상 고수준 함수
- `pool_*` — SpatialCanvasPool 대상
- `canvas_*` — SpatialCanvas 대상
- `grid_*` — SpatialGrid 대상
- `agg_*` — AggTables 대상
- `refine_*` — Task D 관련
- `context_*` — Task A 관련

새 함수 추가 시 이 규칙을 따를 것.

---

## `.spai` 태그 명명

신규 태그는 `SPAI_TAG_*` 접두사 + 의미 있는 이름:

- `SPAI_TAG_SEQMETA` — sequence_id / timestamp_us
- `SPAI_TAG_CONTEXT_POOL` — context_pool 저장 (명시 플래그 시만)
- 등등

**기존 태그의 값 변경 또는 순서 변경 금지.** 항상 트레일링 태그로 추가.

---

## 지양해야 할 표현 (주석/커밋 메시지/문서에서)

- "AI가 기억한다" → "engine accumulates keyframes"
- "머릿속에" → 사용 금지
- "매우 혁신적" → 사용 금지
- "SOTA와 동등" → 사용 금지 (근거 없는 주장)
- "거의 perplexity와 같은 것" → 사용 금지 (이 엔진은 LM이 아님)

### 권장 표현

- "structured prior over byte positions"
- "coarse-to-fine refinement"
- "retrieval-heavy engine with keyframe-based memory"
- "feasibility prototype for X"
- "experimental path parallel to baseline"
