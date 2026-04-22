# 01 — 설계 원칙 (v4 corrected)

> **먼저 읽어야 하는 문서입니다.** v4가 이전 버전의 어떤 단정을 수정했는지 정확히 파악해야 구현 단계에서 같은 오류를 반복하지 않습니다.

---

## v4 핵심 수정 (이전 버전과의 차이)

이 6가지는 **구현 착수 전에 반드시 이해해야 합니다.**

1. **장기 기억의 기준 저장소는 기존 `SpatialAI`의 `keyframes` / `deltas` / EMA이다.** `SpatialCanvasPool`은 보조 컨텍스트/탐색 계층으로 사용할 수 있으나, 기존 장기 기억을 대체한다고 가정하지 말 것. 현재 `ai_generate_next()`는 `ai->kf_count`, `ai->keyframes[...]`를 직접 사용한다.

2. **`ai_generate_next()`는 유지한다.** 새 생성 경로 `ai_generate_refine()`는 실험적 병렬 경로다.

3. **`AggTables.A_sum` 고값 셀은 high-confidence prior seed이지, 기본적으로 hard anchor가 아니다.**

4. **slot 0 reserved anchor는 옵션이다.** 기본 구현은 기존 32 content slots를 유지한다. 현재 `SpatialCanvas`는 8×4 = 32 슬롯 콘텐츠 구조다.

5. **`topic_hash`는 1차 grouping 용도로만 사용한다.** bit-level distance를 의미 거리로 해석하지 말 것.

6. **이미지 확장은 feasibility prototype로 제한한다.** production image generator 또는 classical diffusion equivalent를 약속하지 말 것.

---

## 원칙 1 — ContextPool은 별도의 작업 기억층이다

- 기존 `SpatialAI`의 `keyframes` / `deltas` / EMA는 **장기 기억**으로 유지한다.
- 대화/추론 중 쌓이는 문맥은 신규 `context_pool`에 기록한다.
- 추론 경로에서 기존 학습 메모리를 **수정하지 않는다**.

---

## 원칙 2 — 기존 인터페이스 비파괴

- 기존 baseline API (`ai_generate_next`)의 시그니처와 동작을 **바꾸지 않는다**.
- 새 기능은 **새 함수 / 새 구조체 / 새 태그**로만 추가한다.
- `.spai` 직렬화는 **신규 태그(`SPAI_TAG_*`)로 확장**한다.

---

## 원칙 3 — 채널 의미는 보존하되 과도하게 고정 해석하지 않는다

현재 README/SPEC 기준 채널 해석:

| 채널 | 의미 |
|---|---|
| A | 활성도 / 중요도 |
| R | 의미·형태소 성격 (semantic / morpheme-like) |
| G | 기능·대체 관계 (function / substitution-like) |
| B | 문맥·순서 흐름 (context / order-like) |

기본 coarse-to-fine 스케줄은 **B → G → R**을 사용한다. 다만 이를 절대적인 문법 단위 1:1 대응으로 간주하지 말 것. 레벨 정책은 scoring weight + neighborhood radius의 조합으로 구현하며, 고정 라벨이 아니다.

---

## 원칙 4 — 생성은 structured initial field 위의 iterative refinement다

이 경로는 **classical diffusion model과 동일하지 않다.** 하지만 다음 의미에서는 denoising-like refinement로 설명할 수 있다:

- 출발점은 pure random noise가 아니라 **structured initial field**
- 반복적으로 후보를 재점수화하고
- 충분히 신뢰도가 높은 후보를 **resolved / anchor-like state로 승격**하며
- **coarse-to-fine**으로 수렴시킨다

### 권장 용어

- `structured initial field`
- `draft refinement`
- `iterative refinement`
- `convergence`

### 피해야 할 표현

- "classical diffusion"
- "reverse noise process learned"
- "full diffusion model equivalent"

필요 시 **"denoising-like refinement"** 정도로 제한적으로 표현할 것.

---

## 이 원칙들을 위반하는 구현이 떠오를 때

즉시 멈추고 다음을 확인하라:

- **`ai_generate_next()`를 수정하려는 충동** → 원칙 2 위반. 병렬 경로로 전환.
- **`AggTables` 상위 N%를 일괄 `CELL_ANCHOR`로 만들려는 충동** → 수정 포인트 3 위반. strong prior seed로만.
- **`topic_hash ^ other_hash`의 popcount로 유사도 계산** → 수정 포인트 5 위반. exact match + `canvas_delta_*` 비용으로 대체.
- **`canvas_pool`만 보고 생성하려는 충동** → 수정 포인트 1 위반. 기본 경로는 `agg_build(ai)` (keyframes 기반).
- **기존 `.spai` 태그 순서나 크기 변경** → 원칙 2 위반. 신규 `SPAI_TAG_*` 태그 추가.

확신이 없으면 구현을 멈추고 사람에게 질문하라.
