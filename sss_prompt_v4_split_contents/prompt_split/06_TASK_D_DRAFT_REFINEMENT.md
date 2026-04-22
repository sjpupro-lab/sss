# 06 — Task D: Hierarchical Draft Refinement (★ 실험적)

**우선순위**: A, B, C 완료 후
**의존성**: Task A (context_pool 조회)
**브랜치 권장**: **별도 실험 브랜치에서 작업**. main에 직접 머지하지 말 것.

---

## 목적

`ai_generate_next()`를 **대체하지 않고**, 별도의 실험적 생성 경로 `ai_generate_refine()`를 추가한다.

---

## Baseline과 새 경로의 관계

### 현재 baseline (`ai_generate_next()`)

- input encode
- unified match (`MATCH_GENERATE`)
- next keyframe 선택
- UTF-8 aware row decode

### 새 경로 (`ai_generate_refine()`)

- **structured initial field** 생성
- 후보 셀 **반복 재점수화**
- **resolved promotion**
- **coarse-to-fine convergence**

**절대 원칙**: 두 경로는 독립적으로 동작하며, `ai_generate_refine()`는 `ai_generate_next()`의 시그니처나 동작에 영향을 주지 않는다.

---

## D.0 — 핵심 개념

출발점은 pure random noise가 아니라 **부분적으로 정렬된 초안**이다.

이 초안은 다음에서 온다:

- long-term prior (`AggTables`)
- optional short-term prior (`context_pool`)
- input prompt anchors

### 중요

- **`AggTables.A_sum` 고값 셀은 기본적으로 hard anchor가 아니라 strong prior seed다.**
- **hard anchor는 우선적으로 입력 프롬프트에서 직접 유도된 셀에만 부여한다.**
- **`AggTables` 기반 hard-anchor화는 설정 가능한 옵션으로만 둘 것** (`allow_prior_anchors`).

---

## D.1 — 레벨 개념

텍스트용 기본 레벨 해석:

| 레벨 | neighborhood | dominant channel |
|---|---|---|
| L0 | large | B |
| L1 | medium | G |
| L2 | small | R |

### 주의

- **레벨을 `A_sum` 크기로 고정 배정하지 말 것.**
- **동일 candidate 집합에 대해 단계별 scoring policy를 바꾸는 방식이 기본이다.**
- 각 레벨은 candidate 자체를 바꾸는 게 아니라, scoring weight와 neighborhood radius를 바꾼다.

---

## D.2 — 자료 구조

```c
#define REFINE_TOPK_MAX 16

typedef enum {
    CELL_EMPTY     = 0,
    CELL_CANDIDATE = 1,
    CELL_RESOLVED  = 2,
    CELL_ANCHOR    = 3
} CellStatus;

typedef struct {
    uint8_t  status;
    uint8_t  value;          /* valid for RESOLVED / ANCHOR */
    uint8_t  n_cand;
    uint8_t  cand_values[REFINE_TOPK_MAX];
    float    cand_scores[REFINE_TOPK_MAX];
    float    confidence;
} CellState;

typedef struct {
    CellState cells[GRID_SIZE * GRID_SIZE];
    uint32_t  n_anchor;
    uint32_t  n_candidate;
    uint32_t  n_resolved;
    uint32_t  n_promoted_this_iter;
} DraftField;
```

### 상태 전이 규칙

- `CELL_EMPTY` → 관심 대상 아님 (A_sum 거의 0).
- `CELL_CANDIDATE` → scoring 루프가 Top-K를 갱신하는 대상.
- `CELL_RESOLVED` → 이번 반복에서 confidence 임계치를 넘어 확정.
- `CELL_ANCHOR` → 입력 프롬프트 유도 또는 사용자 옵션으로 강제. **절대 overwrite 금지**.

---

## D.3 — 공개 API

`include/spatial_generate.h`에 추가:

```c
typedef struct {
    float    ch_weights[3][3];      /* [level][R,G,B] */
    uint32_t topk[3];
    float    promote_threshold[3];
    uint32_t max_iter[3];
    float    converge_rate[3];
    uint32_t neighbor_radius[3];    /* large / medium / small */
    float    temperature;           /* 0 = argmax */
    int      use_context_pool;
    int      allow_prior_anchors;   /* off by default */
} RefineConfig;

RefineConfig refine_config_default_text(void);
RefineConfig refine_config_default_image(void);

uint32_t ai_generate_refine(SpatialAI* ai,
                            const char* input_text,
                            char* out, uint32_t max_out,
                            const RefineConfig* cfg,
                            float* out_confidence,
                            uint32_t* out_iterations);
```

---

## D.4 — 기본 텍스트 프리셋

```c
RefineConfig refine_config_default_text(void) {
    RefineConfig c;
    memset(&c, 0, sizeof(c));
    c.ch_weights[0][0] = 0.1f; c.ch_weights[0][1] = 0.3f; c.ch_weights[0][2] = 1.0f;
    c.ch_weights[1][0] = 0.3f; c.ch_weights[1][1] = 1.0f; c.ch_weights[1][2] = 0.3f;
    c.ch_weights[2][0] = 1.0f; c.ch_weights[2][1] = 0.3f; c.ch_weights[2][2] = 0.1f;
    c.topk[0] = 4;  c.topk[1] = 8;  c.topk[2] = 16;
    c.promote_threshold[0] = 0.55f;
    c.promote_threshold[1] = 0.65f;
    c.promote_threshold[2] = 0.75f;
    c.max_iter[0] = 12; c.max_iter[1] = 20; c.max_iter[2] = 30;
    c.converge_rate[0] = 0.02f;
    c.converge_rate[1] = 0.02f;
    c.converge_rate[2] = 0.01f;
    c.neighbor_radius[0] = 16;
    c.neighbor_radius[1] = 8;
    c.neighbor_radius[2] = 2;
    c.temperature = 0.0f;
    c.use_context_pool = 1;
    c.allow_prior_anchors = 0;
    return c;
}
```

---

## D.5 — 초기장 생성 규칙

`draft_field_init(...)` 기본 규칙:

1. **`input_text`에서 직접 유도된 셀은 `CELL_ANCHOR`**
2. 나머지는 `CELL_CANDIDATE` 또는 `CELL_EMPTY`로 초기화
3. `AggTables`는 **candidate scoring prior**로 사용
4. 필요 시에만 `allow_prior_anchors=1`일 때 high-A_sum seed를 `CELL_RESOLVED` 또는 `CELL_ANCHOR`로 승격

### 명시 금지

- `argmax_byte_for_row()`를 모든 high-A cell에 직접 꽂는 식의 초기화 **금지**.
- row-wise 최빈값 하나를 전체 cell value로 복사하는 식의 구현 **금지**.
- 이유: 그런 초기화는 한 행의 최빈값 하나로 그 행의 모든 고-A 셀을 동일하게 덮어 쓰는 효과를 낸다. 이는 학습된 다양성을 잃는다.

---

## D.6 — neighbor signature 규칙

기본 구현:

- **`CELL_ANCHOR`만 사용해** local signature 계산
- anchor가 없으면 **`CELL_RESOLVED`를 약한 보조**로 포함 가능
- 그래도 없으면 `input_signature_get()` 또는 global mean fallback 사용

### 명시 금지

- **"anchor만 쓴다"와 "candidate도 절반 가중"을 동시에 쓰지 말 것.**
- 둘 중 하나의 규칙을 일관되게 적용.

---

## D.7 — refine 루프 규칙

각 level마다:

1. 현재 level 정책 (B/G/R weights + radius)으로 candidate score update
2. `confidence >= threshold`인 cell을 `CELL_RESOLVED`로 승격
3. promote rate가 convergence threshold 아래면 다음 level로 이동
4. 최종적으로 unresolved candidates는 Top-1 또는 fallback rule로 마감

### 주의

- **`CELL_ANCHOR`는 절대 overwrite 금지.** 디버그 빌드에서 `assert(c->status != CELL_ANCHOR)` 강제.
- `CELL_RESOLVED`는 다음 level에서 **사실상 anchor-like source**로 사용 가능.
- 극저 anchor density에서 반복이 무의미하면 **내부적으로 `ai_generate_next()` fallback 허용**. 로그에 `"refine_fallback"` 기록.

---

## D.8 — benchmark

신규:

- `tests/test_refine.c`
- `tests/bench_refine.c`

### 측정 항목

- UTF-8 validity
- avg confidence
- iteration count
- promotion curve
- diversity (temperature > 0)
- baseline 대비 retrieval consistency

### 주의

- **`perplexity`라는 용어는 쓰지 말 것.** 현재 엔진은 standard LM probability model이 아니다.
- 수치 목표(다양성 2배, Top-1 +5%p 등)는 **참고용**이며 merge-blocking hard gate로 두지 말 것.

---

## 검증 체크리스트

```
[ ] make all          경고 없이 빌드
[ ] make test         기존 69 tests PASS (회귀 없음)
[ ] test_refine       anchor immutability + convergence path 검증
[ ] test_refine       CELL_ANCHOR overwrite 시도 시 assert 트리거
[ ] bench_refine      refinement trace / diversity / consistency 리포트 출력
[ ] ai_generate_next  동작 불변 확인 (별도 테스트로)
```

---

## 다음 단계

Task D 실험이 안정화되면 `07_TASK_E_REPL_INTEGRATION.md`의 `:gen refine`, `:refine cfg`, `:refine trace` 명령을 통합한다. 안정성이 충분히 검증되기 전까지는 main 브랜치 머지 보류.
