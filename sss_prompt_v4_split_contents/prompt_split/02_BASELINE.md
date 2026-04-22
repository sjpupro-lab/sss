# 02 — 현재 레포 아키텍처 (기준선)

> 이 문서를 읽기 전에 **실제 소스를 먼저 읽어야** 의미가 있습니다. 여기 요약된 표만 보고 진행하면 `ai_generate_next()`의 현재 구현을 모른 채로 "대체가 아니라 병렬 경로"의 의미를 이해할 수 없습니다.

---

## 작업 시작 전 필독

레포 루트 기준 다음 파일들을 순서대로 정독하세요:

- `README.md`, `README_KO.md`
- `SPEC.md`, `SPEC-ENGINE.md`
- `include/spatial_canvas.h`
- `include/spatial_subtitle.h`
- `include/spatial_generate.h`
- `include/spatial_keyframe.h`

추가로 실제 생성 경로를 파악하려면:

- `src/spatial_generate.c` — `ai_generate_next()` 본체
- `src/spatial_match.c` — `MATCH_GENERATE` 경로
- `src/spatial_io.c` — `.spai` 직렬화 태그 체계
- `tools/chat.c` — REPL 통합 지점

---

## 구현 상태 표

| 컴포넌트 | 역할 | 상태 |
|---|---|---|
| `SpatialGrid` 256×256 | clause 1개를 담는 grid | 구현됨 |
| `SpatialCanvas` 2048×1024 | 32-slot canvas | 구현됨 |
| `SpatialCanvasPool` | canvas 시퀀스 / subtitle index | 구현됨 |
| `CanvasFrameType` | canvas-level I/P 분류 | 구현됨 |
| `scene_change_classify` | x264-style scene change heuristic | 구현됨 |
| `AggTables` | keyframe 집계 기반 prior field | 구현됨 |
| `agg_build_from_pool` | pool 집계 기반 prior field | 구현됨 |
| `ai_generate_next` | retrieval + next-frame decode baseline | 구현됨 |

---

## 중요 관찰 (놓치지 말 것)

### `ai_generate_next()`는 keyframe 기반이다

현재 구현은:

- `find_next_in_topic()`
- `ai->kf_count`
- `ai->keyframes[...]`

를 **직접** 사용한다. `canvas_pool`을 거치지 않는다.

**함의**: 새 생성 경로가 `canvas_pool`만 조회하도록 구현하면 학습된 키프레임 기반 지식을 우회하게 된다. 장기 prior는 반드시 `agg_build(ai)`(keyframes 집계)를 포함해야 한다.

### 이 엔진은 retrieval-heavy이다

README에 명시: **"not a generative LLM replacement"**.

**함의**:

- `perplexity` 같은 autoregressive LM 지표는 사용하지 말 것
- 출력 품질은 retrieval consistency + coarse-to-fine convergence로 평가
- classical diffusion model의 평가 관행(FID 등)을 그대로 가져오지 말 것

### 현재 테스트 규모

README 기준 **69 tests** 전체 PASS 상태가 baseline이다.

**함의**: 회귀 테스트 기준은 69 tests. 새 기능 추가 후에도 이 전체가 PASS 유지.

---

## `SpatialCanvas` 구조 (Task B 관련)

현재 `SlotMeta`의 기존 필드:

```c
typedef struct {
    DataType type;
    float    boundary_weight;
    uint32_t byte_length;
    uint32_t topic_hash;
    int      occupied;
} SlotMeta;
```

캔버스 내 32 content slots (8×4)가 기본 구조다. slot 0을 예약하는 방식은 **layout core를 건드리는 변경**이므로 기본 필수가 아니다. Task B에서 다룬다.

---

## `AggTables` 해석 주의

`AggTables.A_sum[y*GRID_SIZE + x]` 값이 크다는 것은:

- "이 (y, x) 셀이 학습 중 자주 활성화되었다"는 **통계적 prior**일 뿐
- **"이 셀은 반드시 이 값이어야 한다"는 hard constraint가 아니다**

**함의**: Task D의 `draft_field_init()`에서 고-A 셀을 즉시 `CELL_ANCHOR`로 승격하고 `argmax_byte_for_row()`로 값을 꽂는 구현은 금지. 해당 셀은 scoring 단계에서 strong prior seed로 기여할 뿐, hard anchor는 **입력 프롬프트에서 직접 유도된 셀**에만 부여한다.

---

## 착수 전 확인 체크리스트

- [ ] `SpatialCanvas` / `SpatialCanvasPool` 사용처 grep 완료
- [ ] `ai_generate_next` 호출 경로 확인 완료
- [ ] `spatial_io.c` 저장/로드 태그 경로 확인 완료
- [ ] README/SPEC에서 keyframe 중심 baseline 재확인 완료
- [ ] refine 경로는 baseline 대체가 아니라 **병렬 실험 경로**임을 이해 완료
- [ ] image task는 **feasibility prototype** 수준임을 이해 완료

모두 체크되면 `03_TASK_A_CONTEXT_POOL.md`로 진행.
