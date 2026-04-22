# 09 — 체크리스트, 수용 기준, 구현 순서, 절대 금지

> 이 문서는 구현 전체에 걸쳐 **지속적으로 참조**해야 합니다. 각 Task 종료 시 해당 섹션을 다시 확인하세요.

---

## 파일 체크리스트

### 신규 파일

- `include/spatial_recluster.h`
- `src/spatial_recluster.c`
- `tests/test_recluster.c`
- `tests/test_refine.c`
- `tests/bench_refine.c`
- `tests/bench_context.c`

### 이미지 실험 (Task F)

- `include/spatial_image.h`
- `src/spatial_image.c`
- `tools/img2grid.c`
- `tools/grid2img.c`
- `tests/test_image_roundtrip.c`

### 수정 파일

- `include/spatial_keyframe.h` (context_pool)
- `include/spatial_canvas.h` (sequence metadata)
- `include/spatial_generate.h` (`RefineConfig`, `ai_generate_refine`)
- `src/spatial_keyframe.c`
- `src/spatial_generate.c`
- `tools/chat.c`
- `Makefile`
- `SPEC.md`
- 필요 시 `spatial_io.c` (**new optional tags only**)

### 건드리지 않을 파일

- `src/spatial_canvas.c` — slot 0 reserved anchor 변경은 기본 필수 아님. Reserved slot을 실험하지 않는다면 canvas layout core는 건드리지 말 것.
- `dict/` — 한국어 형태소 사전 수정 금지.

---

## 수용 기준

```
[ ] make all            경고 없이 빌드
[ ] make test           기존 전체 테스트 PASS 유지 (69 tests)
[ ] 기존 .spai 로드 가능
[ ] ai_generate_next()  동작 불변
[ ] test_refine         anchor immutability + convergence path 검증
[ ] bench_context       context on/off 비교 가능
[ ] bench_refine        refinement trace / diversity / consistency 리포트 출력
[ ] Task F              roundtrip feasibility 확인까지만 우선
```

### 수치 목표에 대한 정책

다음은 **초기 실험 목표**일 뿐이며, **merge-blocking hard gate로 두지 말 것**:

- 다양성 2배
- Top-1 +5%p
- PSNR 30dB

실제 결과가 이에 못 미쳐도 병렬 경로로서 가치가 있으면 머지 가능. 단, 기존 69 tests 회귀는 **절대 불가**.

---

## 구현 순서

각 단계 종료 시 **반드시** 다음을 실행:

- `make all`
- `make test`
- minimal manual probe (REPL로 기본 동작 확인)

순서:

1. **Task A** — ContextPool
2. **Task E 일부** — `:ctx` 명령 및 context visibility
3. **Task B** — sequence metadata
4. **Task C** — re-clustering (offline path)
5. **Task D** — hierarchical refine (**별도 실험 브랜치**)
6. **Task E 나머지** — refine control / traces / benches
7. **Task F** — image prototype (A~E 안정화 후)

### 왜 이 순서인가

- **Task A가 가장 먼저**: 다른 모든 작업의 기반. 여기서 회귀 없음 확인.
- **Task E 일부를 B 이전에**: `:ctx` 명령으로 Task A 검증.
- **Task B가 C 이전에**: sequence_id 없이는 재클러스터링 후 순서 복원 불가.
- **Task D는 별도 브랜치**: 실험 실패 시 main 보호.
- **Task F가 마지막**: 가장 실험적. 실패 비용이 프로젝트 전체에 영향 없도록.

---

## 절대 금지

다음은 **예외 없이** 금지:

- **existing `keyframes` / `deltas` / EMA를 추론 과정에서 mutate 금지.**
- **`ai_generate_next` 시그니처/동작 변경 금지.**
- **기존 `.spai` 태그 순서/레이아웃 직접 변경 금지.** 신규 태그 추가만 허용.
- **hard anchor overwrite 금지.** `CELL_ANCHOR` 상태 셀의 value는 어떤 경로에서도 덮어쓰지 말 것.
- **`topic_hash` bit distance를 의미 거리로 오해하지 말 것.**
- **image task에서 "already equivalent to diffusion model" 같은 주장 금지.**
- **한국어 형태소 사전 (`dict/`) 수정 금지.**
- **`perplexity` 지표 사용 금지.** 현재 엔진은 standard LM probability model이 아니다.

---

## 착수 전 확인

Task A 착수 전 다음을 모두 체크:

- [ ] `SpatialCanvas` / `SpatialCanvasPool` 사용처 grep 완료
- [ ] `ai_generate_next` 호출 경로 확인 완료
- [ ] `spatial_io.c` 저장/로드 태그 경로 확인 완료
- [ ] README/SPEC에서 keyframe 중심 baseline 재확인 완료
- [ ] refine 경로는 **baseline 대체가 아니라 병렬 실험 경로**임을 이해 완료
- [ ] image task는 **feasibility prototype** 수준임을 이해 완료

---

## 매 Task 종료 시 확인

- [ ] `make all` 경고 없이 빌드
- [ ] `make test` 69 tests 전체 PASS
- [ ] 수동 REPL 동작 확인
- [ ] 기존 `.spai` 파일 로드 정상 여부 확인
- [ ] 신규 테스트 추가되었는지 확인
- [ ] 커밋 메시지 명확히 (Task 번호 포함)

---

## 혼동이 생길 때 질문하기

다음 상황에서는 **즉시 멈추고 질문**:

- `ai_generate_next()`의 내부를 수정해야 할 것 같을 때
- `AggTables.A_sum`의 상위 N%를 일괄적으로 `CELL_ANCHOR`로 만들고 싶을 때
- `topic_hash` XOR 비트 카운트로 유사도를 계산하고 싶을 때
- 기존 `.spai` 파일 포맷을 바꿔야 해결될 것 같을 때
- `canvas_pool`만으로 학습된 지식에 접근할 수 있다고 확신할 때
- classical diffusion model의 수식을 그대로 가져오고 싶을 때

이런 충동은 대체로 **v4 수정 사항을 잊고 있다는 신호**다. `01_DESIGN_PRINCIPLES.md`로 돌아가 재확인할 것.
