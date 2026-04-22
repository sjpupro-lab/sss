# SPATIAL-PATTERN-AI 확장 구현 프롬프트 (v4, corrected) — 분할본

> Claude Code에게: 이 프롬프트는 `sss` 레포(SPATIAL-PATTERN-AI / CANVAS) 위에 **ContextPool**, **Hierarchical Draft Refinement**, **Canvas Re-clustering**, **Image Modality Prototype**을 추가하기 위한 구현 지침입니다. 기존 baseline 경로를 **보존**하고, 새 기능은 **추가 경로**로 구현하세요.

---

## 이 패키지의 구성

긴 지침서를 Task 단위로 분할했습니다. **권장 읽기 순서는 파일명 번호 순서**입니다.

| 파일 | 내용 | 성격 |
|---|---|---|
| `00_README.md` | 이 문서. 전체 구조 안내 | 메타 |
| `01_DESIGN_PRINCIPLES.md` | v4 핵심 수정, 4대 원칙 | **필독** |
| `02_BASELINE.md` | 현재 레포 아키텍처 기준선 | **필독** |
| `03_TASK_A_CONTEXT_POOL.md` | Task A — 별도 작업 기억 공간 | 최우선 구현 |
| `04_TASK_B_SEQUENCE_METADATA.md` | Task B — 시퀀스 메타데이터 | C의 선행 |
| `05_TASK_C_RECLUSTERING.md` | Task C — 주기적 재클러스터링 | offline |
| `06_TASK_D_DRAFT_REFINEMENT.md` | Task D — 계층적 초안 정제 | **★ 실험적, 별도 브랜치** |
| `07_TASK_E_REPL_INTEGRATION.md` | Task E — REPL/도구 통합 | 교차 |
| `08_TASK_F_IMAGE_PROTOTYPE.md` | Task F — 이미지 모달리티 프로토타입 | 마지막 |
| `09_CHECKLIST_AND_ORDER.md` | 파일 체크리스트, 수용 기준, 구현 순서, 절대 금지 | **필독** |
| `10_GLOSSARY.md` | 용어 가이드 | 참조용 |

---

## 착수 전 반드시 할 일

1. **`01_DESIGN_PRINCIPLES.md`을 먼저 읽는다.** v4가 v1~v3의 어떤 단정을 수정했는지 파악하지 않으면 같은 오류를 반복한다.
2. **`02_BASELINE.md`의 필독 목록을 실제로 읽는다.** `ai_generate_next()`의 현재 구현을 모르면 "대체가 아니라 병렬 경로"의 의미를 이해할 수 없다.
3. **`09_CHECKLIST_AND_ORDER.md`의 구현 순서를 준수한다.** Task A → E일부 → B → C → D → E나머지 → F 순. Task D는 별도 브랜치.
4. 각 단계 종료 시 `make all` + `make test`로 **69 tests 기준 회귀 없음** 확인.

---

## 한 장으로 보는 철학

- **장기 기억은 기존 `SpatialAI`의 `keyframes` / `deltas` / EMA다.** `SpatialCanvasPool`은 보조 컨텍스트 계층일 뿐.
- **`ai_generate_next()`는 유지한다.** `ai_generate_refine()`는 실험적 병렬 경로로 추가.
- **`AggTables.A_sum` 고값 셀은 strong prior seed이지, 기본 hard anchor가 아니다.**
- **hard anchor는 입력 프롬프트에서 직접 유도된 셀에 한정한다.**
- **`topic_hash`는 exact-match 1차 grouping에만 쓴다.** bit distance를 의미 거리로 해석하지 말 것.
- **이미지 task는 feasibility prototype.** diffusion equivalent를 약속하지 말 것.
- **"classical diffusion" 주장 금지.** 필요 시 "denoising-like refinement"로만.

---

## 질문이나 혼동이 생기면

다음 중 하나에 해당하면 **즉시 멈추고 질문하라**:

- `ai_generate_next()`의 동작을 바꿔야 할 것 같다 → 멈춰라. 병렬 경로가 맞는지 재확인.
- `AggTables.A_sum` 최댓값을 전체 셀에 덮어쓰고 싶어진다 → 멈춰라. hard anchor 규칙 재확인.
- `topic_hash`의 XOR/해밍 거리를 쓰고 싶어진다 → 멈춰라. exact match + delta/RLE 비용으로 대체.
- 기존 `.spai` 바이너리 레이아웃을 바꿔야 할 것 같다 → 멈춰라. 신규 태그 추가로 해결.

준비가 끝났으면 `03_TASK_A_CONTEXT_POOL.md`부터 시작하세요.
