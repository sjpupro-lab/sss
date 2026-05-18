# restore/pre-phase1-pr39 브랜치 안내

## 목적

`main`은 Phase 5 (단일 `sss_gen` 진입점 + `feature_bank.sfb` 기반)을
유지한다. 그러나 Phase 1 (PR #44, 커밋 `78be7f7`)에서 학습 위상
저장을 폐기하고 amp-only radio tuning으로 전환한 이후, 64×64 sanrio
데이터 25장으로 학습한 모델이 형태/색이 보이지 않는 격자 패턴만
출력하는 문제가 발생했다.

이 브랜치는 **PR #39 머지 직전 상태**(`490c837`,
`sss: 25%×4 subband decomposition + cross-attention restore pipeline`)
를 그대로 복원한 것이다. 25%×4 Haar subband 복원 + sculpt 후보
경쟁이 살아 있던 시점이라 형태 합성이 가능하다.

`main`은 변경되지 않는다. 두 브랜치는 독립적으로 공존한다.

## 동작 확인된 기능

- `tools/sss_ingest.py` — 25%×4 Haar subband + canny + color + amp +
  phase + fp 셀 5종을 ROW_BLOCK 단위로 저장
- `tools/sss_unified.CEMemory` — `subbands` kwarg 포함 셀 저장/검색
- `tools/sss_restore_infer.py` / `tools/sss_restore.py` — PR #38
  cross-attention 기반 row-level restore
- `scripts/sss_train.py` — `.sss` 모델 학습 (낮은 freq + 위상 포함)
- `tools/sss_gen.c` → `build/sss_gen` — sculpt 후보 경쟁 + 후처리

## 사용 불가 기능

`main`에서 추가된 다음 컴포넌트는 이 브랜치에 존재하지 않는다.

- `feature_bank.sfb` 포맷 (PR #45, Phase 2)
- motif trainer + `.sfb` compositor (PR #47, Phase 3)
- condition-interaction paradigm + sfb v2 (PR #48, Phase 4)
- Phase 5 단일 진입점 통합 (PR #50)

## Termux / Linux 사용 가이드

```sh
cd ~/sss
git fetch origin
git checkout restore/pre-phase1-pr39

# 빌드 (pybridge는 학습 시 필요)
make sss_gen
make pybridge

# 학습 (sanrio 25장 → .sss 모델)
python3 scripts/sss_train.py \
    --labels data/sanrio/labels.tsv \
    --root data/sanrio \
    --out build/sanrio.sss \
    --size 64

# 생성: MODEL.sss PROMPT OUT.ppm [seed] [detail] [steps]
./build/sss_gen build/sanrio.sss "kitty"   out_kitty.ppm    1 1.5 60
./build/sss_gen build/sanrio.sss "mymelody" out_mymelody.ppm 1 1.5 60
./build/sss_gen build/sanrio.sss "keroppi" out_keroppi.ppm  1 1.5 60

# PPM → PNG 변환 (ImageMagick 또는 Pillow)
magick out_kitty.ppm out_kitty.png
# 또는
python3 -c "from PIL import Image; Image.open('out_kitty.ppm').save('out_kitty.png')"

# Termux에서 갤러리로 내보내기
cp out_kitty.png ~/storage/downloads/
```

### 학습 검증 (이 브랜치에서 측정)

`sanrio` 25장 / 64×64 / detail=1.5 / steps=24:

- 모델 크기: 212,763 bytes (9 cells, NF=33, NF_LOW=11)
- seed 변동: seed 1~5 간 평균 픽셀 차이 4.4~4.6 (격자 고정 아님)
- 프롬프트 변동: kitty / mymelody / keroppi 간 평균 픽셀 차이 34~39
- RGB 채널이 서로 다름 (흑백 아님)
- 시각 확인: kitty 실루엣 + 리본, mymelody 토끼 귀, keroppi 눈 등
  학습 모티프가 식별 가능

## `main`과의 관계

| | `main` | `restore/pre-phase1-pr39` |
|---|---|---|
| 베이스 커밋 | `bf467a9` (PR #50) | `490c837` (PR #39) |
| 모델 포맷 | `feature_bank.sfb` | `.sss` |
| 진입점 | `sss_gen` 통합 (Phase 5) | 옛 `sss_gen.c` |
| 위상 저장 | 폐기 (amp-only) | 유지 |
| Subband 25%×4 | — | 사용 |
| Sculpt 후보 경쟁 | — | 사용 |
| 형태 합성 | 격자 패턴만 출력 | 모티프 식별 가능 |

이 브랜치는 **`main`을 대체하지 않는다.** 형태 합성이 필요한 작업에
한해 일시적으로 체크아웃해서 사용하고, 평소에는 `main`을 유지하는
것을 전제로 한다. Phase 1 회귀의 원인 분석이 끝나면 폐기 가능.

## 브랜치 전환

```sh
# 이 브랜치로 전환
git checkout restore/pre-phase1-pr39
make clean && make sss_gen && make pybridge

# main으로 복귀
git checkout main
make clean && make sss_gen
```
