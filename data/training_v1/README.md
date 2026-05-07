# training_v1 — 확장 학습 데이터셋

본 디렉터리는 사용자가 업로드한 `ssstrainingdata.zip` (sss_data) 원본 학습 세트와,
이를 보강하기 위해 자동 생성된 확장 데이터를 함께 담고 있습니다. 모든 이미지는
**64×64 PPM(P6)** 포맷으로, 기존 `sss/data/sss_demo`/`sss_demo_1k` 와 동일한
규격이며 `images/labels.tsv` 의 `파일명\t라벨` 쌍으로 학습에 바로 사용할 수
있습니다.

## 구성 요약

| 항목 | 파일/폴더 | 설명 |
|---|---|---|
| 한국어 기초 문장 | `korean_train.txt` | 색·형상·동물·표정·풍경을 다루는 한국어 문장 모음 (확장 12문장 추가) |
| 영어 기초 문장 | `descriptive_en.txt` | 동일 의미축의 영어 캡션 코퍼스 (확장 12문장 추가) |
| 이미지 라벨 | `images/labels.tsv` | TSV 형식, 좌측 PPM 파일명 / 우측 자유 텍스트 라벨 |
| 이미지 자료 | `images/*.ppm` | 색×형상×표정, 동물, 캐릭터, 모션, 풍경, 그라데이션, 꽃 등 |

## 라벨 카테고리 (확장 후 490 라인)

| 카테고리 | 개수 | 비고 |
|---|---:|---|
| 색×형상×표정 (red/blue/green/yellow ×4도형 ×4표정) | 80 | 원본 |
| 추가 색상(brown/pink/purple/orange/cyan/white) | 96 | 원본 |
| 동물 (8종 × happy/sleepy) | 16 | 원본 |
| 꽃 (rose/tulip/sunflower/daisy/violet/lily ×3 variant) | 18 | 원본 |
| mix·gradient·landscape | 42 | 원본 |
| 캐릭터 (skin × hair × outfit × emotion) | 32 + **180 확장** | 분홍 머리/탠 피부/보라 코트 신규 추가 |
| 모션 (walk / wave / petal × 8 frame) | 24 | 원본 |
| 모션 (jump / spin / bow × 8 frame) | **24 확장** | 신규 |

## 확장 데이터 생성 방법

확장 캐릭터·모션 PPM 은 `extend_dataset.py`(레포 외부 빌드 스크립트)로 생성되었으며,
기존 데이터셋과 동일한 64×64 캔버스, 동일한 색 팔레트 컨벤션을 따릅니다. 신규 항목은
다음 축으로 다양성을 확장합니다.

- **신규 헤어컬러**: `pink_hair`
- **신규 피부톤**: `tan`
- **신규 의상**: `purple_coat`
- **신규 모션**: `jump`(점프 포물선), `spin`(회전 + 모션 라인), `bow`(인사 숙임)

## 사용 예시

```bash
# 라벨 텍스트 빠르게 확인
head data/training_v1/images/labels.tsv

# 한국어 문장 라인 수
wc -l data/training_v1/korean_train.txt
```

라벨은 공백으로 구분된 토큰 시퀀스이므로 `tools/sss_ingest.py` 의 CSV 인제스트
경로(`path,label`)와 호환되도록 간단한 변환만으로 바로 학습에 투입할 수
있습니다.
