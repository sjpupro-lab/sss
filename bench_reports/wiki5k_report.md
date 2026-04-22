# CANVAS — 5000 clause benchmark report

**Corpus**: `data/wiki5k.txt` — 5,500 English Wikipedia intro clauses (593 KB, fetched via MediaWiki REST API, 10+ bytes / clause, deduped).
**Host**: Windows 11, MSYS2 gcc 15.2.0, `-O2 -std=gnu11`.
**Model size on disk**: 1,525 MB (`build/models/wiki5k.spai`).

## 1. Streaming ingest (`stream_train`, 5,000 clauses)

| Metric | Value |
|---|---|
| Clauses ingested | 5,000 (1 skipped) |
| Keyframes (I) | 4,649 |
| Deltas (P) | 351 |
| Active cells (total) | 530,658 |
| Ingest time | 75.83 s |
| Throughput | ~66 c/s (last window: 78 c/s) |
| Wall time (incl. verify) | 93.35 s |
| Model file | 1,525 MB |

**Verify pass (unseen query on last 500 clauses)**
- Matched (>0): 500 / 500
- Mean similarity: 0.1222 (min 0.0351 / max 1.0000)
- Hits ≥ 0.90: 1 · ≥ 0.50: 1 · ≥ 0.10: 344

## 2. `bench_word_predict` (70/30 split, 5,000 clauses)

| Metric | Value |
|---|---|
| Train / Test clauses | 3,500 / 1,500 |
| In-vocab predictions | 14,699 |
| OOV skipped | 6,592 |
| **Top-1 accuracy** | **2.79 %** (410 / 14,699) |
| **Top-5 accuracy** | **4.29 %** (630 / 14,699) |
| Random baseline (avg 333 same-length cands) | 0.30 % |
| **Lift over random** | **9.3×** |
| Word-level perplexity | 2,222.06 |
| Prediction speed | 3,323 preds/s |

## 3. `bench_perplexity` (byte-level, 5,000 clauses)

Test set: 500 held-out clauses (55,340 bytes).

| Metric | Value |
|---|---|
| Avg NLL | 3.5119 nats/byte |
| Entropy | 5.0666 bits/byte |
| **Byte-level perplexity** | **33.51** (baseline 256) |
| Unseen byte events | 1.10 % |
| ASCII (<0x80) PPL | 31.16 (n=54,572) |
| CJK UTF-8 leaders PPL | 3,677.71 (n=74) |
| Other UTF-8 bytes PPL | 6,223.51 (n=694) |
| Active cells | 18,522 / 65,536 (28.3 %) |
| Training time | 0.5 s (4,500 clauses) |

## 4. Skipped

- `bench_stsb` — `data/stsb.tsv` not present.
- `bench_qa` — `data/qa.tsv` not present.

## Observations

- Ingest cost is dominated by keyframe insertion; with threshold 0.30, the model classifies 93 % of clauses as new keyframes (4,649 I vs 351 P) on this heterogeneous Wikipedia corpus. A lower threshold (e.g. 0.15) would likely trade accuracy for a tighter model.
- Byte-level model generalises well to ASCII (PPL 31) but poorly to UTF-8 multi-byte (PPL 3 k–6 k) because the English corpus contains almost no CJK content — a consistent result.
- Word-prediction Top-1 (2.79 %, 9.3× random) is "clear signal" per the engine's own threshold but well below token-LM baselines; this reflects the design (frame-match, no softmax, candidate sets of ~333 words).
