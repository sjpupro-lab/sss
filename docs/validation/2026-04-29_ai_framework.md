# AI Framework Validation Summary

Branch: `claude/validate-ai-framework-dbyUb`  Date: 2026-04-29

## 1. Unit tests (ce_core) — ALL PASS

| target                   | result   | notable assertion |
|--------------------------|----------|-------------------|
| test_hybrid_vae          | 16/16    | solid PSNR 6.3 dB / synth PSNR 13.2 dB |
| test_masked_train        | 14/14    | "loss bounded" only — does **not** assert monotonic decrease |
| test_slig_persist        | ALL PASS | CE_TYPE_SLIG roundtrip + tick-sort byte-for-byte |
| test_residual_codebook   | ALL PASS | descriptor pack/unpack + 16-bit position roundtrip |
| test_residual_decode     | ALL PASS | residual book changes 984 px / max-dev 77 |

## 2. End-to-end pipeline

`train_demo data/demo build/models/demo_cb --masked-epochs 50 --residual-codebook` →
10 rows, 21 419 entries, 96 RESIDUAL descriptors (codebook size = 5 patterns).

`gen_image_ce demo_cb.ces "red apple" out.ppm 0 50 200 --hybrid` runs and writes a 256×256 PPM.

## 3. The four user-stated quality checks

### 3.1 SLIG detail actually loads → ✓ but not enough
`ce_storage_load_slig_sets` returns 23–123 SLIG cells per cid. SLIG persist test confirms byte-for-byte roundtrip.

### 3.2 residual ON vs OFF differs → ⚠ partially
- During training: per-image final loss is **byte-identical** between cb_on and cb_off (76.14, 65.13, 62.42 …) — codebook only changes how cells are stored, never how loss is measured.
- At decode: PPMs differ in 640 bytes — but only because masked-train uses CE_TYPE_SLIG (cb_on) vs CE_TYPE_IMAGE (cb_off) and the storage-key encodings collide differently with the SLIG loader. Not the intended residual-stamp effect.
- **Root cause**: `gen_image_ce.c` never sets `hcfg.residual_book`, and `ce_storage_save/load` does not persist `CEResidualCodebook`. So the residual stamping branch in `hybrid_vae_decode` (lines 476–539) is a no-op at inference. Codebook patches exist in RAM during training and are thrown away.

### 3.3 mask epoch ↑ → loss ↓ → ✗
- epochs=0, 10, 50 produce **byte-identical** generated PPMs (`cb_50 == e0`, `e10 == e50`).
- `epochs_run` caps at 5–11 because of `loss_patience = 3` early stopping; `target_loss = 8.0` is never reached (stuck around 60–80).
- `masked_train_image` "best predicted" cells *are* written to storage, but the SLIG sets that the decoder reads come from the original `hybrid_vae_encode` pass, not from the masked-train output.

### 3.4 codebook composition vs canvas_id recall → ✗ recall confirmed
| prompt          | cid routed       | md5 of output                       |
|-----------------|------------------|-------------------------------------|
| `red`           | 0x731e137f       | a96143db…                           |
| `red apple`     | 0x731e137f       | a96143db… (**identical to "red"**)  |
| `apple`         | 0xa366baaf       | ea6a682b…                           |
| `yellow banana` | 0xfa7a09fe       | 1214e941… (routes to "yellow")      |

Multi-morpheme prompts collapse to the single highest-voting morpheme's cid. There is no mixing across cids — pure canvas_id recall.

## 4. Concrete gaps to close

1. **Persist residual codebook**: extend `ce_storage_save/load` to serialize `CEResidualCodebook`, and have `gen_image_ce` set `hcfg.residual_book = &loaded_book`.
2. **Wire masked-train output into decode**: either store the converged best_predicted as canonical CE_TYPE_SLIG entries that `ce_storage_load_slig_sets` actually picks up (slot/block_idx encoding currently doesn't match the encoder's `(scale, channel)` layout), or read masked cells directly in `hybrid_decode_detail_only`.
3. **Bypass early stopping for the loss trajectory test**: expose `loss_patience` as a CLI arg, or add a sweep tool that captures loss per epoch via the `on_epoch` callback.
4. **Compositional voting**: `vote_canvas_id` currently winner-takes-all. To compose "red" + "apple" you need either multi-cid blending in `hybrid_vae_decode`, or a TEXT→token cross-attention into the codebook rather than a single cid lookup.

## 5. What works (don't lose this)

- 16×16 atomic block stamps roundtrip correctly (test_hybrid_vae).
- SLIG cells persist losslessly to .ces and reload (test_slig_persist).
- Residual codebook in-memory does change the decoded plane when wired (test_residual_decode, 984 px / max-dev 77).
- Masked-train loss is bounded and converges within patience window.
- Canvas routing per prompt morpheme is deterministic and reproducible.

So Phase 1–13 plumbing is in. The output is wired, but four wires (codebook persist, masked→decode, patience, multi-cid blend) still need to be soldered before the system is a generator instead of a recall lookup.
