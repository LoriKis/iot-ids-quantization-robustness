# Milestone 2 deadline experiment report

## Setup

- Model: PAVE Standard Transformer, frozen split_v2, seed 1 (one seed). Two FP32 parents (ordinary and adversarially trained) trained on the same 22412 training rows with the same optimizer, batch size and checkpoint rule (final epoch; no validation-based selection); each parent has one INT8 post-training-quantization (PTQ) copy calibrated on training rows only.
- Epochs per parent: 25, equal to the 25-epoch recipe.
- Rows: 22412 training, 3821 validation, 3764 test.
- Clean accuracy and macro-F1 use all 3764 test rows. Robustness uses a frozen stratified test subset of n=270 rows (target 10 per class); the same example IDs are used for every model. Selected rows per class (class index:count): 0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:10, 12:10, 13:10, 14:10, 15:10, 16:10, 17:10, 18:10, 19:10, 20:10, 21:10, 22:10, 23:10, 24:10, 25:10, 26:10.
- Attacks (offline, feature space): FGSM; PGD with 5 steps, 1 start(s), random start on, step size 0.25 x epsilon; epsilons 0.0, 0.05, 0.1. The epsilon-0 point is computed once on the subset and shared by both curves.
- FP32 models: white-box gradient of the FP32 model, scored on the same model. INT8 models: APPROXIMATE INT8 GRADIENT (BPDA): forward pass is the real INT8 model (asserted bit-equal logits); backward pass goes through each of the 23 INT8 Linear layers using their dequantized weights and ignores activation quantization; softmax, LayerNorm and residuals differentiate normally; candidates are scored by the real INT8 model. Not a straight-through estimator on fake-quant and not an FP32 shadow. Robustness of INT8 models may be over-estimated if this gradient is weak.

## Table 1. Four-model comparison

| Model | Parent | Precision | Epochs | Accuracy, full test (n=3764) | Macro-F1, full test | State size (MB) | Latency b1 median (ms) | Latency b1 p95 (ms) | Latency b256 per example (ms) | Peak RSS (MB) | Peak RSS delta (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32-ordinary (V1) | - | FP32 | 25 | 61.96% | 56.80% | 1.449 | 2.8759 | 3.1439 | 1.24162 | 446.44 | 122.03 |
| INT8-PTQ-ordinary (V2) | V1 | INT8 | 25 | 61.77% | 56.89% | 0.497 | 5.3579 | 5.8583 | 1.05289 | 475.71 | 135.82 |
| FP32-adversarial (V3) | - | FP32 | 25 | 53.85% | 49.90% | 1.449 | 2.9001 | 3.2030 | 1.24674 | 446.56 | 122.04 |
| INT8-PTQ-adversarial (V4) | V3 | INT8 | 25 | 53.93% | 49.94% | 0.497 | 5.3973 | 6.0385 | 1.08363 | 471.94 | 131.63 |

Notes:
- Values are medians over 3 repetitions; each repetition runs V1..V4 interleaved, every measurement in a fresh child process, on 1 thread(s) (batch 1: 20 warm-up and 200 timed calls; batch 256: 5 warm-up and 30 timed calls; 256 fixed validation rows).
- Size: bytes of the serialized state (torch.save of the state_dict for FP32; the packed state.pt for INT8). MB = 1,000,000 bytes.
- Latency: eval mode, inference mode, one thread; timed region is row -> repeat to 50 positions -> forward -> argmax. Batch 1 and batch 256 are in the CSV; the per-example figure is the batch-256 median divided by the batch size.
- Memory: peak_rss_bytes = Linux VmHWM of a fresh child process after the timed runs; peak_rss_delta_bytes = VmHWM minus RSS right after model load (the baseline includes the loaded weights and the torch footprint, so the delta is inference-time working memory); peak_rss_sampled_delta_bytes = 10 ms sampler cross-check; weight memory = state_dict_bytes. MB = 1,000,000 bytes.
- Accuracy and macro-F1 are on the full held-out test set (macro-F1 over all 27 labels, zero_division=0). The robustness-subset clean values are separate and appear in the robustness table.
- Epochs: 25 (recipe: 25). Seed 1, one run per parent.

## Table 2. Robustness on the subset (n=270)

| Model | Epsilon | FGSM accuracy | FGSM macro-F1 | PGD-5 accuracy | PGD-5 macro-F1 |
|---|---|---|---|---|---|
| FP32-ordinary | 0.0 (clean subset) | 68.52% | 64.54% | 68.52% | 64.54% |
| FP32-ordinary | 0.05 | 26.30% | 23.71% | 20.74% | 17.38% |
| FP32-ordinary | 0.1 | 6.67% | 6.27% | 4.44% | 4.29% |
| INT8-PTQ-ordinary (approx. BPDA gradient) | 0.0 (clean subset) | 66.67% | 62.38% | 66.67% | 62.38% |
| INT8-PTQ-ordinary (approx. BPDA gradient) | 0.05 | 28.52% | 26.14% | 18.52% | 16.09% |
| INT8-PTQ-ordinary (approx. BPDA gradient) | 0.1 | 7.41% | 7.17% | 4.81% | 4.37% |
| FP32-adversarial | 0.0 (clean subset) | 60.74% | 54.78% | 60.74% | 54.78% |
| FP32-adversarial | 0.05 | 48.52% | 43.63% | 48.52% | 43.63% |
| FP32-adversarial | 0.1 | 35.93% | 32.12% | 31.85% | 27.88% |
| INT8-PTQ-adversarial (approx. BPDA gradient) | 0.0 (clean subset) | 60.00% | 53.97% | 60.00% | 53.97% |
| INT8-PTQ-adversarial (approx. BPDA gradient) | 0.05 | 49.26% | 44.32% | 48.89% | 43.60% |
| INT8-PTQ-adversarial (approx. BPDA gradient) | 0.1 | 37.04% | 32.65% | 32.96% | 28.63% |

Accuracy under attack is the accuracy on the perturbed inputs. It can rise on rows the model got wrong when clean, so the curves are not guaranteed to fall. One subset example is 0.37 pp.

## Q1. What changed after PTQ?

**INT8-PTQ-ordinary vs FP32-ordinary.** On the full test set, accuracy is essentially unchanged (-0.19 pp) and macro-F1 is essentially unchanged (+0.09 pp). The INT8 model and its FP32 parent predict the same class on 3524 of 3764 test rows (93.62%); 240 rows differ. Under attack (FGSM and PGD-5 at epsilon 0.05 and 0.1), subset accuracy is higher in 2, lower in 1 and essentially unchanged in 1 of 4 settings.

| Condition | FP32-ordinary accuracy | INT8-PTQ-ordinary accuracy | Accuracy change (V2 vs V1) | FP32-ordinary macro-F1 | INT8-PTQ-ordinary macro-F1 | Macro-F1 change (V2 vs V1) |
|---|---|---|---|---|---|---|
| Clean, full test (n=3764) | 61.96% | 61.77% | essentially unchanged (-0.19 pp) | 56.80% | 56.89% | essentially unchanged (+0.09 pp) |
| Clean, robustness subset (n=270) | 68.52% | 66.67% | lower by 1.85 pp | 64.54% | 62.38% | lower by 2.16 pp |
| FGSM, eps 0.05 | 26.30% | 28.52% | higher by 2.22 pp | 23.71% | 26.14% | higher by 2.43 pp |
| FGSM, eps 0.1 | 6.67% | 7.41% | higher by 0.74 pp | 6.27% | 7.17% | higher by 0.90 pp |
| PGD-5, eps 0.05 | 20.74% | 18.52% | lower by 2.22 pp | 17.38% | 16.09% | lower by 1.29 pp |
| PGD-5, eps 0.1 | 4.44% | 4.81% | essentially unchanged (+0.37 pp) | 4.29% | 4.37% | essentially unchanged (+0.08 pp) |

**INT8-PTQ-adversarial vs FP32-adversarial.** On the full test set, accuracy is essentially unchanged (+0.08 pp) and macro-F1 is essentially unchanged (+0.05 pp). The INT8 model and its FP32 parent predict the same class on 3319 of 3764 test rows (88.18%); 445 rows differ. Under attack (FGSM and PGD-5 at epsilon 0.05 and 0.1), subset accuracy is higher in 3, lower in 0 and essentially unchanged in 1 of 4 settings.

| Condition | FP32-adversarial accuracy | INT8-PTQ-adversarial accuracy | Accuracy change (V4 vs V3) | FP32-adversarial macro-F1 | INT8-PTQ-adversarial macro-F1 | Macro-F1 change (V4 vs V3) |
|---|---|---|---|---|---|---|
| Clean, full test (n=3764) | 53.85% | 53.93% | essentially unchanged (+0.08 pp) | 49.90% | 49.94% | essentially unchanged (+0.05 pp) |
| Clean, robustness subset (n=270) | 60.74% | 60.00% | lower by 0.74 pp | 54.78% | 53.97% | lower by 0.81 pp |
| FGSM, eps 0.05 | 48.52% | 49.26% | higher by 0.74 pp | 43.63% | 44.32% | higher by 0.69 pp |
| FGSM, eps 0.1 | 35.93% | 37.04% | higher by 1.11 pp | 32.12% | 32.65% | higher by 0.53 pp |
| PGD-5, eps 0.05 | 48.52% | 48.89% | essentially unchanged (+0.37 pp) | 43.63% | 43.60% | essentially unchanged (-0.03 pp) |
| PGD-5, eps 0.1 | 31.85% | 32.96% | higher by 1.11 pp | 27.88% | 28.63% | higher by 0.75 pp |

## Q2. Did adversarial training help the INT8 model?

Comparing INT8-PTQ-adversarial with INT8-PTQ-ordinary: on the full test set, clean accuracy is lower by 7.84 pp and macro-F1 is lower by 6.94 pp. On the attacked subset, adversarial training raised INT8 accuracy under attack (higher in 4, lower in 0, essentially unchanged in 0 of 4 attack settings; differences below 0.5 pp count as unchanged).

| Condition | INT8-PTQ-ordinary accuracy | INT8-PTQ-adversarial accuracy | Accuracy change (V4 vs V2) | INT8-PTQ-ordinary macro-F1 | INT8-PTQ-adversarial macro-F1 | Macro-F1 change (V4 vs V2) |
|---|---|---|---|---|---|---|
| Clean, full test (n=3764) | 61.77% | 53.93% | lower by 7.84 pp | 56.89% | 49.94% | lower by 6.94 pp |
| Clean, robustness subset (n=270) | 66.67% | 60.00% | lower by 6.67 pp | 62.38% | 53.97% | lower by 8.41 pp |
| FGSM, eps 0.05 | 28.52% | 49.26% | higher by 20.74 pp | 26.14% | 44.32% | higher by 18.18 pp |
| FGSM, eps 0.1 | 7.41% | 37.04% | higher by 29.63 pp | 7.17% | 32.65% | higher by 25.48 pp |
| PGD-5, eps 0.05 | 18.52% | 48.89% | higher by 30.37 pp | 16.09% | 43.60% | higher by 27.51 pp |
| PGD-5, eps 0.1 | 4.81% | 32.96% | higher by 28.15 pp | 4.37% | 28.63% | higher by 24.26 pp |

INT8 attack gradients are approximate (BPDA); a robustness gain or loss for INT8 may not hold under stronger attacks.

## Q3. What did we gain or lose in size, speed and memory?

Each line compares the INT8 copy with its FP32 parent (ratio = INT8 / FP32; differences within 5% count as about the same).

- INT8-PTQ-ordinary vs FP32-ordinary, size (serialized state): 65.7% smaller (ratio 0.34x).
- INT8-PTQ-ordinary vs FP32-ordinary, batch-1 latency (median): 86.3% slower (ratio 1.86x).
- INT8-PTQ-ordinary vs FP32-ordinary, batch-256 latency per example (median): 15.2% faster (ratio 0.85x).
- INT8-PTQ-ordinary vs FP32-ordinary, peak RSS (VmHWM): 6.6% larger (ratio 1.07x).
- INT8-PTQ-ordinary vs FP32-ordinary, peak RSS delta above the post-load baseline: 11.3% larger (ratio 1.11x).
- INT8-PTQ-adversarial vs FP32-adversarial, size (serialized state): 65.7% smaller (ratio 0.34x).
- INT8-PTQ-adversarial vs FP32-adversarial, batch-1 latency (median): 86.1% slower (ratio 1.86x).
- INT8-PTQ-adversarial vs FP32-adversarial, batch-256 latency per example (median): 13.1% faster (ratio 0.87x).
- INT8-PTQ-adversarial vs FP32-adversarial, peak RSS (VmHWM): 5.7% larger (ratio 1.06x).
- INT8-PTQ-adversarial vs FP32-adversarial, peak RSS delta above the post-load baseline: 7.9% larger (ratio 1.08x).

## Limitations

- One seed (seed 1) and one training run per parent. No cross-seed variation is measured or reported, so score differences are single-run observations.
- Actual epochs: 25 per parent, equal to the 25-epoch recipe.
- Small robustness subset: actual n=270 (target 10 per class, up to 270). One example equals 0.37 pp, so differences of a few examples are within noise. Classes below the target: none.
- Fixed attack budget: FGSM and PGD-5 with 1 start(s), step size 0.25 x epsilon, epsilons 0.05, 0.1. Stronger attacks, more restarts and other attack sources were not run.
- Perturbations are in feature space and do not guarantee valid network traffic.
- INT8 gradients are approximate (APPROXIMATE INT8 GRADIENT (BPDA)); robustness numbers for INT8 models may differ under other gradient estimators.
- Efficiency was measured on one machine with 1 thread(s). The memory baseline includes the loaded weights and the torch footprint, so the RSS delta shows inference-time working memory, not total footprint.
- Recorded epsilon-0 consistency (subset predictions vs full-test predictions of the same model; record-only, never used to stop the run): V1: 0 of 270 rows differ (max abs logit difference not recorded); V2: 0 of 270 rows differ (max abs logit difference not recorded); V3: 0 of 270 rows differ (max abs logit difference not recorded); V4: 0 of 270 rows differ (max abs logit difference not recorded).
- Extra seeds, transfer studies, layer ablations and full QAT training were not run.

## QAT feasibility

The existing tiny QAT feasibility test (`artifacts/m2a-pilot/qat/qat_feasibility.json`) reports status 'supported', 23 converted layers, serialization flag True, and ran 1 epoch(s) on 2,700 training rows. This demonstrates the feasibility of the conversion, serialization and INT8 inference path only; no QAT performance result is claimed.

## Provenance

- run_id: m2-deadline-v1
- config_sha256: 4f12845e86270944be7c9abd885c9af6c9cfb98282ae3a5f4581c0c0fd1c888a
- lock_sha256: 1b1ced84ab3a68569c2ace4654b468c6679926dc7546bf7752da4fcfd72f6ccb
- protocol_sha256: 62adf121b39054f8109fd7c89a2c3577a4303f9f1610d60742521fd7dcf47b36
- subset_ids_sha256: 2ca2619321ba903a900904e1e4a13bf875d0fcbc7072a71005dac925e1900a2d
- v4_manifest_sha256: 5f49c93af30704a181ea77d3c09c23e99062accee40efc6c93ec9365e8a88d99
