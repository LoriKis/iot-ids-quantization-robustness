# Milestone 2A pilot report

> **Pilot results are non-scientific development checks (1 seed, 2 epochs, small subsets, validation data only) and carry no accuracy or robustness conclusion.**

This is a new group-disjoint evaluation partition within a previously examined dataset, not a completely untouched external benchmark. The dataset was seen by Milestone 1; the earlier test partition was evaluated; identical feature vectors with conflicting labels remain in the data.

Feature-space attacks; no guarantee that perturbed vectors correspond to valid network traffic. Perturbations are tied to the original feature row (one 39-d delta broadcast to all 50 synthetic positions).

## Setup

- seed 1, 2 epochs, batch size 512, training threads 2, attack and evaluation threads 1
- split seed 20260, split manifest SHA-256 `7275ede47c840d8820d6d5892ff6886af8ab23de9373f868a3e2d1c62c969cd2`
- source digest `2c757a08da27d1c9d8a896e0d80d4ddcd3c21bd22aca13d7d8843d2eaa73f8e7`, git revision `466faf724aaab5507d5cf12f74281c5f2a62ae28` (no commit made)
- test partition: not loaded, not evaluated (guard phase `m2a` allows train and validation only)

| subset | rows | ID SHA-256 |
|---|---|---|
| train | 2700 | 8c81bd065ba12e39... |
| validation | 540 | 72038babb091b00a... |
| attack | 270 | 0f7dc8db74d0d34a... |
| calibration | 540 | 1ceef169151c888f... |
| e4_plumbing | 54 | 3d649eb0e1cda0a2... |

## Training (V1 ordinary, V3 adversarial)

| variant | updates | wall clock s | mean step s (first discarded) | final loss | init state hash | final state hash |
|---|---|---|---|---|---|---|
| V1 | 12 | 23.8884 | 1.8591 | 3.1341 | e07454f5ec4d | fea8857c297c |
| V3 | 12 | 158.7413 | 12.5802 | 3.1807 | e07454f5ec4d | b5e2ec278bf4 |

V1 and V3 share the same initial weights: **True**; equal optimizer updates: **True**.

AT extra compute: measured wall-clock ratio 6.77x (first step of each run discarded, 11 steps, indicative only); nominal pass-count ratio 10.67x. Counters per AT step: {'attack_grad_passes_per_step': 6, 'attack_scoring_forwards_per_step': 8, 'train_forward_backward_per_step': 2}; first-step forward-hook counts on the real model: {'attack_side_forwards': 14, 'train_side_forwards': 2, 'expected_attack_side': 14, 'expected_train_side': 2, 'matches_logged_counters': True}.

## Clean evaluation on the validation subset

| model | accuracy | rows |
|---|---|---|
| V1 | 0.1426 | 540 |
| V3 | 0.1056 | 540 |
| V2 | 0.1463 | 540 |
| V4 | 0.1056 | 540 |

INT8 vs FP32 parent argmax agreement: V2_vs_V1 0.9907, V4_vs_V3 0.9963

## INT8 post-training quantization evidence

### V2 (from V1)

- backend `fbgemm`, mode `static`, scope `all_linear`, calibration 540 train rows (ID SHA-256 `1ceef169151c888f...`)
- islands converted: 23 (21 classification path, 2 anomaly scorer); packed weights {'torch.qint8': 23} {'torch.per_channel_affine': 23}
- profiler quantized::linear calls on one forward: 23
- identity wrapper max abs error before observers: 0.0; reload check: bitwise logits True, argmax identical True over 540 rows
- INT8 weight elements 347632 of 351612 parameters (98.87 %); quant/dequant boundaries 46

### V4 (from V3)

- backend `fbgemm`, mode `static`, scope `all_linear`, calibration 540 train rows (ID SHA-256 `1ceef169151c888f...`)
- islands converted: 23 (21 classification path, 2 anomaly scorer); packed weights {'torch.qint8': 23} {'torch.per_channel_affine': 23}
- profiler quantized::linear calls on one forward: 23
- identity wrapper max abs error before observers: 0.0; reload check: bitwise logits True, argmax identical True over 540 rows
- INT8 weight elements 347632 of 351612 parameters (98.87 %); quant/dequant boundaries 46

Operations that remain FP32 (hybrid model, not integer-only):

- attention QK^T matmul
- attention softmax
- attention AV matmul
- LayerNorm
- residual adds
- ReLU
- sinusoidal positional-encoding add
- mean pooling over the 50 positions
- anomaly-scorer sigmoid
- dropout (inactive in eval)
- inter-island activations (each island quantizes its input and dequantizes its output)
- Linear biases (kept in FP32 inside the packed parameters)

## Adversarial evaluation (validation attack subset)

Attack subset: 270 rows. Invalid units: 0. Sources are never merged; `bpda_int8` is the headline INT8 row (forward through the real INT8 model, approximate backward pass).

### fgsm, models V1/V2: robust accuracy by epsilon

| source | eps 0 | eps 0.01 | eps 0.025 | eps 0.05 | eps 0.1 |
|---|---|---|---|---|---|
| V1 fp32_direct | 0.1370 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |
| V2 transfer_fp32_to_int8 | 0.1407 | 0.1370 | 0.1333 | 0.1333 | 0.1296 |
| V2 transfer_fp32_to_int8_targetsel | 0.1407 | 0.1370 | 0.1333 | 0.1333 | 0.1296 |
| V2 bpda_int8 | 0.1407 | 0.1370 | 0.1333 | 0.1333 | 0.1296 |
| V2 surrogate_fakequant | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |
| V2 surrogate_fakequant_targetsel | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |

### fgsm, models V3/V4: robust accuracy by epsilon

| source | eps 0 | eps 0.01 | eps 0.025 | eps 0.05 | eps 0.1 |
|---|---|---|---|---|---|
| V3 fp32_direct | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 transfer_fp32_to_int8 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 transfer_fp32_to_int8_targetsel | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 bpda_int8 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 surrogate_fakequant | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 surrogate_fakequant_targetsel | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |

### pgd20, models V1/V2: robust accuracy by epsilon

| source | eps 0 | eps 0.01 | eps 0.025 | eps 0.05 | eps 0.1 |
|---|---|---|---|---|---|
| V1 fp32_direct | 0.1370 | 0.1333 | 0.1333 | 0.1333 | 0.1259 |
| V2 transfer_fp32_to_int8 | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |
| V2 transfer_fp32_to_int8_targetsel | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |
| V2 bpda_int8 | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1259 |
| V2 surrogate_fakequant | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1296 |
| V2 surrogate_fakequant_targetsel | 0.1407 | 0.1333 | 0.1333 | 0.1333 | 0.1259 |

### pgd20, models V3/V4: robust accuracy by epsilon

| source | eps 0 | eps 0.01 | eps 0.025 | eps 0.05 | eps 0.1 |
|---|---|---|---|---|---|
| V3 fp32_direct | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 transfer_fp32_to_int8 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 transfer_fp32_to_int8_targetsel | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 bpda_int8 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0926 |
| V4 surrogate_fakequant | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0963 |
| V4 surrogate_fakequant_targetsel | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0926 |

Per-example union rows (an example counts as robust only if no INT8 source broke it; separately labelled):

| target | attack | epsilon | robust accuracy |
|---|---|---|---|
| V2 | fgsm | 0.0100 | 0.1333 |
| V2 | fgsm | 0.0250 | 0.1333 |
| V2 | fgsm | 0.0500 | 0.1333 |
| V2 | fgsm | 0.1000 | 0.1296 |
| V2 | pgd20 | 0.0100 | 0.1333 |
| V2 | pgd20 | 0.0250 | 0.1333 |
| V2 | pgd20 | 0.0500 | 0.1333 |
| V2 | pgd20 | 0.1000 | 0.1259 |
| V4 | fgsm | 0.0100 | 0.1000 |
| V4 | fgsm | 0.0250 | 0.1000 |
| V4 | fgsm | 0.0500 | 0.1000 |
| V4 | fgsm | 0.1000 | 0.0963 |
| V4 | pgd20 | 0.0100 | 0.1000 |
| V4 | pgd20 | 0.0250 | 0.1000 |
| V4 | pgd20 | 0.0500 | 0.1000 |
| V4 | pgd20 | 0.1000 | 0.0926 |

### Gradient-masking diagnostics and surrogate fidelity

- V2 `direct_int8_grad`: raises RuntimeError (no grad_fn through quantized operators); undefined/zero fraction 1.0000 over 270 rows; counted as robust: False
- V2 surrogate: surrogate scales differ from deployed INT8 scales (max diff = 0.00408237); max logit difference to the INT8 model 0.053297; argmax agreement 0.9926
- V4 `direct_int8_grad`: raises RuntimeError (no grad_fn through quantized operators); undefined/zero fraction 1.0000 over 270 rows; counted as robust: False
- V4 surrogate: surrogate scales differ from deployed INT8 scales (max diff = 0.00396915); max logit difference to the INT8 model 0.034515; argmax agreement 0.9963

## E4 budget-adequacy plumbing (not an adequacy decision)

plumbing check only; establishes nothing about adequacy. Plumbing subset 54 rows; threshold would be 1. `pgd20` stays provisional; adequacy is decided in campaign step 0 on fully trained checkpoints. Real campaign E4 subset size for this split: 1727 rows (data.stratified_cap with per_class_cap=70).

| model | epsilon | n_extra | pgd20 s | pgd50x5 s |
|---|---|---|---|---|
| V1 | 0.0250 | 0 | 14.9512 | 58.4105 |
| V1 | 0.0500 | 0 | 18.8065 | 75.8022 |
| V1 | 0.1000 | 0 | 20.4728 | 94.1042 |
| V2 | 0.0250 | 0 | 31.4100 | 125.9277 |
| V2 | 0.0500 | 0 | 37.3901 | 140.4510 |
| V2 | 0.1000 | 0 | 30.3170 | 119.0564 |
| V3 | 0.0250 | 0 | 27.7922 | 96.1965 |
| V3 | 0.0500 | 0 | 21.7953 | 96.2118 |
| V3 | 0.1000 | 0 | 22.5026 | 117.3388 |
| V4 | 0.0250 | 0 | 40.1358 | 129.2008 |
| V4 | 0.0500 | 0 | 33.8355 | 117.7598 |
| V4 | 0.1000 | 0 | 27.2175 | 96.7530 |

## Efficiency (1 thread, fresh process per measurement)

Executed order: V1, V2, V3, V4, V1, V2, V3, V4 (interleaved: True, retries flagged *). Reduced pilot repetitions: 2; warm-up/timed {'batch1': {'warmup': 10, 'timed': 50}, 'batch256': {'warmup': 2, 'timed': 10}}.

| variant | state_dict MB | bs1 median ms | bs1 p95 ms | bs256 median ms | bs256 samples/s | baseline RSS MB | peak wset - baseline MB |
|---|---|---|---|---|---|---|---|
| V1 | 1.4487 | 4.6001 | 6.1719 | 437.4014 | 587.5704 | 269.0621 | 115.2573 |
| V2 | 0.4974 | 11.1168 | 13.8334 | 423.4516 | 609.3983 | 279.1731 | 112.8468 |
| V3 | 1.4487 | 6.4609 | 10.6817 | 447.8203 | 598.9918 | 269.6745 | 115.2737 |
| V4 | 0.4974 | 13.4603 | 19.4716 | 483.1949 | 529.9911 | 279.0584 | 112.9615 |

Outcomes are descriptive observations on one laptop CPU; no speed or memory advantage is assumed or claimed.
Process-wide RSS is coarse and includes allocator slack and imported packages.

## QAT feasibility (optional V5, bounded check)

- status **supported**, stage `inference`, 1 epoch on 2700 train rows, 34.6 s
- error: None; module: None
- packed dtypes: {'torch.qint8': 23}; INT8 inference: {'rows': 540, 'logits_finite': True, 'argmax_agreement_with_fp32_parent': 0.894444465637207, 'accuracy_on_eval_rows': 0.1796296238899231}
- bounded feasibility check only; not a QAT result and not comparable with PTQ without the matched FP32-FT controls

## Phase timings (seconds)

| phase | seconds |
|---|---|
| load_data | 1.3553 |
| train_V1 | 29.5072 |
| train_V3 | 159.3865 |
| ptq_V2 | 14.8715 |
| ptq_V4 | 13.3001 |
| clean_eval | 2.4099 |
| attacks | 6730.8956 |
| e4_plumbing | 1597.1509 |
| efficiency | 226.0599 |
| qat_feasibility | 34.6459 |

Peak sampled RSS of the pilot process: 1968.0 MB; minimum available RAM seen: 20.4 MB.

> **Pilot results are non-scientific development checks (1 seed, 2 epochs, small subsets, validation data only) and carry no accuracy or robustness conclusion.**
