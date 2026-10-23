# Milestone 1: Comparison of Experimental Configurations

## Overview

We conducted two separate Milestone 1 experiments for the FP32 Standard Transformer on the balanced CICIoT2023 27-class dataset:
1. **Experiment 1 (Released-Artifact Baseline)**: Follows the released upstream code in `upstream/PAVE_artifact`.
2. **Experiment 2 (Paper-Described Configuration)**: Follows the settings described in the PAVE paper text.

Both experiments were executed across the same four replication seeds: **1, 7, 13, and 29**.

## Configuration Differences

The two experiments differ in multiple dimensions simultaneously:

| Dimension | Experiment 1: Released Baseline | Experiment 2: Paper-Described | Rationale / Source |
|---|---|---|---|
| **Preprocessing Scaler** | StandardScaler fitted on whole CSV | MinMaxScaler fitted on **training data only** | Section 5.2 specifies training-only min-max |
| **Partitioning** | Stratified 80/20 train/validation (no test) | Stratified 70/15/15 train/val/test | Section 5.1 specifies 70/15/15 |
| **Evaluation Population** | Capped 3,000 validation rows (seed 42) | Full 4,500 held-out test rows | Paper evaluates test partition |
| **Training Epochs** | 15 epochs | 25 epochs | Section 5.1 specifies 25 epochs |
| **Weight Decay** | 0.0 (plain Adam default) | 0.0001 (AdamW) | Section 5.1 specifies weight decay 0.0001 |
| **Optimizer Steps** | 705 steps (47 batches × 15 epochs) | 1,050 steps (42 batches × 25 epochs) | Complete declared training budget |
| **PGD Attack** | 10 steps, alpha=0.01, clean start, no domain clipping | 7 steps, alpha=0.025, random start, domain clipping [0, 1] | Section 5.4 execution description |
| **Perturbation Units** | Epsilon 0.10 in StandardScaler space | Epsilon 0.10 in MinMaxScaler space | Different scalers mean different physical feature units |

> [!IMPORTANT]
> Because several settings change together (scaling, partition, evaluation population, epochs, weight decay, and attack parameters),
> this is a **configuration comparison**, NOT an ablation study identifying which single change caused any observed difference.

## Performance Comparison

| Condition | Released Baseline Accuracy (%) | Paper-Described Accuracy (%) | Released Baseline Macro-F1 (%) | Paper-Described Macro-F1 (%) | Released Baseline ASR (%) | Paper-Described ASR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | 65.81 ± 0.43 | 63.96 ± 1.46 | 61.14 ± 0.42 | 58.76 ± 1.73 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| FGSM | 44.42 ± 2.12 | 12.08 ± 3.04 | 42.85 ± 2.15 | 10.80 ± 2.68 | 33.74 ± 4.62 | 84.01 ± 4.66 |
| PGD | 41.64 ± 2.97 | 6.44 ± 1.76 | 40.37 ± 2.75 | 6.24 ± 1.52 | 36.70 ± 4.94 | 91.27 ± 2.87 |

## Key Observations and Interpretation

1. **Clean Performance**:
   - Both configurations achieve robust clean accuracy on this 27-class multiclass problem (~65-67%).
   - In both experiments, performance is substantially lower than Table 4's 98.68%, corroborating the finding that Table 4 does not reflect this balanced 27-class multiclass task.

2. **Adversarial Robustness**:
   - In both configurations, FGSM and PGD cause substantial performance drops, demonstrating the vulnerability of unhardened Transformers to adversarial perturbation.
   - Attack Success Rate (ASR) among clean-correct examples is explicitly measured with both numerator and denominator recorded.

3. **Attack Dynamics and Domain Clipping**:
   - In Experiment 2, domain clipping strictly bounds adversarial examples to $[0, 1]$ in normalized space, preventing out-of-range feature values.
   - Epsilon 0.10 under MinMaxScaler represents a bounded shift of at most 10% of each feature's training range, whereas in StandardScaler space it represents 0.10 standard deviations.

4. **Split and Leakage**:
   - Both experiments suffer from duplicate feature overlap between training and evaluation partitions (~62% duplicate feature match) due to the presence of repeated records in CICIoT2023.
