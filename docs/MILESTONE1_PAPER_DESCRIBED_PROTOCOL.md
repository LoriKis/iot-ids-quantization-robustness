# Protocol: Milestone 1 Paper-Described Configuration

**Frozen before any full training run. 2026-10-21.**
This document must not be altered after the first full seed completes.

## Purpose and scope

This is the second Milestone 1 experiment: a multiclass Standard Transformer
evaluation using paper-described settings as closely as the available evidence
allows. It is not a verified Table 4 reproduction — the classification mode,
dataset provenance, seed list and other details of Table 4 remain unresolved.

The first experiment (released-code baseline) is preserved unchanged in
`configs/milestone1_artifact_full.json` and `reports/milestone1-artifact/`.
This experiment uses `configs/milestone1_paper_described.json`,
`runs/milestone1-paper-described/` and `reports/milestone1-paper-described/`.

## Paper settings verification

| Setting | Value | Classification | Source in paper | Conflict |
|---|---|---|---|---|
| Scaler | MinMaxScaler, training data only | A | Section 5.2, printed p.10 / PDF p.14 | None |
| Partition | Stratified 70% train / 15% validation / 15% test | A | Section 5.1, printed p.9 / PDF p.13 | None |
| Optimizer | Adam, lr=0.001 | A | Section 5.1, printed p.9 / PDF p.13 | None |
| Weight decay | 0.0001 | A | Section 5.1, printed p.9 / PDF p.13 | None |
| Epochs | 25 | A | Section 5.1, printed p.9 / PDF p.13 | None |
| Batch size | 512 | A | Section 5.1, printed p.9 / PDF p.13 | None |
| FGSM epsilon | 0.10 | A | Section 5.4, printed p.11 / PDF p.15 | None |
| PGD epsilon | 0.10 | A | Section 5.4, printed p.11 / PDF p.15 | None |
| PGD steps | 7 | A | Section 5.4 | None |
| PGD alpha | 0.025 (epsilon/4) | A | Section 5.4 | Conflict with Section 3.3 |
| PGD init | Random start | A | Section 5.4 | None |
| Domain clipping | Per-feature [min_train, max_train] | A | Section 5.4 | None |
| Architecture | Standard Transformer 96/4/3/384 | A | Section 5.3 / Table 3 | None |
| Classification mode | Multiclass 27-class | C | Assignment requirement; Table 4 mode unverified | Binary is stronger textual reading |
| Sequence construction | Repeat-50 + noise 0.01 | C | Section 5.3 describes length 10 for RNN; Transformer unspecified | Sequence length unresolved |
| Seeds | {1, 7, 13, 29} | C | Phase 1 seed list not established | Phase 2 lists {1,7,13,21,42} |
| Split seed | 2025 | C | Not in paper | Predetermined before execution |
| Checkpoint rule | Final epoch 25 | C | Not stated | Predetermined before execution |

A = Explicitly stated in paper. B = Inherited from released code. C = Assumption.

## PGD conflict: Section 5.4 vs Section 3.3

Section 5.4 (execution description): "7 steps, alpha=epsilon/4=0.025, random start."
Section 3.3 (general formula): "alpha=epsilon/K" — with K=7 this gives 0.10/7≈0.0143.

**Resolution before execution:** We follow Section 5.4 as the execution-specific
description. Section 3.3 presents the general algorithmic formula; Section 5.4
overrides it with the specific values used in the paper's experiments. This choice
is recorded here and in the configuration before any training run starts. It is not
chosen to move results toward any target value.

## Domain clipping interaction

After min-max scaling on training data, test/validation features outside the
training range produce values outside [0, 1]. Section 5.4 describes clipping to
normalized feature bounds. We apply per-feature [min_train, max_train] bounds
(the training range in scaled space: [0,1] for in-range features, but the actual
min and max observed in training) after each PGD step and after FGSM. This
interacts with the epsilon box: the effective perturbation is the intersection
of the epsilon sphere and the per-feature domain.

Perturbations operate on the **scaled sequence tensor** (same as the artifact
baseline). Epsilon 0.10 is in MinMaxScaler units. Because different scalers are
used, epsilon 0.10 in MinMaxScaler space is not the same perturbation in original
feature units as epsilon 0.10 in StandardScaler space.

Unconstrained attacks do not produce protocol-valid network traffic. Class labels
are retained from the original dataset.

## Preprocessing

1. Load full CSV; numeric coerce; missing/infinite → 0.
2. Stratified split (split_seed=2025): 70% train (20,997 rows), 15% validation
   (4,500 rows), 15% test (4,500 rows). Implemented as two sklearn `train_test_split`
   calls to maintain stratification across all three partitions.
3. **Fit MinMaxScaler on training rows only.** Transform training, validation and
   test using the fitted scaler. No information from validation or test enters the
   scaler.
4. Fit LabelEncoder on training labels only (alphabetical; all 27 classes present
   in training due to stratification).
5. Construct 50-token synthetic sequences: each scaled row repeated 50 times.
   Training sequences add independent Gaussian noise std=0.01. Evaluation uses
   noise-free repetition.

Missing values outside the training range that produce out-of-[0,1] scaled values
are not clipped at preprocessing time; they appear as-is in sequences. PGD domain
clipping uses the per-feature [min_train, max_train] bounds recorded at fit time.

## Training

Each seed trains from scratch:
- AdamW, lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8
- 25 epochs, batch 512, cross-entropy, no scheduler or gradient clipping
- Shuffle training data each epoch (torch.randperm)
- Save epoch checkpoint after every epoch; final epoch 25 is the evaluation checkpoint
- 20,997 training rows → 42 batches/epoch (41 full batches of 512 + 1 remainder batch of 5 rows) → 1,050 optimizer steps total across 25 epochs

## Evaluation

The full held-out test partition (4,500 rows) is used for all three conditions.
The same 4,500 test row IDs are shared across all four training seeds (shared split).
Row IDs differ from the released-artifact baseline's capped validation evaluation.

1. **Clean**: evaluate test rows directly.
2. **FGSM**: one sign-gradient step on test sequences; epsilon=0.10; domain clip.
3. **PGD**: 7 steps, alpha=0.025, random start seeded per batch with training_seed;
   project to [x-eps, x+eps] and domain clip after each step; evaluate final iterate.

All conditions use the same checkpoint and the same row IDs. model.eval() throughout.
No checkpoint selection using evaluation results.

## Split overlap audit

After constructing the split, the runner measures and records:
- Number of feature-identical rows that appear in both train and test partitions
  (exact float32 match after numeric cleaning)
- Number of label-identical rows duplicated across partitions
- Whether these represent leakage or simply duplicate entries in the source CSV

This audit is diagnostic and does not alter the split. The result is reported in
the split metadata. Row IDs are disjoint by construction; the audit
concerns feature-level duplication, not row-level identity.
Measured counts for the frozen split (split_seed=2025):
- Full CSV contains 18,044 repeated feature rows (11,953 unique feature patterns).
- Test rows matching a train row in features: 2,804 / 4,500 (62.31%).
- Test rows matching a train row in (features, label): 2,757 / 4,500 (61.27%).
- Validation rows matching a train row in features: 2,793 / 4,500 (62.07%).
- This reflects source-data duplication rather than row-index overlap. Distinct row IDs do not imply absence of duplicate feature leakage.

## Smoke run

Before full execution, a smoke run with reduced budget:
- 270 training rows (10 per class), 135 test rows (5 per class), 1 epoch, seed 1
- Checks: shapes, label alignment, training-only scaler fit, split coverage,
  optimizer step count, checkpoint reload consistency, model.eval() during evaluation,
  finite nonzero input gradients, epsilon-zero identity, PGD perturbation bounds,
  domain clip bounds, no model weight changes during evaluation, same examples
  across conditions

Smoke results are in `runs/milestone1-paper-described-smoke/` and are labelled
as non-scientific component checks; they are not substituted for full results.

## Comparison with released-artifact baseline

Multiple settings change together between the two experiments:

| Item | Released-artifact (Exp 1) | Paper-described (Exp 2) |
|---|---|---|
| Scaler | StandardScaler, whole CSV | MinMaxScaler, training only |
| Partition | 80/20 train/val, no test | 70/15/15 with held-out test |
| Evaluation pop. | Capped 3,000 from validation | Full 4,500 test rows |
| Training | Adam, no decay, 15 epochs | AdamW, decay 0.0001, 25 epochs |
| PGD config | 10 steps, alpha=0.01, clean start | 7 steps, alpha=0.025, random start |
| Domain clipping | None | Per-feature [min_train, max_train] |

Because these settings change simultaneously, any performance difference is a
**configuration-pair comparison**, not an ablation identifying which individual
change is responsible.

## Resume and completion

Training runs sequentially, one seed at a time. Each epoch checkpoint records
model weights, optimizer state and torch RNG. Resuming a seed restores from the
last completed epoch; completed seeds are reused without retraining. Completed
seeds are never overwritten. If execution is interrupted, resume with:

```bash
PYTHONPATH=. .venv/bin/python -m research.paper_described_baseline \
  --run-dir runs/milestone1-paper-described --resume
```

A new run directory is required for any configuration change or failed seed.

## Authorship and provenance

Configuration frozen: `configs/milestone1_paper_described.json`
Protocol frozen: `docs/MILESTONE1_PAPER_DESCRIBED_PROTOCOL.md`
Runner: `research/paper_described_baseline.py`
Report generator: `research/paper_described_report.py`
Run directory: `runs/milestone1-paper-described/`
Report output: `reports/milestone1-paper-described/`
