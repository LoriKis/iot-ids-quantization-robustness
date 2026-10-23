# PAVE Transformer robustness–efficiency study

## Milestone 1 Experiments

This project includes two completed and verified Milestone 1 experiments for the Standard Transformer on the balanced CICIoT2023 multiclass dataset across seeds 1, 7, 13, and 29:

### Experiment 1: Released-Code Baseline (Completed 2026-10-20)
- **Settings**: StandardScaler fitted on whole CSV, 80/20 train/val partition, 15 epochs, Adam lr=0.001 with zero weight decay, evaluation on capped 3,000 validation rows, clean-start PGD (10 steps, alpha=0.01).
- **Navigation**:
  - [Course report](reports/milestone1-artifact/REPORT.md)
  - [Per-seed results](reports/milestone1-artifact/per_seed.csv)
  - [Summary mean ± sample SD](reports/milestone1-artifact/summary.csv)
  - [Results figure](reports/milestone1-artifact/results.png)
  - Frozen configuration: `configs/milestone1_artifact_full.json`
- **Measured Results**: Clean **65.81 ± 0.43%**, FGSM $\epsilon=0.10$ **44.42 ± 2.12%**, PGD $\epsilon=0.10$ **41.64 ± 2.97%**.
- **Execution Command**:
  ```bash
  PYTHONPATH=. .venv/bin/python -B -m research.artifact_baseline --run-dir runs/milestone1-artifact-full --resume
  ```

### Experiment 2: Paper-Described Configuration (Completed 2026-10-23)
- **Settings**: MinMaxScaler fitted on **training data only** (Sec 5.2), stratified 70/15/15 train/val/test split (Sec 5.1, split seed 2025), 25 epochs (Sec 5.1), AdamW lr=0.001, weight decay=0.0001 (Sec 5.1), batch size 512, evaluation on full 4,500 held-out test rows, FGSM $\epsilon=0.10$ with normalized domain clipping, PGD $\epsilon=0.10$ (7 steps, alpha=0.025, random start, domain clipping [0, 1], Sec 5.4).
- **Navigation**:
  - [Course report](reports/milestone1-paper-described/REPORT.md)
  - [Cross-experiment comparison](reports/milestone1-paper-described/COMPARISON.md) (also available at [reports/milestone1-comparison.md](reports/milestone1-comparison.md))
  - [Protocol document](docs/MILESTONE1_PAPER_DESCRIBED_PROTOCOL.md)
  - [Per-seed results](reports/milestone1-paper-described/per_seed.csv)
  - [Summary mean ± sample SD](reports/milestone1-paper-described/summary.csv)
  - [Results figure](reports/milestone1-paper-described/results.png)
  - Frozen configuration: `configs/milestone1_paper_described.json`
- **Measured Results**: Clean **63.96 ± 1.46%**, FGSM $\epsilon=0.10$ **12.08 ± 3.04%** (ASR: 84.01%), PGD $\epsilon=0.10$ **6.44 ± 1.76%** (ASR: 91.27%).
- **Execution Command**:
  ```bash
  PYTHONPATH=. .venv/bin/python -B -m research.paper_described_baseline --config configs/milestone1_paper_described.json --run-dir runs/milestone1-paper-described --resume
  ```

> [!NOTE]
> As documented in [COMPARISON.md](reports/milestone1-paper-described/COMPARISON.md), these two experiments represent a **configuration comparison** changing multiple settings simultaneously (scaling space, split, evaluation population, epochs, weight decay, and attack dynamics). Epsilon 0.10 in MinMaxScaler space represents 10% of each feature's entire range, whereas in StandardScaler space it represents 0.10 standard deviations. Published Table 4 numbers are provided as separate context; exact Table 4 reproduction remains unestablished. Milestone 2 has not started.

**Active scope: Milestone 1 only.** The full supplied paper has now been read.
The Table 4 values are verified, but the exact multiclass reference remains
unresolved. See [reference investigation](docs/MILESTONE1_REFERENCE.md) and
[unsent clarification draft](docs/MILESTONE1_CLARIFICATION.md).
`research.milestone1 --route paper --dry-run` reports the target; paper training
is gated. The dedicated artifact component smoke is separately labelled and is
not scientific reproduction. The full-study commands below are retained as
historical preparation and are outside the active scope.

Local implementation for the MSc question: **What robustness–efficiency trade-off
results from combining Transformer intrusion detection, INT8 quantization and
adversarial training?**

The corrected pipeline uses PAVE's unchanged Standard Transformer and its
processed balanced CICIoT2023 multiclass data. Full scientific results remain
pending; smoke/pilot output establishes implementation and runtime only.

Read [audit](docs/AUDIT.md), [protocol](docs/PROTOCOL.md),
[setup and commands](docs/SETUP.md), [verified outcomes](docs/VERIFIED_RESULTS.md)
and [progress/resume](PROGRESS.md).

## Windows PowerShell (the tested platform)

From this repository root, after setup:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m research.runner --config configs/smoke.json --run-dir runs/smoke-new --phase both
.\.venv\Scripts\python.exe -m research.analyze runs/smoke-new --phase both --output artifacts/smoke-new
```

Use a new run/output directory for a new experiment. A compatible existing run
supports `--resume` for completed units only. A failed/partial unit is preserved
and requires a new run directory. Do not claim mid-training resume.

## Full study (explicitly launch after reviewing pilot estimates)

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase validation
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase test --resume
.\.venv\Scripts\python.exe -m research.analyze runs/full-study --phase test --output artifacts/full-study-test
```

Four training seeds: 1,7,13,29; split seed 2026. Validation selects checkpoints
and freezes the attack budget before test. Main variants receive full curves;
ablations and four transfer directions use the focused epsilon .1.

Default quantization is **static PTQ of INT8 Linear islands**, including explicit
attention Q/K/V/output projections. Attention matmuls, softmax and normalization
remain floating point. INT8 attacks use actual target forward kernels and
approximate BPDA gradients; they are not exact INT8 white-box attacks.

PAVE's sequences are repeated tabular rows, not recorded temporal traffic.
Perturbations study normalized feature-space vulnerability, not realizable packet
attacks. The dataset is balanced and does not establish natural class prevalence.

## Separate artifact reproduction

```powershell
.\.venv\Scripts\python.exe -m research.reproduce --out runs/reproduction-new --dry-run
```

Remove `--dry-run` to execute the original `exp_proj` in a separate local copy.
This retains upstream limitations; no exact paper-table match is claimed.
See the new reference investigation for the supplied full paper and the still
unverified experiment correspondence; the earlier access failure is historical.

## Evaluation-only Milestone 1 entry points

The repaired wrappers require an existing run and never train:

```powershell
.\.venv\Scripts\python.exe experiments/milestone1/run_clean.py --run-dir runs/smoke-new --partition test
.\.venv\Scripts\python.exe experiments/milestone1/run_fgsm.py --run-dir runs/smoke-new --partition test
.\.venv\Scripts\python.exe experiments/milestone1/run_pgd.py --run-dir runs/smoke-new --partition test
```

All load the same ordinary baseline checkpoint per seed. They write unique
manual evaluation directories; they do not invoke the upstream defense sweeps.

## Local-only and attribution

No code pushes, publishes, uploads data, opens PRs, or triggers remote CI.
Datasets, environments, checkpoints, upstream clones, caches and runs are ignored.
PAVE is attributed to its original authors; its upstream MIT license is preserved
in the pinned, unmodified reference checkout. `provenance.json` pins both starting
revisions and the processed data hash. Historical milestone notes describe the
starting implementation and are superseded by the new audit/protocol.
