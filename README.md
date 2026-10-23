# Transformer intrusion detection: robustness and efficiency

Research question: **How do INT8 quantization and adversarial training change the
robustness–efficiency trade-off of a Transformer for IoT intrusion detection?**

**Released-artifact multiclass baseline completed across four seeds.
Matched-condition reproduction of the published Table 4 results remains
unresolved.**

The PAVE Standard Transformer predicts one of **27 classes**: benign traffic or
one of 26 attack labels in the released, balanced CICIoT subset. It receives a
synthetic length-50 sequence made by repeating a row of 39 standardized features;
these are not 50 chronological network flows.

## Start here

1. [Milestone 1 report](reports/milestone1-artifact/REPORT.md) — authoritative
   methods, results, validity limits and official requirements checklist.
2. [Setup and commands](docs/SETUP.md) — environment, dataset acquisition/hash,
   baseline reproduction and report regeneration.
3. [Frozen baseline configuration](configs/milestone1_artifact_full.json),
   [per-seed metrics](reports/milestone1-artifact/per_seed.csv),
   [aggregate metrics](reports/milestone1-artifact/summary.csv) and
   [compact experimental evidence](reports/milestone1-artifact/evidence/README.md).
4. [Historical paper-described AdamW report](reports/milestone1-paper-described/REPORT.md),
   [cross-experiment comparison](reports/milestone1-paper-described/COMPARISON.md),
   [protocol](docs/MILESTONE1_PAPER_DESCRIBED_PROTOCOL.md) and
   [configuration](configs/milestone1_paper_described.json).
5. [Repository inclusion and verification guide](docs/REPOSITORY_REVIEW.md) —
   what belongs in Git, what stays local, and lightweight review checks.
6. [Corrected paper-described Adam protocol](docs/MILESTONE1_PAPER_DESCRIBED_ADAM_PROTOCOL.md),
   [configuration](configs/milestone1_paper_described_adam.json), and
   [repair handoff](docs/REPAIR_HANDOFF.md).

## What Milestone 1 executed

The **released-artifact multiclass baseline and documented paper-reproduction
attempt** used the entire 29,997-row processed CSV, ordinary FP32 Standard
Transformer only, and seeds **1, 7, 13, 29**. Each seed completed 15 epochs,
705 Adam updates, and evaluation of the same final checkpoint on 3,000 aligned
validation examples under clean, FGSM and PGD conditions. Clean means unperturbed;
the original class labels are retained under both attacks.

Mean ± **sample standard deviation across four seeds** (percentage points for
accuracy and F1):

| Condition | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |
| --- | --- | --- | --- |
| Clean | 65.81 ± 0.43 | 61.14 ± 0.42 | 61.15 ± 0.41 |
| FGSM, ε = 0.10 | 44.42 ± 2.12 | 42.85 ± 2.15 | 42.86 ± 2.16 |
| PGD, ε = 0.10 | 41.64 ± 2.97 | 40.37 ± 2.75 | 40.38 ± 2.77 |

![Four-seed measured performance](reports/milestone1-artifact/results.png)

The report also contains MCC, clean-correct untargeted ASR, runtime and per-class
evidence. PGD uses 10 steps, α = 0.01, clean initialization and epsilon-box
projection; neither attack enforces protocol constraints. Training took 4,397.54 s
and evaluation 369.76 s; the overall run span was 1 h 20 min 48 s on a local CPU.

These are capped **validation** results under inherited artifact choices:
whole-file StandardScaler before splitting, stratified 80/20 partitions varying
with seed, and possible duplicate leakage. They do not establish held-out test
generalization, natural class prevalence or physically realizable network evasion.

The classification mode and generating configuration of the paper's Table 4 row
have not been established. Its metric definitions and experiment provenance are
also unresolved. Our baseline and the published row are potentially different
experiments; numerical reproduction under matched conditions is **not demonstrated**.
See the report and [bounded reference investigation](docs/MILESTONE1_REFERENCE.md).

## Historical second Milestone 1 experiment: paper-described AdamW variant

A teammate added a historical four-seed (1, 7, 13, 29) FP32 Standard Transformer
experiment that follows the settings described in the PAVE paper text. It is the
**AdamW variant** of the paper-described configuration: the paper says "Adam with
weight decay 0.0001"; AdamW was used as a separate historical variant. It is **not**
independently verified and is **not** an exact reproduction of Table 4; review
issues remain outstanding. Table 4's classification mode, dataset subset, F1
averaging and ASR denominator are unresolved, so its numbers are context only.

Settings: MinMaxScaler fitted on training data only; stratified 70/15/15
train/validation/test split (split seed 2025); 25 epochs; AdamW, lr = 0.001,
weight decay = 0.0001, batch size 512 (1,050 optimizer steps); FGSM ε = 0.10 and
PGD ε = 0.10 (7 steps, α = 0.025, random start) with clipping to [0, 1].

Reported results (mean ± sample SD over four seeds, **full 4,500-row held-out
test partition**; ASR among clean-correct examples):

| Condition | Accuracy (%) | Macro-F1 (%) | ASR (%) |
| --- | --- | --- | --- |
| Clean | 63.96 ± 1.46 | 58.76 ± 1.73 | 0.00 ± 0.00 |
| FGSM, ε = 0.10 | 12.08 ± 3.04 | 10.80 ± 2.68 | 84.01 ± 4.66 |
| PGD, ε = 0.10 | 6.44 ± 1.76 | 6.24 ± 1.52 | 91.27 ± 2.87 |

Links: [report](reports/milestone1-paper-described/REPORT.md),
[comparison](reports/milestone1-paper-described/COMPARISON.md) (also
[reports/milestone1-comparison.md](reports/milestone1-comparison.md)),
[protocol](docs/MILESTONE1_PAPER_DESCRIBED_PROTOCOL.md),
[configuration](configs/milestone1_paper_described.json),
[per-seed metrics](reports/milestone1-paper-described/per_seed.csv),
[summary](reports/milestone1-paper-described/summary.csv) and
[figure](reports/milestone1-paper-described/results.png).

### How the two experiments differ

The released baseline's results are on a **capped 3,000-row validation**
evaluation; the second experiment's are on the **full 4,500-row test partition**.
They are not the same evaluation population. Several settings also differ at once:
scaler (whole-file StandardScaler vs training-only MinMaxScaler), split (80/20 vs
70/15/15), epochs (15 vs 25), optimizer (Adam, no decay vs AdamW, decay 0.0001),
steps (705 vs 1,050) and PGD (10 steps, α = 0.01, clean start, no domain clipping
vs 7 steps, α = 0.025, random start, clipping). **Epsilon uses different scaling:**
0.10 is 0.10 standard deviations in StandardScaler space but up to 10% of each
feature's training range in MinMaxScaler space, so the attack budgets are not
equivalent. This is a configuration comparison, not an ablation; differences
cannot be attributed to any single setting.

Reproduction command:

```bash
PYTHONPATH=. .venv/bin/python -B -m research.paper_described_baseline --config configs/milestone1_paper_described.json --run-dir runs/milestone1-paper-described --resume
```

## Corrected paper-described Adam experiment

Variant C uses conventional `torch.optim.Adam` with the paper-described
`weight_decay=0.0001`, a new run directory, and the new protocol above. It does
not rewrite or reuse variant B. The authorized four-seed run completed on the
shared 4,500-row test partition:

| Condition | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | ASR (%) |
| --- | ---: | ---: | ---: | ---: |
| Clean | 63.64 ± 0.94 | 58.31 ± 1.02 | 58.33 ± 1.02 | 0.00 ± 0.00 |
| FGSM, ε = 0.10 | 15.42 ± 4.28 | 12.92 ± 3.98 | 12.92 ± 3.99 | 77.27 ± 5.51 |
| PGD, ε = 0.10 | 7.92 ± 1.66 | 6.52 ± 1.09 | 6.51 ± 1.08 | 89.03 ± 3.74 |

Links: [report](reports/milestone1-paper-described-adam/REPORT.md),
[comparison](reports/milestone1-paper-described-adam/COMPARISON.md),
[protocol](docs/MILESTONE1_PAPER_DESCRIBED_ADAM_PROTOCOL.md),
[configuration](configs/milestone1_paper_described_adam.json),
[per-seed metrics](reports/milestone1-paper-described-adam/per_seed.csv),
[summary](reports/milestone1-paper-described-adam/summary.csv),
[compact evidence](reports/milestone1-paper-described-adam/evidence/),
and [independent checker](scripts/check_milestone1_adam_review.py). The run
used 42 batches per epoch, 1,050 optimizer steps and 524,925 sample exposures
per seed and completed at
`runs/milestone1-paper-described-adam-rerun/`. The earlier
`runs/milestone1-paper-described-adam/` directory is the preserved failed
first attempt, not a replacement run. No exact Table 4 reproduction is claimed.

The saved overlap audit finds 2,804 of 4,500 test rows (62.31%) with identical
feature patterns in training. Row-disjoint IDs therefore do not establish
duplicate-free or independent-device generalization. Historical smoke exposure
is disclosed in the repair records; the test set was not completely untouched
throughout development. Feature-space adversarial examples are not guaranteed
to be protocol-valid network traffic. Experiments A and C are the two main
implemented configurations; B remains a historical AdamW variant with missing
executed-source and checkpoint evidence. This is a completed practical
baseline and paper-described comparison study, not an exact matched-condition
Table 4 reproduction or external approval of every official Milestone 1
requirement.

## Code ownership and attribution

- **Upstream PAVE:** the pinned, unmodified checkout at
  `upstream/PAVE_artifact/` supplies the Standard Transformer, data loader,
  training helper, cap/inference functions and FGSM/PGD implementations. It is
  restored during setup and excluded from Git. Revision and source hashes are
  recorded in the frozen configuration.
- **Our group:** [artifact runner](research/artifact_baseline.py),
  [report generator](research/milestone1_report.py), checkpoint/resume and
  verification instrumentation, tests, configurations, reports and setup tools.
  The separate corrected research pipeline is also in `research/`.
- [PAVE attribution and license](NOTICE.md) identify the upstream authors and
  retain the upstream MIT license. No license for the group's original code is
  inferred from PAVE's license.

`research.reproduce` is a legacy mixed-model artifact wrapper, and
`research.milestone1` is the historical reference gate/component smoke.
`experiments/milestone1/run_*.py` evaluate corrected-pipeline checkpoints.
**The completed baseline uses `research.artifact_baseline`**, not these older paths.

## Later milestones

- **Milestone 2:** run the corrected protocol's full FP32/INT8 and adversarial
  training comparisons, attack curves and efficiency measurements.
- **Milestone 3:** complete the planned layer ablations, transfer/class-wise
  analyses and final robustness–efficiency synthesis.

Much of that functionality was implemented and tested on bounded smoke/pilot
runs; the full later-stage research experiments remain pending. Those tests are
not Milestone 1 baseline results. Read the [corrected protocol](docs/PROTOCOL.md)
and [historical implementation results](docs/VERIFIED_RESULTS.md) for context.
[PROGRESS.md](PROGRESS.md) preserves the execution history. No later milestone is
launched automatically.
