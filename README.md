# PAVE Transformer robustness–efficiency study

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
