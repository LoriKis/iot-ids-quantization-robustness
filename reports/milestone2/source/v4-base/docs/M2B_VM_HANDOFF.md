# Milestone 2B VM handoff (preparation notes)

This note records what was changed to make the Milestone 2 campaign executable on a Linux VM, what deviates from the frozen Milestone 2A files, and what was and was not run. The commands are in `VM_RUNBOOK.md` (top of the transfer folder; source copy `docs/VM_RUNBOOK.md`). `docs/MILESTONE2*` and `configs/m2/*` are unchanged.

## What was run locally, and what was not

Run locally (no model training, no attacks, no calibration or benchmark on the real dataset, no full test suite): file inspection, hashing, split construction and audit from dataset structure, syntax checks, `bash -n` through the available bash, and tests on tiny synthetic fixtures with the local guard `M2_LOCAL_NO_TRAIN=1` (`tests/conftest.py` blocks optimizer steps, models larger than 2 layers or `d_model` 64, and `evaluate_unit`).

Not run: any training (including seed 1), the pilot, FGSM/PGD, PTQ/QAT on real data, efficiency measurements, `preflight` and `smoke` on real data, any Linux or torch-on-Linux execution, `pip install` of the Linux pins. All of that is **pending VM verification**.

Existing tests that were not executed locally (they train or attack; they run in VM stage 4): `tests/test_m2_training.py` (including the bit-identity test of `train_unit` against `run_epochs`), `tests/test_m2_quant.py`, `tests/test_m2_attacks.py`, `tests/test_m2_pilot.py`, the real-model parts of `tests/test_m2_efficiency.py`, `tests/test_pipeline.py`, the Milestone 1 tests (`test_artifact_baseline.py`, `test_paper_described_baseline.py`, `test_adam_config_and_optimizer.py`, `test_milestone1_gate.py`, `test_resume_fingerprints.py`, `test_adversarial_domain.py`, `test_smoke_partitions.py`) and `tests/test_m2b_vm_only.py`. They were only collected locally (import check, 0 collection errors).

## Changes

Split: `split_v2` (algorithm version 2 in `research/m2_split.py`, config `configs/m2b/data_v2.json`) allocates per class by distinct feature groups as well as rows. Classes with exactly one distinct training vector: 15 in the original M2A split, 0 in `split_v2`. Rows and labels are preserved, identical vectors (including conflicting-label groups) stay together, group overlap is 0 for every partition pair. Rows train/validation/test: 21,925/4,036/4,036 before, 22,412/3,821/3,764 after; distinct groups 5,330/3,327/3,296 before, 7,023/2,467/2,463 after. The original M2A split files are unchanged. Details: `reports/milestone2b/split_v2/split_report.md` and `split_comparison_v1_v2.md`.

Campaign runner: `research/m2b_*.py` (plan, state and fingerprints, I/O durability, executors, scheduler and CLI, preflight and smoke, report). `m2b_validation` and `m2b_test` in `research/m2_campaign.py` delegate to the runner; `CampaignNotImplementedError` is kept for import compatibility and is never raised.

Modified existing files: `research/m2_training.py` (epoch-boundary checkpoints with optimizer and RNG state, resume), `research/m2_quant.py` (FP32 fine-tuning control and QAT units), `research/m2_efficiency.py` (Linux memory metric `VmHWM`, QAT arms), `research/m2_split.py` (algorithm v2, `purpose="final_test"` guard), `research/common.py` (git-free package mode), `research/attacks.py`, `tests/test_m2_guard.py`, `pytest.ini` (markers).

New configs: `configs/m2b/campaign_v2.json`, `data_v2.json`, `efficiency_v2.json`. They store the SHA-256 of the frozen `configs/m2/*` files they build on.

VM scripts: `scripts/vm/*.sh` (one per stage, none chains another), `verify_package.py`, `check_shell.py`, `check_runbook_commands.py`, `scan_test_dependencies.py`. Packager: `scripts/package_m2_vm.py`.

## Deviations from the frozen M2A files

1. Step 0 decision: `configs/m2/campaign.json` says the PGD budget decision creates a new campaign revision. Frozen configs stay immutable here; the decision is written to `output/<run>/step0/budget_decision.json` and bound into the evaluation freeze.
2. Split: `split_v2` replaces the M2A split for all M2B work, which is a new partition and so remains a new evaluation inside a previously examined dataset.
3. Efficiency memory: Linux reports `VmHWM`; the pilot reported Windows peak working set. The two are not comparable and every efficiency table says so.
4. Attack cost: the handoff estimate (about 257 h at one attack thread, 868 h with `pgd50x5`) is unchanged; no subset or reduced source set was chosen.

## Git-free package mode

The package has no `.git`. `research/common.py::git` serves a fixed answer table from `PACKAGE_IDENTITY.json` when `.git` is absent and the file exists, after verifying the PAVE source tree against the recorded tree id. Any other git call raises. Call sites reviewed: `common.git` (patched), `milestone1.git` (a second independent function; left unedited because its source hash is recorded in Milestone 1 evidence), `artifact_baseline` and `paper_described_baseline` (import `milestone1.git`), `provenance.py` (receives a callable), `scripts/check_m2_integrity.py::_git` (not used on the VM). Milestone 1 tests call `milestone1.git` indirectly; `tests/conftest.py` redirects those module-level names to the shim only when running in the package. This is pending VM verification.

## Known limits

See section 8 of the runbook. In short: gating and scheduling were proven with stub executors; real-data loading, resume equality, PTQ/QAT conversion on Linux, attack units and efficiency children are untested until the VM run.
