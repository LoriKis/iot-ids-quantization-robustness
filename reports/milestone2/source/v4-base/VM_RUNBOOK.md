# Milestone 2 VM runbook

This folder runs the Milestone 2 campaign on a Linux VM: ordinary and adversarially trained FP32 Standard Transformers, INT8 PTQ derived from each, and an optional QAT stage, for seeds 1, 7, 13 and 29, on the revised group-disjoint split (`split_v2`). The split is a new group-disjoint partition inside a previously examined dataset, not an untouched external benchmark.

**Status: no VM run has been performed.** Everything below that depends on Linux, PyTorch, the INT8 backend, timings or the full test suite is **pending VM verification**. Passing local checks (manifest, syntax, runbook command resolution, synthetic tests) say nothing about the VM run.

All commands run from the top of this folder. Every stage has exactly one entry point. No script starts the next stage.

## 1. Weights and model source

- Model architecture and source: `upstream/PAVE_artifact/code/models/standard_transformer.py` (pinned PAVE commit, verified by stage 1).
- Trained weights are **not** in this folder and are not missing. FP32 checkpoints (V1 ordinary, V3 adversarial) are produced by stages 5 and 6. INT8 states are produced by PTQ inside the same stages. QAT and FP32 fine-tuning states are produced by stage 7.
- The pilot's untrained-quality models are deliberately not shipped.

Key paths:

| What | Relative path |
|---|---|
| Dataset | `upstream/PAVE_artifact/data/balanced_CICIoT.csv` |
| PAVE model source | `upstream/PAVE_artifact/code/models/standard_transformer.py` |
| Frozen split | `reports/milestone2b/split_v2/split_m2.json` (with `scaler_minmax.json`, `split_freeze.json`) |
| Campaign configuration | `configs/m2b/campaign_v2.json` (siblings `data_v2.json`, `efficiency_v2.json`; frozen `configs/m2/*`) |
| Package identity and manifest | `PACKAGE_IDENTITY.json`, `MANIFEST.sha256` |
| Handoff notes and deviations | `docs/M2B_VM_HANDOFF.md` |

## 2. Supported platform

| Item | Requirement |
|---|---|
| OS | Linux x86-64: Ubuntu 22.04 or 24.04, or Debian 12 (glibc 2.28 or newer) |
| CPU | AVX2 required (the INT8 backend is fbgemm). AVX-512 is optional |
| Python | 3.11.x as `python3.11` with `venv` (Ubuntu: `sudo apt install python3.11 python3.11-venv`) |
| PyTorch | `torch==2.5.1+cpu` (pinned in `requirements-linux-cpu.txt`) |
| Device | CPU only. No GPU is used |
| Disk / RAM | at least 5 GB free disk, at least 8 GB RAM and about 3 GB per parallel worker |
| Internet | Stage 2 only (PyPI and `https://download.pytorch.org/whl/cpu`). All other stages are offline. For an offline stage 2 set `M2_WHEELHOUSE=/path/to/wheels` |

Dependency pins were copied from the Windows lock file and are not yet resolved on Linux: **pending VM verification**.

## 3. The nine stages

Environment variables used by the scripts: `RUN_NAME` (default `m2b-v1`), `WORKERS` (default: the value recommended by stage 3 in `output/<RUN_NAME>/preflight.json`; 1 if stage 3 has not written it). Use the same `RUN_NAME` for every stage of one campaign.

### Stage 1. Verify transfer integrity

```
bash scripts/vm/01_verify_transfer.sh
```

Checks `MANIFEST.sha256` with `sha256sum`, the file set, symlinks, LF line endings, the dataset SHA-256 (`0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed`, 5,444,531 bytes), the 19 recorded PAVE source hashes, the PAVE tree id, and `bash -n` on the scripts. Creates no environment and imports no torch. Pass: the script prints `Stage 01 passed.` On failure send back the printed list of missing, mismatched or unlisted files.

### Stage 2. Create the Linux environment and install pinned dependencies

```
bash scripts/vm/02_setup_env.sh
```

Needs internet (see section 2). Creates `.venv`, installs `requirements-linux-cpu.txt`, runs `pip check` and prints the torch version and `supported_engines`. It never reads the dataset and never trains. Offline variant:

```
M2_WHEELHOUSE=/path/to/wheels bash scripts/vm/02_setup_env.sh
```

### Stage 3. Non-training preflight

```
bash scripts/vm/03_preflight.sh
```

Checks imports and versions, the fbgemm INT8 backend (a random 8x8 layer and a tiny QAT prepare/convert without optimizer steps), dataset hash and header, PAVE source tree, the frozen split, config hashes, fingerprints, disk, RAM, cores, writable output paths and `bash -n`. No model forward on real data and no training. Writes `output/<RUN_NAME>/preflight.json` including `recommended_workers`. Pass: no FAIL line. If fbgemm is missing, stop (see troubleshooting): the workflow does not fall back to another engine.

### Stage 4. Regression tests and bounded smoke test

```
bash scripts/vm/04_tests_and_smoke.sh
```

Runs the whole test suite (`pytest -rs tests`), then verifies that the tests did not change the package file set, then a bounded smoke campaign (`research.m2b_campaign smoke`, at most 45 minutes, train and validation subsets only, in `output/<RUN_NAME>-smoke/`). It never touches the real run state and never reads the test partition. The suite includes tests that train tiny models (`vm_only`) and real-data checks; the existing suite took tens of minutes on the laptop, the VM time is unknown. Pass: 0 failed, 0 errors; every skip is printed with its reason. Send back the `-rs` skip list. Expected skip reasons are only the documented package skips and `vm_only` markers that do not apply; anything else is a finding. If the smoke restart check reports WARN "bit-exactness not claimed", that is allowed.

### Stage 5. Fully trained seed-1 validation only

Two parts. Part 5a trains V1 and V3, runs PTQ, clean evaluation and the pair check, then prints status and a time estimate. Read the estimate before starting 5b.

```
bash scripts/vm/05_seed1_validation.sh 5a
```

Part 5b runs step 0 (the E4 grid that decides between `pgd20` and `pgd50x5`) and the seed-1 validation attacks.

```
bash scripts/vm/05_seed1_validation.sh 5b
```

Cost: V1 and V3 training is 25 epochs each (the laptop handoff scale was about 2.3 h and 4.3 h at 2 threads); step 0 is about 14 h single-thread; seed-1 validation attacks are about 28 h single-thread; parallel workers divide the attack time. The handoff projected about 257 hours for all attacks of the campaign at one attack thread (868 hours if `pgd50x5` is chosen). These are laptop-scale estimates, **pending VM measurement**. The `estimate` printed at the end of each part uses the VM's own measured timings. Stage 5 never runs the test partition, other seeds or QAT.

Step 0 decides the PGD budget once, from seed 1, and records it in `output/<RUN_NAME>/step0/budget_decision.json`. If the E4 subset has fewer than 1000 rows the unit is blocked and the runner tells you to stop and ask; do not work around it.

### Stage 6. Resume the remaining seeds and validation work

```
bash scripts/vm/06_resume_remaining.sh
```

Runs validation for seeds 1, 7, 13 and 29. Completed seed-1 units are verified and reused, never rerun or overwritten. Seeds 7, 13 and 29 use the step-0 decision made by seed 1.

### Stage 7. Optional QAT and matched controls

Only if you choose it. The flag is required.

```
bash scripts/vm/07_qat_optional.sh --i-want-qat
```

Runs, per seed, `FP32-FT` (3 epochs of FP32 fine-tuning of V1, the matched control), `FP32-FT-PTQ` (PTQ of that control) and `QAT-FT` (3 epochs of QAT from V1), with their clean and attack units. The QAT benefit is read only as `QAT-FT` minus `FP32-FT-PTQ`; the other two comparisons measure extra training, not quantization. If an arm cannot be built on Linux the unit is recorded as `blocked` with the stage, module and exception, and the other units continue. Exit code stays 0 in that case: check `blocked_units.md` in the validation report.

### Stage 8. Freeze, then launch final-test evaluation

Stage 8a only freezes. It runs no inference and prints the freeze SHA-256.

```
bash scripts/vm/08a_freeze_eval.sh
```

With QAT arms in the test phase (requires stage 7):

```
bash scripts/vm/08a_freeze_eval.sh --include-qat
```

Stage 8b is the only command that reads the test partition. It needs the SHA-256 that 8a printed (also in `output/<RUN_NAME>/freezes/eval_freeze.json.sha256`). Replace the placeholder with the 64 hex characters:

```
bash scripts/vm/08b_run_final_test.sh <SHA256-from-08a>
```

Review the validation report (stage 9b) before running 8b. The test phase cannot train, quantize, calibrate or change the attack budget. A freeze file is never overwritten; to freeze again, use a new `--freeze-name` through the CLI.

### Stage 9. Final tables, robustness curves and efficiency report

Efficiency measurement must run alone on a quiet machine (the runner refuses when other campaign processes run or the load is high). Use the same hardware, thread counts and batch sizes for all models. Pass `--include-qat` only if the freeze used `--include-qat`.

```
bash scripts/vm/09a_efficiency.sh
```

```
bash scripts/vm/09a_efficiency.sh --include-qat
```

Then the reports:

```
bash scripts/vm/09b_report.sh
```

`09b` always writes the validation report; the efficiency report only if 09a finished; the test report only if 08b finished. Outputs: `output/<RUN_NAME>/reports/validation/`, `reports/test/`, `reports/efficiency/`. They contain the long and aggregated tables (mean +/- sample standard deviation over seeds), the paired differences, accuracy and macro-F1 versus epsilon plots for FGSM and PGD (`acc_vs_eps_*.png`, `f1_vs_eps_*.png`, with CSV of the plotted points), per-class tables with weak-evidence flags, and serialized size, latency and memory. Recomputed metrics come from the saved per-example `predictions.npz` (ids, labels, predictions). Validation and final-test results are in separate folders and each file carries a banner naming its phase. Pilot numbers are never read.

## 4. Interruption and restart

Training writes an atomic checkpoint at every epoch boundary (model, optimizer, random-generator state, fingerprints; last two kept). Evaluation units are written once and never overwritten. To interrupt, press Ctrl-C or send SIGTERM to the process; workers finish the current epoch checkpoint and exit 130. To resume, run the same stage script again. Completed units are skipped, an interrupted training unit continues from its last epoch checkpoint.

```
bash scripts/vm/05_seed1_validation.sh 5a
```

Long runs survive a closed terminal when started this way:

```
nohup bash scripts/vm/05_seed1_validation.sh 5a > output/m2b-v1/logs/nohup.out 2>&1 &
```

```
tmux new -s m2
```

```
tail -f output/m2b-v1/logs/nohup.out
```

```
kill -TERM <pid>
```

(`output/m2b-v1/logs/` exists after stage 1 has run.) Resuming restarts from the last epoch; bit-identical resume is checked by the smoke test and the `vm_only` tests (pending VM verification), and run-to-run bit-exactness is not claimed.

### Fingerprints

Every unit records fingerprints of the source files it depends on, the configs, the dataset and the split. On restart the runner compares them and refuses to continue on a mismatch (exit 2, nothing is written). If you changed something on purpose, the runner can archive only the affected units (moved to `output/<RUN_NAME>/archive/<TAG>-<UTC>/`, never deleted) and rerun them:

```
.venv/bin/python -m research.m2b_campaign run --phase validation --seeds 1 --archive-and-restart-affected fix1
```

Rules for edits on the VM: do not edit anything under `configs/`, `upstream/` or the split files. Edit `research/` only to fix a defect, record it in `output/<RUN_NAME>/EDITS.md`, and rerun stage 3. `MANIFEST.sha256` will no longer verify for an edited file. Orchestration files never invalidate finished units (they only produce a WARN). A change to a file in the training set invalidates all units. The exact lists are generated from the code and printed below:

<!-- fingerprint-sets:begin -->
| file (research/) | unit kinds invalidated if it is edited |
|---|---|
| `adversarial_domain.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `attacks.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `common.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `data.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `evaluation.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, ptq, qat, step0 |
| `m2_attacks.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `m2_efficiency.py` | efficiency |
| `m2_quant.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, ptq, qat, step0 |
| `m2_split.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `m2_training.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `m2b_io.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `milestone1.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `paper_described_adam.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `provenance.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `quantization.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, paircheck, ptq, qat, step0, train |
| `runner.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, ptq, qat, step0 |
| `training.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, ptq, qat, step0 |
| `validate.py` | attack, calib, clean, diag, e4, efficiency, fp32_ft, ft_ptq, ptq, qat, step0 |
<!-- fingerprint-sets:end -->

## 5. Where things are written, and which units completed

| Path under `output/<RUN_NAME>/` | Content |
|---|---|
| `logs/` | one log per stage run (`<stage>-<UTC>.log`) |
| `stage_markers/`, `output/.stage_markers/` | advisory order markers (`NN.ok`); the authority is `state/units/*.json` |
| `state/units/` | one record per unit (status, timings, output hashes, fingerprints) |
| `seeds/s<seed>/models/`, `calibration/`, `paircheck/` | checkpoints, INT8 states, calibration ids and tensors, pair check |
| `validation/`, `test/` | clean results, attack units, diagnostics, per-example `predictions.npz` |
| `step0/` | E4 cells and `budget_decision.json` |
| `efficiency/` | per-seed efficiency measurements and `hardware.json` |
| `freezes/` | `eval_freeze.json` and its SHA-256 |
| `reports/{validation,test,efficiency}/` | tables, plots, CSV |
| `archive/` | units moved aside by `--archive-and-restart-affected` |

Which units have completed:

```
bash scripts/vm/status.sh
```

```
bash scripts/vm/status.sh --phase validation --seed 1 --verify
```

`--verify` re-hashes the outputs of completed units. Statuses: `completed`, `failed`, `invalid` (attack checks failed; recorded, not retried), `blocked`, `interrupted`. Exit codes of the campaign runner: 0 ok; 2 config, data or fingerprint error; 3 partition or freeze guard; 4 a unit failed (rerun resumes); 5 invalid attack units recorded; 130 interrupted.

## 6. Comparisons

| Name | Model |
|---|---|
| V1 | FP32 Standard Transformer, ordinary training (25 epochs) |
| V2 | INT8 PTQ derived from the same seed's V1 |
| V3 | FP32 Standard Transformer, adversarial training (25 epochs, same recipe otherwise) |
| V4 | INT8 PTQ derived from the same seed's V3 |
| FP32-FT, FP32-FT-PTQ, QAT-FT | optional (stage 7), derived from V1 |

V1 and V3 share the initial state and update counts (checked by `paircheck`). Calibration for PTQ and QAT uses training rows only (540 rows, 20 per class, same ids for V2, V4 and QAT). Preprocessing is fitted on the training partition only. The split is the same for every variant and seed. Only the factor under comparison changes.

## 7. How INT8 is evaluated

| Item | Procedure |
|---|---|
| Model | Eager-mode static PTQ with fbgemm: all 23 `nn.Linear` layers become quantized linear layers (qint8 per-channel weights, quint8 activations). Attention matmuls, softmax, LayerNorm, residuals, ReLU, positional add, pooling, the sigmoid, dropout and biases stay FP32. It is a hybrid model, not integer-only. |
| Clean INT8 predictions | The real converted INT8 module on CPU with the fbgemm engine set explicitly, evaluation mode, `inference_mode`. |
| FP32 attacks | `fp32_direct`: gradient and evaluation on the FP32 model. |
| INT8 attacks | The offline feature-space evaluation of `research/m2_attacks.py`. Sources: `bpda_int8` (forward through the real INT8 model, backward through the dequantized weights; the primary white-box INT8 curve), `transfer_fp32_to_int8` (pure transfer from the same seed's FP32 parent), `transfer_fp32_to_int8_targetsel`, `surrogate_fakequant`, `surrogate_fakequant_targetsel` (query-assisted variants are labelled). The final robust evaluation is always on the real INT8 target. `direct_int8_grad` is a diagnostic only. |
| Attacks | FGSM and PGD at the frozen epsilon grid (0, 0.01, 0.025, 0.05, 0.1). The PGD budget is `pgd20` or `pgd50x5` as decided by step 0. One 39-dimensional perturbation is shared by all 50 sequence positions. |
| Accuracy definitions | Clean accuracy uses the clean predictions; accuracy under attack uses the adversarial predictions; robust accuracy needs both correct. Macro-F1 is over all 27 classes on the unit's rows. |
| Checks | Each unit records finite values, L-infinity budget, feasible set, model mode, unchanged weights and an identity check at epsilon 0. A failed check makes the unit `invalid`. |

## 8. Known limitations

- Split: identical feature vectors stay together in one partition, so classes dominated by one vector (15 classes in the original split) still have few distinct vectors in validation and test; per-class numbers carry a weak-evidence flag. See `reports/milestone2b/split_v2/split_report.md`.
- Hybrid INT8 only. The backward pass of `bpda_int8` is an approximation (BPDA), so its robustness numbers are not guaranteed upper bounds on attack success.
- Attacks are feature-space; perturbed vectors need not be valid network traffic.
- Run-to-run bit-exactness is not claimed.
- Efficiency numbers need a quiet machine. Windows pilot memory (peak working set) and Linux memory (VmHWM) are not comparable. No speed or memory advantage of INT8 is assumed.
- The attack campaign is expensive (see stage 5). Choices that cut cost (subsets, fewer sources) change what the report can claim; ask before using them.

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| `sha256sum` or `verify_package.py` reports a mismatch | Transfer was corrupted or a text conversion changed bytes. Transfer again. Do not edit files to make it pass. |
| `python3.11 not found` | Install Python 3.11 (`sudo apt install python3.11 python3.11-venv`) or use pyenv or uv, then rerun stage 2. |
| `pip` cannot resolve a pin | Send back the message. Do not loosen pins without recording it in `EDITS.md`. |
| `fbgemm` missing from `supported_engines`, or AVX2 absent | Stop. The campaign needs fbgemm; there is no fallback. Use an x86-64 VM with AVX2. |
| glibc too old | Use Ubuntu 22.04 or newer. |
| Worker killed (OOM) | Rerun with fewer workers, for example `WORKERS=1 bash scripts/vm/06_resume_remaining.sh`. Units resume. |
| Disk full | Free space under `output/`; nothing is deleted automatically. |
| Exit 2 "fingerprint mismatch" | A source, config, data or split file differs from the recorded value. See section 4. |
| Exit 3 | A guard refused (QAT without opt-in, missing or wrong freeze SHA, test requested without a freeze). Read the message. |
| Unit `invalid` or `blocked` | Recorded with its reason in `invalid_units.md` and `blocked_units.md`. Not retried automatically; `--retry-failed` retries failed units only. |
