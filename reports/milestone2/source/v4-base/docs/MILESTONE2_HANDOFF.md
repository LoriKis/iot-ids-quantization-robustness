# Milestone 2A handoff

> **Pilot results are non-scientific development checks (1 seed, 2 epochs, small subsets, validation data only) and carry no accuracy or robustness conclusion.**

This is a new group-disjoint evaluation partition within a previously examined dataset, not a completely untouched external benchmark. The dataset was seen by Milestone 1; the earlier test partition was evaluated; identical feature vectors with conflicting labels remain in the data.

Feature-space attacks; no guarantee that perturbed vectors correspond to valid network traffic. Perturbations are tied to the original feature row (one 39-d delta broadcast to all 50 synthetic positions).

Nothing is committed or pushed. The test partition was never loaded for inference. The full four-seed campaign was not started. Authoritative design: `docs/MILESTONE2_PROTOCOL.md` (revision 3 plus protocol revision 1). Audit: `docs/MILESTONE2_IMPLEMENTATION_AUDIT.md`. Ordered plan: `docs/.m2a/plan.md`.

## 1. What exists

All paths are relative to the repository root. The interpreter is `.venv\Scripts\python.exe` (Python 3.11.9, torch 2.5.1+cpu, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, psutil 6.1.1). No dependency was added or upgraded.

| Module | Role |
|---|---|
| `research/m2_split.py` | Group-disjoint split, train-only `MinMaxScaler`, `load_arrays` guard, range audit |
| `research/m2_training.py` | One loop for V1 (ordinary) and V3 (PGD adversarial training); logs compute counters |
| `research/m2_attacks.py` | Row-tied wrapper, feasible-set policy, `checked_attack`, `CheckedPair`, surrogate, `evaluate_unit`, E4 helpers |
| `research/m2_quant.py` | Calibration adapter, INT8 PTQ unit, conversion verification, reload check, QAT feasibility |
| `research/m2_efficiency.py` | Efficiency harness: fresh child process per measurement, interleaved schedule |
| `research/m2_pilot.py` | Bounded pilot driver (phase `m2a`) |
| `research/m2_campaign.py` | Phase guards and frozen-config validation; `m2b_validation` and `m2b_test` validate and then stop |
| `research/m2_estimate.py` | Pilot-to-campaign planning estimate |
| `research/m2_report.py` | `pilot_report.md` and `attack_results.csv` |
| `research/attacks.py` | The only M1 module edited: one optional keyword `selector=None`; call structure unchanged |
| `scripts/check_m2_integrity.py` | Protected-path snapshot and compare |
| `tests/test_m2_*.py`, `tests/m2_helpers.py` | 305 tests in the whole suite (71 pre-existing), all passing |

Reused unchanged: `common.make_model`, `paper_described_adam.{build_optimizer,derived_seed,to_sequence,seed_everything}`, `attacks.attack`, `quantization.{ptq,prepare_static,inventory,load_int8,bpda_model}`, `runner.save_quantized`, `provenance.{state_hash,MemoryMonitor}`, `data.stratified_cap`.

Frozen configs in `configs/m2/` (SHA-256 of the files as on disk; `campaign.json` stores the hashes of its siblings, so editing any sibling requires a new `campaign.json` revision):

| File | SHA-256 |
|---|---|
| `data.json` | `15bb457b2f4313cc64fa3fe3fd4e289a7194abf03ba51369e71b7a9915104ad0` |
| `training.json` | `7be76132a32999f980e0a0e2474a4852439f0622f2f222a6e15d74b81167aaed` |
| `quantization.json` | `6bb20f37e56a1a0e627f3741b8a616e37983a1c93d92b42b414d240129c655fe` |
| `attacks.json` | `7ff7b518cc76b393a71a09574d1ec4a3f024a998edb4d81842b319a37d5fc8a4` |
| `efficiency.json` | `c3357b941f154c813054ac0cddc76ae343b98503f919b4ca218493fd8d69f4db` |
| `pilot.json` | `36fe9028e6b535ff82d2941e57714734fccd92529dc7a73e87111a0464ad9ce3` |
| `campaign.json` | `18d853b5135c856fdb8197f11b14bdc47addc0f0cbfb53c80524289dd0e650cd` (the value `--confirm-freeze` must equal for `m2b_test`) |

## 2. Frozen split

Split seed 20260 (the first listed seed passed). Dataset SHA-256 `0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed`, 29,997 rows. Manifest SHA-256 `7275ede47c840d8820d6d5892ff6886af8ab23de9373f868a3e2d1c62c969cd2`.

| Partition | Rows | Fraction | Target | Distinct feature vectors | ID SHA-256 |
|---|---|---|---|---|---|
| train | 21,925 | 73.09 % | 70 % | 5,330 | `0d273b3aca29b4e2705bc9227d6b868299f6228048f884e0cdc8996d364f2717` |
| validation | 4,036 | 13.45 % | 15 % | 3,327 | `23eeb96d093358c66cf2b1ff22d866da2c7fe2a4b941add4fe322cdd9866a827` |
| test | 4,036 | 13.45 % | 15 % | 3,296 | `083d16578dab7c771897d1f540e983cf3e446f7576b91dc1676cfb1ec6a50f70` |

Why not 70/15/15: 15 groups of identical feature vectors hold 51.4 % of all rows, and several span three classes, so each group has to stay in one partition. All 27 classes appear in every partition (minimum rows per class: train 640, validation 38, test 38). The acceptance band (train 0.68 to 0.78, validation and test 0.10 to 0.17) was declared before any training and was not widened.

Protocol revision 1 (made before any model was trained): the first run failed for all three seeds because feature `Number` equals 10 in 29,994 of 29,997 rows, and the three exceptions all landed outside train. The user chose to pin feature variability in the split objective (seven at-risk features). The failed attempt is kept as `reports/milestone2a/split/split_attempts_infeasible_rev0.json`.

Conflicting labels among identical vectors are reported, not changed: 229 groups, 8,697 rows (train 150 groups / 8,102 rows, accuracy ceiling 0.8091; validation 40 / 337, ceiling 0.9730; test 39 / 258, ceiling 0.9752).

Input range (clean inputs are never clipped): validation has 7 rows with 10 coordinates outside [0, 1] (largest excursion 3.81 above, 5.00 below, the `Number = 4` row 29510); test has 9 rows with 13 coordinates (largest 11.91 above); train has none. The pilot attack subset contains one out-of-range row, reported in `attack_checks.json`, never dropped. No feasible interval is empty for any epsilon in the grid.

Something to know before reading any per-class result: 15 attack classes have a single distinct feature vector in train (one group repeated 656 to 1,035 times) and 23 to 163 distinct vectors in validation and test. Models trained on this split see one point for those classes. This follows from the dataset's duplicate structure and is a property of the task, not of the split.

## 3. Commands

```powershell
Set-Location "<WORKSPACE>\IOT-network-traffic-DL"
$py = ".\.venv\Scripts\python.exe"
& $py scripts\check_m2_integrity.py --write artifacts\m2a-integrity\before.json            # done first; refuses to overwrite
& $py -B -m research.m2_split --config configs\m2\data.json                                # done; refuses a second run
& $py -B -m research.m2_split --config configs\m2\data.json --audit-only                   # read-only audit
& $py -B -m pytest -p no:cacheprovider -q                                                  # 305 passed
& $py -B -m research.m2_pilot --config configs\m2\pilot.json                               # done; exit 0; refuses an existing artifacts\m2a-pilot
& $py -B -m research.m2_estimate --pilot artifacts\m2a-pilot --out artifacts\m2a-pilot\campaign_estimate.json
& $py scripts\check_m2_integrity.py --compare artifacts\m2a-integrity\before.json --write artifacts\m2a-integrity\after.json
& $py -B -m research.m2_report --run artifacts\m2a-pilot --out <new file>                  # re-render the pilot report
```

`research.m2_campaign --phase m2b_validation` and `--phase m2b_test --confirm-freeze <SHA-256 of campaign.json>` validate the frozen configs and then raise `CampaignNotImplementedError`. The four-seed run is specified in `configs/m2/campaign.json` and is not implemented.

Pilot output: `artifacts\m2a-pilot` (git-ignored), with `pilot_report.md`, `pilot_runtime.json`, `campaign_estimate.json`, `attack_results.json/.csv`, `attack_budget_check.json`, `gradient_diagnostics.json`, `efficiency\efficiency.json`, `qat\qat_feasibility.json`, per-unit `attacks\**\attack_checks.json`, `models\V1..V4`. An earlier pilot attempt (`artifacts\m2a-pilot-attempt1-report-bug`) completed every measurement but crashed while rendering the report (a presentation bug in `m2_report`, since fixed and covered by a test). Its `status.json` says failed. The final pilot is a complete second run on the fixed code, with source digest `2c757a08da27d1c9d8a896e0d80d4ddcd3c21bd22aca13d7d8843d2eaa73f8e7`.

## 4. Pilot summary (non-scientific)

One seed (1), 2 epochs, batch 512, 12 optimizer updates per model. Subsets from the frozen train and validation ID lists: 2,700 train rows (100 per class), 540 validation rows (20 per class), 270 attack rows (10 per class), 540 calibration rows (20 per class, from the pilot train subset). Subset hashes are in `subsets.json`. Total run time 8,812 s, 0 invalid units among the 120 main attack units and the 24 E4 plumbing units.

- V1 and V3 start from identical weights (init hash `e07454f5ec4d...`) and perform identical optimizer updates (12). V3 uses PGD training with epsilon 0.05, 5 steps, step 0.025, random start, 1 restart, loss `0.5 CE(clean) + 0.5 CE(adv)`, attack in `eval()` mode, row-tied.
- AT extra compute: measured wall-clock ratio 6.77x (first step of each run discarded, 11 steps, indicative only); nominal pass-count ratio 10.67x. Counters per AT step 6 gradient passes, 8 scoring forwards, 2 training passes; a forward-hook count on the real model confirmed 14 attack-side and 2 train-side forwards.
- Clean accuracy on the 540 validation rows after 2 epochs is 0.143 (V1), 0.146 (V2), 0.106 (V3), 0.106 (V4): the models are nearly untrained, so none of the attack tables says anything about robustness. INT8 and FP32 argmax agreement is 0.991 (V2 vs V1) and 0.996 (V4 vs V3).
- All six attack sources (`fp32_direct`, `transfer_fp32_to_int8`, `transfer_fp32_to_int8_targetsel`, `bpda_int8`, `surrogate_fakequant`, `surrogate_fakequant_targetsel`) ran FGSM and pgd20 over the full grid for both pairs. Every unit passed the E5 checks: finite inputs, logits and gradients, L-infinity budget, feasible set, model mode, unchanged weights, no parameter gradients, unchanged labels, epsilon 0 identity, selector call counts 65 (pgd20) and 4 (FGSM) per batch, and BPDA forward error exactly 0.0.
- `direct_int8_grad` raises `RuntimeError` (no `grad_fn` through quantized operators) on 270 of 270 rows for V2 and V4. This is counted as an undefined gradient, never as robustness.
- Surrogate fidelity: the fake-quant surrogate's scales differ from the deployed INT8 scales (largest difference 0.0041 for V2 and 0.0040 for V4), maximum logit difference 0.053 and 0.035, argmax agreement 0.993 and 0.996. Every surrogate row carries that note.

Full tables: `artifacts\m2a-pilot\pilot_report.md`.

## 5. PTQ conversion evidence (V2 and V4)

Workflow (frozen): PyTorch eager-mode static PTQ, fbgemm backend, per-channel `qint8` weights, `quint8` activations (`HistogramObserver`), explicit `QuantStub -> Linear -> DeQuantStub` islands around all 23 `nn.Linear`. FX graph mode and PT2E fail on the unchanged model (audit section 2) and are not used. Calibration: 540 training rows from the shared ID list, identical for V2, V4, the surrogate and the QAT check.

For both models: the island wrapper reproduces FP32 logits with max error 0.0 before observers; 23 `torch.ao.nn.quantized.Linear` modules with `torch.qint8` / `per_channel_affine` packed weights; no `FakeQuantize`, observer, stub or float `nn.Linear` remains; the profiler shows 23 `quantized::linear` calls on one forward; the saved state reloads through `quantization.load_int8` with bitwise-equal logits and identical argmax over 540 rows. INT8 weight elements: 347,632 of 351,612 parameters (98.87 %); 46 quantize/dequantize boundaries; 21 classification-path islands and 2 anomaly-scorer islands. The `state_dict` shrinks from 1.449 MB to 0.497 MB.

Operations that remain FP32 (the model is hybrid, not integer-only): attention QK^T matmul, attention softmax, attention AV matmul, LayerNorm, residual adds, ReLU, sinusoidal positional-encoding add, mean pooling over the 50 positions, anomaly-scorer sigmoid, dropout (inactive in eval), the activations between islands, and Linear biases (FP32 inside the packed parameters).

## 6. QAT feasibility (optional V5)

Status: supported, with no unsupported operation or backend limit observed. Check: `prepare_static(qat=True)`, 1 epoch of Adam (lr 1e-4) on the 2,700-row pilot train subset with training noise, `disable_observer`, `convert`, then real INT8 inference on 540 validation rows. Result: 23 `qint8` per-channel packed weights, finite logits, serialized and reloaded bitwise, 34.6 s in total. This is a feasibility check on a 2-epoch FP32 parent and not a QAT result. QAT takes extra fine-tuning epochs, so any full experiment needs the matched controls specified in `quantization.json` and `campaign.json`: FP32-FT (same k = 3 epochs, lr 1e-4, no fake quantization) and FP32-FT -> PTQ. A QAT benefit would be attributed only relative to FP32-FT -> PTQ. V5 does not block V1 to V4.

## 7. E4 budget-adequacy plumbing

`pgd20` stays `provisional`. The pilot ran the E4 code path (pgd20 versus pgd50x5, 12 paired cells for V1 to V4 at epsilon 0.025, 0.05, 0.10) on a 54-row stratified subset (2 per class) of the 270-row attack subset, not on all 270 rows. Reason: pgd50x5 costs about 3.9x pgd20 on this CPU, and the 54-row subset kept the pilot at about 27 minutes for this phase. All 12 `n_extra` values were 0, which says nothing about adequacy because the models are nearly untrained. No adequacy decision was recorded (`adequate` and `decision` are null). The decision belongs to campaign step 0 on fully trained seed-1 checkpoints. For this split the real E4 subset would have 1,727 rows (`data.stratified_cap`, cap 70, validation has more than 2,500 rows), above the 1,000-row minimum, with allowed `n_extra <= ceil(0.01 * 1727) = 18` per cell.

## 8. Efficiency observations

Harness: fresh child process per measurement, 1 thread, `torch.inference_mode`, 256 fixed validation rows, timed region row tensor -> repeat 50 -> forward -> argmax, batch 1 (warm-up 10, timed 50) and batch 256 (warm-up 2, timed 10), 2 repetitions, executed order V1, V2, V3, V4, V1, V2, V3, V4 recorded in `run_order` with timestamps, CPU load and available RAM before each start. No retries or failures.

Observed on this laptop, with background load (pre-run CPU load 37 to 93 %, available RAM 0.6 to 0.8 GB; the pilot process peaked at 1.97 GB RSS and the machine's available RAM fell to 20 MB during the run):

| Variant | state_dict MB | batch-1 median ms | batch-256 median ms | peak working set minus baseline MB |
|---|---|---|---|---|
| V1 | 1.449 | 4.60 | 437 | 115 |
| V2 | 0.497 | 11.12 | 423 | 113 |
| V3 | 1.449 | 6.46 | 448 | 115 |
| V4 | 0.497 | 13.46 | 483 | 113 |

Read these as noisy. The discarded first attempt, run with a different background load, gave V1 batch-1 2.19 ms and V2 5.89 ms, so absolute values shift by 2x between runs. What both runs agree on: the INT8 state is 0.34x the FP32 size, INT8 is slower than its FP32 parent at batch 1 (about 2.1x for V4 against V3 and 2.4x for V2 against V1 here), and batch-256 times are close. No speed or memory advantage is claimed. A usable efficiency measurement needs a quiet machine and a fresh process (the pilot parent holds the training and attack models in memory).

## 9. Campaign estimate (planning figure, not a measurement)

Source: `artifacts\m2a-pilot\campaign_estimate.json`, scaled from pilot timings (each run's first step discarded). Assumptions: 4 seeds, 25 epochs, 1,075 updates per run, 2 training threads, 1 attack thread, validation and test each evaluated once with 4,036 rows, the six sources per pair and four epsilons above 0.

| Component | Seconds | Hours |
|---|---|---|
| V1 training, 4 seeds | 9,173 | 2.5 |
| V3 (AT) training, 4 seeds | 62,071 | 17.2 |
| PTQ, 8 units | 113 | 0.03 |
| Step 0 E4 check (12 cells on 1,727 rows) | 50,973 | 14.2 |
| Efficiency harness | 480 | 0.1 |
| Attacks with pgd20, validation | 400,869 | 111.4 |
| Attacks with pgd20, test | 400,869 | 111.4 |

Point estimate with pgd20: about 257 hours. If step 0 forces pgd50x5 everywhere (measured ratio 3.9x, range 3.2x to 5.2x over the 12 cells): about 868 hours. Planning range: 128 to 1,627 hours (point times 0.5, up to the pgd50x5 scenario times 1.5 times 1.25 for thermal throttling and RAM paging). Disk: about 0.13 GB estimated against a 2 GB budget; 49 GB free. The pilot models are nearly untrained, and attack cost per example for trained models may differ. The fake-quant surrogate sources cost about 2.6x (pgd20: 1.05 s/example against 0.40 for `bpda_int8` and 0.27 for `fp32_direct`).

Attacks dominate, not training. This size is not a run-it-overnight campaign on this machine, which is why section 12 lists it as a decision.

## 10. Known unverified items

- Run-to-run bit-exactness is not claimed (training does not enable deterministic algorithms, as in Experiment C). `train_unit(adversarial=False)` is bit-identical to `run_epochs` at equal thread count (tested at 1 thread).
- Dropout masks cannot be paired between V1 and V3 (V3 performs two train-mode forwards per update).
- The PT2E and FX failure cause was not diagnosed; Linux and WSL were not tested; the AVX2-only CPU is the plausible reason INT8 is slower here (unverified).
- `pgd20` adequacy, the AT effect after 25 epochs, the QAT benefit and any robustness difference between V1 to V4 are unknown. The pilot says nothing about them.
- BPDA uses an approximate backward pass (dequantized weights); the forward is the real INT8 model, with exact forward agreement checked on every scored candidate.
- Efficiency numbers were taken on a loaded machine (section 8).
- The surrogate's observers use moving-average QAT settings, so its scales differ from the deployed PTQ scales; this is reported and not hidden.
- Out-of-range held-out features (the `Number` singleton rows) dominate the largest excursions; per-class results on those rows are not robustness evidence.

## 11. Integrity

`py scripts\check_m2_integrity.py --compare artifacts\m2a-integrity\before.json --write artifacts\m2a-integrity\after.json` returned OK: 1,144 protected Milestone 1 files unchanged (reports, configs, runs, artifacts, docs, upstream), git HEAD unchanged (`466faf724aaab5507d5cf12f74281c5f2a62ae28`). `git status` shows only new or intended files: the `selector` edit in `research/attacks.py`, a `slow` marker in `pytest.ini`, and the new M2 files. Editing `research/`, `configs/` and `tests/` changed the provenance source digest, so Milestone 1 run roots can no longer be resumed (expected).

## 12. Open decisions for the user

1. **Acceptance band.** Keep 73.1 / 13.5 / 13.5 (inside the band declared before training), or change the band. A different band means a new split and a new protocol revision before any further training.
2. **Campaign size.** About 257 hours at 1 attack thread (868 if pgd50x5 is needed). Options that keep the design: evaluate attacks on a stratified validation subset (the E4 subset has 1,727 rows) instead of all 4,036 rows, run fewer sources on the test partition (for example `fp32_direct`, `bpda_int8`, pure transfer), use more attack threads, or accept a longer run. Each changes what the report can claim and needs your decision before the campaign.
3. **QAT inclusion.** Feasible and cheap to run (about 35 s per pilot unit). Include the V5 arms with their matched FP32-FT controls (12 extra units), or leave QAT out of the campaign.
4. **Per-position sensitivity check.** The inherited per-position perturbation mode (50 x 39 free variables) is specified only as a labelled seed-1 validation sensitivity check. Include it, or leave it out.
5. **Training diversity.** 15 classes have a single training vector (section 2). Keep the task as it is and state this beside every per-class result, or change the task definition. Neither is done silently.
6. **Efficiency run.** Run the efficiency harness on a quiet machine in its own process, as a separate step, instead of inside the pilot.
