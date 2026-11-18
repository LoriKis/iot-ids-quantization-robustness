# Milestone 2A implementation audit

**Date:** audit performed on branch `local/research-pipeline-fixes`, HEAD
`466faf724aaab5507d5cf12f74281c5f2a62ae28`, clean working tree before and after the audit.
**Scope:** read-only. No code, config, split, checkpoint, report or artifact was modified.
No model inference was run on any test partition. The only commands executed were
`pip freeze`/`pip check`, version and API probes, in-memory data audits, and the existing
test suite (`pytest -p no:cacheprovider`, **71 passed in 25.4 s** on this machine today).

Interpreter for everything below:
`<WORKSPACE>\IOT-network-traffic-DL\.venv\Scripts\python.exe`
(abbreviated `.venv python`). Run from the repo root as `.\.venv\Scripts\python.exe -B -m ...`.

---------------------------------------------------------------------------------------

## 1. Repository rules and Milestone 1 evidence to preserve

- There is no `AGENTS.md`, `CONTRIBUTING.md` or `.kiro/` steering in the repo. The binding
  instructions are `README.md`, `docs/SETUP.md`, `docs/PROTOCOL.md`, `docs/REPAIR_HANDOFF.md`,
  `docs/MILESTONE1_CORRECTION_NOTES.md`, `docs/MILESTONE1_PAPER_DESCRIBED_ADAM_PROTOCOL.md`,
  `docs/REPOSITORY_REVIEW.md`, `PROGRESS.md`.
- Rules that constrain M2A: never overwrite an existing run directory (new `runs/<name>` per
  execution; `research.common.unit_status` raises `FileExistsError` otherwise); any change to
  `research/`, `configs/`, `scripts/` or `tests/` changes `provenance()["source_digest"]`, so the
  historical run roots **cannot** be resumed after edits (this is intended; use new roots);
  `runs/`, `artifacts/`, `upstream/`, `work/`, `*.pt`, `*.log`, `.venv/` are git-ignored; the
  group has not chosen a code license (see `NOTICE.md`).
- **Preserve, never edit:** `reports/milestone1-artifact/**`, `reports/milestone1-paper-described/**`,
  `reports/milestone1-paper-described-adam/**` (including `provenance/`, `evidence/`),
  `reports/milestone1-repair/**`, `reports/milestone1-comparison.md`, `configs/milestone1_*.json`,
  `configs/{smoke,pilot,full}.json`, `runs/**` (including `milestone1-paper-described-adam/` failed
  first attempt and `milestone1-paper-described-adam-rerun/`), `artifacts/{smoke-verified,pilot-verified,pilot-estimate}/**`,
  `docs/MILESTONE1_*`, `docs/VERIFIED_RESULTS.md`, `docs/PROTOCOL.md` (historical corrected-study protocol).
  New M2 material must go in new files: `configs/milestone2_*.json`, `docs/MILESTONE2_*.md`,
  `runs/milestone2a-*`, `artifacts/milestone2a-*`, new `tests/test_m2_*.py`.
- `docs/MILESTONE1_REFERENCE.md` and the README state that Table 4 reproduction is unresolved.
  Nothing in M2A may tune against Table 4 (paper values are 98.68% clean accuracy, ours ~64%).
- Experiments: A = `research.artifact_baseline` (released code, StandardScaler, 80/20,
  15 epochs); B = historical AdamW (`research/paper_described_baseline.py`, source and checkpoints
  missing locally); C = `research/paper_described_adam.py` + `configs/milestone1_paper_described_adam.json`.

## 2. Environment, hardware and quantization APIs (measured today)

| Item | Finding |
|---|---|
| OS / shell | Windows 10.0.26200, PowerShell |
| Python | 3.11.9 (portable build extracted under `work/python-3.11.9`, venv `.venv`) |
| torch | **2.5.1+cpu**; `pip check`: no broken requirements |
| Other pins | numpy 1.26.4, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.5.2, matplotlib 3.9.4, psutil 6.1.1, pytest 8.3.4 (`requirements.txt`, `requirements-lock-windows.txt`) |
| `torchao` | **not installed** (do not add; not needed) |
| CPU | Intel Core i5-10210U, 4 physical / 8 logical cores, AVX2 (`torch.backends.cpu.get_cpu_capability() == "AVX2"`), no AVX-512/VNNI |
| GPU | none usable (Intel UHD Graphics only; `torch.cuda.is_available()` False, `torch.version.cuda` None) |
| RAM | 8,331,935,744 bytes total; only **~795 MB available** at probe time. Keep batch sizes modest, one training process at a time, avoid holding many models. |
| Disk | ~56.07 GB free of 510 GB on C: (project drive) |
| Threads | torch default 4 intra / 4 inter; the research runner pins 1/1 (`config["threads"]`); C used 2 threads (`resources.torch_threads`) |
| Quantized engines | `['none', 'onednn', 'x86', 'fbgemm']`, **default `x86`**; the project forces `fbgemm` explicitly (`research.quantization.prepare_static`, `torch.backends.quantized.engine = backend`) |
| MKL / MKLDNN | both available |

### Quantization API probes (torch 2.5.1, this machine)

| API | Result | Consequence |
|---|---|---|
| Eager-mode static PTQ with explicit `QuantStub -> Linear -> DeQuantStub` islands (`research.quantization.LinearIsland`) | Works; 71 tests pass including real converted modules, packed-weight inventory (`qint8`, per-channel affine), save/reload bit-exact | **Use this as the single supported CPU INT8 workflow** (fbgemm, per-channel qint8 weights, `HistogramObserver` quint8 activations via `aq.get_default_qconfig("fbgemm")`) |
| `torch.ao.quantization.quantize_dynamic` | Works on all 23 `nn.Linear`; runs under fbgemm, x86 and onednn | Already implemented as optional `mode="dynamic"`; must stay separately labeled |
| FX graph mode (`prepare_fx`) on the unchanged `StandardTransformerIDS` (also behind a logits-only wrapper) | **Fails**: `TypeError: slice indices must be integers or None or have an __index__ method` (symbolic tracing of `_SinusoidalPE.forward`, `self.pe[:, :x.size(1), :]`) | Cannot be used without editing upstream model code (forbidden: the model must stay PAVE's). Do not pursue |
| PT2E (`torch.export.export_for_training` -> `prepare_pt2e` with `X86InductorQuantizer`) | Export succeeds, **`prepare_pt2e` fails with `IndexError`** (not investigated further; 2.5.1 PT2E is prototype) | Not used. Do not upgrade torch to chase it |
| Eager QAT (`prepare_qat` on the islands, train-mode fake quant, `disable_observer`, `convert`) | Works end to end (`tests/test_pipeline.py::test_qat_real_conversion`, `runs/smoke-verified/*/qat/s1/qat_int8_feasibility/feasibility.json`: actual converted INT8 inference verified, 216 rows, 1 epoch) | QAT is feasible on the 23 Linear islands; the remaining question is a matched FP32 fine-tune control |

Official references used: PyTorch quantization docs (<https://pytorch.org/docs/main/quantization.html>,
API reference <https://pytorch.org/docs/main/quantization-support.html>), the eager static
quantization tutorial (<https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html>)
and PT2E x86 docs (<https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html>).
Eager mode with QuantStub/DeQuantStub is the documented mechanism for manually delimited
quantized regions, which is exactly what `LinearIsland` does.

Quantization scope actually realized (unchanged model, `selected_names`): `all_linear` = **23**
`nn.Linear`: embedding (1), per block W_q/W_k/W_v/W_o (12) and FFN two linears (6), classification head
(2), anomaly scorer (2). Operations that stay FP32: attention QK^T and AV matmuls, softmax, LayerNorm,
residual adds, dropout (inactive in eval), ReLU between island boundaries, sinusoidal PE add, mean pooling,
sigmoid of the anomaly scorer, and every inter-island tensor (each island quantizes its input and
dequantizes its output; this is a hybrid INT8-Linear model, not integer-only). Note the anomaly scorer
branch is computed but unused for classification; it inflates "converted module" counts.

Efficiency caveat from `docs/VERIFIED_RESULTS.md` (pilot): on this CPU the all-linear INT8 model was
**slower** than FP32 at batch 1 (20.5 ms vs 14.0 ms) and at batch 32 for the ordinary model, with a
2.9x smaller serialized state (497,448 vs 1,448,741 bytes). AVX2 without VNNI and per-island
quant/dequant overhead are the plausible causes (unverified). No speedup claim is justified in advance.

## 3. Experiment C: exact configuration (the M2 starting point)

Sources: `configs/milestone1_paper_described_adam.json` (schema 2, frozen 2026-10-23),
`research/paper_described_adam.py`, `docs/MILESTONE1_PAPER_DESCRIBED_ADAM_PROTOCOL.md`.

| Aspect | Experiment C value |
|---|---|
| Model | `StandardTransformerIDS(input_dim=39, d_model=96, n_heads=4, n_layers=3, d_ff=384, n_classes=27, max_seq_len=50, dropout=0.1)` loaded unchanged by `research.common.make_model` from the pinned upstream (`e73911bb...`), 351,612 params. Forward returns a dict (`logits`, `anomaly_score`, `embeddings`, `attention_weights`); only `logits` enters the loss |
| Precision / device | FP32, CPU |
| Data | `upstream/PAVE_artifact/data/balanced_CICIoT.csv`, SHA-256 `0d869e33...a53ed`, 29,997x39, 27 classes, sorted label names, 1,111 rows/class |
| Split | saved stratified row split, seed 2025, 20,997/4,500/4,500 (`reports/milestone1-paper-described/split_ids.json`, SHA `f4624a70...514b`); **not group-aware** (see sec 5) |
| Scaler | `sklearn.preprocessing.MinMaxScaler` fitted on training rows only (`provenance.fit_scaler`); held-out values outside [0,1] retained, never clamped; no constant training feature |
| Sequence | each scaled row repeated 50 times -> `(B, 50, 39)`; synthetic, not chronological |
| Training noise | Gaussian, std 0.01, added **after** repetition to the full `(B,50,39)` tensor, i.e. independent noise per position; fresh noise every batch and every epoch; evaluation noise-free |
| Optimizer | `torch.optim.Adam(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999), eps=1e-8, amsgrad=False, maximize=False)`, L2-in-gradient decay (not AdamW). `build_optimizer` asserts the class |
| Schedule / clipping | none / none |
| Loss | unweighted `F.cross_entropy` on logits |
| Batching | batch 512, `drop_last=False`: 41 batches of 512 + 1 of 5 = 42 steps/epoch, 25 epochs, **1,050 steps**, 524,925 sample exposures |
| Checkpoint rule | final epoch 25, no validation/test selection |
| Dropout | 0.1 active in train mode, drawn from the **global** torch RNG |
| RNG policy | `seed_everything(seed)` once (Python, NumPy, torch global) before model construction; per-epoch shuffle uses `torch.Generator` seeded `derived_seed("shuffle", seed, epoch)`; per-batch noise uses `derived_seed("noise", seed, epoch, batch)` (`sha256` of the pipe-joined parts, 63-bit). Shuffle and noise are therefore fully isolated from the global stream and from any attack generator; only initialization and dropout use the global stream. No `torch.use_deterministic_algorithms` call in C |
| Seeds | 1, 7, 13, 29 |
| Attacks (M1 test) | FGSM eps 0.10; PGD eps 0.10, 7 steps, alpha 0.025 (= eps/4), uniform random start, final iterate; per-position perturbation on `(B,50,39)`; feasible set = eps-box intersect [0,1]; infeasible rows raise `InfeasibleConstraintError` |
| M1 outcome (context only) | test clean accuracy 63.64 +/- 0.94 %, macro-F1 58.31 +/- 1.02 %, FGSM 15.42 %, PGD 7.92 % at eps 0.10; 62.31 % of M1 test rows share an exact feature vector with training |

Differences between C and the existing generic M2-capable pipeline (`research/training.py`,
`configs/full.json`): scaler (MinMax vs StandardScaler), split (stratified 2025 vs group 2026),
batch 32 vs 512, weight decay 0 vs 1e-4, 15 vs 25 epochs, gradient clipping 1.0 vs none, validation-selected
checkpoint vs final epoch, shared generator (shuffle + noise drawn from one `torch.Generator`) vs derived
isolated generators. **The generic pipeline does not implement Experiment C and must not be run as-is for M2.**

## 4. Functionality classification

Legend: **V** implemented and verified (covered by passing tests and/or executed evidence);
**R** implemented but needs repair, adaptation or validation for M2; **M** missing.
"Tests" refers to files in `tests/` (71 tests pass today).

### 4.1 Implemented and verified (reuse unchanged)

| Component | Location and signature | Evidence |
|---|---|---|
| Upstream model loader, pin/hash verification | `research.common.make_model(config)`, `verify_upstream()`; `PIN`, `DATA_SHA`, `UPSTREAM` | `test_actual_standard_attention_inventory`, run fingerprints |
| Canonical hashing, atomic writes, unit lifecycle | `common.digest/canonical/sha256/atomic_json/atomic_torch`, `common.unit_status(folder, identity, resume=False)`; `provenance.atomic_*`, `state_hash(model)`, `canonical_hash`, `MemoryMonitor` | `test_unit_failure_resume_and_collision`, `test_resume_fingerprints.py` |
| Feature-row group hashing | `research/data.py::build_data` (inline `sha256(row.astype("<f8").tobytes())`) | used for the 2026 split manifest; exact hashing independent of label |
| Eager static/dynamic PTQ of explicit Linear islands | `research.quantization.ptq(model, scope, calibration, mode="static", backend="fbgemm") -> (model, inventory)`; `prepare_static(model, scope, example, *, qat=False, backend)`; `selected_names(model, scope)`; `LinearIsland`; `replace`; `inventory(model, requested, mode)` | `test_actual_quantization_and_bpda` (static+dynamic x 4 scopes), identity wrapper check `atol=0` before observers |
| Packed-weight inventory (dtype/qscheme/shape, fails if not converted) | `quantization.inventory` raises `RuntimeError` if a requested module is not a quantized Linear or weight is not `qint8` | same tests; `conversion.json` per unit |
| INT8 serialization + reload equality | `quantization.load_int8(float_architecture, scope, calibration, checkpoint, compatibility, mode, backend)` (rebuilds converted modules then `load_state_dict(strict=True)`); `runner.save_quantized` asserts bitwise equal logits | `test_int8_checkpoint_roundtrip`, smoke/pilot runs |
| Exact-forward BPDA | `quantization.BPDALinear`, `_BPDA`, `bpda_model(target, requested, example)` asserts `atol=0` forward agreement with the real INT8 target; backward = grad @ dequantized weight | `test_actual_quantization_and_bpda`; `evaluate()` re-checks full-set forward error |
| Strongest-candidate FGSM/PGD | `research.attacks.attack(model, x, y, *, kind, epsilon, steps=10, alpha=None, restarts=1, random_start=True, lower=None, upper=None, mutable=None, seed=0)`; `attacks.logits(model, x)` | `test_attack_budget_zero_modes_weights_and_gradients` (eps 0 identity, mixed module modes restored, no `.grad` written, weights unchanged), `test_pgd_retains_fgsm_and_more_restarts`; uses `torch.autograd.grad` on the input only; local `torch.Generator` seeded by caller; finite and `Linf <= eps` asserted |
| Pure-function domain attacks (policy C) | `research.adversarial_domain.fgsm(model, x_np, y, eps, batch_size=64)`, `pgd(model, x_np, y, eps, alpha, steps, base_seed, batch_size=64, return_trajectory=False)`, `feasible_bounds`, `project`, `batch_seed`, `compute_logits`, `InfeasibleConstraintError` | `tests/test_adversarial_domain.py` (11 tests); executed in M1 C four-seed run |
| Experiment C training loop | `research.paper_described_adam.run_epochs(model, opt, X, y, seed, start_epoch, epochs, batch_size, seq_len, noise_std, on_epoch=None)`, `build_optimizer(cfg, model)`, `derived_seed(*parts)`, `epoch_batches`, `to_sequence`, `seed_everything` | `tests/test_adam_config_and_optimizer.py` (7 tests: Adam not AdamW, L2 decay enters gradient, 42 batches / final batch of 5, 1,050 steps) |
| Resume/fingerprint gate | `research.provenance.compute_fingerprint`, `check_resume`, `check_checkpoint_fingerprint` | `tests/test_resume_fingerprints.py` |
| Metrics and ASR definition | `research.evaluation.metrics(y, pred, clean, labels)` (accuracy, robust accuracy, macro-F1, ASR with clean-correct denominator, `None` when denominator 0, classwise report, confusion matrix); C's `evaluate_seed` also reports weighted-F1 and MCC | `test_incomplete_run_and_finite_metrics`; M1 reports |
| Transfer attacks (FP32->INT8, INT8(BPDA)->FP32, AT->INT8, FP32->AT-INT8) | `evaluation.transfer(source, target, x, y, ids, data, spec, config, folder, name, seed, source_kind)`; orchestrated in `runner.run` | pilot execution (`artifacts/pilot-verified/transfers.csv`) |
| CPU benchmark | `evaluation.benchmark(model, example, config)` (median/p95/throughput, raw samples, state bytes, RSS) | pilot execution |
| Run validation / measured-only analysis | `research.validate.validate_run(root, phase)`, `research.analyze.analyze(root, phase, output)`, `research.estimate.estimate(pilot, output)` | smoke/pilot runs |
| Dataset-schema fail-closed checks | `data.build_data` (hash, 39 numeric features, finite, 27 labels incl. `Benign_Final`) | `test_real_schema_split_and_sequences` |

### 4.2 Implemented but needs repair, adaptation or validation

| Component | Problem for M2A | Recommended action |
|---|---|---|
| `research/data.py::build_data` and `partitions(y, groups, split_seed)` | Uses `StandardScaler`, per-class caps, and `StratifiedGroupKFold(5)` twice, giving **71.2/17.9/10.9 %** (21,365/5,363/3,269) and validation classes as small as 6 rows. Isolation is correct (0 shared groups) but proportions miss 70/15/15, and the scaler is wrong for M2 (MinMax). Persisted manifest is per-run, not a standalone frozen split artifact | Add a **new** M2 split module (do not edit `build_data`); reuse its group-hash code and `partitions` assertions pattern |
| 2026 group split as an evaluation split | The pilot/smoke runs performed inference on capped subsets of its **test** partition (108 rows pilot, 54 rows smoke; `docs/VERIFIED_RESULTS.md`), so it is partially exposed; also its validation was used for the historical attack-budget selection | If reused, disclose; the recommendation in sec 5 is a new split |
| `research/training.py::train` | Not Experiment C (see sec 3 diff table). Also shares one `torch.Generator` (`seed+1000`) between `DataLoader` shuffling and noise draws; AT validation selection uses attacks per epoch; saves `best_state`. AT branch is correct in structure (attack in eval mode, restored; `autograd.grad` on inputs; 0.5/0.5 CE mix; fresh init) but uses StandardScaler-unit eps 0.1, 3 steps, FGSM candidate retention, per-position perturbation | Reuse structure only. Build the M2 trainer by calling C's `run_epochs` pattern (derived isolated generators, Adam 1e-3/1e-4, batch 512, final epoch) plus an AT branch; do not copy `train()` hyperparameters |
| `research/attacks.py::attack` semantics | (a) Perturbation is independent per `(position, feature)` over the repeated tensor (50x39 free variables per row); (b) PGD returns the **best candidate over FGSM + random start + every iterate + restarts** (strongest-of), not the final iterate, which is stricter than C's final-iterate policy and must be named explicitly; (c) feasible set = `[min(train_lo,x), max(train_hi,x)]` intersect `x +/- eps` (extends bounds to include the clean point, so it is never empty and eps=0 is identity); (d) no explicit guard that `lower/upper` correspond to a MinMax [0,1] space | Keep as the core engine; add the M2 decisions on row-tied vs per-position perturbation and final-iterate vs best-of (sec 7), parameterize bounds as [0,1] for MinMax |
| `research/adversarial_domain.py` (policy C) | `feasible_bounds` **raises** when `max(x-eps,0) > min(x+eps,1)`. M1 split's validation contained 2 rows with value 2.0 (infeasible at eps 0.1; they were documented, not attacked). That fails the M2 requirement of a consistent policy over the whole grid | Do not use as the M2 domain policy. Use `attacks.attack`'s extended-bounds policy (sec 7) |
| `research/evaluation.py::evaluate` | Works on any partition named by `config["evaluation_partition"]`; the pipeline has no guard against `test`. Gradient diagnostic probes only 2 examples (`x[:2]`) | Add a hard M2A guard (validation only; refuse `test`), enlarge diagnostics to the full evaluated set (zero-gradient fraction per example) |
| `research/runner.py::run` and `select_attack` | Monolithic; hard-wires `train(...)`, StandardScaler data, the generic epsilon list (`0, .025, .05, .1, .2`), `phase test` flow; attack-budget selection uses seed-1 FP32 vs all-linear INT8 on a validation subset with candidates `10/1, 20/1, 20/3` | Reuse its unit orchestration, `expected_units`, `calibration`, `save_quantized`; add an M2 config and an M2A mode that cannot touch `test`. Do not fork a second runner framework |
| `runner.calibration` | Selects `calibration_per_class` training rows with `stratified_cap(..., split_seed+99)`; applies no training noise (clean repeated sequences). Verified to use training only | Reuse. Same IDs must be reused for FP32-derived and AT-derived PTQ (already the case: `cal` is computed once) |
| `runner.qat_feasibility` | Fine-tunes the **ordinary** FP32 model for `qat_epochs` (config; 1 in smoke) at lr 1e-4 on clean repeated sequences with no training noise, batch 32, one seed. Conversion and actual INT8 inference verified, but there is **no matched FP32 fine-tuning control**, so any QAT benefit would be confounded with extra epochs | Keep as feasibility check; add the matched control to the proposed full plan (sec 8) |
| `quantization.load_int8` | Re-runs PTQ calibration on reload to rebuild observers then overwrites with the saved state; correct but requires the same calibration tensor | Document; reuse |
| `evaluation.benchmark` | INT8 slower than FP32 at batch 1 on this CPU in the pilot; RSS is process-wide | Report honestly; keep as secondary |
| `milestone1_report.py`, `paper_described_adam_report.py`, `analyze.py` | Tied to M1 run layouts and StandardScaler wording ("training-standardized feature units"; plot label in `analyze.py` line ~79) | Write M2 reporting as additions; do not edit M1 report modules |

### 4.3 Missing (must be built; extend existing modules)

1. Frozen M2 group-aware split artifact (standalone JSON with row IDs, group hashes, seed, per-class counts, conflicting-label report, SHA-256s) and a MinMax scaler fitted on train rows only, shared by every variant and seed.
2. Read-only partition-integrity / class-coverage / input-range audit tool for the new split (no model inference).
3. `configs/milestone2_*.json` freezing: Experiment C training, AT recipe, PTQ policy, epsilon grid `0, 0.01, 0.025, 0.05, 0.10`, exact PGD steps / step-size rule / restarts / random-start seeds, calibration selection, and a validation-only M2A guard.
4. Isolated RNG for AT: generating adversarial examples must not consume the clean model's shuffle/noise stream (C's derived generators already give this; AT needs its own derived seeds). Dropout masks cannot be paired between clean and AT (AT does extra forward passes in train mode) — document.
5. Adversarial-training trainer matching C's optimizer/budget, with recorded additional compute (attack forward/backward passes, wall-clock, update count).
6. Row-tied perturbation mode (single 39-d delta broadcast over 50 positions) and an explicit decision on per-position vs row-tied.
7. PGD-budget adequacy check on validation only (steps / restarts sweep, loss monotonicity, saturation), plus epsilon-zero and non-regression checks for the new grid.
8. Fake-quantization/surrogate attack (QAT-prepared model as gradient source) and an explicit "direct gradient through actual INT8 kernels" diagnostic (the quantized `torch.ops` backward is undefined/zero) so missing gradients are never read as robustness. Only transfer and BPDA exist today.
9. Matched FP32 fine-tuning control for QAT; QAT with training noise and C-compatible budget.
10. Bounded M2A pilot driver (training and validation only) and the M2 documentation (protocol, pilot report, full-campaign plan with compute estimate).
11. Tests for all of the above (`tests/test_m2_*.py`).

## 5. Data and existing-split audit (read-only; test partition touched only for ID/count/range)

Dataset: 29,997 rows x 39 features, 27 classes x 1,111 rows. All values finite.

### 5.1 Duplicate-feature structure (group = SHA-256 of the float64 feature row, label excluded)

- **11,953 unique feature vectors**; 18,044 rows are repeats of an earlier vector.
- Group sizes: 11,154 singletons, then 416 pairs, 138 triples, ... ; **15 groups hold >= 100 rows and
  together contain 15,412 rows (51.4 % of the dataset).**
- Largest groups (rows: label composition): 2,980 (class 7: 1,020, class 9: 1,030, class 15: 930);
  2,020 (class 10: 985, class 16: 1,035); 1,814 (class 11: 1,008, class 17: 803, plus one row each of 0, 23, 26);
  1,024 (class 3 only); 997 (class 6); 990 (class 5); 975 (class 20); 947 (class 19); 937 (class 21);
  814 (class 4); 727 (class 12); 656 (class 1).
- **229 feature groups carry conflicting labels**, containing 8,697 rows. They cannot be classified
  correctly by a deterministic feature-only model; they must stay in one partition and be reported,
  never relabeled or dropped.
- Many classes are dominated by one identical vector (e.g. class 3: 1,024 of 1,111 rows; class 6: 997;
  class 5: 990; class 20: 975). Under group isolation such a class has very few "other" vectors to supply
  validation/test rows. Groups per class range from 47 (class 16) to 1,078 (class 13).

### 5.2 Existing partitions

| Split | train / validation / test | Shared feature groups between partitions | All 27 classes in each | Notes |
|---|---|---|---|---|
| M1 stratified row split (seed 2025; used by B and C) | 20,997 / 4,500 / 4,500 (70/15/15) | **train-val 294, train-test 315, val-test 165** (62 % of test rows match train) | yes (166-167 per class held-out) | Not group-isolated. Conflict groups inside partitions: 148 / 38 / 43. **Not valid for M2** |
| Research split, `data.partitions(y, groups, 2026)` | 21,365 / 5,363 / 3,269 (71.2 / 17.9 / 10.9 %) | **0 / 0 / 0** | yes; min class count train 53, validation **6**, test 13 | Valid isolation; imbalanced fractions; inference exposure of capped test subsets in smoke/pilot (sec 4.2) |

### 5.3 Is a 70/15/15 group split feasible?

- A plain `StratifiedGroupKFold` composition is unsuitable: 20 folds recombined 14/3/3 (seed 2026)
  gave 91.5 / 7.1 / 1.4 % and **lost a class** in validation and in test, because the giant groups land in
  whole folds. This is a concrete constraint, not an option to be silently relaxed.
- A deterministic, label-count-only local search (objective: total fractions near 70/15/15 plus
  per-class fraction deviation; **no model results used**) reached **74.8 / 12.6 / 12.6 %**
  (22,445 / 3,776 / 3,776 rows), all 27 classes present in every partition, minimum per-class counts
  38 (validation and test) and 678 (train). A greedy variant gave 78.7 / 10.7 / 10.6 %.
  Exact 70/15/15 is therefore unreachable while keeping every class represented, because 15 groups hold
  51 % of the rows and several span three classes. These runs were exploratory; nothing was saved.
- **Recommendation:** build one new deterministic split with a label-count-only optimizer, fixed split
  seed declared in the config before any training, whole groups assigned atomically (labels irrelevant),
  every one of 27 classes required in all three partitions with a documented minimum, and report the
  achieved fractions. Reusing the 2026 split is the fallback if the user prefers continuity, at the cost
  of the 10.9 % test fraction, class-6-row validation classes and the pilot exposure.
- Describe the result as a **new group-disjoint evaluation partition within a previously examined dataset**,
  not an untouched external benchmark; the dataset was seen by A/B/C, the 2025 test partition was
  evaluated, and 61 distinct test rows were exposed to the old smoke selection
  (`reports/milestone1-repair/old_smoke_overlap.json`).

### 5.4 Input ranges under MinMax (train-fit; read-only)

| Split | Held-out values outside [0,1] | Rows infeasible for `max(x-eps,0) <= min(x+eps,1)` at eps 0.01 .. 0.10 |
|---|---|---|
| M1 2025 split | validation: 2 cells > 1 (max **2.0**); test: none | validation 2 rows (row IDs 14554, 3420); test 0 |
| 2026 group split | validation: none; test: **1 cell > 1 (max 2.0)** | test 1 row at every eps in the grid |

Interpretation: a few rare rows have a feature at twice the training maximum under any split that
leaves them held out. The infeasible-row problem is partition-dependent and cannot be assumed away.
The M2 domain policy must keep these rows, keep clean inputs unclipped, never enlarge epsilon and
never produce an empty feasible set (sec 7). No feature is constant in training under either split
(`scaler.data_max_ == data_min_` count 0).

## 6. Adversarial and PTQ functionality details for later steps

- Epsilon units: in MinMax space eps is a fraction of each feature's training range; the generic pipeline
  and M1 A used standard-deviation units. The frozen grid `0, 0.01, 0.025, 0.05, 0.10` is a new,
  incomparable scale; report it as such.
- M1 C PGD: 7 steps, alpha 0.025, random start, final iterate, per-position; existing pipeline AT/eval
  PGD: 3-step (training) and 10/20-step with 1-3 restarts (evaluation), alpha = eps/4, best-of-iterates.
  No M2 value is frozen yet.
- Gradient masking: INT8 static kernels have no autograd. Existing safeguards: BPDA with zero forward error
  (asserted), FP32->INT8 and AT->INT8 transfer, nonzero-gradient probe on 2 examples. Missing: full-set
  zero-gradient accounting and a fake-quant surrogate attack.
- AT compute: pilot measured 74.35 s vs 11.23 s (6.62x) at identical epochs/updates with the 3-step PGD
  plus FGSM candidate; Experiment C budgets are 1,050 steps per model, so AT adds roughly (steps+2)
  extra forward/backward passes per step. On this i5-10210U (4 cores, 1 thread) estimate in the pilot
  suggested ~200 h for the old full matrix; the M2 campaign must be re-estimated from a bounded M2A pilot.

## 7. Reuse-versus-build recommendations

Build **no second framework**. Extend the existing `research` package by adding small, separately named modules:

| New artifact | Reuse from | Notes |
|---|---|---|
| `research/m2_split.py` (frozen split + MinMax scaler + audit) | group hashing and isolation assertions from `research/data.py`; `provenance.fit_scaler`/`canonical_hash`; `common.atomic_json` | write split to a new directory, e.g. `reports/milestone2a/split/` or `runs/milestone2a-*/data/`; never touch `reports/milestone1-*` |
| `configs/milestone2_baseline.json` (+ docs) | schema/validation style of `read_adam_config` | Experiment C values verbatim, plus split/AT/PTQ/attack/epsilon sections |
| `research/m2_training.py` | `paper_described_adam.run_epochs`, `build_optimizer`, `derived_seed`, `to_sequence`, `seed_everything`; AT inner step from `research.attacks.attack` | final epoch checkpoint; AT uses derived attack seeds and eval-mode attack generation restoring train mode |
| attack extension (in `research/attacks.py` or new `research/m2_attacks.py`) | `attacks.attack` engine, extended-bounds policy | add row-tied option; set bounds `[0,1]` extended to include clean point; document "best-of" vs "final iterate" |
| PTQ | `quantization.ptq/prepare_static/inventory/load_int8/bpda_model`, `runner.calibration/save_quantized` | one workflow: eager static fbgemm per-channel qint8 Linear islands, calibration = fixed training rows, same IDs for ordinary and AT models |
| QAT | `quantization.prepare_static(qat=True)`, `runner.qat_feasibility` pattern | bounded feasibility only; matched FP32 control specified, not run in M2A |
| Evaluation/validation | `evaluation.evaluate/metrics/transfer`, `validate.validate_run` | add a refuse-`test` guard for M2A |
| Tests | `tests/test_pipeline.py` fixtures (`small_model`, `cfg`), `test_adversarial_domain.py` patterns | new `tests/test_m2_*.py`; existing 71 must keep passing |

### Proposed M2 attack-domain policy (recommendation to adopt in design)

Perturbation space: MinMax-scaled feature space. Feasible set for eps > 0:
`lower = max(x - eps, min(0, x))`, `upper = min(x + eps, max(1, x))` (identical to `attacks.attack` with
`lower=0, upper=1`). Never empty; eps=0 returns an exact copy; clean values are never clipped or dropped;
epsilon is never enlarged. An out-of-range coordinate can only move toward the interior within `[x-eps, x]`
(or `[x, x+eps]`), which is documented as a conservative restriction for those rows. Apply it to all five epsilons
identically. Because the policy extends beyond [0,1] only for the clean point, adversarial points stay
inside the clean data's own support.

### Decisions the design step must make (with recommended defaults)

1. **Row-tied vs per-position perturbation.** Recommend **row-tied** as primary (one 39-d delta added to
   every repeated position; the sequence stays a repetition of one feature row, so the attack is a feature-row
   attack), with the inherited per-position mode kept only as a labeled sensitivity check. Rationale: the
   synthetic sequence has no temporal meaning, and 50 independent positions give the attacker 50x more
   freedom than a single feature row has. Training-time noise remains independent per position (inherited).
2. **PGD definition.** Recommend final-iterate PGD with fixed steps, alpha = eps/4, uniform random start,
   explicit restart count, seeds derived by hash, evaluated both as final-iterate and best-of; validate
   adequacy on validation with a steps/restarts sweep (loss and accuracy saturation).
3. **AT recipe.** Starting choice: PGD on training examples against the current model in eval mode, 50 %
   clean CE + 50 % adversarial CE, fresh init (not from the FP32 checkpoint), same Adam/budget as C.
4. **Which evaluation partitions.** M2A: validation only. The test partition is read-only for integrity,
   coverage and range checks.

## 8. Constraints and risks for the implementing steps

- **Memory:** ~0.8 GB RAM free; train/evaluate one model at a time, keep pilot batch sizes small, do not
  keep many converted models resident, and avoid parallel jobs.
- **Time:** 1 thread, 4 physical cores. The bounded pilot should be small (e.g., seed 1, a few epochs on a
  reduced budget or a stratified training subset) and clearly labeled as a pilot; the full four-seed
  campaign must not be started.
- **Determinism:** `common.seed_all` enables `torch.use_deterministic_algorithms(True)`; static INT8 kernels
  hit an unsupported op (`quantized_resize_cpu_`) so `LinearIsland.forward` temporarily disables the guard
  per call. C's own runner does not set deterministic mode. Choose one policy for M2 and record it.
- **Source-digest coupling:** adding or editing files under `research/`, `configs/`, `scripts/`, `tests/` changes
  `provenance()["source_digest"]`; completed historical run roots will no longer pass resume checks and that
  is expected. Never try to bypass the gate; never resume historical runs.
- **Git:** the working tree was clean at audit time and the user forbids commits/pushes in this task.
- **Test-set integrity:** `runner.run --phase test` and `evaluate(... evaluation_partition="test")` will run
  inference on test. M2A tooling must refuse it.
- **Historical-run safety:** use new names such as `runs/milestone2a-pilot`, `artifacts/milestone2a-pilot`.
- Known unverified items: PT2E failure cause not diagnosed; Linux/WSL not tested; AVX2-only CPU explains
  slow INT8 only as a hypothesis; the 74.8/12.6/12.6 % split figure comes from an exploratory optimizer run
  that was not saved (the later step must regenerate and persist its split deterministically).
