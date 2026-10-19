# Local progress and resume

## Active released-artifact baseline execution (2026-10-20)

The user now authorizes the full local FP32 multiclass baseline for replication
seeds 1, 7, 13, 29, overriding the earlier execution block and two-hour threshold.
No professor/author clarification is a prerequisite for this artifact attempt.
Exact Table 4 correspondence remains unresolved; the required claim is
"Released-artifact multiclass baseline and documented paper-reproduction attempt."

- No valid full-budget artifact runs existed at the start; all previous runs and
  the untracked user file `IOT DL.zip` are preserved.
- Frozen configuration: configs/milestone1_artifact_full.json. Full dataset,
  upstream StandardScaler-before-80/20-split, seed-dependent partitions, synthetic
  length-50 sequences with .01 training noise, ordinary 96/4/3/384 Transformer,
  Adam .001 / zero weight decay, 15 epochs, batch 512, final checkpoint only.
- Evaluation: same 3000 capped validation rows per seed, cap seed 42, clean/FGSM
  .10/PGD .10 with alpha .01 and 10 clean-start steps, epsilon-box projection
  without domain clipping. Upstream attack functions, evaluation batch 32.
- Entry point: research.artifact_baseline. It instruments the upstream train_seq
  only for epoch checkpoint/resume, with optimizer and torch RNG state. Corrected
  research protocol and historical smoke/target gate are unchanged.
- Run path: runs/milestone1-artifact-full. Completion requires each seed's saved
  final checkpoint, 15 epochs / 705 optimizer steps, aligned predictions, finite
  recomputed metrics, sanity checks and artifact manifest; a launch is not completion.
- Exact continuation from the repository root:
  `.\.venv\Scripts\python.exe -B -m research.artifact_baseline --run-dir runs/milestone1-artifact-full --resume`
- Full completion including verification, report/figure and local review ZIP:
  `.\.venv\Scripts\python.exe -B scripts/complete_milestone1.py`
  If the controller is already active, its OS lock prevents a duplicate controller.
  The current controller waits for the first seed to finish and validate before
  launching the remaining sequential seeds. `runs/milestone1-artifact-full/controller.json`
  records its current state; `controller.log` records commands and failures.
- Preflight: six tests passed, including exact training-adapter and interrupted-
  resume equivalence, upstream epsilon-zero/bounds and unchanged model parameters.
- Full training has started; this source/progress commit does NOT establish a
  completed experiment. Only verified per-seed manifests and the generated
  completion.json count towards the result package.
- Seed 1 subsequently completed and passed independent verification: all 15
  epochs / actual 705 Adam steps, 3000 aligned evaluation rows, final checkpoint
  reload, metric recomputation, confusion matrices, epsilon-zero/bounds and
  unchanged evaluation parameters. Accuracy clean/FGSM/PGD =
  66.0333% / 43.4000% / 39.6667%; macro-F1 =
  61.2889% / 41.2185% / 38.7235%. One seed has no across-seed SD.
  Training 1293.75s; evaluation 94.99s; invocation after imports 1400.41s.
- The completion controller reused seed 1 and started seed 7 only after independent
  validation. Seeds 13 and 29 remain queued. Generated report/figure currently
  include completed seeds only; their completion.json is the authoritative count.
- No quantization, adversarial training, transfers, ablations or remote writes.

The earlier execution-block entries below are historical and superseded by this
authorization. The exact-paper scientific limitation is retained separately.

## Active Milestone 1 reference investigation (2026-10-18)

- Scope is Milestone 1 only. No full reference seed, later-milestone study,
  quantization, adversarial training, ablation or multi-seed run authorized here.
- Read supplied full PAVE PDF; visually verified all seven candidate Table 4
  CICIoT Std.Trans. values. Exact multiclass reference is NOT identified.
- Paper describes binary context, attack-positive F1 and separate macro-F1;
  Table 4 mode, F1 field, ASR denominator and seed aggregation are not proven.
- Confirmed paper-versus-called-code differences in data composition, scaling,
  partition, optimizer decay, epochs, PGD and output metrics. The paper also
  contains conflicting PGD step-size descriptions. See docs/MILESTONE1_REFERENCE.md.
- configs/, docs/, experiments/ exist and are tracked. IOT DL.zip omitted them;
  previous source-only export contains them. Added read-only source export checker.
- Separate public inspection clone matches pinned e73911b; two commits, main,
  v1.0; release API has no attached assets. No Table 4 generator, checkpoints,
  split IDs, original configs or per-seed outputs found in the public history.
  Anonymous file API exposes root/README and six scientific code files matching
  public text/AST. Client script, nested listings and processed CSV returned 403;
  remaining inventory unknown. All requests bounded; no repeated failing URLs.
- Added configs/milestone1_target.json and separate research/milestone1.py:
  route A training blocked, full B training disabled; explicit bounded B helper smoke.
- Four gate tests passed. Final real-data component smoke: one seed/one epoch,
  54 training and 27 validation rows; Standard Transformer only. Exact saved-scaler
  reconstruction, checkpoint reload, same-checkpoint clean/FGSM/PGD, attack bound,
  saved artifact checksums/coverage passed. Two development/final component smokes
  are preserved separately; neither is a scientific result.
- Final smoke train 8.45 s; clean/FGSM/PGD 0.06/0.18/1.06 s; total after imports
  16.51 s. Full reference runtime/command remain unknown, rather than guessed.
- Clarification drafted in docs/MILESTONE1_CLARIFICATION.md; NOT sent.
- Earlier corrected pipeline, environments, results and commits are preserved.
  Pinned upstream is unchanged. All work local; only read-only remote inspection.
- Milestone 1 remains INCOMPLETE pending reference identity/materials. The next
  scientific step is resolving that clarification, then preparing one reference seed.

The sections below record earlier work and are not a mandate to resume the full study.

Project: C:\Users\loran\Documents\Codex\2026-10-17\files-pasted-by-the-user-you\IOT-network-traffic-DL
Branch: local/research-pipeline-fixes.
Starting state: clean main, c511dfe468036ad3bf92137b2ee809ce30bdc482.
Reference: e73911bbdb241f3fb6c5de346139010c4aadefd4, unchanged.
No applicable AGENTS.md found. All work remains local.

## Completed

- Inspected native Windows/PowerShell, CPU, RAM, disk, Python installations,
  Intel GPU and absence of usable CUDA; selected a coherent CPU-only workflow.
- Installed portable Python 3.11.9 and isolated .venv with torch 2.5.1+cpu.
  No global environment, PATH or Git configuration changed.
- Downloaded processed PAVE data; pinned its origin/hash/schema/labels.
- Audited and repaired wrong FGSM selection, repeated training/output collisions,
  validation mislabeled as test, pre-split preprocessing, incomplete seeding,
  dataset-switching quick mode, interpreter selection, pin/failure checks.
- Implemented duplicate-group splits, training-only scaling, isolated synthetic
  sequences, ordinary/AT checkpoints, shared clean/FGSM/PGD evaluation, static
  PTQ, dynamic quantization support, four ablations, real INT8 exact-forward BPDA,
  four transfers, converted QAT feasibility, CPU efficiency, result validation,
  completed-unit resume, analysis and full-study configurations.
- Real data has 18,044 duplicate feature rows, 17,721 duplicate labeled rows,
  and 229 conflicting-label feature groups. Full grouped split counts:
  21,365 train / 5,363 validation / 3,269 test.
- Documentation: docs/AUDIT.md, docs/PROTOCOL.md, docs/SETUP.md,
  docs/VERIFIED_RESULTS.md, docs/environment-windows.json.
- Local commits: 3ff878d (setup/protocol), 222892c (pipeline/tests),
  14c387e (INT8 backend guard), 39c3f2c (artifact/runtime hardening),
  followed by the final verification/documentation commit.

## Verified commands and outcomes

Commands used explicit .venv/Scripts/python.exe in native PowerShell.

- pip install -r requirements.txt: succeeded in the project environment.
- pip check: no broken requirements.
- scripts/setup.ps1: succeeded, reused the resolved environment and verified pin.
- python -m pytest -q --junitxml=work/tests.xml:
  23 passed (115.27 s console; XML has test-suite timing).
- Focused INT8 regression: 12 passed, 11 deselected, 92.95 s.
- Focused runtime/resume/completeness regression: 4 passed, 20 deselected, 68.36 s.
- python -m research.runner --config configs/smoke.json --run-dir runs/smoke-verified --phase both:
  27 completed units, 72 measured metric rows. Real CICIoT, 216/54/54 rows,
  seed 1, one epoch; includes actual converted QAT.
- python -m research.runner --config configs/pilot.json --run-dir runs/pilot-verified --phase both:
  25 completed units, 82 measured metric rows. Real CICIoT, 864/108/108 rows,
  seed 1, one epoch; all mandatory ablations and transfer paths.
- python -m research.analyze for smoke/pilot: succeeded; plots visually checked.
- python -m research.estimate runs/pilot-verified --output artifacts/pilot-estimate:
  central 200.84 hours, heuristic planning range 200.84–803.34 hours;
  conservative output-storage estimate 2.76 GB; sampled process RSS about 243 MB.
  Memory/storage/runtime forecasts are estimates, not measured full-study peaks.
- work/check_completed_artifacts.py: six checks passed using an isolated copy:
  compatible completed-training resume preserves checkpoint, incompatible resume
  rejected, output collision rejected, missing metric row rejected, empty
  predictions rejected, corrupt INT8 state rejected. Original runs untouched.
- FP32/actual INT8 reload predictions identical; BPDA clean forward error zero.
- research.reproduce and repaired FGSM wrapper dry-runs succeeded.
- Upstream remains clean; no remote actions were performed.

Pilot test FP32 clean accuracy 0.3148, PGD robust accuracy 0.2315 at epsilon .1,
macro-F1 clean 0.2601. All-linear PTQ clean 0.3241, PGD robust 0.2037. These are
undertrained one-seed/one-epoch pilot measurements, not full scientific results.
Pilot training: FP32 11.23 seconds; AT 74.35 seconds. Full tables are in the report.

## Decisions and limitations

- Full architecture remains unchanged: 351,612 parameters, width 96, 4 heads,
  3 blocks, FFN 384, 27 classes, 50 synthetic repeated-row positions.
- Static PTQ converts 23 Linear modules in the all-linear case, including all
  12 explicit attention projections. Other operators remain floating point.
- Float deterministic guard is strict; actual static Linear kernels alone are
  narrowly exempted because torch 2.5.1 rejects quantized_resize_cpu_ under it.
- INT8 gradients are approximate BPDA with real target forward kernels.
- Calibration and AT use training only; validation selects/fixes checkpoints
  and attack budgets. No test performance was used to alter full hyperparameters.
- Test data has missing capture/device/flow provenance and substantial duplicates.
  Feature-space attacks do not establish realizable packet attacks.
- Smoke revision 222892c had native warning timing overhead; use pilot efficiency
  measurements, not smoke timings. Pilot revision was 14c387e.
- Later hardening changed source fingerprints; current CLI intentionally rejects
  resuming the historical smoke/pilot roots after source changes. Their completed
  artifacts remain validated/analyzable. New experiments require a fresh root.
- Completed-unit resume only; no mid-training optimizer/RNG recovery is claimed.
- Full publisher HTML/PDF returned 403/cache failures. Accessible publisher
  abstract and pinned source were read; no exact matching paper baseline claimed.
- QAT actual conversion is feasible and verified in a bounded real-data smoke.
  A full four-seed QAT comparison, full paper reproduction, Linux execution and
  isolated per-model peak memory remain unverified.

## Remaining tasks

- User-initiated full four-seed study; no overnight/multi-day matrix launched.
- Inspect full validation diagnostics/frozen settings, then execute final test.
- Draw research conclusions only after full results and paired seed uncertainty.
- Optional expanded QAT study and exact paper mapping when full text is available.

## Next exact commands (PowerShell, from project root)

The full study is lengthy. Review docs/VERIFIED_RESULTS.md before launching.

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase validation
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase test --resume
.\.venv\Scripts\python.exe -m research.analyze runs/full-study --phase test --output artifacts/full-study-test
```

To rerun bounded verification, choose NEW directories:

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/smoke.json --run-dir runs/smoke-new --phase both
```

Nothing was pushed, published, uploaded, or sent to remote CI. No unrelated user
changes were staged, reset, stashed, discarded or overwritten.
