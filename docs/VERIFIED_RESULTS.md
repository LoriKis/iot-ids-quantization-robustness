# Verified local implementation and bounded experiments

> **Status/context:** Historical corrected-pipeline smoke/pilot results (2026-10-17). These are bounded implementation checks, not the completed Milestone 1 artifact baseline or full later-stage research results. The paper-access failure and earlier baseline status below are historical; the full paper was subsequently inspected.
> Current authoritative results: [Milestone 1 report](../reports/milestone1-artifact/REPORT.md).

All changes and results remain local. Nothing was pushed, published, uploaded, or sent to remote CI.

Project: repository root (`IOT-network-traffic-DL/`).
Branch: `local/research-pipeline-fixes`.
Starting revision: c511dfe468036ad3bf92137b2ee809ce30bdc482 (clean main).
PAVE: e73911bbdb241f3fb6c5de346139010c4aadefd4 (unmodified).

## What is implemented

Training-only scaling and fixed duplicate-group splits; saved baseline/AT checkpoints; evaluation-only clean/FGSM/PGD wrappers; strongest-candidate attacks; actual static INT8 classifier/FFN/attention/all-linear ablations; exact-forward BPDA; AT→PTQ; all four transfer directions; actual converted QAT feasibility; controlled CPU benchmarking; provenance, failure handling and compatible completed-unit resume; measured-only analysis and full-study configurations.

The model is PAVE's unchanged 351,612-parameter Standard Transformer: width 96, four heads, three blocks, FFN 384, 27 classes, 50 repeated-row positions. The sequences are synthetic repetitions, not measured temporal traffic. Static PTQ converts 23 Linear modules for all-linear, including 12 explicit Q/K/V/output projections; attention matmuls/softmax/LayerNorm remain FP32. BPDA uses real INT8 forward kernels and approximate gradients, not exact INT8 white-box differentiation.

## Verified execution

| Check | Outcome |
|---|---|
| complete suite | 23 passed, 115.13 seconds |
| INT8 regression | 12 passed, 92.67 seconds |
| resume/completeness regression | 4 passed, 68.39 seconds |
| Dependency compatibility | pip check: no broken requirements |
| Real-CICIoT smoke | 27 completed units, 72 metric rows; 216 train / 54 validation / 54 test, seed 1, one epoch |
| Real-CICIoT pilot | 25 completed units, 82 metric rows; 864 train / 108 validation / 108 test, seed 1, one epoch |
| Checkpoint reloads | FP32 and actual INT8 round-trip predictions identical |
| Completed-artifact negative checks | Compatible training resume preserves checkpoint; incompatible resume/collisions/missing rows/empty predictions/corrupt INT8 state rejected |
| BPDA agreement | Full evaluated clean sets: zero target/surrogate forward error |
| QAT | Real-data smoke converted all 23 Linear modules and evaluated actual CPU INT8 inference |
| Upstream reproduction | Command dry-run only; original algorithm is isolated and row-completeness checked |

The full suite was followed by focused INT8 regressions after narrowing the backend determinism exception. Strict deterministic float execution remains enabled; unsupported static INT8 kernels are exempted only during their calls. Smoke timings had native warning I/O and are not used for efficiency conclusions.

Measured run revisions: smoke 222892c7e5b5ff8a14c9b6110e05fcfd9dbd9f18; pilot 14c387ead76c90fea7016e39fbabcb78483c0257. Later changes strengthened validation/runtime matching and localized plotting caches; model/training/attack mathematics were unchanged. Current strict source fingerprints intentionally reject CLI resume of those historical run roots after source hardening. Their completed artifacts remain validated and analyzable; use a fresh run root for a new execution.

Pilot training measured 11.23 s FP32 and 74.35 s AT (6.62×) at the same epoch/update budget. This includes validation selection but excludes environment/setup. Validation froze PGD 5 steps, 2 restarts, alpha=epsilon/4.

## Pilot test measurements — undertrained, not full scientific results

108 matched test examples (four per class), one seed, one epoch. No between-seed standard deviation can be estimated. Clean/robust accuracy and macro-F1 below are fractions, not percentages. Robust accuracy requires both clean and perturbed correctness. PGD uses epsilon .1 in training-standardized feature units.

| Variant | Clean accuracy | Clean macro-F1 | PGD robust accuracy | PGD macro-F1 | ASR successes/clean-correct |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.3148 | 0.2601 | 0.2315 | 0.1816 | 9/34 |
| ptq_all_linear | 0.3241 | 0.2665 | 0.2037 | 0.1607 | 13/35 |
| at_fp32 | 0.2963 | 0.2201 | 0.2222 | 0.1579 | 8/32 |
| at_ptq_all_linear | 0.2870 | 0.2132 | 0.2222 | 0.1587 | 7/31 |
| ptq_classifier | 0.3241 | 0.2750 | 0.2037 | 0.1594 | 13/35 |
| ptq_ffn | 0.3148 | 0.2601 | 0.2315 | 0.1822 | 9/34 |
| ptq_attention | 0.3056 | 0.2537 | 0.2315 | 0.1824 | 8/33 |

These sample-limited, one-epoch values do not establish whether AT or INT8 improves the research trade-off. The data is a balanced processed subset with substantial duplication, not a natural traffic prevalence benchmark.

## Pilot CPU efficiency

One CPU intra-op thread, one inter-op thread, FBGEMM, sequence length 50, warmup 5 and 20 repetitions. Complete upstream forward on preallocated matched input; excludes data preparation/attack generation. Raw samples, p95 latency and throughput are retained. Small repetition counts, thermal throttling and background activity limit precision.

| Variant | State bytes | Batch-one median ms | Batch-32 median ms | Batch-32 examples/s |
|---|---:|---:|---:|---:|
| fp32 | 1448741 | 13.953 | 102.409 | 302.0 |
| ptq_all_linear | 497448 | 20.478 | 129.727 | 234.5 |
| at_fp32 | 1448741 | 8.602 | 93.193 | 265.8 |
| at_ptq_all_linear | 497448 | 20.249 | 85.909 | 325.1 |
| ptq_classifier | 1436545 | 14.267 | 106.212 | 296.5 |
| ptq_ffn | 820281 | 9.244 | 82.885 | 370.9 |
| ptq_attention | 1159916 | 13.890 | 156.130 | 201.7 |

Serialized state includes packed quantized weights and metadata. Process RSS includes other resident models and allocator caches; no isolated model-memory reduction is claimed. This is a CPU-vs-CPU comparison.

## Pilot transfer measurements

PGD epsilon .1; INT8→FP32 generates examples through actual INT8 forward plus BPDA gradients. AT-FP32→INT8 targets the ordinary quantized model. Target ASR denominator is target clean-correct examples; both-source-and-target eligibility is separately retained in raw metrics.

| Direction | Target robust accuracy | Target ASR successes/clean-correct |
|---|---:|---:|
| at_fp32_to_int8 | 0.2407 | 9/35 |
| fp32_to_at_int8 | 0.2407 | 5/31 |
| fp32_to_int8 | 0.2407 | 9/35 |
| int8_to_fp32 | 0.2315 | 9/34 |

## Data and planning estimate

The 29,997-row CSV has 18,044 duplicate feature rows, 17,721 duplicate labeled rows and 229 conflicting-label feature groups. The fixed group-isolated full split is 21,365 / 5,363 / 3,269 rows. Capture/device/flow grouping identifiers are absent. No exact duplicate feature group crosses partitions; near-duplicate/capture leakage cannot be excluded.

A linear pilot extrapolation gives **200.8 hours**, with a conservative planning range **200.8–803.3 hours** for the full four-seed mandatory study, including validation and test. This is a heuristic, not a confidence interval. It assumes the largest predeclared full PGD setting (20 steps, three restarts), includes eight trainings, all mandatory curves/ablations/transfers, and excludes optional QAT, reproduction and setup.
Conservative output-storage extrapolation: 2635 MiB; observed pilot process RSS 232 MiB; planning process RAM 463 MiB. These are not measured full-study peaks; the existing environment/cache and downloaded reference also consume disk.

The full matrix was **not launched**. Full scientific results, four-seed uncertainty and any full QAT comparison remain pending. QAT conversion is feasible on this backend; only its bounded deployment feasibility was measured. Linux/WSL execution and isolated per-model peak memory are unverified. Exact paper reproduction is unverified because publisher full HTML/PDF access failed; accessible primary abstract and pinned source were read, and no directly matching reported baseline was established.

## Next exact command

After reviewing the estimate, explicitly start the lengthy full validation phase:

```powershell
# Run from your checkout root.
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase validation
```

After validation completes and the attack protocol is frozen:

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase test --resume
.\.venv\Scripts\python.exe -m research.analyze runs/full-study --phase test --output artifacts/full-study-test
```

Completed-unit resume requires unchanged source/config/environment and validates artifacts. Failed or partial units are preserved and require a new run directory; there is no mid-training resume.

Audit, protocol, setup, raw metrics, plots and estimate are included in this local report folder. Large data, environments and checkpoints remain ignored in the local project. Nothing was pushed or published.
