# Audit and requirements coverage

Audit starting point: clean `main` at c511dfe468036ad3bf92137b2ee809ce30bdc482.
Reference PAVE: e73911bbdb241f3fb6c5de346139010c4aadefd4. New local branch:
`local/research-pipeline-fixes`. No applicable AGENTS.md found in the workspace
or its ancestors, or in either checkout. Historical files remain accessible in
the starting commit. There was no pre-existing user work to preserve.

## Confirmed findings

File references below are repository-relative; upstream references refer to the
unchanged pinned `upstream/PAVE_artifact` checkout.

| Historical finding | Verified evidence at the starting revision | Correction |
|---|---|---|
| Wrong FGSM experiment | `experiments/milestone1/run_fgsm.py:build_cmd` selected `tuning`; upstream `code/revision_experiments.py:exp_tuning` trains AGT SAM/Jacobian configurations | Evaluation-only wrappers load ordinary saved Standard Transformer checkpoints |
| Repeated training and output collisions | `run_clean.py:build_cmd` and `run_pgd.py:build_cmd` both selected `proj`, three datasets and identical `--out`; upstream `exp_proj` trains both models on every invocation | One training unit per variant/seed; separate immutable evaluation directories |
| Validation described as test | Wrapper docstrings claimed test; `exp_proj` consumes `X_val_raw/y_val` | Persist distinct train/validation/test IDs and label the partition in every row |
| Pre-split preprocessing | `code/data_loader.py:load_dataset` fits StandardScaler before `train_test_split` | Fit scaler only on selected training IDs |
| Dataset shape/classes | `data/balanced_CICIoT.csv`, inspected by `research.data:build_data` | Strict 39-feature/27-class checks, sorted mapping, SHA256 |
| No pretrained checkpoints | `git ls-files '*.pt' '*.pth' '*.ckpt'` returned none in PAVE | Train, save, hash and reload each baseline |
| Child-shell activation | `run_all.sh` used bare `python` after setup in a child shell | Scripts invoke `.venv` Python explicitly |
| Portability and unsafe recreation | Setup assumed POSIX venv paths and offered recursive environment deletion | Separate PowerShell/Linux scripts; reuse compatible env or fail; never remove one |
| Pinning not enforced | Original wrappers resolved a script from stale notes/README | Verify exact upstream revision and clean tree before work |
| Swallowed failures | Upstream `revision_experiments.py:main` catches individual exceptions then continues; CSV written only when rows exist | Isolated reproduction wrapper checks complete seed/model rows and finite values |
| Three seeds and partial seeding | Upstream argparse `[1,7,13]`; `train_seq` random initialization/permutations without complete seeding | Main corrected seeds `[1,7,13,29]`, separate split seed 2026, Python/NumPy/torch seeded |
| Wrong smoke dataset | Upstream `main` sets `dsets=['CICIoMT']` with `--quick` | Smoke configuration strictly requires CICIoT |
| No established research completion | Starting tree contained wrappers/notes and a tracked pyc; no results or INT8 implementation | New pipeline; measured results are explicitly smoke/pilot, full study pending |

## Architecture and representation

The authoritative model is `code/models/standard_transformer.py:StandardTransformerIDS`,
imported directly by `research.common:make_model`. Corresponding runner dimensions
come from upstream `revision_experiments.py:build_transformer` and `sota_compare.py:run`:
39 inputs, width 96, four heads, three blocks, FFN 384, dropout 0.1, 27 logits,
50 positions. It embeds features with Linear, adds sinusoidal positions, performs
explicit Q/K/V/output Linear attention, post-residual LayerNorm and ReLU FFNs,
then mean pooling and a 96→48→27 classifier. A separate 96→64→1 sigmoid anomaly
head is computed but is not supervised by upstream ordinary `train_seq`; our CE
baseline also does not train it. It remains part of full-forward benchmarking.

There are 23 Linear modules: one embedding, 12 attention projections, six FFN
layers, two classifier layers and two anomaly layers. Attention is explicit;
there are no packed `MultiheadAttention` projections hiding from conversion.

`data_loader.py:_expand_to_sequence` repeats each single row 50 times and adds
independent Gaussian noise of std 0.01 in standardized space. The upstream
revision evaluation instead uses `sota_compare.py:expand`, repeating a row with
no noise. No observed packet ordering or temporal structure is available. Our
pipeline constructs these synthetic repeated-row sequences only after splitting;
noise is used for training only and regenerated during training. This last detail
differs from upstream's single fixed noisy expansion and is recorded as a deviation.

## Primary-source comparison and reproduction boundary

Primary sources read: pinned PAVE README, model, loader, SOTA/revision runners,
attack code and configuration; publisher-indexed abstract/data-availability text
for [PAVE, DOI 10.1016/j.hcc.2026.100426](https://www.sciencedirect.com/science/article/pii/S2667295226000450).
Publisher HTML and PDF fetches returned HTTP 403/cache errors. The accessible
abstract discusses protocol realizability and does not provide a directly
matching Standard Transformer CICIoT multiclass clean/FGSM/PGD baseline table.
Its data-availability text says checkpoints are included, but the pinned Git
artifact has none. This discrepancy remains unresolved.

Therefore no exact paper configuration/result correspondence is claimed.
`research.reproduce` preserves the original artifact's `exp_proj` behavior in an
isolated copy, with CICIoT, original three seeds, 15 epochs, width 96/3 blocks,
standardized epsilon 0.1, PGD alpha 0.01 and 10 iterations, training-only noise
expansion and noiseless validation evaluation. It preserves pre-split scaler
fitting and incomplete original seeding. Its output is artifact reproduction,
not our corrected held-out study. It is not automatically launched because it
also trains AGT and may be lengthy. The exact paper table mapping remains pending
access to the full paper; this is a specific primary-source limitation.

## Coverage and verification locations

| Requirement | Implementation | Verification / limitation |
|---|---|---|
| Correct baseline, one checkpoint/seed | `research/training.py:train`, `load_checkpoint` | Unit tests + real-data smoke; full four-seed training pending |
| Leak-resistant protocol and metadata | `research/data.py:partitions`, `build_data` | Group disjointness, train scaler values, feature/label/sequence tests; device/capture provenance absent |
| FGSM and multistart PGD | `research/attacks.py:attack` | Fixed weights, preserved modes, finite input gradients, zero epsilon and budget tests, FGSM candidate included in PGD |
| Static PTQ and requested ablations | `research/quantization.py:prepare_static`, `ptq`, `inventory` | Actual qint8 weights and CPU kernels; explicit attention checks; state roundtrip |
| Dynamic quantization distinction | `ptq(mode='dynamic')` | Actual conversion/BPDA/reload tests; optional mode, not default main-study result |
| INT8 gradients | `bpda_model`, `_BPDA` | Exact actual-target forward; dequantized-weight derivative approximation; not exact INT8 white-box |
| AT and AT→PTQ | `training.py:train`, `runner.py:run` | Training-only attacks and validation selection; real-data smoke/pilot |
| Four transfer directions | `evaluation.py:transfer` | Actual target predictions; INT8 source uses actual forward/BPDA gradients |
| QAT | `runner.py:qat_feasibility` | Real converted INT8 deployment pilot; one seed, no full QAT comparison claimed |
| CPU efficiency | `evaluation.py:benchmark` | Batch 1 and 32, fixed threads, warmup/repetitions/raw samples/size; isolated per-model peak memory not measured |
| Safe results/resume | `common.py:unit_status`, `runner.py:run`, `validate.py:validate_run` | Completed units only, incompatible/failed work rejected, artifact completeness checked; no mid-training resume |
| Analysis | `research/analyze.py` | Measured-only CSVs, mean/std, paired deltas, curves, clean confusion matrices and full class-wise attack metrics |
| Local-only | All scripts | No push/PR/upload/CI/publish code; references downloaded read-only |

Exact measured outcomes, runtime estimates and remaining commands are in
`PROGRESS.md` and `docs/VERIFIED_RESULTS.md` after execution.
