# Milestone 1: PAVE reference investigation

Investigation: 2026-10-17 to 2026-10-18, Europe/Budapest. LOCAL ONLY.

## Decision

**The exact multiclass reference experiment is not identified.** The Table 4
numbers are verified, and part of the paper's intended configuration is known.
However, its binary context does not establish the assignment's multiclass
reference, and the available executable artifact materially differs from the
paper. Exact reproduction (A) is blocked; Milestone 1 remains incomplete.

Keep these claims separate:

- **A: verified paper experiment reproduction.** Requires the actual Table 4
  generator/configuration, data/splits and metric/seed/checkpoint provenance.
  The new entry point refuses execution under this claim.
- **B: released artifact behavior.** The existing `research.reproduce` runs the
  released projected-PGD experiment with both Standard Transformer and AGT. The
  new narrow smoke checks only the released Transformer helpers with explicitly
  reduced budgets and added seeding/saving. Neither establishes Table 4.
- **C: reconstruction using paper settings plus assumptions.** Not performed.
  The target JSON records unknowns instead of supplying guessed values.

The corrected research pipeline and its previous results remain separate and
unchanged. No full reference seed or later-milestone experiment was launched.

## Paper examined

Full user-supplied 31-page pre-proof:
`C:/Users/loran/Downloads/1-s2.0-S2667295226000450-main.pdf`.
SHA-256: `be81c110fd12901c8be6455b217ba53ca942c1b0bb9c52bb5dda84bf655b6e90`.
DOI: https://doi.org/10.1016/j.hcc.2026.100426.
Text extracted page by page; Sections 3, 5, 9, 10 and data/code availability
inspected, with visual checks of Table 4 and its surrounding methodology pages.
Printed manuscript page numbers differ from PDF page indices by four for the
main text; later figure pages have inconsistent printed totals.

Table 4, printed p.12 / PDF p.16, CICIoT / **Std.Trans.**, gives:

| Evaluation | Accuracy (%) | F1 (%) | ASR (%) |
|---|---:|---:|---:|
| Clean | 98.68 | 98.63 | not reported |
| FGSM, epsilon 0.10 | 98.19 | 97.95 | 0.81 |
| PGD, epsilon 0.10 | 98.19 | not reported | 0.81 |

All seven candidate values match the supplied PDF. No PGD F1 value is provided
by this table; it must not be invented.

### Classification mode, metrics and aggregation

Section 3.1 (printed p.5 / PDF p.9) explicitly describes a binary attack-versus-
benign setting with C=2. Section 3.6 (p.7 / PDF p.11) defines the confusion matrix
with **attack as positive**, and ordinary F1 from binary precision and recall;
macro-F1 is a separate arithmetic mean over class F1 scores. Sections 8.5 and 9
(pp.16–17 / PDF pp.20–21) discuss multiclass extensions as future work, although
Section 6.1 also says Phase 2 includes binary and multiclass modes. **Binary is
the strongest textual reading of the Phase 1 row, but Table 4 itself has no mode
label or generator provenance. It is not a verified multiclass or mode-aggregated
row.** The assignment asking for multiclass and macro-F1 cannot determine what
this paper row measured.

For F1, the paper's formal definition is binary attack-positive F1, whereas the
released helper's `f1` field is support-weighted F1 and `f1_macro` is separate.
Which field produced Table 4 cannot be established. Calling its 98.63 macro-F1
would be unsupported.

For ASR, Section 3.6 describes misclassification relative to clean predictions,
but gives no exact numerator or explicit conditioning denominator. In released
`pave_audit.py::_asr` (line 38) it is
`100 * count(clean_correct AND adversarial_wrong) / count(clean_correct)`;
in `revision_experiments.py::exp_adaptive` (line 300), `attack_success_pct` is
`100 * count(adversarial_wrong) / N`. The latter includes pre-existing errors.
Neither routine is a verified Table 4 generator. Prediction changes over all N,
new errors over all N, and new errors over clean-correct N are different metrics.
The .81% values and accuracy difference do not identify the denominator.

Phase 1 describes seed-rotated splits but does not explicitly link a seed list
or reduction rule to Table 4. The paper generally claims five seeds, and Section
6.1 explicitly lists **Phase 2** seeds {1,7,13,21,42}. This does not prove those
exact seeds generated Table 4. Equal-weight means of seed metrics, pooling all
predictions, aggregation over modes, rounding order, and any evaluation cap are
unresolved. No per-seed Table 4 files were found.

## Local state and packaging

Project root:
`C:/Users/loran/Documents/Codex/2026-10-17/files-pasted-by-the-user-you/IOT-network-traffic-DL`.
Branch `local/research-pipeline-fixes`; starting HEAD for this investigation
`472be3c36691d286ab0e7c0c9e42ca714c00ed0d`.
Existing environments, prior runs, code and local commits were retained.
No applicable AGENTS.md was found in the project/ancestors or inspection clone.

`configs/`, `docs/`, and `experiments/` **exist locally and are tracked at the
starting HEAD**. There was no reason to reconstruct or recover these directories.
The inspected `IOT DL.zip` had 2,834 entries and omitted all three. This is an
**export omission**, not evidence of missing local implementation. Its SHA-256
was `0c67856baae4868a5fc43c306dd487aa170664bc9747027f1bc00fdc7d5b1085`.
The ZIP was only read; it was no longer present at its project-root path in a
later filesystem check. No tool action in this investigation deleted or moved it.

The existing `../outputs/local-project-source.zip`, 53 entries, does contain all
three directories; SHA-256
`19ee044c7b12401321b2970ec4ffc2c854e68b081df5a8e07cabb35557821701`.
Both inventories were saved under `work/milestone1_reference/zip-inspection.json`.
A new read-only `scripts/check_source_export.py` checks every tracked file,
all three directories and ZIP CRCs. A fresh task handover export is produced
separately; existing exports are not overwritten.

## Artifact investigation and access limits

Pinned checkout `upstream/PAVE_artifact` was not fetched into or edited. It stays
clean at `e73911bbdb241f3fb6c5de346139010c4aadefd4` (v1.0).
A separate inspection clone is at `work/milestone1_reference/public_artifact`.

| Source | Observed outcome | Relevant materials |
|---|---|---|
| https://github.com/robotumbel/PAVE_artifact | Successful separate read-only clone; main and v1.0 resolve to the pinned commit | README, 19 Python files, MIT license, three balanced CSVs |
| Public history, all available refs | Two commits: `75c4bce8a95aa71ca743306a966051f3e18258e6` has only LICENSE; `e73911b...` adds the release files | No older Table 4 generator, splits, checkpoint, configuration or result files |
| https://api.github.com/repos/robotumbel/PAVE_artifact/branches | HTTP 200; one branch, main, pinned SHA | No additional branch with experiment records |
| https://api.github.com/repos/robotumbel/PAVE_artifact/releases | HTTP 200; one v1.0 release, published 2026-08-23; `assets: []` | Automatic source archives only; no checkpoint/split/result attachments |
| https://github.com/robotumbel/PAVE_artifact/releases/tag/v1.0 | Identified by the release API; direct web-tool release-page request had cache miss | API result resolves the release inventory |
| https://anonymous.4open.science/r/PAVE_artifact-9413/ | Web tool unable to open; bounded local HTTP GET returned 200, but only client HTML shell | File inventory could not be verified |
| https://anonymous.4open.science/script/core.3b07189932.min.js | URL observed in that HTML; HTTP 403 Forbidden | Stopped; no repeated equivalent retries or claims about absent anonymous files |

Initial sandboxed HTTP requests failed DNS resolution; the bounded approved
network request succeeded for the HTML and GitHub APIs. This was read-only
network inspection, not a remote write. Access records and responses are saved
under `work/milestone1_reference/`. No raw CICIoT dataset was downloaded.

The README links to CIC's dataset index and maps reviewer revision outputs such
as `rev_proj.csv` to experiments. It has **no explicit Table 4 command or link to
a separate checkpoint/configuration bundle**. The released repository contains
no trained `.pt/.pth/.ckpt` files, original per-seed results, frozen processed
train/validation/test partitions, source-row identifiers or dependency lock.
The three CSVs are processed **subsets**, not stored partition assignments.
The paper's availability statement (printed p.18 / PDF p.22) promises processed
splits and trained checkpoints; that promise is not satisfied by the inspected
public release. Contents of the inaccessible anonymous repository remain unknown.

## Evidence table: paper versus code actually executed

All code locators below refer to pinned upstream `e73911b...`, except the local
wrapper explicitly identified. Runtime paths were traced through function calls;
unrelated `TrainingConfig` defaults do not determine this execution.

| Requirement | Paper evidence | Actual executable code/data | Match or discrepancy | Resolution or unknown |
|---|---|---|---|---|
| Target/mode | Table 4; binary context in 3.1 and 3.6; multiclass extensions discussed in 8.5/9 | `get_data(..., mode="multiclass")`, `sota_compare.py:49`; `exp_proj` supplies no mode override | No verified multiclass Table 4 target | Ask which mode/row is the assignment reference |
| Data and sampling | 5.2, pp.9–10: CICIoT2023, 33 attack categories at source, approximately 60% benign after sampling; actual subset size/recipe not given | `balanced_CICIoT.csv`: 29,997 rows, 39 numeric features, 27 label values, each 1,111; `Benign_Final` = 3.7037%; `get_data` forces full file use | Composition differs substantially; raw source statistics are not subset identifiers | Need original subset/sampling and class mapping; do not substitute this CSV silently |
| Labels/features/cleaning | Numeric source fields; no complete Table 4 schema/mapping/cleaning recipe | `data_loader.py:86–113`: all non-Label columns, numeric coercion, missing/infinite -> 0; alphabetical LabelEncoder for multiclass; benign->0, attack->1 for binary | Released behavior known; paper correspondence unknown | Saved released schema/class counts; author metadata needed |
| Scaling | 5.2, p.10: training-only min–max | `data_loader.py:116–117`: StandardScaler fitted to whole file before splitting | Confirmed discrepancy and evaluation leakage relative to paper | Retain for B; do not silently repair and call it A |
| Partition | 5.1/5.2: stratified 70/15/15, seed rotates with model | `data_loader.py:121–127`: stratified 80/20 train/validation; no test partition | Confirmed discrepancy | Need actual 70/15/15 row IDs and split procedure |
| Evaluation population | Paper says test partition; Table 4 cap not specified | `cap_aligned`, `sota_compare.py:68`, caps validation to 3,000 using its own default seed 42 | Different population; capping unknown for paper | Need original test IDs/cap |
| Sequence construction | 5.3 describes length 10 + Gaussian .01 explicitly for RNN; Transformer length/token provenance not specified | `get_data`/`_expand_to_sequence`: repeat each row 50 times, Gaussian .01 for train/val; `expand` at evaluation removes noise | Exact Transformer input protocol not established | Do not infer chronological sequences or port the RNN description to Transformer |
| Architecture | 5.3/Table 3: 3 layers, 4 heads, width 96, FFN 384, fixed sinusoidal PE, mean pooling, two-layer MLP | `revision_experiments.py::build_transformer`, `sota_compare.py::run` override constructor defaults with those dimensions; max length 50, dropout .1 | Listed core dimensions match; sequence/dropout/head output size not fully linked to Table 4 | Reuse upstream class. It also instantiates an auxiliary anomaly scorer unused by CE; no architectural rewrite |
| Training optimizer | 5.1 pp.9: Adam, lr .001, weight decay .0001 | `sota_compare.py::train_seq`, line 131: `Adam(model.parameters(), 1e-3)`, CE, default weight decay zero | LR/optimizer match; decay differs | `config.py` AdamW/SAM/etc. defaults are not called by this helper |
| Training duration/batch | 5.1: 25 epochs, batch 512 | Local `research.reproduce` and upstream CLI default 15 epochs; called `train_seq` default batch 512 | Epoch discrepancy; batch matches | No paper command/config proving actual values |
| FGSM | 5.4: epsilon .10 and normalized feature-bound clipping | `seq_fgsm`, line 177: CE gradient sign; no domain clipping; not called by local `research.reproduce` | Budget available; clipping/output path differs | Narrow B smoke invokes helper; A needs actual implementation |
| PGD | 5.4 p.11: epsilon .10, 7 steps, alpha=epsilon/4=.025, random start; 3.3 instead states alpha=epsilon/K | `PGD` constant + `seq_pgd`, line 188: .10/10 steps/.01; starts from clean tensor; projects only to x±epsilon, no domain clipping | Confirmed difference; internal paper alpha inconsistency also exists | Need generating script/config; do not silently choose an implementation |
| F1 | 3.6 p.7 defines binary attack-positive F1, macro separately; Table 4 says F1 | `metrics`, line 206: `f1` weighted; `f1_macro` separate; values rounded to 4 decimals | Averaging for Table 4 unverified | Need Table 4 metric column and reduction rule |
| ASR | 3.6 verbal definition relative to clean predictions | `_asr` uses clean-correct denominator; `exp_adaptive` uses all-row adversarial error; `exp_proj` emits no ASR | Multiple non-equivalent definitions; no Table 4 trace | Need exact numerator, denominator and evaluated subset |
| Seeds/aggregation | General five-seed claims; Phase 2 lists 1,7,13,21,42; Phase 1 rotates splits | Wrapper/CLI defaults 1,7,13; split/noise use seed; training helper does not seed torch; `sota_compare::build_table` averages macro-F1 over seeds for a different table | No proven Phase 1/Table 4 seed protocol or aggregation | Need actual per-seed rows, RNG handling and aggregation script |
| Checkpoint selection | 25 epochs stated; no final-vs-best validation rule found | `train_seq` returns final-epoch model; no checkpoint saving/selection in this path | Release rule known; Table 4 rule unknown | Need supplied checkpoint(s) and selection criterion |
| Runner output | Table 4 clean/FGSM/PGD accuracy/F1/ASR | `research.reproduce -> revision_experiments::exp_proj`: Standard Transformer AND AGT, clean/unconstrained-PGD/projected-PGD MCC and validity | Not Table 4 experiment | Keep B wrapper; dedicated A execution remains gated |

### Called functions, rather than configuration comments

`research.reproduce.main` builds `revision_experiments.py --exp proj --datasets
CICIoT --seeds ... --epochs ...` in an isolated copy. `exp_proj` calls
`sota_compare.get_data -> data_loader.load_dataset`, constructs the 96/4/3/384
Transformer, and calls `sota_compare.train_seq`. It then uses `cap_aligned`,
noise-free `expand`, `seq_logits`, `seq_pgd`, `seq_pgd_projected`, and `metrics`.
The loop subsequently trains AGT. No Table 4 F1/ASR comparison or FGSM call is
part of that execution. `config.py::TrainingConfig` does not override the
optimizer or batch settings inside `train_seq`.

`sota_compare.run` has clean/FGSM/PGD evaluation for its Transformer, but also
trains a surrogate and GRU and evaluates other attack families; its table is a
macro-F1/retention summary, not Table 4. Calling that whole runner is unnecessary
for the active narrow task and would still not verify A.

## Preparation and checks actually executed

Added `configs/milestone1_target.json` and `research/milestone1.py`. The paper
manifest contains verified observations and explicit unresolved entries. Route A
refuses training before importing ML libraries or creating a run directory;
editing its status alone cannot bypass this implementation gate. Full-budget B
training is also disabled in the new entry point. The old B wrapper is preserved.

The component smoke imports a separate copy of unchanged upstream code, redirects
its import-time `AGT_OUT` directory and points `BAL` at the pinned CSV. These are
path compatibility measures, not scientific fixes. It reuses `get_data`,
`cap_aligned`, `expand`, `StandardTransformerIDS`, `train_seq`, `seq_logits`,
`seq_fgsm`, `seq_pgd`, and `metrics`. No upstream algorithm source was patched.

The smoke explicitly changes the budget to seed 1, one epoch, 54 training rows
(two per class), 27 validation rows (one per class), batch 18/attack batch 9,
CPU/two threads, and explicit torch/numpy seeding. Removing surrogate training
also changes upstream RNG consumption. These deviations are saved in its config;
this is a B **component check**, not a full B experiment or A reproduction.

Commands executed from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m research.milestone1 --route paper --dry-run
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_milestone1_gate.py --junitxml=work/milestone1_reference/gate-tests.xml
.\.venv\Scripts\python.exe -B -m research.milestone1 --route artifact --smoke --run-dir runs/milestone1-artifact-component-smoke-verified
.\.venv\Scripts\python.exe -B -m research.milestone1 --verify runs/milestone1-artifact-component-smoke-verified
.\.venv\Scripts\python.exe -B -m research.milestone1 --route paper --smoke --run-dir runs/milestone1-paper-must-not-run
```

- Four gate/completeness tests passed (0.20 s reported by pytest).
- Help/import/parse checks passed. Paper execution was rejected with exit 2;
  `milestone1-paper-must-not-run` was never created.
- Final smoke completed. One earlier development smoke is preserved in
  `runs/milestone1-artifact-component-smoke`; it was repeated after adding
  persisted-scaler reload and bundle semantic checks. Both use only 54/27 rows,
  one epoch, one seed. There was no full-budget or multi-seed experiment.
- The actual full-file StandardScaler and 80/20 upstream split were reproduced
  and checked against reconstructed CSV row identifiers before selecting smoke
  rows. Training/validation contain 23,997/6,000 rows; smoke uses the small caps.
- Persisted scaler parameters reconstruct the saved clean inputs exactly.
  Saved checkpoint reload gives identical logits. All three evaluations load/use
  that same checkpoint; measured perturbations satisfy the epsilon bound.
- Each smoke emits config, feature/class metadata, complete split IDs, selected
  row IDs, scaler parameters, checkpoint, inputs, labels, predictions/logits,
  metrics, timing, provenance/source hashes and logs. Checksums, partition coverage,
  evaluation row coverage and shared checkpoint identity pass validation.
- Smoke accuracy is 7.41% for clean/FGSM/PGD. This undertrained small-sample result
  has no scientific comparison value; it is not an attempted 98.68% reproduction.
  Weighted and macro-F1 happen to coincide on the balanced one-row-per-class
  evaluation and therefore cannot resolve the paper's F1 definition.
- New smoke metrics separately name clean-correct-conditioned ASR, prediction
  changes over all rows, and all-row error. None is labelled verified paper ASR.

### Runtime and launch readiness

Final smoke measured 8.45 s for three training batches (54 rows), 0.06 s clean,
0.18 s FGSM and 1.06 s PGD (27 rows); total after imports 16.51 s, data step 3.98 s.
Environment: Python 3.11.9, torch 2.5.1 CPU, numpy 1.26.4, pandas 2.2.3,
scikit-learn 1.5.2; i5-10210U, 8 GB RAM, no usable CUDA.

**No defensible full reference-seed runtime estimate or exact launch command is
available.** Its sample count, sequence construction, mode and checkpoint
protocol remain unresolved. Extrapolating small batches to batch 512 would add
unmeasured memory and throughput effects. The earlier 200.84-hour full-study
estimate belongs to the corrected multi-variant research protocol, not this
reference experiment, and must not be reused here.

The exact A command is deliberately `null` in the target manifest. Once the
missing materials identify it, prepare and validate a Standard-Transformer-only
entry point using that evidence, then run one full reference seed. Do not run the
old `--exp all`, substitute the balanced 27-class data, or choose assumptions just
to obtain a plausible accuracy. The clarification draft is in
`docs/MILESTONE1_CLARIFICATION.md`; it has not been sent.
