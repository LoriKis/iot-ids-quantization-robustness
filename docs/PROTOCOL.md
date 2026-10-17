# Corrected study protocol

This studies feature-space robustness and CPU efficiency on PAVE's balanced
CICIoT2023 subset. It does not establish realizable network evasion or natural
class prevalence. Reproduction and corrected study results must never be pooled.

## Fixed data protocol

Use the pinned 29,997-row, 39-feature, 27-class CSV, each original class having
1,111 rows including Benign_Final. `provenance.json` pins its content hash.
No raw CIC dataset download is needed. Numeric rows must be finite; schema
changes fail. Feature order and alphabetically sorted labels are persisted.

Hash the raw numeric feature row independently of its label. Exact duplicates,
including any conflicting-label groups, remain in one split using shuffled
StratifiedGroupKFold: first outer fold test (about 20%); first five-fold split of
the remaining population validation (about 16%); the rest training (about 64%).
Split seed is 2026 and inner seed 2027, independent of training seeds. Large
duplicate groups prevent exact 64/16/20 proportions: the measured full split
is 21,365 training, 5,363 validation and 3,269 test rows (71.2/17.9/10.9%). There
are 18,044 repeated feature rows and 229 feature groups with conflicting labels;
the latter imply ambiguity for a deterministic feature-only classifier. The
processed CSV has no flow ID, device ID, capture ID, or timestamp column suitable
for provenance grouping. Exact duplicate isolation cannot exclude near-duplicate
or shared-capture leakage; no unseen-device generalization claim is justified.

Smoke and pilot select fixed per-class caps **within** these partitions. Fit
StandardScaler only on the selected training rows, including in smoke. Persist
the scaler parameters, full and selected split IDs, group hashes, label mapping,
feature order, selected/calibration IDs and all content hashes. All variants and
training seeds use the same examples. Gaussian training noise std 0.01 is added
after scaling and splitting. Each sample repeats its own row 50 times; it is
synthetic, not a temporal sequence. Evaluation has no Gaussian noise.

## Training and selection

Train ordinary PAVE StandardTransformerIDS from scratch per seed [1,7,13,29].
Use the unchanged upstream width-96/3-block/4-head/FFN-384 architecture, ordinary
27-class CE, Adam lr 1e-3, no weight decay, constant schedule, gradient clip 1,
15 epochs, batch 32, CPU, one intra-op and one inter-op thread. Upstream uses
batch 512, no clipping, and no checkpoint selection; these are declared deviations
for memory safety and validation selection. Larger effective batch is not implied.
Select baseline checkpoint by maximum validation clean macro-F1, earliest tie.
Clean/FGSM/PGD evaluations all reload that checkpoint. Full runs train four
ordinary baselines and four AT models, not a new model for each attack.

AT starts from the same paired random initialization, not the trained baseline.
Use training rows only, PGD epsilon .1, step .025, three iterations, one random
restart plus the FGSM candidate, and mean 50% clean CE + 50% adversarial CE.
Attack generation uses eval mode and restores train mode; subsequent optimization
uses dropout. Optimizer and epoch budget match ordinary training, while gradient
compute is larger. Select AT checkpoint by mean validation clean and training-
budget PGD macro-F1. Report actual update counts and training seconds; equal
epochs do not mean equal training compute. No early stopping or implicit schedule.

Train/data-loader/augmentation Python, NumPy and torch RNGs are explicitly seeded;
workers=0 and a seeded loader generator avoid platform worker differences.
Float training and attacks use strict deterministic algorithm selection. PyTorch
2.5.1 does not register static quantized_resize_cpu_ as deterministic. Each actual
static Linear island temporarily disables that guard only for its kernel call
and immediately restores the prior policy, including on error. This avoids
per-operator C++ warning I/O contaminating benchmarks. Within-run forward/reload
agreement is checked; cross-platform bitwise reproducibility is not claimed.
CUDA RNGs/deterministic settings are set when available, but this implementation's
verified execution and quantized inference are CPU-only. Do not infer CUDA support.

## Attacks and frozen evaluation settings

Untargeted L-infinity perturbations are measured in training-standardized feature
units. Epsilon .1 means .1 training standard deviations per feature, not 10% raw
traffic change. All 39 numeric coordinates at each of the 50 synthetic positions
are mutable. Both benign and malicious examples are attacked. Independent
timestep perturbations are an additional feature-tensor modeling freedom, not
a physical traffic construction. Domain clipping uses training min/max per feature,
extended per example to include its original value. Intersect it with x±epsilon;
this preserves epsilon-zero identity for out-of-training-range evaluation rows.
Integer, categorical, protocol coupling and packet conservation are not enforced.

FGSM is one CE sign-gradient step from the clean point. PGD uses uniform random
starts in the epsilon ball, projects every iterate, includes the original and
FGSM candidates, and retains per-example successful candidates preferentially,
then highest CE among equally successful candidates, across iterations/restarts.
Weights remain fixed. Gradients use autograd.grad on inputs, without writing
model parameter gradients. Finiteness, bounds and epsilon-zero predictions are
checked. No claim of complete absence of gradient masking follows from these
checks. More iterations/restarts and the retained FGSM candidate provide a
bounded strength diagnostic, not exhaustive security certification.

On a fixed, at-most-four-per-class **validation** subset from seed-1 ordinary
FP32 and all-linear INT8, compare PGD (10 steps,1 restart), (20,1), (20,3), alpha
epsilon/4. Freeze the candidate with lowest mean validation adversarial accuracy;
ties favor more steps/restarts. Record every candidate and selected IDs in
protocol_selection/frozen.json before test access for reporting. These numeric
choices are predeclared; smoke/pilot use smaller candidate grids explicitly.
For the full study run validation first, inspect its diagnostics, then invoke
the test phase against the same compatible run. Test never selects preprocessing,
checkpoint, calibration or attack settings. Re-running test does not tune them.

Static INT8 kernels have no useful autograd. BPDA runs actual converted Linear
islands in forward and approximates their input derivative with the dequantized
weight, treating quantize/dequantize rounding and clipping as identity in backward.
All other operations differentiate normally. Full clean-set forward logits must
agree exactly with the real INT8 target. Attacks are evaluated against the actual
converted model. This is an exact-forward BPDA approximation, **not** exact INT8
white-box evaluation and **not** merely an FP32-copy attack. Large perturbed
out-of-range saturation can weaken this derivative approximation; report that
limitation with the masking diagnostics.

Attack-success denominator is target clean-correct examples; numerator is those
misclassified by the attack. Zero denominator is undefined (null), not fabricated
zero. Robust accuracy counts correct-on-clean AND correct-on-adversarial examples
over **all** evaluated rows. Also report adversarial accuracy, macro-F1, per-class
precision/recall/F1/support and confusion matrices. Transfer reports this target
denominator and the subset correct on both source and target. Matched raw row IDs,
predictions and perturbation norms are saved with every unit.

## Quantization scope and efficiency

Default main path is static PTQ of Linear islands using FBGEMM CPU kernels,
per-channel qint8 weights and calibrated quint8 activations. Each chosen Linear
gets QuantStub→Linear→DeQuantStub. Before observer insertion the wrapped FP32
model must have bitwise-identical logits. Calibration uses only fixed training
rows (up to 16/class in full). It never sees validation/test rows.

| Ablation | Converted modules | Count |
|---|---|---:|
| Classifier-only | classification_head Linear layers | 2 |
| FFN-only | blocks.*.ff.0 and .3 | 6 |
| Attention-only | blocks.*.attn.W_q/W_k/W_v/W_o | 12 |
| All-linear | all above, embedding and anomaly head | 23 |

Matmul attention, softmax, LayerNorm, residuals, positional encoding, pooling and
the interfaces between islands remain floating point. This is a hybrid INT8-
Linear model, not an integer-only Transformer. Every requested and converted
module, weight dtype/qscheme/shape and backend is listed. Missing conversion fails.
Serialized packed states are independently reconstructed and checked for identical
predictions after loading. Optional dynamic quantization is implemented/tested:
weights are prequantized and activations quantized at runtime, with no static
calibration. It is distinctly labeled and is not substituted for static results.

All efficiency comparisons use CPU for both FP32 and actual converted INT8 with
identical sequence length, batches 1/32, one thread and recorded warmup/repetitions.
Measure serialized state bytes, median/p95 latency, throughput, and retain raw
timing samples. Scope is complete upstream forward (including anomaly/attention
outputs and output-finiteness check) with preallocated input. Data preprocessing
and attack generation are excluded; attack generation has separate timing.
Record process RSS but do not treat it as model-specific memory: multiple models
and allocator caches are resident. Isolated per-model peak memory is unmeasured.
CPU thermal throttling/background work and small smoke repetition counts limit
timing precision. No GPU-versus-CPU speedup claim is allowed.

QAT is a focused optional feasibility path after mandatory paths: initialize
from the ordinary checkpoint, add train-mode fake-quant observers to the same
all-linear islands, fine-tune only training data, convert to actual static INT8,
reload/evaluate actual deployment kernels. Its one-seed smoke results establish
conversion feasibility only. A publishable QAT comparison needs a separately
declared four-seed budget and validation protocol; fake quant alone is no result.

## Bounded experiment matrix

| Work | Seeds | Budgets | Scope |
|---|---|---|---|
| FP32, all-linear PTQ, AT-FP32, AT→PTQ | 1,7,13,29 | Clean; FGSM/PGD eps 0,.025,.05,.1,.2 | Full validation/test fixed splits |
| Ordinary classifier/FFN/attention PTQ ablations | same four | Clean; FGSM/PGD eps .1 | Same examples; focused budget |
| FP32→ordinary INT8 | same four | PGD eps .1 | Same examples |
| ordinary INT8→FP32 | same four | Generate with INT8 exact-forward/BPDA PGD eps .1, evaluate FP32 | Same examples |
| AT-FP32→ordinary INT8 | same four | PGD eps .1 | Same examples; target is ordinary INT8 |
| FP32→AT-INT8 | same four | PGD eps .1 | Same examples |
| QAT conversion feasibility | first seed only, optional | Clean/FGSM/PGD eps .1 | Not a main four-seed result |

The ambiguous AT-FP32→INT8 target is explicitly defined as the ordinary PTQ target;
AT→PTQ itself receives direct BPDA evaluation as a main variant. No unspecified
full cross-product of source/target/ablation/training methods is launched.
Per evaluation partition: 176 main metric rows, 36 ablation rows and 16 transfer
rows, 228 total, plus optional feasibility rows. Epsilon-zero is deliberately
reported per attack as a consistency check, not an independent observation.

## Provenance, failures, resume and analysis

Every run snapshots config/ID, project revision/dirty status, source file hashes,
upstream pin/model hash, environment and data manifests. Units have running,
completed or failed status, atomic checkpoints/JSON, raw logs, checkpoint source,
metrics and predictions. Existing directories are never silently reused without
--resume. Resume permits compatible completed units and validates their outputs;
failed or partial units require a new run directory, preserving evidence.
There is no mid-training resume claim: optimizer/scheduler/RNG recovery is not
implemented. Never copy an isolated checkpoint between incompatible runs.

Analysis refuses incomplete runs and separates smoke, pilot and full output roots.
Report per-seed values, mean and sample standard deviation (ddof=1; undefined
for one seed), paired differences, robustness curves, class-wise results,
confusions, ablations, transfers and efficiency CSVs. Four seeds give limited
uncertainty precision; do not present variability as population confidence bounds.
No numeric full-study outcome exists until those runs have actually completed.
