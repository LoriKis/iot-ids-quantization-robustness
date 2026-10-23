# Milestone 1 comparison: published context and local experiments

This comparison covers the published PAVE Table 4 context and the two main
implemented local configurations. Experiment B is retained as a historical
AdamW variant, not treated as a third independently verified configuration.
No result below establishes an exact Table 4 reproduction.

## Main results

Our means and sample standard deviations are computed from the saved four-seed
metrics (`reports/milestone1-artifact/per_seed.csv` and
`reports/milestone1-paper-described-adam/per_seed.csv`), using percentage
points for accuracy and macro-F1. Experiment A evaluates a capped 3,000-row
validation population; Experiment C evaluates the shared 4,500-row test
partition.

| Condition | Published Table 4 accuracy (%) | Experiment A accuracy (%) | Experiment A macro-F1 (%) | Experiment C accuracy (%) | Experiment C macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| Clean | 98.68 | 65.808 ± 0.428 | 61.140 ± 0.423 | 63.644 ± 0.940 | 58.305 ± 1.016 |
| FGSM, ε = 0.10 | 98.19 | 44.417 ± 2.119 | 42.850 ± 2.145 | 15.422 ± 4.278 | 12.919 ± 3.981 |
| PGD, ε = 0.10 | 98.19 | 41.642 ± 2.967 | 40.372 ± 2.755 | 7.922 ± 1.660 | 6.524 ± 1.092 |

The published F1 values are deliberately separate from our macro-F1 values:

| Condition | Published Table 4 F1 (%) | Status of published F1 definition |
|---|---:|---|
| Clean | 98.63 | Averaging and classification mode unverified |
| FGSM, ε = 0.10 | 97.95 | Averaging and classification mode unverified |
| PGD, ε = 0.10 | Not reported | Not applicable |

The published table's F1 is not presented here as macro-F1 or as verified
binary attack-positive F1. Its ASR values (0.81 for FGSM and PGD) also have an
unverified denominator and are not included in the main local-results table.

## Historical Experiment B: AdamW

Experiment B is the teammate's historical paper-described AdamW variant:

| Condition | Accuracy (%) | Macro-F1 (%) | ASR (%) |
|---|---:|---:|---:|
| Clean | 63.956 ± 1.455 | 58.758 ± 1.734 | 0.000 ± 0.000 |
| FGSM | 12.083 ± 3.040 | 10.801 ± 2.680 | 84.009 ± 4.662 |
| PGD | 6.439 ± 1.758 | 6.241 ± 1.518 | 91.269 ± 2.867 |

These figures are preserved as historical context in
`reports/milestone1-paper-described/`. The teammate's Mac run, byte-exact
executed `executor_snapshot.py`, checkpoints, verification records, and full
smoke artifacts are unavailable locally. The current runner is not substituted
for that missing historical source, and Experiment B is not called independently
verified here.

## Settings comparison

| Setting | Published/Table 4 evidence | Experiment A: released code | Experiment B: historical AdamW | Experiment C: corrected Adam |
|---|---|---|---|---|
| Classification task | Mode unresolved; binary context appears elsewhere in the paper | 27-class multiclass | 27-class multiclass | 27-class multiclass by project requirement |
| Data | Exact generating subset unresolved | Released balanced CSV, 29,997 rows | Same verified CSV and saved split | Same verified CSV, 29,997 rows |
| Preprocessing/scaler fitting | Paper describes normalized inputs; exact pipeline correspondence unresolved | StandardScaler fit on all rows before split | MinMaxScaler fit on training partition only | MinMaxScaler fit on training partition only |
| Split | Exact data and population unresolved | Seed-dependent stratified 80/20; 23,997 train / 6,000 validation, then cap 3,000 for evaluation | Shared stratified 70/15/15 split, 20,997 / 4,500 / 4,500 | Same shared split, seed 2025 |
| Evaluation population | Unresolved | 3,000 capped validation rows per seed | Full 4,500-row test partition | Full 4,500-row test partition |
| Optimizer | Paper names Adam with weight decay 0.0001 | Adam, zero weight decay | AdamW, lr 0.001, weight decay 0.0001 | Conventional `torch.optim.Adam`, lr 0.001, weight decay 0.0001 |
| Epochs / batch | Paper describes 25 / 512 | 15 / 512 | 25 / 512 | 25 / 512 |
| Sequence construction | Transformer format not fully established | Repeat each row to length 50; training noise σ=0.01 | Same inherited construction | Same inherited construction |
| Training noise | Unresolved | Fixed generated training tensor per seed where source/evidence support it | Historical execution cannot be independently confirmed; source snapshot unavailable | Fresh Gaussian noise per batch and epoch from derived local generators |
| Checkpoint selection | Unresolved | Final epoch, no test selection | Reported final epoch; historical reload evidence unavailable locally | Final epoch 25, no validation/test selection |
| FGSM | Paper ε=0.10 context | ε=0.10 in StandardScaler space; upstream implementation | ε=0.10 in MinMaxScaler space, random/domain policy as historical report | ε=0.10 in normalized sequence space with explicit feasible-set projection |
| PGD | Paper reports ε=0.10; step-size descriptions conflict | 10 steps, α=0.01, clean start | 7 steps, α=0.025, random start | 7 steps, α=0.025, random start |
| Environment | Not established for Table 4 | Local Windows CPU environment | Historical Mac/arm64 environment, not runtime-equivalent | Windows Python 3.11.9, PyTorch 2.5.1+cpu, CPU, two threads |

The paper statements, inherited implementation choices, and assumptions are
not interchangeable. In particular, Experiment C's Adam interpretation is a
direct reading of “Adam with weight decay”; AdamW is a different procedure.
The exact training-noise execution of B cannot be promoted from source
assumption to historical fact because its executed source is missing.

## Why the results do not isolate one cause

Experiment A and Experiment C change multiple factors at once: scaler and
preprocessing fit scope, split/evaluation population, epoch budget, optimizer
and weight decay, sequence-noise generation, attack implementation, and
checkpoint/evaluation instrumentation. They are therefore a practical
configuration comparison, not an optimizer-only ablation. The accuracy gap
from either local experiment to Table 4 cannot be assigned causally to
classification mode, duplicates, optimizer, scaling, data subset, or any
other single factor. Lower local accuracy does **not** prove that Table 4 used
a different classification task.

Epsilon `0.10` also has different meaning under the two local scalers. In
StandardScaler space it is a fraction of each feature's training standard
deviation; in MinMaxScaler space it is up to 10% of each feature's training
range, subject to the declared normalized-domain policy. These perturbations
are not directly comparable physical budgets.

## Data and validity limitations

The saved overlap audit reports that **2,804 of 4,500 test rows (62.31%)**
have an identical feature pattern in the training partition. Row IDs are
disjoint, but row-disjoint partitions therefore do not establish independence
from duplicate feature patterns. The historical smoke audit also found 61
distinct rows overlapping the saved test partition. The test set must not be
described as completely untouched throughout development.

All attacks operate in continuous normalized feature space. They are not
guaranteed to produce protocol-valid network traffic, preserve flag
combinations, or maintain flow/timing semantics. The repeated length-50
sequences are synthetic representations, not chronological flow traces.

Milestone 1 is complete as a practical released-code baseline plus a
paper-described comparison study. Exact matched-condition reproduction of
Table 4 remains unresolved. This package does not claim that every official
Milestone 1 requirement was fully met, and it does not claim external approval
of the whole project.
