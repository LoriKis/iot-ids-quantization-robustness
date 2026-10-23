# Milestone 1: Corrected Paper-Described Adam Protocol

**Frozen for variant C on 2026-10-23.** This is a new protocol for the
corrected experiment. The historical AdamW protocol and results remain
unchanged as variant B.

## Scope and evidence

The supplied paper directly describes Adam with learning rate `0.001`, weight
decay `0.0001`, 25 epochs, and batch size 512. It does not establish AdamW,
decoupled decay, the Table 4 classification mode, or the Table 4 metric
reduction. The local wording check is
`reports/milestone1-paper-described-adam/paper_wording_check.json`; the supplied
PDF hash is recorded there. This experiment is multiclass 27-class by project
requirement and is **not** an exact Table 4 reproduction.

## Fixed data and model procedure

- Dataset: the verified 29,997-row, 39-feature CICIoT2023 CSV, SHA-256
  `0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed`.
- Saved stratified split, seed 2025: 20,997 train, 4,500 validation, 4,500
  test rows. Row IDs are disjoint, but duplicate feature overlap is measured
  and reported; no independent-device generalization claim is made.
- MinMaxScaler is fitted on training rows only. Non-finite raw values are
  fatal; held-out finite values outside `[0,1]` are retained. No training
  feature is constant, and all 27 expected classes occur in training.
- The PAVE Standard Transformer is unchanged. Each row is repeated into a
  length-50 sequence; training adds Gaussian noise with standard deviation
  `0.01`, while evaluation is noiseless. These are inherited assumptions, not
  freshly established paper details.
- Each seed (1, 7, 13, 29) starts from fresh weights. Adam is
  `torch.optim.Adam(lr=0.001, weight_decay=0.0001, betas=(0.9,0.999),
  eps=1e-8, amsgrad=False, maximize=False)`. Decoupled weight decay is
  explicitly disabled when the installed API exposes that option.
- Batch size 512 with `drop_last=False` gives 41 batches of 512 and a final
  batch of 5: 42 steps per epoch, 1,050 steps over 25 epochs, and 524,925
  training sample exposures per seed. Epoch 25 is selected without validation
  or test-based checkpoint selection.

## Attacks and evaluation

FGSM uses epsilon `0.10`. PGD uses epsilon `0.10`, seven steps, alpha `0.025`,
and a deterministic local-generator random start. The paper's Section 5.4
experiment-specific alpha `epsilon/4` is used over the conflicting generic
Section 3.3 formula `epsilon/K = 0.0142857`; this choice was fixed before
execution and was not selected from results.

For epsilon greater than zero, the feasible set is the intersection of the
epsilon L-infinity box around the original normalized input and `[0,1]`:
`lower=max(x-epsilon,0)`, `upper=min(x+epsilon,1)`. Initialization and every
update are projected into that intersection. Infeasible constraints raise an
error; epsilon is never enlarged and clean inputs are never modified. Epsilon
zero returns an exact copy. Clean evaluation accepts finite out-of-range values
without clamping. Labels remain unchanged and the attacks provide no guarantee
of physically or protocol-valid traffic.

All three conditions use the same 4,500 test rows, labels, row order, and
checkpoint. Evaluation runs in `model.eval()` mode and verifies unchanged
weights, finite outputs, bounds, and checkpoint reload consistency.

## Smoke, resume, and environment

New smoke checks select 10 training and 5 validation rows per class from the
saved partitions only, save their IDs and seed, assert disjointness from test,
and never reuse smoke weights or optimizer state. The historical AdamW smoke
used whole-CSV selection; its reconstructed exposure is documented separately
in `reports/milestone1-repair/old_smoke_overlap.json`.

Resume compatibility is checked before any run-directory write using the
configuration, executed source, pinned upstream, dataset, split/scaler, and
environment fingerprints. Epoch checkpoints contain model, optimizer, RNG, and
fingerprint state. Windows uses the pinned `requirements-lock-windows.txt`;
the Mac environment of variant B is historical and not runtime-equivalent.

The completed four-seed run is
`runs/milestone1-paper-described-adam-rerun/`; its compact report and evidence
are in `reports/milestone1-paper-described-adam/`. The earlier
`runs/milestone1-paper-described-adam/` directory is the preserved first
attempt, which stopped after a metadata-only verification failure and is not
reused. A future run should use a new run directory after passing the
fingerprint gate; it must not overwrite either historical directory.

### Training noise and shuffle provenance

Variant C does not materialize one fixed noisy training tensor. For every
epoch, it derives a local CPU generator for the batch permutation and a
separate local CPU generator for each batch's Gaussian noise. Consequently,
fresh noise is generated for every batch and epoch, while replay remains
deterministic under the declared `(seed, epoch, batch)` policy.

The released-code implementation used by A materializes training sequences
once with NumPy Gaussian noise from the seed-derived generator before training;
the resulting noisy tensor is then reused across epochs while the training
loader shuffles batches. This is a source-code observation supported by the
released runner and its saved configuration. B's executed AdamW source and
full run are unavailable, so its historical noise reuse and shuffle behavior
cannot be independently verified from execution evidence; the current source
must not be substituted for that claim.

These differences, along with scaler, split, optimizer, evaluation population,
and attack differences, prevent an optimizer-only interpretation of A/B/C.
