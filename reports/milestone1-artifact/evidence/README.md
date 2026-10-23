# Compact measured evidence

These are byte-for-byte copies from the **completed** local full-budget run,
not newly executed evaluations. Seeds: **1, 7, 13, 29**.

Each `seed-N/` directory contains:

- `metrics.csv`: three conditions; final-checkpoint hash and explicit ASR counts.
- `predictions.csv`: 9,000 condition/row-ID/label/prediction/checkpoint records
  (3,000 underlying examples × three conditions), approximately 0.74 MB.
- `split_ids.json`: all 23,997 train / 6,000 validation assignments and 3,000
  selected evaluation IDs, approximately 0.38 MB; IDs are zero-based CSV row positions.
- `per_class.csv`: 27 classes × three conditions, precision/recall/F1/support.
- `preprocessing.json` and `preprocessing.npz`: feature/class order and small
  fitted-scaler arrays, not raw dataset rows.
- `confusion_matrices.npz`: three small 27×27 count matrices.
- `verification.json`, `training_complete.json`, `runtime.json`: the original
  saved checks, budget/completion and measured times.
- `training_adapter.py`: the instrumented upstream training helper; see the
  [PAVE MIT license](../../../licenses/PAVE-MIT.txt).
- `manifest.json`: the **original full-run** checksums, including excluded files.
  Its checkpoint/logit references do not imply those payloads are in Git.

[files_sha256.json](files_sha256.json) covers the copies included here. It is a
new review manifest; it does not replace the original run manifests.
All four directories total approximately 4.6 MB; their predictions and split
assignments are retained deliberately for independent review.

From the repository root, `python scripts/check_milestone1_review.py` recomputes
metrics, checks row alignment, splits, runtime totals and review-file hashes
without PyTorch, the dataset or checkpoints. Saved verification records report
the checks performed during the experiment; this lightweight check does not
re-execute attack, gradient or checkpoint-reload checks.

To restore these copies from an existing full local run, use
`python scripts/check_milestone1_review.py --collect --run-dir runs/milestone1-artifact-full`.
It verifies original manifest hashes and refuses to overwrite different evidence.
See the [authoritative report](../REPORT.md) and
[repository review guide](../../../docs/REPOSITORY_REVIEW.md).
