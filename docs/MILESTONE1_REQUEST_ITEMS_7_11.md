# Verbatim request items 7-11 (user instruction, supplied for the repair workflow)

## 7. MAKE THE ENVIRONMENT REPRODUCIBLE

requirements.txt currently leaves torch unpinned, and the new run used a
different platform/version from the released baseline.

Record the installed versions and hardware actually used. Preserve the
existing verified Windows dependency specification. Add a separate tested
platform-specific dependency specification for the teammate's environment
where needed.

Do not upgrade dependencies unnecessarily or claim cross-platform equivalence.
Any environment difference must be visible in the comparison report.

## 8. VERIFY AND RUN THE CORRECTED EXPERIMENT

First run the focused tests, relevant regression tests and a training/validation
smoke. Resolve concrete failures before the full experiment.

Then execute the corrected Adam experiment locally:

- Seeds 1, 7, 13 and 29.
- Shared recorded 20,997 / 4,500 / 4,500 split.
- Training-only preprocessing.
- Unchanged PAVE Standard Transformer.
- 25 epochs, batch size 512.
- Final epoch checkpoint.
- Clean, FGSM and PGD evaluation on all 4,500 test rows.
- Same checkpoint, labels and row order across conditions.
- Sequential runs and recoverable epoch checkpoints.

This prompt authorizes these four full runs and their evaluations. Do not
launch hyperparameter searches, additional optimizer variants, quantization,
adversarial training, or later-milestone experiments.

Verify actual optimizer-state step counts, expected sample exposures,
checkpoint reload consistency, row/label alignment, perturbation bounds,
finite outputs and unchanged model parameters during attacks.

Select no hyperparameters or checkpoints using test performance. Do not
repeat runs simply because the results are disappointing.

If work is interrupted, preserve progress and provide an exact compatible
resume command. Report only completed runs as completed.

## 9. PACKAGE REVIEWABLE EVIDENCE

For the existing AdamW run and corrected Adam run, include compact evidence:

- Per-example prediction CSVs for all conditions and seeds.
- Per-class metrics and confusion matrices.
- Split IDs and overlap audit.
- Preprocessing metadata and small fitted-scaler arrays.
- Training-completion, verification and runtime records.
- Original manifests and relevant evaluation metadata.
- Raw training logs of reasonable size.
- Executed source/configuration snapshots and hashes.
- Dataset and checkpoint hashes.

Keep large datasets, checkpoints, environments and large tensor dumps local.

Create or extend a lightweight review checker that independently recomputes
accuracy, macro-F1, weighted-F1, MCC, clean-correct ASR, per-class results,
confusion matrices, means and sample standard deviations from the included
predictions. Check file hashes and alignment across seeds/conditions.

Document which checks use included evidence and which require excluded
checkpoints/full run artifacts. Do not present saved "passed" flags as newly
executed independent checks.

Verify the compact package in isolation from ignored local run directories.

## 10. CORRECT THE REPORTS

Provide clear navigation among:

A. Released-code baseline.
B. Historical AdamW paper-described variant.
C. Corrected paper-described Adam experiment.

Preserve historical evidence and add correction notes rather than silently
rewriting old measured results or protocol history.

Correct these unsupported statements:

- (10a) Lower accuracy does not prove Table 4 used a different classification task.
- (10b) The paper's F1 must not be called "binary attack-positive F1" without
  verified evidence establishing that definition.
- (10c) AdamW is not equivalent to the requested Adam procedure.
- (10d) Duplicate-feature overlap is measured; its origin should not be asserted
  without evidence.
- (10e) Checkpoint reload verification must state its actual sample coverage.

Report published Table 4 values separately as context. Classification mode,
dataset correspondence, F1 definition and ASR denominator remain unresolved
unless new evidence genuinely resolves them.

Explain that the released-code and paper-described configurations differ in
multiple settings and evaluation populations. Their comparison cannot identify
which individual change caused the result. Different scaling also changes the
meaning of epsilon 0.10.

If comparing AdamW with Adam as an optimizer-specific comparison, establish
that all other relevant settings and executed behavior match; otherwise label
it as a broader configuration comparison.

Update setup instructions, dependency notes and commands for the tested
platforms. Preserve attribution and useful historical documentation.

## 11. FINAL HANDOFF

Leave changes uncommitted for independent review. Do not create a ZIP.

Finish with:

- Each review issue and how it was resolved.
- Any missing historical evidence or remaining limitation.
- Completed runs and measured results.
- Tests and independent evidence checks actually executed.
- Measured runtime.
- Locations of the configurations, reports and compact evidence.
- Git status and a concise changed-file summary.

Do not claim exact Table 4 reproduction. Do not claim independent reviewer
approval before the new package has been reviewed.
