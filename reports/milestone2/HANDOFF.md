# Milestone 2 handoff (pre-commit cleanup, for human review)

**Publication note:** The 312-file byte-for-byte comparison reported below was performed before privacy edits. Seven public copies were subsequently changed: the Python executable path was redacted in `pilot_runtime.json`, the default V4 path was made relative in the overlay test fixture (`tests/conftest.py`), and personal paths were replaced in five Markdown documents. The original VM copies and historical hash manifests were left unchanged. These edits did not change predictions, labels, or reported measurements.

Status: **UNCOMMITTED, UNSTAGED, UNPUSHED** (branch `m2-review`, start commit `466faf724aaab5507d5cf12f74281c5f2a62ae28`). Nothing was trained, run or benchmarked. The two VM folders were only read and hashed. The local `.agents/` folder stays local and is not part of the commit.

## What to stage

Run from the repository root. This stages the modified `README.md` and the new files, and nothing else:

```powershell
git add -- README.md reports/milestone2 verification
```

Do **not** stage `.agents/`. No file needed `git add -f`.

## Checks run in this cleanup (each PASS or FAIL)

POST-RUN checks. The temporary index and the temporary checkouts were created outside the repository and removed afterwards; the real index was never touched. Commands and raw output: [`cleanup_verification_postrun.md`](../../verification/results/cleanup_verification_postrun.md).

| # | Check | Result |
|---|---|---|
| 1 | **README block removal.** Removing the marked block (`<!-- M2-NAV-BEGIN -->` through the line ending after `<!-- M2-NAV-END -->`) gives bytes identical to `README.md` at the start commit (SHA-256 `de9628ef...f7ad8`, 11,035 bytes; LF, no BOM). The extra blank line is removed. | PASS |
| 2 | **No other tracked file changed.** `git diff --name-status HEAD` lists only `M README.md` (12 lines added, 0 deleted). With the temporary index: 1 `M`, 337 `A`, 0 deleted, 0 renamed. | PASS |
| 3 | Real index empty, `HEAD` still the start commit, status shows only `README.md` modified plus untracked paths. | PASS |
| 4 | Clean temporary checkout from the temporary index: `.agents/`, datasets, `*.pt`/`*.pth`/`*.ckpt`, virtual environments, `__pycache__`, `.mplconfig` are absent. | PASS |
| 5 | Checkout contains the checker inputs, both original plots, both recomputed plots and the source trees (21 required paths; 123 source files, 188 evidence files). | PASS |
| 6 | Portable checker run from the checkout: `verify_m2.py` 174 PASS, 0 FAIL, 25 NOTE; its output is byte-identical to the saved output. | PASS |
| 7 | Link checker run from the checkout (includes this file): 0 problems. | PASS |
| 8 | SHA-256 of every publication file, checkout vs working copy, with default settings and with `core.autocrlf=true`: 0 mismatches. The first run found 5 mismatches in `verification/results/` (CRLF files rewritten to LF by the root `.gitattributes` rule `text=auto eol=lf`). Cause fixed by adding `verification/.gitattributes` (`* -text`); no historical manifest was changed. | PASS |
| 9 | The 312 copied evidence and source files, checkout vs the VM originals: 0 mismatches (default and `autocrlf=true`). `reports/milestone2/.gitattributes` (`* -text`) is in the checkout and effective. | PASS |
| 10 | Checkpoint hashes recorded in the stage markers and `ckpt/latest.json`, recomputed directly on the VM files without loading a model ([`vm_hash_check_postrun.tsv`](../../verification/results/vm_hash_check_postrun.tsv)): 59 of 59 match. | PASS |
| 11 | Campaign logs read (section "What the logs show"). | PASS (read) |
| 12 | No publication Markdown links into `.agents/`; no personal absolute path in the active docs. | PASS |

## What changed in this cleanup

- Root `README.md`: removed the extra blank line after the navigation block.
- `reports/milestone2/README.md`, `REPORT.md`, `CHECKLIST.md`, `verification/results/discrepancies.md`: now use the same labels (**recomputed from predictions**, **confirmed from a config or log**, **reported but not independently verified**) and keep apart: the four measured models vs the QAT feasibility pilot (full QAT deferred; no comparable QAT accuracy result and no measured runtime estimate; the pilot only shows that one epoch ran, converted to INT8 and serialized); 3,764 test rows vs the same 270 selected rows; one training seed vs three efficiency repetitions; recomputed metrics vs settings supported only by a record; exact VM source vs post-run adaptation.
- Restart behaviour, checkpoint hashes and the scaler/calibration statements were updated from the VM copies (below). Nothing was removed from the unverified list unless it was actually resolved.
- `verification/copy_check.ps1` is **not published** (see decision below); `check_links.py` now also checks this file; `verification/.gitattributes` added; new `results/vm_hash_check_postrun.tsv` and `results/cleanup_verification_postrun.md`.
- Unchanged: `original/`, `evidence/` and `source/` copies (byte-identical to the VM files), the hash tables, all existing M1 files.

## What the logs show (restart behaviour)

The first launch (12:31:49Z) ran preflight and freeze and stopped on request. The resume at 12:32:00Z found freeze valid and ran nothing, so it is not evidence of recovery. A second resume at 12:37:16Z ran training, PTQ, validation, test, robustness, efficiency and report once, in order, each stage with one start line and one marker, and no error or interruption. Training was never interrupted according to the logs, so recovery from an interrupted run was not exercised. An independent scaler refit and a calibration reconstruction were **NOT performed**; those statements rest on configs and hashes.

## Decisions about `.agents/` and `verification/`

- `.agents/` stays local. Its two findings that matter are summarized here and in the reports: (1) files that conflict with the repository were left unchanged and exist only as M2 copies under `reports/milestone2/source/`: `research/attacks.py` and `research/common.py` (M2-extended), `.gitattributes`, `pytest.ini`, `tests/conftest.py` (V4 and overlay differ from each other and from the root). The root `.gitattributes` has `* text=auto eol=lf`, so a nested `.gitattributes` (`* -text`) keeps copied bytes. Do not replace the M1 root files with these. (2) V4 copies of M1 files that differ from the submitted M1 versions (mostly line endings: `configs/milestone1_artifact_full.json`, M1 docs, `reports/milestone1-*` data files, `research/paper_described_adam_report.py`, `scripts/check_milestone1_adam_review.py`) were not copied. The root `.gitignore` ignores `*.log`, `*.pt`, `artifacts/`, `data/`, so logs are stored as `*.log.txt` and `artifacts/` folders were renamed (`m2a-pilot/`, `provenance/m2b-integrity/`).
- Kept in `verification/`: `verify_m2.py`, `check_links.py` (run from any checkout, read files only), the saved outputs, `discrepancies.md`, `figure_review.md`, the recomputed figures. `copy_check.txt`, `copy_map.tsv` and `vm_hash_check_postrun.tsv` are kept as labelled historical records.
- Excluded: `copy_check.ps1`. It needs both VM folders and the VM file manifests that live only in `.agents/`, so a teammate cannot run it. The same byte comparison was repeated in this cleanup (check 9) and the SHA-256 columns in `evidence/README.md` and `source/README.md` remain the reference.

## Remaining issues that affect what can be concluded

- One training seed per parent, no confidence intervals. PTQ differences (-0.19 pp, +0.08 pp on 3,764 rows) are not shown to be meaningful.
- Robustness uses 270 rows and feature-space FGSM and 5-step PGD only. INT8 attacks use an approximate gradient and may be too weak.
- Efficiency was measured on one machine at one thread; its CPU model is not recorded. The INT8 batch-256 timings vary widely.
- The official Milestone 2 assignment text is not in the repository or the VM folders; [`CHECKLIST.md`](CHECKLIST.md) is based on the review request and the protocol.
- QAT was deferred. There is no QAT accuracy result and no runtime estimate.
- The reorganized layout was not tested end to end. Only the two checkers were run on a clean checkout. The dataset is not in the repository.
- The personal Windows path was removed from the public copies. The original hashes predate the seven edits described in the publication note. Some historical records still contain Linux VM paths.
- `evidence/deadline-v1/efficiency/fixed_rows.npy` contains the 256 fixed validation rows used for efficiency measurements.
- The root README still contains Milestone 1-era wording; the navigation block explains this.
