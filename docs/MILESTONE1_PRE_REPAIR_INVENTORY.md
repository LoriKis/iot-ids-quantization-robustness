# Milestone 1 pre-repair inventory

Taken on Lorand's Windows machine before any repair edits, at HEAD
`da89a2ce70c6ddbbe31b8421da8ff7866e51a277` on branch `local/research-pipeline-fixes`
(merge of `7216cef` Milestone 1 cleanup and `0a78d51` paper-described AdamW experiment).
The working tree was clean. This pass only read existing files and wrote this
document and its machine-readable twin,
[milestone1_pre_repair_inventory.json](milestone1_pre_repair_inventory.json)
(every hash below is in that file, plus per-file SHA-256 for all 138 tracked
files, all 177 files of the released-baseline run, and the smaller smoke/pilot runs).

Nothing was committed, pushed, reset, merged again or started as Milestone 2. No
historical run directory, frozen configuration or recorded hash was changed.
Read-only network use: `git ls-remote` only (no fetch). One temporary `pypdf`
install in `%TEMP%` (outside the repository) was used to read the paper.

## 1. What exists and what does not

| Item | Local status | Note |
| --- | --- | --- |
| Released-code baseline run `runs/milestone1-artifact-full/` | PRESENT, 177 files, 60 epoch snapshots, 4 final checkpoints | Every manifest entry re-hashes correctly |
| Released-code baseline report `reports/milestone1-artifact/` | PRESENT (tracked) | Includes compact evidence and `review_metadata.json` |
| Original machine records (`environment/completion/provenance.json`, `docs/environment-windows.json`) | PRESENT, untracked by design | Hashes match `review_metadata.json` |
| Finalized housekeeping (README structure, `NOTICE.md`, `licenses/PAVE-MIT.txt`, `docs/REPOSITORY_REVIEW.md`, `scripts/check_milestone1_review.py`, `.gitignore`/`.gitattributes`, evidence) | PRESENT in HEAD (added in `7216cef`) | Not on `milestone1/main` (remote main is `0a78d51`), so local only |
| Paper-described AdamW compact reports `reports/milestone1-paper-described/` | PRESENT (14 tracked files) | Report only |
| Paper-described AdamW runner/config/protocol/tests | PRESENT, tracked | Runner is NOT the executed source (section 4) |
| **Paper-described AdamW full run `runs/milestone1-paper-described/`** | **MISSING** | Ran on the teammate's Mac |
| **`executor_snapshot.py` for the AdamW run** | **MISSING** | Only snapshot found in the workspace belongs to the released baseline |
| **AdamW smoke run / `smoke_checks`** | **MISSING** | |
| Paper PDF | PRESENT at `../outputs/milestone1-reference-review/evidence/PAVE-source.pdf` | SHA-256 matches recorded `be81c110...6e90` |
| Upstream `upstream/PAVE_artifact` | PRESENT, clean at pinned `e73911bb...fd4` | Dataset SHA-256 matches `0d869e33...53ed` |
| Other runs: `smoke-verified`, `pilot-verified`, `milestone1-artifact-component-smoke(-verified)`, `artifacts/*` | PRESENT | Aggregate digests in the JSON |
| Review ZIP the independent reviewer received | NOT FOUND | The 7 local ZIPs in `../outputs` all predate 2026-10-23 and contain no paper-described files |

## 2. Export omission versus repository omission

- AdamW run directory and `executor_snapshot.py`: not a repository deletion.
  `.gitignore` contains `runs/`, so they were never tracked. They are absent from
  this machine because they were never copied from the Mac.
- Released-baseline run directory: present here (ignored by policy). If the
  reviewed ZIP lacked it, that is an export omission.
- Housekeeping files: tracked in `7216cef` and HEAD. If the reviewed ZIP lacked
  them, that is an export omission. The ZIP was probably built from a tree based
  on `e7c41e6`/`0a78d51`, which predates `7216cef`.
- Original machine-record JSONs and the paper PDF: intentionally excluded from Git;
  both exist locally and their hashes match.
- Remote: `git ls-remote --heads milestone1` gives `main = 0a78d51`; `origin`
  returns "Repository not found". There is no newer finalized material remotely.

## 3. Hash verification outcomes

Released-code baseline (`runs/milestone1-artifact-full`), all OK:

- Recorded results reproduce from the files. Mean accuracy over seeds 1, 7, 13, 29,
  recomputed from run `predictions.csv`, from the Git evidence copies, from
  `per_seed.csv` and from `summary.csv`: Clean **65.808333%**, FGSM **44.416667%**,
  PGD **41.641667%**. All four sources match to six decimals; per-seed run vs CSV
  difference is 5.6e-17.
- `executor_snapshot.py` SHA-256 `962d1beecf7d2b7233716294e81f9331ca9ce6da23fbbf7b834e0deec2747588`
  equals the recorded `executor_sha256` and equals the `research/artifact_baseline.py`
  blob at run revision `3412d41`. The current `artifact_baseline.py` differs from
  it, because `0a78d51` added a CRLF-tolerant `sha_matches` helper.
- Seed manifests, checkpoints and every manifest entry re-hash and match
  `review_metadata.json`; all 48 `files_sha256.json` entries match.
- Run `configuration.json` hash `a243604a...c26ddf` matches `review_metadata.json`.
- Run `reference/code` snapshot is byte-identical to the pinned upstream code
  (19 files). Upstream source hashes in the frozen config match.
- Checkpoints: seed 1 `0fc97092...`, seed 7 `765a2408...`, seed 13 `467f4d5a...`,
  seed 29 `b7e4dae9...` (full values in the JSON).

Paper-described AdamW (only what survives in tracked reports):

- `reports/.../configuration.json` hash `ae3a5a25...e620` equals the recorded
  `configuration_sha256`. The committed config (`7533b2a6...1cc3`) is
  semantically identical and has the same canonical runner `config_hash`
  (`370e691b...ef8ea`), but a different byte layout.
- The saved split `split_ids.json` (20,997/4,500/4,500) is exactly reproduced
  locally with `random_state=2025` (train, validation and test all equal).
- AdamW checkpoint and manifest hashes are only known from `provenance.json`; the
  files themselves are missing, so they cannot be verified here.

Pre-repair checks: `pytest` gives 33 passed. `scripts/check_milestone1_review.py`
FAILS at its README numeric assertion because the merged README contains a UTF-8
BOM and mojibake (see section 6). That is a pre-existing defect, left unchanged.

## 4. Executed-source mismatch (AdamW run)

- Recorded `executor_sha256` in `reports/milestone1-paper-described/environment.json`:
  `619ad2f42eb399c3d65b679d4389e141cd524ad5278da7af04276f9d3d37726f`.
- Current `research/paper_described_baseline.py`
  (identical to blob at `0a78d51`, its only version in Git):
  `1b6893204fd992346a331f5f2d8dd40aad739d5c61d85e9cf967df7a1ea4f44c`.
- No tested normalization reproduces the recorded value: CRLF, LF, no trailing
  newline and UTF-8 BOM variants were tried (see the JSON for each).
- At run start (`environment.json`) the runner was untracked and
  `research/artifact_baseline.py` was modified, so no commit contains the executed
  bytes. The runner was edited between launch and commit, or the snapshot differs
  for another reason.
- **Status: the original executed source is UNAVAILABLE locally.** The current
  runner must not be presented as the executed snapshot. A diff against the
  snapshot (training, data prep, attacks, evaluation or reporting-only?) is
  impossible until the teammate supplies `executor_snapshot.py`.
- `research/milestone1.py`, the only project module the runner imports, was not
  listed as modified at launch and equals its `e7c41e6` blob (`f363aabf...1058`).
- Upstream source snapshot for the AdamW run (`runs/.../reference/`) is also
  missing. The local pinned upstream is verified clean at the recorded revision.
- The AdamW environment was a Mac (arm64, Python 3.11.0, torch 2.14.1, numpy 1.26.4,
  pandas 2.2.3, scikit-learn 1.5.2). This machine uses Windows, Python 3.11.9,
  torch 2.5.1+cpu with the same numpy/pandas/scikit-learn. `requirements.txt` was
  changed in `0a78d51` from `torch==2.5.1+cpu` to an unpinned `torch`.

## 5. Paper wording checked directly

Read from the supplied PDF (SHA-256 verified, pypdf text, symbols partly lost):

- PDF p.13 (printed p.9), Section 5.1: the shared optimiser is **Adam** with
  learning rate 10^-3 and weight decay 10^-4, batch size 512, 25 epochs. PDF p.19
  repeats "Adam at learning rate 10^-3". The paper never says AdamW, decoupled
  decay or L2 in the gradient. AdamW is therefore not established as the paper's
  optimizer, and `torch.optim.Adam(..., weight_decay=1e-4)` is the direct reading.
- PDF p.15: PGD at epsilon 0.10 with seven inner steps (alpha = epsilon/4,
  l-infinity ball, random start); perturbations built in normalised feature space
  and clipped to the feature-space boundaries. PDF p.10: Section 3.3 gives a
  different step-size formula, which is the documented conflict.

Claims to remove later: the config's `optimizer_type_evidence` ("mathematically
equivalent ... AdamW more principled"), README/REPORT wording "AdamW chosen for
correct decoupled decay", and the `C_assumption` AdamW line.

## 6. Other findings for later steps (not fixed here)

- **Smoke partition use.** `run_smoke` takes, per class, the first 10 CSV rows as
  "train" and the next 5 as "val" from the whole CSV. A preliminary reconstruction
  shows 43 of 270 smoke-train IDs and 18 of 135 smoke-val IDs fall in the saved
  final test set (61 distinct test rows). The smoke used them for one optimizer
  step (AdamW), epsilon-zero/bounds/gradient-free checks and checkpoint reload.
  Full-model weights are initialised fresh per seed in `run_seed`, so this is
  exposure of test rows to a smoke check, not shown to be full-model training
  leakage. The exact measurement still has to be repeated in the repair step.
  The smoke artifacts are not on this machine.
- **Domain handling.** Docs and config describe per-feature
  `[min_train, max_train]` clipping; the code clamps to scalar `[0, 1]`. FGSM clamps
  to `[0, 1]` then to `x +/- eps`, PGD does the reverse order for its updates, which
  is inconsistent for clean values outside `[0, 1]` (validation has two such
  values). The PGD random start uses `seed*10000 + batch_idx`, while the config says
  the training seed is used.
- **Resume.** Resume compares only `configuration_hash` and seed. There are no
  executor, upstream, data or environment fingerprints.
- **README.** The merged README has a BOM and mojibake (for example `ΓÇô`, `╬╡`,
  `┬▒`) that breaks the review checker.
- **PROGRESS.md** still says housekeeping changes are uncommitted.
- **Runner-level `read_config`** hard-codes the AdamW frozen values, so a new Adam
  config needs a new reader.

## 7. Files to request from the teammate

Teammate path root:
`/Users/jamshid/Documents/uni/deeplearning/iot-ids-quantization-robustness/runs/milestone1-paper-described/`
Copy into a new local folder (for example `external_evidence/teammate-mac/`), never
into `runs/` over existing data, then verify hashes and record the results.

Critical:

1. `executor_snapshot.py`, expected SHA-256
   `619ad2f42eb399c3d65b679d4389e141cd524ad5278da7af04276f9d3d37726f`.
   If it does not match, report the mismatch; do not substitute.
2. `configuration.json`, expected `ae3a5a25129c3809a1c052fb6061314f1a3a72fde64a48808cb3e9633237e620`.
3. `environment.json` (expected identical to tracked copy, `b0530568...501d3`).
4. `reference/` (upstream source snapshot `reference/code/**` and `reference/LICENSE`).
5. The smoke output: `../milestone1-paper-described-smoke/` and/or `smoke_checks/`
   (`smoke_checks_passed.json`, `smoke_chk.pt`), plus the command/log that ran it.

Per seed (`seed-N/`):

| Seed | `checkpoint.pt` expected SHA-256 | `manifest.json` expected SHA-256 |
| --- | --- | --- |
| 1 | `89a1c80cc46ac8aa5dae46a5e3463705702745c8fc115eb4c6df09cf31008440` | `68bcd851988434f46e2aa0d958e7c610b0e0625c27c5f5567965a791e901a6d4` |
| 7 | `117f10e773536a69c6d8b2987a2f84698402d2d1593a1d618128108bae5e0748` | `ddaa4e16843f826ca9f600d55e846a059f2605747c499f0444a96f760e7ad601` |
| 13 | `c071832301abd80e36e7b743c574d541a58aa605515122b089d966b30d7a8740` | `867c5ba86d66ebe5691c6ed29f1da2f7644902a602c25d1fdf5cbb93efa43bf7` |
| 29 | `9a2cfc0a0ae01002a3f6de46bd5b94a1001929e5cec92d8309fcae13c772a73b` | `693b521969c4f63ab87b7ad88a34e911329fc349a7afb709533333fa5e580d3d` |

Also per seed: `epochs/epoch-01.pt` ... `epoch-25.pt`, `training_complete.json`,
`runtime.json`, `verification.json`, `predictions.csv`, `metrics.csv`, `metrics.json`,
`per_class.csv`, `confusion_matrices.npz`, `evaluations/{Clean,FGSM,PGD}.{json,npz}`,
`raw.log`, `status.json`, `split_ids.json`, `split_overlap_audit.json`,
`preprocessing.json`, `preprocessing.npz`. Their hashes are inside each
`manifest.json` (verify after copying), except epoch files.

Run-level: `split_ids.json`, `split_overlap_audit.json`, `preprocessing.json`,
`preprocessing.npz`, `requested_run_complete.json`, `upstream_auxiliary/` listing.

Environment and clarifications:

- `pip freeze` of the Mac `.venv` (torch 2.14.1 recorded).
- `git diff` and `git status` from launch time, if still available.
- Which ZIP/commit the reviewer received, whether the runner was edited between
  launch and `0a78d51`, and when the smoke test ran relative to the full run.

## 8. Disclosed limits of this inventory

- The AdamW experiment's reported numbers (accuracy 63.96/12.08/6.44%) come only
  from tracked report files; they are not independently verifiable here.
- The paper text was machine-extracted with partial symbol loss; the optimizer and
  PGD sentences are consistent with the config quotes, but a visual check of PDF
  pp.10, 13, 15 is still advisable.
- The smoke-overlap numbers are a preliminary reconstruction from current code
  (the executed snapshot is unavailable); they are not a final measurement.
