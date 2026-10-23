# Repository inclusion and review guide

The [Milestone 1 report](../reports/milestone1-artifact/REPORT.md) is authoritative.
This housekeeping pass changes documentation, Git inclusion and compact packaging,
not the training or attacks. It leaves changes uncommitted for review.

## Included in Git

- Source, tests, setup scripts, dependency specifications, configurations and
  [attribution/license material](../NOTICE.md).
- README, setup, the authoritative report, historical documentation with status
  banners, summary/per-seed metrics, comparison table and measured figure.
- [Portable review metadata](../reports/milestone1-artifact/review_metadata.json):
  dataset/source/checkpoint hashes, measured environment and completion, relative
  artifact locations, and hashes of the unchanged original local metadata records.
- [Compact per-seed evidence](../reports/milestone1-artifact/evidence/README.md):
  all 36,000 prediction rows, split IDs, per-class scores, small confusion matrices,
  fitted-scaler metadata and verification/runtime records. No raw feature rows.

CSV, JSON, reports and outputs are not blanket-ignored. The largest newly
included individual files are the roughly 0.74 MB prediction CSVs; split files
are roughly 0.38 MB. These are useful review evidence, not redundant tensor dumps.
The per-seed copies have a `.gitattributes` exception that preserves their original
bytes/line endings, so Git normalization cannot invalidate the saved checksums.
Configuration equality is checked as JSON; historical raw configuration hashes
describe the unchanged run copy, rather than normalized documentation copies.

## Excluded from Git, preserved locally

| Location/pattern | Purpose and recovery |
| --- | --- |
| `.venv/`, `.venv_pave_m1/`, `work/`, caches | Local runtimes, packages, scratch, previous check logs; recreate using setup instructions |
| `upstream/`, `data/` | Pinned PAVE checkout and processed/raw data; obtain and hash-check using [setup](SETUP.md) |
| `runs/`, `artifacts/` | Full experimental evidence and redundant generated analysis; existing files remain intact |
| `*.pt`, `*.pth`, `*.ckpt` | Final/epoch checkpoints and optimizer state; regenerate with the frozen baseline command |
| ZIP/archive exports | Historical review exports and the user-owned `IOT DL.zip`; remain local |
| Secret/key/local-settings patterns | Real credentials, private keys and machine-specific settings; never needed for this public-data experiment |
| Four original machine-record JSON files below | Kept byte-for-byte locally; portable derivative included instead |

The original records removed **only from Git tracking** are
`docs/environment-windows.json` and the report directory's `environment.json`,
`completion.json`, `provenance.json`. Their local contents are unchanged. They
contain machine-specific paths, not required portable entry points. The baseline
report generator may recreate them locally; the ignore rules keep them excluded.

Full Milestone 1 evidence remains at **`runs/milestone1-artifact-full/`**, including
all 60 epoch snapshots, final checkpoints, original scaler/split/prediction files,
per-example clean/attack logits and perturbation records, manifests and raw logs.
Hashes of final checkpoints and original manifests are in portable review
metadata. Earlier smoke/pilot runs remain distinct under `runs/` and `artifacts/`.
Historical review exports remain in the sibling `../outputs/` directory; the
completed package was `milestone1-practical-review-final.zip`. That export predates
this housekeeping pass and is not a snapshot of the current uncommitted tree.

Historical run records, frozen configuration, provenance hashes and exported ZIPs
were not reformatted to remove local paths. Historical investigation/notes may
still quote original paths as evidence. Such paths are not instructions to use
that machine or a prerequisite for a fresh checkout.

## Reproduce and regenerate

Follow [setup](SETUP.md), then use the report's exact commands. A fresh full
baseline run needs the pinned upstream checkout, matching CSV, Python environment
and a **new** run directory. The original run directory supports compatible resume.
Report/figure regeneration reads existing completed checkpoints; it does not train
or rerun attacks, but requires the excluded full evidence and original supplied
paper. The generator currently locates that PDF at
`../outputs/milestone1-reference-review/evidence/PAVE-source.pdf`; its recorded
SHA-256 must match. An arbitrary newer publisher PDF is not assumed identical.

After regenerating a report, run the compact-evidence collection command in setup.
It refreshes derived portable metadata without changing the original run records.
The generator's source hash in the existing portable provenance describes the
**original** report generation; it is not a claim that the later documentation
template is byte-identical. Training executor/config/checkpoint hashes are retained.

## Review checks and requirements

From the repository root, using any compatible Python 3.11+ interpreter:

```powershell
python -B scripts/check_milestone1_review.py
git diff --check
git diff --cached --check
git status --short --untracked-files=all
```

The Python check uses only stdlib: included-file hashes; four expected seeds;
12 metric rows; all 36,000 prediction rows and aligned labels/IDs/checkpoints;
split coverage and disjointness; precision/recall/F1/accuracy/MCC/ASR recomputation;
four-seed means/sample SD; report/README numeric consistency; measured runtime
totals; local Markdown file/image links. It reads saved verification claims,
but cannot independently establish attack correctness or full training from CSVs.

The stronger `research.artifact_baseline --verify-only` command requires the
original full run, pinned upstream and PyTorch (keep the matching dataset available
for the runner's setup). It loads saved optimizer checkpoints, verifies every
manifest payload, and recomputes metrics from saved logits, including recorded
perturbation bounds. It checks saved reload/gradient verification records; it does
not itself rerun the attacks or gradient probes. This full-artifact check was not
rerun in this pass. Pytest includes training/attack fixtures;
it was also not rerun here. Previously passing checks are documented in the report.

## Local inclusion/security audit

The pre-cleanup index held 65 files, approximately 0.40 MB; no tracked generated
file was bulky. All reachable local Git history was inspected (119 distinct blobs
inventoried and eligible text scanned; no blob reached 1 MB). No credential-pattern hits
or large artifact payloads were found. The current candidate files are also scanned
for private-key/token/assigned-secret patterns without printing matched values.
This is a focused pattern audit, not a guarantee against every possible secret.
An isolated preview index and temporary checkout also passed the lightweight
check, demonstrating that the included evidence works without ignored local files
and that Git preserves all 48 per-seed copies byte-for-byte. The review candidate
is approximately 5 MB; its largest file is below 1 MB. The real index was not
used to stage these new files. A [compact check record](../reports/milestone1-artifact/housekeeping_checks.json)
records this pass.

Earlier Git history contains machine-specific path records in nine file paths
(including old reports and historical notes). Removing current tracking does
**not** remove those earlier versions. No history was rewritten. Review that
disclosure before a later GitHub push; this local task does not push or publish.

The group-wide source license remains a team decision; the upstream MIT notice
is now included. Exact matched-condition Table 4 reproduction also remains
unresolved, independently of repository readiness.
