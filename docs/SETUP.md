# Reproducible setup and local commands

Start with the [authoritative Milestone 1 report](../reports/milestone1-artifact/REPORT.md).
The completed baseline uses `research.artifact_baseline`; the corrected-pipeline
smoke/pilot and full-study commands farther below are separate historical/planned
work, not baseline reproduction instructions. No command runs automatically.

All commands run from the repository root. Only native Windows CPU execution
has been tested. Linux instructions are supported by the code but untested here;
the only installed WSL distribution was docker-desktop, so it was not reused as
a research environment. Never share a .venv between Windows and WSL/Linux.

## Actual detected machine

Windows 10.0.26200, PowerShell 7.6.5, Intel i5-10210U (4 physical/8 logical CPUs),
8,331,935,744 RAM bytes, about 56 GB initially free on C:, Intel UHD graphics.
No nvidia-smi or usable CUDA GPU found. Portable completed-baseline metadata is in
[review_metadata.json](../reports/milestone1-artifact/review_metadata.json).
The original local `docs/environment-windows.json` and run snapshots are retained
outside Git. The Python launcher listed a
missing Python 3.11 path; the working global Python 3.9.7 was preserved.

The tested environment uses portable Python 3.11.9 from the official Python NuGet
package, extracted under work/python-3.11.9, with a project .venv. Archive SHA256:
9283876d58c017e0e846f95b490da3bca0fc0a6ee1134b2870677cfb7eec3c67.
No global Python packages, Git configuration or PATH were changed.

## Windows bootstrap

With an existing working Python 3.11:

```powershell
& ./scripts/setup.ps1 -Python 'C:/path/to/Python311/python.exe'
```

Without one, use the project-local portable bootstrap:

```powershell
& ./scripts/portable-python.ps1
```

The scripts never delete/recreate an existing environment. They verify Python
3.11 and install the resolved Windows lock if available, otherwise direct pins.
If PowerShell execution policy blocks scripts, the equivalent commands are:

```powershell
# Once a valid 3.11 interpreter is available:
& 'C:/path/to/Python311/python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir work/pip-cache -r requirements-lock-windows.txt
.\.venv\Scripts\python.exe scripts/setup_upstream.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

PyTorch 2.5.1+cpu is deliberately pinned for the verified torch.ao eager static,
dynamic and QAT backends with explicit Linear islands. Newer torchao APIs are
not assumed compatible. Direct package pins are in requirements.txt; all resolved
versions in requirements-lock-windows.txt. No unrelated environment is upgraded.

## Obtain and verify the released dataset

The setup scripts use `scripts/setup_upstream.py` to obtain the public PAVE
repository only when absent, then verify the pinned revision
`e73911bbdb241f3fb6c5de346139010c4aadefd4` and a clean checkout. They do not
reset an existing checkout. Its processed `data/balanced_CICIoT.csv` is the
required data; do not download the full raw CICIoT dataset for this baseline.
If Git/network policy prevents setup, obtain the public repository by an approved
read-only method, preserve the exact file bytes, and verify the same pin and hash.

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts/setup_upstream.py
$baselineConfig = Get-Content configs/milestone1_artifact_full.json -Raw | ConvertFrom-Json
$datasetHash = (Get-FileHash $baselineConfig.dataset.path -Algorithm SHA256).Hash.ToLowerInvariant()
if ($datasetHash -ne $baselineConfig.dataset.sha256) { throw 'Dataset hash mismatch; stop.' }
```

Expected CSV SHA-256:
`0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed`.
The CSV contains 29,997 rows, 39 features and 27 equally represented classes;
the frozen config records feature order and label mapping. The repository
[attribution notice](../NOTICE.md) distinguishes code licensing from dataset rights.

## Reproduce the completed Milestone 1 artifact baseline

For an independent rerun, choose a **new** directory:

```powershell
.\.venv\Scripts\python.exe -B -m research.artifact_baseline --config configs/milestone1_artifact_full.json --run-dir runs/milestone1-artifact-replication
```

This trains/evaluates four sequential full-budget seeds, not a smoke test. Do not
run it for documentation review. The original completed evidence remains in
`runs/milestone1-artifact-full/`. Its compatible continuation command is:

```powershell
.\.venv\Scripts\python.exe -B -m research.artifact_baseline --run-dir runs/milestone1-artifact-full --resume
```

Reuse the existing environment and source/config snapshots when resuming. Do not
edit saved records or replace data to bypass compatibility checks.

## Check and regenerate reports from completed evidence

Lightweight review requires only the included evidence and Python stdlib:

```powershell
python -B scripts/check_milestone1_review.py
git diff --check
git diff --cached --check
```

To regenerate measured tables/figure/report from the original local run, with
the project environment and excluded evidence available:

```powershell
.\.venv\Scripts\python.exe -B -m research.milestone1_report --run-dir runs/milestone1-artifact-full --output reports/milestone1-artifact
.\.venv\Scripts\python.exe -B scripts/check_milestone1_review.py --collect
```

The report generator validates completed bundles; it does not train or rerun
adversarial evaluations. It also checks the original supplied pre-proof PDF at
`../outputs/milestone1-reference-review/evidence/PAVE-source.pdf`, SHA-256
`be81c110fd12901c8be6455b217ba53ca942c1b0bb9c52bb5dda84bf655b6e90`.
Obtain it through the [paper DOI](https://doi.org/10.1016/j.hcc.2026.100426) or
the original supplied copy; do not redistribute the PDF in Git or assume a
different version has this hash. Current included summaries can be reviewed
without that PDF or any excluded data/checkpoint.

`--collect` copies compact original files after checksum verification and derives
portable metadata. The original machine-specific metadata remains unchanged and
ignored. See [repository review](REPOSITORY_REVIEW.md) for inclusion rules,
historical source-hash context and stronger checks requiring full artifacts.

## Bounded verification and pilot (historical later-study preparation)

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/smoke.json --run-dir runs/smoke-new --phase both
.\.venv\Scripts\python.exe -m research.validate runs/smoke-new --phase both
.\.venv\Scripts\python.exe -m research.analyze runs/smoke-new --phase both --output artifacts/smoke-new
.\.venv\Scripts\python.exe -m research.runner --config configs/pilot.json --run-dir runs/pilot-new --phase both
.\.venv\Scripts\python.exe -m research.analyze runs/pilot-new --phase both --output artifacts/pilot-new
.\.venv\Scripts\python.exe -m research.estimate runs/pilot-new --output artifacts/pilot-new-estimate
```

Smoke: 216 training, 54 validation, 54 test rows, seed 1, one epoch; full upstream
architecture and all mandatory paths plus QAT deployment feasibility. Pilot: 864
training, 108 validation, 108 test rows, seed 1, one epoch, stronger attacks, no QAT.
All counts are class caps after fixed partitioning, never dataset switches.

## Full study (do not launch automatically)

This later-study matrix remains pending. Read the historical planning estimate
in [VERIFIED_RESULTS.md](VERIFIED_RESULTS.md) before any separately authorized execution:

```powershell
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase validation
.\.venv\Scripts\python.exe -m research.runner --config configs/full.json --run-dir runs/full-study --phase test --resume
.\.venv\Scripts\python.exe -m research.validate runs/full-study --phase test
.\.venv\Scripts\python.exe -m research.analyze runs/full-study --phase test --output artifacts/full-study-test
```

Review validation diagnostics and the frozen protocol before the second command.
Any source/config/environment change invalidates completed-unit resume: use a new
run root. Editing documentation alone does not invalidate source compatibility.
Failed/partial units require a new root; no training state is silently recovered.
Completed units are reused only with --resume and artifact validation.

## Linux / WSL (untested)

Use an independent Python 3.11 installation and virtual environment:

```bash
bash scripts/setup.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m research.runner --config configs/smoke.json --run-dir runs/linux-smoke-new --phase both
.venv/bin/python -m research.runner --config configs/full.json --run-dir runs/linux-full --phase validation
.venv/bin/python -m research.runner --config configs/full.json --run-dir runs/linux-full --phase test --resume
.venv/bin/python -m research.analyze runs/linux-full --phase test --output artifacts/linux-full-test
```

FBGEMM requires a compatible x86 CPU/build; unsupported engines fail explicitly.
Linux uses direct requirements pins because the Windows-resolved transitive lock
is not advertised as cross-platform tested.

## Artifact layout and source pin

```text
upstream/PAVE_artifact/   pinned read-only source and processed CSVs (ignored)
.venv/, work/            local runtime, caches, scratch and invocation logs (ignored)
research/               artifact baseline/report tools and separate corrected study
configs/                frozen artifact baseline; separate smoke/pilot/full study
tests/                  scientific invariants and failure behavior
runs/<name>/            snapshots, manifests, checkpoints, statuses and raw results (ignored)
artifacts/<name>/       measured-only CSVs, plots, estimate (ignored)
docs/                   audit, protocol, setup, measured verification
reports/milestone1-artifact/ authoritative report, tables, figure, compact evidence
PROGRESS.md              local progress and resume commands
```

The data is already included in PAVE. No enormous raw download is necessary.
setup_upstream.py clones only if absent, detaches the pinned revision for a new
clone and verifies both pin and clean state. If existing upstream differs, it
fails without resetting/stashing/deleting user work. No network access is used
for training or analysis after setup. None of these commands push or publish.
