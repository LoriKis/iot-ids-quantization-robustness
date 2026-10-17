# Reproducible setup and local commands

All commands run from the repository root. Only native Windows CPU execution
has been tested. Linux instructions are supported by the code but untested here;
the only installed WSL distribution was docker-desktop, so it was not reused as
a research environment. Never share a .venv between Windows and WSL/Linux.

## Actual detected machine

Windows 10.0.26200, PowerShell 7.6.5, Intel i5-10210U (4 physical/8 logical CPUs),
8,331,935,744 RAM bytes, about 56 GB initially free on C:, Intel UHD graphics.
No nvidia-smi or usable CUDA GPU found. Exact runtime metadata is captured in
environment-windows.json and each run snapshot. The Python launcher listed a
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

## Bounded verification and pilot

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

Read the measured planning estimate in VERIFIED_RESULTS.md before executing:

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
research/               corrected implementation, evaluation and analysis
configs/                bounded smoke, pilot and four-seed full protocols
tests/                  scientific invariants and failure behavior
runs/<name>/            snapshots, manifests, checkpoints, statuses and raw results (ignored)
artifacts/<name>/       measured-only CSVs, plots, estimate (ignored)
docs/                   audit, protocol, setup, measured verification
PROGRESS.md              local progress and resume commands
```

The data is already included in PAVE. No enormous raw download is necessary.
setup_upstream.py clones only if absent, detaches the pinned revision for a new
clone and verifies both pin and clean state. If existing upstream differs, it
fails without resetting/stashing/deleting user work. No network access is used
for training or analysis after setup. None of these commands push or publish.
