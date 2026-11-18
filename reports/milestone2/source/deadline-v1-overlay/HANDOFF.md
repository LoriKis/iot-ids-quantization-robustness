# Milestone 2 deadline experiment (m2-deadline-v1): handoff

This folder is an add-on to the verified V4 package already on the Ubuntu VM. It adds a timing command, a launch
command, a resume command and a report command. It changes nothing inside V4.

Nothing was trained, benchmarked, attacked or evaluated on the laptop. The laptop only ran synthetic checks of the
budget, sampling, protocol, marker, report and manifest logic. Every number in the final report comes from the VM.

What the experiment does: one seed (1), the verified dataset, PAVE Standard Transformer, frozen split_v2. Two FP32
parents (ordinary and adversarially trained) with the same rows, epochs, optimizer, batch size and final-epoch
checkpoint. An INT8 post-training-quantization (PTQ) copy of each, calibrated on training rows only. That gives four
models. QAT is feasibility only: the report quotes what the existing tiny QAT test demonstrated and claims no QAT
performance result.

Placeholders you must replace: `<VM_USER>`, `<VM_HOST>`, `<HOURS_REMAINING>`, `<HOURS_REMAINING_NOW>`.
The commands assume V4 is at `~/IOT-network-traffic-DL-M2-VM-v4` on the VM. If it is elsewhere, run
`export V4_ROOT=/path/to/IOT-network-traffic-DL-M2-VM-v4` first in every VM shell.

## 1. Pre-transfer check

**Windows PowerShell (laptop)**

```powershell
$O  = '<WORKSPACE>\IOT-network-traffic-DL\IOT-network-traffic-DL-M2-VM-deadline-v1'
$V4 = '<WORKSPACE>\IOT-network-traffic-DL-M2-VM-v4'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = $O
Set-Location $O
python -m m2_deadline.verify_overlay --overlay $O --v4-root $V4
```

Expected: the last line is `OK`. Any `FAIL` line names the file. Do not transfer until it prints `OK`.

## 2. Transfer (no archive)

**Windows PowerShell (laptop)**

```powershell
scp -r "$O" <VM_USER>@<VM_HOST>:~/
```

V4 and its `.venv` are already on the VM, so nothing else is copied. This creates
`~/IOT-network-traffic-DL-M2-VM-deadline-v1` on the VM.

## 3. Verify on the VM

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
bash scripts/vm_deadline/verify.sh
```

It uses the system `python3`, checks every overlay file against `OVERLAY_MANIFEST.sha256`, checks that V4's
`MANIFEST.sha256` is the one this overlay expects, and re-checks V4's files. The last line must be `OK`.
A `WARN ... unlisted files` line means V4 contains extra files (for example a `__pycache__`). Delete them before
using V4's own stage scripts. Exit 3 means a real mismatch: stop and do not edit anything.

## 4. Record of V4's Stage 04 smoke run, and a contention check

Status as reported from the VM, kept as is: V4 passed all 478 regression tests. Its Stage 04 smoke run did **not**
pass: it took 48 minutes against a 45-minute limit, the restart check inside it was skipped, and the test partition
was not loaded. No pass marker exists for it and none is created here. The old E4 sweep is not rerun. This overlay
never reads files from that smoke run, and every path containing `smoke` is refused.

Check that nothing from that run is still using the CPU (the timing command measures CPU speed and exits 5 on
contention).

**Ubuntu VM**

```bash
pgrep -af research.m2b_campaign
ls -dt ~/IOT-network-traffic-DL-M2-VM-v4/output/*/ | head
```

If `pgrep` prints a process, wait until it ends by itself. Do not kill it. The new outputs go to
`artifacts/m2-deadline-v1` inside this overlay.
## 5. Timing estimate (measures and stops; starts no training)

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
bash scripts/vm_deadline/timing.sh <HOURS_REMAINING>
```

`<HOURS_REMAINING>` is the real number of hours left until the 12-hour deadline right now (for example `11.5`).
The probe uses training and validation rows only. It never loads test data and records no accuracy or F1.
It prints a table and the launch command, and writes
`artifacts/m2-deadline-v1/timing/timing_estimate.json` and, when feasible, `timing/budget_lock.json`.

The lock holds the epoch count (25, 10 or 5, largest that fits) and the robustness subset size (10 or 5 rows per
class). It is chosen from runtime alone, before any performance number exists.

Exit codes of this command: 0 feasible and locked; 4 even the smallest plan does not fit (nothing is locked and
nothing should be launched; report that to your instructor or cut scope); 5 CPU contention or noisy timings (stop
other work and rerun, or add `--allow-contention` to accept an over-estimate); 2 bad input or the probe ran over its
time limit; 3 an estimate or lock already exists (see section 12).

## 6. Launch (separate command; hours remaining is mandatory)

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
nohup bash scripts/vm_deadline/launch.sh <HOURS_REMAINING> > artifacts/m2-deadline-v1/logs/launch.nohup.out 2>&1 &
```

Or inside tmux:

```bash
tmux new -s m2
bash scripts/vm_deadline/launch.sh <HOURS_REMAINING>
```

Use the hours left at the moment you launch, not the figure from the timing run. The script exits 2 without it.
Launch exits 2 if there is no lock ("run the timing command first") and exits 4 if the locked plan no longer fits
the hours you give. It reads the epoch count from the lock only; there is no epochs flag.

Stages, in order: preflight, freeze, train_ordinary, train_adversarial, ptq, val_check, test_clean, robustness,
efficiency, report. Each stage writes its completion marker only after its outputs validate.

### 6a. Short restart check (part of the real run, no extra training)

The reduced workflow replaces V4's skipped restart check with three checks on the real run (no separate smoke run). None runs test inference or computes a metric. The freeze stage reads only test row IDs and labels to fix the subset.

1. Start the run, which stops after the freeze stage (preflight and freeze only, minutes):

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
bash scripts/vm_deadline/launch.sh <HOURS_REMAINING> --stop-after freeze
bash scripts/vm_deadline/status.sh
```

2. Run resume with the same stop point. It must report that nothing needs to run and exit 0, and `status.sh` must show the same
markers as before:

**Ubuntu VM**

```bash
bash scripts/vm_deadline/resume.sh <HOURS_REMAINING_NOW> --stop-after freeze
bash scripts/vm_deadline/status.sh
```

3. Interrupt check during training. Start the campaign with resume (section 8 command), wait until
`artifacts/m2-deadline-v1/models/V1/ckpt/latest.json` exists (the first epoch is saved), then send SIGTERM and resume:

**Ubuntu VM**

```bash
pkill -TERM -f m2_deadline.campaign
# it stops at the end of the current epoch and exits 130; wait for the process to end, then:
bash scripts/vm_deadline/resume.sh <HOURS_REMAINING_NOW>
```

The log must say it continues from the saved epoch, not from epoch 1. If it restarts from scratch or exits with a
guard error, stop and report it; do not edit files.
## 7. Status

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
bash scripts/vm_deadline/status.sh
tail -f "$(ls -t artifacts/m2-deadline-v1/logs/campaign-*.log | head -n 1)"
```

`status.sh` prints each stage marker, whether it still validates, and elapsed times. It is read-only.
(If the `tail` pattern matches nothing, list `artifacts/m2-deadline-v1/logs/` and tail the newest file.)

## 8. Resume

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
nohup bash scripts/vm_deadline/resume.sh <HOURS_REMAINING_NOW> > artifacts/m2-deadline-v1/logs/resume.nohup.out 2>&1 &
```

Use the hours left now. Resume restarts at the first missing or invalid stage. Training continues from the last
completed epoch checkpoint. An interruption, kill or VM shutdown in the middle of an epoch loses that epoch, and
resume restarts it from the previous checkpoint. A single adversarial epoch can take tens of minutes, so a stop
request (Ctrl-C or SIGTERM) may only take effect when the current epoch ends. Hard kills are safe because
checkpoints are written atomically and checked by SHA-256.

Resume checks only the remaining planned work against `<HOURS_REMAINING_NOW> - 3`. If it exits 4, nothing was
changed. Decide by hand what to cut (for example, stop and report the finished stages).

## 9. Regenerate the report

**Ubuntu VM**

```bash
cd ~/IOT-network-traffic-DL-M2-VM-deadline-v1
bash scripts/vm_deadline/report.sh
```

It reads result files only and loads no model. It needs the `test_clean`, `robustness` and `efficiency` markers
to validate (exit 3 otherwise) and overwrites only `results/` and `report/`. The previous report is first copied to
`report/old/<utc>/`. It can be run as often as needed.

## 10. The 9 h and 3 h rule

You asked for at most 9 hours of computation inside 12 elapsed hours. The 3-hour reserve is a design choice, not
your wording: the planned compute cap is `min(9, H - 3)` where `H` is the hours you pass in. At `H = 12` the cap is
9 h. At smaller `H` it shrinks so 3 h stay free for problems and write-up. Planned hours include a 1.25 safety
multiplier and assume sequential execution (no parallel speedup).

## 11. Where the results are

**Ubuntu VM**, all under `~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/`:

| What | File |
|---|---|
| Four-model table (accuracy, macro-F1, size, latency, memory) | `results/model_comparison.csv` and `results/model_comparison.md` |
| FGSM curves (accuracy and macro-F1) | `report/figures/fgsm_curves.png` |
| PGD curves (accuracy and macro-F1) | `report/figures/pgd5_curves.png` |
| Short report | `report/m2_deadline_report.md` |
| Curve data | `results/robustness_curves.csv` |
| Per-example labels, predictions, IDs (full test) | `test/predictions_full_test.csv` |
| Per-example labels, predictions, IDs (robustness subset) | `robustness/predictions_subset.csv` |
| Frozen protocol and subset IDs and counts | `protocol/` |
| Budget estimate and lock | `timing/` |

### Copy back to the laptop

**Windows PowerShell (laptop)**, one command per folder (no brace expansion, which PowerShell does not support):

```powershell
$DEST = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'm2-deadline-results'
New-Item -ItemType Directory -Force $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/results $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/report $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/protocol $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/timing $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/test $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/robustness $DEST
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/efficiency $DEST
```

Alternative (everything, including model checkpoints, calibration data and logs; large):

```powershell
scp -r <VM_USER>@<VM_HOST>:~/IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1 $DEST
```

No archive is created on either side.

## 12. Epoch fallback and the lock

- The timing command picks the largest matched budget from 25, 10, 5 epochs that fits, using runtime only, and
  also picks 10 or 5 robustness rows per class. A reduced epoch count is recorded in the protocol, the model table
  and the report as a deviation from the 25-epoch recipe.
- The lock is written before any performance number exists. Do not edit `configs/m2_deadline_v1.json`, the estimate
  or the lock. Edits are detected by hashes and exit 3.
- `--force-relock` (added to `timing.sh`) archives the old estimate and lock and measures again. Use it only before
  launch. It is refused once any stage marker exists.
- If hours shrink between timing and launch and the plan no longer fits, launch exits 4. Rerun timing with the new
  hours and `--force-relock`.
- Settings are never chosen based on which model scores better. The robustness subset, attack settings and
  epsilons are frozen before test inference.

## 13. Exit codes (all commands)

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Unexpected error (traceback in the log) |
| 2 | Usage, config or data error, or a missing lock or hours. Nothing written |
| 3 | Guard failure: changed protocol, lock, config, V4 manifest or sources; smoke path; existing lock |
| 4 | Budget infeasible, or a blocking correctness gate failed. Do not rerun with changed settings |
| 5 | CPU contention or noisy timing (timing only) |
| 130 | Interrupted at an epoch or stage boundary. Use resume |

## 14. What the results can and cannot say

One seed, the actual epoch count that was locked, a small robustness subset (one example moves a score by
100/n percentage points), a fixed attack budget (FGSM, PGD with 5 steps and 1 start), feature-space perturbations,
and approximate INT8 gradients (BPDA). No cross-seed standard deviations are reported. QAT shows feasibility of the
conversion, serialization and INT8 path only. Score differences are in percentage points.
