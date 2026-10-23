# PAVE Milestone 1 repair handoff

**Updated:** 2026-10-24, documentation and verification cleanup complete.
**Branch:** `local/research-pipeline-fixes`.
**Current revision:** `da89a2ce70c6ddbbe31b8421da8ff7866e51a277` plus uncommitted
repair files. No commit, push, ZIP, or Milestone 2 work was performed.

## Actual status

| Area | Status and evidence |
|---|---|
| Released baseline (A) | Completed and preserved. `scripts/check_milestone1_review.py` passes; recorded means remain Clean 65.808333%, FGSM 44.416667%, PGD 41.641667%. |
| Historical AdamW variant (B) | Preserved as historical and not relabelled. The Mac run, executed `executor_snapshot.py`, checkpoints, and smoke artifacts are unavailable locally. |
| Paper provenance | PDF hash and wording check are recorded in `reports/milestone1-paper-described-adam/paper_wording_check.json`; the paper says Adam, not AdamW. |
| Corrected implementation (C) | New config, protocol, Adam runner, domain policy, fingerprint gate, smoke checks, and tests are present. |
| Corrected smoke | Passed in `runs/milestone1-paper-described-adam-smoke/`; selected IDs are partition-safe and smoke state is not reused. |
| Corrected full run (C) | Completed and verified in `runs/milestone1-paper-described-adam-rerun/`; seeds 1, 7, 13, 29 each used 25 epochs, 1,050 Adam steps and 524,925 sample exposures. |
| Domain preflight | Completed at `reports/milestone1-repair/preflight_domain_scan.json`: test infeasible rows 0; validation infeasible rows 2 and retained as a documented diagnostic. |
| Tests/checkers | Focused reload regression tests and full `pytest -q` pass after the later source patch; corrected four-seed verification and independent compact checker remain recorded-run checks. |

## Corrected run results

The shared 4,500-row test partition produced the following mean ± sample SD:

| Condition | Accuracy | Macro-F1 | Weighted-F1 | MCC | ASR |
|---|---:|---:|---:|---:|---:|
| Clean | 63.644 ± 0.940% | 58.305 ± 1.016% | 58.326 ± 1.016% | 0.628 ± 0.009 | 0.000 ± 0.000% |
| FGSM | 15.422 ± 4.278% | 12.919 ± 3.981% | 12.916 ± 3.993% | 0.124 ± 0.044 | 77.266 ± 5.509% |
| PGD | 7.922 ± 1.660% | 6.524 ± 1.092% | 6.509 ± 1.085% | 0.045 ± 0.018 | 89.030 ± 3.744% |

Measured aggregate runtime was 5,338.999 training seconds, 252.553 attack
seconds, and 5,664.549 invocation seconds across the four seeds. The compact
report and evidence are in `reports/milestone1-paper-described-adam/`.

The saved overlap audit reports 2,804/4,500 test rows (62.31%) with identical
feature patterns in training. Row-disjoint IDs do not establish independence
from duplicate feature patterns. The historical smoke reconstruction reports
61 distinct test rows exposed by the old smoke selection; the test set is not
described as completely untouched throughout development. Adversarial examples
remain feature-space tensors without protocol-valid traffic guarantees.

## Historical evidence gap

Request from the teammate: the byte-exact AdamW `executor_snapshot.py`
(expected SHA-256 `619ad2f42eb399c3d65b679d4389e141dc524ad5278da7af04276f9d3d37726f`),
the Mac run's `reference/`, manifests, checkpoints, verification/log files,
historical smoke outputs, and source environment records. Do not substitute the
current runner or copy them over the corrected run.

## Next actions

The full run, report generation, compact export and independent checks are
complete. The exact recorded-environment verification command was:

```powershell
.\.venv\Scripts\python.exe -B -m research.paper_described_adam --config configs\milestone1_paper_described_adam.json --run-dir runs\milestone1-paper-described-adam-rerun --verify-only
```

The compact checker command was:

```powershell
.\.venv\Scripts\python.exe -B scripts\check_milestone1_adam_review.py --collect --run-dir runs\milestone1-paper-described-adam-rerun --package reports\milestone1-paper-described-adam
```

The first failed run in `runs/milestone1-paper-described-adam/` remains
preserved; it is not reused because the source fingerprint changed after its
metadata-only verification failure.

## Post-run source patch and provenance

After the completed C run, `research/paper_described_adam.py` was strengthened
so a checkpoint reload mismatch raises an explicit `RuntimeError` before
verification/completion metadata can be written. Focused regression tests cover
matching and mismatching clean logits. This changes the current source hash;
the completed results used the preserved pre-patch executed source. The
run-level `executor_snapshot.py`, fingerprints, manifests, checkpoints and
compact evidence were not rewritten. The modified source must not be used to
bypass the resume fingerprint gate or resume either historical run.

## Final Milestone 1 status

Milestone 1 is complete as a practical released-code baseline plus a
paper-described comparison study: A and C are the two main implemented
configurations, and B is preserved as a historical AdamW variant. Exact
matched-condition reproduction of Table 4 remains unresolved. This handoff
does not claim full satisfaction of every official requirement or external
reviewer approval of the whole project.

## Cleanup checks run

- `.\.venv\Scripts\python.exe -B -m pytest -q tests\test_resume_fingerprints.py -k checkpoint_reload` — **2 passed**.
- `.\.venv\Scripts\python.exe -B -m pytest -q` — **71 passed**, 7 existing warnings.
- `.\.venv\Scripts\python.exe -B scripts\check_milestone1_review.py` — **PASS** for A.
- `.\.venv\Scripts\python.exe -B scripts\check_milestone1_adam_review.py --package reports\milestone1-paper-described-adam` — **PASS** for C.
- Saved-metric recomputation reproduced the comparison values in
  `reports/milestone1-paper-described-adam/COMPARISON.md`.
- `git diff --check` and the local Markdown-link audit — **PASS**.
- Preserved C run-level executed-source snapshots matched all four recorded
  hashes. The current source intentionally differs after the reload guard
  patch, so no post-patch `--verify-only` or resume was attempted.

No training, attack evaluation, dependency installation, commit, push, or
publication occurred during this cleanup.

The corrected experiment remains a multiclass configuration comparison, not an
exact Table 4 reproduction, and no independent reviewer approval is claimed.
