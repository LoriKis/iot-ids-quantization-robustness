# Milestone 2 post-run verification

**Post-run work, performed on a laptop from saved files. No model was loaded, and no training, inference, adversarial evaluation or benchmark was run.** None of this was part of the VM campaign.

| File | What it does |
|---|---|
| [`verify_m2.py`](verify_m2.py) | Recomputes clean accuracy and macro-F1 (all 27 labels, `zero_division=0`) for all 3,764 test ids, the 270-id robustness tables, eps-0 agreement, efficiency statistics; checks split, scaler, calibration, hashes, parents, epochs and provenance; regenerates both figures from saved predictions. |
| [`check_links.py`](check_links.py) | Checks that every relative link in the new Markdown files (including `reports/milestone2/HANDOFF.md`) resolves, and that the numbers in the REPORT tables equal `verify_m2.json`. |
| `results/copy_check.txt`, `results/copy_map.tsv` | **Historical record**, not repeatable from this repository. Output and path map of a one-time byte-for-byte comparison of the copied files against the VM originals. The PowerShell script that produced it needed both VM folders and the full VM file manifests of the setup step, so it is **not published**. |
| `results/vm_hash_check_postrun.tsv` | **POST-RUN, VM-only.** SHA-256 of the checkpoint and other files recorded in the stage markers, computed directly on the VM folder (no model loaded). Repeatable only with that VM folder. |
| `results/verify_m2.txt`, `verify_m2.json`, `console_output.txt` | Output of the run (`PASS`/`FAIL`/`NOTE` lines). |
| [`results/discrepancies.md`](results/discrepancies.md), [`results/figure_review.md`](results/figure_review.md) | Where sources disagree; visual review of the original figures. |
| `figures/*_recomputed.png` | **Post-run regeneration** from saved predictions. The original figures are in `reports/milestone2/original/figures/`. |

## How to run

Python 3 with numpy and matplotlib (scikit-learn optional). Run from outside the repository tree so that no cache files are created inside it:

```
set PYTHONDONTWRITEBYTECODE=1
python <repo>/verification/verify_m2.py --evidence-root <repo>/reports/milestone2/evidence --out <repo>/verification/results
python <repo>/verification/check_links.py
```

The script writes only into `--out` (and `--figures`). The previously reported values (61.96 %, 61.77 %, 53.85 %, 53.93 %, 4.81 %, 32.96 %) are cross-checks only; the script never changes data to match.
The last run: 174 PASS, 0 FAIL, 25 NOTE.
