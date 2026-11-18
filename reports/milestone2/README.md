# Milestone 2: INT8 quantization and adversarial training (entry point)

**Read this first.** Milestone 1 (M1) is unchanged in this repository (see the [root README](../../README.md) and the [M1 report](../milestone1-artifact/REPORT.md)).
This folder adds Milestone 2 (M2).

**What is original and what is new.** Folders [`evidence/`](evidence/README.md) and [`source/`](source/README.md) hold byte-identical copies of files produced or
written on the VM. [`original/`](original/README.md) holds the report the VM generated. **Everything else here is post-run work** written afterwards from those files:
this README, [`REPORT.md`](REPORT.md), [`CHECKLIST.md`](CHECKLIST.md), [`post-run-adaptation/`](post-run-adaptation/README.md) and the
[`verification/`](../../verification/README.md) folder at the repository root.

## What did we compare?

Four models of the same small Transformer, which predicts one of **27 labels** (benign or one of 26 malicious-traffic types) from a row of 39 traffic features:

| ID | Model |
|---|---|
| V1 | ordinary FP32 (trained 25 epochs, seed 1) |
| V2 | V1 converted to INT8 by post-training quantization (PTQ; no further training) |
| V3 | adversarially trained FP32 (25 epochs, seed 1) |
| V4 | V3 converted to INT8 by PTQ (no further training) |

We measured clean accuracy and macro-F1 on 3,764 held-out rows, accuracy and macro-F1 under FGSM and 5-step PGD attacks on a fixed 270-row subset (eps 0.05 and 0.10), and model
size, latency and memory. Terms are explained in [REPORT.md, section 1](REPORT.md#1-plain-language-background). Which earlier experiment is which:
[REPORT.md, section 2](REPORT.md#2-which-experiment-is-this).

## What happened?

Accuracy and robustness numbers were **recomputed from predictions** saved by the run (they match the values from the VM run after rounding). Clean accuracy uses all **3,764** test rows. The robustness column uses the **same 270 selected rows for every model**, so the two columns have different denominators and must not be mixed:

| | Clean acc., 3,764 rows | PGD-5, eps 0.10, 270 rows |
|---|---|---|
| V1 FP32 ordinary | 61.96 % | 4.44 % |
| V2 INT8 of V1 | 61.77 % | 4.81 % |
| V3 FP32 adversarial | 53.85 % | 31.85 % |
| V4 INT8 of V3 | 53.93 % | 32.96 % |

- PTQ changed clean accuracy by -0.19 pp (V2 vs V1) and +0.08 pp (V4 vs V3). Under attack the differences are small and of mixed sign.
- Adversarial training cost about 8 pp of clean accuracy and raised accuracy under attack from under 5 % to about 32 % at eps 0.10. This comes from **one training seed** per parent.
- INT8 serialized size is 34 % of FP32. On this machine INT8 was slower for one row at a time (about 5.4 vs 2.9 ms), faster per example in batches of 256 (about 1.05 to 1.08 vs 1.24 ms) and did not use less process memory. These timings are **three efficiency measurement repetitions** of the same trained models, recomputed from the raw measurement files. They are not three training runs.

## Five things this folder keeps apart

1. **Four measured models vs the QAT pilot.** Only V1 to V4 were measured. QAT (quantization-aware training) was **deferred**. A separate small feasibility pilot shows only that one QAT epoch ran, converted to INT8 and serialized without error; there is **no comparable QAT accuracy result and no measured runtime estimate** ([REPORT.md section 6](REPORT.md#6-qat-deferred)).
2. **3,764 vs 270 rows.** Clean accuracy and macro-F1 use all 3,764 test rows. Robustness uses the same 270 selected rows (10 per class) for every model.
3. **One seed vs three repetitions.** One training seed (seed 1) per parent. The three repetitions apply only to the efficiency timing.
4. **Recomputed vs supported only by a record.** Labels used everywhere: **recomputed from predictions**, **confirmed from a config or log**, **reported but not independently verified**. No scaler refit and no calibration reconstruction was performed. The checkpoint file hashes recorded by the run were recomputed directly on the VM files after the run (POST-RUN, no model loaded) and match.
5. **Exact VM source vs post-run adaptation.** [`source/v4-base/`](source/README.md) and [`source/deadline-v1-overlay/`](source/README.md) are byte-identical copies of what ran. [`post-run-adaptation/`](post-run-adaptation/README.md) is a documentation note only. The M2 versions of files that conflict with the repository (`research/attacks.py`, `research/common.py`, `.gitattributes`, `pytest.ini`, `tests/conftest.py`) are under `source/` and **must not be silently replaced with the M1 versions at the repository root**. The reorganized layout was **not** tested end to end; only the checks of the portable checker and link checker were run on a clean checkout of the files to be committed.

Details, figures and tables: [REPORT.md](REPORT.md).

## How do I inspect the evidence?

1. Open [`REPORT.md`](REPORT.md) section 4.5: each headline result has a link to the file that supports it and a tag (**recomputed from predictions**, **confirmed from a config or log**, **reported but not independently verified**).
2. Re-run the checks (needs Python 3 with numpy and matplotlib; scikit-learn is optional; they read only the files in this repository, no model is loaded; set `PYTHONDONTWRITEBYTECODE=1` and send the checker output outside the repository):
   `python verification/verify_m2.py` and `python verification/check_links.py` (see [`verification/README.md`](../../verification/README.md)). Saved outputs: [`verify_m2.txt`](../../verification/results/verify_m2.txt).
3. Raw data: [`evidence/README.md`](evidence/README.md) (prediction files, efficiency raw measurements, protocol, split identity, completion records, with SHA-256 per file).
4. Code that produced the run: [`source/README.md`](source/README.md) (V4 base and deadline overlay, kept apart).
5. Where sources disagree: [`discrepancies.md`](../../verification/results/discrepancies.md).
6. Requirement checklist: [`CHECKLIST.md`](CHECKLIST.md).

## What did we not establish?

- Anything beyond **one seed and one training run per parent** (the three efficiency repetitions are not three training runs).
- **Restart or recovery behaviour.** The campaign logs show no interruption during training; the no-op resume after the freeze stage is not evidence of recovery ([REPORT.md section 7](REPORT.md#7-discrepancies-and-provenance)).
- That the small PTQ differences are meaningful: no equivalence, statistical significance or inherent INT8 robustness benefit is shown.
- Robustness outside FGSM and 5-step PGD at eps 0.05 and 0.10, or for traffic that can really be sent (perturbations are in feature space).
  INT8 attacks use an approximate gradient and may be too weak.
- A comparison with the PAVE paper's Table 4, or any cause of differences between M1 and M2. The valid PTQ comparisons are V2 vs V1 and V4 vs V3 within M2 only.
- **QAT (quantization-aware training) comparison: deferred, not done.** No comparable QAT accuracy result and no measured runtime estimate exist. Only a separate small feasibility pilot exists ([REPORT.md section 6](REPORT.md#6-qat-deferred)).
- An end-to-end rerun from this repository layout. None was performed ([`post-run-adaptation/`](post-run-adaptation/README.md)). An independent scaler refit and a calibration reconstruction were also not performed.
- **Milestone 3 is pending**; nothing planned for it is reported as a result.

## Links

[REPORT.md](REPORT.md) | [CHECKLIST.md](CHECKLIST.md) | [original/](original/README.md) | [evidence/](evidence/README.md) | [source/](source/README.md) |
[verification/](../../verification/README.md) | [post-run-adaptation/](post-run-adaptation/README.md) |
M1: [artifact report](../milestone1-artifact/REPORT.md), [AdamW variant](../milestone1-paper-described/REPORT.md), [Adam recreation](../milestone1-paper-described-adam/REPORT.md)
