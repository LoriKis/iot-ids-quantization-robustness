# Original generated Milestone 2 report (preserved)

These three files are byte-identical copies of what the VM report stage generated for campaign `m2-deadline-v1`.
**Do not edit them.** The edited, reader-facing version is [`../REPORT.md`](../REPORT.md).

| Clone path (relative to `reports/milestone2/`) | VM path | Bytes | SHA-256 (copy = VM file) |
|---|---|---:|---|
| `original/m2_deadline_report.md` | `artifacts/m2-deadline-v1/report/m2_deadline_report.md` | 12510 | `5b9cb0a0bfd1a43623310eee7e497e0f5774387c3bef6b1c0faa77ba4a71ae88` |
| `original/figures/fgsm_curves.png` | `artifacts/m2-deadline-v1/report/figures/fgsm_curves.png` | 140222 | `6bf20f1dce242fab5944595e23e698a2579bfbf56b6a6a56ef554a778edd3680` |
| `original/figures/pgd5_curves.png` | `artifacts/m2-deadline-v1/report/figures/pgd5_curves.png` | 139916 | `7b6aa1e7af427bd504c18cc0b85668f799e576fee0f0a4a4eed5916e5dc86752` |

## Wording in the original that verification found imprecise (original text left unchanged)

See [`../../../verification/results/discrepancies.md`](../../../verification/results/discrepancies.md). In short:

- Table 1 prints "Epochs 25" for the INT8 rows V2 and V4. Those models were converted from the finished parents and had 0 extra training epochs.
- "Peak RSS" in Table 1 is the Linux VmHWM of a fresh process; its baseline already includes the loaded weights and the torch runtime.
- The report says QAT "demonstrates the feasibility of the conversion, serialization and INT8 inference path only", citing the separate pilot file; it contains no comparable QAT model.
- Paths such as `artifacts/m2a-pilot/qat/qat_feasibility.json` refer to VM paths. In this repository that file is at
  [`../evidence/v4-base/m2a-pilot/qat/qat_feasibility.json`](../evidence/v4-base/m2a-pilot/qat/qat_feasibility.json).
- Every number in the original Tables 1 and 2 was recomputed from the saved predictions and matches (see `../../../verification/results/verify_m2.txt`).

Figures: [`figures/fgsm_curves.png`](figures/fgsm_curves.png), [`figures/pgd5_curves.png`](figures/pgd5_curves.png).
