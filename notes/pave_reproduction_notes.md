# PAVE Artifact — Discovery Report

> **Status/context:** Historical starting-checkout discovery snapshot. Preserve its original paths and experimental context; it is not the completed four-seed baseline report.
> Current authoritative results: [Milestone 1 report](../reports/milestone1-artifact/REPORT.md).
## Generated: 2026-10-17T19:05:02.899958Z
## Upstream: C:\Users\Q4C\Desktop\Deep Project\upstream\PAVE_artifact

---
## 1. Processed Data Splits

- `PAVE_artifact\data\balanced_CICIoMT.csv` (5.12 MB)
- `PAVE_artifact\data\balanced_CICIoT.csv` (5.19 MB)
- `PAVE_artifact\data\balanced_CICIoV.csv` (9.15 MB)

---
## 2. Standard Transformer Checkpoints

- **NONE FOUND** — searched *.pt / *.pth matching keywords: transformer, standard, ciciot
  > NOTE: PAVE trains models from scratch at evaluation time. No pre-trained checkpoint files are shipped in the repository.

---
## 3. Evaluation Script (from README)

- README parsed: `PAVE_artifact\README.md`
- Python invocations found in README code blocks:
  ```
  python revision_experiments.py --exp all                    # 3 datasets x 3 seeds
  ```
  ```
  python revision_experiments.py --exp all --quick            # smoke: 1 dataset, 1 seed
  ```
  ```
  python summarize_revision.py                                 # -> summary_*.csv
  ```

---
## 4. FGSM Attack Implementation

- `PAVE_artifact\code\config.py`
  - line 190: `Transfer Attack (FGSM on a surrogate DNN) — zero-query black-box`
  - line 194: `fgsm_epsilons: List[float] = field(`
- `PAVE_artifact\code\defense_benchmark.py`
  - line 12: `clean vs FGSM):`
  - line 30: `flat_fgsm, seq_logits, seq_fgsm, seq_pgd, seq_square,`
- `PAVE_artifact\code\revision_experiments.py`
  - line 11: `fastat    R1.10            Fast-AT (FGSM random-start) vs PGD-TRADES: robustness + train time.`
  - line 27: `from sota_compare import (get_data, cap_aligned, expand, FlatMLP, train_flat, flat_fgsm,`
- `PAVE_artifact\code\sota_compare.py`
  - line 7: `FGSM on one shared flat surrogate MLP, evaluated on every target. White-box`
  - line 8: `FGSM/PGD/Square are additionally reported for every differentiable model.`
- `PAVE_artifact\code\summarize_revision.py`
  - line 48: `g = gmean(d, ["defense", "hp", "dataset"], ["clean_mcc", "fgsm_mcc", "pgd_mcc"])`
  - line 50: `# best hp per (defense,dataset) by fgsm_mcc`
- `PAVE_artifact\code\attacks\adversarial.py`
  - line 2: `adversarial.py — FGSM, PGD, and Gaussian Noise adversarial attackers.`
  - line 10: `FGSM: ε ∈ {0.01, 0.05, 0.10, 0.15, 0.20}`
- `PAVE_artifact\code\models\agt.py`
  - line 157: `feature distributions even though no FGSM/PGD examples are seen during`
  - line 488: ``mc_predict()` this gives gradient-obfuscated robustness — FGSM/PGD`
- `PAVE_artifact\code\models\robust_eval.py`
  - line 76: `FGSM-style attacks need gradients to flow back to the input`

---
## 5. PGD Attack Implementation

- `PAVE_artifact\code\config.py`
  - line 188: `V3: PGD (white-box) is REPLACED with two blackbox attacks:`
  - line 218: `# Legacy PGD fields kept for backward compatibility with older runs/tests;`
- `PAVE_artifact\code\defense_benchmark.py`
  - line 9: `V7 AT (control)   PGD-TRADES adversarial training (K=7, alpha=eps/4, beta=6)`
  - line 30: `flat_fgsm, seq_logits, seq_fgsm, seq_pgd, seq_square,`
- `PAVE_artifact\code\pave_audit.py`
  - line 9: `mcc_unconstrained   MCC under PGD with NO validity constraint (attacker perturbs freely)`
  - line 10: `mcc_valid           MCC under the SAME PGD projected into the valid protocol domain (PAVE)`
- `PAVE_artifact\code\revision_experiments.py`
  - line 6: `proj      R1.2/R1.14/R2.2  projected-PGD (constraint-respecting attacker):`
  - line 7: `MCC clean / unconstrained-PGD / projected-PGD + validity,`
- `PAVE_artifact\code\sota_compare.py`
  - line 8: `FGSM/PGD/Square are additionally reported for every differentiable model.`
  - line 10: `* XGBoost was removed: as a non-differentiable tree ensemble, FGSM/PGD/Square`
- `PAVE_artifact\code\summarize_revision.py`
  - line 37: `# ---- R1.2/R1.14/R2.2 : projected-PGD ----`
  - line 41: `["mcc_clean", "mcc_pgd_unconstrained", "mcc_pgd_projected",`
- `PAVE_artifact\code\attacks\adversarial.py`
  - line 2: `adversarial.py — FGSM, PGD, and Gaussian Noise adversarial attackers.`
  - line 11: `PGD:  ε=0.10, α=0.01, iterations=10`
- `PAVE_artifact\code\models\agt.py`
  - line 157: `feature distributions even though no FGSM/PGD examples are seen during`
  - line 488: ``mc_predict()` this gives gradient-obfuscated robustness — FGSM/PGD`

---
## 6. Default Attack Parameters (Extracted from Source)

- **FGSM_epsilon** = `0.1`
- **PGD_alpha** = `0.01`
- **PGD_eps** = `0.1`
- **PGD_iters** = `10`

---
## 7. Protocol Validity Caveat

Unconstrained feature-space adversarial perturbations (FGSM, PGD) applied to network traffic features may produce feature vectors that do not correspond to any physically realizable network packet sequence. Results should be interpreted in the feature-space threat model only.

The PAVE framework explicitly evaluates both **unconstrained** and **constraint-respecting (projected)** variants of PGD to quantify how much robustness is overstated when protocol validity is ignored.
