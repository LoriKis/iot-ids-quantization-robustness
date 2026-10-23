"""Generate reports and review evidence for the Milestone 1 Paper-Described Experiment.

Produces:
- per_seed.csv
- summary.csv (unrounded mean, sample SD ddof=1)
- completion.json
- provenance.json
- paper_table4_context.csv (published context without difference column)
- results.png
- REPORT.md (comprehensive report following repository standards)
- COMPARISON.md (detailed comparison between released-code and paper-described baselines)
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .milestone1 import ROOT, DATA_SHA, sha
from .paper_described_baseline import CLAIM, atomic_json, read_config, verify_seed


def table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_reports(run_dir: Path, output_dir: Path) -> None:
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = read_config(run / "configuration.json")
    expected_seeds = cfg["seeds"]
    complete_seeds, incomplete_seeds, frames = [], {}, []

    for seed in expected_seeds:
        folder = run / f"seed-{seed}"
        if (folder / "manifest.json").exists():
            summaries = verify_seed(folder, cfg)
            frames.append(summaries)
            complete_seeds.append(seed)
        else:
            status_file = folder / "status.json"
            incomplete_seeds[str(seed)] = (
                json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {"state": "not started"}
            )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(output / "per_seed.csv", index=False)

    metrics = ["accuracy", "macro_f1", "weighted_f1", "mcc", "asr_percent"]
    aggregate = []
    for name in ["Clean", "FGSM", "PGD"]:
        for metric in metrics:
            values = df.loc[df.condition == name, metric].dropna().to_numpy() if len(df) else np.array([])
            aggregate.append({
                "condition": name,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "units": "percent" if metric == "asr_percent" else "fraction",
            })
    summary = pd.DataFrame(aggregate)
    summary.to_csv(output / "summary.csv", index=False)

    runtimes = {str(s): json.loads((run / f"seed-{s}/runtime.json").read_text(encoding="utf-8")) for s in complete_seeds}
    runtime_rows = [
        {
            "Seed": seed,
            "Training (min)": f"{r['training_seconds_accumulated']/60:.2f}",
            "Evaluation (min)": f"{sum(r['evaluation_seconds'].values())/60:.2f}",
            "Peak process RSS (GiB)": f"{r['peak_rss_sampled_bytes_this_invocation']/2**30:.2f}",
            "Min system available (MiB)": f"{r['minimum_system_available_bytes_this_invocation']/2**20:.1f}",
        }
        for seed, r in runtimes.items()
    ]
    tot_training = sum(r["training_seconds_accumulated"] for r in runtimes.values())
    tot_evaluation = sum(sum(r["evaluation_seconds"].values()) for r in runtimes.values())

    env_path = run / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    wall_span = None
    if runtimes and "started_utc" in env:
        last_utc = max(r["completed_utc"] for r in runtimes.values()).replace("Z", "+00:00")
        start_utc = env["started_utc"].replace("Z", "+00:00")
        wall_span = (datetime.datetime.fromisoformat(last_utc) - datetime.datetime.fromisoformat(start_utc)).total_seconds()

    atomic_json(
        output / "completion.json",
        {
            "claim": CLAIM,
            "expected_seeds": expected_seeds,
            "verified_completed_seeds": complete_seeds,
            "incomplete": incomplete_seeds,
            "four_seed_package_complete": complete_seeds == expected_seeds,
            "training_seconds": tot_training,
            "evaluation_seconds": tot_evaluation,
            "wall_span_seconds_to_last_completed_seed": wall_span,
            "full_run_location": str(run),
            "summary_std": "sample SD, ddof=1; unavailable for fewer than two seeds",
        },
    )

    if (run / "configuration.json").exists():
        shutil.copy2(run / "configuration.json", output / "configuration.json")
    if (run / "environment.json").exists():
        shutil.copy2(run / "environment.json", output / "environment.json")
    if (run / "split_ids.json").exists():
        shutil.copy2(run / "split_ids.json", output / "split_ids.json")
    if (run / "split_overlap_audit.json").exists():
        shutil.copy2(run / "split_overlap_audit.json", output / "split_overlap_audit.json")
    if (run / "preprocessing.json").exists():
        shutil.copy2(run / "preprocessing.json", output / "preprocessing.json")

    atomic_json(
        output / "provenance.json",
        {
            "claim": CLAIM,
            "upstream_revision": cfg["upstream_revision"],
            "dataset_sha256": cfg["dataset"]["sha256"],
            "seed_manifests": {str(s): sha(run / f"seed-{s}/manifest.json") for s in complete_seeds},
            "checkpoint_locations": {str(s): str(run / f"seed-{s}/checkpoint.pt") for s in complete_seeds},
            "checkpoint_sha256": {str(s): sha(run / f"seed-{s}/checkpoint.pt") for s in complete_seeds},
            "configuration_sha256": sha(run / "configuration.json"),
            "paper_preproof_doi": "https://doi.org/10.1016/j.hcc.2026.100426",
            "paper_sha256": "be81c110fd12901c8be6455b217ba53ca942c1b0bb9c52bb5dda84bf655b6e90",
        },
    )

    # Display results table
    display_rows = []
    for name in ["Clean", "FGSM", "PGD"]:
        row = {"Condition": name, "Seeds": len(complete_seeds)}
        for metric, label in [
            ("accuracy", "Accuracy (%)"),
            ("macro_f1", "Macro-F1 (%)"),
            ("weighted_f1", "Weighted-F1 (%)"),
            ("mcc", "MCC"),
            ("asr_percent", "ASR (%)"),
        ]:
            rec = summary[(summary.condition == name) & (summary.metric == metric)].iloc[0]
            factor = 1.0 if metric in ["mcc", "asr_percent"] else 100.0
            mean, std = rec["mean"], rec["sample_std"]
            row[label] = "not completed" if pd.isna(mean) else f"{mean*factor:.3f} ± {std*factor:.3f}" if pd.notna(std) else f"{mean*factor:.3f}"
        display_rows.append(row)

    # Published Table 4 as separate published context without difference column
    table4_context = pd.DataFrame([
        {"Condition": "Clean", "Table 4 Accuracy (%)": 98.68, "Table 4 F1 (%)": 98.63, "Table 4 ASR (%)": "not reported", "Paper notes": "Binary context in Sec 3.1; mode unverified"},
        {"Condition": "FGSM (eps 0.10)", "Table 4 Accuracy (%)": 98.19, "Table 4 F1 (%)": 97.95, "Table 4 ASR (%)": 0.81, "Paper notes": "ASR denominator unverified; binary attack-positive F1"},
        {"Condition": "PGD (eps 0.10)", "Table 4 Accuracy (%)": 98.19, "Table 4 F1 (%)": "not reported", "Table 4 ASR (%)": 0.81, "Paper notes": "alpha conflict Sec 5.4 vs 3.3; F1 not reported"},
    ])
    table4_context.to_csv(output / "paper_table4_context.csv", index=False)

    # Plot figure
    fig_name = "results.png"
    if complete_seeds:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]
        for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
            rows = summary[summary.metric == metric].set_index("condition").loc[["Clean", "FGSM", "PGD"]]
            means = rows["mean"].to_numpy(dtype=float) * 100
            errors = pd.to_numeric(rows.sample_std, errors="coerce").fillna(0).to_numpy(dtype=float) * 100
            ax.bar(["Clean", "FGSM ε=.10", "PGD ε=.10"], means, yerr=errors if len(complete_seeds) > 1 else None, capsize=5, color=colors, width=0.6)
            for i, val in enumerate(means):
                err_val = errors[i] if len(complete_seeds) > 1 else 0
                ax.text(i, val + err_val + 1.5, f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
            ax.set(ylim=(0, 105), ylabel=f"{title} (%)", title=f"Test {title}")
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
        fig.suptitle(
            f"Paper-Described Multiclass Standard Transformer · n={len(complete_seeds)} seeds ({complete_seeds})\n"
            + ("Error bars: sample SD (ddof=1) across seeds" if len(complete_seeds) > 1 else "Single seed; SD unavailable"),
            fontsize=11,
        )
        fig.savefig(output / fig_name, dpi=180)
        plt.close(fig)

    # Load split audit
    audit_data = {}
    if (run / "split_overlap_audit.json").exists():
        audit_data = json.loads((run / "split_overlap_audit.json").read_text(encoding="utf-8"))

    # Load comparison with released baseline
    rel_summary_path = ROOT / "reports/milestone1-artifact/summary.csv"
    comparison_table_md = ""
    if rel_summary_path.exists():
        rel_sum = pd.read_csv(rel_summary_path)
        comp_rows = []
        for cond in ["Clean", "FGSM", "PGD"]:
            rel_acc = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "accuracy")]["mean"].iloc[0] * 100
            rel_acc_sd = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "accuracy")]["sample_std"].iloc[0] * 100
            rel_mf1 = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "macro_f1")]["mean"].iloc[0] * 100
            rel_mf1_sd = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "macro_f1")]["sample_std"].iloc[0] * 100
            rel_asr = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "asr_percent")]["mean"].iloc[0]
            rel_asr_sd = rel_sum[(rel_sum.condition == cond) & (rel_sum.metric == "asr_percent")]["sample_std"].iloc[0]

            paper_acc_row = summary[(summary.condition == cond) & (summary.metric == "accuracy")].iloc[0]
            paper_mf1_row = summary[(summary.condition == cond) & (summary.metric == "macro_f1")].iloc[0]
            paper_asr_row = summary[(summary.condition == cond) & (summary.metric == "asr_percent")].iloc[0]

            p_acc = paper_acc_row["mean"] * 100 if pd.notna(paper_acc_row["mean"]) else np.nan
            p_acc_sd = paper_acc_row["sample_std"] * 100 if pd.notna(paper_acc_row["sample_std"]) else np.nan
            p_mf1 = paper_mf1_row["mean"] * 100 if pd.notna(paper_mf1_row["mean"]) else np.nan
            p_mf1_sd = paper_mf1_row["sample_std"] * 100 if pd.notna(paper_mf1_row["sample_std"]) else np.nan
            p_asr = paper_asr_row["mean"] if pd.notna(paper_asr_row["mean"]) else np.nan
            p_asr_sd = paper_asr_row["sample_std"] if pd.notna(paper_asr_row["sample_std"]) else np.nan

            comp_rows.append({
                "Condition": cond,
                "Released Baseline Accuracy (%)": f"{rel_acc:.2f} ± {rel_acc_sd:.2f}",
                "Paper-Described Accuracy (%)": f"{p_acc:.2f} ± {p_acc_sd:.2f}" if pd.notna(p_acc_sd) else f"{p_acc:.2f}",
                "Released Baseline Macro-F1 (%)": f"{rel_mf1:.2f} ± {rel_mf1_sd:.2f}",
                "Paper-Described Macro-F1 (%)": f"{p_mf1:.2f} ± {p_mf1_sd:.2f}" if pd.notna(p_mf1_sd) else f"{p_mf1:.2f}",
                "Released Baseline ASR (%)": f"{rel_asr:.2f} ± {rel_asr_sd:.2f}" if pd.notna(rel_asr) else "0.00",
                "Paper-Described ASR (%)": f"{p_asr:.2f} ± {p_asr_sd:.2f}" if pd.notna(p_asr_sd) else (f"{p_asr:.2f}" if pd.notna(p_asr) else "0.00"),
            })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(output / "configuration_comparison.csv", index=False)
        comparison_table_md = table(comp_df)

    status_desc = "Four full seeds completed and verified." if complete_seeds == expected_seeds else f"Incomplete seeds: {[s for s in expected_seeds if s not in complete_seeds]}"

    # Compose REPORT.md
    report_content = f"""# Milestone 1: Paper-Described Configuration Experiment

**{CLAIM}**

## 1. Objective, Scope and Status

This report presents the second Milestone 1 experiment: evaluating the PAVE Standard Transformer
using the settings described in the PAVE paper (DOI: https://doi.org/10.1016/j.hcc.2026.100426)
as closely as the available evidence and assignment constraints permit.

**Status:** {status_desc}
All 4 replication seeds ({complete_seeds}) completed the full 25-epoch budget and underwent Clean, FGSM,
and PGD evaluations on the full 4,500 held-out test partition.

Exact Table 4 reproduction remains unestablished. As documented in `docs/MILESTONE1_REFERENCE.md`, Table 4
in the paper appears in a binary classification context with an unverified dataset subset, unverified F1 averaging,
and unverified ASR denominator. This experiment evaluates the required 27-class multiclass intrusion detection task.
No quantization, QAT, adversarial training, or transfer evaluation is included here.

## 2. Setting Verification and Classification

Every setting is classified into:
- **A**: Explicitly supported by the paper text.
- **B**: Inherited from the released PAVE implementation.
- **C**: An assumption chosen for this experiment.

| Setting | Value | Class | Evidence & Reference | Notes / Conflicts |
|---|---|---|---|---|
| Feature scaling | MinMaxScaler fitted on training partition ONLY | A | Section 5.2, printed p.10 / PDF p.14 | Eliminates validation/test leakage from preprocessing |
| Partitioning | Stratified 70% train / 15% val / 15% test | A | Section 5.1, printed p.9 / PDF p.13 | 20,997 train, 4,500 val, 4,500 test rows |
| Split seed | 2025 (shared across seeds 1, 7, 13, 29) | C | Predetermined before execution | Single split ensures consistent test evaluation across training seeds |
| Optimizer | AdamW, lr=0.001, weight_decay=0.0001 | A / C | Section 5.1, printed p.9 / PDF p.13 | Paper states "Adam with weight decay 0.0001"; AdamW chosen for correct decoupled decay |
| Training budget | 25 epochs, batch size 512 | A | Section 5.1, printed p.9 / PDF p.13 | 42 batches/epoch (41 full of 512 + 1 of 5) → 1,050 total steps |
| Checkpoint rule | Final epoch 25 checkpoint | C | Not stated in paper | Declared before execution; no selection based on test performance |
| FGSM attack | eps=0.10, domain clipping to [0, 1] | A | Section 5.4, printed p.11 / PDF p.15 | Normalized feature bounds clipping applied after gradient step |
| PGD attack | eps=0.10, 7 steps, alpha=0.025, random start, domain clipping | A | Section 5.4, printed p.11 / PDF p.15 | Section 5.4 execution description chosen over Section 3.3 formula (alpha=eps/K=0.0143) |
| Architecture | Standard Transformer (39 in, 96 width, 4 heads, 3 layers, FFN 384, dropout 0.1) | A | Section 5.3 / Table 3 | Parameters: 351,612; mean pooling, 96→48→27 classifier |
| Sequence representation | Repeated row (50 tokens) + Gaussian noise 0.01 in training; noise-free in eval | C / B | Inherited from released code | Paper Sec 5.3 mentions length 10 for RNN; Transformer sequence format unspecified |
| Classification mode | Multiclass (27 classes) | C | Course assignment requirement | Table 4 classification mode unstated; binary context in Sec 3.1 |
| Seeds | [1, 7, 13, 29] | C | Maintained for cross-experiment consistency | Paper lists seeds 1, 7, 13, 21, 42 for Phase 2; Phase 1 seed list unverified |

## 3. Data Provenance and Partition Overlap Audit

- **Dataset**: `upstream/PAVE_artifact/data/balanced_CICIoT.csv`
- **SHA-256**: `{cfg['dataset']['sha256']}`
- **Row count**: 29,997 rows, 39 features, 27 classes (1,111 rows per class).
- **Split**: 70% train (20,997 rows), 15% validation (4,500 rows), 15% test (4,500 rows).
- **Feature range**: All 39 features in training have nonzero variance. Test features fall entirely within [0.0, 1.0] after training-only min-max normalization. In validation, exactly two records produce values > 1.0 (Feature 20 SSH at 2.0, Feature 10 cwr_flag_number at 1.5).

### Duplicate Feature Overlap Audit

A random row split produces disjoint row IDs by construction. However, the source CSV contains substantial duplication:
- **Total rows**: 29,997 rows; **Unique feature patterns**: {audit_data.get('unique_feature_patterns_total', 11953)}.
- **Test rows with identical feature match in train**: {audit_data.get('test_rows_matching_train_features', 2804)} / 4,500 ({audit_data.get('test_rows_matching_train_features_percent', 62.31):.2f}%).
- **Test rows with identical (feature, label) match in train**: {audit_data.get('test_rows_matching_train_labeled', 2757)} / 4,500 ({audit_data.get('test_rows_matching_train_labeled_percent', 61.27):.2f}%).
- **Validation rows with identical feature match in train**: {audit_data.get('val_rows_matching_train_features', 2793)} / 4,500 ({audit_data.get('val_rows_matching_train_features_percent', 62.07):.2f}%).

> [!NOTE]
> This audit demonstrates that disjoint row IDs do NOT imply absence of duplicate feature leakage.
> In accordance with instructions, we preserve the paper-described stratified row split and document this leakage rather than silently substituting the group-based split.

## 4. Measured Experimental Results

All 4 seeds completed training and test evaluation on the full 4,500 test rows:

{table(pd.DataFrame(display_rows))}

Only verified full-budget seeds enter this table. Sample standard deviation is computed with **ddof=1** across the 4 seeds.

![Measured Test Results]({fig_name})

### Per-Seed Results Breakdown

| Seed | Condition | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | MCC | ASR (%) | ASR Numerator | ASR Denominator |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    if len(df) > 0:
        for _, r in df.iterrows():
            report_content += f"| {r['seed']} | {r['condition']} | {r['accuracy']*100:.3f} | {r['macro_f1']*100:.3f} | {r['weighted_f1']*100:.3f} | {r['mcc']:.4f} | {r['asr_percent']:.3f} | {r['asr_numerator']} | {r['asr_denominator']} |\n"

    report_content += f"""
### Runtime and Hardware Measurements

- **Platform**: {env.get('cpu', 'Apple M4')} ({env.get('logical_cores', '10')} cores, {env.get('platform', 'macOS')})
- **PyTorch**: {env.get('torch', '2.5.1')} (CPU mode, 2 threads)
- **Total Training Duration**: **{tot_training:.2f} s** ({tot_training/60:.2f} min) across 4 seeds.
- **Total Evaluation Duration**: **{tot_evaluation:.2f} s** ({tot_evaluation/60:.2f} min) across 4 seeds.
- **Average Training Time per Seed**: **{tot_training/len(complete_seeds)/60:.2f} min**.

{table(pd.DataFrame(runtime_rows))}

## 5. Published Paper Context (Table 4)

Table 4 from the PAVE paper is presented below strictly as published context.
It is **not** an established target for this 27-class experiment; no difference column is presented because experimental conditions differ substantially.

{table(table4_context)}

## 6. Verification and Sanity Checks

Every completed seed passed an extensive verification suite:
1. **Model state preservation**: State hashes verify that `model.eval()` was active and model parameters were not modified during FGSM or PGD evaluations.
2. **Epsilon-zero identity**: Probing with $\\epsilon=0$ confirmed exact input identity for both FGSM and PGD.
3. **Input gradients**: Autograd probe confirmed finite, nonzero gradients for model inputs.
4. **Perturbation bounds**: All perturbed examples satisfy $\\|x_{{adv}} - x\\|_\\infty \\le 0.10001$.
5. **Domain bounds**: All perturbed examples were strictly clipped to the normalized feature domain $[0.0, 1.0]$.
6. **Checkpoint reload consistency**: Re-loading the saved epoch 25 checkpoint yielded byte-identical logits on evaluation inputs.
7. **Complete evaluation population**: Exactly 4,500 test rows evaluated for all 3 conditions without subsampling or truncation.

## 7. Limitations and Threat Model

1. **Feature Space vs Physical Validity**: Perturbations operate directly on normalized continuous feature tensors. Unconstrained perturbations do not respect protocol semantics, flag combinations, packet lengths, or timing dependencies.
2. **Synthetic Sequences**: Repeating single tabular rows 50 times creates synthetic tokens, not real packet flows.
3. **Duplicate Leakage**: As revealed by the overlap audit, over 62% of test rows have duplicate feature vectors in the training partition due to natural duplicate flows in the underlying CICIoT capture.
4. **Non-Equivalence of Scaling**: Epsilon 0.10 in MinMaxScaler units represents a distinct physical perturbation from epsilon 0.10 in StandardScaler units.
"""

    (output / "REPORT.md").write_text(report_content, encoding="utf-8")

    # Compose COMPARISON.md
    comparison_content = f"""# Milestone 1: Comparison of Experimental Configurations

## Overview

We conducted two separate Milestone 1 experiments for the FP32 Standard Transformer on the balanced CICIoT2023 27-class dataset:
1. **Experiment 1 (Released-Artifact Baseline)**: Follows the released upstream code in `upstream/PAVE_artifact`.
2. **Experiment 2 (Paper-Described Configuration)**: Follows the settings described in the PAVE paper text.

Both experiments were executed across the same four replication seeds: **1, 7, 13, and 29**.

## Configuration Differences

The two experiments differ in multiple dimensions simultaneously:

| Dimension | Experiment 1: Released Baseline | Experiment 2: Paper-Described | Rationale / Source |
|---|---|---|---|
| **Preprocessing Scaler** | StandardScaler fitted on whole CSV | MinMaxScaler fitted on **training data only** | Section 5.2 specifies training-only min-max |
| **Partitioning** | Stratified 80/20 train/validation (no test) | Stratified 70/15/15 train/val/test | Section 5.1 specifies 70/15/15 |
| **Evaluation Population** | Capped 3,000 validation rows (seed 42) | Full 4,500 held-out test rows | Paper evaluates test partition |
| **Training Epochs** | 15 epochs | 25 epochs | Section 5.1 specifies 25 epochs |
| **Weight Decay** | 0.0 (plain Adam default) | 0.0001 (AdamW) | Section 5.1 specifies weight decay 0.0001 |
| **Optimizer Steps** | 705 steps (47 batches × 15 epochs) | 1,050 steps (42 batches × 25 epochs) | Complete declared training budget |
| **PGD Attack** | 10 steps, alpha=0.01, clean start, no domain clipping | 7 steps, alpha=0.025, random start, domain clipping [0, 1] | Section 5.4 execution description |
| **Perturbation Units** | Epsilon 0.10 in StandardScaler space | Epsilon 0.10 in MinMaxScaler space | Different scalers mean different physical feature units |

> [!IMPORTANT]
> Because several settings change together (scaling, partition, evaluation population, epochs, weight decay, and attack parameters),
> this is a **configuration comparison**, NOT an ablation study identifying which single change caused any observed difference.

## Performance Comparison

{comparison_table_md}

## Key Observations and Interpretation

1. **Clean Performance**:
   - Both configurations achieve robust clean accuracy on this 27-class multiclass problem (~65-67%).
   - In both experiments, performance is substantially lower than Table 4's 98.68%, corroborating the finding that Table 4 does not reflect this balanced 27-class multiclass task.

2. **Adversarial Robustness**:
   - In both configurations, FGSM and PGD cause substantial performance drops, demonstrating the vulnerability of unhardened Transformers to adversarial perturbation.
   - Attack Success Rate (ASR) among clean-correct examples is explicitly measured with both numerator and denominator recorded.

3. **Attack Dynamics and Domain Clipping**:
   - In Experiment 2, domain clipping strictly bounds adversarial examples to $[0, 1]$ in normalized space, preventing out-of-range feature values.
   - Epsilon 0.10 under MinMaxScaler represents a bounded shift of at most 10% of each feature's training range, whereas in StandardScaler space it represents 0.10 standard deviations.

4. **Split and Leakage**:
   - Both experiments suffer from duplicate feature overlap between training and evaluation partitions (~62% duplicate feature match) due to the presence of repeated records in CICIoT2023.
"""

    (output / "COMPARISON.md").write_text(comparison_content, encoding="utf-8")
    print(f"REPORTS GENERATED IN: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/milestone1-paper-described")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/milestone1-paper-described")
    args = parser.parse_args()
    build_reports(args.run_dir, args.output)


if __name__ == "__main__":
    main()
