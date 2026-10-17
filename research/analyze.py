"""Summarize only measured, completed units. Smoke/pilot are labeled explicitly."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import atomic_json
from .validate import validate_run


def analyze(root, phase, output):
    root, output = Path(root), Path(output)
    validate_run(root, phase)
    if output.exists():
        raise FileExistsError(f"Use a new analysis directory: {output}")
    output.mkdir(parents=True)
    cfg = json.loads((root / "run.json").read_text())["config"]
    partitions = ["validation", "test"] if phase == "both" else [phase]
    rows, class_rows, transfers, bench_rows = [], [], [], []
    for partition in partitions:
        for file in sorted((root / partition).glob("evaluation/s*/*/metrics.json")):
            for row in json.loads(file.read_text()):
                rows.append({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
                for label, values in row["classwise"].items():
                    if isinstance(values, dict) and label not in ("macro avg", "weighted avg"):
                        class_rows.append({"partition": partition, "variant": row["variant"], "seed": row["seed"],
                                           "attack": row["attack"], "epsilon": row["epsilon"], "label": label, **values})
                if row["attack"] == "clean":
                    fig, ax = plt.subplots(figsize=(9, 8))
                    matrix = np.asarray(row["confusion_matrix"])
                    image = ax.imshow(matrix, cmap="Blues")
                    labels = list(row["classwise"])[:len(matrix)]
                    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
                           xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True",
                           title=f"{cfg['tier'].upper()} {partition} {row['variant']} s{row['seed']} n={row['n']}")
                    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
                    plt.setp(ax.get_yticklabels(), fontsize=6)
                    fig.colorbar(image, ax=ax, shrink=0.6)
                    fig.tight_layout()
                    fig.savefig(output / f"confusion_{partition}_{row['variant']}_s{row['seed']}.png", dpi=140)
                    plt.close(fig)
            bench = json.loads((file.parent / "benchmark.json").read_text())
            for b in bench["measurements"]:
                bench_rows.append({"partition": partition, "seed": int(file.parent.parent.name[1:]),
                                   "variant": file.parent.name, "serialized_state_bytes": bench["serialized_state_bytes"],
                                   **{k: v for k, v in b.items() if not isinstance(v, (dict, list))}})
        for file in sorted((root / partition).glob("transfer/s*/*/metrics.json")):
            transfers += [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in json.loads(file.read_text())]
    df = pd.DataFrame(rows)
    df.to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "classwise.csv", index=False)
    pd.DataFrame(transfers).to_csv(output / "transfers.csv", index=False)
    pd.DataFrame(bench_rows).to_csv(output / "efficiency.csv", index=False)
    group = ["partition", "variant", "attack", "epsilon"]
    aggregated = df.groupby(group)[["accuracy", "robust_accuracy", "macro_f1", "attack_success_rate"]].agg(["count", "mean", "std"])
    aggregated.to_csv(output / "mean_std.csv")
    paired = df.pivot(index=["partition", "seed", "attack", "epsilon"], columns="variant", values="robust_accuracy")
    for a, b in (("fp32", "ptq_all_linear"), ("fp32", "at_fp32"), ("at_fp32", "at_ptq_all_linear")):
        paired[f"delta_{b}_minus_{a}"] = paired[b] - paired[a]
    paired.to_csv(output / "paired_seed_differences.csv")
    for partition in partitions:
        for kind in ("fgsm", "pgd"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for variant in ("fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear"):
                part = df[(df.partition == partition) & (df.attack == kind) & (df.variant == variant)]
                grouped = part.groupby("epsilon").robust_accuracy
                mean, std = grouped.mean(), grouped.std()
                ax.plot(mean.index, mean.values, marker="o", label=variant)
                if len(cfg["seeds"]) > 1:
                    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
            n = int(df[df.partition == partition].iloc[0].n)
            ax.set(xlabel="L-inf epsilon (training-standardized feature units)", ylabel="Robust accuracy",
                   title=f"{cfg['tier'].upper()}: {kind.upper()} {partition}, n={n}, seeds={len(cfg['seeds'])}", ylim=(0, 1))
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(output / f"robustness_{partition}_{kind}.png", dpi=160)
            plt.close(fig)
    text = [f"# {cfg['tier'].upper()} measured results", "",
            f"Training seeds: {cfg['seeds']}; split seed: {cfg['split_seed']}.",
            "Smoke/pilot results test implementation and runtime, not scientific effectiveness.",
            "One seed has no estimable between-seed standard deviation; CSV std cells are blank.",
            "ASR is undefined (null/blank) if no target examples are correctly classified clean.",
            "Robust accuracy counts examples correct on both clean and perturbed input.",
            "Macro-F1 and confusion matrices use adversarial predictions; raw matched IDs are in predictions.npz.",
            "The balanced processed subset cannot establish natural minority-class prevalence.", "",
            "| Partition | Variant | Seed | n | Clean accuracy | Macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in df[df.attack == "clean"].itertuples():
        text.append(f"| {row.partition} | {row.variant} | {row.seed} | {row.n} | {row.accuracy:.4f} | {row.macro_f1:.4f} |")
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    atomic_json(output / "source.json", {"run": str(root.resolve()), "tier": cfg["tier"], "phase": phase})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--phase", default="both", choices=("validation", "test", "both"))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.run_dir, a.phase, a.output)


if __name__ == "__main__":
    main()
