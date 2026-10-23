"""Generate compact aggregate reporting for completed corrected Adam seeds."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd


def summarize(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for seed_dir in sorted(run_dir.glob("seed-*")):
        if (seed_dir / "manifest.json").is_file():
            frames.append(pd.read_csv(seed_dir / "metrics.csv"))
    if len(frames) != 4:
        raise ValueError(f"Expected four verified seed bundles, found {len(frames)}")
    per_seed = pd.concat(frames, ignore_index=True)
    rows = []
    for condition, group in per_seed.groupby("condition", sort=False):
        for name in ("accuracy", "macro_f1", "weighted_f1", "mcc", "asr_percent"):
            values = [float(v) for v in group[name].dropna()]
            rows.append({"condition": condition, "metric": name, "n": len(values),
                         "mean": statistics.mean(values),
                         "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                         "units": "percent" if name == "asr_percent" else "fraction"})
    return per_seed, pd.DataFrame(rows)


def render(run_dir: Path, output: Path) -> None:
    per_seed, summary = summarize(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    lines = [
        "# Corrected paper-described Adam experiment",
        "",
        "Variant C uses conventional `torch.optim.Adam`; it is not an exact "
        "reproduction of Table 4. Historical AdamW results remain variant B.",
        "",
        "| Condition | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | MCC | ASR (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("Clean", "FGSM", "PGD"):
        sub = summary[summary.condition == condition].set_index("metric")
        def fmt(name, scale=1):
            r = sub.loc[name]
            return f"{r['mean'] * scale:.3f} +/- {r['sample_std'] * scale:.3f}"
        lines.append(f"| {condition} | {fmt('accuracy', 100)} | {fmt('macro_f1', 100)} | "
                     f"{fmt('weighted_f1', 100)} | {fmt('mcc')} | {fmt('asr_percent')} |")
    lines += [
        "",
        "The test population is the shared 4,500-row saved split. ASR uses the "
        "clean-correct denominator. Duplicate overlap, attack policy, smoke "
        "exposure, environment, and unresolved Table 4 correspondence are "
        "documented in the protocol and correction notes.",
        "",
        "## Evidence scope",
        "",
        "Included prediction tables and manifests support metric and alignment "
        "recomputation. Checkpoint-dependent claims require the local run "
        "directory; historical AdamW checkpoint claims remain unavailable here.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "COMPARISON.md").write_text(
        "# Milestone 1 comparison\n\n"
        "Variant A is the released-code baseline, variant B is the historical "
        "AdamW configuration, and variant C is conventional Adam. A/B/C differ "
        "in more than optimizer and evaluation population, so this is not an "
        "optimizer-only ablation. Published Table 4 values are context only.\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    render(args.run_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
