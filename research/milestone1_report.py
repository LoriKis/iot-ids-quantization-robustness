"""Build the local review package exclusively from verified completed seeds.

Incomplete seeds are listed as incomplete and are excluded from every aggregate.
"""
from __future__ import annotations
import argparse
import datetime
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .artifact_baseline import CLAIM, ROOT, read_config, verify_seed, atomic_json, sha


def table(frame):
    # Avoid an optional tabulate dependency in the project's frozen environment.
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def build(run, output):
    cfg = read_config(run / "configuration.json")
    expected = cfg["seeds"]
    complete, incomplete, frames = [], {}, []
    for seed in expected:
        folder = run / f"seed-{seed}"
        if (folder / "manifest.json").exists():
            frames.append(verify_seed(folder, cfg))
            complete.append(seed)
        else:
            status = folder / "status.json"
            incomplete[str(seed)] = json.loads(status.read_text()) if status.exists() else {"state": "not completed; no epoch checkpoint status yet"}
    output.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    adapter_hashes = {str(s): sha(run / f"seed-{s}/training_adapter.py") for s in complete}
    if len(set(adapter_hashes.values())) > 1:
        raise ValueError("Training adapter source differs across completed seeds; inspect before aggregating")
    df.to_csv(output / "per_seed.csv", index=False)
    metrics = ["accuracy", "macro_f1", "weighted_f1", "mcc", "asr_percent"]
    aggregate = []
    for name in ["Clean", "FGSM", "PGD"]:
        for metric in metrics:
            values = df.loc[df.condition == name, metric].dropna().to_numpy() if len(df) else np.array([])
            aggregate.append({"condition": name, "metric": metric, "n": len(values), "mean": float(values.mean()) if len(values) else None, "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None, "units": "percent" if metric == "asr_percent" else "fraction"})
    summary = pd.DataFrame(aggregate)
    summary.to_csv(output / "summary.csv", index=False)
    runtimes = {str(s): json.loads((run / f"seed-{s}/runtime.json").read_text()) for s in complete}
    runtime_rows = [{"Seed": seed, "Training (min)": f"{r['training_seconds_accumulated']/60:.2f}", "Evaluation (min)": f"{sum(r['evaluation_seconds'].values())/60:.2f}", "Peak process RSS (GiB)": f"{r['peak_rss_sampled_bytes_this_invocation']/2**30:.2f}", "Min system available (MiB)": f"{r['minimum_system_available_bytes_this_invocation']/2**20:.1f}"} for seed,r in runtimes.items()]
    training = sum(r["training_seconds_accumulated"] for r in runtimes.values())
    evaluation = sum(sum(r["evaluation_seconds"].values()) for r in runtimes.values())
    environment = json.loads((run / "environment.json").read_text())
    wall_span = (datetime.datetime.fromisoformat(max(r["completed_utc"] for r in runtimes.values()).replace("Z", "+00:00")) - datetime.datetime.fromisoformat(environment["started_utc"].replace("Z", "+00:00"))).total_seconds() if runtimes else None
    atomic_json(output / "completion.json", {"claim": CLAIM, "expected_seeds": expected, "verified_completed_seeds": complete, "incomplete": incomplete, "four_seed_package_complete": complete == expected, "training_seconds": training, "evaluation_seconds": evaluation, "wall_span_seconds_to_last_completed_seed": wall_span, "full_run_location": str(run), "summary_std": "sample SD, ddof=1; unavailable for fewer than two seeds"})
    shutil.copy2(run / "configuration.json", output / "configuration.json")
    shutil.copy2(run / "environment.json", output / "environment.json")
    atomic_json(output / "provenance.json", {"claim": CLAIM, "upstream_revision": cfg["upstream_revision"], "dataset_sha256": cfg["dataset"]["sha256"], "seed_manifests": {str(s): sha(run / f"seed-{s}/manifest.json") for s in complete}, "checkpoint_locations": {str(s): str(run / f"seed-{s}/checkpoint.pt") for s in complete}, "checkpoint_sha256": {str(s): sha(run / f"seed-{s}/checkpoint.pt") for s in complete}, "training_adapter_sha256": adapter_hashes, "initial_executor_snapshot_sha256": sha(run / "executor_snapshot.py"), "report_generator_sha256": sha(Path(__file__)), "configuration_sha256": sha(run / "configuration.json"), "paper_source": str(ROOT.parent / "outputs/milestone1-reference-review/evidence/PAVE-source.pdf"), "paper_sha256": "be81c110fd12901c8be6455b217ba53ca942c1b0bb9c52bb5dda84bf655b6e90"})
    display_rows = []
    for name in ["Clean", "FGSM", "PGD"]:
        row = {"Condition": name, "Seeds": len(complete)}
        for metric, label in [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1 (%)"), ("weighted_f1", "Weighted-F1 (%)"), ("mcc", "MCC"), ("asr_percent", "ASR (%)")]:
            rec = summary[(summary.condition == name) & (summary.metric == metric)].iloc[0]
            factor = 1 if metric in ["mcc", "asr_percent"] else 100
            mean, std = rec["mean"], rec.sample_std
            row[label] = "not completed" if pd.isna(mean) else f"{mean*factor:.3f} ± {std*factor:.3f}" if pd.notna(std) else f"{mean*factor:.3f} (SD unavailable)"
        display_rows.append(row)
    comparisons = []
    for name, paper_acc, paper_f1, paper_asr in [("Clean", 98.68, 98.63, None), ("FGSM", 98.19, 97.95, .81), ("PGD", 98.19, None, .81)]:
        ours = summary[(summary.condition == name) & (summary.metric == "accuracy")]["mean"].iloc[0]
        comparisons.append({"Condition": name, "Paper accuracy (%)": paper_acc, "Our accuracy (%)": f"{100*ours:.3f}" if pd.notna(ours) else "not completed", "Descriptive difference (pp)": f"{100*ours-paper_acc:+.3f}" if pd.notna(ours) else "not available", "Paper F1 (averaging unverified)": paper_f1 if paper_f1 is not None else "not reported", "Paper ASR (denominator unverified)": paper_asr if paper_asr is not None else "—"})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "paper_comparison.csv", index=False)
    fig_name = "results.png"
    if complete:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        colors = ["#246b91", "#d88427", "#ae4247"]
        for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
            rows = summary[summary.metric == metric].set_index("condition").loc[["Clean", "FGSM", "PGD"]]
            means = rows["mean"].to_numpy(dtype=float)*100
            errors = pd.to_numeric(rows.sample_std, errors="coerce").fillna(0).to_numpy(dtype=float)*100
            ax.bar(["Clean", "FGSM ε=.10", "PGD ε=.10"], means, yerr=errors if len(complete)>1 else None, capsize=5, color=colors, width=.65)
            for i, value in enumerate(means): ax.text(i, value+2, f"{value:.2f}", ha="center", fontsize=9)
            ax.set(ylim=(0, 105), ylabel=title+" (%)", title=title)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(f"Released-artifact multiclass baseline · n={len(complete)} completed seeds\n" + ("Error bars: sample SD across seeds (ddof=1)" if len(complete)>1 else "One completed seed; variability not yet available"), fontsize=11)
        fig.savefig(output / fig_name, dpi=180)
        plt.close(fig)
    statuses = "Four full seeds completed and verified." if complete == expected else f"INCOMPLETE: completed seeds {complete}; outstanding {[s for s in expected if s not in complete]}. No unfinished run is counted as a result."
    attack_order = []
    for seed in complete:
        values = df[df.seed == seed].set_index("condition").accuracy
        if values["PGD"] > values["FGSM"] + 1e-12:
            attack_order.append(seed)
    attack_note = f"PGD has higher accuracy than FGSM for seed(s) {attack_order}. This is recorded as an attack-strength sanity concern. The released clean-start PGD retains only its final iterate and does not select the stronger of FGSM/PGD candidates. We preserve the behavior and evidence, without repairing it or claiming worst-case robustness." if attack_order else "No completed seed has PGD accuracy higher than FGSM accuracy. This ordering check alone does not establish a strongest-attack or worst-case robustness guarantee."
    labels = pd.DataFrame([{"ID": i, "Label": name, "CSV rows": cfg['dataset']['class_counts'][name]} for i,name in enumerate(cfg['dataset']['class_names'])])
    epochs = {str(s): json.loads((run / f"seed-{s}/training_complete.json").read_text()) for s in complete}
    checklist = [
        ["1. Clone PAVE; read README and paper", "Completed with evidence", "Pinned e73911b, full local PDF and previous reference investigation"],
        ["2. Environment and processed CICIoT data", "Completed with documented limitation", "Hashed released 27-class balanced subset, not the raw 33-attack dataset composition"],
        ["3. Standard Transformer multiclass pipeline", "Completed with evidence" if complete == expected else "Not completed for all required seeds", f"Verified full-budget seeds: {complete}"],
        ["4. Reproduce published clean results", "Unresolved / not demonstrated", "Numerical reproduction under matched conditions has NOT been established"],
        ["5. Provided FGSM and PGD evaluation", "Completed with evidence" if complete == expected else "Not completed for all required seeds", "Same saved final checkpoint and row IDs; unchanged upstream attack functions"],
        ["6. Compare with published results", "Completed with documented limitation" if complete else "Not completed: awaiting first measured seed", "Descriptive comparison only; potentially different experiments"],
        ["7. Explain architecture, pipeline, attacks and validity", "Completed with evidence", "Explanations and executed configuration below"]
    ]
    source = ROOT.parent / "outputs/milestone1-reference-review/evidence/PAVE-source.pdf"
    if sha(source) != "be81c110fd12901c8be6455b217ba53ca942c1b0bb9c52bb5dda84bf655b6e90":
        raise ValueError("Unexpected local paper source")
    report = f'''# Milestone 1: released-artifact multiclass baseline

**{CLAIM}**

## Objective, scope and completion

Run the ordinary FP32 PAVE Standard Transformer on released CICIoT data, evaluate
clean (unperturbed), FGSM and PGD inputs, and document the attempted paper comparison.
Clean includes unperturbed examples from **all 27 classes**, including malicious
traffic. Clean does not mean benign.

{statuses}

Exact Table 4 reproduction remains unresolved. This does not block the authorized
artifact attempt. No quantization, QAT, adversarial training, transfer experiment,
layer ablation, remote write or external compute is part of this work.

## Provenance and data

- Upstream: https://github.com/robotumbel/PAVE_artifact, revision `{cfg['upstream_revision']}`.
- Dataset: `{cfg['dataset']['path']}`, SHA-256 `{cfg['dataset']['sha256']}`.
- Entire CSV: 29,997 rows, 39 numeric features, 27 classes, 1,111 rows per class.
  Benign_Final is 3.7037% of this subset. This balanced subset does not establish
  natural traffic prevalence or natural minority-class behavior.
- Full feature order and alphabetical class mapping are frozen in `configuration.json`.
- Interpreter, package versions, CPU, RAM and thread count: `environment.json`.
  Local CPU only, two PyTorch threads; no concurrent training jobs.
- Paper: supplied pre-proof, DOI https://doi.org/10.1016/j.hcc.2026.100426,
  Table 4 printed p.12 / PDF p.16. The seven values were rechecked against the
  existing local extraction; the source PDF hash is recorded in `provenance.json`.
- Source/called-function hashes: `configuration.json`; checkpoint and per-seed
  manifest hashes/locations: `provenance.json`. Raw evidence remains in `{run}`.

{table(labels)}

## Architecture and input representation

A CSV row has 39 features. The artifact standardizes it and repeats it into a
synthetic sequence of 50 tokens. Training adds independent Gaussian noise with
standard deviation .01 at each time step. Evaluation repeats the row without noise.
These are not chronological observations of 50 real flows.

The unchanged upstream `StandardTransformerIDS` maps 39 features to width 96,
adds fixed sinusoidal positional encoding and dropout .1, then applies three
post-norm residual Transformer encoder blocks. Each has four attention heads
(24 dimensions each), Q/K/V/output linear projections, and a ReLU feed-forward
network of widths 96→384→96. Mean pooling over tokens feeds the classifier
96→48→27 logits. Cross-entropy trains the classifier; argmax selects the label.
The upstream 96→64→1 anomaly scorer remains instantiated but receives no auxiliary
loss in this ordinary baseline. The complete model has 351,612 parameters.

## Executed pipeline

1. Use all CSV rows; numeric coercion and missing/infinite values are replaced by zero.
2. Fit the released **StandardScaler on the entire CSV before splitting**. Retain
   this inherited methodological limitation, rather than silently substituting min–max.
3. Use upstream stratified `train_test_split(test_size=.20, random_state=seed)`:
   23,997 training rows and 6,000 validation rows. No independent test partition exists.
4. Construct the upstream synthetic training sequences with the replication seed.
5. Train ordinary FP32 for 15 epochs, batch 512 (including final short batch),
   Adam lr .001, default betas .9/.999, eps 1e-8, **zero weight decay** and ordinary
   cross-entropy. No scheduler, gradient clipping, adversarial samples or tuning.
6. Save the **final epoch 15**, without selecting it using evaluation performance.
   Each completed seed must contain 705 optimizer steps and 359,955 sample exposures.
7. Select 3,000 validation rows using unchanged `cap_aligned` with cap seed 42.
   Its per-class sampling plus remainder fill is preserved. Use exactly those row IDs
   for all three conditions and keep model.eval() during attacks and inference.
8. Use upstream attacks and inference with evaluation batch 32 to limit memory.
   This uses their supported `bs` argument; training batch remains 512.

Replication seeds are **1, 7, 13, 29**, predetermined before training; they are not
a claim about the paper's seeds. Python/numpy/torch training RNGs are explicitly
seeded as our addition. The same replication seed controls the inherited split
and sequence-noise generation. Variation therefore includes both training
randomness and changing partitions/evaluation examples. The cap seed stays 42,
but operates on each seed's different validation population.

The upstream `train_seq` statements are reused through AST instrumentation that
only adds an epoch range, optimizer/RNG restore and an end-of-epoch callback.
The optimizer, loss, shuffle, dropout and update statements are unchanged. Added
checkpoints save model, Adam state and CPU torch RNG after each complete epoch.
Tests establish exact parameter equality against uninterrupted upstream training,
including interruption/resume. Atomic saves and logs are our additions. Skipping
unneeded surrogate/AGT/GRU training also changes RNG consumption relative to the
artifact's mixed-model scripts; we disclose that isolation and seed the baseline.

## FGSM, PGD and metric definitions

FGSM takes one gradient-sign step that increases true-label cross-entropy:

`x_adv = x + ε sign(∇x CE(f(x), y))`, with `ε = .10`.

PGD repeatedly takes smaller steps, projecting each result to the epsilon box:

`x⁰ = x`

`xᵗ⁺¹ = clip(xᵗ + .01 sign(∇xᵗ CE(f(xᵗ), y)), x−.10, x+.10)`, for 10 steps.

These are untargeted attacks in the standardized **sequence tensor**. PGD starts
from the clean input without random initialization. Neither upstream function
clips to feature-domain boundaries or projects to protocol-valid traffic. Epsilon
.10 is in StandardScaler units, not ten percent of each raw feature's range.
Model parameters are unchanged during evaluation; input gradients are enabled.
Perturbation checks permit 1e-5 float32 roundoff, and retain each condition's
measured maximum. An observed maximum such as .10000038 is rounding at epsilon
.10, not an intentional larger attack budget.

- Accuracy: proportion of correctly classified evaluated rows.
- Macro-F1: arithmetic mean of all 27 per-class F1 values, zero_division=0.
- Weighted-F1: mean of class F1 values weighted by evaluation support.
- MCC: sklearn multiclass Matthews correlation.
- Untargeted ASR (%): 100 × newly incorrect examples among clean-correct examples
  divided by the number of clean-correct examples. Empty denominator → null.
  A wrong attack-family prediction counts; this is not a targeted
  malicious-to-benign evasion rate.
- Upstream `pave_audit._asr` is saved separately; its nonempty formula agrees,
  while its empty-denominator fallback returns zero. Prediction-change rate and
  total adversarial error are also separate diagnostics, not paper-equivalent ASR.

Metrics are saved without the upstream four-decimal rounding. Per-class
precision/recall/F1/support, confusion matrices, labels, predictions/logits and
row IDs are retained. Sample SD uses **ddof=1** across completed seeds; fewer
than two seeds cannot establish variability.

## Measured results

{table(pd.DataFrame(display_rows))}

Only completed, verified full-budget seeds enter this table and the figure.
`per_seed.csv` contains individual measurements; `summary.csv` contains unrounded
means and sample SD. Missing seeds are not filled with zeros or smoke results.

{f'![Measured clean/FGSM/PGD results]({fig_name})' if complete else 'No results figure is generated before a full-budget seed completes.'}

Completed-seed training time: **{training:.2f} seconds**. Completed evaluation
time: **{evaluation:.2f} seconds**. These are measured accumulated training and
evaluation durations, not estimates of uncompleted seeds; runtime records also
include per-invocation loading time and sampled peak RSS. Epoch checkpoints retain
progress if the session ends; `completion.json` lists outstanding work.
Wall-clock span from initial run setup to the last completed seed is
{f'{wall_span:.0f} seconds' if wall_span is not None else 'not available before a seed completes'};
it includes loading, verification and any idle gaps between sequential launches.

{table(pd.DataFrame(runtime_rows)) if runtime_rows else 'No completed-seed runtime table yet.'}

Memory values are sampled, not exact allocation peaks. System-available memory
includes the effect of other applications; it is distinct from this process's RSS.
The run uses one training process, two CPU threads and evaluation batches of 32.

## Verification and failures

Every included seed passed dataset/hash/class/model checks, full-budget checkpoint
validation, exact checkpoint-reload logit equality over the evaluation population,
aligned examples across conditions, finite recomputed accuracy/F1/MCC, epsilon
bounds and epsilon-zero identity for actual upstream FGSM/PGD. An explicit input
gradient probe is finite/nonzero; state hashes establish no parameter updates
during attacks. Saved prediction coverage, confusion matrices, per-class support,
checkpoint identity and artifact checksums are verified. A process exit code alone
is not accepted as completion. Raw logs and any unfinished checkpoint/status files
remain available; incomplete work is visible in `completion.json`.

{attack_note}

## Paper comparison and limits

{table(comparison)}

Accuracy differences above are **descriptive differences between potentially
different experiments**, not reproduction-error estimates. Paper F1 is not labelled
macro-F1, and the paper's ASR denominator is not established; no F1 or ASR difference
is presented as a matched metric comparison. Table 4 does not prove a multiclass
target. Its binary context, mode ambiguity, metric definitions, seed aggregation,
dataset/split provenance and checkpoint rule remain unresolved.

Established differences (see `source/docs/MILESTONE1_REFERENCE.md` in the ZIP):

| Item | Paper | Executed released artifact |
|---|---|---|
| CICIoT composition | approximately 60% benign after sampling; exact subset unknown | 27 equal classes, 3.70% benign |
| Scaling | training-only min–max | whole-file StandardScaler before split |
| Partition | stratified 70/15/15 | stratified 80/20, capped validation evaluation |
| Training | Adam .001, decay .0001, 25 epochs | Adam .001, no decay, 15 epochs |
| PGD | Section 5.4: 7 steps, alpha .025, random start; Section 3.3 conflicts on alpha | 10 steps, alpha .01, clean start |
| Domain clipping | paper describes normalized feature-bound clipping | epsilon-box projection only |
| F1 and ASR | Table 4 definitions/generator unverified | explicit macro/weighted F1 and clean-correct ASR |

The artifact's execution does not establish absence of leakage. Scaling uses
validation information, and random splitting does not separate duplicate feature
groups. The previous audit found 18,044 repeated feature rows, 17,721 repeated
labeled rows and 229 conflicting-label feature groups. Reproducing those choices
retains them; a high accuracy alone would not validate their generalization.

Unconstrained feature-space attacks may create fractional flags, negative counts,
out-of-range protocol values or changes to immutable statistics. Valid protocol
traffic has integer, range and immutability constraints, and sometimes joint-field
relationships. No packet or valid traffic trace is constructed by these attacks.
Their measured robustness concerns normalized feature tensors. PAVE's validity
audit explains why apparent feature-space damage may overstate realizable attacks;
this baseline report does not claim protocol-valid robustness or measure a new
validity rate.

The later corrected research protocol is separate: duplicate-group separation,
training-only scaling and distinct validation/test handling address different
questions. None of those changes is silently inserted into this artifact attempt.
This report does not launch Milestone 2 or claim literal exact paper reproduction.

## Official Milestone 1 checklist

{table(pd.DataFrame(checklist,columns=['Requirement','Status','Evidence/limitation']))}

The practical four-seed baseline package is {'ready for review, with the published-result requirement explicitly unresolved' if complete == expected else 'NOT ready as a completed four-seed package; resume outstanding work'}.

## Exact local commands and excluded artifacts

Run from `{ROOT}` using its existing environment:

```powershell
.\\.venv\\Scripts\\python.exe -B -m research.artifact_baseline --run-dir runs/milestone1-artifact-full --resume
.\\.venv\\Scripts\\python.exe -B -m research.artifact_baseline --run-dir runs/milestone1-artifact-full --verify-only
.\\.venv\\Scripts\\python.exe -B -m research.milestone1_report --run-dir runs/milestone1-artifact-full --output reports/milestone1-artifact --export ../outputs/milestone1-practical-review.zip
```

The first command reuses verified completed seeds and resumes the last complete
epoch of incomplete seeds, restoring Adam/RNG state. It never reduces the full
budget to smoke. For an independent rerun, use a new run directory with the same
frozen config. The unchanged pinned checkout and hashed CSV are required; the
existing setup instructions explain restoring those excluded dependencies.

Large data/checkpoints, environments and package caches are excluded from the
review ZIP. Original checkpoint/preprocessing/evaluation files remain under
`{run}`; exact paths/hashes are in `provenance.json`. The ZIP includes source,
frozen config, this report, summaries/figure, small per-seed evidence and an export
manifest. `reference-source/` is an inspection copy of the upstream source, not a
replacement Git checkout. Existing environments, runs and local commits are retained.
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return complete


def export_package(run, output, destination):
    # Assemble a fresh staging directory; never traverse ignored data/environments.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="m1-review-", dir=ROOT / "work") as tmp:
        stage = Path(tmp)
        source = stage / "source"
        tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().strip("\0").split("\0")
        for name in tracked:
            path = ROOT / name
            if path.is_file():
                dest = source / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        shutil.copytree(ROOT / "upstream/PAVE_artifact/code", stage / "reference-source/code", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "upstream/PAVE_artifact/LICENSE", stage / "reference-source/LICENSE")
        for path in output.iterdir():
            if path.is_file(): shutil.copy2(path, stage / path.name)
        shutil.copytree(output, source / "reports/milestone1-artifact", dirs_exist_ok=True)
        validation = ROOT / "work/milestone1-full-preflight.xml"
        if validation.exists():
            (stage / "validation").mkdir(exist_ok=True)
            shutil.copy2(validation, stage / "validation/preflight.xml")
        for folder in sorted(run.glob("seed-*")):
            for name in ["metrics.csv", "metrics.json", "per_class.csv", "predictions.csv", "split_ids.json", "preprocessing.json", "preprocessing.npz", "runtime.json", "verification.json", "training_complete.json", "training_adapter.py", "raw.log", "status.json", "manifest.json", "confusion_matrices.npz"]:
                path = folder / name
                if path.is_file():
                    dest = stage / "results" / folder.name / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
        references = ["research/artifact_baseline.py", "research/milestone1.py", "research/milestone1_report.py", "configs/milestone1_artifact_full.json", "configs/milestone1_target.json", "docs/MILESTONE1_REFERENCE.md", "README.md", "PROGRESS.md"]
        for name in references:
            if not (source / name).is_file():
                raise ValueError(f"Referenced project file missing from source export: {name}; stage/commit new source first")
        manifest = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        atomic_json(stage / "export_manifest.json", {"files_sha256": manifest, "local_git_revision_at_export": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "local_dirty_status_at_export": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip(), "referenced_project_files_checked": references, "excluded": {"dataset": "29997-row CSV; existing project path/hash in configuration.json", "checkpoints_and_evaluation_npz": "existing full run directory and regeneration command in REPORT.md/provenance.json", "environments_and_caches": "reuse existing .venv or follow source/docs/SETUP.md"}, "completion": json.loads((output / "completion.json").read_text())})
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".building.zip")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(stage).as_posix())
        with zipfile.ZipFile(temp) as archive:
            assert archive.testzip() is None
            for name, digest in manifest.items():
                import hashlib
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        if destination.exists():
            raise FileExistsError(f"Preserve existing review ZIP; choose a new filename: {destination}")
        temp.replace(destination)
    print(f"Review ZIP verified: {destination} ({destination.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    completed = build(args.run_dir.resolve(), args.output.resolve())
    print(f"Report uses completed seeds only: {completed}")
    if args.export:
        export_package(args.run_dir.resolve(), args.output.resolve(), args.export.resolve())


if __name__ == "__main__":
    main()
