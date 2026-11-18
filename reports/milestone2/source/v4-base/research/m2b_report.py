"""M2B reporting (DESIGN 4.14): tables, robustness curves, per-class table and the efficiency report.

`python -m research.m2b_campaign report --phase validation|test|efficiency` writes under
output/<run>/reports/<phase>/. Every metric is recomputed from the saved per-example predictions.npz files
(accuracy and macro-F1 over all 27 classes) and cross-checked against the unit's attack_result.json or
clean_result.json (a difference above 1e-12 raises). Aggregates are mean +/- sample standard deviation
(np.std(ddof=1)) over seeds; one seed prints "n/a". Pilot outputs are never read: only output/<run>/<phase>/ is scanned.
Every written file names its phase (VALIDATION, FINAL TEST) and the feature-space framing sentence.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np

from .m2b_io import durable_write_bytes, durable_write_json, sha256_file, utc_now

N_CLASSES = 27
TOLERANCE = 1e-12
FRAMING = "feature-space attacks; perturbed vectors need not be valid network traffic"
BANNERS = {"validation": "VALIDATION", "test": "FINAL TEST", "efficiency": "VALIDATION ROWS (EFFICIENCY)"}
BASE_MODELS = ("V1", "V2", "V3", "V4")
QAT_MODELS = ("FP32-FT", "FP32-FT-PTQ", "QAT-FT")
INT8_MODELS = ("V2", "V4", "FP32-FT-PTQ", "QAT-FT")
PRIMARY_SOURCE = {"V1": "fp32_direct", "V3": "fp32_direct", "FP32-FT": "fp32_direct", "V2": "bpda_int8", "V4": "bpda_int8",
                  "FP32-FT-PTQ": "bpda_int8", "QAT-FT": "bpda_int8"}
METRICS = ("accuracy", "macro_f1", "robust_accuracy", "attack_success_rate")
PAIRS = (("V3", "V1"), ("V4", "V2"), ("V4", "V3"), ("V2", "V1"))
QAT_PAIRS = (("FP32-FT", "V1"), ("FP32-FT-PTQ", "V2"), ("QAT-FT", "FP32-FT-PTQ"))
QAT_SENTENCE = ("QAT benefit is the third row only (QAT-FT minus FP32-FT->PTQ). The first two rows are effects of the "
                "extra fine-tuning epochs, not of quantization.")
NO_CLAIM = "No speed or memory advantage is assumed; all numbers are descriptive measurements."
MEMORY_NOTE = "Windows pilot (peak working set) and Linux (VmHWM) memory numbers are not comparable."
WEAK_DISTINCT, WEAK_ROWS = 10, 30
EPS_DIR = re.compile(r"^eps([0-9.eE+-]+)$")


class ReportError(RuntimeError):
    """Missing inputs or a refused report."""


class ReportConsistencyError(ReportError):
    """A value recomputed from predictions.npz differs from the unit's stored result beyond 1e-12."""


# ------------------------------------------------------------------------------------------ metrics
def macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0))


def mean_sd(values):
    """(mean, sample sd with ddof=1, n). sd is None for fewer than two values."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return None, None, 0
    return float(v.mean()), (float(np.std(v, ddof=1)) if v.size > 1 else None), int(v.size)


def fmt(mean, sd, digits=4) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f} ± " + ("n/a" if sd is None else f"{sd:.{digits}f}")


def banner_text(phase: str) -> str:
    return f"{BANNERS[phase]} | {FRAMING}"


def attack_row(y, pred_clean, pred_adv):
    """Recompute one attack unit's row from per-example predictions (definitions of DESIGN 4.10)."""
    y, pc, pa = (np.asarray(a) for a in (y, pred_clean, pred_adv))
    clean_ok, adv_ok = pc == y, pa == y
    robust = clean_ok & adv_ok
    n_clean = int(clean_ok.sum())
    return {"n": int(len(y)), "accuracy": float(adv_ok.mean()), "macro_f1": macro_f1(y, pa),
            "robust_accuracy": float(robust.mean()),
            "attack_success_rate": (int((clean_ok & ~adv_ok).sum()) / n_clean) if n_clean else None}, robust


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOLERANCE


# ------------------------------------------------------------------------------------------ collection
def _eps_of(folder: Path) -> float:
    match = EPS_DIR.match(folder.name)
    if not match:
        raise ReportError(f"unexpected epsilon directory {folder}")
    return float(match.group(1))


def collect(run_dir, phase, models=BASE_MODELS):
    """Scan output/<run>/<phase>/s*/ and return (rows, arrays).

    rows: dicts with the results_long.csv columns. arrays: {(seed, model, source, attack, eps): (y, pred, robust, ids)}.
    """
    run_dir = Path(run_dir)
    if "pilot" in run_dir.parts or "m2a-pilot" in run_dir.parts:
        raise ReportError("pilot outputs are never reported by the campaign report")
    base = run_dir / phase
    rows, arrays = [], {}
    if not base.is_dir():
        raise ReportError(f"no {phase} results under {base}")
    for seed_dir in sorted(base.glob("s*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not seed_dir.name[1:].isdigit():
            continue
        seed = int(seed_dir.name[1:])
        for model_dir in sorted((seed_dir / "clean").glob("*")) if (seed_dir / "clean").is_dir() else []:
            if model_dir.name not in models or not (model_dir / "predictions.npz").is_file():
                continue
            with np.load(model_dir / "predictions.npz") as z:
                ids, y, pred = z["ids"], z["y"], z["pred"]
            acc, f1 = float((pred == y).mean()), macro_f1(y, pred)
            stored = json.loads((model_dir / "clean_result.json").read_text(encoding="utf-8"))
            if not (_close(acc, stored["accuracy"]) and _close(f1, stored["macro_f1"]) and stored["n"] == len(y)):
                raise ReportConsistencyError(f"{model_dir}: accuracy/macro-F1 recomputed from predictions.npz differ from clean_result.json")
            rows.append({"phase": phase, "seed": seed, "model": model_dir.name, "source": "clean", "attack": "clean",
                         "epsilon": 0.0, "n": int(len(y)), "accuracy": acc, "macro_f1": f1, "robust_accuracy": acc,
                         "attack_success_rate": None, "valid": True})
            arrays[(seed, model_dir.name, "clean", "clean", 0.0)] = (y, pred, pred == y, ids)
        attacks = seed_dir / "attacks"
        if not attacks.is_dir():
            continue
        for result_path in sorted(attacks.glob("*/*/*/eps*/attack_result.json")):
            eps_dir = result_path.parent
            attack_dir = eps_dir.parent
            source_dir, model_dir = attack_dir.parent, attack_dir.parent.parent
            model, source, attack = model_dir.name, source_dir.name, attack_dir.name
            if model not in models:
                continue
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            valid = bool(stored.get("valid", False))
            npz = eps_dir / "predictions.npz"
            if not npz.is_file():
                rows.append({"phase": phase, "seed": seed, "model": model, "source": source, "attack": attack,
                             "epsilon": _eps_of(eps_dir), "n": stored.get("n"), "accuracy": None, "macro_f1": None,
                             "robust_accuracy": None, "attack_success_rate": None, "valid": False})
                continue
            with np.load(npz) as z:
                ids, y, pc, pa = z["ids"], z["y"], z["pred_clean"], z["pred_adv"]
            row, robust = attack_row(y, pc, pa)
            if not (row["n"] == stored["n"] and _close(row["accuracy"], stored["accuracy_under_attack"])
                    and _close(row["robust_accuracy"], stored["robust_accuracy"])
                    and _close(row["attack_success_rate"], stored["attack_success_rate"])):
                raise ReportConsistencyError(f"{eps_dir}: values recomputed from predictions.npz differ from attack_result.json")
            eps = _eps_of(eps_dir)
            rows.append({"phase": phase, "seed": seed, "model": model, "source": source, "attack": attack, "epsilon": eps,
                         **row, "valid": valid})
            arrays[(seed, model, source, attack, eps)] = (y, pa, robust, ids)
    if not rows:
        raise ReportError(f"no predictions found under {base}")
    return rows, arrays


# ------------------------------------------------------------------------------------------ aggregation
def _key(r):
    return (r["model"], r["source"], r["attack"], r["epsilon"])


def per_seed_values(rows, metric):
    """{(model, source, attack, eps): {seed: value}} over valid rows with a defined metric."""
    out = {}
    for r in rows:
        if r["valid"] and r[metric] is not None:
            out.setdefault(_key(r), {})[r["seed"]] = r[metric]
    return out


def aggregate(rows):
    """List of dicts: model, source, attack, epsilon, metric, n_seeds, mean, sd, per_seed."""
    out = []
    for metric in METRICS:
        for key, by_seed in sorted(per_seed_values(rows, metric).items()):
            mean, sd, n = mean_sd(by_seed.values())
            out.append({"model": key[0], "source": key[1], "attack": key[2], "epsilon": key[3], "metric": metric,
                        "n_seeds": n, "mean": mean, "sd": sd, "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)}})
    return out


def paired_differences(rows, pairs=PAIRS):
    """Per-seed difference a - b at each model's primary source; mean +/- sd over seeds."""
    out = []
    for a, b in pairs:
        for metric in METRICS:
            values = per_seed_values(rows, metric)
            conditions = {(k[2], k[3]) for k in values if k[0] in (a, b)}
            for attack, eps in sorted(conditions):
                ka = (a, PRIMARY_SOURCE[a] if attack != "clean" else "clean", attack, eps)
                kb = (b, PRIMARY_SOURCE[b] if attack != "clean" else "clean", attack, eps)
                if ka not in values or kb not in values:
                    continue
                seeds = sorted(set(values[ka]) & set(values[kb]))
                diffs = {s: values[ka][s] - values[kb][s] for s in seeds}
                mean, sd, n = mean_sd(diffs.values())
                if n:
                    out.append({"pair": f"{a}-{b}", "attack": attack, "epsilon": eps, "metric": metric, "n_seeds": n,
                                "mean": mean, "sd": sd, "per_seed": {str(s): diffs[s] for s in seeds}})
    return out


# ------------------------------------------------------------------------------------------ writing helpers
def _write_text(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    durable_write_bytes(path, text.encode("utf-8"))


def _write_csv(path: Path, phase: str, header, records) -> None:
    buffer = io.StringIO()
    buffer.write(f"# {banner_text(phase)}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for record in records:
        writer.writerow(["" if record.get(h) is None else record.get(h) for h in header])
    _write_text(path, buffer.getvalue())


def _md_table(header, records) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in records]
    return "\n".join(lines)


def _per_seed_text(per_seed: dict) -> str:
    return "; ".join(f"s{s}={v:.4f}" for s, v in per_seed.items())


# ------------------------------------------------------------------------------------------ tables
def write_results_long(out, phase, rows):
    header = ["phase", "seed", "model", "source", "attack", "epsilon", "n", "accuracy", "macro_f1", "robust_accuracy",
              "attack_success_rate", "valid"]
    _write_csv(out / "results_long.csv", phase, header, sorted(rows, key=lambda r: (r["seed"], r["model"], r["source"], r["attack"], r["epsilon"])))


def write_aggregate(out, phase, rows, models_used):
    agg = aggregate(rows)
    header = ["model", "source", "attack", "epsilon", "metric", "n_seeds", "mean", "sd", "per_seed"]
    _write_csv(out / "results_agg.csv", phase, header,
               [{**a, "sd": "n/a" if a["sd"] is None else a["sd"], "per_seed": _per_seed_text(a["per_seed"])} for a in agg])
    lines = [f"# {banner_text(phase)}", "", "Mean ± sample standard deviation (ddof=1) over seeds; `n/a` means one seed.",
             "Accuracy and macro-F1 are on the attacked predictions (clean rows: attack `clean`); robust accuracy needs both "
             "the clean and the attacked prediction to be correct. Invalid units are excluded (see invalid_units.md).", ""]
    for metric in METRICS:
        lines += [f"## {metric}", ""]
        table = [(a["model"], a["source"], a["attack"], f"{a['epsilon']:g}", a["n_seeds"], fmt(a["mean"], a["sd"]),
                  _per_seed_text(a["per_seed"])) for a in agg if a["metric"] == metric]
        lines += [_md_table(["model", "source", "attack", "epsilon", "seeds", "mean ± sd", "per seed"], table), ""]
    paired = paired_differences(rows, PAIRS)
    lines += ["## Paired differences per seed (each model at its primary source: V1, V3 fp32_direct; V2, V4 bpda_int8)", ""]
    lines += [_md_table(["pair", "attack", "epsilon", "metric", "seeds", "mean ± sd", "per seed"],
                        [(p["pair"], p["attack"], f"{p['epsilon']:g}", p["metric"], p["n_seeds"], fmt(p["mean"], p["sd"]),
                          _per_seed_text(p["per_seed"])) for p in paired]), ""]
    _write_csv(out / "paired_differences.csv", phase, ["pair", "attack", "epsilon", "metric", "n_seeds", "mean", "sd", "per_seed"],
               [{**p, "sd": "n/a" if p["sd"] is None else p["sd"], "per_seed": _per_seed_text(p["per_seed"])} for p in paired])
    if any(m in models_used for m in QAT_MODELS):
        qat = paired_differences(rows, QAT_PAIRS)
        labels = {"FP32-FT-V1": "FP32-FT - V1 (extra-training effect)", "FP32-FT-PTQ-V2": "FP32-FT->PTQ - V2 (extra-training effect)",
                  "QAT-FT-FP32-FT-PTQ": "QAT-FT - FP32-FT->PTQ (QAT benefit)"}
        lines += ["## QAT attribution (paired per seed)", ""]
        lines += [_md_table(["comparison", "attack", "epsilon", "metric", "seeds", "mean ± sd"],
                            [(labels[p["pair"]], p["attack"], f"{p['epsilon']:g}", p["metric"], p["n_seeds"], fmt(p["mean"], p["sd"]))
                             for p in qat]), "", QAT_SENTENCE, ""]
        _write_csv(out / "qat_paired.csv", phase, ["pair", "attack", "epsilon", "metric", "n_seeds", "mean", "sd", "per_seed"],
                   [{**p, "sd": "n/a" if p["sd"] is None else p["sd"], "per_seed": _per_seed_text(p["per_seed"])} for p in qat])
    _write_text(out / "results_agg.md", "\n".join(lines) + "\n")
    return agg


# ------------------------------------------------------------------------------------------ plots
def _families(rows):
    names = sorted({r["attack"] for r in rows if r["attack"] != "clean"})
    return {"fgsm": [n for n in names if n == "fgsm"], "pgd": [n for n in names if n.startswith("pgd")]}


def _curve(agg, model, source, attacks, metric):
    """[(eps, mean, sd)] for one model/source over the given attack names; the clean point only at eps 0 when absent."""
    pts = {}
    for a in agg:
        if a["model"] == model and a["source"] == source and a["attack"] in attacks and a["metric"] == metric:
            pts[a["epsilon"]] = (a["mean"], a["sd"] or 0.0, a["n_seeds"])
    return [(e, *pts[e]) for e in sorted(pts)]


def _save_figure(fig, path: Path, csv_path: Path, phase, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        fig.savefig(handle, format="png", dpi=130)
    tmp.replace(path)
    _write_csv(csv_path, phase, ["curve", "epsilon", "mean", "sd", "n_seeds"],
               [{"curve": c, "epsilon": e, "mean": m, "sd": s, "n_seeds": n} for c, e, m, s, n in points])


def write_plots(out, phase, rows, arrays, agg, models_used):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    families = _families(rows)
    labels = {"accuracy": "accuracy under attack", "macro_f1": "macro-F1 under attack"}
    prefix = {"accuracy": "acc", "macro_f1": "f1"}

    def figure(title, curves, ylabel, name, subdir=None):
        if not any(c for _, c in curves):
            return
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        points = []
        for label, curve in curves:
            if not curve:
                continue
            e, m, s = [c[0] for c in curve], [c[1] for c in curve], [c[2] for c in curve]
            ax.errorbar(e, m, yerr=s, marker="o", capsize=3, label=label)
            points += [(label, *c) for c in curve]
        ax.set_xlabel("epsilon (L-inf, MinMax-scaled feature space)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{BANNERS[phase]}: {title}", fontsize=9)
        ax.text(0.01, 0.01, FRAMING, transform=ax.transAxes, fontsize=6, va="bottom")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        folder = out / subdir if subdir else out
        _save_figure(fig, folder / f"{name}.png", folder / f"{name}_points.csv", phase, points)
        plt.close(fig)
        written.append(str((folder / f"{name}.png").relative_to(out)))

    models = [m for m in BASE_MODELS + QAT_MODELS if m in models_used]
    for family, attacks in families.items():
        if not attacks:
            continue
        tag = "fgsm" if family == "fgsm" else "pgd"
        for metric in ("accuracy", "macro_f1"):
            curves = [(f"{m} ({PRIMARY_SOURCE[m]})", _curve(agg, m, PRIMARY_SOURCE[m], attacks, metric)) for m in models]
            figure(f"{labels[metric]} vs attack strength, {'/'.join(attacks)}, mean ± sd over seeds", curves,
                   labels[metric], f"{prefix[metric]}_vs_eps_{tag}")
            for model in [m for m in models if m in INT8_MODELS]:
                sources = sorted({a["source"] for a in agg if a["model"] == model and a["attack"] in attacks})
                figure(f"{model}, all sources, {labels[metric]}", [(s, _curve(agg, model, s, attacks, metric)) for s in sources],
                       labels[metric], f"{prefix[metric]}_vs_eps_{tag}_{model}", subdir="by_source")
        union = []
        for model in [m for m in models if m in INT8_MODELS]:
            by_seed_eps = {}
            for (seed, mo, source, attack, eps), (y, pred, robust, ids) in arrays.items():
                if mo == model and attack in attacks and source != "clean":
                    by_seed_eps.setdefault((seed, eps), []).append((source, robust, ids))
            per_eps = {}
            for (seed, eps), entries in by_seed_eps.items():
                if any(not np.array_equal(entries[0][2], e[2]) for e in entries):
                    raise ReportError(f"example ids differ across sources for {model} seed {seed} eps {eps}")
                per_eps.setdefault(eps, []).append(float(np.logical_and.reduce([e[1] for e in entries]).mean()))
            curve = []
            for eps in sorted(per_eps):
                mean, sd, n = mean_sd(per_eps[eps])
                curve.append((eps, mean, sd or 0.0, n))
            union.append((f"{model} any-source (union of all sources)", curve))
        figure(f"robust accuracy, any-source union (separately labelled), {'/'.join(attacks)}", union, "robust accuracy",
               f"any_source_robust_vs_eps_{tag}", subdir="by_source")
    return written


# ------------------------------------------------------------------------------------------ per-class table
def distinct_vectors_per_class(split_dir, partition):
    """{class index: distinct feature vectors in the partition} from the frozen split, or None when unavailable."""
    try:
        split_dir = Path(split_dir)
        names = json.loads((split_dir / "split_m2.json").read_text(encoding="utf-8"))["dataset"]["class_names"]
        groups = json.loads((split_dir / "group_manifest.json").read_text(encoding="utf-8"))["groups"]
    except (OSError, KeyError, ValueError):
        return None
    index = {n: i for i, n in enumerate(names)}
    counts = {i: 0 for i in range(len(names))}
    for g in groups.values():
        if g["partition"] == partition:
            for label in g["label_counts"]:
                counts[index[label]] += 1
    return counts


def write_per_class(out, phase, rows, arrays, models_used, distinct=None):
    """Per-class recall: clean and under the largest-epsilon attack of each family (primary source), mean ± sd over seeds."""
    top = {}
    for (seed, model, source, attack, eps) in arrays:
        if attack != "clean":
            top[attack] = max(top.get(attack, 0.0), eps)
    conditions = [("clean", "clean", 0.0)] + [(a, a, e) for a, e in sorted(top.items())]
    records = []
    for model in [m for m in BASE_MODELS + QAT_MODELS if m in models_used]:
        for label, attack, eps in conditions:
            source = "clean" if attack == "clean" else PRIMARY_SOURCE[model]
            per_class = {c: {} for c in range(N_CLASSES)}
            support = {}
            for (seed, mo, so, at, e), (y, pred, robust, ids) in arrays.items():
                if (mo, so, at, e) != (model, source, attack, eps):
                    continue
                for c in range(N_CLASSES):
                    mask = y == c
                    if mask.any():
                        per_class[c][seed] = float((pred[mask] == c).mean())
                        support[c] = int(mask.sum())
            for c in range(N_CLASSES):
                if not per_class[c]:
                    continue
                mean, sd, n = mean_sd(per_class[c].values())
                weak = support[c] < WEAK_ROWS or (distinct is not None and distinct.get(c, WEAK_DISTINCT) < WEAK_DISTINCT)
                records.append({"model": model, "condition": f"{label}" + ("" if attack == "clean" else f" eps={eps:g}"),
                                "source": source, "class": c, "rows": support[c],
                                "distinct_vectors": None if distinct is None else distinct.get(c),
                                "recall_mean": mean, "recall_sd": "n/a" if sd is None else sd, "n_seeds": n,
                                "weak_per_class_evidence": "yes" if weak else "no"})
    header = ["model", "condition", "source", "class", "rows", "distinct_vectors", "recall_mean", "recall_sd", "n_seeds",
              "weak_per_class_evidence"]
    _write_csv(out / "per_class.csv", phase, header, records)
    lines = [f"# {banner_text(phase)}", "", "Per-class recall (mean ± sd over seeds). `weak_per_class_evidence` is `yes` when the class has "
             f"fewer than {WEAK_ROWS} rows or fewer than {WEAK_DISTINCT} distinct feature vectors in the evaluated partition; recall for those "
             "classes rests on few independent vectors. A class whose rows are mostly one repeated vector (single dominant vector) "
             "contributes many rows but little independent evidence.", ""]
    lines += [_md_table(["model", "condition", "class", "rows", "distinct vectors", "recall", "weak evidence"],
                        [(r["model"], r["condition"], r["class"], r["rows"], r["distinct_vectors"] if r["distinct_vectors"] is not None else "n/a",
                          fmt(r["recall_mean"], None if r["recall_sd"] == "n/a" else r["recall_sd"]), r["weak_per_class_evidence"])
                         for r in records]), ""]
    _write_text(out / "per_class.md", "\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------ invalid / blocked
def write_status_files(out, phase, run_dir, rows):
    from .m2b_state import CampaignState
    state = CampaignState(Path(run_dir) / "state")
    records = [r for r in state.list_units() if r["phase"] == phase] if state.exists() else []
    invalid = [r for r in records if r["status"] == "invalid"]
    seen = {r["id"] for r in invalid}
    lines = [f"# {banner_text(phase)}", "", "Invalid attack units failed an invariant check (E5). They are kept as evidence and excluded "
             "from every table and plot.", ""]
    table = [(r["id"], ", ".join((r.get("extra") or {}).get("failures", [])) or "see attack_checks.json") for r in invalid]
    table += [(f"s{r['seed']}:{r['model']}/{r['source']}/{r['attack']}/eps{r['epsilon']:g}", "valid=false in attack_result.json")
              for r in rows if not r["valid"] and f"s{r['seed']}.{'val' if phase == 'validation' else 'test'}.attack.{r['model']}.{r['source']}.{r['attack']}.eps{r['epsilon']:g}" not in seen]
    lines += [_md_table(["unit", "reason"], table) if table else "No invalid units.", ""]
    _write_text(out / "invalid_units.md", "\n".join(lines) + "\n")
    blocked = [r for r in records if r["status"] == "blocked"]
    lines = [f"# {banner_text(phase)}", "", "Blocked units and the recorded reason (QAT stage blocks are documented, not failures).", ""]
    table = [(r["id"], (r.get("error") or {}).get("type", ""), (r.get("error") or {}).get("message", "")[:300]) for r in blocked]
    lines += [_md_table(["unit", "type", "reason"], table) if table else "No blocked units.", ""]
    _write_text(out / "blocked_units.md", "\n".join(lines) + "\n")
    return len(invalid) + sum(1 for r in rows if not r["valid"]), len(blocked)


# ------------------------------------------------------------------------------------------ efficiency report
def read_peak_memory_status(text: str):
    """VmHWM in bytes from the text of /proc/self/status, or None."""
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _median_over_processes(measurements, variant, getter):
    values = []
    for m in measurements:
        if m["variant"] == variant:
            try:
                value = getter(m["result"])
            except (KeyError, TypeError):
                value = None
            if value is not None:
                values.append(value)
    return float(np.median(values)) if values else None


EFFICIENCY_FIELDS = {
    "size_bytes": lambda r: r["state_dict_bytes"],
    "batch1_median_ms": lambda r: r["batch1"]["median_ms"], "batch1_p95_ms": lambda r: r["batch1"]["p95_ms"],
    "batch256_median_ms": lambda r: r["batch256"]["median_ms"], "batch256_p95_ms": lambda r: r["batch256"]["p95_ms"],
    "memory_delta_bytes": lambda r: r["peak_wset_minus_baseline_bytes"]}


def collect_efficiency(run_dir):
    """{seed: {threads: {'record': efficiency.json, 'per_variant': {variant: {field: median over processes}}}}}, hardware."""
    base = Path(run_dir) / "efficiency"
    out, hardware = {}, {}
    for seed_dir in sorted(base.glob("s*")) if base.is_dir() else []:
        if not seed_dir.name[1:].isdigit():
            continue
        seed = int(seed_dir.name[1:])
        if (seed_dir / "hardware.json").is_file():
            hardware[seed] = json.loads((seed_dir / "hardware.json").read_text(encoding="utf-8"))
        for pass_dir in sorted(seed_dir.glob("threads*")):
            path = pass_dir / "efficiency.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            per_variant = {v: {f: _median_over_processes(record["measurements"], v, g) for f, g in EFFICIENCY_FIELDS.items()}
                           for v in record["variants"]}
            out.setdefault(seed, {})[int(pass_dir.name[len("threads"):])] = {"record": record, "per_variant": per_variant}
    if not out:
        raise ReportError(f"no efficiency results under {base}; run the efficiency stage first")
    return out, hardware


def aggregate_efficiency(collected):
    """{threads: {variant: {field: (mean, sd, n)}}} over seeds (mean ± sample sd of the per-seed medians)."""
    out = {}
    passes = sorted({t for per in collected.values() for t in per})
    for threads in passes:
        variants = []
        for per in collected.values():
            for v in per.get(threads, {}).get("per_variant", {}):
                if v not in variants:
                    variants.append(v)
        out[threads] = {v: {f: mean_sd([per[threads]["per_variant"][v][f] for per in collected.values()
                                         if threads in per and per[threads]["per_variant"].get(v, {}).get(f) is not None])
                            for f in EFFICIENCY_FIELDS} for v in variants}
    return out


def write_efficiency(out, run_dir):
    phase = "efficiency"
    collected, hardware = collect_efficiency(run_dir)
    agg = aggregate_efficiency(collected)
    metrics = sorted({m for per in collected.values() for p in per.values() for m in p["record"].get("peak_memory_metric", [])})
    first_hw = next(iter(hardware.values()), {})
    busy = sorted(s for s, h in hardware.items() if h.get("busy_override"))
    lines = [f"# {banner_text(phase)}", "",
             "Artifact size, latency and memory were measured on the same VM, with the same rows, thread counts and batch sizes for every "
             "compared model, each in a fresh child process (interleaved order).",
             f"Peak memory metric: {', '.join(metrics) or 'n/a'} (memory delta = peak minus the baseline RSS after model load). {MEMORY_NOTE}",
             f"Hardware (hardware.json): {first_hw.get('cpu_model', 'n/a')}, nproc {first_hw.get('nproc', 'n/a')}, MemTotal "
             f"{first_hw.get('mem_total', 'n/a')}, kernel {first_hw.get('kernel', 'n/a')}, torch {first_hw.get('torch', 'n/a')}, quantized engine "
             f"{first_hw.get('quantized_engine', 'n/a')}, scaling governor {first_hw.get('scaling_governor') or 'n/a'}.",
             f"Seeds measured: {sorted(collected)}. Busy-machine override recorded for seeds: {busy or 'none'}.",
             NO_CLAIM, ""]
    csv_rows = []
    for threads, variants in agg.items():
        lines += [f"## {threads} thread(s): mean ± sample sd over seeds of the per-seed medians", ""]
        table = []
        for v, fields in variants.items():
            table.append((v, *[(fmt(*fields[f][:2], digits=1 if f.endswith("bytes") else 3)) for f in EFFICIENCY_FIELDS]))
            for f, (mean, sd, n) in fields.items():
                csv_rows.append({"threads": threads, "variant": v, "field": f, "mean": mean, "sd": "n/a" if sd is None else sd, "n_seeds": n})
        lines += [_md_table(["variant", "size (bytes)", "batch-1 median (ms)", "batch-1 p95 (ms)", "batch-256 median (ms)",
                             "batch-256 p95 (ms)", f"memory delta (bytes, {'/'.join(metrics) or 'n/a'})"], table), ""]
    lines += ["## Per seed", ""]
    per_seed = []
    for seed, per in sorted(collected.items()):
        for threads, entry in sorted(per.items()):
            for v, fields in entry["per_variant"].items():
                per_seed.append((seed, threads, v, *[("n/a" if fields[f] is None else f"{fields[f]:.3f}") for f in EFFICIENCY_FIELDS]))
    lines += [_md_table(["seed", "threads", "variant", *EFFICIENCY_FIELDS], per_seed), ""]
    missing = [m for m in QAT_MODELS if not any(m in v for v in agg.values())]
    if missing:
        lines += [f"QAT arms {missing} have no efficiency numbers: they were not part of the measured freeze, or the QAT stage was not run or was blocked "
                  "(see blocked_units.md of the validation report).", ""]
    _write_text(out / "efficiency.md", "\n".join(lines) + "\n")
    _write_csv(out / "efficiency_agg.csv", phase, ["threads", "variant", "field", "mean", "sd", "n_seeds"], csv_rows)
    return agg


# ------------------------------------------------------------------------------------------ entry points
def build_report(run_dir, phase, *, include_qat=False, split_dir=None, freeze_sha=None):
    """Write every report file of one phase; returns a summary dict."""
    run_dir = Path(run_dir)
    out = run_dir / "reports" / phase
    out.mkdir(parents=True, exist_ok=True)
    if phase == "efficiency":
        write_efficiency(out, run_dir)
        summary = {"phase": phase, "out": str(out), "files": sorted(p.name for p in out.glob("*"))}
        durable_write_json(out / "report_manifest.json", {"banner": banner_text(phase), "created_utc": utc_now(), **summary})
        return summary
    models = BASE_MODELS + (QAT_MODELS if include_qat else ())
    rows, arrays = collect(run_dir, phase, models)
    models_used = {r["model"] for r in rows}
    write_results_long(out, phase, rows)
    agg = write_aggregate(out, phase, rows, models_used)
    plots = write_plots(out, phase, rows, arrays, agg, models_used)
    distinct = distinct_vectors_per_class(split_dir, "validation" if phase == "validation" else "test") if split_dir else None
    write_per_class(out, phase, rows, arrays, models_used, distinct)
    n_invalid, n_blocked = write_status_files(out, phase, run_dir, rows)
    summary = {"phase": phase, "out": str(out), "units": len(rows), "seeds": sorted({r["seed"] for r in rows}), "plots": plots,
               "invalid": n_invalid, "blocked": n_blocked, "freeze_sha256": freeze_sha}
    durable_write_json(out / "report_manifest.json", {"banner": banner_text(phase), "created_utc": utc_now(), **summary})
    return summary


def run(args, root) -> int:
    """`m2b_campaign report` (args: config, run_name, out_root, allow_outside_root, phase, freeze_name, include_qat)."""
    from .m2b_campaign import EXIT_CONFIG, EXIT_OK, ExitError, load_campaign, resolve_paths
    run_dir, campaign_path = resolve_paths(args, root)
    cfg = load_campaign(campaign_path, root)
    freeze_sha = None
    if args.phase == "test":
        freeze = run_dir / "freezes" / f"{args.freeze_name}.json"
        if not freeze.is_file():
            raise ExitError(EXIT_CONFIG, f"the final-test report needs the evaluation freeze {freeze}; run freeze-eval and the test phase first")
        freeze_sha = sha256_file(freeze)
    data_cfg = json.loads((Path(root) / cfg["siblings"]["data_v2"]["path"]).read_text(encoding="utf-8"))
    split_dir = Path(root) / data_cfg["split"]["output_dir"]
    try:
        summary = build_report(run_dir, args.phase, include_qat=args.include_qat, split_dir=split_dir, freeze_sha=freeze_sha)
    except ReportError as error:
        raise ExitError(EXIT_CONFIG, str(error)) from error
    print(json.dumps(summary, indent=2))
    return EXIT_OK
