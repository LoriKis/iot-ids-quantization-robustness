"""Report generator: four-model table, robustness curves, figures and the Markdown report (DESIGN 7).

Reads only result files under ``OUT``; never recomputes a model result, loads a model or imports torch.
Standard library + numpy at import time; matplotlib is imported inside the figure function.
Python 3.9 compatible.

    python -m m2_deadline.report --out OUT [--v4-root PATH]

Exit codes: 0 ok; 1 unexpected error; 2 an input file is missing or inconsistent; 3 a stage marker or the
frozen protocol does not validate.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import markers, metrics, protocol
from .common import (
    ATTACK_NAMES,
    CLEAN_ATTACK,
    CLEAN_UNIT,
    CSV_COLUMNS,
    DISPLAY_NAMES,
    DataIntegrityError,
    DeadlineError,
    EFFICIENCY_LATENCY_KEYS,
    EFFICIENCY_RESULT_KEYS,
    EPS0_CONSISTENCY_KEYS,
    GuardError,
    INT8_LABEL,
    INT8_MODELS,
    MODEL_IDS,
    N_CLASSES,
    PARENTS,
    PATHS,
    PRECISION,
    UNIT_RESULT_KEYS,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    check_required_keys,
    eps_str,
    exit_code_for,
    load_config,
    out_path,
    read_json,
    sha256_file,
    unit_id,
    utc_stamp,
)
from .metrics import accuracy, macro_f1, pp_diff

RECIPE_EPOCHS = 25
REQUIRED_STAGES = ("test_clean", "robustness", "efficiency")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m2_deadline_v1.json"
BPDA_SENTENCE = (
    "INT8 attack gradients are approximate (BPDA); a robustness gain or loss for INT8 may not hold under "
    "stronger attacks."
)
ATTACK_LABELS = {"fgsm": "FGSM", "pgd5": "PGD-5"}
PAIRS = {
    "q1_ptq_ordinary": ("V1", "V2"),
    "q1_ptq_adversarial": ("V3", "V4"),
    "q2_adversarial_vs_ordinary_int8": ("V2", "V4"),
}


# ---------------------------------------------------------------- wording helpers (all computed from values)
def describe_pp(d: float, tol: float = 0.5) -> str:
    """Wording of a score difference ``d`` in percentage points (``b - a``)."""
    if abs(d) < tol:
        return "essentially unchanged (%+.2f pp)" % d
    return "%s by %.2f pp" % ("higher" if d > 0 else "lower", abs(d))


def describe_ratio(ratio: Optional[float], kind: str, tol: float = 0.05) -> str:
    """Wording of ``ratio = b / a`` for ``kind`` in size, memory (smaller/larger) or latency (faster/slower)."""
    if ratio is None or not math.isfinite(ratio) or ratio <= 0:
        return "not available"
    if abs(ratio - 1.0) <= tol + 1e-9:
        return "about the same (ratio %.2fx)" % ratio
    low, high = ("faster", "slower") if kind == "latency" else ("smaller", "larger")
    return "%.1f%% %s (ratio %.2fx)" % (abs(ratio - 1.0) * 100.0, low if ratio < 1.0 else high, ratio)


def _ratio(b: Optional[float], a: Optional[float]) -> Optional[float]:
    if a is None or b is None or not a:
        return None
    return float(b) / float(a)


def _pct(x: float) -> str:
    return "%.2f%%" % (100.0 * x)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- reading and validating inputs
def _input_files(out: Path) -> Dict[str, Path]:
    keys = (
        "run_identity", "protocol", "protocol_sha256", "subset_ids", "subset_counts", "clean_results",
        "predictions_full_test", "robustness_results", "predictions_subset", "efficiency",
    )
    return {k: out_path(out, k) for k in keys}


def _read_csv(path: Path, columns: Sequence[str]) -> List[Dict[str, str]]:
    with open(str(path), "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise DataIntegrityError("%s: header %s differs from %s" % (path.name, reader.fieldnames, list(columns)))
        return list(reader)


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9


def _verify_markers(out: Path, ident: Dict[str, Any]) -> None:
    ctx = markers.context(
        config_sha256=ident["config_sha256"],
        lock_sha256=ident["lock_sha256"],
        protocol_sha256=sha256_file(out_path(out, "protocol")),
        epochs=int(ident["epochs"]),
    )
    problems = []
    for stage in REQUIRED_STAGES:
        ok, reason = markers.verify(out, stage, ctx)
        if not ok:
            problems.append("%s: %s" % (stage, reason))
    if problems:
        raise GuardError("stage markers do not validate: " + "; ".join(problems))


def load_inputs(out: Any) -> Dict[str, Any]:
    """Read and cross-check every result file. ``DataIntegrityError`` (exit 2) lists a missing file;
    ``GuardError`` (exit 3) is a marker or protocol failure."""
    out = Path(str(out))
    files = _input_files(out)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise DataIntegrityError("missing input files:\n  " + "\n  ".join(missing))
    ident = read_json(files["run_identity"])
    check_required_keys(ident, "run_identity", "run_identity.json")
    _verify_markers(out, ident)
    protocol.assert_frozen(out)  # raises ProtocolChangedError (exit 3)

    proto = read_json(files["protocol"])
    check_required_keys(proto, "protocol", "protocol.json")
    counts = read_json(files["subset_counts"])
    check_required_keys(counts, "subset_counts", "robustness_subset_counts.json")
    clean = read_json(files["clean_results"])
    check_required_keys(clean, "clean_results", "clean_results.json")
    rob = read_json(files["robustness_results"])
    check_required_keys(rob, "robustness_results", "robustness_results.json")
    eff = read_json(files["efficiency"])
    check_required_keys(eff, "efficiency", "efficiency.json")
    if int(ident["epochs"]) != int(proto["epochs"]):
        raise DataIntegrityError("run_identity epochs differ from protocol epochs")

    subset_ids, subset_y = protocol.parse_ids_csv(files["subset_ids"].read_bytes())
    full_rows = _read_csv(files["predictions_full_test"], CSV_COLUMNS["predictions_full_test"])
    sub_rows = _read_csv(files["predictions_subset"], CSV_COLUMNS["predictions_subset"])
    data = {
        "out": out, "identity": ident, "protocol": proto, "counts": counts, "clean": clean, "robustness": rob,
        "efficiency": eff, "subset_ids": subset_ids, "subset_y": subset_y, "full_rows": full_rows,
        "sub_rows": sub_rows, "protocol_sha256": sha256_file(files["protocol"]),
    }
    data["full"] = _check_full_test(data)
    _check_subset(data)
    data["efficiency_rows"] = _efficiency_rows(data)
    return data


def _check_full_test(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = data["full_rows"]
    clean = data["clean"]
    n_test = int(clean["n_test"])
    if len(rows) != n_test or int(data["protocol"]["split"]["n_test"]) != n_test:
        raise DataIntegrityError("predictions_full_test.csv has %d rows, expected n_test=%d" % (len(rows), n_test))
    ids = np.array([int(r["example_id"]) for r in rows], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise DataIntegrityError("predictions_full_test.csv has duplicate example ids")
    y = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
    flagged = sorted(int(i) for i, r in zip(ids, rows) if str(r["in_robust_subset"]).lower() == "true")
    if flagged != sorted(int(i) for i in data["subset_ids"]):
        raise DataIntegrityError("in_robust_subset flags differ from the frozen subset ids")
    preds = {m: np.array([int(r["%s_pred" % m]) for r in rows], dtype=np.int64) for m in MODEL_IDS}
    for m in MODEL_IDS:
        entry = clean["models"].get(m)
        if entry is None or any(k not in entry for k in ("accuracy", "macro_f1", "n")):
            raise DataIntegrityError("clean_results.json lacks model %s or its keys" % m)
        if int(entry["n"]) != n_test:
            raise DataIntegrityError("clean_results.json n for %s differs from n_test" % m)
        if not _close(entry["accuracy"], accuracy(y, preds[m])) or not _close(entry["macro_f1"], macro_f1(y, preds[m])):
            raise DataIntegrityError("clean_results.json for %s differs from predictions_full_test.csv" % m)
    return {"ids": ids, "y": y, "preds": preds}


def _expected_units(data: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    eps = [float(e) for e in data["protocol"]["robustness"]["epsilons"]]
    units = [(CLEAN_ATTACK, 0.0, CLEAN_UNIT)]
    for attack in ATTACK_NAMES:
        units += [(attack, e, unit_id(attack, e)) for e in eps if e != 0.0]
    return units


def _check_subset(data: Dict[str, Any]) -> None:
    ids = data["subset_ids"]
    label_of = {int(i): int(y) for i, y in zip(ids, data["subset_y"])}
    rob = data["robustness"]
    n = int(rob["n"])
    if n != len(ids) or int(data["counts"]["n_actual"]) != n or int(data["protocol"]["robustness"]["n_actual"]) != n:
        raise DataIntegrityError("subset size differs between ids file, counts, protocol and results")
    groups: Dict[Tuple[str, str, str], Dict[int, Tuple[int, int]]] = {}
    for r in data["sub_rows"]:
        key = (r["model_id"], r["attack"], r["epsilon"])
        i = int(r["example_id"])
        if i in groups.setdefault(key, {}):
            raise DataIntegrityError("duplicate example %d in predictions_subset.csv for %s" % (i, key))
        groups[key][i] = (int(r["y_true"]), int(r["y_pred"]))
    expected = {(m, a, eps_str(e)) for m in MODEL_IDS for a, e, _ in _expected_units(data)}
    if set(groups) != expected:
        raise DataIntegrityError(
            "predictions_subset.csv groups differ from the expected units (missing %s, extra %s)"
            % (sorted(expected - set(groups))[:3], sorted(set(groups) - expected)[:3])
        )
    wanted = set(label_of)
    for (m, a, e), rows in groups.items():
        if set(rows) != wanted:
            raise DataIntegrityError("predictions_subset.csv ids for %s/%s/%s differ from the frozen subset" % (m, a, e))
        if any(rows[i][0] != label_of[i] for i in rows):
            raise DataIntegrityError("predictions_subset.csv y_true for %s/%s/%s differs from the frozen labels" % (m, a, e))
    for m in MODEL_IDS:
        units = rob["units"].get(m)
        if units is None:
            raise DataIntegrityError("robustness_results.json lacks model %s" % m)
        if any(k not in rob["eps0_consistency"].get(m, {}) for k in EPS0_CONSISTENCY_KEYS):
            raise DataIntegrityError("robustness_results.json eps0_consistency for %s lacks keys" % m)
        for a, e, uid in _expected_units(data):
            u = units.get(uid)
            if u is None or any(k not in u for k in UNIT_RESULT_KEYS):
                raise DataIntegrityError("robustness_results.json lacks unit %s/%s" % (m, uid))
            rows = groups[(m, a, eps_str(e))]
            y = np.array([rows[int(i)][0] for i in ids], dtype=np.int64)
            p = np.array([rows[int(i)][1] for i in ids], dtype=np.int64)
            if int(u["n"]) != n or not _close(u["accuracy"], accuracy(y, p)) or not _close(u["macro_f1"], macro_f1(y, p)):
                raise DataIntegrityError("robustness_results.json %s/%s differs from predictions_subset.csv" % (m, uid))


def _median(values: Sequence[Any]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def _efficiency_rows(data: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    eff = data["efficiency"]
    if eff["any_failed"] or not eff["interleaved"]:
        raise DataIntegrityError("efficiency.json reports a failed or non-interleaved measurement")
    reps = int(data["protocol"]["efficiency"]["repetitions"])
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for m in MODEL_IDS:
        runs = [e["result"] for e in eff["measurements"] if e.get("variant") == m]
        if len(runs) != reps:
            raise DataIntegrityError("efficiency.json has %d measurements for %s, expected %d" % (len(runs), m, reps))
        for r in runs:
            if any(k not in r for k in EFFICIENCY_RESULT_KEYS):
                raise DataIntegrityError("efficiency.json measurement for %s lacks keys" % m)
            for b in ("batch1", "batch256"):
                if any(k not in r[b] for k in EFFICIENCY_LATENCY_KEYS):
                    raise DataIntegrityError("efficiency.json %s lacks latency keys for %s" % (b, m))
        batch = [r["batch256"].get("batch_size", 256) for r in runs]
        b256 = _median([r["batch256"]["median_ms"] for r in runs])
        result[m] = {
            "state_dict_bytes": _median([r["state_dict_bytes"] for r in runs]),
            "latency_b1_median_ms": _median([r["batch1"]["median_ms"] for r in runs]),
            "latency_b1_p95_ms": _median([r["batch1"]["p95_ms"] for r in runs]),
            "latency_b256_median_ms": b256,
            "latency_b256_p95_ms": _median([r["batch256"]["p95_ms"] for r in runs]),
            "latency_b256_per_example_median_ms": None if b256 is None else b256 / float(_median(batch) or 256),
            "peak_rss_bytes": _median([r["peak_wset_bytes"] for r in runs]),
            "peak_rss_delta_bytes": _median([r["peak_wset_minus_baseline_bytes"] for r in runs]),
            "peak_rss_sampled_delta_bytes": _median([r["sampled_peak_minus_baseline_bytes"] for r in runs]),
        }
    return result


# ---------------------------------------------------------------- tables
def _fnum(value: Optional[float], digits: Optional[int] = None) -> str:
    if value is None:
        return ""
    if digits is not None:
        return repr(round(float(value), digits))
    return repr(float(value))


def comparison_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    clean = data["clean"]["models"]
    eff = data["efficiency_rows"]
    rows = []
    for m in MODEL_IDS:
        e = eff[m]
        row = {
            "model_id": m,
            "display_name": DISPLAY_NAMES[m],
            "parent": PARENTS[m] or "",
            "precision": PRECISION[m],
            "epochs": int(data["protocol"]["epochs"]),
            "n_test": int(data["clean"]["n_test"]),
            "clean_accuracy_full_test": _fnum(clean[m]["accuracy"]),
            "macro_f1_full_test": _fnum(clean[m]["macro_f1"]),
            "state_dict_bytes": "" if e["state_dict_bytes"] is None else str(int(e["state_dict_bytes"])),
            "state_dict_mb": _fnum(None if e["state_dict_bytes"] is None else e["state_dict_bytes"] / 1e6, 4),
        }
        for k in ("latency_b1_median_ms", "latency_b1_p95_ms", "latency_b256_median_ms", "latency_b256_p95_ms",
                  "latency_b256_per_example_median_ms"):
            row[k] = _fnum(e[k])
        for k in ("peak_rss_bytes", "peak_rss_delta_bytes", "peak_rss_sampled_delta_bytes"):
            row[k] = "" if e[k] is None else str(int(e[k]))
        rows.append(row)
    return rows


def _csv_text(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("true" if r[k] is True else "false" if r[k] is False else r[k]) for k in columns})
    return buf.getvalue()


def _mb(b: Optional[float]) -> str:
    return "n/a" if b is None else "%.2f" % (b / 1e6)


def comparison_markdown(data: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    proto = data["protocol"]
    eff = proto["efficiency"]
    headers = ["Model", "Parent", "Precision", "Epochs", "Accuracy, full test (n=%d)" % data["clean"]["n_test"],
               "Macro-F1, full test", "State size (MB)", "Latency b1 median (ms)", "Latency b1 p95 (ms)",
               "Latency b256 per example (ms)", "Peak RSS (MB)", "Peak RSS delta (MB)"]
    body = []
    for r in rows:
        body.append([
            "%s (%s)" % (r["display_name"], r["model_id"]), r["parent"] or "-", r["precision"], r["epochs"],
            _pct(float(r["clean_accuracy_full_test"])), _pct(float(r["macro_f1_full_test"])),
            "%.3f" % float(r["state_dict_mb"]) if r["state_dict_mb"] else "n/a",
            "%.4f" % float(r["latency_b1_median_ms"]), "%.4f" % float(r["latency_b1_p95_ms"]),
            "%.5f" % float(r["latency_b256_per_example_median_ms"]),
            _mb(float(r["peak_rss_bytes"]) if r["peak_rss_bytes"] else None),
            _mb(float(r["peak_rss_delta_bytes"]) if r["peak_rss_delta_bytes"] else None),
        ])
    notes = [
        "Notes:",
        "- Values are medians over %d repetitions; each repetition runs V1..V4 interleaved, every measurement in a "
        "fresh child process, on %d thread(s) (batch 1: %d warm-up and %d timed calls; batch 256: %d warm-up and "
        "%d timed calls; %d fixed validation rows)." % (
            eff["repetitions"], eff["threads"], eff["batch1"]["warmup"], eff["batch1"]["timed"],
            eff["batch256"]["warmup"], eff["batch256"]["timed"], eff["rows"]),
        "- Size: bytes of the serialized state (torch.save of the state_dict for FP32; the packed state.pt for "
        "INT8). MB = 1,000,000 bytes.",
        "- Latency: eval mode, inference mode, one thread; timed region is row -> repeat to 50 positions -> forward "
        "-> argmax. Batch 1 and batch 256 are in the CSV; the per-example figure is the batch-256 median divided "
        "by the batch size.",
        "- Memory: " + eff["memory_metric"] + ". MB = 1,000,000 bytes.",
        "- Accuracy and macro-F1 are on the full held-out test set (macro-F1 over all 27 labels, zero_division=0). "
        "The robustness-subset clean values are separate and appear in the robustness table.",
        "- Epochs: %d (recipe: %d). Seed 1, one run per parent." % (proto["epochs"], RECIPE_EPOCHS),
    ]
    return "\n".join(["# Four-model comparison", "", _md_table(headers, body), ""] + notes) + "\n"


# ---------------------------------------------------------------- differences
def score_points(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    n = int(data["robustness"]["n"])
    points = [("full_test", "Clean, full test (n=%d)" % data["clean"]["n_test"]),
              (CLEAN_UNIT, "Clean, robustness subset (n=%d)" % n)]
    for attack, e, uid in _expected_units(data)[1:]:
        points.append((uid, "%s, eps %s" % (ATTACK_LABELS[attack], eps_str(e))))
    return points


def _score(data: Dict[str, Any], model: str, point: str) -> Dict[str, float]:
    if point == "full_test":
        e = data["clean"]["models"][model]
    else:
        e = data["robustness"]["units"][model][point]
    return {"accuracy": float(e["accuracy"]), "macro_f1": float(e["macro_f1"])}


def compute_differences(data: Dict[str, Any], tol: float) -> List[Dict[str, Any]]:
    records = []
    for name, (a, b) in PAIRS.items():
        for point, label in score_points(data):
            sa, sb = _score(data, a, point), _score(data, b, point)
            for metric in ("accuracy", "macro_f1"):
                d = pp_diff(sa[metric], sb[metric])
                records.append({
                    "comparison": name, "a": a, "b": b, "point": point, "label": label, "metric": metric,
                    "a_value": sa[metric], "b_value": sb[metric], "pp": d, "wording": describe_pp(d, tol),
                })
    return records


def _tally(records: Sequence[Dict[str, Any]], tol: float) -> Tuple[int, int, int]:
    higher = sum(1 for r in records if r["pp"] >= tol)
    lower = sum(1 for r in records if r["pp"] <= -tol)
    return higher, lower, len(records) - higher - lower


def _attack_records(diffs: Sequence[Dict[str, Any]], name: str, metric: str) -> List[Dict[str, Any]]:
    return [r for r in diffs if r["comparison"] == name and r["metric"] == metric
            and r["point"] not in ("full_test", CLEAN_UNIT)]


def _diff_table(diffs: Sequence[Dict[str, Any]], name: str) -> str:
    a_id, b_id = PAIRS[name]
    by_point: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in diffs:
        if r["comparison"] == name:
            by_point.setdefault(r["point"], {})[r["metric"]] = r
    rows = []
    for point, entry in by_point.items():
        acc, f1 = entry["accuracy"], entry["macro_f1"]
        rows.append([acc["label"], _pct(acc["a_value"]), _pct(acc["b_value"]), acc["wording"],
                     _pct(f1["a_value"]), _pct(f1["b_value"]), f1["wording"]])
    head = ["Condition", "%s accuracy" % DISPLAY_NAMES[a_id], "%s accuracy" % DISPLAY_NAMES[b_id],
            "Accuracy change (%s vs %s)" % (b_id, a_id),
            "%s macro-F1" % DISPLAY_NAMES[a_id], "%s macro-F1" % DISPLAY_NAMES[b_id],
            "Macro-F1 change (%s vs %s)" % (b_id, a_id)]
    return _md_table(head, rows)


# ---------------------------------------------------------------- QAT sentence
def qat_sentence(v4_root: Optional[Any], rel_path: str) -> str:
    tail = ("This demonstrates the feasibility of the conversion, serialization and INT8 inference path only; "
            "no QAT performance result is claimed.")
    path = None if v4_root is None else Path(str(v4_root)) / rel_path
    if path is None or not path.is_file():
        return ("The existing tiny QAT feasibility artifact (`%s` under the V4 package) was not available when "
                "this report was generated, so its numbers are omitted. %s" % (rel_path, tail))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ("The QAT feasibility artifact `%s` could not be parsed, so its numbers are omitted. %s" % (rel_path, tail))
    packed = doc.get("packed_dtypes", {}).get("weight_dtype", {})
    layers = sum(int(v) for v in packed.values()) if isinstance(packed, dict) and packed else None
    rows = doc.get("train_rows")
    return (
        "The existing tiny QAT feasibility test (`%s`) reports status '%s', %s converted layers, serialization "
        "flag %s, and ran %s epoch(s) on %s training rows. %s" % (
            rel_path, doc.get("status", "unknown"), "an unknown number of" if layers is None else layers,
            doc.get("serialized", "unknown"), doc.get("epochs", "unknown"),
            "an unknown number of" if rows is None else "{:,}".format(int(rows)), tail)
    )


# ---------------------------------------------------------------- report text
def _ratio_lines(data: Dict[str, Any], tol: float) -> Tuple[List[str], List[List[str]]]:
    eff = data["efficiency_rows"]
    items = (
        ("size (serialized state)", "state_dict_bytes", "size"),
        ("batch-1 latency (median)", "latency_b1_median_ms", "latency"),
        ("batch-256 latency per example (median)", "latency_b256_per_example_median_ms", "latency"),
        ("peak RSS (VmHWM)", "peak_rss_bytes", "memory"),
        ("peak RSS delta above the post-load baseline", "peak_rss_delta_bytes", "memory"),
    )
    sentences, table = [], []
    for parent, child in (("V1", "V2"), ("V3", "V4")):
        for label, key, kind in items:
            r = _ratio(eff[child][key], eff[parent][key])
            text = describe_ratio(r, kind, tol)
            sentences.append("- %s vs %s, %s: %s." % (DISPLAY_NAMES[child], DISPLAY_NAMES[parent], label, text))
            table.append([DISPLAY_NAMES[child], DISPLAY_NAMES[parent], label, text])
    return sentences, table


def build_report(data: Dict[str, Any], cfg: Dict[str, Any], v4_root: Optional[Any],
                 diffs: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    tol = float(cfg["report"]["score_pp_tolerance"])
    rtol = float(cfg["report"]["ratio_tolerance"])
    proto, rob, counts = data["protocol"], data["robustness"], data["counts"]
    ident = data["identity"]
    E, n_sub, k = int(proto["epochs"]), int(rob["n"]), int(rob["per_class_target"])
    n_test = int(data["clean"]["n_test"])
    pgd = proto["robustness"]["attacks"]["pgd5"]
    int8 = proto["robustness"]["methods"]["int8"]
    resolution = 100.0 / n_sub
    nonzero = [eps_str(e) for e in proto["robustness"]["epsilons"] if float(e) != 0.0]
    sel = {int(c): v["selected"] for c, v in counts["counts"].items()}
    short = sorted(c for c, v in counts["counts"].items() if v["selected"] < k)
    per_class = ", ".join("%d:%d" % (c, sel[c]) for c in sorted(sel))
    epoch_note = ("below the %d-epoch recipe (a runtime-based reduction recorded in the budget lock)" % RECIPE_EPOCHS
                  if E < RECIPE_EPOCHS else "equal to the %d-epoch recipe" % RECIPE_EPOCHS)

    L: List[str] = ["# Milestone 2 deadline experiment report", ""]
    L += ["## Setup", "",
          "- Model: PAVE Standard Transformer, frozen split_v2, seed %d (one seed). Two FP32 parents (ordinary and "
          "adversarially trained) trained on the same %d training rows with the same optimizer, batch size and "
          "checkpoint rule (%s); each parent has one INT8 post-training-quantization (PTQ) copy calibrated on "
          "training rows only." % (proto["seed"], proto["split"]["n_train"], proto["checkpoint_rule"]),
          "- Epochs per parent: %d, %s." % (E, epoch_note),
          "- Rows: %d training, %d validation, %d test." % (proto["split"]["n_train"], proto["split"]["n_val"], n_test),
          "- Clean accuracy and macro-F1 use all %d test rows. Robustness uses a frozen stratified test subset of "
          "n=%d rows (target %d per class); the same example IDs are used for every model. Selected rows per class "
          "(class index:count): %s." % (n_test, n_sub, k, per_class),
          "- Attacks (offline, feature space): FGSM; PGD with %d steps, %d start(s), random start %s, step size "
          "%.2f x epsilon; epsilons %s. The epsilon-0 point is computed once on the subset and shared by both curves."
          % (pgd["steps"], pgd["restarts"], "on" if pgd["random_start"] else "off", pgd["alpha_fraction"],
             ", ".join(eps_str(e) for e in proto["robustness"]["epsilons"])),
          "- FP32 models: %s. INT8 models: %s: %s" % (
              proto["robustness"]["methods"]["fp32"]["description"], int8["label"], int8["description"]),
          ""]

    L += ["## Table 1. Four-model comparison", "", comparison_markdown(data, rows).split("\n", 2)[2].rstrip(), ""]

    pts = score_points(data)
    attack_pts = [p for p in pts if p[0] not in ("full_test", CLEAN_UNIT)]
    rob_rows = []
    for m in MODEL_IDS:
        label = DISPLAY_NAMES[m] + (" (%s)" % INT8_LABEL if m in INT8_MODELS else "")
        c = _score(data, m, CLEAN_UNIT)
        rob_rows.append([label, "0.0 (clean subset)", _pct(c["accuracy"]), _pct(c["macro_f1"]),
                         _pct(c["accuracy"]), _pct(c["macro_f1"])])
        for e in nonzero:
            f, p = _score(data, m, unit_id("fgsm", float(e))), _score(data, m, unit_id("pgd5", float(e)))
            rob_rows.append([label, e, _pct(f["accuracy"]), _pct(f["macro_f1"]), _pct(p["accuracy"]), _pct(p["macro_f1"])])
    L += ["## Table 2. Robustness on the subset (n=%d)" % n_sub, "",
          _md_table(["Model", "Epsilon", "FGSM accuracy", "FGSM macro-F1", "PGD-5 accuracy", "PGD-5 macro-F1"], rob_rows),
          "",
          "Accuracy under attack is the accuracy on the perturbed inputs. It can rise on rows the model got wrong "
          "when clean, so the curves are not guaranteed to fall. One subset example is %.2f pp." % resolution, ""]

    # Q1
    agree = {}
    for parent, child in (("V1", "V2"), ("V3", "V4")):
        same = int(np.sum(data["full"]["preds"][parent] == data["full"]["preds"][child]))
        agree[child] = (same, n_test)
    L += ["## Q1. What changed after PTQ?", ""]
    for name in ("q1_ptq_ordinary", "q1_ptq_adversarial"):
        a, b = PAIRS[name]
        full = {r["metric"]: r for r in diffs if r["comparison"] == name and r["point"] == "full_test"}
        hi, lo, same_n = _tally(_attack_records(diffs, name, "accuracy"), tol)
        n_att = hi + lo + same_n
        L += ["**%s vs %s.** On the full test set, accuracy is %s and macro-F1 is %s. The INT8 model and its FP32 "
              "parent predict the same class on %d of %d test rows (%s); %d rows differ. Under attack (FGSM and "
              "PGD-5 at epsilon %s), subset accuracy is higher in %d, lower in %d and essentially unchanged in %d "
              "of %d settings." % (
                  DISPLAY_NAMES[b], DISPLAY_NAMES[a], full["accuracy"]["wording"], full["macro_f1"]["wording"],
                  agree[b][0], agree[b][1], _pct(agree[b][0] / float(agree[b][1])), agree[b][1] - agree[b][0],
                  " and ".join(nonzero), hi, lo, same_n, n_att),
              "", _diff_table(diffs, name), ""]

    # Q2
    name = "q2_adversarial_vs_ordinary_int8"
    full = {r["metric"]: r for r in diffs if r["comparison"] == name and r["point"] == "full_test"}
    hi, lo, same_n = _tally(_attack_records(diffs, name, "accuracy"), tol)
    n_att = hi + lo + same_n
    if hi and not lo:
        verb = "raised"
    elif lo and not hi:
        verb = "lowered"
    elif not hi and not lo:
        verb = "did not change"
    else:
        verb = "had mixed effects on"
    L += ["## Q2. Did adversarial training help the INT8 model?", "",
          "Comparing %s with %s: on the full test set, clean accuracy is %s and macro-F1 is %s. On the attacked "
          "subset, adversarial training %s INT8 accuracy under attack (higher in %d, lower in %d, essentially "
          "unchanged in %d of %d attack settings; differences below %.1f pp count as unchanged)." % (
              DISPLAY_NAMES["V4"], DISPLAY_NAMES["V2"], full["accuracy"]["wording"], full["macro_f1"]["wording"],
              verb, hi, lo, same_n, n_att, tol),
          "", _diff_table(diffs, name), "", BPDA_SENTENCE, ""]

    # Q3
    ratio_lines, _ = _ratio_lines(data, rtol)
    L += ["## Q3. What did we gain or lose in size, speed and memory?", "",
          "Each line compares the INT8 copy with its FP32 parent (ratio = INT8 / FP32; differences within %d%% "
          "count as about the same)." % round(rtol * 100)] + [""] + ratio_lines + [""]

    # Limitations
    e0 = rob["eps0_consistency"]
    e0_text = "; ".join(
        "%s: %s of %s rows differ (max abs logit difference %s)" % (
            m, e0[m]["eps0_vs_full_test_mismatches"], e0[m]["n_compared"],
            "not recorded" if e0[m]["eps0_max_abs_logit_diff"] is None else "%.3g" % e0[m]["eps0_max_abs_logit_diff"])
        for m in MODEL_IDS)
    L += ["## Limitations", "",
          "- One seed (seed %d) and one training run per parent. No cross-seed variation is measured or reported, so "
          "score differences are single-run observations." % proto["seed"],
          "- Actual epochs: %d per parent, %s." % (E, epoch_note),
          "- Small robustness subset: actual n=%d (target %d per class, up to %d). One example equals %.2f pp, so "
          "differences of a few examples are within noise. Classes below the target: %s." % (
              n_sub, k, N_CLASSES * k, resolution, ", ".join(short) if short else "none"),
          "- Fixed attack budget: FGSM and PGD-%d with %d start(s), step size %.2f x epsilon, epsilons %s. Stronger "
          "attacks, more restarts and other attack sources were not run." % (
              pgd["steps"], pgd["restarts"], pgd["alpha_fraction"], ", ".join(nonzero)),
          "- Perturbations are in feature space and do not guarantee valid network traffic.",
          "- INT8 gradients are approximate (%s); robustness numbers for INT8 models may differ under other "
          "gradient estimators." % int8["label"],
          "- Efficiency was measured on one machine with %d thread(s). The memory baseline includes the loaded "
          "weights and the torch footprint, so the RSS delta shows inference-time working memory, not total "
          "footprint." % proto["efficiency"]["threads"],
          "- Recorded epsilon-0 consistency (subset predictions vs full-test predictions of the same model; "
          "record-only, never used to stop the run): %s." % e0_text,
          "- Extra seeds, transfer studies, layer ablations and full QAT training were not run.",
          ""]

    L += ["## QAT feasibility", "", qat_sentence(v4_root, cfg["report"]["qat_artifact"]), ""]

    L += ["## Provenance", "",
          "- run_id: %s" % ident["run_id"],
          "- config_sha256: %s" % ident["config_sha256"],
          "- lock_sha256: %s" % ident["lock_sha256"],
          "- protocol_sha256: %s" % data["protocol_sha256"],
          "- subset_ids_sha256: %s" % rob["subset_ids_sha256"],
          "- v4_manifest_sha256: %s" % ident["v4_manifest_sha256"],
          ""]
    return "\n".join(L)


# ---------------------------------------------------------------- figures
def _figure_bytes(data: Dict[str, Any], attack: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [float(e) for e in data["protocol"]["robustness"]["epsilons"]]
    n = int(data["robustness"]["n"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    try:
        for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro-F1")):
            for m in MODEL_IDS:
                ys = []
                for e in eps:
                    uid = CLEAN_UNIT if e == 0.0 else unit_id(attack, e)
                    ys.append(100.0 * float(data["robustness"]["units"][m][uid][metric]))
                label = DISPLAY_NAMES[m] + (" (%s)" % INT8_LABEL if m in INT8_MODELS else "")
                ax.plot(eps, ys, marker="o", linestyle="--" if m in INT8_MODELS else "-", label=label)
            ax.set_xlabel("epsilon (feature space)")
            ax.set_ylabel("%s (%%)" % title)
            ax.set_xticks(eps)
            ax.set_title("%s, %s" % (ATTACK_LABELS[attack], title))
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("%s robustness curves, test subset n=%d (seed 1, %d epochs)" % (
            ATTACK_LABELS[attack], n, int(data["protocol"]["epochs"])))
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        return buf.getvalue()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- generate
def _archive_previous(out: Path) -> Optional[Path]:
    md = out_path(out, "report_md")
    figures = [out_path(out, "fgsm_figure"), out_path(out, "pgd5_figure")]
    old = [p for p in [md] + figures if p.is_file()]
    if not old:
        return None
    base = out / PATHS["report_dir"] / "old"
    dest = base / utc_stamp()
    n = 1
    while dest.exists():
        n += 1
        dest = base / ("%s-%d" % (utc_stamp(), n))
    dest.mkdir(parents=True)
    for p in old:
        shutil.move(str(p), str(dest / p.name))
    return dest


def generate(out: Any, v4_root: Optional[Any] = None, config_path: Optional[Any] = None) -> Dict[str, Any]:
    """Validate inputs and write ``results/*`` and ``report/*``. Returns the written paths."""
    out = Path(str(out))
    cfg = load_config(config_path or DEFAULT_CONFIG)
    data = load_inputs(out)
    tol = float(cfg["report"]["score_pp_tolerance"])

    rows = comparison_rows(data)
    units = data["robustness"]["units"]
    curves = metrics.build_curves(units, MODEL_IDS, DISPLAY_NAMES,
                                  epsilons=[float(e) for e in data["protocol"]["robustness"]["epsilons"]])
    diffs = compute_differences(data, tol)
    text = build_report(data, cfg, v4_root, diffs, rows)
    figures = {"fgsm_figure": _figure_bytes(data, "fgsm"), "pgd5_figure": _figure_bytes(data, "pgd5")}

    archived = _archive_previous(out)
    atomic_write_text(out_path(out, "model_comparison_csv"), _csv_text(CSV_COLUMNS["model_comparison_csv"], rows))
    atomic_write_text(out_path(out, "model_comparison_md"), comparison_markdown(data, rows))
    atomic_write_text(out_path(out, "robustness_curves_csv"), _csv_text(CSV_COLUMNS["robustness_curves_csv"], curves))
    atomic_write_json(out_path(out, "pp_differences"), {
        "schema": 1, "kind": "m2_deadline_pp_differences", "run_id": data["identity"]["run_id"],
        "unit": "percentage points, (b - a) * 100 rounded to 2 decimals", "tolerance_pp": tol,
        "differences": diffs,
    })
    for key, blob in figures.items():
        atomic_write_bytes(out_path(out, key), blob)
    atomic_write_text(out_path(out, "report_md"), text)
    return {"archived_previous": None if archived is None else str(archived),
            "paths": {k: str(out_path(out, k)) for k in (
                "model_comparison_csv", "model_comparison_md", "robustness_curves_csv", "pp_differences",
                "fgsm_figure", "pgd5_figure", "report_md")}}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m m2_deadline.report",
        description="Generate the four-model table, curves, figures and Markdown report from result files "
                    "under OUT. Never loads a model.")
    p.add_argument("--out", required=True, help="run output folder (artifacts/m2-deadline-v1)")
    p.add_argument("--v4-root", default=os.environ.get("V4_ROOT"),
                   help="V4 package folder (optional; used only to read the QAT feasibility artifact)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate(args.out, args.v4_root)
    except KeyboardInterrupt:
        return 130
    except DeadlineError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return exit_code_for(exc)
    except Exception:
        traceback.print_exc()
        return 1
    for key, path in result["paths"].items():
        print("%s: %s" % (key, path))
    if result["archived_previous"]:
        print("previous report archived to %s" % result["archived_previous"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
