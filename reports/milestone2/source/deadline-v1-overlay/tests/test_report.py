"""Report generation on a synthetic OUT folder built from the common.py contract. No torch, no model."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import OVERLAY, fake_efficiency_cfg, make_split_and_csv

from m2_deadline import markers, metrics, protocol, report
from m2_deadline.common import (
    ATTACK_UNITS,
    CSV_COLUMNS,
    DISPLAY_NAMES,
    MODEL_IDS,
    DataIntegrityError,
    GuardError,
    PATHS,
    atomic_write_json,
    check_required_keys,
    eps_str,
    out_path,
    sha256_file,
    unit_id,
)

pytest.importorskip("pandas", reason="synthetic split needs read_test_labels (reported skip, not a pass)")

CLEAN_RATE = {"V1": 0.95, "V2": 0.90, "V3": 0.93, "V4": 0.88}
DROP = {("fgsm", 0.05): 0.15, ("fgsm", 0.1): 0.30, ("pgd5", 0.05): 0.25, ("pgd5", 0.1): 0.50}
DROP_MULT = {"V1": 1.0, "V2": 1.0, "V3": 0.5, "V4": 0.5}
EFF_B1 = {"V1": 2.0, "V2": 1.5, "V3": 2.0, "V4": 1.5}
EFF_SIZE = {"V1": 400000, "V2": 130000, "V3": 400000, "V4": 130000}
EFF_RSS = {"V1": 300e6, "V2": 280e6, "V3": 300e6, "V4": 280e6}


def build_out(out, tmp_path, config, *, epochs=10, k=5, clean_rate=None, copy_parent=(), eff_b1=None, seed=3):
    """Write every file report.generate reads, using only the contract tables of common.py."""
    clean_rate = dict(CLEAN_RATE if clean_rate is None else clean_rate)
    eff_b1 = dict(EFF_B1 if eff_b1 is None else eff_b1)
    split, csv_path = make_split_and_csv(tmp_path)
    test_ids, y_test = protocol.read_test_labels(split, None, csv_path)
    config_sha, lock_sha = "c" * 64, "l" * 64
    freeze = protocol.build_freeze(
        config=config, config_sha256=config_sha, lock={"epochs": epochs, "per_class_target": k}, lock_sha256=lock_sha,
        split=split, split_m2_sha256="s" * 64, scaler_sha256="m" * 64, efficiency_cfg=fake_efficiency_cfg(),
        test_ids=test_ids, y_test=y_test,
    )
    protocol.write_protocol(out, freeze, "2026-01-01T00:00:00Z")
    proto_sha = freeze["protocol_sha256"]
    sub_ids = np.asarray(freeze["subset"]["ids"], dtype=np.int64)
    sub_y = np.asarray(freeze["subset"]["labels"], dtype=np.int64)
    n_sub = len(sub_ids)
    rng = np.random.default_rng(seed)

    def noisy(y, rate):
        wrong = rng.random(len(y)) > rate
        return np.where(wrong, (y + 1 + rng.integers(0, 26, len(y))) % 27, y)

    full_pred, unit_pred = {}, {}
    for m in MODEL_IDS:
        full_pred[m] = noisy(y_test, clean_rate[m])
        unit_pred[m] = {}
        for attack, eps in ATTACK_UNITS:
            rate = clean_rate[m] if attack == "clean" else max(0.02, clean_rate[m] - DROP[(attack, eps)] * DROP_MULT[m])
            unit_pred[m][(attack, eps)] = noisy(sub_y, rate)
    for m in copy_parent:
        parent = "V1" if m == "V2" else "V3"
        full_pred[m] = full_pred[parent].copy()
        unit_pred[m] = {k_: v.copy() for k_, v in unit_pred[parent].items()}

    clean_models = {m: {"accuracy": metrics.accuracy(y_test, full_pred[m]), "macro_f1": metrics.macro_f1(y_test, full_pred[m]),
                        "n": len(y_test)} for m in MODEL_IDS}
    atomic_write_json(out_path(out, "clean_results"), {
        "schema": 1, "kind": "m2_deadline_clean", "run_id": "m2-deadline-v1", "n_test": len(y_test), "labels": 27,
        "batch_size": 64, "models": clean_models})
    with open(str(out_path(out, "predictions_full_test")), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS["predictions_full_test"])
        flagged = set(int(i) for i in sub_ids)
        for pos, i in enumerate(test_ids):
            w.writerow([int(i), int(y_test[pos]), "true" if int(i) in flagged else "false"]
                       + [int(full_pred[m][pos]) for m in MODEL_IDS])

    units = {}
    for m in MODEL_IDS:
        units[m] = {}
        for attack, eps in ATTACK_UNITS:
            p = unit_pred[m][(attack, eps)]
            units[m][unit_id(attack, eps)] = {
                "n": n_sub, "accuracy": metrics.accuracy(sub_y, p), "macro_f1": metrics.macro_f1(sub_y, p),
                "robust_accuracy": 0.5 * metrics.accuracy(sub_y, p)}
    atomic_write_json(out_path(out, "robustness_results"), {
        "schema": 1, "kind": "m2_deadline_robustness", "run_id": "m2-deadline-v1", "n": n_sub, "per_class_target": k,
        "subset_ids_sha256": freeze["subset"]["ids_sha256"], "epsilons": [0.0, 0.05, 0.1], "units": units,
        "eps0_consistency": {m: {"eps0_vs_full_test_mismatches": 1, "eps0_max_abs_logit_diff": 2.5e-6, "n_compared": n_sub}
                             for m in MODEL_IDS}})
    with open(str(out_path(out, "predictions_subset")), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS["predictions_subset"])
        for m in MODEL_IDS:
            for attack, eps in ATTACK_UNITS:
                for i, yv, pv in zip(sub_ids, sub_y, unit_pred[m][(attack, eps)]):
                    w.writerow([m, attack, eps_str(eps), int(i), int(yv), int(pv)])

    measurements = []
    index = 0
    for rep in (1, 2, 3):
        for m in MODEL_IDS:
            j = 1.0 + 0.01 * rep
            measurements.append({"index": index, "repetition": rep, "variant": m, "retry": False, "result": {
                "state_dict_bytes": EFF_SIZE[m], "peak_wset_bytes": EFF_RSS[m] * j,
                "peak_wset_minus_baseline_bytes": 20e6 * j, "sampled_peak_minus_baseline_bytes": 21e6 * j,
                "batch1": {"median_ms": eff_b1[m] * j, "p95_ms": eff_b1[m] * j * 1.5, "batch_size": 1},
                "batch256": {"median_ms": 100.0 * j, "p95_ms": 120.0 * j, "batch_size": 256}}})
            index += 1
    atomic_write_json(out_path(out, "efficiency"), {
        "any_failed": False, "interleaved": True, "repetitions": 3, "threads": 1, "variants": list(MODEL_IDS),
        "measurements": measurements, "summary": {}, "peak_memory_metric": ["VmHWM"]})

    ident = markers.build_run_identity(
        config_sha256=config_sha, lock_sha256=lock_sha, epochs=epochs, protocol_sha256=proto_sha,
        v4_manifest_sha256="v" * 64, source_sha256={"m2_training.py": "a" * 64})
    markers.write_run_identity(out, ident)
    ctx = markers.context(config_sha256=config_sha, lock_sha256=lock_sha, protocol_sha256=proto_sha, epochs=epochs)
    markers.write_after_validate(out, "test_clean", lambda: True,
                                 [PATHS["clean_results"], PATHS["predictions_full_test"]], ctx)
    markers.write_after_validate(out, "robustness", lambda: True,
                                 [PATHS["robustness_results"], PATHS["predictions_subset"]], ctx)
    markers.write_after_validate(out, "efficiency", lambda: True, [PATHS["efficiency"]], ctx)
    return {"clean": clean_models, "units": units, "n_sub": n_sub, "n_test": len(y_test), "ctx": ctx,
            "full_pred": full_pred}


@pytest.fixture
def v4():
    """Report tests do not need the real V4 package; the QAT artifact is faked where needed."""
    return None


@pytest.fixture
def built(out, tmp_path, config):
    return build_out(out, tmp_path, config)


def read_report(out):
    return out_path(out, "report_md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- wording helpers
def test_describe_pp_wording_flips_with_sign_and_tolerance():
    assert report.describe_pp(1.25) == "higher by 1.25 pp"
    assert report.describe_pp(-1.25) == "lower by 1.25 pp"
    assert report.describe_pp(0.49) == "essentially unchanged (+0.49 pp)"
    assert report.describe_pp(-0.49) == "essentially unchanged (-0.49 pp)"
    assert report.describe_pp(0.5) == "higher by 0.50 pp"
    assert report.describe_pp(0.0) == "essentially unchanged (+0.00 pp)"
    assert report.describe_pp(0.8, tol=1.0).startswith("essentially unchanged")


def test_describe_ratio_thresholds_and_words():
    assert report.describe_ratio(1.05, "size").startswith("about the same")
    assert report.describe_ratio(0.95, "latency").startswith("about the same")
    assert report.describe_ratio(0.5, "size") == "50.0% smaller (ratio 0.50x)"
    assert report.describe_ratio(1.5, "memory") == "50.0% larger (ratio 1.50x)"
    assert report.describe_ratio(0.5, "latency") == "50.0% faster (ratio 0.50x)"
    assert report.describe_ratio(2.0, "latency") == "100.0% slower (ratio 2.00x)"
    assert report.describe_ratio(None, "size") == "not available"
    assert report.describe_ratio(float("nan"), "size") == "not available"


# ---------------------------------------------------------------- generation
def test_generate_writes_tables_curves_and_report(out, built, v4):
    result = report.generate(out, v4)
    assert result["archived_previous"] is None
    for key in ("model_comparison_csv", "model_comparison_md", "robustness_curves_csv", "pp_differences", "report_md"):
        assert out_path(out, key).is_file()

    rows = list(csv.DictReader(open(str(out_path(out, "model_comparison_csv")), encoding="utf-8")))
    assert tuple(rows[0].keys()) == CSV_COLUMNS["model_comparison_csv"]
    assert [r["model_id"] for r in rows] == list(MODEL_IDS)
    v1 = rows[0]
    assert v1["epochs"] == "10" and v1["state_dict_bytes"] == "400000" and v1["parent"] == ""
    assert float(v1["clean_accuracy_full_test"]) == built["clean"]["V1"]["accuracy"]
    assert float(v1["latency_b1_median_ms"]) == pytest.approx(2.0 * 1.02)  # median of 3 repetitions
    assert float(v1["latency_b256_per_example_median_ms"]) == pytest.approx(100.0 * 1.02 / 256)
    assert int(v1["peak_rss_bytes"]) == int(300e6 * 1.02)
    assert rows[1]["parent"] == "V1" and rows[3]["parent"] == "V3"

    curves = list(csv.DictReader(open(str(out_path(out, "robustness_curves_csv")),
                                      encoding="utf-8")))
    assert tuple(curves[0].keys()) == CSV_COLUMNS["robustness_curves_csv"]
    assert len(curves) == 4 * 2 * 3
    eps0 = [r for r in curves if r["epsilon"] == "0.0"]
    assert len(eps0) == 8 and all(r["eps0_shared"] == "true" and r["unit_id"] == "clean_eps0.0" for r in eps0)
    assert {r["epsilon"] for r in curves} == {"0.0", "0.05", "0.1"}
    assert all(r["eps0_shared"] == "false" for r in curves if r["epsilon"] != "0.0")
    for m in MODEL_IDS:
        a = [r for r in eps0 if r["model_id"] == m]
        assert a[0]["accuracy"] == a[1]["accuracy"] and {r["attack"] for r in a} == {"fgsm", "pgd5"}

    md = read_report(out)
    for heading in ("## Setup", "## Table 1.", "## Table 2.", "## Q1.", "## Q2.", "## Q3.", "## Limitations",
                    "## QAT feasibility", "## Provenance"):
        assert heading in md
    assert not re.search(r"(?<![\d.])0\.10(?!\d)", md)  # one epsilon string form: 0.1
    assert "approx. BPDA gradient" in md and report.BPDA_SENTENCE in md
    assert "c" * 64 in md and "l" * 64 in md  # config and lock hashes


def test_report_states_seed_epochs_subset_n_and_resolution(out, built, v4):
    report.generate(out, v4)
    md = read_report(out)
    n = built["n_sub"]
    assert "one seed" in md.lower() and "seed 1" in md
    assert "Epochs per parent: 10" in md and "below the 25-epoch recipe" in md
    assert "actual n=%d" % n in md and "n=%d" % n in md
    assert "%.2f pp" % (100.0 / n) in md
    assert "feature space" in md and "BPDA" in md and "PGD with 5 steps, 1 start(s)" in md
    assert "single-run observations" in md and "one machine" in md
    assert "record-only" in md and "2.5e-06" in md


def test_no_standard_deviation_text(out, built, v4):
    report.generate(out, v4)
    for key in ("report_md", "model_comparison_md"):
        text = out_path(out, key).read_text(encoding="utf-8").lower()
        assert "standard deviation" not in text
        assert not re.search(r"\bstd\b", text) and "±" not in text


def test_pp_values_in_text_are_computed_from_inputs(out, built, v4):
    report.generate(out, v4)
    md = read_report(out)
    clean = built["clean"]
    d = metrics.pp_diff(clean["V1"]["accuracy"], clean["V2"]["accuracy"])
    assert report.describe_pp(d) in md
    d2 = metrics.pp_diff(clean["V2"]["accuracy"], clean["V4"]["accuracy"])
    assert report.describe_pp(d2) in md
    diffs = json.loads(out_path(out, "pp_differences").read_text(encoding="utf-8"))
    check_required_keys(diffs, "pp_differences")
    rec = [r for r in diffs["differences"] if r["comparison"] == "q1_ptq_ordinary" and r["point"] == "full_test"
           and r["metric"] == "accuracy"][0]
    assert rec["pp"] == d and rec["wording"] == report.describe_pp(d)
    assert len(diffs["differences"]) == 3 * 6 * 2  # 3 comparisons x (full, clean subset, 4 attack points) x 2 metrics
    assert all(isinstance(r["pp"], float) for r in diffs["differences"])


def test_text_changes_when_inputs_change(tmp_path, config, v4):
    a_out, b_out = tmp_path / "a" / "out", tmp_path / "b" / "out"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a_out.mkdir()
    b_out.mkdir()
    build_out(a_out, tmp_path / "a", config, clean_rate={"V1": 0.95, "V2": 0.55, "V3": 0.93, "V4": 0.98})
    build_out(b_out, tmp_path / "b", config, epochs=5, clean_rate={"V1": 0.55, "V2": 0.95, "V3": 0.93, "V4": 0.60},
              eff_b1={"V1": 2.0, "V2": 4.0, "V3": 2.0, "V4": 1.0})
    report.generate(a_out, v4)
    report.generate(b_out, v4)
    a, b = read_report(a_out), read_report(b_out)
    pat1 = r"INT8-PTQ-ordinary vs FP32-ordinary\.\*\* On the full test set, accuracy is (higher|lower) by"
    assert re.search(pat1, a).group(1) == "lower" and re.search(pat1, b).group(1) == "higher"
    pat2 = r"adversarial training (raised|lowered|did not change|had mixed effects on) INT8 accuracy under attack"
    assert re.search(pat2, a) and re.search(pat2, b)
    assert "Epochs per parent: 10" in a and "Epochs per parent: 5" in b
    # latency ratio of V2 vs V1: 1.5/2.0 (faster) in a, 4.0/2.0 (slower) in b
    line = "INT8-PTQ-ordinary vs FP32-ordinary, batch-1 latency (median): "
    assert line + "25.0% faster (ratio 0.75x)" in a and line + "100.0% slower (ratio 2.00x)" in b
    assert a != b


def test_equal_predictions_are_described_as_essentially_unchanged(out, tmp_path, config, v4):
    build_out(out, tmp_path, config, copy_parent=("V2",))
    report.generate(out, v4)
    md = read_report(out)
    assert "accuracy is essentially unchanged (+0.00 pp) and macro-F1 is essentially unchanged (+0.00 pp)" in md
    assert re.search(r"predict the same class on (\d+) of \1 test rows \(100\.00%\); 0 rows differ", md)


def test_qat_sentence_with_and_without_artifact(out, built, tmp_path):
    root = tmp_path / "v4"
    art = root / "artifacts" / "m2a-pilot" / "qat"
    art.mkdir(parents=True)
    (art / "qat_feasibility.json").write_text(json.dumps({
        "status": "supported", "serialized": True, "epochs": 1, "train_rows": 2700,
        "packed_dtypes": {"weight_dtype": {"torch.qint8": 23}},
        "int8_inference": {"accuracy_on_eval_rows": 0.1796}}), encoding="utf-8")
    report.generate(out, root)
    with_art = read_report(out)
    assert "status 'supported', 23 converted layers, serialization flag True, and ran 1 epoch(s) on 2,700" in with_art
    assert "feasibility of the conversion, serialization and INT8 inference path only" in with_art
    assert "no QAT performance result is claimed" in with_art
    assert "17.96" not in with_art and "0.1796" not in with_art

    report.generate(out, None)
    without = read_report(out)
    assert "was not available" in without and "converted layers" not in without
    assert "no QAT performance result is claimed" in without
    assert "m2a-pilot/qat/qat_feasibility.json" in without


def test_figures_are_png_files(out, built, v4):
    pytest.importorskip("matplotlib", reason="figure assertion skipped without matplotlib (reported)")
    report.generate(out, v4)
    for key in ("fgsm_figure", "pgd5_figure"):
        data = out_path(out, key).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


def test_rerun_archives_previous_report(out, built, v4):
    report.generate(out, v4)
    first = read_report(out)
    r2 = report.generate(out, v4)
    assert r2["archived_previous"] is not None
    old_dirs = list((out / "report" / "old").iterdir())
    assert len(old_dirs) == 1
    assert (old_dirs[0] / "m2_deadline_report.md").read_text(encoding="utf-8") == first
    report.generate(out, v4)
    assert len(list((out / "report" / "old").iterdir())) == 2


def test_only_results_and_report_are_written(out, built, v4):
    def snapshot():
        return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob("*") if p.is_file() and p.relative_to(out).parts[0] not in ("results", "report")}

    before = snapshot()
    report.generate(out, v4)
    report.generate(out, v4)
    assert snapshot() == before


# ---------------------------------------------------------------- failures and exit codes
def test_missing_input_exits_2_and_lists_the_file(out, built, capsys):
    out_path(out, "predictions_subset").unlink()
    assert report.main(["--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "missing input files" in err and "predictions_subset.csv" in err
    assert not out_path(out, "report_md").exists()


def test_invalid_marker_exits_3(out, built, capsys):
    out_path(out, "robustness_results").write_text(
        out_path(out, "robustness_results").read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert report.main(["--out", str(out)]) == 3
    assert "robustness" in capsys.readouterr().err
    markers.clear(out, "efficiency")
    with pytest.raises(GuardError, match="efficiency: no marker"):
        report.load_inputs(out)


def test_changed_protocol_exits_3(out, built):
    p = out_path(out, "protocol")
    p.write_text(p.read_text(encoding="utf-8").replace('"seed": 1', '"seed": 2'), encoding="utf-8")
    assert report.main(["--out", str(out)]) == 3


def test_inconsistent_predictions_are_rejected(out, built):
    path = out_path(out, "robustness_results")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["units"]["V1"]["clean_eps0.0"]["accuracy"] = 0.123
    atomic_write_json(path, doc)
    # re-issue the marker so only the cross-check can fail
    ctx = built["ctx"]
    markers.write_after_validate(out, "robustness", lambda: True,
                                 [PATHS["robustness_results"], PATHS["predictions_subset"]], ctx)
    with pytest.raises(DataIntegrityError, match="differs from predictions_subset.csv"):
        report.load_inputs(out)
    assert report.main(["--out", str(out)]) == 2


def test_subset_ids_must_match_for_every_model(out, built):
    path = out_path(out, "predictions_subset")
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("V3,"))
    parts = lines[idx].split(",")
    parts[3] = "999999"
    lines[idx] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx = built["ctx"]
    markers.write_after_validate(out, "robustness", lambda: True,
                                 [PATHS["robustness_results"], PATHS["predictions_subset"]], ctx)
    with pytest.raises(DataIntegrityError, match="frozen subset"):
        report.load_inputs(out)


# ---------------------------------------------------------------- imports and CLI
def test_help_exits_0_without_importing_torch():
    code = (
        "import sys\n"
        "from m2_deadline import report\n"
        "try:\n"
        "    report.main(['--help'])\n"
        "except SystemExit as e:\n"
        "    rc = e.code\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "sys.exit(rc)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(OVERLAY), capture_output=True, text=True,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert res.returncode == 0, res.stderr
    assert "--out" in res.stdout and "--v4-root" in res.stdout
    res2 = subprocess.run([sys.executable, "-m", "m2_deadline.report", "--help"], cwd=str(OVERLAY),
                          capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert res2.returncode == 0 and "usage" in res2.stdout.lower()


def test_generate_never_imports_torch(out, built, v4):
    report.generate(out, v4)
    assert "torch" not in sys.modules
    assert not any(name == "research" or name.startswith("research.") for name in sys.modules)


def test_main_requires_out(capsys):
    with pytest.raises(SystemExit) as exc:
        report.main([])
    assert exc.value.code == 2


def test_display_names_are_used_in_the_table(out, built, v4):
    report.generate(out, v4)
    text = out_path(out, "model_comparison_md").read_text(encoding="utf-8")
    for m in MODEL_IDS:
        assert DISPLAY_NAMES[m] in text
    assert "Peak RSS" in text and "VmHWM" in text and "1,000,000 bytes" in text
    assert sha256_file(out_path(out, "model_comparison_csv"))
