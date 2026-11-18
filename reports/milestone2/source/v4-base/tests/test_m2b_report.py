"""M2B reporting on synthetic predictions (no model, no training, no attack): recomputation, aggregation, plots, efficiency."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from research import m2b_report as rep

SEEDS = (1, 7, 13, 29)
EPS = (0.0, 0.01, 0.025, 0.05, 0.1)
N_PER_CLASS = 10
PRIMARY = {"V1": "fp32_direct", "V3": "fp32_direct", "V2": "bpda_int8", "V4": "bpda_int8"}


def f1_manual(y, pred):
    scores = []
    for c in range(27):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def noisy(rng, y, p):
    wrong = rng.random(len(y)) > p
    return np.where(wrong, (y + rng.integers(1, 27, len(y))) % 27, y)


def write_attack(folder, rng, y, ids, p_clean, p_adv, *, tamper=0.0, valid=True):
    folder.mkdir(parents=True, exist_ok=True)
    pc, pa = noisy(rng, y, p_clean), noisy(rng, y, p_adv)
    np.savez(folder / "predictions.npz", ids=ids, y=y, pred_clean=pc, pred_adv=pa, robust_correct=(pc == y) & (pa == y),
             linf=np.zeros(len(y), dtype=np.float32))
    clean_ok, adv_ok = pc == y, pa == y
    robust = clean_ok & adv_ok
    asr = float((clean_ok & ~adv_ok).sum() / clean_ok.sum()) if clean_ok.sum() else None
    (folder / "attack_result.json").write_text(json.dumps({
        "n": int(len(y)), "accuracy_under_attack": float(adv_ok.mean()) + tamper, "robust_accuracy": float(robust.mean()),
        "attack_success_rate": asr, "valid": valid}), encoding="utf-8")
    return pa


def write_clean(folder, rng, y, ids, p):
    folder.mkdir(parents=True, exist_ok=True)
    pred = noisy(rng, y, p)
    np.savez(folder / "predictions.npz", ids=ids, y=y, pred=pred)
    (folder / "clean_result.json").write_text(json.dumps({"n": int(len(y)), "accuracy": float((pred == y).mean()),
                                                          "macro_f1": f1_manual(y, pred)}), encoding="utf-8")


def build_run(root: Path, phase="validation", seeds=SEEDS, qat=False, extra_source=True):
    y = np.repeat(np.arange(27), N_PER_CLASS)
    ids = np.arange(len(y))
    models = ["V1", "V2", "V3", "V4"] + (["FP32-FT", "FP32-FT-PTQ", "QAT-FT"] if qat else [])
    primary = dict(PRIMARY, **{"FP32-FT": "fp32_direct", "FP32-FT-PTQ": "bpda_int8", "QAT-FT": "bpda_int8"})
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base = root / phase / f"s{seed}"
        for m in models:
            write_clean(base / "clean" / m, rng, y, ids, 0.9 - 0.01 * models.index(m))
            sources = [primary[m]] + (["transfer_fp32_to_int8"] if extra_source and m in ("V2", "V4") else [])
            for source in sources:
                for attack in ("fgsm", "pgd20"):
                    for e in EPS:
                        write_attack(base / "attacks" / m / source / attack / f"eps{e:g}", rng, y, ids, 0.9, 0.9 - 4 * e)


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        lines = [l for l in handle if not l.startswith("#")]
    return list(csv.DictReader(lines))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    build_run(root)
    return root


@pytest.fixture(scope="module")
def built(run_dir):
    rows, arrays = rep.collect(run_dir, "validation")
    return rows, arrays


def test_recomputation_equals_independent_numpy(built):
    rows, arrays = built
    for r in rows:
        y, pred, robust, ids = arrays[(r["seed"], r["model"], r["source"], r["attack"], r["epsilon"])]
        assert r["accuracy"] == pytest.approx(float((pred == y).mean()), abs=1e-12)
        assert r["macro_f1"] == pytest.approx(f1_manual(y, pred), abs=1e-12)
        assert r["n"] == 27 * N_PER_CLASS and r["valid"]


def test_mean_and_sample_sd_equal_numpy_ddof1(built):
    rows, _ = built
    agg = {(a["model"], a["source"], a["attack"], a["epsilon"], a["metric"]): a for a in rep.aggregate(rows)}
    values = [r["accuracy"] for r in rows if (r["model"], r["source"], r["attack"], r["epsilon"]) == ("V1", "fp32_direct", "fgsm", 0.05)]
    assert len(values) == 4
    got = agg[("V1", "fp32_direct", "fgsm", 0.05, "accuracy")]
    assert got["mean"] == pytest.approx(np.mean(values), abs=1e-15)
    assert got["sd"] == pytest.approx(np.std(values, ddof=1), abs=1e-15)
    assert got["sd"] != pytest.approx(np.std(values, ddof=0))
    assert got["n_seeds"] == 4 and sorted(got["per_seed"]) == ["1", "13", "29", "7"]


def test_single_seed_prints_na(tmp_path):
    build_run(tmp_path, seeds=(1,))
    out = tmp_path / "reports" / "validation"
    summary = rep.build_report(tmp_path, "validation")
    text = (out / "results_agg.md").read_text(encoding="utf-8")
    assert "± n/a" in text and "n/a" in (out / "results_agg.csv").read_text(encoding="utf-8")
    assert summary["seeds"] == [1]
    assert rep.fmt(0.5, None) == "0.5000 ± n/a"


def test_paired_differences_are_per_seed(built):
    rows, _ = built
    paired = {(p["pair"], p["attack"], p["epsilon"], p["metric"]): p for p in rep.paired_differences(rows)}
    got = paired[("V3-V1", "pgd20", 0.05, "accuracy")]
    by = {(r["seed"], r["model"]): r["accuracy"] for r in rows if r["attack"] == "pgd20" and r["epsilon"] == 0.05
          and r["source"] == PRIMARY[r["model"]]}
    diffs = [by[(s, "V3")] - by[(s, "V1")] for s in SEEDS]
    assert got["mean"] == pytest.approx(np.mean(diffs)) and got["sd"] == pytest.approx(np.std(diffs, ddof=1))
    assert {p["pair"] for p in paired.values()} == {"V3-V1", "V4-V2", "V4-V3", "V2-V1"}
    assert ("V3-V1", "clean", 0.0, "macro_f1") in paired


def test_full_report_files_banner_and_plots(run_dir):
    summary = rep.build_report(run_dir, "validation")
    out = run_dir / "reports" / "validation"
    for name in ("acc_vs_eps_fgsm.png", "acc_vs_eps_pgd.png", "f1_vs_eps_fgsm.png", "f1_vs_eps_pgd.png"):
        assert (out / name).stat().st_size > 1000 and (out / name.replace(".png", "_points.csv")).is_file()
    assert any(p.startswith("by_source") for p in summary["plots"])
    assert (out / "by_source" / "any_source_robust_vs_eps_pgd.png").is_file()
    for name in ("results_long.csv", "results_agg.csv", "results_agg.md", "per_class.csv", "per_class.md", "invalid_units.md", "blocked_units.md"):
        first = (out / name).read_text(encoding="utf-8").splitlines()[0]
        assert "VALIDATION" in first and "perturbed vectors need not be valid network traffic" in first
    assert "FINAL TEST" not in (out / "results_long.csv").read_text(encoding="utf-8")
    long_rows = read_csv(out / "results_long.csv")
    assert len(long_rows) == len(rep.collect(run_dir, "validation")[0])
    per_class = read_csv(out / "per_class.csv")
    assert {r["weak_per_class_evidence"] for r in per_class} == {"yes"}  # 10 rows per class is below the 30-row floor


def test_test_phase_banner(tmp_path):
    build_run(tmp_path, phase="test", seeds=(1, 7))
    rep.build_report(tmp_path, "test", freeze_sha="a" * 64)
    out = tmp_path / "reports" / "test"
    assert "FINAL TEST" in (out / "results_agg.md").read_text(encoding="utf-8").splitlines()[0]
    assert not (tmp_path / "reports" / "validation").exists()


def test_stored_value_mismatch_raises(tmp_path):
    y, ids = np.repeat(np.arange(27), N_PER_CLASS), np.arange(27 * N_PER_CLASS)
    write_attack(tmp_path / "validation" / "s1" / "attacks" / "V1" / "fp32_direct" / "fgsm" / "eps0.05", np.random.default_rng(0), y, ids,
                 0.9, 0.5, tamper=1e-9)
    with pytest.raises(rep.ReportConsistencyError):
        rep.collect(tmp_path, "validation")


def test_invalid_units_are_excluded_and_listed(tmp_path):
    build_run(tmp_path, seeds=(1, 7), extra_source=False)
    y, ids = np.repeat(np.arange(27), N_PER_CLASS), np.arange(27 * N_PER_CLASS)
    bad = tmp_path / "validation" / "s7" / "attacks" / "V1" / "fp32_direct" / "fgsm" / "eps0.05"
    for f in bad.iterdir():
        f.unlink()
    write_attack(bad, np.random.default_rng(3), y, ids, 0.9, 0.3, valid=False)
    rep.build_report(tmp_path, "validation")
    out = tmp_path / "reports" / "validation"
    agg = {(a["model"], a["source"], a["attack"], a["epsilon"], a["metric"]): a for a in rep.aggregate(rep.collect(tmp_path, "validation")[0])}
    assert agg[("V1", "fp32_direct", "fgsm", 0.05, "accuracy")]["n_seeds"] == 1
    assert "fgsm" in (out / "invalid_units.md").read_text(encoding="utf-8")


def test_qat_attribution_table(tmp_path):
    build_run(tmp_path, seeds=(1, 7), qat=True, extra_source=False)
    rep.build_report(tmp_path, "validation", include_qat=True)
    text = (tmp_path / "reports" / "validation" / "results_agg.md").read_text(encoding="utf-8")
    assert "QAT-FT - FP32-FT->PTQ (QAT benefit)" in text and "FP32-FT - V1 (extra-training effect)" in text
    assert "QAT benefit is the third row only" in text
    assert (tmp_path / "reports" / "validation" / "qat_paired.csv").is_file()
    rep.build_report(tmp_path, "validation", include_qat=False)
    assert "QAT-FT" not in (tmp_path / "reports" / "validation" / "results_long.csv").read_text(encoding="utf-8")


def test_pilot_paths_refused(tmp_path):
    with pytest.raises(rep.ReportError):
        rep.collect(tmp_path / "artifacts" / "m2a-pilot", "validation")


def test_linux_vmhwm_parser_on_fake_proc_status(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t  900000 kB\nVmHWM:\t  123456 kB\nVmRSS:\t  100000 kB\n", encoding="utf-8")
    assert rep.read_peak_memory_status(status.read_text()) == 123456 * 1024
    from research import m2_efficiency as me
    assert me.peak_memory_bytes(status_path=status, system="linux") == (123456 * 1024, "VmHWM")
    assert rep.read_peak_memory_status("Name:\tx\n") is None


def fake_efficiency(root, seeds=SEEDS, variants=("V1", "V2", "V3", "V4")):
    rng = np.random.default_rng(5)
    for seed in seeds:
        folder = root / "efficiency" / f"s{seed}"
        folder.mkdir(parents=True)
        (folder / "hardware.json").write_text(json.dumps({"cpu_model": "Fake CPU", "nproc": 8, "mem_total": "16 GB", "kernel": "6.1",
                                                          "torch": "2.5.1+cpu", "quantized_engine": "fbgemm", "busy_override": False}), encoding="utf-8")
        for threads in (1, 4):
            meas = []
            for v in variants:
                for _ in range(3):
                    meas.append({"variant": v, "result": {
                        "state_dict_bytes": 1000 + 10 * variants.index(v), "peak_memory_metric": "VmHWM",
                        "batch1": {"median_ms": 1 + rng.random(), "p95_ms": 2 + rng.random()},
                        "batch256": {"median_ms": 10 + rng.random(), "p95_ms": 12 + rng.random()},
                        "peak_wset_minus_baseline_bytes": int(1e6 + rng.integers(0, 1000))}})
            pass_dir = folder / f"threads{threads}"
            pass_dir.mkdir()
            (pass_dir / "efficiency.json").write_text(json.dumps({"variants": list(variants), "measurements": meas,
                                                                  "peak_memory_metric": ["VmHWM"]}), encoding="utf-8")


def test_efficiency_aggregation_and_report(tmp_path):
    fake_efficiency(tmp_path)
    collected, hardware = rep.collect_efficiency(tmp_path)
    agg = rep.aggregate_efficiency(collected)
    per_seed = [collected[s][1]["per_variant"]["V1"]["batch1_median_ms"] for s in SEEDS]
    mean, sd, n = agg[1]["V1"]["batch1_median_ms"]
    assert mean == pytest.approx(np.mean(per_seed)) and sd == pytest.approx(np.std(per_seed, ddof=1)) and n == 4
    rep.build_report(tmp_path, "efficiency")
    text = (tmp_path / "reports" / "efficiency" / "efficiency.md").read_text(encoding="utf-8")
    assert "VmHWM" in text and "not comparable" in text and "No speed or memory advantage is assumed" in text
    assert "Fake CPU" in text and "QAT arms" in text and "1 thread(s)" in text and "4 thread(s)" in text
    assert (tmp_path / "reports" / "efficiency" / "efficiency_agg.csv").is_file()


def test_efficiency_without_results_is_refused(tmp_path):
    with pytest.raises(rep.ReportError):
        rep.collect_efficiency(tmp_path)


def test_report_cli_help_runs(capsys):
    from research import m2b_campaign
    for command in ("report", "preflight", "smoke"):
        assert m2b_campaign.main([command, "--help"]) == 0
    assert "--phase" in capsys.readouterr().out
