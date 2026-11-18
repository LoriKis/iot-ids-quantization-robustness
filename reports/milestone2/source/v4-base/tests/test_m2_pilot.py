"""Pilot support code: subset selection, estimate arithmetic and report helpers (no training, no real data)."""
import csv

import numpy as np
import pytest

from m2_helpers import split_arrays
from research import m2_estimate as est
from research import m2_pilot as mp
from research import m2_report as rep


def test_make_subset_is_stratified_deterministic_and_drawn_from_the_partition():
    part = split_arrays(per_class=150)["train"]
    a = mp.make_subset(part, 100, 7)
    b = mp.make_subset(part, 100, 7)
    c = mp.make_subset(part, 100, 8)
    assert len(a["ids"]) == 2700 and np.bincount(a["y"]).tolist() == [100] * 27
    assert np.array_equal(a["ids"], b["ids"]) and not np.array_equal(a["ids"], c["ids"])
    assert set(a["ids"].tolist()) <= set(part["ids"].tolist())
    assert np.array_equal(a["x"], part["x"][a["local"]])
    nested = mp.make_subset({"ids": a["ids"], "x": a["x"], "y": a["y"]}, 10, 3)
    assert len(nested["ids"]) == 270 and set(nested["ids"].tolist()) <= set(a["ids"].tolist())
    described = mp.describe(nested)
    assert described["n"] == 270 and described["per_class_counts"] == [10] * 27 and len(described["id_sha256"]) == 64


def _log(step_seconds, counters, batch_sizes=(512, 140), epochs=2):
    return {"batch_sizes_epoch_1": list(batch_sizes), "epochs": epochs, "step_seconds": step_seconds,
            "pass_counters_per_step": counters, "attack_seconds": [0.0] + [s / 2 for s in step_seconds[1:]]}


ORDINARY = {"attack_grad_passes_per_step": 0, "attack_scoring_forwards_per_step": 0, "train_forward_backward_per_step": 1}
ADVERSARIAL = {"attack_grad_passes_per_step": 6, "attack_scoring_forwards_per_step": 8, "train_forward_backward_per_step": 2}


def test_per_row_step_seconds_discards_the_first_step():
    log = _log([100.0, 5.12, 1.40, 5.12], ORDINARY)  # the 100 s first step must not count
    assert est.per_row_step_seconds(log) == pytest.approx((5.12 + 1.40 + 5.12) / (140 + 512 + 140))
    with pytest.raises(ValueError):
        est.per_row_step_seconds(_log([1.0], ORDINARY, batch_sizes=(512,), epochs=1))
    with pytest.raises(ValueError):
        est.per_row_step_seconds({**log, "step_seconds": [1.0, 2.0]})


def test_nominal_cost_ratio_is_the_documented_32_over_3():
    assert est.nominal_step_cost(ORDINARY) == 3.0
    assert est.nominal_step_cost(ADVERSARIAL) == 2 * 3 + 6 * 3 + 8
    ratio = est.training_cost_ratio(_log([9.0, 1.0, 1.0, 1.0], ORDINARY), _log([9.0, 6.0, 6.0, 6.0], ADVERSARIAL))
    assert ratio["nominal_pass_count_ratio"] == pytest.approx(32 / 3)
    assert ratio["measured_wall_clock_ratio"] == pytest.approx(6.0) and ratio["steps_compared"] == 3
    assert ratio["first_step_discarded"] is True
    with pytest.raises(ValueError):
        est.training_cost_ratio(_log([1.0, 1.0], ORDINARY, epochs=1), _log([1.0, 1.0, 1.0, 1.0], ADVERSARIAL))


def test_attack_cost_and_budget_ratio_helpers():
    rows = [{"attack_source": "bpda_int8", "attack": "pgd20", "epsilon": e, "n": 10, "attack_seconds": s}
            for e, s in ((0, 0.1), (0.01, 10.0), (0.05, 30.0))]
    cost = est.attack_seconds_per_example(rows)[("bpda_int8", "pgd20")]
    assert cost == {"eps0": pytest.approx(0.01), "eps_positive": pytest.approx(2.0)}
    cells = [{"seconds_pgd20": 10.0, "seconds_pgd50x5": 40.0}, {"seconds_pgd20": 30.0, "seconds_pgd50x5": 150.0},
             {"seconds_pgd20": None, "seconds_pgd50x5": 1.0}]
    ratio = est.budget_ratio(cells)
    assert ratio["n_cells"] == 2 and ratio["ratio"] == pytest.approx(190 / 40) and ratio["min"] == 4.0 and ratio["max"] == 5.0
    assert est.budget_ratio([]) == {"ratio": None, "n_cells": 0}


def test_banner_and_framing_are_the_frozen_sentences():
    assert mp.BANNER == est.BANNER == rep.BANNER
    assert mp.FRAMING.startswith("This is a new group-disjoint evaluation partition within a previously examined dataset")
    assert "not a completely untouched external benchmark" in rep.FRAMING


def test_attack_csv_and_table_helpers(tmp_path):
    rows = [{"model_id": "V2", "attack_source": "bpda_int8", "target": "V2", "attack": "pgd20", "epsilon": 0.05, "n": 270,
             "clean_accuracy": 0.5, "accuracy_under_attack": 0.1, "robust_accuracy": 0.1, "attack_success_rate": None,
             "max_linf": 0.05, "attack_seconds": 1.5, "valid": True, "note": "n", "unrelated": 1}]
    path = tmp_path / "attack_results.csv"
    rep.attack_results_csv(rows, path)
    read = list(csv.DictReader(path.open()))
    assert read[0]["attack_source"] == "bpda_int8" and read[0]["epsilon"] == "0.05" and "unrelated" not in read[0]
    with pytest.raises(FileExistsError):
        rep.attack_results_csv(rows, path)
    text = rep.table(("a", "b"), [(1, 0.123456), (None, True)])
    assert text.splitlines()[0] == "| a | b |" and "0.1235" in text and "n/a" in text and "yes" in text


def _write(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _synthetic_pilot(root):
    """Minimal pilot directory with every file and key the report reads (checks the file schema, not numbers)."""
    sha = "a" * 64
    summary = {"n": 10, "id_sha256": sha}
    spread = {"median": 1.5, "min": 1.0, "max": 2.0, "n": 2}
    _write(root / "pilot_runtime.json", {
        "seed": 1, "epochs": 2, "batch_size": 512, "threads": {"training": 2, "attack_and_evaluation": 1},
        "provenance": {"source_digest": sha, "revision": "r"}, "phases": {"attacks": 1.0},
        "at_cost": {"measured_wall_clock_ratio": 7.0, "nominal_pass_count_ratio": 32 / 3, "steps_compared": 11},
        "campaign_e4_subset": {"n_subset": 1727, "rule": "rule"}, "peak_rss_bytes": 10 ** 9, "min_available_ram_bytes": 10 ** 6})
    _write(root / "subsets.json", {"split_seed": 1, "split_manifest_sha256": sha, **{k: summary for k in
           ("train", "validation", "attack", "calibration", "e4_plumbing")}})
    _write(root / "clean_eval.json", {"models": {v: {"accuracy": 0.1, "n": 5} for v in ("V1", "V2", "V3", "V4")},
                                      "int8_agreement": {"V2_vs_V1": 1.0, "V4_vs_V3": 1.0}})
    rows = [{"model_id": m, "attack": a, "epsilon": e, "attack_source": s, "robust_accuracy": 0.1}
            for m, s in (("V1", "fp32_direct"), ("V2", "bpda_int8"), ("V3", "fp32_direct"), ("V4", "bpda_int8"))
            for a in ("fgsm", "pgd20") for e in (0, 0.05)]
    _write(root / "attack_results.json", {"banner": "b", "framing": "f", "rows": rows, "invalid_units": [], "union_rows": [
        {"target": "V2", "attack": "pgd20", "epsilon": 0.05, "robust_accuracy": 0.1}]})
    diag = {"direct_int8_grad": {"behaviour": "raises", "undefined_or_zero_fraction": 1.0, "n": 5, "counted_as_robust": False},
            "surrogate_fidelity": {"note": "note", "max_logit_diff_to_int8": 0.1, "argmax_agreement_with_int8": 1.0}}
    _write(root / "gradient_diagnostics.json", {"banner": "b", "framing": "f", "V2": diag, "V4": diag})
    _write(root / "attack_budget_check.json", {"note": "plumbing", "n_subset": 54, "threshold": 1, "cell_costs": [
        {"model": "V1", "epsilon": 0.025, "n_extra": 0, "seconds_pgd20": 1.0, "seconds_pgd50x5": 4.0}]})
    stats = {"state_dict_bytes": spread, "batch1_median_ms": spread, "batch1_p95_ms": spread, "batch256_median_ms": spread,
             "batch256_throughput_samples_per_s": spread, "baseline_rss_bytes": spread, "peak_wset_minus_baseline_bytes": spread}
    _write(root / "efficiency/efficiency.json", {
        "run_order": [{"variant": v, "retry": False} for v in ("V1", "V2", "V3", "V4")], "interleaved": True, "repetitions": 1,
        "warmup_and_timed": {}, "summary": {v: stats for v in ("V1", "V2", "V3", "V4")}, "no_claim": "no claim"})
    _write(root / "qat/qat_feasibility.json", {"status": "supported", "stage": "inference", "epochs": 1, "train_rows": 5,
           "seconds": 1.0, "error": None, "module": None, "packed_dtypes": {"weight_dtype": {"torch.qint8": 23}},
           "int8_inference": {"rows": 5}, "claim": "claim"})
    counters = {"attack_grad_passes_per_step": 6, "attack_scoring_forwards_per_step": 8, "train_forward_backward_per_step": 2}
    for v in ("V1", "V3"):
        _write(root / f"models/{v}/train_log.json", {
            "optimizer_updates": 12, "wall_clock_seconds": 1.0, "step_seconds": [1.0, 1.0], "epoch_losses": [3.0, 2.9],
            "init_state_hash": sha, "final_state_hash": sha, "pass_counters_per_step": counters, "first_step_hook_counts": {}})
    for v in ("V2", "V4"):
        _write(root / f"models/{v}/conversion.json", {
            "backend": "fbgemm", "mode": "static", "scope": "all_linear", "calibration": {"n_rows": 540, "id_sha256": sha},
            "requested": ["a"] * 23, "classification_path_islands": 21, "anomaly_scorer_islands": 2,
            "structure_checks": {"weight_dtype_inventory": {"torch.qint8": 23}, "weight_qscheme_inventory": {"x": 23},
                                 "profiler": {"quantized_linear_calls": 23}},
            "architecture_wrapper_fp32_max_abs_error": 0.0,
            "reload_check": {"logits_bitwise_equal": True, "argmax_identical": True, "rows": 5},
            "parameter_counts": {"int8_weight_elements": 9, "total_parameters": 10, "int8_weight_share": 0.9},
            "quant_dequant_boundaries": {"total": 46}, "fp32_remaining_operations": ["LayerNorm"]})


def test_report_renders_from_the_files_the_pilot_writes(tmp_path):
    _synthetic_pilot(tmp_path)
    text = rep.render_pilot_report(tmp_path)
    assert text.startswith("# Milestone 2A pilot report") and rep.BANNER in text and rep.FRAMING in text
    assert "bpda_int8" in text and "V4 `direct_int8_grad`" in text and "status **supported**" in text
    out = tmp_path / "report.md"
    assert rep.main(["--run", str(tmp_path), "--out", str(out)]) == 0 and out.read_text(encoding="utf-8") == text
    with pytest.raises(FileExistsError):
        rep.main(["--run", str(tmp_path), "--out", str(out)])
