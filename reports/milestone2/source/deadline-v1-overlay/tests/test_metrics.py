from __future__ import annotations

import numpy as np
import pytest

from m2_deadline import metrics
from m2_deadline.common import DISPLAY_NAMES, DataIntegrityError, MODEL_IDS


def test_accuracy_basic():
    assert metrics.accuracy([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75
    with pytest.raises(ValueError):
        metrics.accuracy([0, 1], [0])


def test_macro_f1_hand_computed_with_27_labels():
    y = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    # class 0: tp=1 fp=0 fn=1 -> 2/3 ; class 1: tp=2 fp=1 fn=0 -> 4/5 ; 25 absent classes contribute 0
    expected = (2 / 3 + 4 / 5) / 27
    assert metrics.macro_f1(y, pred) == pytest.approx(expected)
    assert metrics.macro_f1(y, pred, labels=[0, 1]) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_class_absent_from_y_contributes_zero_and_predicted_but_absent_lowers_score():
    y = [0, 0, 0, 0]
    perfect = metrics.macro_f1(y, y, labels=[0, 1, 2])
    assert perfect == pytest.approx(1 / 3)  # classes 1 and 2 absent everywhere -> zero_division=0
    wrong = metrics.macro_f1(y, [0, 0, 0, 1], labels=[0, 1, 2])
    # class 0: tp=3 fn=1 -> 6/7 ; class 1 predicted once but absent from y -> 0 ; class 2 -> 0
    assert wrong == pytest.approx((6 / 7) / 3)
    assert wrong < perfect
    ignoring_class_1 = metrics.macro_f1(y, [0, 0, 0, 1], labels=[0, 2])
    assert wrong < ignoring_class_1 or wrong == pytest.approx(ignoring_class_1 * 2 / 3)


def test_zero_division_value_is_used_for_classes_absent_everywhere():
    assert metrics.macro_f1([0], [0], labels=[0, 1], zero_division=0) == pytest.approx(0.5)
    assert metrics.macro_f1([0], [0], labels=[0, 1], zero_division=1) == pytest.approx(1.0)


def test_macro_f1_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = rng.integers(0, 27, 400)
    pred = np.where(rng.random(400) < 0.6, y, rng.integers(0, 27, 400))
    expected = sk.f1_score(y, pred, labels=list(range(27)), average="macro", zero_division=0)
    assert metrics.macro_f1(y, pred) == pytest.approx(expected)
    subset = rng.choice(400, 50, replace=False)  # some classes absent from the subset
    expected = sk.f1_score(y[subset], pred[subset], labels=list(range(27)), average="macro", zero_division=0)
    assert metrics.macro_f1(y[subset], pred[subset]) == pytest.approx(expected)


def test_pp_difference_sign_rounding_and_no_negative_zero():
    assert metrics.pp_diff(0.90, 0.95) == 5.0
    assert metrics.pp_diff(0.95, 0.90) == -5.0
    assert metrics.pp_diff(0.123456, 0.123999) == 0.05
    zero = metrics.pp_diff(0.5, 0.5 - 1e-9)
    assert zero == 0.0 and str(zero) == "0.0"


def test_eps_str_single_form():
    assert [metrics.eps_str(e) for e in (0, 0.0, 0.05, 0.10, 0.1)] == ["0.0", "0.0", "0.05", "0.1", "0.1"]


def test_eps0_consistency_counts_mismatches_and_max_logit_diff():
    full_ids = np.arange(10)
    full_pred = np.arange(10) % 3
    sub_ids = np.array([2, 5, 7])
    sub_pred = full_pred[sub_ids].copy()
    sub_pred[1] = (sub_pred[1] + 1) % 3
    full_logits = np.zeros((10, 3))
    sub_logits = np.zeros((3, 3))
    sub_logits[2, 1] = 0.25
    r = metrics.eps0_consistency(sub_ids, sub_pred, full_ids, full_pred, sub_logits, full_logits)
    assert r["eps0_vs_full_test_mismatches"] == 1 and r["n_compared"] == 3
    assert r["eps0_max_abs_logit_diff"] == pytest.approx(0.25)
    assert r["behaviour"] == "record_only"


def test_eps0_consistency_never_raises_even_for_full_mismatch_or_bad_input():
    full_ids = np.arange(6)
    full_pred = np.zeros(6, dtype=int)
    r = metrics.eps0_consistency(full_ids, np.ones(6, dtype=int), full_ids, full_pred)
    assert r["eps0_vs_full_test_mismatches"] == 6 and r["eps0_max_abs_logit_diff"] is None
    missing = metrics.eps0_consistency([99], [0], full_ids, full_pred)
    assert missing["n_missing_ids"] == 1 and missing["n_compared"] == 0
    broken = metrics.eps0_consistency("not ids", None, None, None)
    assert "error" in broken and broken["eps0_vs_full_test_mismatches"] is None


def unit(n, acc, f1, rob):
    return {"n": n, "accuracy": acc, "macro_f1": f1, "robust_accuracy": rob}


def make_units(skip=None):
    units = {}
    for m, base in zip(MODEL_IDS, (0.9, 0.85, 0.8, 0.75)):
        u = {"clean_eps0.0": unit(270, base, base - 0.02, base)}
        for attack in ("fgsm", "pgd5"):
            for e, drop in (("0.05", 0.2), ("0.1", 0.4)):
                u["%s_eps%s" % (attack, e)] = unit(270, base - drop, base - drop - 0.02, base - drop)
        units[m] = u
    if skip:
        del units[skip[0]][skip[1]]
    return units


def test_build_curves_shares_eps0_between_attacks():
    rows = metrics.build_curves(make_units(), MODEL_IDS, DISPLAY_NAMES)
    assert len(rows) == 4 * 2 * 3
    eps0 = [r for r in rows if r["epsilon"] == "0.0"]
    assert len(eps0) == 8 and all(r["eps0_shared"] and r["unit_id"] == "clean_eps0.0" for r in eps0)
    for m in MODEL_IDS:
        a = [r for r in eps0 if r["model_id"] == m]
        assert len(a) == 2 and a[0]["accuracy"] == a[1]["accuracy"] and a[0]["macro_f1"] == a[1]["macro_f1"]
        assert {r["attack"] for r in a} == {"fgsm", "pgd5"}
    nonzero = [r for r in rows if r["epsilon"] != "0.0"]
    assert all(not r["eps0_shared"] for r in nonzero)
    assert {r["epsilon"] for r in rows} == {"0.0", "0.05", "0.1"}  # never "0.10"
    assert {r["unit_id"] for r in nonzero} == {"fgsm_eps0.05", "fgsm_eps0.1", "pgd5_eps0.05", "pgd5_eps0.1"}


def test_build_curves_rejects_missing_unit_or_keys():
    with pytest.raises(DataIntegrityError, match="clean_eps0.0"):
        metrics.build_curves(make_units(skip=("V2", "clean_eps0.0")), MODEL_IDS, DISPLAY_NAMES)
    with pytest.raises(DataIntegrityError, match="pgd5_eps0.1"):
        metrics.build_curves(make_units(skip=("V3", "pgd5_eps0.1")), MODEL_IDS, DISPLAY_NAMES)
    units = make_units()
    del units["V1"]["fgsm_eps0.05"]["macro_f1"]
    with pytest.raises(DataIntegrityError, match="lacks keys"):
        metrics.build_curves(units, MODEL_IDS, DISPLAY_NAMES)
    with pytest.raises(DataIntegrityError, match="no robustness units"):
        metrics.build_curves({}, MODEL_IDS, DISPLAY_NAMES)


def test_full_test_and_subset_clean_stay_in_separate_fields():
    cols = metrics.clean_columns(
        {"accuracy": 0.91, "macro_f1": 0.88, "n": 3764},
        {"accuracy": 0.93, "macro_f1": 0.90, "n": 270},
    )
    assert cols["clean_accuracy_full_test"] == 0.91 and cols["clean_accuracy_robustness_subset"] == 0.93
    assert cols["n_full_test"] == 3764 and cols["n_robustness_subset"] == 270
    assert cols["macro_f1_full_test"] != cols["macro_f1_robustness_subset"]
