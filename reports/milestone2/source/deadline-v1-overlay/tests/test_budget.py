from __future__ import annotations

import math

import numpy as np
import pytest

from m2_deadline import budget
from m2_deadline.common import (
    COMPONENT_NAMES,
    ConfigError,
    ContentionError,
    GuardError,
    InfeasibleError,
    STAGES,
)


def decision_of(measurements, config, hours):
    d = budget.plan(measurements, config, hours)["decision"]
    return d["status"], d["epochs"], d["per_class"]


# ---------------------------------------------------------------- step statistics
def test_step_used_p90_term_wins_on_skewed_series_and_mean_wins_otherwise():
    skewed = [1.0] * 8 + [10.0, 10.0]
    mean, p90 = float(np.mean(skewed)), float(np.percentile(skewed, 90))
    assert 0.9 * p90 > mean  # the guard term is the larger one here
    assert budget.step_used(mean, p90) == pytest.approx(0.9 * p90)
    assert budget.step_used(1.0, 1.0) == 1.0  # mean wins: 0.9 * p90 < mean


def test_summarize_steps_drops_warmup_and_reports_cv():
    steps = [9.0, 9.0, 1.0, 1.0, 1.0, 1.0]  # the two slow warm-up steps must not count
    s = budget.summarize_steps(steps, warmup=2)
    assert s["n_timed"] == 4 and s["mean"] == 1.0 and s["step_cv"] == 0.0 and s["step_used"] == 1.0
    noisy = budget.summarize_steps([0.0, 0.0, 1.0, 5.0, 1.0, 5.0], warmup=2)
    assert noisy["step_cv"] > 0.5
    with pytest.raises(ConfigError):
        budget.summarize_steps([1.0, 1.0, 1.0], warmup=2)


def test_noisy_classification_threshold():
    ok = {"ordinary": {"step_cv": 0.5}, "adversarial": {"step_cv": 0.1}}
    bad = {"ordinary": {"step_cv": 0.1}, "adversarial": {"step_cv": 0.51}}
    assert budget.is_noisy(ok, 0.5) is False
    assert budget.is_noisy(bad, 0.5) is True


# ---------------------------------------------------------------- cap and hours
@pytest.mark.parametrize("h,expected", [(14, 9.0), (12, 9.0), (11, 8.0), (3.5, 0.5), (3, 0.0), (2, -1.0)])
def test_cap_is_min_of_hard_cap_and_h_minus_reserve(h, expected):
    assert budget.cap_hours(h) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, 0, -1, float("nan"), float("inf"), 12.5, "11", True])
def test_validate_hours_rejects_missing_or_invalid(bad):
    with pytest.raises(ConfigError):
        budget.validate_hours(bad)


def test_validate_hours_accepts_boundary():
    assert budget.validate_hours(12) == 12.0
    assert budget.validate_hours(0.5) == 0.5


# ---------------------------------------------------------------- plan
def test_picks_25_epochs_and_10_per_class_when_cheap(measurements, config):
    assert decision_of(measurements, config, 12) == ("feasible", 25, 10)
    plan = budget.plan(measurements, config, 12)
    assert plan["decision"]["robustness_n_upper_bound"] == 270
    assert plan["decision"]["cap_hours"] == 9.0
    assert plan["decision"]["planned_hours"] <= 9.0


@pytest.mark.parametrize("h", [3.0, 2.0, 1.0, 0.5])
def test_h_at_most_3_is_infeasible_even_when_cheap(measurements, config, h):
    plan = budget.plan(measurements, config, h)
    assert plan["decision"]["status"] == "infeasible"
    assert plan["decision"]["epochs"] is None and plan["decision"]["per_class"] is None
    assert not any(c["fits"] for c in plan["candidates"])
    assert plan["decision"]["cap_hours"] <= 0
    assert plan["decision"]["smallest_plan"] == {"epochs": 5, "per_class": 5}


def test_falls_back_to_10_then_5_epochs_as_training_gets_slower(measurements_factory, config):
    h = 11.0  # cap 8 h
    assert decision_of(measurements_factory(step_ord=2.0, step_adv=20.0), config, h)[1:] == (10, 10)
    assert decision_of(measurements_factory(step_ord=5.0, step_adv=50.0), config, h)[1:] == (5, 10)


def test_keeps_25_epochs_with_5_per_class_when_only_that_fits(measurements_factory, config):
    found = None
    for pgd5 in np.geomspace(0.01, 200.0, 600):
        m = measurements_factory(step_ord=1.0, step_adv=8.0, pgd5=float(pgd5))
        d = decision_of(m, config, 11.0)
        if d[1:] == (25, 5):
            found = float(pgd5)
            break
    assert found is not None, "priority rule must prefer 25 epochs with 5 per class over 10 epochs with 10 per class"
    # one step slower robustness and the k=5 variant of 25 epochs no longer wins
    slower = measurements_factory(step_ord=1.0, step_adv=8.0, pgd5=found * 50)
    assert decision_of(slower, config, 11.0)[1] != 25 or decision_of(slower, config, 11.0)[0] == "infeasible"


def test_infeasible_when_even_the_minimum_plan_is_too_slow(measurements_factory, config):
    status, epochs, per_class = decision_of(measurements_factory(step_ord=20.0, step_adv=200.0), config, 12.0)
    assert (status, epochs, per_class) == ("infeasible", None, None)


def test_candidate_order_and_count(measurements, config):
    cands = budget.plan(measurements, config, 12)["candidates"]
    assert [(c["epochs"], c["per_class"]) for c in cands] == [(25, 10), (25, 5), (10, 10), (10, 5), (5, 10), (5, 5)]
    assert budget.candidate_grid(config) == [(25, 10), (25, 5), (10, 10), (10, 5), (5, 10), (5, 5)]


def test_sequential_sum_and_safety_multiplier_applied_once(measurements, config):
    plan = budget.plan(measurements, config, 12)
    safety = config["budget"]["safety_multiplier"]
    for c in plan["candidates"]:
        assert set(c["components_s"]) == set(COMPONENT_NAMES)
        assert c["raw_s"] == pytest.approx(sum(c["components_s"].values()))  # no parallel factor
        assert c["planned_hours"] == pytest.approx(c["raw_s"] * safety / 3600.0)


def test_component_formulas(measurements, config):
    comp = budget.components_seconds(measurements, config, 10, 10)
    updates = math.ceil(22412 / 512)
    assert updates == 44
    ord_step = max(0.05, 0.9 * 0.05 * 1.1)
    adv_step = max(0.4, 0.9 * 0.4 * 1.1)
    assert comp["train_ordinary"] == pytest.approx(10 * (updates * ord_step + 0.5))
    assert comp["train_adversarial"] == pytest.approx(10 * (updates * adv_step + 0.5))
    assert comp["ptq"] == pytest.approx(10.0)
    assert comp["validation"] == pytest.approx(2 * 6.0 + 2 * 9.0)
    assert comp["test"] == pytest.approx(comp["validation"] * 3764 / 3821)
    n_ub = 27 * 10
    per_fp32 = 0.01 + 2 * 0.02 + 2 * 0.1
    assert comp["robustness"] == pytest.approx(2 * n_ub * per_fp32 + 2 * n_ub * per_fp32 * 2)
    assert comp["efficiency"] == 300.0
    assert comp["report"] == 120.0
    assert comp["overhead"] == pytest.approx(1 * (4.0 + 10.0))


def test_n_upper_bound_is_27_times_k(measurements, config):
    comp10 = budget.components_seconds(measurements, config, 10, 10)
    comp5 = budget.components_seconds(measurements, config, 10, 5)
    assert comp10["robustness"] == pytest.approx(2 * comp5["robustness"])
    assert comp10["train_adversarial"] == comp5["train_adversarial"]


def test_planned_hours_increase_with_epochs(measurements, config):
    cands = {(c["epochs"], c["per_class"]): c["planned_hours"] for c in budget.plan(measurements, config, 12)["candidates"]}
    for k in (10, 5):
        assert cands[(25, k)] > cands[(10, k)] > cands[(5, k)]
    for e in (25, 10, 5):
        assert cands[(e, 10)] > cands[(e, 5)]


def test_p90_term_raises_the_plan_only_when_it_exceeds_the_mean(measurements_factory, config):
    a = budget.plan(measurements_factory(p90_factor=1.0), config, 12)["candidates"][0]["raw_s"]
    b = budget.plan(measurements_factory(p90_factor=1.5), config, 12)["candidates"][0]["raw_s"]
    assert b > a  # p90 term wins once p90 is far above the mean


@pytest.mark.parametrize("field", ["checkpoint_write_s", "ptq_s", "efficiency_s", "load_arrays_s"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_plan_refuses_non_finite_or_negative_inputs(measurements, config, field, bad):
    measurements[field] = bad
    with pytest.raises(ConfigError):
        budget.plan(measurements, config, 12)


def test_plan_refuses_bad_nested_inputs_and_updates_mismatch(measurements, config):
    m = {**measurements, "val_eval_s": {"fp32": float("nan"), "int8": 1.0}}
    with pytest.raises(ConfigError):
        budget.plan(m, config, 12)
    m = {**measurements, "data": {**measurements["data"], "updates_per_epoch": 99}}
    with pytest.raises(ConfigError, match="updates_per_epoch"):
        budget.plan(m, config, 12)
    m = {k: v for k, v in measurements.items() if k != "ptq_s"}
    with pytest.raises(ConfigError, match="ptq_s"):
        budget.plan(m, config, 12)


def test_plan_does_not_mutate_its_inputs(measurements, config):
    import copy

    m0, c0 = copy.deepcopy(measurements), copy.deepcopy(config)
    budget.plan(measurements, config, 12)
    assert measurements == m0 and config == c0


# ---------------------------------------------------------------- lock creation
def make_estimate(measurements, config, hours=12.0, noisy=False, contended=False):
    est = budget.plan(measurements, config, hours)
    est.update({"run_id": "m2-deadline-v1", "measurements": {"noisy": noisy}, "contention": {"contended": contended}})
    return est


def test_noisy_probe_refuses_lock_without_allow_contention(measurements, config):
    est = make_estimate(measurements, config, noisy=True)
    with pytest.raises(ContentionError):
        budget.build_lock(est, "e" * 64, "c" * 64, "2026-01-01T00:00:00Z")
    lock = budget.build_lock(est, "e" * 64, "c" * 64, "2026-01-01T00:00:00Z", allow_contention=True)
    assert lock["performance_inspected"] is False and lock["selection_basis"] == "runtime only"


def test_contended_probe_refuses_lock_without_allow_contention(measurements, config):
    est = make_estimate(measurements, config, contended=True)
    with pytest.raises(ContentionError):
        budget.check_lockable(est)
    budget.check_lockable(est, allow_contention=True)


def test_infeasible_estimate_never_produces_a_lock(measurements, config):
    est = make_estimate(measurements, config, hours=3.0)
    with pytest.raises(InfeasibleError):
        budget.build_lock(est, "e" * 64, "c" * 64, "2026-01-01T00:00:00Z", allow_contention=True)


def test_lock_content_follows_the_decision(measurements, config):
    est = make_estimate(measurements, config)
    lock = budget.build_lock(est, "e" * 64, "c" * 64, "2026-01-01T00:00:00Z")
    d = est["decision"]
    assert (lock["epochs"], lock["per_class_target"]) == (d["epochs"], d["per_class"])
    assert lock["planned_hours"] == d["planned_hours"] and lock["cap_hours"] == d["cap_hours"]
    assert lock["hours_remaining_at_timing"] == 12.0


# ---------------------------------------------------------------- launch check
def lock_and_estimate(measurements, config, hours=12.0):
    est = make_estimate(measurements, config, hours)
    lock = budget.build_lock(est, "e" * 64, "c" * 64, "2026-01-01T00:00:00Z")
    return lock, est


def test_launch_check_passes_with_enough_hours_and_fails_when_hours_shrink(measurements_factory, config):
    m = measurements_factory(step_ord=1.0, step_adv=8.0)
    lock, est = lock_and_estimate(m, config, 12.0)
    planned = lock["planned_hours"]
    assert 3.0 < planned <= 9.0
    ok = budget.launch_check(lock, est, 12.0, estimate_sha256="e" * 64, config_sha256="c" * 64)
    assert ok["ok"] and ok["cap_hours"] == 9.0
    with pytest.raises(InfeasibleError, match="rerun timing"):
        budget.launch_check(lock, est, planned + 3.0 - 0.01)
    budget.launch_check(lock, est, planned + 3.0 + 0.01)  # exactly enough


@pytest.mark.parametrize("bad", [None, 0, -2, 13, float("nan")])
def test_launch_requires_valid_hours(measurements, config, bad):
    lock, est = lock_and_estimate(measurements, config)
    with pytest.raises(ConfigError):
        budget.launch_check(lock, est, bad)


def test_launch_check_guards_inconsistent_locks(measurements, config):
    lock, est = lock_and_estimate(measurements, config)
    with pytest.raises(GuardError, match="estimate_sha256"):
        budget.launch_check(lock, est, 12, estimate_sha256="0" * 64)
    with pytest.raises(GuardError, match="config_sha256"):
        budget.launch_check(lock, est, 12, config_sha256="0" * 64)
    with pytest.raises(GuardError, match="performance_inspected"):
        budget.launch_check({**lock, "performance_inspected": True}, est, 12)
    with pytest.raises(GuardError, match="planned_hours"):
        budget.launch_check({**lock, "planned_hours": lock["planned_hours"] + 1.0}, est, 12)
    with pytest.raises(GuardError, match="no candidate"):
        budget.launch_check({**lock, "epochs": 7}, est, 12)


# ---------------------------------------------------------------- resume check
def test_remaining_seconds_shrinks_as_markers_appear(measurements, config):
    lock, est = lock_and_estimate(measurements, config)
    e, k = lock["epochs"], lock["per_class_target"]
    full = sum(budget.remaining_seconds(est, e, k, []).values())
    after_train = budget.remaining_seconds(est, e, k, ["preflight", "freeze", "train_ordinary", "train_adversarial"])
    assert sum(after_train.values()) < full
    assert after_train["train_ordinary"] == 0.0 and after_train["train_adversarial"] == 0.0
    nearly_all = budget.remaining_seconds(est, e, k, list(STAGES[:-1]))
    assert nearly_all["report"] > 0 and nearly_all["robustness"] == 0.0
    assert nearly_all["overhead"] > 0  # a resume process always pays one array load


def test_mid_training_resume_counts_only_unfinished_epochs(measurements, config):
    lock, est = lock_and_estimate(measurements, config)
    e, k = lock["epochs"], lock["per_class_target"]
    none_done = budget.remaining_seconds(est, e, k, ["preflight", "freeze"])
    half = budget.remaining_seconds(est, e, k, ["preflight", "freeze"], {"train_adversarial": e // 2})
    full_adv = none_done["train_adversarial"]
    assert half["train_adversarial"] == pytest.approx(full_adv * (e - e // 2) / e)
    assert half["train_ordinary"] == none_done["train_ordinary"]
    clamped = budget.remaining_seconds(est, e, k, ["freeze"], {"train_adversarial": 10 * e})
    assert clamped["train_adversarial"] == 0.0


def test_resume_passes_where_the_full_plan_check_would_fail(measurements_factory, config):
    m = measurements_factory(step_ord=1.0, step_adv=8.0)
    lock, est = lock_and_estimate(m, config, 12.0)
    h_now = lock["planned_hours"] + 3.0 - 0.5  # not enough for the full plan
    with pytest.raises(InfeasibleError):
        budget.launch_check(lock, est, h_now)
    done = ["preflight", "freeze", "train_ordinary", "train_adversarial", "ptq"]
    result = budget.resume_check(est, lock, done, h_now)
    assert result["ok"] and result["remaining_planned_hours"] < lock["planned_hours"]
    assert result["limit_hours"] == pytest.approx(h_now - 3.0)


def test_resume_applies_no_nine_hour_ceiling(measurements_factory, config):
    m = measurements_factory(step_ord=1.0, step_adv=8.0)
    lock, est = lock_and_estimate(m, config, 12.0)
    # remaining plan equals the full plan (nothing done) and the limit H - 3 exceeds 9 h: allowed
    result = budget.resume_check(est, lock, [], 12.0)
    assert result["ok"]


def test_resume_failure_does_not_suggest_force_relock(measurements, config):
    lock, est = lock_and_estimate(measurements, config)
    with pytest.raises(InfeasibleError) as err:
        budget.resume_check(est, lock, [], 3.01)
    text = str(err.value)
    assert "force-relock" not in text and "stages left" in text


@pytest.mark.parametrize("bad", [None, 0, 99])
def test_resume_requires_valid_hours(measurements, config, bad):
    lock, est = lock_and_estimate(measurements, config)
    with pytest.raises(ConfigError):
        budget.resume_check(est, lock, [], bad)
