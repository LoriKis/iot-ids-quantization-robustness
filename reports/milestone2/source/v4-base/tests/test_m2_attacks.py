"""Milestone 2A attack plumbing tests (part 1): selector, feasible set, checked_attack, row-tied model, CheckedPair."""
import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.ao import quantization as aq

from m2_helpers import TINY_ISLANDS, split_arrays, tiny_model
from research import attacks
from research import m2_attacks as ma
from research import m2_quant as mq
from research.common import ROOT, make_model
from research.m2_attacks import (AttackInvariantError, CheckedPair, RowTiedModel, checked_attack, feasible_attack,
                                 feasible_bounds, feasible_interval)
from research.m2_split import ConfigError, PartitionGuardError
from research.paper_described_adam import seed_everything
from research.provenance import state_hash
from research.quantization import bpda_model, ptq

GRID = [0.0, 0.01, 0.025, 0.05, 0.10]
TINY = {"input_dim": 39, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "n_classes": 27,
        "max_seq_len": 50, "dropout": 0.1}


@pytest.fixture
def tiny():
    seed_everything(5)
    return make_model(TINY).eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(3)
    return torch.rand(6, 39, generator=g), torch.arange(6) % 27


class Recorder:
    """Selector that records each input and delegates to a function."""

    def __init__(self, fn):
        self.fn, self.inputs = fn, []

    def __call__(self, x):
        self.inputs.append(x.detach().clone())
        return self.fn(x)


def test_selector_none_is_unchanged_and_call_structure(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    lower, upper = feasible_bounds(x)
    base = attacks.attack(model, x, y, kind="pgd", epsilon=0.05, steps=4, alpha=0.0125, restarts=2, lower=lower, upper=upper, seed=9)
    rec = Recorder(lambda z: model(z)["logits"])
    same = attacks.attack(model, x, y, kind="pgd", epsilon=0.05, steps=4, alpha=0.0125, restarts=2, lower=lower, upper=upper, seed=9, selector=rec)
    assert torch.equal(base, same)
    assert len(rec.inputs) == 1 + 1 + 2 * (1 + 4)
    fg = Recorder(lambda z: model(z)["logits"])
    attacks.attack(model, x, y, kind="fgsm", epsilon=0.05, lower=lower, upper=upper, selector=fg)
    assert len(fg.inputs) == 4


def test_pgd20_three_restarts_selector_calls(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    rec = Recorder(lambda z: model(z)["logits"])
    feasible_attack(model, x, y, kind="pgd", epsilon=0.05, steps=20, alpha=0.0125, restarts=3, seed=1, selector=rec)
    assert len(rec.inputs) == 65


def test_epsilon_zero_makes_no_selector_calls(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    rec = Recorder(lambda z: model(z)["logits"])
    out = checked_attack(model, x, y, kind="pgd", epsilon=0.0, steps=5, selector=rec)
    assert torch.equal(out, x) and rec.inputs == []


def test_selector_stub_decides_baseline_and_retention(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    clean_sum = x.sum(1)

    def stub(z):  # correct on the clean point, wrong whenever the perturbation raised the row sum
        out = torch.zeros(len(z), 27)
        wrong = z.sum(1) > clean_sum + 1e-4
        out[torch.arange(len(z)), torch.where(wrong, (y + 1) % 27, y)] = 5.0
        return out

    rec = Recorder(stub)
    adv = feasible_attack(model, x, y, kind="pgd", epsilon=0.05, steps=3, alpha=0.025, restarts=1, seed=4, selector=rec)
    assert torch.equal(rec.inputs[0], x)  # baseline scored by the stub, not the source
    any_wrong = torch.zeros(len(x), dtype=torch.bool)
    for cand in rec.inputs[1:]:
        any_wrong |= stub(cand).argmax(1).ne(y)
    assert any_wrong.any()
    assert torch.equal(stub(adv).argmax(1).ne(y), any_wrong)  # success under the stub is retained


def test_feasible_bounds_must_be_tensors(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    with pytest.raises((TypeError, RuntimeError)):
        attacks.attack(model, x, y, kind="fgsm", epsilon=0.05, lower=0.0, upper=1.0)
    lower, upper = feasible_bounds(x)
    assert lower.shape == upper.shape == x.shape
    attacks.attack(model, x, y, kind="fgsm", epsilon=0.05, lower=lower, upper=upper)


@pytest.mark.parametrize("value", [2.0, -0.3, 0.5])
@pytest.mark.parametrize("kind", ["fgsm", "pgd"])
def test_feasible_policy_over_grid(tiny, value, kind):
    x = torch.full((4, 39), value)
    y = torch.arange(4)
    model = RowTiedModel(tiny)
    for eps in GRID:
        lo, hi = feasible_interval(x, eps)
        assert bool((lo <= x).all() and (x <= hi).all() and (lo <= hi).all())  # never empty, contains x
        out = checked_attack(model, x, y, kind=kind, epsilon=eps, steps=3, alpha=eps / 2 or None, seed=2)
        assert torch.isfinite(out).all()
        if eps == 0.0:
            assert torch.equal(out, x)
        if value == 2.0:  # out-of-range coordinate may only move toward the interior within [x-eps, x]
            assert bool((out <= x).all())
        if value == -0.3:
            assert bool((out >= x).all())
    assert torch.equal(x, torch.full((4, 39), value))  # clean input never clipped


def test_float32_tolerance_no_false_failure(tiny):
    x = torch.full((4, 39), 2.0)
    y = torch.arange(4)
    out = checked_attack(RowTiedModel(tiny), x, y, kind="pgd", epsilon=0.01, steps=5, alpha=0.005, seed=1)
    assert float((out - x).abs().max()) <= 0.01 + 2e-6


class _NanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * float("nan")


class _NanGradModel(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)["logits"]
        return {"logits": out + 0.0 * _NanGrad.apply(x).sum(dim=(1, 2)).unsqueeze(1)}


class _NanLogitModel(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return {"logits": self.inner(x)["logits"] * float("nan")}


def test_injected_nan_gradient_is_attack_invariant_error(tiny, batch):
    x, y = batch
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(RowTiedModel(_NanGradModel(tiny)), x, y, kind="pgd", epsilon=0.05, steps=2, seed=1)
    assert err.value.check_name == "finite_gradients"
    assert isinstance(err.value.__cause__, FloatingPointError)


def test_nan_logits_and_nan_inputs_are_named(tiny, batch):
    x, y = batch
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(RowTiedModel(_NanLogitModel(tiny)), x, y, kind="fgsm", epsilon=0.05, seed=1)
    assert err.value.check_name == "finite_logits"
    bad = x.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(RowTiedModel(tiny), bad, y, kind="fgsm", epsilon=0.05, seed=1)
    assert err.value.check_name == "finite_inputs"


def test_engine_assertion_error_maps_to_linf_budget(tiny, batch, monkeypatch):
    x, y = batch

    def boom(*args, **kwargs):
        raise AssertionError("Attack budget or finiteness violation")

    monkeypatch.setattr(attacks, "attack", boom)
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(RowTiedModel(tiny), x, y, kind="fgsm", epsilon=0.05)
    assert err.value.check_name == "linf_budget"
    assert isinstance(err.value.__cause__, AssertionError)


def test_budget_violation_detected(tiny, batch, monkeypatch):
    x, y = batch
    monkeypatch.setattr(attacks, "attack", lambda *a, **k: x + 0.5)
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(RowTiedModel(tiny), x, y, kind="fgsm", epsilon=0.05)
    assert err.value.check_name == "linf_budget"


def test_weights_and_grads_unchanged_and_modes_restored(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    model.train()
    next(m for m in model.modules() if isinstance(m, torch.nn.Dropout)).eval()  # mixed module modes
    modes = [m.training for m in model.modules()]
    before = state_hash(model)
    report = {}
    checked_attack(model, x, y, kind="pgd", epsilon=0.05, steps=3, alpha=0.025, seed=3, report=report)
    assert [m.training for m in model.modules()] == modes
    assert state_hash(model) == before
    assert all(p.grad is None for p in model.parameters())
    assert report["weights_unchanged"] and report["no_parameter_grad"] and report["mode_during_attack"]


def test_selector_that_changes_weights_or_grads_is_caught(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)

    def corrupt(z):
        with torch.no_grad():
            next(model.parameters()).add_(1e-3)
        return model(z)["logits"]

    with pytest.raises(AttackInvariantError) as err:
        checked_attack(model, x, y, kind="fgsm", epsilon=0.05, selector=corrupt)
    assert err.value.check_name == "weights_unchanged"

    fresh = RowTiedModel(make_model(TINY).eval())
    with pytest.raises(AttackInvariantError) as err:
        checked_attack(fresh, x, y, kind="fgsm", epsilon=0.05,
                       selector=lambda z: (setattr(next(fresh.parameters()), "grad", torch.ones_like(next(fresh.parameters()))), fresh(z)["logits"])[1])
    assert err.value.check_name == "no_parameter_grad"


def test_row_tied_perturbation_is_equal_across_positions(tiny, batch):
    x, y = batch
    seen = []
    tiny.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    x_adv = checked_attack(RowTiedModel(tiny), x, y, kind="pgd", epsilon=0.05, steps=3, alpha=0.025, seed=2)
    assert x_adv.shape == x.shape and not torch.equal(x_adv, x)
    seq = seen[-1]
    assert seq.shape == (6, 50, 39)
    for pos in range(50):
        assert torch.equal(seq[:, pos], seq[:, 0])
    # per-position mode (inherited) perturbs positions independently
    seq_x = x[:, None, :].repeat(1, 50, 1)
    lower, upper = torch.zeros_like(seq_x), torch.ones_like(seq_x)
    per_position = attacks.attack(tiny, seq_x, y, kind="fgsm", epsilon=0.05, lower=lower, upper=upper)
    assert not torch.equal(per_position[:, 0], per_position[:, 7])


def test_row_tied_input_validation(tiny):
    with pytest.raises(ValueError):
        RowTiedModel(tiny)(torch.zeros(2, 50, 39))


def test_fgsm_candidates_duplicate(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    rec = Recorder(lambda z: model(z)["logits"])
    feasible_attack(model, x, y, kind="fgsm", epsilon=0.05, selector=rec)
    assert len(rec.inputs) == 4
    assert torch.equal(rec.inputs[0], x) and torch.equal(rec.inputs[2], x)
    assert torch.equal(rec.inputs[1], rec.inputs[3])  # candidate 1 and candidate 3 are bit-identical
    assert not torch.equal(rec.inputs[1], x)


def test_checked_pair_counts_and_exactness(tiny, batch):
    x, y = batch
    int8 = RowTiedModel(tiny)
    pair = CheckedPair(RowTiedModel(tiny), int8)
    out = checked_attack(int8, x, y, kind="fgsm", epsilon=0.05, selector=pair, seed=1)
    assert out.shape == x.shape
    assert pair.n_selector_calls == 4 and pair.max_abs_err == 0.0 and pair.n_clean_calls == 0
    clean = pair.clean(x)
    assert pair.n_clean_calls == 1 and pair.n_selector_calls == 4
    assert torch.equal(clean, int8(x)["logits"])

    class Offset(torch.nn.Module):
        def forward(self, z):
            return {"logits": int8(z)["logits"] + 1e-3}

    bad = CheckedPair(Offset(), int8)
    bad(x)
    assert bad.max_abs_err > 0.0 and math.isfinite(bad.max_abs_err)


# ============================================================ part 2: evaluation sources
TRAIN_CFG = json.loads((ROOT / "configs/m2/training.json").read_text(encoding="utf-8"))
NB = 2  # 70 rows at evaluation batch size 64


@pytest.fixture(scope="module")
def acfg():
    return ma.load_config()


@pytest.fixture(scope="module")
def stack():
    parent = tiny_model(5)
    arrays = split_arrays()
    cal = mq.calibration_ids(arrays, 20, 20260)
    q, inv = ptq(parent, "all_linear", cal["x"])
    bpda = bpda_model(q, inv["requested"], cal["x"][:2])
    surrogate, names = ma.build_surrogate(parent, cal["x"])
    return {"parent": parent, "arrays": arrays, "cal": cal, "int8": q, "inv": inv, "bpda": bpda,
            "surrogate": surrogate, "names": names}


@pytest.fixture(scope="module")
def models(stack):
    return {k: stack[k] for k in ("int8", "bpda", "surrogate")} | {"fp32": stack["parent"]}


@pytest.fixture(scope="module")
def val(stack):
    v = stack["arrays"]["validation"]
    return torch.from_numpy(v["x"][:70]), torch.from_numpy(v["y"][:70]), v["ids"][:70]


def unit(models, val, acfg, source, attack, eps, **kw):
    x, y, ids = val
    return ma.evaluate_unit(source=source, attack_name=attack, epsilon=eps, x=x, y=y, ids=ids, models=models, cfg=acfg,
                            train_seed=1, model_id="V2", model_ids={"fp32": "V1", "int8": "V2"}, **kw)


def test_attack_config_is_consistent_with_the_engine(acfg):
    assert acfg["epsilon_grid"] == [0, 0.01, 0.025, 0.05, 0.10]
    table = acfg["expected_selector_calls_per_batch"]
    assert table["pgd20"] == {"selector_calls": 65, "candidates": 64}
    assert table["fgsm"] == {"selector_calls": 4, "candidates": 3}
    assert table["epsilon_0"] == {"selector_calls": 0, "candidates": 0}
    assert acfg["attacks"]["pgd20"]["status"] == "provisional" and acfg["attacks"]["pgd20"]["restarts"] == 3
    assert acfg["e4"]["per_class_cap"] == 70 and acfg["e4"]["min_subset"] == 1000 and acfg["e4"]["all_rows_threshold"] == 2500
    assert acfg["evaluation"]["batch_size"] == 64 and acfg["row_tied"] is True
    assert acfg["seed_rule"] == "derived_seed('atk-eval', train_seed, model_id, source, epsilon, batch_index)"
    assert ma.expected_selector_calls("pgd", 20, 3) == {"selector_calls": 65, "candidates": 64}
    assert ma.expected_selector_calls("pgd", 50, 5) == {"selector_calls": 257, "candidates": 256}
    assert ma.expected_selector_calls("fgsm", 1, 1) == {"selector_calls": 4, "candidates": 3}


def test_attack_config_validation(acfg):
    def broken(mutate):
        cfg = copy.deepcopy(acfg)
        mutate(cfg)
        with pytest.raises(ConfigError):
            ma.validate_config(cfg)

    broken(lambda c: c["expected_selector_calls_per_batch"]["pgd20"].update(selector_calls=66))
    broken(lambda c: c.update(epsilon_grid=[0.01, 0.05]))
    broken(lambda c: c.update(epsilon_grid=[0, 0.05, 0.05]))
    broken(lambda c: c.update(epsilon_grid=[0, 0.05, 1.5]))
    broken(lambda c: c.update(row_tied=False))
    broken(lambda c: c.update(unexpected=1))
    broken(lambda c: c["evaluation"].update(evaluation_partition="test"))
    broken(lambda c: c["expected_selector_calls_per_batch"].update(epsilon_0={"selector_calls": 1, "candidates": 0}))
    with pytest.raises(ConfigError):
        ma.load_config(ROOT / "missing.json")


def test_engine_call_count_matches_formula(tiny, batch):
    x, y = batch
    model = RowTiedModel(tiny)
    rec = Recorder(lambda z: model(z)["logits"])
    feasible_attack(model, x, y, kind="pgd", epsilon=0.05, steps=3, alpha=0.0125, restarts=2, seed=1, selector=rec)
    assert len(rec.inputs) == ma.expected_selector_calls("pgd", 3, 2)["selector_calls"]


class CountingTarget(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.calls = inner, 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


def test_pure_transfer_makes_no_target_queries_but_targetsel_does(stack, val, acfg):
    for source, construction in (("transfer_fp32_to_int8", 0), ("transfer_fp32_to_int8_targetsel", 4 * NB)):
        target = CountingTarget(stack["int8"])
        res = unit({"fp32": stack["parent"], "int8": target}, val, acfg, source, "fgsm", 0.05)
        c = res["checks"]
        assert c["target_queries_during_construction"] == construction
        # clean call and the single final evaluation of the retained x_adv are the only other target queries
        assert target.calls == construction + 2 * NB
        assert c["valid"] and c["scoring_target_id"] == "V2" and c["gradient_model_id"] == "V1"
        if construction == 0:
            assert c["n_selector_calls"] is None and c["selector_target_id"] is None
        else:
            assert c["n_selector_calls"] == 4 * NB and c["candidates_scored_on_target"] == 3 * NB
            assert c["selector_target_id"] == "V2" and c["n_clean_calls"] == NB


def test_pure_surrogate_makes_no_target_queries_but_targetsel_does(stack, val, acfg):
    for source, construction in (("surrogate_fakequant", 0), ("surrogate_fakequant_targetsel", 4 * NB)):
        target = CountingTarget(stack["int8"])
        res = unit({"surrogate": stack["surrogate"], "int8": target}, val, acfg, source, "fgsm", 0.05)
        assert res["checks"]["target_queries_during_construction"] == construction
        assert target.calls == construction + 2 * NB


@pytest.mark.parametrize("attack,eps,calls,cands", [("pgd20", 0.05, 65, 64), ("fgsm", 0.05, 4, 3),
                                                     ("pgd20", 0.0, 0, 0), ("fgsm", 0.0, 0, 0)])
def test_bpda_selector_counts_are_read_from_config(models, val, acfg, attack, eps, calls, cands):
    c = unit(models, val, acfg, "bpda_int8", attack, eps)["checks"]
    assert c["n_selector_calls"] == calls * NB and c["candidates_scored_on_target"] == cands * NB
    assert c["expected_n_selector_calls"] == calls * NB and c["counts_ok"] is True
    assert c["n_clean_calls"] == NB  # one explicit clean call per batch, outside attack()
    assert c["valid"] and c["selector_target_id"] == "V2" and len(c["selector_target_state_hash"]) == 64
    if eps == 0:
        assert c["bpda_forward_max_abs_err"] is None and c["bpda_forward_max_abs_err_all_calls"] == 0.0
    else:
        assert c["bpda_forward_max_abs_err"] == 0.0


def test_count_mismatch_with_config_fails_the_unit(models, val, acfg):
    wrong = copy.deepcopy(acfg)
    wrong["expected_selector_calls_per_batch"]["fgsm"]["selector_calls"] = 5
    with pytest.raises(ma.AttackUnitError) as err:
        unit(models, val, wrong, "bpda_int8", "fgsm", 0.05)
    assert "selector_call_counts" in err.value.checks["failures"] and err.value.checks["valid"] is False


def test_bpda_wrapper_equals_real_int8_exactly(stack):
    x = torch.from_numpy(stack["arrays"]["validation"]["x"][:30])
    with torch.no_grad():
        a = RowTiedModel(stack["bpda"])(x)["logits"]
        b = RowTiedModel(stack["int8"])(x)["logits"]
    assert torch.equal(a, b)
    xr = x.clone().requires_grad_()
    g = torch.autograd.grad(RowTiedModel(stack["bpda"])(xr)["logits"].square().sum(), xr)[0]
    assert torch.isfinite(g).all() and g.abs().max() > 0


def test_perturbed_bpda_stub_fails_the_unit(models, val, acfg):
    class Offset(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return {"logits": self.inner(x)["logits"] + 1e-3}

    bad = dict(models, bpda=Offset(models["bpda"]))
    with pytest.raises(ma.AttackUnitError) as err:
        unit(bad, val, acfg, "bpda_int8", "fgsm", 0.05)
    assert "bpda_forward_exact" in err.value.checks["failures"]
    soft = unit(bad, val, acfg, "bpda_int8", "fgsm", 0.05, raise_on_invalid=False)
    assert soft["checks"]["valid"] is False and soft["checks"]["bpda_forward_max_abs_err"] > 0
    # a non-zero error on the clean call alone also fails an epsilon-0 unit while the field stays null
    zero = unit(bad, val, acfg, "bpda_int8", "fgsm", 0.0, raise_on_invalid=False)
    assert zero["checks"]["bpda_forward_max_abs_err"] is None and zero["checks"]["valid"] is False


@pytest.mark.parametrize("source", ma.SOURCES)
def test_epsilon_zero_reproduces_clean_predictions_for_every_source(models, val, acfg, source):
    for attack in ("fgsm", "pgd20"):
        res = unit(models, val, acfg, source, attack, 0.0)
        c, arrays = res["checks"], res["arrays"]
        assert c["valid"] and c["eps0_bitwise_identity"] is True and c["eps0_prediction_identity"] is True
        assert np.array_equal(arrays["pred_clean"], arrays["pred_adv"]) and res["row"]["max_linf"] == 0.0
        assert c["n_selector_calls"] in (0, None) and c["candidates_scored_on_target"] in (0, None)


@pytest.mark.parametrize("source", ma.SOURCES)
def test_every_source_passes_checks_and_keeps_examples_and_labels(models, val, acfg, source):
    x, y, ids = val
    before = x.clone()
    res = unit(models, val, acfg, source, "fgsm", 0.05)
    assert torch.equal(x, before)  # the clean input is not modified
    c, row = res["checks"], res["row"]
    assert c["valid"] and all(c["invariants"].values()) and c["failures"] == []
    assert set(c["invariants"]) >= {"finite_inputs", "finite_logits", "finite_gradients", "linf_budget", "feasible_set",
                                    "weights_unchanged", "no_parameter_grad", "labels_unchanged", "mode_restored"}
    assert c["example_id_sha256"] == ma.id_hash(ids) and c["label_sha256"] == ma._label_hash(y.numpy())
    assert row["n"] == 70 and np.array_equal(res["arrays"]["y"], y.numpy()) and np.array_equal(res["arrays"]["ids"], ids)
    assert row["max_linf"] <= 0.05 + 2e-6 and "Feature-space attacks" in row["framing"]
    assert 0.0 <= row["robust_accuracy"] <= row["clean_accuracy"] <= 1.0


def test_out_of_range_inputs_are_reported_never_clipped_or_dropped(models, acfg):
    g = torch.Generator().manual_seed(1)
    x = torch.rand(70, 39, generator=g)
    x[0, :3] = 2.0
    x[1, 5] = -0.3
    y, ids = torch.arange(70) % 27, np.arange(70)
    before = x.clone()
    res = ma.evaluate_unit(source="fp32_direct", attack_name="pgd20", epsilon=0.05, x=x, y=y, ids=ids, models=models,
                           cfg=acfg, train_seed=1, model_id="V1")
    c = res["checks"]
    assert torch.equal(x, before) and res["row"]["n"] == 70 and len(res["arrays"]["ids"]) == 70
    assert c["n_rows_out_of_range"] == 2 and c["n_out_of_range_coordinates"] == 4 and c["n_one_sided_coordinates"] == 4
    assert c["valid"] and res["row"]["max_linf"] <= 0.05 + 2e-6
    eps0 = ma.evaluate_unit(source="fp32_direct", attack_name="fgsm", epsilon=0.0, x=x, y=y, ids=ids, models=models,
                            cfg=acfg, train_seed=1, model_id="V1")
    assert eps0["checks"]["n_one_sided_coordinates"] == 0 and eps0["checks"]["eps0_bitwise_identity"] is True


def test_evaluation_entry_refuses_test_and_other_partitions(models, val, acfg):
    for partition in ("test", "train", "anything"):
        with pytest.raises(PartitionGuardError):
            unit(models, val, acfg, "fp32_direct", "fgsm", 0.05, evaluation_partition=partition)


def test_label_and_id_hash_checks(models, val, acfg):
    x, y, ids = val
    ok = unit(models, val, acfg, "fp32_direct", "fgsm", 0.05, expected_ids_sha256=ma.id_hash(ids),
              expected_label_sha256=ma._label_hash(y.numpy()))
    assert ok["checks"]["valid"]
    with pytest.raises(ma.AttackUnitError) as err:
        unit(models, val, acfg, "fp32_direct", "fgsm", 0.05, expected_label_sha256="0" * 64)
    assert "labels_match" in err.value.checks["failures"]
    with pytest.raises(ma.AttackUnitError) as err:
        unit(models, val, acfg, "fp32_direct", "fgsm", 0.05, expected_ids_sha256="0" * 64)
    assert "example_ids_match" in err.value.checks["failures"]


def test_attack_seed_rule(models, val, acfg, monkeypatch):
    from research.paper_described_adam import derived_seed
    seen = []
    real = ma.checked_attack
    monkeypatch.setattr(ma, "checked_attack", lambda *a, **k: (seen.append(k["seed"]), real(*a, **k))[1])
    unit(models, val, acfg, "bpda_int8", "pgd20", 0.025)
    assert seen == [derived_seed("atk-eval", 1, "V2", "bpda_int8", 0.025, b) for b in range(NB)]


def test_invariant_failure_invalidates_unit_and_is_recorded(stack, val, acfg, tmp_path):
    class NanLogits(nn.Module):
        def forward(self, x):
            return {"logits": stack["parent"](x)["logits"] * float("nan")}

    with pytest.raises(ma.AttackUnitError) as err:
        unit({"fp32": NanLogits()}, val, acfg, "fp32_direct", "fgsm", 0.05, out_dir=tmp_path)
    assert err.value.checks["failed_check"] == "finite_logits" and err.value.checks["valid"] is False
    saved = json.loads((tmp_path / "attack_checks.json").read_text())
    assert saved["failed_check"] == "finite_logits" and saved["valid"] is False


def test_attack_checks_file_contents_and_no_overwrite(models, val, acfg, tmp_path):
    res = unit(models, val, acfg, "bpda_int8", "fgsm", 0.05, out_dir=tmp_path)
    saved = json.loads((tmp_path / "attack_checks.json").read_text())
    assert saved["selector_target_id"] == "V2" and saved["selector_target_state_hash"] == ma.model_state_hash(models["int8"])
    assert saved["example_id_sha256"] == res["checks"]["example_id_sha256"] and saved["label_sha256"]
    assert saved["n_selector_calls"] == 4 * NB and saved["candidates_scored_on_target"] == 3 * NB
    assert saved["bpda_forward_max_abs_err"] == 0.0 and saved["n_batches"] == NB
    assert json.loads((tmp_path / "attack_result.json").read_text())["attack_source"] == "bpda_int8"
    assert "approximate backward" in res["row"]["note"]
    with pytest.raises(FileExistsError):
        unit(models, val, acfg, "bpda_int8", "fgsm", 0.05, out_dir=tmp_path)


def test_union_row_is_a_separate_labelled_per_example_and(models, val, acfg):
    a = unit(models, val, acfg, "bpda_int8", "fgsm", 0.05)
    b = unit(models, val, acfg, "transfer_fp32_to_int8", "fgsm", 0.05)
    u = ma.union_row([a, b])
    assert u["attack_source"] == "union_any_source" and u["sources"] == ["bpda_int8", "transfer_fp32_to_int8"]
    assert u["robust_accuracy"] == float((a["arrays"]["robust_correct"] & b["arrays"]["robust_correct"]).mean())
    assert u["robust_accuracy"] <= min(a["row"]["robust_accuracy"], b["row"]["robust_accuracy"])
    other = unit(models, val, acfg, "bpda_int8", "fgsm", 0.025)
    with pytest.raises(ValueError):
        ma.union_row([a, other])


# ------------------------------------------------------------------ surrogate
def _scales(model, names):
    return {n: ma._island_scales(model, n) for n in names}


def test_surrogate_build_is_deterministic_dropout_free_and_rng_neutral(stack, monkeypatch):
    seen = []
    real = torch.nn.functional.dropout
    monkeypatch.setattr(torch.nn.functional, "dropout",
                        lambda input, p=0.5, training=True, inplace=False: (seen.append(training), real(input, p, training, inplace))[1])
    before = torch.get_rng_state().clone()
    s1, names = ma.build_surrogate(stack["parent"], stack["cal"]["x"])
    s2, _ = ma.build_surrogate(stack["parent"], stack["cal"]["x"])
    assert torch.equal(before, torch.get_rng_state())
    assert seen and not any(seen)  # dropout was called (eval mode) and never in training mode
    a, b = _scales(s1, names), _scales(s2, names)
    for n in names:
        for key in a[n]:
            assert torch.equal(a[n][key], b[n][key])
    # weights are exactly the parent's and the model is frozen the way the protocol requires
    assert torch.equal(s1.get_submodule("embedding").linear.weight, stack["parent"].embedding.weight)
    assert not any(m.training for m in s1.modules())
    fq = [m for m in s1.modules() if isinstance(m, aq.FakeQuantizeBase)]
    assert fq and all(int(m.observer_enabled) == 0 and int(m.fake_quant_enabled) == 1 for m in fq)


def test_surrogate_uses_the_supplied_calibration_rows(stack):
    other = mq.calibration_ids(split_arrays(seed=7), 20, 20260)
    s_other, names = ma.build_surrogate(stack["parent"], other["x"])
    assert not torch.equal(ma._island_scales(s_other, names[0])["input_scale"],
                           ma._island_scales(stack["surrogate"], names[0])["input_scale"])


def test_surrogate_build_fails_on_training_mode_dropout_or_rng_change(stack, monkeypatch):
    real_prepare = ma.prepare_static

    def keep_training(*args, **kwargs):
        model, names = real_prepare(*args, **kwargs)
        model.eval = lambda: model  # instance attribute shadows Module.eval, so dropout stays in training mode
        return model, names

    monkeypatch.setattr(ma, "prepare_static", keep_training)
    with pytest.raises(RuntimeError, match="Dropout"):
        ma.build_surrogate(stack["parent"], stack["cal"]["x"])
    monkeypatch.setattr(ma, "prepare_static", real_prepare)

    real_disable = aq.disable_observer

    def noisy(module):
        torch.rand(1)
        return real_disable(module)

    monkeypatch.setattr(aq, "disable_observer", noisy)
    with pytest.raises(RuntimeError, match="RNG"):
        ma.build_surrogate(stack["parent"], stack["cal"]["x"])


def test_surrogate_fidelity_reports_scale_differences(stack, val, acfg):
    x = val[0]
    f = ma.surrogate_fidelity(stack["surrogate"], stack["int8"], stack["names"], x)
    assert set(f["islands"]) == set(stack["names"]) and f["n_rows"] == 70
    assert math.isfinite(f["max_logit_diff_to_int8"]) and 0.0 <= f["argmax_agreement_with_int8"] <= 1.0
    assert f["note"].startswith("surrogate scales differ") == (f["max_scale_diff"] > 0)
    res = unit({"surrogate": stack["surrogate"], "int8": stack["int8"]}, val, acfg, "surrogate_fakequant", "fgsm", 0.05,
               notes=[f["note"]])
    assert res["row"]["extra_notes"] == [f["note"]]


# ------------------------------------------------------------------ direct INT8 gradients
class _ZeroGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(39, 27)

    def forward(self, x):
        return {"logits": self.w(x.mean(1)) * 0.0}


def test_direct_int8_grad_is_flagged_never_counted_as_robust(stack, val):
    x, y, _ = val
    d = ma.direct_int8_grad_diagnostic(stack["int8"], x, y)
    assert d["n"] == 70 and d["undefined_gradient_examples"] == 70 and d["undefined_or_zero_fraction"] == 1.0
    assert d["counted_as_robust"] is False and "RuntimeError" in d["first_error"]
    assert "raises RuntimeError" in d["behaviour"]
    z = ma.direct_int8_grad_diagnostic(_ZeroGrad(), x, y)
    assert z["zero_gradient_examples"] == 70 and z["undefined_gradient_examples"] == 0 and z["undefined_or_zero_fraction"] == 1.0
    assert z["behaviour"] == "zero or non-finite gradients"
    from test_m2_attacks import _NanGradModel
    n = ma.direct_int8_grad_diagnostic(_NanGradModel(stack["parent"]), x, y)
    assert n["nonfinite_gradient_examples"] == 70
    ok = ma.direct_int8_grad_diagnostic(stack["parent"], x, y)
    assert ok["undefined_or_zero_fraction"] == 0.0 and ok["behaviour"] == "usable gradients"


# ------------------------------------------------------------------ E4 budget check
def test_e4_subset_rules(acfg):
    ids = np.arange(3000) + 10
    y = np.arange(3000) % 27
    sub = ma.e4_subset(ids, y, acfg, 20260)
    assert sub["n_subset"] == 27 * 70 and set(sub["ids"]) <= set(ids.tolist()) and sub["plumbing_only"] is False
    assert sub["per_class_counts"] == [70] * 27 and sub["id_sha256"] == ma.id_hash(sub["ids"])
    assert ma.e4_subset(ids, y, acfg, 20260)["ids"] == sub["ids"]
    everything = ma.e4_subset(ids[:1200], y[:1200], acfg, 20260)
    assert everything["n_subset"] == 1200 and everything["ids"] == ids[:1200].tolist()
    with pytest.raises(ma.SubsetTooSmallError, match="ask the user"):
        ma.e4_subset(ids[:900], y[:900], acfg, 20260)
    skewed = np.where(np.arange(3000) < 2600, 0, 1 + np.arange(3000) % 26)
    with pytest.raises(ma.SubsetTooSmallError):
        ma.e4_subset(ids, skewed, acfg, 20260)  # capped subset below the minimum: stop, never substitute rows
    plumbing = ma.e4_subset(ids[:270], y[:270], acfg, 20260, plumbing_only=True)
    assert plumbing["n_subset"] == 270 and plumbing["plumbing_only"] is True


def test_paired_extra_count_and_budget_decision(acfg):
    assert ma.paired_n_extra([False, True, False, True], [True, True, False, False]) == 1
    with pytest.raises(ValueError):
        ma.paired_n_extra([True], [True, False])
    cells = {(m, e): 0 for m in acfg["e4"]["models"] for e in acfg["e4"]["epsilons"]}
    assert len(cells) == 12
    ok = ma.budget_check(cells, 1890, acfg)
    assert ok["threshold"] == math.ceil(0.01 * 1890) == 19 and ok["adequate"] is True and ok["decision"] == "pgd20"
    edge = {**cells, ("V3", 0.05): 19}
    assert ma.budget_check(edge, 1890, acfg)["adequate"] is True
    fail = {**cells, ("V3", 0.05): 20}
    result = ma.budget_check(fail, 1890, acfg)
    assert result["adequate"] is False and result["decision"] == "pgd50x5"
    assert [r for r in result["cells"] if not r["within_threshold"]] == [
        {"model": "V3", "epsilon": 0.05, "n_extra": 20, "within_threshold": False}]
    with pytest.raises(ValueError):
        ma.budget_check({k: v for k, v in cells.items() if k != ("V1", 0.1)}, 1890, acfg)
    with pytest.raises(ma.SubsetTooSmallError):
        ma.budget_check(cells, 500, acfg)
    plumbing = ma.budget_check(cells, 270, acfg, plumbing_only=True)
    assert plumbing["adequate"] is None and plumbing["decision"] is None and plumbing["threshold"] == 3


# ------------------------------------------------------------------ real model (slow)
@pytest.mark.slow
def test_real_model_bpda_surrogate_and_direct_int8_gradients(acfg):
    arch = TRAIN_CFG["architecture"]
    model = tiny_model(3, arch)
    assert sum(p.numel() for p in model.parameters()) == 351_612
    arrays = split_arrays(per_class=25, seed=2)
    cal = mq.calibration_ids(arrays, 20, 20260)
    q, inv = ptq(model, "all_linear", cal["x"])
    assert len(inv["requested"]) == 23
    bpda = bpda_model(q, inv["requested"], cal["x"][:2])
    surrogate, names = ma.build_surrogate(model, cal["x"])
    assert len(names) == 23
    v = arrays["validation"]
    val16 = (torch.from_numpy(v["x"][:16]), torch.from_numpy(v["y"][:16]), v["ids"][:16])
    stack = {"fp32": model, "int8": q, "bpda": bpda, "surrogate": surrogate}
    for attack, calls, cands in (("pgd20", 65, 64), ("fgsm", 4, 3)):
        c = unit(stack, val16, acfg, "bpda_int8", attack, 0.05)["checks"]
        assert c["n_selector_calls"] == calls and c["candidates_scored_on_target"] == cands
        assert c["bpda_forward_max_abs_err"] == 0.0 and c["valid"]
    for source in ("fp32_direct", "transfer_fp32_to_int8", "transfer_fp32_to_int8_targetsel", "surrogate_fakequant"):
        assert unit(stack, val16, acfg, source, "fgsm", 0.05)["checks"]["valid"]
    d = ma.direct_int8_grad_diagnostic(q, val16[0], val16[1])
    # observed behaviour on the real converted model: autograd cannot differentiate through quantized::linear
    assert d["undefined_or_zero_fraction"] == 1.0 and d["undefined_gradient_examples"] == 16 and not d["counted_as_robust"]
