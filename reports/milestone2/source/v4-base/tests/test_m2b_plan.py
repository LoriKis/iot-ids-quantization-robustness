import copy
import json

import pytest

from research import m2b_plan as mp
from research.common import ROOT

CFG = json.loads((ROOT / "configs/m2b/campaign_v2.json").read_text(encoding="utf-8"))
SEEDS = [1, 7, 13, 29]
DECISION = "s1.val.step0.decision"


def _plan(**kw):
    kw.setdefault("seeds", SEEDS)
    kw.setdefault("phase", "validation")
    return mp.build_plan(CFG, **kw)


def _by_id(plan):
    return {u.id: u for u in plan}


def test_base_counts_541():
    plan = _plan()
    assert len(plan) == 541
    assert mp.counts_by_kind(plan) == {"attack": 480, "calib": 4, "clean": 16, "diag": 8, "e4": 12, "paircheck": 4,
                                       "ptq": 8, "step0": 1, "train": 8}


def test_qat_adds_144_per_phase_set():
    base = len(_plan())
    plan = _plan(stages=mp.BASE_STAGES + ("qat",))
    assert len(plan) == base + 144
    extra = [u for u in plan if u.stage == "qat"]
    assert mp.counts_by_kind(extra) == {"attack": 120, "clean": 12, "fp32_ft": 4, "ft_ptq": 4, "qat": 4}


def test_ids_unique_deterministic_and_grammar():
    plan = _plan()
    assert [u.id for u in plan] == [u.id for u in _plan()]
    assert len({u.id for u in plan}) == len(plan)
    ids = _by_id(plan)
    for example in ("s1.val.train.V1", "s1.val.ptq.V2", "s1.val.clean.V4", "s1.val.e4.V2.bpda_int8.eps0.05", DECISION,
                    "s7.val.attack.V4.bpda_int8.pgd20.eps0.05", "s1.val.attack.V1.fp32_direct.fgsm.eps0", "s1.val.calib",
                    "s13.val.paircheck", "s1.val.attack.V3.fp32_direct.pgd20.eps0.1"):
        assert example in ids, example
    qat = _by_id(_plan(stages=("qat",)))
    assert "s1.val.qat.FP32-FT" in qat and "s1.val.qat.QAT-FT" in qat and "s1.val.qat.FP32-FT-PTQ" in qat


def test_plan_is_topological_and_external_deps_are_empty_for_full_plan():
    plan = _plan(stages=mp.BASE_STAGES + ("qat",))
    mp.validate_plan(plan)
    assert mp.external_deps(plan) == []


def test_later_seeds_depend_on_step0_decision():
    ids = _by_id(_plan())
    for seed in (7, 13, 29):
        for u in ids.values():
            if u.seed == seed and u.kind in ("train", "calib", "attack"):
                assert DECISION in u.deps, u.id
    assert DECISION not in ids["s1.val.train.V1"].deps and DECISION not in ids["s1.val.calib"].deps
    assert DECISION in ids["s1.val.attack.V1.fp32_direct.fgsm.eps0.05"].deps


def test_attack_dependencies():
    ids = _by_id(_plan())
    fp32 = ids["s7.val.attack.V1.fp32_direct.pgd20.eps0.05"].deps
    assert {"s7.val.train.V1", "s7.val.ptq.V2", "s7.val.paircheck", DECISION} <= set(fp32)
    assert "s7.val.calib" not in fp32 and "s7.val.diag.V2" not in fp32
    int8 = ids["s7.val.attack.V4.surrogate_fakequant.fgsm.eps0.025"].deps
    assert {"s7.val.train.V3", "s7.val.ptq.V4", "s7.val.paircheck", "s7.val.calib", "s7.val.diag.V4", DECISION} <= set(int8)
    for u in ids.values():
        if u.kind == "e4":
            assert {"s1.val.paircheck"} <= set(u.deps)
            if u.params["model"] in ("V2", "V4"):
                assert {"s1.val.calib", f"s1.val.diag.{u.params['model']}"} <= set(u.deps)


def test_ptq_depend_on_calib_and_do_not_write_it():
    ids = _by_id(_plan())
    for seed in SEEDS:
        for model, parent in (("V2", "V1"), ("V4", "V3")):
            ptq = ids[f"s{seed}.val.ptq.{model}"]
            assert set(ptq.deps) == {f"s{seed}.val.train.{parent}", f"s{seed}.val.calib"}
        assert [u.id for u in ids.values() if u.kind == "calib" and u.seed == seed] == [f"s{seed}.val.calib"]
        assert [u.id for u in ids.values() if u.kind == "paircheck" and u.seed == seed] == [f"s{seed}.val.paircheck"]
    assert {u.params["model"] for u in ids.values() if u.kind == "ptq" and u.seed == 1} == {"V2", "V4"}


def test_decision_depends_on_all_e4_and_diag_does_not_depend_on_decision():
    ids = _by_id(_plan())
    e4 = {u.id for u in ids.values() if u.kind == "e4"}
    assert len(e4) == 12 and set(ids[DECISION].deps) == e4
    for u in ids.values():
        if u.kind == "diag":
            assert DECISION not in u.deps and f"s{u.seed}.val.calib" in u.deps


def test_failed_paircheck_blocks_attack_diag_e4_but_not_train_ptq_clean():
    plan = _plan()
    statuses = {u.id: "completed" for u in plan if u.kind in ("train", "calib", "ptq", "clean")}
    statuses["s7.val.paircheck"] = "failed"
    blocked = mp.blocked_units(plan, statuses)
    for u in plan:
        if u.seed == 7 and u.kind in ("attack", "diag"):
            assert blocked.get(u.id) == "paircheck failed", u.id
        if u.kind in ("train", "ptq", "clean", "calib") or u.kind == "paircheck":
            assert u.id not in blocked
    seed1 = {u.id for u in plan if u.seed == 1}
    assert not (seed1 & set(blocked))
    statuses1 = dict(statuses, **{"s1.val.paircheck": "failed"})
    blocked1 = mp.blocked_units(plan, statuses1)
    assert all(blocked1[u.id] == "paircheck failed" for u in plan if u.seed == 1 and u.kind in ("e4", "diag", "attack"))
    assert DECISION in blocked1  # step 0 cannot finish, so later seeds stay blocked
    assert "s7.val.attack.V1.fp32_direct.fgsm.eps0" in blocked1


def test_test_phase_has_no_training_ptq_e4_or_qat_training():
    plan = mp.build_plan(CFG, SEEDS, "test")
    assert {u.kind for u in plan} == {"clean", "attack"}
    assert len(plan) == 16 + 480
    assert all(u.phase == "test" and ".test." in u.id for u in plan)
    assert "s7.test.attack.V4.bpda_int8.pgd20.eps0.05" in _by_id(plan)
    with_qat = mp.build_plan(CFG, SEEDS, "test", stages=("clean", "attack", "qat"))
    assert {u.kind for u in with_qat} == {"clean", "attack"} and len(with_qat) == 496 + 12 + 120
    with pytest.raises(mp.PlanError):
        mp.build_plan(CFG, SEEDS, "test", stages=("train",))


def test_stage_selection_and_unknown_stage():
    only_train = _plan(stages=("train",))
    assert {u.kind for u in only_train} == {"train"} and len(only_train) == 8
    assert DECISION in mp.external_deps(only_train)
    step0 = _plan(seeds=[1], stages=("step0",))
    assert mp.counts_by_kind(step0) == {"e4": 12, "step0": 1}
    with pytest.raises(mp.PlanError):
        _plan(stages=("bogus",))


def test_pgd50x5_budget_renames_attack_ids():
    plan = _plan(pgd_budget="pgd50x5")
    assert len(plan) == 541
    assert "s7.val.attack.V4.bpda_int8.pgd50x5.eps0.05" in _by_id(plan)


def test_seed_subset_plan_keeps_seed1_as_external_dependency():
    plan = _plan(seeds=[7, 13, 29])
    assert all(u.seed != 1 for u in plan) and len(plan) == 3 * (12 + 120)
    assert mp.external_deps(plan) == [DECISION]


def test_efficiency_stage_is_one_unit_per_seed():
    plan = _plan(stages=("efficiency",))
    assert [u.id for u in plan] == [f"s{s}.val.efficiency.all" for s in (1, 7, 13, 29)]
    assert plan[0].params["models"] == ("V1", "V2", "V3", "V4")
