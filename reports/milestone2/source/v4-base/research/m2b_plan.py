"""Pure unit-graph builder for the M2B campaign (DESIGN 4.3). No I/O, no torch.

Unit id grammar: s{seed}.{phase}.{kind}.{model}[.{source}.{attack}.eps{eps:g}] with phase "val" or "test".
Validation base plan for four seeds: 8 train + 8 ptq + 16 clean + 12 e4 + 1 decision + 480 attack + 8 diag + 4 calib
+ 4 paircheck = 541 units; the QAT opt-in adds 12 qat + 12 clean + 120 attack = 144 per phase set.
Dependencies that are not units of the returned plan (a stage not selected, or seed 1 when only later seeds are
requested) are external: the scheduler requires them to be completed in the state directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PHASE_TAG = {"validation": "val", "test": "test"}
BASE_STAGES = ("train", "calib", "paircheck", "ptq", "clean", "step0", "attack")
ALL_STAGES = BASE_STAGES + ("qat", "efficiency")
TEST_STAGES = ("clean", "attack", "qat")
FP32_MODELS = ("V1", "V3")
PARENT = {"V2": "V1", "V4": "V3", "FP32-FT": "V1", "FP32-FT-PTQ": "FP32-FT", "QAT-FT": "V1"}
PAIRS = (("V1", "V2"), ("V3", "V4"))
QAT_ARMS = ("FP32-FT", "FP32-FT-PTQ", "QAT-FT")
QAT_KIND = {"FP32-FT": "fp32_ft", "FP32-FT-PTQ": "ft_ptq", "QAT-FT": "qat"}
QAT_SOURCES = {"FP32-FT": ("fp32_direct",), "FP32-FT-PTQ": ("bpda_int8",), "QAT-FT": ("bpda_int8",)}


class PlanError(ValueError):
    """Invalid configuration, stage selection or plan."""


@dataclass(frozen=True)
class Unit:
    id: str
    kind: str          # fingerprint kind: train, calib, paircheck, ptq, clean, e4, step0, attack, diag, qat, fp32_ft, ft_ptq, efficiency
    seed: int
    phase: str         # "validation" | "test"
    stage: str         # the --stages name that selects this unit
    deps: tuple = ()
    params: dict = field(default_factory=dict, compare=False)


def eps_token(eps) -> str:
    return f"{float(eps):g}"


def _uid(seed, phase, kind, *rest) -> str:
    return ".".join([f"s{seed}", PHASE_TAG[phase], kind, *map(str, rest)])


def attack_id(seed, phase, model, source, attack, eps) -> str:
    return _uid(seed, phase, "attack", model, source, attack, f"eps{eps_token(eps)}")


def _check_cfg(cfg):
    for key in ("seeds", "epsilon_grid", "matrix", "e4", "qat"):
        if key not in cfg:
            raise PlanError(f"campaign config lacks {key!r}")
    if not cfg["epsilon_grid"] or not cfg["matrix"]["sources_by_target"].get("int8"):
        raise PlanError("campaign config has an empty epsilon grid or no INT8 sources")


def build_plan(cfg: dict, seeds, phase: str, stages=None, pgd_budget: str = "pgd20") -> list:
    """Ordered (topological, deterministic) list of Units for `seeds` in `phase`."""
    _check_cfg(cfg)
    if phase not in PHASE_TAG:
        raise PlanError(f"phase must be one of {sorted(PHASE_TAG)}")
    if pgd_budget not in ("pgd20", "pgd50x5"):
        raise PlanError("pgd_budget must be pgd20 or pgd50x5")
    if stages is None:
        stages = TEST_STAGES[:2] if phase == "test" else BASE_STAGES
    stages = tuple(stages)
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise PlanError(f"unknown stages {unknown}; choose from {list(ALL_STAGES)}")
    if phase == "test":
        bad = [s for s in stages if s not in TEST_STAGES]
        if bad:
            raise PlanError(f"the test phase accepts only {list(TEST_STAGES)}; got {bad}")
    first_seed = cfg["seeds"][0]
    seeds = sorted(set(seeds), key=lambda s: (s != first_seed, s))
    sources = cfg["matrix"]["sources_by_target"]
    attacks = ("fgsm", pgd_budget)
    eps_grid = list(cfg["epsilon_grid"])
    e4_eps = list(cfg["e4"]["epsilons"])
    decision_id = _uid(first_seed, "validation", "step0", "decision")
    val = phase == "validation"
    units: list[Unit] = []

    def add(uid, kind, seed, stage, deps=(), **params):
        if stage in stages:
            units.append(Unit(uid, kind, seed, phase, stage, tuple(deps), params))

    for seed in seeds:
        S = lambda kind, *rest: _uid(seed, phase, kind, *rest)  # noqa: E731
        calib, pair_check = S("calib"), S("paircheck")
        gate = () if (not val or seed == first_seed) else (decision_id,)  # step 0 blocks every later seed
        if val:
            add(calib, "calib", seed, "calib", gate)
            for m in FP32_MODELS:
                add(S("train", m), "train", seed, "train", gate, model=m, adversarial=(m == "V3"))
            add(pair_check, "paircheck", seed, "paircheck", (S("train", "V1"), S("train", "V3")))
            for m in ("V2", "V4"):
                add(S("ptq", m), "ptq", seed, "ptq", (S("train", PARENT[m]), calib), model=m, parent=PARENT[m])
        for m in ("V1", "V2", "V3", "V4"):
            add(S("clean", m), "clean", seed, "clean", (S("ptq" if m in ("V2", "V4") else "train", m),) if val else (),
                model=m)
        if val:
            for m in ("V2", "V4"):  # diag runs before e4/attack (surrogate note); no dependency on the decision
                add(S("diag", m), "diag", seed, "attack", (S("ptq", m), calib, pair_check), model=m)

        def pair_deps(model):
            """Both models of the pair, paircheck, and for an INT8 model also calib and its diag."""
            parent = PARENT.get(model, model)
            int8 = model in ("V2", "V4")
            deps = [S("train", parent), pair_check]
            if int8:
                deps += [S("ptq", model), calib, S("diag", model)]
            else:
                deps += [S("ptq", dict(PAIRS)[model])]
            return deps

        def combos():
            for fp32, int8 in PAIRS:
                yield fp32, "fp32_direct"
                for s in sources["int8"]:
                    yield int8, s

        if val and seed == first_seed:
            for model, source in (("V1", "fp32_direct"), ("V2", "bpda_int8"), ("V3", "fp32_direct"), ("V4", "bpda_int8")):
                for e in e4_eps:
                    add(S("e4", model, source, f"eps{eps_token(e)}"), "e4", seed, "step0", pair_deps(model),
                        model=model, source=source, eps=e, attacks=("pgd20", "pgd50x5"))
            add(decision_id, "step0", seed, "step0",
                [u.id for u in units if u.kind == "e4" and u.seed == seed], decision_of="budget")
        if "attack" in stages:
            for model, source in combos():
                for atk in attacks:
                    for e in eps_grid:
                        deps = (pair_deps(model) + [decision_id]) if val else []
                        add(attack_id(seed, phase, model, source, atk, e), "attack", seed, "attack", deps,
                            model=model, source=source, attack=atk, eps=e)

        if "qat" in stages and (not val or cfg["qat"]["arms"]):
            arms = tuple(cfg["qat"]["arms"])
            if set(arms) - set(QAT_ARMS):
                raise PlanError(f"unknown QAT arms {sorted(set(arms) - set(QAT_ARMS))}")
            if val:
                add(S("qat", "FP32-FT"), "fp32_ft", seed, "qat", (S("train", "V1"), calib), model="FP32-FT")
                add(S("qat", "QAT-FT"), "qat", seed, "qat", (S("train", "V1"), calib), model="QAT-FT")
                add(S("qat", "FP32-FT-PTQ"), "ft_ptq", seed, "qat", (S("qat", "FP32-FT"), calib), model="FP32-FT-PTQ")
            for arm in arms:
                add(S("clean", arm), "clean", seed, "qat", (S("qat", arm),) if val else (), model=arm)
            for arm in arms:
                for source in QAT_SOURCES[arm]:
                    for atk in attacks:
                        for e in eps_grid:
                            deps = [S("qat", arm)] + ([S("qat", PARENT[arm])] if arm == "FP32-FT-PTQ" else [])
                            deps = (deps + [calib] + [decision_id]) if val else []
                            add(attack_id(seed, phase, arm, source, atk, e), "attack", seed, "qat", deps,
                                model=arm, source=source, attack=atk, eps=e)
        if "efficiency" in stages and val:
            eff_models = ["V1", "V2", "V3", "V4"] + (list(QAT_ARMS) if "qat" in stages else [])
            add(S("efficiency", "all"), "efficiency", seed, "efficiency",
                [S("ptq" if m in ("V2", "V4") else "train", m) if m in ("V1", "V2", "V3", "V4") else S("qat", m)
                 for m in eff_models], models=tuple(eff_models))
    validate_plan(units)
    return units


def validate_plan(units) -> None:
    """Unique ids, and every internal dependency appears earlier (topological, no cycles)."""
    position = {}
    for i, u in enumerate(units):
        if u.id in position:
            raise PlanError(f"duplicate unit id {u.id}")
        position[u.id] = i
    for i, u in enumerate(units):
        for d in u.deps:
            if d in position and position[d] >= i:
                raise PlanError(f"{u.id} depends on later unit {d}")
            if d == u.id:
                raise PlanError(f"{u.id} depends on itself")


def external_deps(units) -> list:
    ids = {u.id for u in units}
    return sorted({d for u in units for d in u.deps if d not in ids})


def counts_by_kind(units) -> dict:
    out = {}
    for u in units:
        out[u.kind] = out.get(u.kind, 0) + 1
    return dict(sorted(out.items()))


def blocked_units(units, statuses: dict) -> dict:
    """{unit id: reason} for every unit that cannot run because a dependency failed or is blocked.

    `statuses` maps unit id to its recorded status (missing means not run). A failed paircheck gives the reason
    "paircheck failed" (it blocks attack, diag and e4 units of the seed; train, ptq and clean stay valid).
    """
    bad = {"failed", "blocked"}
    blocked = {}
    for u in units:
        if statuses.get(u.id) == "completed":
            continue
        for d in u.deps:
            state = "blocked" if d in blocked else statuses.get(d)
            if state in bad:
                if d.endswith(".paircheck"):
                    blocked[u.id] = "paircheck failed"
                else:
                    blocked[u.id] = blocked.get(d) or f"dependency {d} {state}"
                break
    return blocked
