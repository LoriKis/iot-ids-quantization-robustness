"""Milestone 2A attack plumbing: row-tied model adapter, feasible-set policy, checked attacks, selector pair.

Feature-space attacks only; no guarantee that perturbed vectors correspond to valid network traffic.
Part 1: AttackInvariantError, RowTiedModel, feasible-set wrapper, checked_attack, CheckedPair.
Part 2 (below the part-1 section): configuration, INT8 attack sources, fake-quantization surrogate,
direct-INT8-gradient diagnostic, per-unit evaluation with attack_checks, E4 budget check.
Design: docs/MILESTONE2_PROTOCOL.md sections D, E1-E6.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.ao import quantization as aq

from . import attacks
from .common import ROOT, atomic_json
from .data import stratified_cap
from .m2_split import ConfigError, PartitionGuardError, id_hash
from .paper_described_adam import derived_seed
from .quantization import prepare_static

SEQUENCE_LENGTH = 50
BUDGET_TOLERANCE = 2e-6  # equals the attacks.attack engine tolerance (protocol E5)


def _owned_numpy_bytes(t: torch.Tensor) -> bytes:
    """Raw C-order bytes of a tensor via a private clone, so NumPy never exports storage a live model tensor owns."""
    return t.detach().cpu().clone(memory_format=torch.contiguous_format).numpy().tobytes()


def model_state_hash(model: nn.Module) -> str:
    """SHA-256 over the state_dict; unlike provenance.state_hash it also handles packed INT8 state.

    Plain float tensors are hashed exactly as provenance.state_hash does (key, then raw bytes).
    Quantized tensors contribute integer values, scales and zero points; tuples (packed params) are
    walked; dtypes and other scalars contribute their repr.
    Tensor bytes come from private clones, so hashing never exports live storage to NumPy (a fresh observer's
    min_val/max_val must stay resizable on their first forward).
    """
    h = hashlib.sha256()

    def feed(key, val):
        if isinstance(val, torch.Tensor):
            t = val.detach().cpu()
            if t.is_quantized:
                h.update(f"{key}|quantized|{t.qscheme()}".encode())
                h.update(_owned_numpy_bytes(t.int_repr()))
                if t.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                    h.update(_owned_numpy_bytes(t.q_per_channel_scales()))
                    h.update(_owned_numpy_bytes(t.q_per_channel_zero_points()))
                    h.update(str(t.q_per_channel_axis()).encode())
                else:
                    h.update(repr((t.q_scale(), t.q_zero_point())).encode())
            else:
                h.update(key.encode())
                h.update(_owned_numpy_bytes(t))
        elif isinstance(val, (tuple, list)):
            for i, item in enumerate(val):
                feed(f"{key}[{i}]", item)
        else:
            h.update(f"{key}|{val!r}".encode())

    for key, val in model.state_dict().items():
        feed(key, val)
    return h.hexdigest()


class AttackInvariantError(RuntimeError):
    """An attack-run invariant failed; the unit is invalid. check_name names the failed check."""

    def __init__(self, check_name: str, message: str = ""):
        super().__init__(f"{check_name}: {message}" if message else check_name)
        self.check_name = check_name


class RowTiedModel(nn.Module):
    """Takes (B, 39) rows, repeats each row to (B, 50, 39) and returns {'logits': ...}.

    A perturbation of the row is therefore tied across all 50 synthetic sequence positions.
    """

    def __init__(self, model: nn.Module, length: int = SEQUENCE_LENGTH):
        super().__init__()
        self.model = model
        self.length = length

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"RowTiedModel expects (B, features), got {tuple(x.shape)}")
        seq = x.unsqueeze(1).repeat(1, self.length, 1)
        return {"logits": self.model(seq)["logits"]}


def feasible_bounds(x: torch.Tensor):
    """Tensor bounds for attacks.attack (Python floats are rejected by the engine's torch.minimum)."""
    return torch.zeros_like(x), torch.ones_like(x)


def feasible_interval(x: torch.Tensor, epsilon: float):
    """Protocol E2: lower = max(x - eps, min(0, x)), upper = min(x + eps, max(1, x)); always contains x."""
    zeros, ones = feasible_bounds(x)
    return torch.maximum(x - epsilon, torch.minimum(zeros, x)), torch.minimum(x + epsilon, torch.maximum(ones, x))


def feasible_attack(model, x, y, *, kind, epsilon, steps=10, alpha=None, restarts=1, random_start=True,
                    seed=0, selector=None):
    """attacks.attack under the single feasible-set policy of protocol E2."""
    lower, upper = feasible_bounds(x)
    return attacks.attack(model, x, y, kind=kind, epsilon=epsilon, steps=steps, alpha=alpha, restarts=restarts,
                          random_start=random_start, lower=lower, upper=upper, seed=seed, selector=selector)


def _grad_snapshot(model: nn.Module):
    return [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]


def _grads_equal(model: nn.Module, snapshot) -> bool:
    for p, old in zip(model.parameters(), snapshot):
        if (p.grad is None) != (old is None):
            return False
        if old is not None and not torch.equal(p.grad, old):
            return False
    return True


def _name_floating_point_error(error: FloatingPointError) -> str:
    text = str(error).lower()
    if "gradient" in text:
        return "finite_gradients"
    if "input" in text:
        return "finite_inputs"
    return "finite_logits"


def checked_attack(model, x, y, *, kind, epsilon, steps=10, alpha=None, restarts=1, random_start=True,
                   seed=0, selector=None, report=None):
    """feasible_attack plus the E5 invariants. Returns x_adv; raises AttackInvariantError(check_name).

    `model` is the attack source (e.g. RowTiedModel(...)); its parameters, modes and .grad are checked.
    If `report` is a dict it is filled with one boolean per check that passed.
    """
    report = {} if report is None else report
    if not torch.isfinite(x).all():
        raise AttackInvariantError("finite_inputs", "clean input contains non-finite values")
    modes = [(m, m.training) for m in model.modules()]
    hash_before = model_state_hash(model)
    grads_before = _grad_snapshot(model)
    labels_before = y.detach().clone()
    seen_training = []
    hook = model.register_forward_pre_hook(lambda module, args: seen_training.append(any(m.training for m in module.modules())))
    try:
        x_adv = feasible_attack(model, x, y, kind=kind, epsilon=epsilon, steps=steps, alpha=alpha, restarts=restarts,
                                random_start=random_start, seed=seed, selector=selector)
    except FloatingPointError as error:
        raise AttackInvariantError(_name_floating_point_error(error), str(error)) from error
    except AssertionError as error:
        name = "finite_inputs" if not torch.isfinite(x).all() else "linf_budget"
        raise AttackInvariantError(name, str(error)) from error
    finally:
        hook.remove()
    report["finite_inputs"] = True
    report["finite_logits"] = True
    report["finite_gradients"] = True
    if any(seen_training):
        raise AttackInvariantError("mode_during_attack", "model was in training mode while the attack ran")
    report["mode_during_attack"] = True
    if any(m.training != was for m, was in modes):
        raise AttackInvariantError("mode_restored", "module modes were not restored after the attack")
    report["mode_restored"] = True
    if x_adv.shape != x.shape or not torch.isfinite(x_adv).all():
        raise AttackInvariantError("finite_inputs", "attack output is non-finite or has the wrong shape")
    if float((x_adv - x).abs().max()) > epsilon + BUDGET_TOLERANCE:
        raise AttackInvariantError("linf_budget", f"|x_adv - x|_inf exceeds {epsilon} + {BUDGET_TOLERANCE}")
    report["linf_budget"] = True
    lower, upper = feasible_interval(x, epsilon)
    if bool((x_adv < lower).any() or (x_adv > upper).any()):
        raise AttackInvariantError("feasible_set", "attack output left the feasible set")
    report["feasible_set"] = True
    if model_state_hash(model) != hash_before:
        raise AttackInvariantError("weights_unchanged", "model parameters changed during the attack")
    report["weights_unchanged"] = True
    if not _grads_equal(model, grads_before):
        raise AttackInvariantError("no_parameter_grad", "parameter .grad changed during the attack")
    report["no_parameter_grad"] = True
    if not torch.equal(labels_before, y):
        raise AttackInvariantError("labels_unchanged", "labels were modified during the attack")
    report["labels_unchanged"] = True
    return x_adv


def _as_logits(out):
    return out["logits"] if isinstance(out, dict) else out


class CheckedPair:
    """Selector for BPDA rows: candidates are scored by the real INT8 model; the BPDA wrapper is compared.

    Both models take the same input and may return {'logits': ...} or a logits tensor. The BPDA forward must
    agree exactly with the INT8 forward, so max_abs_err must stay 0.0.
    """

    def __init__(self, bpda_model, int8_model):
        self.bpda_model = bpda_model
        self.int8_model = int8_model
        self.max_abs_err = 0.0
        self.n_selector_calls = 0
        self.n_clean_calls = 0

    def _score(self, x):
        with torch.no_grad():
            a = _as_logits(self.int8_model(x))
            b = _as_logits(self.bpda_model(x))
            err = float((a - b).abs().max())
        self.max_abs_err = float("inf") if not math.isfinite(err) else max(self.max_abs_err, err)
        return a

    def __call__(self, x):
        self.n_selector_calls += 1
        return self._score(x)

    def clean(self, x):
        """Explicit clean-prediction call made outside attack(); counted separately."""
        self.n_clean_calls += 1
        return self._score(x)


# =============================================================== part 2: evaluation sources
DEFAULT_CONFIG = ROOT / "configs/m2/attacks.json"
CLAIM_FRAMING = ("Feature-space attacks; no guarantee that perturbed vectors correspond to valid network traffic. "
                 "Perturbations are tied to the original feature row (one 39-d delta broadcast to all 50 synthetic positions).")
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "epsilon_grid", "space", "row_tied",
            "perturbation_structure", "feasible_set_policy", "budget_tolerance", "attacks",
            "expected_selector_calls_per_batch", "e4", "sources", "surrogate", "seed_name", "seed_rule",
            "evaluation", "bpda_forward_tolerance"}
SOURCES = ("fp32_direct", "transfer_fp32_to_int8", "transfer_fp32_to_int8_targetsel", "bpda_int8",
           "surrogate_fakequant", "surrogate_fakequant_targetsel")
DIAGNOSTIC_SOURCE = "direct_int8_grad"
# gradient model, candidate selector and scoring target for each source (protocol E6)
SOURCE_SPECS = {
    "fp32_direct": {"grad": "fp32", "selector": None, "target": "fp32", "pure": False},
    "transfer_fp32_to_int8": {"grad": "fp32", "selector": None, "target": "int8", "pure": True},
    "transfer_fp32_to_int8_targetsel": {"grad": "fp32", "selector": "count", "target": "int8", "pure": False},
    "bpda_int8": {"grad": "bpda", "selector": "pair", "target": "int8", "pure": False},
    "surrogate_fakequant": {"grad": "surrogate", "selector": None, "target": "int8", "pure": True},
    "surrogate_fakequant_targetsel": {"grad": "surrogate", "selector": "count", "target": "int8", "pure": False},
}
SOURCE_NOTES = {
    "bpda_int8": "approximate backward pass: the forward is the real INT8 model, the backward uses dequantized weights (BPDA)",
    "transfer_fp32_to_int8": "pure transfer: no query to the INT8 target during attack construction",
    "transfer_fp32_to_int8_targetsel": "gradient from the FP32 parent, candidate choice by INT8 target queries (query-assisted, not a pure transfer)",
    "surrogate_fakequant": "pure surrogate: the fake-quantized surrogate is a different function than the deployed INT8 model",
    "surrogate_fakequant_targetsel": "fake-quantized surrogate gradient, candidate choice by INT8 target queries (query-assisted)",
    "fp32_direct": "white-box attack on the FP32 model",
}


class AttackUnitError(RuntimeError):
    """An evaluation unit failed one of the E5 checks; `checks` holds the full attack_checks record."""

    def __init__(self, message, checks):
        super().__init__(message)
        self.checks = checks


class SubsetTooSmallError(RuntimeError):
    """The E4 subset is smaller than the frozen minimum; stop and ask the user (nothing is changed automatically)."""


def expected_selector_calls(kind: str, steps: int, restarts: int) -> dict:
    """Selector calls per batch derived from the attacks.attack control flow (protocol E3)."""
    nr, ns = (1, 1) if kind == "fgsm" else (restarts, steps)
    candidates = 1 + nr * (1 + ns)  # FGSM point + per restart: start point and each iterate
    return {"selector_calls": 1 + candidates, "candidates": candidates}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or set(cfg) != TOP_KEYS:
        raise ConfigError(f"unknown or missing attack configuration keys: "
                          f"{sorted(set(cfg) ^ TOP_KEYS) if isinstance(cfg, dict) else 'not an object'}")
    grid = cfg["epsilon_grid"]
    if (not isinstance(grid, list) or not grid or grid[0] != 0
            or any(isinstance(e, bool) or not isinstance(e, (int, float)) or not 0 <= e <= 1 for e in grid)
            or any(b <= a for a, b in zip(grid, grid[1:]))):
        raise ConfigError("epsilon_grid must be strictly increasing numbers in [0, 1] starting at 0")
    if cfg["row_tied"] is not True:
        raise ConfigError("row_tied must be true (protocol E1)")
    if cfg["bpda_forward_tolerance"] != 0:
        raise ConfigError("bpda_forward_tolerance must be 0")
    if not _is_int(cfg["evaluation"]["batch_size"]) or not 1 <= cfg["evaluation"]["batch_size"] <= 4096:
        raise ConfigError("evaluation.batch_size must be an integer in 1..4096")
    if cfg["evaluation"]["evaluation_partition"] != "validation":
        raise ConfigError("evaluation_partition must be 'validation' in M2A")
    table = cfg["expected_selector_calls_per_batch"]
    for name, spec in cfg["attacks"].items():
        if spec["kind"] not in ("fgsm", "pgd") or not _is_int(spec["steps"]) or spec["steps"] < 1 \
                or not _is_int(spec["restarts"]) or spec["restarts"] < 1:
            raise ConfigError(f"invalid attack definition {name!r}")
        if spec["kind"] == "pgd" and not (0 < spec["alpha_fraction"] <= 1):
            raise ConfigError(f"invalid alpha_fraction for {name!r}")
        want = expected_selector_calls(spec["kind"], spec["steps"], spec["restarts"])
        if name not in table or table[name] != want:
            raise ConfigError(f"expected selector-call table for {name!r} differs from the attack control flow {want}")
    if table.get("epsilon_0") != {"selector_calls": 0, "candidates": 0}:
        raise ConfigError("epsilon_0 must expect 0 selector calls and 0 candidates")
    for name in (cfg["e4"]["reference_budget"], cfg["e4"]["candidate_budget"]):
        if name not in cfg["attacks"]:
            raise ConfigError(f"E4 budget {name!r} is not defined in attacks")
    return cfg


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"attack configuration not found: {path}")
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def attack_spec(cfg: dict, name: str, epsilon: float) -> dict:
    """Keyword arguments for checked_attack for the named budget at `epsilon`."""
    if name not in cfg["attacks"]:
        raise ValueError(f"unknown attack budget {name!r}")
    a = cfg["attacks"][name]
    if a["kind"] == "fgsm":
        return {"kind": "fgsm", "steps": 1, "alpha": None, "restarts": 1, "random_start": False}
    return {"kind": "pgd", "steps": a["steps"], "alpha": a["alpha_fraction"] * epsilon,
            "restarts": a["restarts"], "random_start": a["random_start"]}


def expected_counts(cfg: dict, name: str, epsilon: float) -> dict:
    """Expected selector calls and candidates per batch, read from attacks.json (not hard-coded)."""
    return cfg["expected_selector_calls_per_batch"]["epsilon_0" if epsilon == 0 else name]


class CountingSelector:
    """Selector for *_targetsel rows: scores every candidate on the INT8 target and counts the calls."""

    max_abs_err = None

    def __init__(self, model):
        self.model = model
        self.n_selector_calls = 0
        self.n_clean_calls = 0

    def _score(self, x):
        with torch.no_grad():
            return _as_logits(self.model(x))

    def __call__(self, x):
        self.n_selector_calls += 1
        return self._score(x)

    def clean(self, x):
        self.n_clean_calls += 1
        return self._score(x)


class QueryCounter(nn.Module):
    """Wraps the scoring target and counts every forward call made on it."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.model(x)


# ------------------------------------------------------------------ surrogate (E6)
def _island_scales(model, name):
    island = model.get_submodule(name)
    quant, linear = island.quant, island.linear
    if hasattr(quant, "activation_post_process"):  # fake-quant surrogate (QAT-prepared island)
        weight = linear.weight_fake_quant
        return {"input_scale": quant.activation_post_process.scale.detach().flatten().float(),
                "input_zero_point": quant.activation_post_process.zero_point.detach().flatten().double(),
                "output_scale": linear.activation_post_process.scale.detach().flatten().float(),
                "output_zero_point": linear.activation_post_process.zero_point.detach().flatten().double(),
                "weight_scale": weight.scale.detach().flatten().float()}
    weight = linear.weight()  # converted INT8 island
    return {"input_scale": quant.scale.detach().flatten().float(),
            "input_zero_point": quant.zero_point.detach().flatten().double(),
            "output_scale": torch.tensor([float(linear.scale)]),
            "output_zero_point": torch.tensor([float(linear.zero_point)], dtype=torch.float64),
            "weight_scale": weight.q_per_channel_scales().detach().float()}


def build_surrogate(parent, calibration_x, *, backend: str = "fbgemm", batch_size: int = 32):
    """Fake-quantized surrogate of `parent` (protocol E6): same 23 islands, observers calibrated on the PTQ rows.

    Returns (surrogate, island_names); the surrogate is in eval mode with observers disabled and fake
    quantization enabled. Raises RuntimeError if a Dropout is in training mode while it calibrates or if the
    global torch RNG state changed.
    """
    rng_before = torch.get_rng_state().clone()
    surrogate, names = prepare_static(parent, "all_linear", calibration_x[:2], qat=True, backend=backend)
    # prepare_qat builds torch.ao.nn.qat.Linear modules whose throwaway random initialisation draws from the
    # global RNG before from_float overwrites it with the parent's weights. Restore the state so the surrogate
    # build leaves it untouched; the calibration pass below is then checked to consume nothing.
    torch.set_rng_state(rng_before)
    surrogate.eval()  # prepare_static leaves train() on, which would activate dropout
    surrogate.apply(aq.enable_observer)
    surrogate.apply(aq.enable_fake_quant)
    violations = []
    handles = [m.register_forward_pre_hook(lambda module, args: violations.append(module.training))
               for m in surrogate.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
    try:
        with torch.no_grad():
            for batch in calibration_x.split(batch_size):
                surrogate(batch)
    finally:
        for handle in handles:
            handle.remove()
    surrogate.apply(aq.disable_observer)
    surrogate.eval()
    if any(violations):
        raise RuntimeError("a Dropout module was in training mode while the surrogate was calibrated")
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError("the global torch RNG state changed while the surrogate was built")
    for p in surrogate.parameters():
        p.requires_grad_(True)
    return surrogate, names


def surrogate_fidelity(surrogate, int8, names, x_rows) -> dict:
    """Scale/zero-point differences to the deployed INT8 model and the maximum logit difference on `x_rows`."""
    per_island, worst_scale, worst_zero_point = {}, 0.0, 0.0
    for name in names:
        s, q = _island_scales(surrogate, name), _island_scales(int8, name)
        scale = max(float((s[k] - q[k]).abs().max()) for k in ("input_scale", "output_scale", "weight_scale"))
        zero_point = max(float((s[k] - q[k]).abs().max()) for k in ("input_zero_point", "output_zero_point"))
        per_island[name] = {"max_scale_diff": scale, "max_zero_point_diff": zero_point}
        worst_scale, worst_zero_point = max(worst_scale, scale), max(worst_zero_point, zero_point)
    with torch.no_grad():
        a, b = RowTiedModel(surrogate).eval()(x_rows)["logits"], RowTiedModel(int8).eval()(x_rows)["logits"]
    fidelity = {"islands": per_island, "max_scale_diff": worst_scale, "max_zero_point_diff": worst_zero_point,
                "max_logit_diff_to_int8": float((a - b).abs().max()),
                "argmax_agreement_with_int8": float(a.argmax(1).eq(b.argmax(1)).float().mean()),
                "n_rows": int(len(x_rows))}
    fidelity["note"] = (f"surrogate scales differ from deployed INT8 scales (max diff = {worst_scale:.6g})"
                        if worst_scale > 0 else "surrogate scales equal the deployed INT8 scales")
    return fidelity


# ------------------------------------------------------------ direct INT8 gradients (diagnostic)
def direct_int8_grad_diagnostic(int8, x, y, batch_size: int = 64) -> dict:
    """Autograd straight through the converted modules. Never reported as robustness.

    Counts examples whose gradient is undefined (a RuntimeError is counted as fraction 1.0 for the batch),
    all-zero or non-finite, over the full evaluated set.
    """
    model = RowTiedModel(int8).eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    n = len(x)
    undefined = zero = nonfinite = 0
    first_error = None
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size].detach().clone().requires_grad_(True)
        yb = y[start:start + batch_size]
        loss = F.cross_entropy(model(xb)["logits"], yb)  # forward errors are real errors and propagate
        try:
            grad = torch.autograd.grad(loss, xb)[0]
        except RuntimeError as error:
            undefined += len(xb)
            first_error = first_error or f"{type(error).__name__}: {error}"
            continue
        finite = torch.isfinite(grad).flatten(1).all(1)
        nonfinite += int((~finite).sum())
        zero += int((finite & grad.flatten(1).eq(0).all(1)).sum())
    bad = undefined + zero + nonfinite
    behaviour = ("raises RuntimeError (no grad_fn through quantized operators)" if undefined == n
                 else "zero or non-finite gradients" if bad else "usable gradients")
    return {"source": DIAGNOSTIC_SOURCE, "n": n, "undefined_gradient_examples": undefined,
            "zero_gradient_examples": zero, "nonfinite_gradient_examples": nonfinite,
            "undefined_or_zero_fraction": bad / n if n else None, "behaviour": behaviour,
            "first_error": first_error, "counted_as_robust": False,
            "note": "gradient-masking diagnostic only; a missing or zero gradient is never robustness"}


# ------------------------------------------------------------------ evaluation unit
def _label_hash(y) -> str:
    return hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else repr(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_new(path: Path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    atomic_json(path, _json_safe(value))


def evaluate_unit(*, source, attack_name, epsilon, x, y, ids, models, cfg, train_seed, model_id,
                  evaluation_partition="validation", model_ids=None, notes=None, out_dir=None,
                  expected_ids_sha256=None, expected_label_sha256=None, raise_on_invalid=True) -> dict:
    """Run one (model, source, attack, epsilon) unit on identical rows and return {row, checks, arrays}.

    x: (N, 39) MinMax-scaled rows (never clipped, never dropped); y: (N,) original labels; ids: row IDs.
    models: {'fp32', 'int8', 'bpda', 'surrogate'} sequence models returning {'logits': ...}; only those the
    source needs must be present. Only the validation partition is accepted in M2A.
    """
    if evaluation_partition != cfg["evaluation"]["evaluation_partition"]:
        raise PartitionGuardError(f"evaluation_partition={evaluation_partition!r} is not allowed in M2A "
                                  f"(only {cfg['evaluation']['evaluation_partition']!r})")
    if source not in SOURCE_SPECS:
        raise ValueError(f"unknown attack source {source!r}")
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    spec = SOURCE_SPECS[source]
    missing = [k for k in (spec["grad"], spec["target"]) if k not in models]
    if missing:
        raise ValueError(f"source {source} needs models {missing}")
    if not (len(x) == len(y) == len(ids)) or x.dim() != 2:
        raise ValueError("x, y and ids must describe the same N rows; x must be (N, features)")
    model_ids = {"fp32": "fp32", "int8": "int8", "bpda": "bpda_int8", "surrogate": "surrogate_fakequant",
                 **(model_ids or {})}
    kwargs_template = attack_spec(cfg, attack_name, epsilon)
    expected = expected_counts(cfg, attack_name, epsilon)
    batch = cfg["evaluation"]["batch_size"]
    n = len(x)
    n_batches = (n + batch - 1) // batch

    grad_model = RowTiedModel(models[spec["grad"]]).eval()
    target = QueryCounter(RowTiedModel(models[spec["target"]])).eval()
    if spec["selector"] == "pair":
        selector = CheckedPair(grad_model, target)
    elif spec["selector"] == "count":
        selector = CountingSelector(target)
    else:
        selector = None
    target_hash_before = model_state_hash(models[spec["target"]])
    grad_hash_before = model_state_hash(models[spec["grad"]])

    checks = {"model_id": model_id, "source": source, "attack": attack_name, "epsilon": epsilon,
              "train_seed": train_seed, "evaluation_partition": evaluation_partition,
              "selector_target_id": model_ids[spec["target"]] if selector is not None else None,
              "selector_target_state_hash": target_hash_before if selector is not None else None,
              "scoring_target_id": model_ids[spec["target"]], "gradient_model_id": model_ids[spec["grad"]],
              "gradient_model_state_hash": grad_hash_before, "n_examples": n, "n_batches": n_batches,
              "example_id_sha256": id_hash(ids), "label_sha256": _label_hash(y),
              "expected_per_batch": expected, "invariants": {}, "failures": [], "valid": None}
    failures = checks["failures"]
    if expected_ids_sha256 is not None and checks["example_id_sha256"] != expected_ids_sha256:
        failures.append("example_ids_match")
    if expected_label_sha256 is not None and checks["label_sha256"] != expected_label_sha256:
        failures.append("labels_match")

    pred_clean, pred_adv, linf = [], [], []
    eps0_bitwise, construction_queries, clean_queries = True, 0, 0
    passed = {}
    started = time.perf_counter()
    try:
        for b, start in enumerate(range(0, n, batch)):
            xb, yb = x[start:start + batch], y[start:start + batch]
            before = target.calls
            clean = selector.clean(xb) if selector is not None else target(xb)["logits"]  # outside attack()
            clean_queries += target.calls - before
            report = {}
            before = target.calls
            x_adv = checked_attack(grad_model, xb, yb, epsilon=epsilon, report=report,
                                   seed=derived_seed(cfg["seed_name"], train_seed, model_id, source, epsilon, b),
                                   selector=selector, **kwargs_template)
            construction_queries += target.calls - before
            for key, ok in report.items():
                passed[key] = passed.get(key, True) and ok
            with torch.no_grad():
                adv_logits = target(x_adv)["logits"]
            if not torch.isfinite(adv_logits).all():
                raise AttackInvariantError("finite_logits", "non-finite logits on the retained candidate")
            eps0_bitwise = eps0_bitwise and torch.equal(x_adv, xb)
            pred_clean.append(clean.argmax(1))
            pred_adv.append(adv_logits.argmax(1))
            linf.append((x_adv - xb).abs().amax(1))
    except AttackInvariantError as error:
        failures.append(error.check_name)
        checks["invariants"] = passed
        checks["failed_check"] = error.check_name
        checks["failure_message"] = str(error)
        checks["valid"] = False
        if out_dir is not None:
            _write_new(Path(out_dir) / "attack_checks.json", checks)
        if raise_on_invalid:
            raise AttackUnitError(f"{source}/{attack_name}/eps={epsilon}: invariant {error.check_name} failed", checks) from error
        return {"row": None, "checks": checks, "arrays": None}
    seconds = time.perf_counter() - started
    pred_clean, pred_adv, linf = torch.cat(pred_clean), torch.cat(pred_adv), torch.cat(linf)
    checks["invariants"] = passed

    # E5 count rules
    if selector is not None:
        checks["n_selector_calls"] = selector.n_selector_calls
        checks["n_clean_calls"] = selector.n_clean_calls
        checks["candidates_scored_on_target"] = selector.n_selector_calls - (n_batches if epsilon > 0 else 0)
        want_calls = expected["selector_calls"] * n_batches
        want_candidates = expected["candidates"] * n_batches
        checks["expected_n_selector_calls"] = want_calls
        checks["expected_candidates_scored_on_target"] = want_candidates
        checks["counts_ok"] = (selector.n_selector_calls == want_calls
                               and checks["candidates_scored_on_target"] == want_candidates
                               and selector.n_clean_calls == n_batches)
        if not checks["counts_ok"]:
            failures.append("selector_call_counts")
    else:
        checks.update(n_selector_calls=None, n_clean_calls=None, candidates_scored_on_target=None,
                      expected_n_selector_calls=None, expected_candidates_scored_on_target=None, counts_ok=None)
    checks["target_queries_during_construction"] = construction_queries
    checks["target_queries_clean_predictions"] = clean_queries
    checks["target_queries_final_evaluation"] = n_batches
    if spec["pure"] and construction_queries != 0:
        failures.append("pure_source_queried_target")
    if spec["selector"] is not None and construction_queries != selector.n_selector_calls:
        failures.append("target_query_accounting")

    # BPDA forward exactness (null at epsilon 0, but a non-zero clean-call error still fails the unit)
    if source == "bpda_int8":
        error = selector.max_abs_err
        checks["bpda_forward_max_abs_err"] = None if epsilon == 0 else error
        checks["bpda_forward_max_abs_err_all_calls"] = error
        if not (math.isfinite(error) and error == cfg["bpda_forward_tolerance"]):
            failures.append("bpda_forward_exact")
    else:
        checks["bpda_forward_max_abs_err"] = None
        checks["bpda_forward_max_abs_err_all_calls"] = None

    if epsilon == 0:
        checks["eps0_bitwise_identity"] = bool(eps0_bitwise)
        checks["eps0_prediction_identity"] = bool(torch.equal(pred_clean, pred_adv))
        if not (checks["eps0_bitwise_identity"] and checks["eps0_prediction_identity"]):
            failures.append("epsilon_zero_identity")
    else:
        checks["eps0_bitwise_identity"] = None
        checks["eps0_prediction_identity"] = None

    if (model_state_hash(models[spec["target"]]) != target_hash_before
            or model_state_hash(models[spec["grad"]]) != grad_hash_before):
        failures.append("models_unchanged")

    # out-of-range accounting (reported, never clipped or dropped)
    outside = (x < 0) | (x > 1)
    checks["n_rows_out_of_range"] = int(outside.any(1).sum())
    checks["n_out_of_range_coordinates"] = int(outside.sum())
    checks["n_one_sided_coordinates"] = int(outside.sum()) if epsilon > 0 else 0
    checks["valid"] = not failures

    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    clean_correct, adv_correct = pred_clean.eq(yt), pred_adv.eq(yt)
    robust = clean_correct & adv_correct
    n_clean_correct = int(clean_correct.sum())
    row = {"attack_source": source, "target": model_ids[spec["target"]], "model_id": model_id, "attack": attack_name,
           "epsilon": epsilon, "n": n, "clean_accuracy": float(clean_correct.float().mean()),
           "accuracy_under_attack": float(adv_correct.float().mean()),
           "robust_accuracy": float(robust.float().mean()),
           "attack_success_rate": (int((clean_correct & ~adv_correct).sum()) / n_clean_correct) if n_clean_correct else None,
           "max_linf": float(linf.max()) if len(linf) else 0.0, "attack_seconds": seconds,
           "example_id_sha256": checks["example_id_sha256"], "label_sha256": checks["label_sha256"],
           "valid": checks["valid"], "note": SOURCE_NOTES[source], "extra_notes": list(notes or []),
           "framing": CLAIM_FRAMING}
    arrays = {"ids": np.asarray(ids), "y": yt.numpy(), "pred_clean": pred_clean.numpy(),
              "pred_adv": pred_adv.numpy(), "robust_correct": robust.numpy(), "linf": linf.numpy()}
    if out_dir is not None:
        _write_new(Path(out_dir) / "attack_checks.json", checks)
        _write_new(Path(out_dir) / "attack_result.json", row)
    if failures and raise_on_invalid:
        raise AttackUnitError(f"{source}/{attack_name}/eps={epsilon}: failed checks {failures}", checks)
    return {"row": row, "checks": checks, "arrays": arrays}


def union_row(results: list) -> dict:
    """Per-example 'any source succeeded' row over units with identical examples (separately labelled)."""
    if not results:
        raise ValueError("no results to combine")
    first = results[0]["row"]
    for r in results:
        row = r["row"]
        if (row["example_id_sha256"], row["label_sha256"], row["attack"], row["epsilon"], row["target"]) != \
                (first["example_id_sha256"], first["label_sha256"], first["attack"], first["epsilon"], first["target"]):
            raise ValueError("union rows need identical examples, labels, attack, epsilon and target")
    robust = np.logical_and.reduce([r["arrays"]["robust_correct"] for r in results])
    clean = results[0]["arrays"]["pred_clean"] == results[0]["arrays"]["y"]
    return {"attack_source": "union_any_source", "target": first["target"], "attack": first["attack"],
            "epsilon": first["epsilon"], "n": first["n"], "sources": [r["row"]["attack_source"] for r in results],
            "robust_accuracy": float(robust.mean()),
            "attack_success_rate": float((clean & ~robust).sum() / clean.sum()) if clean.sum() else None,
            "example_id_sha256": first["example_id_sha256"],
            "note": "per-example union: an example counts as robust only if no listed source broke it; separately labelled, not a single attack"}


# ------------------------------------------------------------------ E4 budget adequacy
def e4_subset(val_ids, val_y, cfg: dict, split_seed: int, *, plumbing_only: bool = False) -> dict:
    """Fixed validation attack subset for the budget check (protocol E4). Stops (raises) if it is too small."""
    e4 = cfg["e4"]
    val_ids, val_y = np.asarray(val_ids), np.asarray(val_y)
    n = len(val_y)
    if n <= e4["all_rows_threshold"]:
        local = np.arange(n)
        rule = f"all validation rows (n <= {e4['all_rows_threshold']})"
    else:
        local = stratified_cap(np.arange(n), val_y, e4["per_class_cap"], derived_seed(e4["seed_name"], split_seed))
        rule = f"data.stratified_cap with per_class_cap={e4['per_class_cap']}"
    if len(local) < e4["min_subset"] and not plumbing_only:
        raise SubsetTooSmallError(f"E4 subset has {len(local)} rows, fewer than {e4['min_subset']}: stop and ask the user "
                                  "(the cap and threshold are not changed automatically)")
    return {"ids": [int(i) for i in val_ids[local]], "local_index": local, "n_subset": int(len(local)), "rule": rule,
            "id_sha256": id_hash(val_ids[local]), "per_class_counts": np.bincount(val_y[local], minlength=int(val_y.max()) + 1).tolist(),
            "plumbing_only": bool(plumbing_only)}


def paired_n_extra(reference_robust_correct, candidate_robust_correct) -> int:
    """Examples the reference budget breaks (not robust) that the candidate budget does not."""
    ref, cand = np.asarray(reference_robust_correct, dtype=bool), np.asarray(candidate_robust_correct, dtype=bool)
    if ref.shape != cand.shape:
        raise ValueError("paired comparison needs identical examples")
    return int((~ref & cand).sum())


def budget_check(cells: dict, n_subset: int, cfg: dict, *, plumbing_only: bool = False) -> dict:
    """E4 decision over the 12 (model, epsilon) cells: n_extra <= ceil(0.01 * n_subset) in every cell.

    cells: {(model, epsilon): n_extra} with exactly the cells listed in attacks.json["e4"].
    In plumbing_only mode (M2A) the arithmetic runs but no adequacy decision is recorded.
    """
    e4 = cfg["e4"]
    keys = {(m, float(e)) for m in e4["models"] for e in e4["epsilons"]}
    given = {(m, float(e)) for m, e in cells}
    if given != keys:
        raise ValueError(f"E4 needs exactly the {len(keys)} cells {sorted(keys)}; got {sorted(given)}")
    if n_subset < e4["min_subset"] and not plumbing_only:
        raise SubsetTooSmallError(f"E4 subset has {n_subset} rows, fewer than {e4['min_subset']}: stop and ask the user")
    threshold = math.ceil(e4["max_extra_fraction"] * n_subset)
    rows = [{"model": m, "epsilon": float(e), "n_extra": int(v), "within_threshold": int(v) <= threshold}
            for (m, e), v in sorted(cells.items(), key=lambda kv: (kv[0][0], float(kv[0][1])))]
    adequate = all(r["within_threshold"] for r in rows)
    return {"n_subset": int(n_subset), "threshold": threshold, "cells": rows, "n_cells": len(rows),
            "rule": e4["rule"], "plumbing_only": bool(plumbing_only),
            "adequate": None if plumbing_only else adequate,
            "decision": None if plumbing_only else (e4["candidate_budget"] if adequate else e4["reference_budget"]),
            "note": ("plumbing check only; establishes nothing about adequacy" if plumbing_only
                     else "pgd20 adequate" if adequate else "pgd20 inadequate: use the reference budget for every model, source and epsilon")}
