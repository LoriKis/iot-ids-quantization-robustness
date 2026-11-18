"""Local compute guard (DESIGN 9.1): makes training, model-size construction and attacks fail in local test runs.

Used by tests/conftest.py (autouse, enabled by env M2_LOCAL_NO_TRAIN=1) and by tests/test_m2b_guard_local.py.
"""
from __future__ import annotations

import importlib.machinery
import inspect
import sys

import torch

MAX_LOCAL_LAYERS = 2     # n_layers >= 3 is refused
MAX_LOCAL_D_MODEL = 64   # d_model > 64 is refused


class LocalComputeRuleViolation(RuntimeError):
    """Raised by the local guard when a test tries to train, build a full-size model or run an attack."""


def _raiser(what):
    def _raise(*args, **kwargs):
        raise LocalComputeRuleViolation(f"{what} is not allowed on this machine (M2_LOCAL_NO_TRAIN=1); "
                                        "run it on the VM and mark the test vm_only")
    return _raise


def _wrap_model_init(cls):
    original = cls.__init__
    if getattr(original, "_m2_local_guard", False):
        return
    signature = inspect.signature(original)

    def guarded(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        n_layers, d_model = bound.arguments.get("n_layers", 0), bound.arguments.get("d_model", 0)
        if n_layers > MAX_LOCAL_LAYERS or d_model > MAX_LOCAL_D_MODEL:
            raise LocalComputeRuleViolation(
                f"StandardTransformerIDS(n_layers={n_layers}, d_model={d_model}) is too large to build locally")
        return original(self, *args, **kwargs)

    guarded._m2_local_guard = True
    cls.__init__ = guarded


def install_local_guard(monkeypatch):
    """Apply the three patches with a pytest monkeypatch (undone automatically)."""
    # (a) every optimizer step. Concrete optimizers override Optimizer.step, so patch each subclass too.
    optimizers = [torch.optim.Optimizer] + [
        obj for obj in vars(torch.optim).values()
        if inspect.isclass(obj) and issubclass(obj, torch.optim.Optimizer) and obj is not torch.optim.Optimizer]
    for cls in optimizers:
        if "step" in vars(cls):
            monkeypatch.setattr(cls, "step", _raiser(f"{cls.__name__}.step"))

    # (b) StandardTransformerIDS.__init__: the class is re-loaded by research.common.make_model from a file on every call, so
    # wrap SourceFileLoader.exec_module and guard the class right after each load of the PAVE module.
    original_exec = importlib.machinery.SourceFileLoader.exec_module

    def exec_module(self, module):
        original_exec(self, module)
        cls = getattr(module, "StandardTransformerIDS", None)
        if cls is not None:
            _wrap_model_init(cls)

    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "exec_module", exec_module)
    for module in list(sys.modules.values()):
        cls = getattr(module, "StandardTransformerIDS", None) if module is not None else None
        if inspect.isclass(cls) and cls.__module__ and "standard_transformer" in (getattr(module, "__file__", "") or ""):
            monkeypatch.setattr(cls, "__init__", cls.__init__)  # restore the original at teardown
            _wrap_model_init(cls)

    # (c) evaluate_unit (FGSM/PGD evaluation), also where other research modules imported it by name.
    from research import m2_attacks
    original_eval = m2_attacks.evaluate_unit
    blocked = _raiser("research.m2_attacks.evaluate_unit")
    monkeypatch.setattr(m2_attacks, "evaluate_unit", blocked)
    for name, module in list(sys.modules.items()):
        if name.startswith("research.") and module is not m2_attacks and getattr(module, "evaluate_unit", None) is original_eval:
            monkeypatch.setattr(module, "evaluate_unit", blocked)
