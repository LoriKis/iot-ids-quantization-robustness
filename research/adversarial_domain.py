"""Adversarial attacks with an explicit, checked domain policy (pure functions, no I/O).

Policy
------
* Epsilon is an L-infinity budget around the ORIGINAL normalized input ``x``.
* For eps > 0 the feasible set is the intersection of the epsilon ball with [0, 1]:
      lower = maximum(x - eps, 0)      upper = minimum(x + eps, 1)
  (float32 arithmetic). The random start and EVERY attack update are projected
  onto [lower, upper].
* If any lower > upper (clean value farther than eps outside [0, 1]) the call raises
  ``InfeasibleConstraintError``. Epsilon is never enlarged, clean inputs are never
  modified and no example is dropped.
* For eps == 0 an exact unchanged copy is returned before any model call. No
  perturbation or domain correction is performed, even for out-of-range inputs.
* Clean evaluation accepts finite values outside [0, 1]; nothing is clamped.
* Perturbations live on normalized sequence tensors (batch, 50, 39); they carry no
  guarantee of protocol validity. PGD evaluates the FINAL iterate.
* PGD random-start generator seed: sha256("pgd-random-start|<training_seed>|<batch_index>");
  a local torch.Generator is used, the global RNG is never reseeded or consumed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F


class InfeasibleConstraintError(ValueError):
    """Raised when max(x-eps,0) > min(x+eps,1) for at least one element."""

    def __init__(self, count: int, indices: list, max_violation: float, eps: float):
        self.count = count
        self.indices = indices
        self.max_violation = max_violation
        self.eps = eps
        super().__init__(
            f"Infeasible adversarial constraint: {count} element(s) have lower > upper for eps={eps} "
            f"(max violation {max_violation:.8g}; first (row, step, feature) indices {indices}). "
            "Epsilon was not enlarged, inputs were not modified, no example was dropped."
        )


def batch_seed(training_seed: int, batch_index: int) -> int:
    digest = hashlib.sha256(f"pgd-random-start|{training_seed}|{batch_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def feasible_bounds(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """lower=max(x-eps,0), upper=min(x+eps,1) in float32; raises if any lower > upper."""
    e = torch.tensor(eps, dtype=torch.float32)
    lower = torch.clamp(x - e, min=0.0)
    upper = torch.clamp(x + e, max=1.0)
    bad = lower > upper
    if bool(bad.any()):
        idx = torch.nonzero(bad)[:10].tolist()
        raise InfeasibleConstraintError(int(bad.sum()), idx, float((lower - upper)[bad].max()), float(eps))
    return lower, upper


def project(x_adv: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(x_adv, lower), upper)


def assert_clean_inputs_finite(x) -> None:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("Empty input array")
    if not np.isfinite(arr).all():
        first = np.argwhere(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"Non-finite value at index {first}")


def _validate(x, y, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(eps) or eps < 0:
        raise ValueError(f"eps must be finite and >= 0, got {eps}")
    x = np.asarray(x)
    y = np.asarray(y)
    if x.dtype != np.float32 or x.ndim != 3:
        raise ValueError("x must be a float32 array of shape (n, steps, features)")
    assert_clean_inputs_finite(x)
    if len(x) != len(y):
        raise ValueError("x and y length differ")
    if y.min() < 0 or y.max() > 26:
        raise ValueError("labels must be in [0, 26]")
    return x, y


class _Frozen:
    """Eval mode + parameters frozen for the duration of an attack; requires_grad restored."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.saved = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self.model

    def __exit__(self, *exc):
        for p, flag in zip(self.model.parameters(), self.saved):
            p.requires_grad_(flag)


def _input_grad(model, x_in: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    x_in = x_in.clone().detach().requires_grad_(True)
    assert not model.training
    loss = F.cross_entropy(model(x_in)["logits"], yb, reduction="sum")
    return torch.autograd.grad(loss, x_in)[0]


def fgsm(model, x, y, eps: float, batch_size: int = 64) -> np.ndarray:
    x, y = _validate(x, y, eps)
    if eps == 0:
        return np.array(x, copy=True)
    xt_all = torch.from_numpy(x)
    lower_all, upper_all = feasible_bounds(xt_all, eps)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    out = []
    with _Frozen(model):
        for i in range(0, len(x), batch_size):
            xb, yb = xt_all[i:i + batch_size], yt[i:i + batch_size]
            grad = _input_grad(model, xb, yb)
            adv = project(xb + torch.tensor(eps, dtype=torch.float32) * torch.sign(grad),
                          lower_all[i:i + batch_size], upper_all[i:i + batch_size])
            out.append(adv.detach().numpy())
    return np.concatenate(out, axis=0)


def pgd(model, x, y, eps: float, alpha: float, steps: int, base_seed: int,
        batch_size: int = 64, return_trajectory: bool = False):
    x, y = _validate(x, y, eps)
    if alpha <= 0 or steps < 1:
        raise ValueError("alpha must be > 0 and steps >= 1")
    if eps == 0:
        result = np.array(x, copy=True)
        return (result, [result.copy()]) if return_trajectory else result
    xt_all = torch.from_numpy(x)
    lower_all, upper_all = feasible_bounds(xt_all, eps)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    a = torch.tensor(alpha, dtype=torch.float32)
    e = torch.tensor(eps, dtype=torch.float32)
    finals, trajectories = [], []
    with _Frozen(model):
        for b, i in enumerate(range(0, len(x), batch_size)):
            x0, yb = xt_all[i:i + batch_size], yt[i:i + batch_size]
            lower, upper = lower_all[i:i + batch_size], upper_all[i:i + batch_size]
            gen = torch.Generator(device="cpu")
            gen.manual_seed(batch_seed(base_seed, b))
            u = (torch.rand(x0.shape, generator=gen, dtype=torch.float32) * 2 - 1) * e
            adv = project(x0 + u, lower, upper)
            traj = [adv.numpy().copy()]
            for _ in range(steps):
                grad = _input_grad(model, adv, yb)
                adv = project(adv + a * torch.sign(grad), lower, upper).detach()
                traj.append(adv.numpy().copy())
            finals.append(adv.numpy())
            trajectories.append(traj)
    result = np.concatenate(finals, axis=0)
    if return_trajectory:
        return result, [np.concatenate([t[k] for t in trajectories], axis=0) for k in range(steps + 1)]
    return result


def compute_logits(model, x, batch_size: int = 64) -> np.ndarray:
    """Clean/adversarial logits. Finite out-of-[0,1] inputs pass through unchanged."""
    x = np.asarray(x, dtype=np.float32)
    assert_clean_inputs_finite(x)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[i:i + batch_size]))["logits"].cpu())
    return torch.cat(out, dim=0).numpy()
