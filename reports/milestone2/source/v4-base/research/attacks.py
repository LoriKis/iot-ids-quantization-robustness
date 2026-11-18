"""Untargeted L-infinity feature attacks with strongest-candidate retention."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def logits(model, x):
    out = model(x)["logits"]
    if not torch.isfinite(out).all():
        raise FloatingPointError("Non-finite model outputs")
    return out


def attack(model, x, y, *, kind, epsilon, steps=10, alpha=None, restarts=1,
           random_start=True, lower=None, upper=None, mutable=None, seed=0, selector=None):
    if kind not in ("fgsm", "pgd") or epsilon < 0 or steps < 1 or restarts < 1:
        raise ValueError("Invalid attack configuration")
    x = x.detach()
    if epsilon == 0:
        return x.clone()
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    # Include original points outside training support, so clipping itself
    # cannot violate epsilon=0 or the epsilon ball. This is not protocol validity.
    lo, hi = x - epsilon, x + epsilon
    if lower is not None:
        lo = torch.maximum(lo, torch.minimum(lower, x))
        hi = torch.minimum(hi, torch.maximum(upper, x))
    if mutable is not None:
        lo = torch.where(mutable, lo, x)
        hi = torch.where(mutable, hi, x)
    alpha = epsilon if kind == "fgsm" else (alpha if alpha is not None else epsilon / 4)
    nsteps = 1 if kind == "fgsm" else steps
    nr = 1 if kind == "fgsm" else restarts
    generator = torch.Generator(device=x.device).manual_seed(seed)
    # selector (optional): scores the clean baseline and every candidate; gradients always use model.
    def score(point):
        if selector is None:
            return logits(model, point)
        out = selector(point)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite model outputs")
        return out
    try:
        with torch.no_grad():
            original = score(x)
            best_loss = F.cross_entropy(original, y, reduction="none")
            best_wrong = original.argmax(1).ne(y)
        best = x.clone()

        def consider(candidate):
            nonlocal best_loss, best_wrong, best
            with torch.no_grad():
                out = score(candidate)
                loss = F.cross_entropy(out, y, reduction="none")
                wrong = out.argmax(1).ne(y)
                # Success takes priority; within each success state maximize CE.
                take = (wrong & ~best_wrong) | ((wrong == best_wrong) & (loss > best_loss))
                best[take] = candidate[take]
                best_loss[take] = loss[take]
                best_wrong[take] = wrong[take]

        # Include FGSM in every PGD candidate pool, preventing a weaker PGD
        # report merely because random initialization missed a known candidate.
        def gradient(point):
            with torch.enable_grad():
                point = point.detach().requires_grad_(True)
                loss = F.cross_entropy(logits(model, point), y)
                grad = torch.autograd.grad(loss, point, only_inputs=True)[0]
            if not torch.isfinite(grad).all():
                raise FloatingPointError("Non-finite input gradient")
            return grad

        initial_gradient = gradient(x)
        consider(torch.clamp(x + epsilon * initial_gradient.sign(), lo, hi))
        for restart in range(nr):
            if kind == "pgd" and random_start:
                point = torch.clamp(x + (torch.rand(x.shape, generator=generator, device=x.device) * 2 - 1) * epsilon, lo, hi)
            else:
                point = x.clone()
            consider(point)
            for _ in range(nsteps):
                g = gradient(point)
                point = torch.clamp(point + alpha * g.sign(), lo, hi).detach()
                consider(point)
        if not torch.isfinite(best).all() or (best - x).abs().max() > epsilon + 2e-6:
            raise AssertionError("Attack budget or finiteness violation")
        return best.detach()
    finally:
        # Preserve mixed module modes as well as the top-level state.
        for module, training in modes:
            module.training = training
