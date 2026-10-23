import numpy as np
import pytest
import torch

from research import adversarial_domain as ad
from research.paper_described_adam import DEFAULT_CONFIG, read_adam_config
from research.provenance import state_hash

EPS = 0.10


class Tiny(torch.nn.Module):
    """Linear model on flattened (n, 50, 39) input; returns the same dict interface as the real model."""

    def __init__(self, seed=0, dropout=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc = torch.nn.Linear(50 * 39, 27)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(self.fc.weight.shape, generator=g) * 0.05)
        self.drop = torch.nn.Dropout(dropout)
        self.modes = []

    def forward(self, x):
        self.modes.append(self.training)
        return {"logits": self.fc(self.drop(x.flatten(1)))}


def grid(value, n=3):
    return np.full((n, 50, 39), value, dtype=np.float32)


def bounds(x):
    e = np.float32(EPS)
    return np.maximum(x - e, np.float32(0)), np.minimum(x + e, np.float32(1))


Y = np.array([0, 1, 2], dtype=np.int64)


def test_in_range_inputs():
    x = np.random.default_rng(0).uniform(0.2, 0.8, (3, 50, 39)).astype(np.float32)
    m = Tiny().eval()
    for out in (ad.fgsm(m, x, Y, EPS), ad.pgd(m, x, Y, EPS, 0.025, 7, 1)):
        assert out.min() >= 0 and out.max() <= 1
        assert np.abs(out - x).max() <= EPS + 1e-6


def test_out_of_range_feasible_inputs_stay_in_intersection():
    x = grid(1.05)
    x[1] = np.float32(-0.05)
    x[2] = np.float32(0.5)
    lo, hi = bounds(x)
    m = Tiny()
    fg = ad.fgsm(m, x, Y, EPS)
    pg, traj = ad.pgd(m, x, Y, EPS, 0.025, 7, 1, return_trajectory=True)
    assert len(traj) == 8  # initialization + 7 iterates
    for arr in [fg, pg, *traj]:
        assert (arr >= lo).all() and (arr <= hi).all()


@pytest.mark.parametrize("value", [1.2, -0.2])
def test_infeasible_inputs_fail_clearly(value):
    x = grid(value)
    before = x.copy()
    m = Tiny()
    for call in (lambda: ad.fgsm(m, x, Y, EPS), lambda: ad.pgd(m, x, Y, EPS, 0.025, 7, 1)):
        with pytest.raises(ad.InfeasibleConstraintError, match="Infeasible") as info:
            call()
        assert info.value.count == x.size and len(info.value.indices) == 10
    assert np.array_equal(x, before)


def test_infeasible_boundary_matches_float32_formula():
    x = grid(1.10)
    lo, hi = bounds(x)
    feasible = not (lo > hi).any()
    try:
        ad.fgsm(Tiny(), x, Y, EPS)
        assert feasible
    except ad.InfeasibleConstraintError:
        assert not feasible


def test_epsilon_zero_identity_without_model_call():
    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("model must not be called for eps=0")

    x = grid(1.7)  # out of range: still returned unchanged
    for out in (ad.fgsm(Boom(), x, Y, 0.0), ad.pgd(Boom(), x, Y, 0.0, 0.025, 7, 1)):
        assert np.array_equal(out, x) and not np.shares_memory(out, x)


def test_perturbation_and_domain_bounds_on_unit_interval():
    x = np.random.default_rng(1).uniform(0, 1, (3, 50, 39)).astype(np.float32)
    lo, hi = bounds(x)
    m = Tiny(3)
    for out in (ad.fgsm(m, x, Y, EPS), ad.pgd(m, x, Y, EPS, 0.025, 7, 5)):
        assert (out >= lo).all() and (out <= hi).all()
        assert np.abs(out - x).max() <= EPS + 1e-6


def test_finite_nonzero_gradients_and_hand_computed_fgsm():
    m = Tiny(2).eval()
    x = torch.from_numpy(np.random.default_rng(2).uniform(0.2, 0.8, (3, 50, 39)).astype(np.float32))
    g = ad._input_grad(m, x, torch.from_numpy(Y))
    assert torch.isfinite(g).all() and (g.flatten(1).abs().sum(1) > 0).all()
    # hand computed: softmax-CE input gradient of a linear model is (p - onehot) @ W
    p = torch.softmax(m.fc(x.flatten(1)), 1)
    p[torch.arange(3), torch.from_numpy(Y)] -= 1
    expected = (p @ m.fc.weight).reshape(3, 50, 39)
    assert torch.allclose(g, expected, atol=1e-5)
    out = ad.fgsm(m, x.numpy(), Y, EPS)
    lo, hi = bounds(x.numpy())
    manual = np.minimum(np.maximum(x.numpy() + np.float32(EPS) * np.sign(expected.detach().numpy()), lo), hi)
    assert np.array_equal(out, manual)


def test_eval_mode_and_unchanged_weights():
    m = Tiny(4, dropout=0.5)
    m.train()
    before = state_hash(m)
    x = grid(0.5)
    ad.pgd(m, x, Y, EPS, 0.025, 7, 1)
    ad.fgsm(m, x, Y, EPS)
    assert not any(m.modes) and state_hash(m) == before
    assert all(p.grad is None and p.requires_grad for p in m.parameters())


def test_deterministic_replay_and_global_rng_untouched():
    m = Tiny(5)
    x = np.random.default_rng(3).uniform(0.2, 0.8, (5, 50, 39)).astype(np.float32)
    y = np.arange(5)
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    a = ad.pgd(m, x, y, EPS, 0.025, 7, 1, batch_size=2)
    assert torch.equal(torch.get_rng_state(), state)
    b = ad.pgd(m, x, y, EPS, 0.025, 7, 1, batch_size=2)
    c = ad.pgd(m, x, y, EPS, 0.025, 7, 2, batch_size=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_clean_evaluation_accepts_out_of_range_and_rejects_nan():
    x = grid(1.5)
    before = x.copy()
    logits = ad.compute_logits(Tiny(), x)
    assert np.isfinite(logits).all() and np.array_equal(x, before)
    bad = grid(0.5)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ad.compute_logits(Tiny(), bad)


def test_config_attack_parameters_match_function_defaults():
    cfg = read_adam_config(DEFAULT_CONFIG)
    a = cfg["attack"]
    assert (a["fgsm"]["epsilon"], a["pgd"]["epsilon"], a["pgd"]["iterations"], a["pgd"]["alpha"], a["pgd"]["random_start"]) == (0.10, 0.10, 7, 0.025, True)
    assert a["attack_batch_size"] == 64 and a["final_iterate_evaluation"] is True
    assert "maximum(x-eps,0)" in a["domain_policy"]["feasible_set_eps_gt_0"]
