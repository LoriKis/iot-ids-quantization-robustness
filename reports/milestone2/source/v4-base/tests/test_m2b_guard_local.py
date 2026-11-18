"""The local compute guard raises for each blocked operation (no model is trained)."""
import pytest
import torch

from m2_guard_support import LocalComputeRuleViolation, install_local_guard
from research import common, m2_attacks


@pytest.fixture
def guard(monkeypatch):
    install_local_guard(monkeypatch)


def test_optimizer_step_blocked(guard):
    param = torch.nn.Parameter(torch.zeros(2))
    for opt in (torch.optim.Adam([param], lr=1e-3), torch.optim.SGD([param], lr=1e-3)):
        with pytest.raises(LocalComputeRuleViolation):
            opt.step()


def test_full_size_model_blocked_tiny_model_allowed(guard):
    tiny = dict(input_dim=4, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_classes=3, max_seq_len=5, dropout=0.0)
    assert common.make_model(tiny) is not None
    for bad in (dict(tiny, n_layers=3), dict(tiny, d_model=96, n_heads=4)):
        with pytest.raises(LocalComputeRuleViolation):
            common.make_model(bad)


def test_evaluate_unit_blocked(guard):
    with pytest.raises(LocalComputeRuleViolation):
        m2_attacks.evaluate_unit()


def test_guard_is_undone_when_its_monkeypatch_context_ends():
    before = (torch.optim.Adam.step, m2_attacks.evaluate_unit)
    with pytest.MonkeyPatch.context() as mp:
        install_local_guard(mp)
        assert torch.optim.Adam.step is not before[0]
    assert (torch.optim.Adam.step, m2_attacks.evaluate_unit) == before
