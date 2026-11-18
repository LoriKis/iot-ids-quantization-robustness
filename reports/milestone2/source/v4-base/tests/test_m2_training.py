"""Milestone 2A training tests: Experiment C equivalence, V1/V3 pairing, AT safeguards, compute counters."""
import copy
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research import m2_training
from research.common import ROOT, make_model, sha256
from research.m2_attacks import RowTiedModel, checked_attack
from research.m2_split import ConfigError
from research.m2_training import TrainingDivergedError, attack_pass_counts, load_config, train_unit, weighted_loss
from research.paper_described_adam import (build_optimizer, derived_seed, epoch_batches, read_adam_config, run_epochs,
                                           seed_everything)
from research.provenance import state_hash

TINY = {"input_dim": 39, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "n_classes": 27,
        "max_seq_len": 50, "dropout": 0.1}
CONFIG_C = ROOT / "configs/milestone1_paper_described_adam.json"


@pytest.fixture(scope="module")
def real_cfg():
    return load_config()


@pytest.fixture
def cfg(real_cfg):
    c = copy.deepcopy(real_cfg)
    c["architecture"] = dict(TINY)
    c["training"]["batch_size"] = 16
    c["training"]["epochs"] = 2
    return c


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return rng.random((40, 39)).astype(np.float32), (np.arange(40) % 27).astype(np.int64)


def test_config_equals_experiment_c(real_cfg):
    c = read_adam_config(CONFIG_C)
    assert sha256(CONFIG_C) == real_cfg["source_experiment"]["sha256"]
    assert real_cfg["architecture"] == c["architecture"]
    assert real_cfg["sequence"]["length"] == c["sequence"]["length"] == 50
    assert real_cfg["sequence"]["train_noise_std"] == c["sequence"]["train_noise_std"] == 0.01
    assert real_cfg["seeds"] == c["seeds"] == [1, 7, 13, 29]
    t, tc = real_cfg["training"], c["training"]
    for key in ("precision", "optimizer", "optimizer_kwargs", "decoupled_weight_decay", "epochs", "batch_size",
                "drop_last", "loss", "gradient_clipping", "schedule"):
        assert t[key] == tc[key], key
    assert real_cfg["resources"]["torch_threads"] == c["resources"]["torch_threads"] == 2
    assert real_cfg["upstream_revision"] == c["upstream_revision"]


def test_adam_not_adamw(real_cfg):
    model = make_model(real_cfg["architecture"])
    opt = build_optimizer(real_cfg, model)
    assert type(opt) is torch.optim.Adam and not isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 0.001 and opt.defaults["weight_decay"] == 0.0001
    assert sum(p.numel() for p in model.parameters()) == 351612
    bad = copy.deepcopy(real_cfg)
    bad["training"]["optimizer"] = "torch.optim.AdamW"
    with pytest.raises(ConfigError):
        _load_from_dict(bad)


def _load_from_dict(cfg):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "training.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return load_config(p)


def test_config_validation_rejects_unknown_key_and_bad_weights(real_cfg):
    extra = copy.deepcopy(real_cfg)
    extra["surprise"] = 1
    with pytest.raises(ConfigError):
        _load_from_dict(extra)
    bad = copy.deepcopy(real_cfg)
    bad["adversarial"]["adversarial_weight"] = 0.7
    with pytest.raises(ConfigError):
        _load_from_dict(bad)


def test_step_counts_for_new_train_size():
    batches = epoch_batches(21925, 512, list(range(21925)))
    assert len(batches) == 43 and len(batches[-1]) == 421
    assert 25 * len(batches) == 1075


def test_v1_matches_run_epochs(cfg, data):
    X, y = data
    seed = 1
    previous = torch.get_num_threads()
    torch.set_num_threads(1)  # same thread count as train_unit(threads=1); summation order depends on it
    try:
        seed_everything(seed)
        model = make_model(cfg["architecture"])
        opt = build_optimizer(cfg, model)
        run_epochs(model, opt, X, y, seed, 0, 2, 16, 50, 0.01)
    finally:
        torch.set_num_threads(previous)
    unit = train_unit(cfg, X, y, seed, adversarial=False, threads=1)
    assert unit["state_hash"] == state_hash(model)
    assert unit["log"]["optimizer_updates"] == 6


def test_shared_init_and_shuffle(cfg, data, monkeypatch):
    X, y = data
    seen = {}

    def record(n_rows, batch_size, perm):
        batches = epoch_batches(n_rows, batch_size, perm)
        seen.setdefault(flag[0], []).append([b.tolist() for b in batches])
        return batches

    monkeypatch.setattr(m2_training, "epoch_batches", record)
    flag = ["v1"]
    v1 = train_unit(cfg, X, y, 1, adversarial=False, epochs=2, threads=1)
    flag[0] = "v3"
    v3 = train_unit(cfg, X, y, 1, adversarial=True, epochs=2, threads=1)
    assert v1["init_state_hash"] == v3["init_state_hash"]
    assert seen["v1"] == seen["v3"]
    assert v1["state_hash"] != v3["state_hash"]  # AT really changes the training
    assert v1["log"]["optimizer_updates"] == v3["log"]["optimizer_updates"] == 6
    assert v1["log"]["steps_per_epoch"] == v3["log"]["steps_per_epoch"] == 3


def test_interleaved_attack_leaves_ordinary_weights_unchanged(cfg, data):
    X, y = data
    plain = train_unit(cfg, X, y, 1, adversarial=False, epochs=1, threads=1)

    state = {"calls": 0, "busy": False}

    def factory():
        model = make_model(cfg["architecture"])

        def interleave(module, args):
            if module.training and not state["busy"]:
                state["calls"] += 1
                if state["calls"] > 1:  # the first step is left alone so its hook count stays exact
                    state["busy"] = True
                    try:
                        rows = args[0][:, 0, :].detach()
                        checked_attack(RowTiedModel(module), rows, torch.zeros(len(rows), dtype=torch.long), kind="pgd",
                                       epsilon=0.05, steps=3, alpha=0.025, seed=derived_seed("atk-train", 1, 0, state["calls"]))
                    finally:
                        state["busy"] = False

        model.register_forward_pre_hook(interleave)
        return model

    interleaved = train_unit(cfg, X, y, 1, adversarial=False, epochs=1, model_factory=factory, threads=1)
    assert state["calls"] == 3
    assert interleaved["state_hash"] == plain["state_hash"]


def test_attack_does_not_touch_params_grads_or_modes(cfg, data):
    X, y = data
    seed_everything(2)
    model = make_model(cfg["architecture"]).train()
    F.cross_entropy(model(torch.from_numpy(X[:8, None, :]).expand(-1, 50, -1).contiguous())["logits"], torch.from_numpy(y[:8])).backward()
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    before, modes = state_hash(model), [m.training for m in model.modules()]
    checked_attack(RowTiedModel(model), torch.from_numpy(X[:8]), torch.from_numpy(y[:8]), kind="pgd", epsilon=0.05, steps=5,
                   alpha=0.025, restarts=1, random_start=True, seed=11)
    assert state_hash(model) == before
    assert [m.training for m in model.modules()] == modes and model.training
    assert all((p.grad is None) if g is None else torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))


def test_loss_weights_from_known_logits():
    logits_clean = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]])
    logits_adv = torch.tensor([[0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    y = torch.tensor([0, 1])

    def ce(lg):
        p = np.exp(lg.numpy()) / np.exp(lg.numpy()).sum(1, keepdims=True)
        return float(-np.log(p[np.arange(2), y.numpy()]).mean())

    got = float(weighted_loss(logits_clean, logits_adv, y, 0.5, 0.5))
    assert math.isclose(got, 0.5 * ce(logits_clean) + 0.5 * ce(logits_adv), rel_tol=1e-6)


def test_logged_counters_match_forward_hooks(cfg, data):
    X, y = data
    assert attack_pass_counts(cfg, True) == {"attack_grad_passes_per_step": 6, "attack_scoring_forwards_per_step": 8,
                                             "train_forward_backward_per_step": 2}
    assert attack_pass_counts(cfg, False) == {"attack_grad_passes_per_step": 0, "attack_scoring_forwards_per_step": 0,
                                              "train_forward_backward_per_step": 1}
    for adversarial, expected_total in ((True, 16), (False, 1)):
        total = {"n": 0}

        def factory():
            model = make_model(cfg["architecture"])
            model.register_forward_pre_hook(lambda module, args: total.__setitem__("n", total["n"] + 1))
            return model

        unit = train_unit(cfg, X, y, 1, adversarial=adversarial, epochs=1, model_factory=factory, threads=1)
        hooks = unit["log"]["first_step_hook_counts"]
        assert total["n"] == expected_total * unit["log"]["optimizer_updates"]  # every step, independent count
        if adversarial:
            assert (hooks["attack_side_forwards"], hooks["train_side_forwards"]) == (14, 2)
            assert unit["log"]["pass_counters_per_step"] == {"attack_grad_passes_per_step": 6,
                                                             "attack_scoring_forwards_per_step": 8,
                                                             "train_forward_backward_per_step": 2}
            assert len(unit["log"]["attack_seconds"]) == unit["log"]["optimizer_updates"]
        else:
            assert (hooks["attack_side_forwards"], hooks["train_side_forwards"]) == (0, 1)
            assert unit["log"]["attack_seconds"] == []
        assert hooks["matches_logged_counters"]


def test_divergence_raises(cfg, data):
    X, y = data
    bad = X.copy()
    bad[0] = np.nan
    with pytest.raises(TrainingDivergedError):
        train_unit(cfg, bad, y, 1, adversarial=False, epochs=1, threads=1)


def test_outputs_written_once_and_threads_restored(cfg, data, tmp_path):
    X, y = data
    before = torch.get_num_threads()
    unit = train_unit(cfg, X, y, 1, tmp_path / "unit", adversarial=True, epochs=1, threads=1)
    assert torch.get_num_threads() == before
    log = json.loads((tmp_path / "unit" / "train_log.json").read_text(encoding="utf-8"))
    assert log["final_state_hash"] == unit["state_hash"] and log["adversarial"] is True
    assert log["checkpoint_sha256"] == sha256(tmp_path / "unit" / "checkpoint.pt")
    ck = torch.load(tmp_path / "unit" / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert ck["epoch"] == 1 and sorted(p.name for p in (tmp_path / "unit").iterdir()) == ["checkpoint.pt", "train_log.json"]
    with pytest.raises(FileExistsError):
        train_unit(cfg, X, y, 1, tmp_path / "unit", adversarial=True, epochs=1, threads=1)
