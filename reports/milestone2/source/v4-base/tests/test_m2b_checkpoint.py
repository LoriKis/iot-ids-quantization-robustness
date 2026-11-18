"""Epoch-boundary checkpointing with a nn.Linear, an untouched Adam and a stub epoch (no backward, no optimizer step)."""
import copy
import random

import numpy as np
import pytest
import torch

from research import m2_training as mt
from research.m2_training import (CheckpointError, TrainingInterrupted, load_epoch_checkpoint, restore_rng,
                                  save_epoch_checkpoint, train_unit)

FP = {"source_train": "a" * 64, "config": "b" * 64, "data": "c" * 64, "split": "d" * 64}


def _model():
    return torch.nn.Linear(3, 2)


def _save(path, epoch, model, opt, fingerprints=FP, threads=2, keep_last=2):
    return save_epoch_checkpoint(path, epoch=epoch, model=model, opt=opt, updates=epoch * 4, epoch_losses=[0.5] * epoch,
                                 step_seconds=[0.1], attack_seconds=[], init_state_hash="h", seed=1, adversarial=False,
                                 threads=threads, fingerprints=fingerprints, keep_last=keep_last)


def test_roundtrip_restores_model_optimizer_and_rng(tmp_path):
    torch.manual_seed(5), np.random.seed(5), random.seed(5)
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _save(tmp_path, 2, model, opt)
    expected = (torch.rand(3), np.random.rand(3), random.random())
    torch.manual_seed(99), np.random.seed(99), random.seed(99)  # disturb every generator
    payload = load_epoch_checkpoint(tmp_path, threads=2, fingerprints=FP)
    restore_rng(payload["rng"])
    got = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(expected[0], got[0]) and np.array_equal(expected[1], got[1]) and expected[2] == got[2]
    other = _model()
    other.load_state_dict(payload["model_state"])
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
    opt2 = torch.optim.Adam(other.parameters(), lr=1e-3)
    opt2.load_state_dict(payload["optimizer_state"])
    assert opt2.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    assert payload["epoch_done"] == 2 and payload["updates"] == 8


def test_no_checkpoint_returns_none_and_torn_tmp_is_ignored(tmp_path):
    assert load_epoch_checkpoint(tmp_path / "missing", threads=2) is None
    (tmp_path / "epoch_001.pt.abc.tmp").write_bytes(b"torn")
    assert load_epoch_checkpoint(tmp_path, threads=2) is None
    assert not list(tmp_path.glob("*.tmp")) and (tmp_path / "old" / "epoch_001.pt.abc.tmp").exists()


def test_torn_tmp_next_to_valid_checkpoint(tmp_path):
    model = _model()
    _save(tmp_path, 1, model, torch.optim.Adam(model.parameters()))
    (tmp_path / "epoch_002.pt.zz.tmp").write_bytes(b"torn")
    assert load_epoch_checkpoint(tmp_path, threads=2)["epoch_done"] == 1


def test_sha_threads_and_fingerprint_mismatch_rejected(tmp_path):
    model = _model()
    path = _save(tmp_path, 1, model, torch.optim.Adam(model.parameters()))
    with pytest.raises(CheckpointError, match="threads"):
        load_epoch_checkpoint(tmp_path, threads=3, fingerprints=FP)
    with pytest.raises(CheckpointError, match="fingerprints"):
        load_epoch_checkpoint(tmp_path, threads=2, fingerprints=dict(FP, split="e" * 64))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_epoch_checkpoint(tmp_path, threads=2)


def test_keep_last_moves_older_epochs_to_old(tmp_path):
    model = _model()
    opt = torch.optim.Adam(model.parameters())
    for e in (1, 2, 3, 4):
        _save(tmp_path, e, model, opt, keep_last=2)
    assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == ["epoch_003.pt", "epoch_004.pt"]
    assert sorted(p.name for p in (tmp_path / "old").glob("epoch_*.pt")) == ["epoch_001.pt", "epoch_002.pt"]
    assert load_epoch_checkpoint(tmp_path, threads=2)["epoch_done"] == 4


# ---- train_unit resume with a stub epoch ---------------------------------------------------------------------------

def _cfg():
    cfg = mt.load_config()
    cfg = copy.deepcopy(cfg)
    cfg["resources"]["torch_threads"] = 1
    return cfg


def _stub_epoch(ep, env):
    """Draws from all three global generators and perturbs the weights; no backward, no optimizer step."""
    with torch.no_grad():
        for p in env["model"].parameters():
            p.add_(torch.rand_like(p) * 0.01 + float(np.random.rand()) * 0.001 + random.random() * 0.0001)
    return {"total_loss": float(torch.rand(1)) * 10, "updates": 4, "step_seconds": [0.0] * 4, "attack_seconds": [],
            "hook_counts": None, "batch_sizes": [8, 8, 8, 8]}


def _run(tmp_path, name, epochs, stop_flag=None):
    return train_unit(_cfg(), np.zeros((32, 39)), np.zeros(32, dtype=int), 7, None, epochs=epochs,
                      model_factory=_model, threads=1, checkpoint_dir=tmp_path / name, stop_flag=stop_flag,
                      epoch_fn=_stub_epoch, fingerprints=FP)


def test_stub_resume_two_plus_three_equals_straight_five(tmp_path):
    straight = _run(tmp_path, "straight", 5)
    rng_straight = (torch.rand(2), np.random.rand(2), random.random())

    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] == 2

    with pytest.raises(TrainingInterrupted):
        _run(tmp_path, "resumed", 5, stop_flag=stop_after_two)
    assert load_epoch_checkpoint(tmp_path / "resumed", threads=1)["epoch_done"] == 2
    torch.manual_seed(1234), np.random.seed(1234), random.seed(1234)  # a "new process" starts elsewhere
    resumed = _run(tmp_path, "resumed", 5)
    rng_resumed = (torch.rand(2), np.random.rand(2), random.random())

    assert resumed["state_hash"] == straight["state_hash"]
    assert resumed["log"]["optimizer_updates"] == straight["log"]["optimizer_updates"] == 20
    assert resumed["log"]["epoch_losses"] == straight["log"]["epoch_losses"]
    assert resumed["log"]["resumed_from_epoch"] == 2
    assert torch.equal(rng_straight[0], rng_resumed[0]) and np.array_equal(rng_straight[1], rng_resumed[1])
    assert rng_straight[2] == rng_resumed[2]


def test_resume_rejects_other_seed(tmp_path):
    _run(tmp_path, "u", 2)
    with pytest.raises(CheckpointError, match="seed"):
        train_unit(_cfg(), np.zeros((32, 39)), np.zeros(32, dtype=int), 8, None, epochs=3, model_factory=_model,
                   threads=1, checkpoint_dir=tmp_path / "u", epoch_fn=_stub_epoch, fingerprints=FP)


def test_completed_checkpoint_resumes_to_finalise_without_epochs(tmp_path):
    first = _run(tmp_path, "done", 3)
    again = _run(tmp_path, "done", 3)
    assert again["state_hash"] == first["state_hash"]
