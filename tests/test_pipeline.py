import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from research.attacks import attack
from research.common import ROOT, atomic_json, atomic_torch, digest, make_model, seed_all, unit_status, verify_upstream
from research.data import build_data, partitions, sequences
from research.evaluation import metrics
from research.quantization import bpda_model, inventory, load_int8, prepare_static, ptq, selected_names
from research.runner import validate_config
from research.training import load_checkpoint
from research.validate import finite, validate_run


@pytest.fixture(scope="session", autouse=True)
def controlled_threads():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def cfg():
    return json.loads((ROOT / "configs/smoke.json").read_text())


@pytest.fixture(scope="session")
def real_data(tmp_path_factory, cfg):
    verify_upstream()
    return build_data(cfg, tmp_path_factory.mktemp("data"))


@pytest.fixture
def small_model():
    seed_all(5)
    return make_model({"input_dim": 39, "d_model": 16, "n_heads": 4, "n_layers": 1,
                       "d_ff": 32, "n_classes": 27, "max_seq_len": 50, "dropout": 0.1}).eval()


def test_real_schema_split_and_sequences(real_data, cfg):
    d = real_data
    m = d["manifest"]
    assert m["rows"] == 29997
    assert len(m["features"]) == 39 and len(d["labels"]) == 27
    assert set(m["label_counts"].values()) == {1111}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(m["full_split_row_ids"][left]).isdisjoint(m["full_split_row_ids"][right])
        assert set(m["selected_group_hashes"][left]).isdisjoint(m["selected_group_hashes"][right])
    for partition in ("train", "validation", "test"):
        seq = sequences(d["x"][partition], 50)
        assert seq.shape == (len(d["y"][partition]), 50, 39)
        torch.testing.assert_close(seq[:, 0], d["x"][partition], rtol=0, atol=0)
        torch.testing.assert_close(seq[:, -1], d["x"][partition], rtol=0, atol=0)
    assert d["preprocessing"]["fit_row_ids"] == m["selected_row_ids"]["train"]
    # Mean zero only for fitted training population; no evaluation fit.
    np.testing.assert_allclose(d["x"]["train"].numpy().mean(0), 0, atol=2e-6)


def test_preprocessing_matches_training_rows(real_data):
    import pandas as pd
    from research.common import UPSTREAM
    df = pd.read_csv(UPSTREAM / "data/balanced_CICIoT.csv")
    x = df.drop(columns="Label").to_numpy()
    ids = real_data["ids"]["train"]
    np.testing.assert_allclose(real_data["preprocessing"]["mean"], x[ids].mean(0), rtol=1e-12)
    # This checks values against the actual rows rather than a metadata label.
    assert not np.allclose(x[ids].mean(0), x.mean(0), rtol=1e-5)


def test_duplicate_groups_never_cross():
    groups = np.repeat(np.arange(100), 2)
    y = np.repeat(np.arange(100) % 5, 2)
    p = partitions(y, groups, 2026)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(groups[p[a]]).isdisjoint(groups[p[b]])


def test_checkpoint_roundtrip_and_compatibility(tmp_path, small_model):
    config = {"input_dim": 39, "d_model": 16, "n_heads": 4, "n_layers": 1,
              "d_ff": 32, "n_classes": 27, "max_seq_len": 50, "dropout": 0.1}
    path = tmp_path / "checkpoint.pt"
    atomic_torch(path, {"format_version": 1, "model_config": config,
                        "compatibility": {"split": "a"}, "state_dict": small_model.state_dict()})
    loaded, _ = load_checkpoint(path, {"split": "a"})
    x = torch.randn(3, 50, 39)
    torch.testing.assert_close(small_model(x)["logits"], loaded(x)["logits"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(path, {"split": "b"})


@pytest.mark.parametrize("kind", ["fgsm", "pgd"])
def test_attack_budget_zero_modes_weights_and_gradients(small_model, kind):
    model = small_model.train()
    model.blocks[0].attn.dropout.eval()  # Mixed modes must be restored.
    modes = [m.training for m in model.modules()]
    original = copy.deepcopy(model.state_dict())
    x = torch.randn(3, 50, 39)
    y = torch.tensor([0, 1, 2])
    zero = attack(model, x, y, kind=kind, epsilon=0)
    torch.testing.assert_close(x, zero, atol=0, rtol=0)
    adv = attack(model, x, y, kind=kind, epsilon=0.1, steps=3, restarts=2,
                 lower=torch.zeros(39), upper=torch.ones(39), seed=7)
    assert torch.isfinite(adv).all()
    assert (adv - x).abs().max() <= 0.100002
    assert modes == [m.training for m in model.modules()]
    assert all(p.grad is None for p in model.parameters())
    for name, weight in original.items():
        torch.testing.assert_close(weight, model.state_dict()[name], atol=0, rtol=0)


def test_pgd_retains_fgsm_and_more_restarts(small_model):
    x = torch.randn(4, 8, 39)
    y = small_model(x)["logits"].argmax(1)
    a = attack(small_model, x, y, kind="fgsm", epsilon=0.1)
    b = attack(small_model, x, y, kind="pgd", epsilon=0.1, steps=3, restarts=2)
    ac = small_model(a)["logits"].argmax(1).eq(y)
    bc = small_model(b)["logits"].argmax(1).eq(y)
    assert not torch.any(~ac & bc)
    c = attack(small_model, x, y, kind="pgd", epsilon=0.1, steps=3, restarts=3)
    cc = small_model(c)["logits"].argmax(1).eq(y)
    assert not torch.any(~bc & cc)


@pytest.mark.parametrize("scope,count", [("classifier", 2), ("ffn", 2), ("attention", 4), ("all_linear", 11)])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_actual_quantization_and_bpda(small_model, scope, count, mode):
    x = torch.randn(6, 10, 39)
    q, inv = ptq(small_model, scope, x, mode)
    assert len(inv["converted"]) == count
    assert set(inv["requested"]) == set(selected_names(small_model, scope))
    bpda = bpda_model(q, inv["requested"], x)
    torch.testing.assert_close(q(x)["logits"], bpda(x)["logits"], atol=0, rtol=0)
    xi = x.clone().requires_grad_()
    g = torch.autograd.grad(bpda(xi)["logits"].square().sum(), xi)[0]
    assert torch.isfinite(g).all() and g.abs().max() > 0
    y = q(x)["logits"].argmax(1)
    adv = attack(bpda, x, y, kind="pgd", epsilon=0.1, steps=2)
    assert torch.isfinite(q(adv)["logits"]).all()


def test_qat_real_conversion(small_model):
    from torch.ao import quantization as aq
    x = torch.randn(5, 8, 39)
    qat, names = prepare_static(small_model, "all_linear", x[:2], qat=True)
    opt = torch.optim.Adam(qat.parameters(), 0.001)
    out = qat(x)["logits"]
    opt.zero_grad()
    out.square().mean().backward()
    opt.step()
    qat.eval()
    qat.apply(aq.disable_observer)
    aq.convert(qat, inplace=True)
    assert len(inventory(qat, names, "qat_static")["converted"]) == 11
    assert torch.isfinite(qat(x)["logits"]).all()


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_int8_checkpoint_roundtrip(tmp_path, small_model, mode):
    x = torch.randn(5, 8, 39)
    q, _ = ptq(small_model, "attention", x, mode)
    path = tmp_path / "q.pt"
    atomic_torch(path, {"state_dict": q.state_dict(), "compatibility": {"data": "a"}})
    restored, _ = load_int8(small_model, "attention", x, path, {"data": "a"}, mode)
    torch.testing.assert_close(q(x)["logits"], restored(x)["logits"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        load_int8(small_model, "attention", x, path, {"data": "b"}, mode)


def test_upstream_swallowed_failure_is_detected(tmp_path):
    import pandas as pd
    from research.reproduce import check_results
    file = tmp_path / "rev_proj.csv"
    # A zero-exit upstream run can still omit AGT or an entire seed.
    pd.DataFrame([{"seed": 1, "model": "Standard Transformer", "dataset": "CICIoT"}]).to_csv(file, index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        check_results(file, [1])


def test_actual_standard_attention_inventory():
    m = make_model({"input_dim": 39, "d_model": 96, "n_heads": 4, "n_layers": 3,
                    "d_ff": 384, "n_classes": 27, "max_seq_len": 50, "dropout": 0.1})
    names = selected_names(m, "attention")
    assert len(names) == 12
    for layer in range(3):
        assert all(f"blocks.{layer}.attn.W_{projection}" in names for projection in ("q", "k", "v", "o"))
    assert len(selected_names(m, "all_linear")) == 23


def test_dataset_and_seed_selection_fail_closed(cfg):
    validate_config(cfg)
    for key, value in (("dataset", "CICIoMT"), ("model", "AGT"), ("sequence_length", 1)):
        changed = {**cfg, key: value}
        with pytest.raises(ValueError):
            validate_config(changed)
    with pytest.raises(ValueError, match="four"):
        validate_config({**cfg, "tier": "full"})


def test_unit_failure_resume_and_collision(tmp_path):
    folder = tmp_path / "unit"
    with pytest.raises(RuntimeError, match="deliberate"):
        with unit_status(folder, "id"):
            raise RuntimeError("deliberate")
    assert json.loads((folder / "status.json").read_text())["status"] == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        with unit_status(folder, "id", resume=True):
            pass
    with pytest.raises(RuntimeError, match="Incompatible"):
        with unit_status(folder, "wrong", resume=True):
            pass
    complete = tmp_path / "complete"
    with unit_status(complete, "id") as execute:
        assert execute
    with unit_status(complete, "id", resume=True) as execute:
        assert not execute
    with pytest.raises(FileExistsError):
        with unit_status(complete, "id"):
            pass


def test_incomplete_run_and_finite_metrics(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_run(tmp_path, "both")
    with pytest.raises(ValueError, match="Non-finite"):
        finite({"metric": float("nan")})
    result = metrics([0, 1], [1, 1], [1, 0], ["a", "b"])
    assert result["asr_denominator"] == 0 and result["attack_success_rate"] is None
