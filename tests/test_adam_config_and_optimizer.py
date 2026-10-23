import copy
import json

import numpy as np
import pytest
import torch

from research import paper_described_adam as runner
from research import provenance as pv

CFG = runner.DEFAULT_CONFIG


def test_config_values_and_historical_config_untouched():
    cfg = runner.read_adam_config(CFG)
    t = cfg["training"]
    assert t["optimizer"] == "torch.optim.Adam" and t["decoupled_weight_decay"] is False
    assert t["optimizer_kwargs"] == {"lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999], "eps": 1e-08, "amsgrad": False, "maximize": False}
    assert (t["epochs"], t["batch_size"], cfg["seeds"], cfg["split"]["split_seed"]) == (25, 512, [1, 7, 13, 29], 2025)
    assert cfg["architecture"]["n_classes"] == 27
    old = json.loads((runner.ROOT / "configs/milestone1_paper_described.json").read_text(encoding="utf-8"))
    assert old["training"]["optimizer"] == "torch.optim.AdamW"  # frozen historical variant B stays as recorded


def test_reader_rejects_historical_adamw_and_modified_settings(tmp_path):
    with pytest.raises(ValueError):
        runner.read_adam_config(runner.ROOT / "configs/milestone1_paper_described.json")
    base = json.loads(CFG.read_text(encoding="utf-8"))
    for mutate in (lambda c: c["training"].update(optimizer="torch.optim.AdamW"),
                   lambda c: c["training"].update(decoupled_weight_decay=True),
                   lambda c: c["training"]["optimizer_kwargs"].update(lr=0.01),
                   lambda c: c["training"].update(epochs=26),
                   lambda c: c["split"].update(sha256="0" * 64),
                   lambda c: c.update(surprise=1)):
        cfg = copy.deepcopy(base)
        mutate(cfg)
        path = tmp_path / "c.json"
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            runner.read_adam_config(path)


def test_optimizer_is_plain_adam_with_l2_weight_decay():
    cfg = runner.read_adam_config(CFG)
    model = torch.nn.Linear(3, 2)
    opt = runner.build_optimizer(cfg, model)
    assert type(opt) is torch.optim.Adam and not isinstance(opt, torch.optim.AdamW)
    d = opt.defaults
    assert (d["lr"], d["weight_decay"], d["betas"], d["eps"], d["amsgrad"]) == (0.001, 0.0001, (0.9, 0.999), 1e-08, False)
    assert d.get("decoupled_weight_decay", False) is False


def test_adam_weight_decay_enters_the_gradient_unlike_adamw():
    w0 = torch.tensor([1.0, -2.0])
    outs = {}
    for cls in (torch.optim.Adam, torch.optim.AdamW):
        p = torch.nn.Parameter(w0.clone())
        opt = cls([p], lr=0.1, weight_decay=0.5)
        p.grad = torch.zeros_like(p)  # zero loss gradient: only the decay acts
        opt.step()
        outs[cls.__name__] = p.detach().clone()
    # Adam: decay is added to the gradient, so Adam's normalised step is ~lr * sign(w); AdamW shrinks multiplicatively.
    assert torch.allclose(outs["Adam"], w0 - 0.1 * torch.sign(w0), atol=1e-4)
    assert not torch.allclose(outs["Adam"], outs["AdamW"], atol=1e-3)


def test_batch_counts_42_with_final_batch_of_5():
    perm = torch.randperm(20997, generator=torch.Generator().manual_seed(1))
    batches = runner.epoch_batches(20997, 512, perm)
    sizes = [len(b) for b in batches]
    assert len(sizes) == 42 and sizes[:41] == [512] * 41 and sizes[-1] == 5
    assert sum(sizes) == 20997 and 25 * len(sizes) == 1050
    assert sorted(torch.cat(batches).tolist()) == list(range(20997))


def test_optimizer_step_counts_after_epochs_on_real_loop():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(50 * 39, 5)

        def forward(self, x):
            return {"logits": self.fc(x.flatten(1))}

    X = np.random.default_rng(0).uniform(size=(1029, 39)).astype(np.float32)  # 2*512 + 5
    y = np.zeros(1029, dtype=np.int64)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    seen = []
    runner.run_epochs(model, opt, X, y, 1, 0, 2, 512, 50, 0.01, lambda n, loss, sizes: seen.append(sizes))
    assert seen[0] == [512, 512, 5]
    assert runner.optimizer_steps(opt) == {6}


def test_fresh_initialization_is_independent_of_other_state():
    cfg = runner.read_adam_config(CFG)
    infra = pv.import_upstream_readonly(runner.ROOT)
    runner.seed_everything(1)
    h1 = pv.state_hash(infra.StandardTransformerIDS(**cfg["architecture"]))
    torch.manual_seed(999)  # simulate smoke-trained state elsewhere
    runner.seed_everything(1)
    h2 = pv.state_hash(infra.StandardTransformerIDS(**cfg["architecture"]))
    assert h1 == h2
    assert len(runner.build_optimizer(cfg, torch.nn.Linear(2, 2)).state_dict()["state"]) == 0
