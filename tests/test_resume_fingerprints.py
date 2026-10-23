import builtins
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from research import paper_described_adam as runner
from research import provenance as pv


def fake_fp(**overrides):
    comps = {n: {"sha256": hashlib.sha256(n.encode()).hexdigest()} for n in ("config", "executed_source", "upstream_source", "data", "environment")}
    for k, v in overrides.items():
        comps[k] = {"sha256": v}
    return {"components": comps, "combined_sha256": pv.canonical_hash({k: v["sha256"] for k, v in sorted(comps.items())})}


def tree_hash(root: Path):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_absent_or_empty_run_dir_is_not_refused(tmp_path):
    pv.check_resume(tmp_path / "absent", fake_fp())
    (tmp_path / "empty").mkdir()
    pv.check_resume(tmp_path / "empty", fake_fp())


def test_matching_fingerprint_passes_and_each_component_mismatch_is_rejected(tmp_path):
    fp = fake_fp()
    (tmp_path / "fingerprint.json").write_text(json.dumps(fp))
    pv.check_resume(tmp_path, fp)
    for comp in fp["components"]:
        with pytest.raises(pv.ResumeIncompatibleError, match=comp):
            pv.check_resume(tmp_path, fake_fp(**{comp: "0" * 64}))


def test_content_without_fingerprint_and_unparsable_fingerprint_are_rejected(tmp_path):
    (tmp_path / "seed-1").mkdir()
    with pytest.raises(pv.ResumeIncompatibleError, match="no fingerprint"):
        pv.check_resume(tmp_path, fake_fp())
    (tmp_path / "fingerprint.json").write_text("{not json")
    with pytest.raises(pv.ResumeIncompatibleError, match="unreadable"):
        pv.check_resume(tmp_path, fake_fp())
    (tmp_path / "fingerprint.json").write_text(json.dumps({"components": {}}))
    with pytest.raises(pv.ResumeIncompatibleError):
        pv.check_resume(tmp_path, fake_fp())


def test_checkpoint_fingerprint_mismatch_is_rejected():
    fp = fake_fp()
    pv.check_checkpoint_fingerprint({"fingerprint_sha256": fp["combined_sha256"]}, fp)
    with pytest.raises(pv.ResumeIncompatibleError):
        pv.check_checkpoint_fingerprint({"fingerprint_sha256": "x"}, fp)


class WriteSpy:
    def __init__(self, monkeypatch):
        self.calls = []
        real_open = builtins.open

        def spy_open(file, mode="r", *a, **k):
            if any(c in str(mode) for c in "wax+"):
                self.calls.append(("open", str(file), mode))
                raise AssertionError(f"write attempted: open({file}, {mode})")
            return real_open(file, mode, *a, **k)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(io, "open", spy_open)
        for mod, names in ((os, ["replace", "rename", "makedirs", "mkdir"]), (shutil, ["copyfile", "copy2", "copytree"]),
                           (tempfile, ["mkstemp"]), (Path, ["write_bytes", "write_text", "mkdir", "touch"])):
            for name in names:
                monkeypatch.setattr(mod, name, self.make(name))

    def make(self, name):
        def spy(*a, **k):
            self.calls.append((name, a))
            raise AssertionError(f"write attempted: {name}")
        return spy


@pytest.fixture
def gate_env(monkeypatch):
    monkeypatch.setattr(pv, "compute_fingerprint", lambda cfg, root, with_scaler=True: fake_fp())
    monkeypatch.setattr(runner, "build_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("data prep reached")))


@pytest.mark.parametrize("case", ["mismatch", "missing_fingerprint"])
def test_rejected_launch_performs_zero_writes_and_exits_3(tmp_path, monkeypatch, gate_env, case):
    run = tmp_path / "run"
    (run / "seed-1" / "epochs").mkdir(parents=True)
    (run / "seed-1" / "epochs" / "epoch-01.pt").write_bytes(b"checkpoint")
    if case == "mismatch":
        (run / "fingerprint.json").write_text(json.dumps(fake_fp(config="1" * 64)))
    before = tree_hash(run)
    spy = WriteSpy(monkeypatch)
    for extra in ([], ["--resume"], ["--verify-only"]):
        code = runner.main(["--run-dir", str(run), *extra])
        assert code == 3
    smoke_code = runner.main(["--smoke", "--run-dir", str(run)])
    assert smoke_code == 3
    monkeypatch.undo()
    assert spy.calls == []
    assert tree_hash(run) == before
    assert not (run / "run.lock").exists() and not (run / "status.json").exists()


def test_fresh_start_over_incomplete_seed_without_resume_exits_3(tmp_path, monkeypatch, gate_env):
    run = tmp_path / "run"
    (run / "seed-1").mkdir(parents=True)
    (run / "seed-1" / "raw.log").write_text("x")
    (run / "fingerprint.json").write_text(json.dumps(fake_fp()))
    before = tree_hash(run)
    assert runner.main(["--run-dir", str(run), "--seeds", "1"]) == 3
    assert tree_hash(run) == before


def test_compatible_fingerprint_passes_the_gate_then_writes_begin(tmp_path, monkeypatch, gate_env):
    run = tmp_path / "run"
    monkeypatch.setattr(runner.pv, "environment_record", lambda *a: (_ for _ in ()).throw(ValueError("stop after gate")))
    assert runner.main(["--run-dir", str(run)]) == 4  # not 3: gate passed, first write step reached
    assert not (run / "run.lock").exists()  # lock released


def test_full_run_refuses_smoke_directory(tmp_path):
    assert runner.main(["--run-dir", str(runner.SMOKE_RUN / "inner")]) == 4


def test_compute_fingerprint_is_read_only_deterministic_and_sensitive(monkeypatch):
    cfg = runner.read_adam_config(runner.DEFAULT_CONFIG)
    first = pv.compute_fingerprint(cfg, runner.ROOT)
    spy = WriteSpy(monkeypatch)
    second = pv.compute_fingerprint(cfg, runner.ROOT)
    monkeypatch.undo()
    assert spy.calls == [] and first == second
    assert set(first["components"]["executed_source"]["files"]) == set(pv.EXECUTED_SOURCE_FILES)
    changed = copy.deepcopy(cfg)
    changed["training"]["optimizer_kwargs"]["lr"] = 0.002
    assert pv.compute_fingerprint(changed, runner.ROOT, with_scaler=False)["components"]["config"] != first["components"]["config"]


class DropNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(50 * 39, 5)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        return {"logits": self.fc(self.drop(x.flatten(1)))}


def make_training(tmp_seed=3):
    g = torch.Generator().manual_seed(7)
    X = torch.rand(100, 39, generator=g).numpy()
    y = (torch.rand(100, generator=g) * 5).long().numpy()
    runner.seed_everything(tmp_seed)
    model = DropNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    return model, opt, X, y


def train(model, opt, X, y, start, end):
    runner.run_epochs(model, opt, X, y, 3, start, end, 32, 50, 0.01)


def test_resume_equivalence_with_negative_control():
    model, opt, X, y = make_training()
    train(model, opt, X, y, 0, 3)
    reference = pv.state_hash(model)

    def resumed(restore_rng):
        m, o, _, _ = make_training()
        train(m, o, X, y, 0, 1)
        saved = {"s": copy.deepcopy(m.state_dict()), "o": copy.deepcopy(o.state_dict()), "r": torch.get_rng_state()}
        m2, o2, _, _ = make_training(tmp_seed=99)  # different process state
        m2.load_state_dict(saved["s"]); o2.load_state_dict(saved["o"])
        torch.set_rng_state(saved["r"] if restore_rng else torch.manual_seed(12345).get_state())
        train(m2, o2, X, y, 1, 3)
        return pv.state_hash(m2)

    assert resumed(True) == reference
    assert resumed(False) != reference  # control: dropout consumes the global RNG


class ReloadNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(50 * 39, 5)

    def forward(self, x):
        return {"logits": self.fc(x.flatten(1))}


def test_checkpoint_reload_matching_logits_pass(tmp_path):
    model = ReloadNet()
    folder = tmp_path / "seed-1"
    folder.mkdir()
    torch.save({"state_dict": model.state_dict()}, folder / "checkpoint.pt")
    x = torch.randn(3, 50, 39)
    clean = runner.ad.compute_logits(model, x.numpy(), 3)
    result = runner.checkpoint_reload_check(ReloadNet, folder, x.numpy(), clean, 3)
    assert result["bitwise_equal"] is True
    assert result["rows_checked"] == 3
    assert result["conditions_checked"] == ["Clean"]


def test_checkpoint_reload_mismatching_logits_fail(tmp_path):
    model = ReloadNet()
    folder = tmp_path / "seed-1"
    folder.mkdir()
    torch.save({"state_dict": model.state_dict()}, folder / "checkpoint.pt")
    x = torch.randn(3, 50, 39)
    clean = runner.ad.compute_logits(model, x.numpy(), 3)
    clean[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="Checkpoint reload mismatch"):
        runner.checkpoint_reload_check(ReloadNet, folder, x.numpy(), clean, 3)
