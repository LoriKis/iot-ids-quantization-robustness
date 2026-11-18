"""Helpers for the stub-based runner tests: a fake V4 folder, a locked run, and call-recording stubs.

Nothing here imports torch or research.*. The stubs replace the ``deps`` object of ``m2_deadline.stages``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from conftest import CONFIG_PATH, fake_efficiency_cfg, make_measurements, make_split_and_csv

from m2_deadline import markers, protocol, sampling, timing
from m2_deadline.common import (
    MODEL_IDS,
    atomic_write_json,
    checkpoint_path,
    load_config,
    model_dir,
    out_path,
    sha256_file,
    utc_now,
)

SOURCE_FILES = markers.SOURCE_FILES


def make_fake_v4(root, test_counts=None):
    """Create a minimal V4-shaped folder and return ``(root, split_dict)``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    split, _csv = make_split_and_csv(root, test_counts=test_counts)  # dataset.csv lands in root
    (root / "MANIFEST.sha256").write_text("0" * 64 + "  README.md\n", encoding="utf-8")
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (root / "research").mkdir()
    for name in SOURCE_FILES:
        (root / "research" / name).write_text("# stub %s\n" % name, encoding="utf-8")
    cfg = load_config(CONFIG_PATH)
    for key, rel in cfg["siblings"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "efficiency":
            doc = fake_efficiency_cfg()
        elif key == "data":
            doc = {"dataset": {"path": split["dataset"]["path"]}}
        else:
            doc = {"stub": key}
        path.write_text(json.dumps(doc), encoding="utf-8")
    split_dir = root / cfg["split_dir"]
    split_dir.mkdir(parents=True)
    (split_dir / "split_m2.json").write_text(json.dumps(split), encoding="utf-8")
    (split_dir / "scaler_minmax.json").write_text(json.dumps({"stub": "scaler"}), encoding="utf-8")
    return root, split


def cheap_contention(config):
    return {"contended": False, "load": 1.0, "busy_processes": []}


def make_locked_run(tmp_path, hours=11.0, **measure_kwargs):
    """Fake V4 + OUT with a real estimate and lock produced by ``timing.main`` with injected measurements."""
    v4, split = make_fake_v4(Path(tmp_path) / "v4")
    out = Path(tmp_path) / "artifacts" / "m2-deadline-v1"
    code = timing.main(
        ["--hours-remaining", str(hours), "--v4-root", str(v4), "--out", str(out), "--config", str(CONFIG_PATH)],
        measure_fn=lambda env: make_measurements(**measure_kwargs),
        contention_fn=cheap_contention,
    )
    assert code == 0, code
    return SimpleNamespace(v4=v4, out=out, split=split, config=load_config(CONFIG_PATH))


# ---------------------------------------------------------------- stage stubs
class CallLog:
    def __init__(self):
        self.calls = []

    def add(self, name, **kw):
        self.calls.append((name, kw))

    def names(self):
        return [c[0] for c in self.calls]

    def of(self, name):
        return [c[1] for c in self.calls if c[0] == name]


class FakeModel:
    def __init__(self, tag, shift=0):
        self.tag = tag
        self.shift = shift


class StageStubDeps:
    """Records every call of the deps object; returns synthetic data. No torch."""

    def __init__(self, split, v4_root, n_train=64, n_val=54, epochs_in_log=None, quantized=23):
        self.log = CallLog()
        self.split = split
        self.v4_root = Path(v4_root)
        self.n_train = n_train
        self.n_val = n_val
        self.epochs_in_log = epochs_in_log
        self.quantized = quantized
        self.id_hash_override = None
        self.verify_overlay_rc = 0
        self.logits_nonfinite = False
        self.efficiency_failed = False
        rng = np.random.default_rng(3)
        self._train = {
            "ids": np.arange(n_train, dtype=np.int64) + 100000,
            "x": rng.random((n_train, 39)).astype(np.float32),
            "y": (np.arange(n_train) % 27).astype(np.int64),
        }
        self._val = {
            "ids": np.arange(n_val, dtype=np.int64) + 200000,
            "x": rng.random((n_val, 39)).astype(np.float32),
            "y": (np.arange(n_val) % 27).astype(np.int64),
        }
        parent = self

        # ---- research.m2_training
        class TrainingInterrupted(RuntimeError):
            pass

        def train_unit(cfg, X, y, seed, out_dir=None, **kw):
            parent.log.add("train_unit", out_dir=out_dir, seed=seed, **kw)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "checkpoint.pt").write_bytes(b"ckpt-" + str(kw["adversarial"]).encode())
            log = {
                "epochs": kw["epochs"] if parent.epochs_in_log is None else parent.epochs_in_log,
                "adversarial": kw["adversarial"], "seed": seed, "train_rows": int(len(X)),
                "init_state_hash": "init", "final_state_hash": "final-%s" % kw["adversarial"],
                "optimizer_updates": 7, "checkpoint_sha256": sha256_file(out_dir / "checkpoint.pt"),
            }
            (out_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
            return {"model": FakeModel("trained"), "log": log}

        self.mt = SimpleNamespace(
            TrainingInterrupted=TrainingInterrupted,
            load_config=lambda p: {"architecture": {"d_model": 4}, "training": {"batch_size": 512}},
            train_unit=train_unit,
        )

        # ---- research.m2_quant
        def calibration_ids(arrays, per_class, split_seed):
            parent.log.add("calibration_ids", per_class=per_class, split_seed=split_seed)
            ids = [int(i) for i in arrays["train"]["ids"][:20]]
            if getattr(parent, "calibration_outside_train", False):
                ids[0] = -5
            return {"ids": ids, "id_sha256": "cal" * 21 + "c", "n": len(ids), "per_class": per_class,
                    "x": np.zeros((len(ids), 50, 39), dtype=np.float32)}

        def save_calibration(cal, path):
            Path(path).write_text(json.dumps({"id_sha256": cal["id_sha256"]}), encoding="utf-8")

        def ptq_unit(model, cal, cfg, out_dir=None, **kw):
            parent.log.add("ptq_unit", out_dir=out_dir, **kw)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "state.pt").write_bytes(b"int8-" + kw["variant"].encode())
            conversion = {
                "variant": kw["variant"], "requested": ["layer%d" % i for i in range(parent.quantized)],
                "calibration": {"id_sha256": cal["id_sha256"]},
                "reload_check": {"logits_bitwise_equal": True, "argmax_identical": True},
            }
            (out_dir / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
            return {"model": FakeModel("int8"), "conversion": conversion}

        self.mq = SimpleNamespace(
            load_config=lambda p: {"ptq": {"calibration_per_class": 20, "scope": "all_linear", "backend": "fbgemm",
                                           "expected_quantized_linear": 23}},
            calibration_ids=calibration_ids, save_calibration=save_calibration, ptq_unit=ptq_unit,
        )

        # ---- research.m2_attacks
        def evaluate_unit(**kw):
            parent.log.add("evaluate_unit", **{k: v for k, v in kw.items() if k not in ("x", "models", "cfg")})
            y = np.asarray(kw["y"], dtype=np.int64)
            ids = np.asarray(kw["ids"], dtype=np.int64)
            eps = float(kw["epsilon"])
            pred_adv = y.copy() if eps == 0.0 else (y + 1) % 27
            out_dir = Path(kw["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "attack_checks.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
            (out_dir / "attack_result.json").write_text(json.dumps({"n": int(len(y))}), encoding="utf-8")
            valid = not getattr(parent, "invalid_unit", None) == (kw["model_id"], kw["attack_name"], eps)
            checks = {"valid": valid, "failures": [] if valid else ["linf_budget"],
                      "example_id_sha256": kw["expected_ids_sha256"]}
            return {
                "row": {"robust_accuracy": float(np.mean(pred_adv == y)), "attack_seconds": 0.01},
                "checks": checks,
                "arrays": {"ids": ids, "y": y, "pred_clean": y.copy(), "pred_adv": pred_adv,
                           "robust_correct": pred_adv == y, "linf": np.zeros(len(y), dtype=np.float32)},
            }

        self.ma = SimpleNamespace(
            load_config=lambda p: {"attacks": {"fgsm": {}}, "expected_selector_calls_per_batch": {},
                                   "evaluation": {"batch_size": 64, "evaluation_partition": "validation"}},
            validate_config=lambda cfg: cfg,
            evaluate_unit=evaluate_unit,
        )

        # ---- research.m2_efficiency
        def run_efficiency(models, rows, cfg, out_dir, **kw):
            parent.log.add("run_efficiency", models=sorted(models), out_dir=out_dir, **kw)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fixed_rows.json").write_text("{}", encoding="utf-8")
            reps = 3
            record = {
                "any_failed": parent.efficiency_failed, "interleaved": True, "repetitions": reps, "threads": kw["threads"],
                "variants": list(models), "summary": {}, "peak_memory_metric": ["VmHWM"],
                "measurements": [{"variant": m, "result": {}} for _ in range(reps) for m in models],
            }
            (out_dir / "efficiency.json").write_text(json.dumps(record), encoding="utf-8")
            return record

        self.me = SimpleNamespace(
            load_config=lambda p: fake_efficiency_cfg(),
            select_rows=lambda ids, x, count, split_seed: {"ids": list(ids[:count]), "x": x[:count]},
            run_efficiency=run_efficiency,
        )

        # ---- research.m2_split / m2b_state / m2b_exec / quantization
        def load_arrays(split_path, scaler_path, partitions=("train", "validation"), *, purpose=None, freeze=None, **kw):
            parent.log.add("load_arrays", partitions=tuple(partitions), purpose=purpose, freeze=freeze)
            out = {}
            for p in partitions:
                if p == "train":
                    out[p] = parent._train
                elif p == "validation":
                    out[p] = parent._val
                elif p == "test":
                    assert purpose == "final_test" and freeze is not None, "test arrays need a verified freeze"
                    ids, y = protocol.read_test_labels(parent.split, None, parent.v4_root / parent.split["dataset"]["path"])
                    x = np.zeros((len(ids), 39), dtype=np.float32)
                    x[:, 0] = y
                    out[p] = {"ids": ids, "x": x, "y": y}
            return out

        self.m2_split = SimpleNamespace(
            load_arrays=load_arrays,
            id_hash=lambda ids: parent.id_hash_override or sampling.ids_sha256(ids),
        )

        def verify_eval_freeze(path, sha, **kw):
            parent.log.add("verify_eval_freeze", path=Path(path), sha=sha)
            assert sha256_file(path) == sha
            return SimpleNamespace(path=Path(path), sha256=sha, data={})

        self.m2b_state = SimpleNamespace(verify_eval_freeze=verify_eval_freeze)
        self.m2b_exec = SimpleNamespace(deep_copy_test_attack_cfg=self._deep_copy_test_cfg)
        self.quantization = SimpleNamespace(bpda_model=lambda q, requested, example: FakeModel("bpda"))

    @staticmethod
    def _deep_copy_test_cfg(cfg):
        out = copy.deepcopy(cfg)
        out["evaluation"]["evaluation_partition"] = "test"
        return out

    # ---- deps helper methods used directly by the stages
    def configure_inference(self, threads=1, engine="fbgemm"):
        self.log.add("configure_inference", threads=threads)

    def sequences(self, x):
        return np.asarray(x)

    def logits(self, model, x, batch_size=64):
        n = len(x)
        out = np.zeros((n, 27), dtype=np.float32)
        out[np.arange(n), np.arange(n) % 27] = 1.0
        if self.logits_nonfinite and model.tag == "bad":
            out[0, 0] = np.nan
        return out

    def predict(self, model, x, batch_size=64):
        y = np.asarray(x)[:, 0].astype(np.int64)
        pred = y.copy()
        pred[::5] = (pred[::5] + model.shift) % 27
        return pred

    def load_fp32(self, checkpoint, architecture):
        return FakeModel("fp32")

    def load_int8(self, spec):
        return FakeModel("int8")

    def load_tensor(self, path):
        return np.zeros((20, 50, 39), dtype=np.float32)

    def save_tensor(self, value, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"tensor")

    def parameter_count(self, model):
        return 123

    def import_check(self):
        self.log.add("import_check")

    def run_verify_overlay(self, overlay_root, v4_root):
        self.log.add("run_verify_overlay")
        return self.verify_overlay_rc, "stub verify output"


def make_ctx(tmp_path, epochs=10, per_class=10, **deps_kw):
    """RunContext over a fake V4 and a temp OUT with a stub deps object (lock/estimate hand-built)."""
    from m2_deadline import stages

    v4, split = make_fake_v4(Path(tmp_path) / "v4")
    out = Path(tmp_path) / "artifacts" / "m2-deadline-v1"
    out.mkdir(parents=True)
    config = load_config(CONFIG_PATH)
    deps = StageStubDeps(split, v4, **deps_kw)
    lock = {
        "schema": 1, "kind": "m2_deadline_budget_lock", "run_id": config["run_id"], "locked_utc": utc_now(),
        "epochs": epochs, "per_class_target": per_class, "planned_hours": 3.0, "cap_hours": 8.0,
        "hours_remaining_at_timing": 11.0, "estimate_sha256": "e" * 64, "config_sha256": sha256_file(CONFIG_PATH),
        "selection_basis": "runtime only", "performance_inspected": False,
    }
    estimate = {"data": {"n_train": deps.n_train, "n_val": deps.n_val, "n_test": len(split["row_ids"]["test"])},
                "config_sha256": sha256_file(CONFIG_PATH), "v4_manifest_sha256": sha256_file(v4 / "MANIFEST.sha256")}
    ctx = stages.RunContext(
        out=out, v4_root=v4, config=config, config_sha256=sha256_file(CONFIG_PATH), lock=lock,
        lock_sha256="l" * 64, estimate=estimate, estimate_sha256="e" * 64, deps=deps,
        overlay_root=Path(CONFIG_PATH).parents[1], config_path=CONFIG_PATH,
    )
    return ctx


def preload_models(ctx, shifts=None, write_conversion=True):
    """Put fake models into the context cache (skips deadline_meta verification) and write conversion files."""
    shifts = shifts or {"V1": 0, "V2": 1, "V3": 2, "V4": 3}
    for m in MODEL_IDS:
        ctx._cache["model_" + m] = FakeModel(m, shifts[m])
    for m in (("V2", "V4") if write_conversion else ()):
        d = model_dir(ctx.out, m)
        d.mkdir(parents=True, exist_ok=True)
        (d / "conversion.json").write_text(json.dumps({"requested": ["a", "b"]}), encoding="utf-8")
    cal = out_path(ctx.out, "calibration_x")
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_bytes(b"tensor")
