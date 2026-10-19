"""Full LOCAL released-artifact FP32 multiclass baseline; never a Table 4 claim.

Keeps the legacy smoke/target gate and corrected research pipeline intact.
Training uses the upstream train_seq AST with epoch callbacks/resume only.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from .milestone1 import ROOT, PIN, DATA_SHA, Tee, git, sha

CLAIM = "Released-artifact multiclass baseline and documented paper-reproduction attempt."
DEFAULT_CONFIG = ROOT / "configs/milestone1_artifact_full.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, val in model.state_dict().items():
        h.update(key.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = (cfg["dataset"]["sha256"], cfg["seeds"], cfg["training"]["epochs"], cfg["training"]["batch_size"], cfg["evaluation"]["cap"])
    if expected != (DATA_SHA, [1, 7, 13, 29], 15, 512, 3000):
        raise ValueError("Expected the frozen full released-artifact configuration, not smoke/paper/corrected settings")
    return cfg


def import_reference(root, cfg):
    upstream = ROOT / "upstream/PAVE_artifact"
    if git("rev-parse", "HEAD", cwd=upstream) != PIN or git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
        raise ValueError("Pinned upstream changed")
    for name, digest in cfg["upstream_source_sha256"].items():
        if sha(upstream / name) != digest:
            raise ValueError(f"Upstream source changed: {name}")
    if sha(upstream / "data/balanced_CICIoT.csv") != DATA_SHA:
        raise ValueError("Dataset hash changed")
    reference = root / "reference"
    if not reference.exists():
        shutil.copytree(upstream / "code", reference / "code", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(upstream / "LICENSE", reference / "LICENSE")
    for name, digest in cfg["upstream_source_sha256"].items():
        if sha(reference / name) != digest:
            raise ValueError(f"Isolated source changed: {name}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference / "code"))
    os.environ["AGT_OUT"] = str(root / "upstream_auxiliary")
    infra = importlib.import_module("sota_compare")
    infra.BAL = str(upstream / "data")
    infra.DEV = "cpu"
    return infra


def instrument_training(infra, resume, callback):
    """Retain original statements; add epoch range, state restore and callback.

    No new optimizer, loss, sampler, clipping, schedule or model operations.
    """
    tree = ast.parse(inspect.getsource(infra.train_seq))
    func = tree.body[0]
    loop = next(node for node in func.body if isinstance(node, ast.For))
    if ast.unparse(loop.iter) != "range(epochs)" or ast.unparse(loop.target) != "_":
        raise ValueError("Unexpected upstream training loop; refuse instrumentation")
    pos = func.body.index(loop)
    func.body.insert(pos, ast.parse("_m1_restore(opt)").body[0])
    loop.iter = ast.parse("range(_m1_start, epochs)", mode="eval").body
    loop.body.append(ast.parse("_m1_callback(_ + 1, model, opt)").body[0])
    ast.fix_missing_locations(tree)

    def restore(opt):
        if resume is not None:
            opt.load_state_dict(resume["optimizer"])
            torch.set_rng_state(resume["torch_rng"])

    namespace = dict(infra.__dict__)
    namespace.update(_m1_restore=restore, _m1_callback=callback, _m1_start=0 if resume is None else resume["epoch"])
    exec(compile(tree, "<upstream-train_seq-with-epoch-checkpoints>", "exec"), namespace)
    return namespace["train_seq"], ast.unparse(tree) + "\n"


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.min_available = psutil.virtual_memory().available
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def data_for_seed(infra, cfg, seed, folder):
    tick = time.perf_counter()
    d = infra.get_data("CICIoT", seed, "multiclass")
    frame = pd.read_csv(infra.BAL + "/balanced_CICIoT.csv")
    classes = list(map(str, d["class_names"]))
    labels = d["label_encoder"].transform(frame.Label)
    tr, va = train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed, stratify=labels)
    positions = np.arange(len(va))[:, None]
    selected = infra.cap_aligned(positions, positions, d["y_val"], n=3000, seed=42)[0].ravel()
    numeric = frame.drop(columns="Label").apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)
    scaled = d["scaler"].transform(numeric.values)
    np.testing.assert_array_equal(scaled[tr], d["X_train_raw"])
    np.testing.assert_array_equal(scaled[va], d["X_val_raw"])
    np.testing.assert_array_equal(labels[tr], d["y_train"])
    np.testing.assert_array_equal(labels[va], d["y_val"])
    assert (len(frame), len(tr), len(va), len(selected), d["n_features"], d["n_classes"]) == (29997, 23997, 6000, 3000, 39, 27)
    assert classes == cfg["dataset"]["class_names"] and d["feature_names"] == cfg["dataset"]["feature_names"]
    assert not set(tr).intersection(va)
    sc = d["scaler"]
    metadata = {"seed": seed, "split_seed": seed, "cap_seed": 42, "train": tr.tolist(), "validation": va.tolist(), "evaluation": va[selected].tolist(), "test": None, "row_id": "zero-based CSV data row; duplicate records retain distinct row IDs", "dataset_sha256": DATA_SHA}
    metadata_path = folder / "split_ids.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Existing split IDs differ")
    atomic_json(metadata_path, metadata)
    np.savez_compressed(folder / "preprocessing.npz", mean=sc.mean_, scale=sc.scale_, var=sc.var_)
    atomic_json(folder / "preprocessing.json", {"type": "StandardScaler", "fit_scope": "all rows BEFORE split", "n_samples_seen": int(sc.n_samples_seen_), "features": d["feature_names"], "class_mapping": dict(enumerate(classes)), "training_sequences": "row repeated 50 times plus Gaussian sigma=.01; numpy default_rng(split seed)", "evaluation_sequences": "row repeated 50 times; NO noise"})
    clean = infra.expand(d["X_val_raw"][selected])
    with np.load(folder / "preprocessing.npz") as saved:
        np.testing.assert_array_equal(((numeric.values[va[selected]] - saved["mean"]) / saved["scale"]).astype(np.float32), clean[:, 0])
    del d["X_val_seq"]
    return d, clean, np.asarray(d["y_val"])[selected], va[selected], time.perf_counter() - tick


def evaluate(infra, cfg, model, clean, labels, ids, folder, checkpoint_sha):
    model.eval()
    before = state_hash(model)
    batch = cfg["resources"]["evaluation_batch_size"]
    probe = clean[:2]
    fg0 = infra.seq_fgsm(model, probe, labels[:2], 0, bs=2)
    pg0 = infra.seq_pgd(model, probe, labels[:2], eps=0, alpha=.01, iters=10, bs=2)
    np.testing.assert_array_equal(fg0, probe)
    np.testing.assert_array_equal(pg0, probe)
    xt = torch.tensor(probe, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(model(xt)["logits"], torch.tensor(labels[:2], dtype=torch.long))
    grad = torch.autograd.grad(loss, xt)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    results = {}
    for name in ["Clean", "FGSM", "PGD"]:
        destination = folder / "evaluations" / (name + ".npz")
        meta_path = destination.with_suffix(".json")
        if destination.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if sha(destination) != meta["sha256"] or meta["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("Saved evaluation changed")
            with np.load(destination) as saved:
                np.testing.assert_array_equal(saved["row_ids"], ids)
                np.testing.assert_array_equal(saved["labels"], labels)
                results[name] = {"logits": saved["logits"].copy(), "meta": meta}
            continue
        tick = time.perf_counter()
        if name == "Clean":
            attacked = clean
        elif name == "FGSM":
            attacked = infra.seq_fgsm(model, clean, labels, .1, bs=batch)
        else:
            attacked = infra.seq_pgd(model, clean, labels, eps=.1, alpha=.01, iters=10, bs=batch)
        bounds = np.max(np.abs(attacked - clean), axis=(1, 2))
        if not np.isfinite(attacked).all() or bounds.max() > .10001:
            raise ValueError(f"{name}: finite-input/perturbation-bound failure")
        logits = infra.seq_logits(model, attacked, bs=batch)
        if not np.isfinite(logits).all() or state_hash(model) != before or model.training:
            raise ValueError("Evaluation changed parameters/model mode, or logits are not finite")
        destination.parent.mkdir(exist_ok=True)
        np.savez_compressed(destination, labels=labels, row_ids=ids, logits=logits, linf=bounds)
        meta = {"condition": name, "rows": len(labels), "seconds": time.perf_counter() - tick, "sha256": sha(destination), "checkpoint_sha256": checkpoint_sha, "max_linf": float(bounds.max()), "model_mode": "eval", "parameters_unchanged": True}
        atomic_json(meta_path, meta)
        results[name] = {"logits": logits, "meta": meta}
        print(f"EVALUATED {name}: {meta['seconds']:.1f}s; max Linf {meta['max_linf']:.8f}", flush=True)
    audit = importlib.import_module("pave_audit")
    pc = results["Clean"]["logits"].argmax(1)
    correct = pc == labels
    denominator = int(correct.sum())
    metrics, class_rows, predictions, matrices = [], [], [], {}
    for name, result in results.items():
        pred = result["logits"].argmax(1)
        numerator = int((correct & (pred != labels)).sum())
        row = {"seed": folder.name.split("-")[-1], "condition": name, "rows": len(labels), "accuracy": float(accuracy_score(labels, pred)), "macro_f1": float(f1_score(labels, pred, labels=np.arange(27), average="macro", zero_division=0)), "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)), "mcc": float(matthews_corrcoef(labels, pred)), "asr_numerator": numerator, "asr_denominator": denominator, "asr_percent": 100 * numerator / denominator if denominator else None, "upstream_pave_audit_asr_percent": audit._asr(labels, pc, pred), "prediction_change_all_percent": float((pred != pc).mean()) * 100, "all_row_error_percent": float((pred != labels).mean()) * 100, "checkpoint_sha256": checkpoint_sha}
        if not all(np.isfinite(row[k]) for k in ["accuracy", "macro_f1", "weighted_f1", "mcc"]):
            raise ValueError("Non-finite metrics")
        metrics.append(row)
        report = classification_report(labels, pred, labels=np.arange(27), output_dict=True, zero_division=0)
        for c in range(27):
            class_rows.append({"seed": row["seed"], "condition": name, "class_id": c, "class_name": cfg["dataset"]["class_names"][c], **report[str(c)]})
        matrices[name] = confusion_matrix(labels, pred, labels=np.arange(27))
        predictions.extend({"condition": name, "row_id": int(r), "label": int(y), "prediction": int(p), "checkpoint_sha256": checkpoint_sha} for r, y, p in zip(ids, labels, pred))
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    atomic_json(folder / "metrics.json", metrics)
    pd.DataFrame(class_rows).to_csv(folder / "per_class.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / "predictions.csv", index=False)
    np.savez_compressed(folder / "confusion_matrices.npz", **matrices)
    atomic_json(folder / "verification.json", {"epsilon_zero_fgsm_identity": True, "epsilon_zero_pgd_identity": True, "input_gradient_finite_nonzero": True, "input_gradient_max_abs": float(grad.abs().max()), "evaluation_weights_unchanged": state_hash(model) == before, "evaluation_mode": "eval", "same_examples_all_conditions": True, "checkpoint_sha256": checkpoint_sha})
    return metrics


def verify_seed(folder, cfg):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    required = {"checkpoint.pt", "metrics.csv", "metrics.json", "per_class.csv", "predictions.csv", "split_ids.json", "preprocessing.npz", "preprocessing.json", "confusion_matrices.npz", "verification.json", "runtime.json", "training_complete.json", "training_adapter.py"}
    if not required.issubset(manifest):
        raise ValueError("Missing required artifacts")
    for name, digest in manifest.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Changed/incomplete artifact: {path}")
    training = json.loads((folder / "training_complete.json").read_text())
    if (training["epochs"], training["train_rows"], training["optimizer_steps"], training["configuration_hash"]) != (15, 23997, 705, config_hash(cfg)):
        raise ValueError("Full training budget not established")
    epoch_files = sorted((folder / "epochs").glob("epoch-*.pt"))
    if len(epoch_files) != 15:
        raise ValueError("Expected all 15 recoverable epoch checkpoints")
    final_training_state = torch.load(epoch_files[-1], map_location="cpu", weights_only=True)
    steps = {int(value["step"].item()) for value in final_training_state["optimizer"]["state"].values() if "step" in value}
    if final_training_state["epoch"] != 15 or steps != {705} or final_training_state["configuration_hash"] != config_hash(cfg):
        raise ValueError("Actual saved optimizer state does not establish 705 updates")
    checks = json.loads((folder / "verification.json").read_text())
    if not checks.get("checkpoint_reload_logits_identical") or not checks["evaluation_weights_unchanged"]:
        raise ValueError("Checkpoint/evaluation verification missing")
    splits = json.loads((folder / "split_ids.json").read_text())
    if sorted(splits["train"] + splits["validation"]) != list(range(29997)) or len(set(splits["evaluation"])) != 3000 or not set(splits["evaluation"]).issubset(splits["validation"]):
        raise ValueError("Invalid split/evaluation coverage")
    predictions = pd.read_csv(folder / "predictions.csv")
    if len(predictions) != 9000 or predictions.duplicated(["condition", "row_id"]).any():
        raise ValueError("Incomplete/duplicate predictions")
    summaries = pd.read_csv(folder / "metrics.csv")
    expected_seed = int(folder.name.split("-")[-1])
    if len(summaries) != 3 or set(summaries.seed) != {expected_seed} or expected_seed not in cfg["seeds"] or set(summaries.condition) != {"Clean", "FGSM", "PGD"} or not np.isfinite(summaries[["accuracy", "macro_f1", "weighted_f1", "mcc"]]).all().all():
        raise ValueError("Incomplete/non-finite metrics")
    class_rows = pd.read_csv(folder / "per_class.csv")
    if len(class_rows) != 81:
        raise ValueError("Missing per-class metrics")
    for name in ["Clean", "FGSM", "PGD"]:
        group = predictions[predictions.condition == name]
        if group.row_id.tolist() != splits["evaluation"] or set(group.checkpoint_sha256) != {manifest["checkpoint.pt"]}:
            raise ValueError("Mismatched checkpoint/evaluation examples")
        with np.load(folder / "evaluations" / (name + ".npz")) as saved:
            np.testing.assert_array_equal(saved["row_ids"], group.row_id)
            np.testing.assert_array_equal(saved["labels"], group.label)
            np.testing.assert_array_equal(saved["logits"].argmax(1), group.prediction)
            assert np.isfinite(saved["logits"]).all() and saved["linf"].max() <= .10001
        row = summaries[summaries.condition == name].iloc[0]
        recomputed = [accuracy_score(group.label, group.prediction), f1_score(group.label, group.prediction, labels=np.arange(27), average="macro", zero_division=0), f1_score(group.label, group.prediction, average="weighted", zero_division=0), matthews_corrcoef(group.label, group.prediction)]
        np.testing.assert_allclose(row[["accuracy", "macro_f1", "weighted_f1", "mcc"]].to_numpy(dtype=float), recomputed, atol=1e-12, rtol=0)
        with np.load(folder / "confusion_matrices.npz") as saved:
            np.testing.assert_array_equal(saved[name], confusion_matrix(group.label, group.prediction, labels=np.arange(27)))
        assert class_rows[class_rows.condition == name].support.sum() == 3000
    return summaries


def run_seed(infra, cfg, seed, root, resume_allowed):
    folder = root / f"seed-{seed}"
    if (folder / "manifest.json").exists():
        verify_seed(folder, cfg)
        print(f"REUSED verified completed seed {seed}", flush=True)
        return
    if folder.exists() and not resume_allowed:
        raise ValueError(f"Existing incomplete seed {seed}; use --resume")
    folder.mkdir(exist_ok=True)
    started = time.perf_counter()
    with (folder / "raw.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)), MemoryMonitor() as memory:
        print(f"START seed {seed}, pid={os.getpid()}, full 15 epochs, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        d, clean, labels, ids, data_seconds = data_for_seed(infra, cfg, seed, folder)
        model = infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu")
        assert sum(p.numel() for p in model.parameters()) == 351612
        epoch_files = sorted((folder / "epochs").glob("epoch-*.pt"))
        resume = torch.load(epoch_files[-1], map_location="cpu", weights_only=True) if epoch_files else None
        if resume:
            if resume["configuration_hash"] != config_hash(cfg) or resume["seed"] != seed:
                raise ValueError("Checkpoint/configuration mismatch")
            model.load_state_dict(resume["state_dict"], strict=True)
            print(f"RESUME from completed epoch {resume['epoch']}; optimizer and torch RNG restored", flush=True)
        previous_seconds = 0 if resume is None else resume["training_seconds"]
        training_tick = time.perf_counter()
        epoch_tick = training_tick

        def callback(epoch, current, opt):
            nonlocal epoch_tick
            now = time.perf_counter()
            elapsed = previous_seconds + now - training_tick
            atomic_save(folder / "epochs" / f"epoch-{epoch:02}.pt", {"state_dict": current.state_dict(), "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(), "epoch": epoch, "seed": seed, "training_seconds": elapsed, "configuration_hash": config_hash(cfg)})
            atomic_json(folder / "status.json", {"state": "training", "seed": seed, "epoch_completed": epoch, "epochs_required": 15, "train_rows": 23997, "training_seconds": elapsed, "last_epoch_seconds": now - epoch_tick, "rss_bytes": psutil.Process().memory_info().rss, "updated_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            print(f"EPOCH {epoch}/15 seed={seed}: {now - epoch_tick:.2f}s; cumulative train={elapsed:.2f}s; RSS={psutil.Process().memory_info().rss / 2**30:.2f} GiB", flush=True)
            epoch_tick = time.perf_counter()

        trainer, adapter = instrument_training(infra, resume, callback)
        (folder / "training_adapter.py").write_text(adapter, encoding="utf-8")
        trainer(model, d["X_train_seq"], np.asarray(d["y_train"]), epochs=15, bs=512)
        del d["X_train_seq"]
        training_seconds = previous_seconds + time.perf_counter() - training_tick
        latest = torch.load(folder / "epochs/epoch-15.pt", map_location="cpu", weights_only=True)
        if latest["epoch"] != 15:
            raise ValueError("Final epoch missing")
        if not (folder / "checkpoint.pt").exists():
            atomic_save(folder / "checkpoint.pt", {"state_dict": model.state_dict(), "epoch": 15, "architecture": cfg["architecture"], "configuration_hash": config_hash(cfg), "claim": CLAIM, "seed": seed})
        before = infra.seq_logits(model, clean, bs=cfg["resources"]["evaluation_batch_size"])
        saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["configuration_hash"] != config_hash(cfg) or saved["epoch"] != 15:
            raise ValueError("Saved final checkpoint mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        np.testing.assert_array_equal(before, infra.seq_logits(model, clean, bs=cfg["resources"]["evaluation_batch_size"]))
        atomic_json(folder / "training_complete.json", {"epochs": 15, "train_rows": 23997, "samples_processed": 359955, "optimizer_steps": 705, "batch_size": 512, "checkpoint_rule": "final epoch, no evaluation-based selection", "configuration_hash": config_hash(cfg), "training_seconds": training_seconds})
        metrics = evaluate(infra, cfg, model, clean, labels, ids, folder, sha(folder / "checkpoint.pt"))
        verification = json.loads((folder / "verification.json").read_text())
        verification["checkpoint_reload_logits_identical"] = True
        atomic_json(folder / "verification.json", verification)
        atomic_json(folder / "runtime.json", {"data_seconds_this_invocation": data_seconds, "training_seconds_accumulated": training_seconds, "evaluation_seconds": {n: json.loads((folder / "evaluations" / (n + ".json")).read_text())["seconds"] for n in ["Clean", "FGSM", "PGD"]}, "invocation_seconds_after_imports": time.perf_counter() - started, "peak_rss_sampled_bytes_this_invocation": memory.peak, "minimum_system_available_bytes_this_invocation": memory.min_available, "completed_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print("RESULTS", json.dumps(metrics), flush=True)
    manifest = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name not in {"status.json", "manifest.json"} and not p.name.endswith(".tmp") and "epochs" not in p.parts}
    atomic_json(folder / "manifest.json", manifest)
    verify_seed(folder, cfg)
    atomic_json(folder / "status.json", {"state": "complete", "seed": seed, "full_budget_and_outputs_verified": True})
    print(f"COMPLETED AND VERIFIED SEED {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    seeds = args.seeds or cfg["seeds"]
    if len(set(seeds)) != len(seeds) or any(s not in cfg["seeds"] for s in seeds):
        parser.error("Use only the frozen predetermined replication seeds")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "configuration.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != cfg:
            parser.error("Existing run configuration differs")
    else:
        atomic_json(frozen, cfg)
        atomic_json(root / "environment.json", {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(), "cpu": platform.processor(), "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total, "cuda_available": torch.cuda.is_available(), "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"), "executor_sha256": sha(Path(__file__)), "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        shutil.copy2(__file__, root / "executor_snapshot.py")
    if args.verify_only:
        for seed in seeds:
            verify_seed(root / f"seed-{seed}", cfg)
        print("All requested seed bundles verified")
        return
    lock = root / "run.lock"
    if lock.exists():
        old = json.loads(lock.read_text())
        if psutil.pid_exists(old["pid"]) and abs(psutil.Process(old["pid"]).create_time() - old["create_time"]) < 1:
            parser.error(f"Run already active as PID {old['pid']}")
        lock.rename(root / f"stale-lock-{time.time_ns()}.json")
    with lock.open("x") as stream:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)
    try:
        torch.set_num_threads(cfg["resources"]["torch_threads"])
        torch.use_deterministic_algorithms(True)
        infra = import_reference(root, cfg)
        for seed in seeds:
            # Complete and independently validate each seed before starting another.
            run_seed(infra, cfg, seed, root, args.resume)
            verify_seed(root / f"seed-{seed}", cfg)
        atomic_json(root / "requested_run_complete.json", {"seeds_completed_this_request": seeds, "claim": CLAIM})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
