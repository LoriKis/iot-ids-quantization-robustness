"""Runner for the corrected paper-described Adam experiment (variant C).

torch.optim.Adam(lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8) -- NOT AdamW.
Independent of research/paper_described_baseline.py (historical variant B, untouched).

Order of operations in main() (a contract, see docs/MILESTONE1_REPAIR_DESIGN.md 3.4):
  1. read config  2. compute_fingerprint (read-only)  3. check_resume (read-only)
  4. only then: lock, status, log, fingerprint + source snapshots, data prep, training.
Exit codes: 3 resume/partition gate refusal, 4 fatal validation error, 5 run active,
6 infeasible adversarial constraint.
Not an exact reproduction of the paper's Table 4.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from . import adversarial_domain as ad
from . import provenance as pv
from .milestone1 import DATA_SHA, PIN, ROOT, Tee, git, sha

CLAIM = "Corrected paper-described Adam experiment (C); multiclass 27-class; not an exact reproduction of Table 4."
DEFAULT_CONFIG = ROOT / "configs/milestone1_paper_described_adam.json"
DEFAULT_RUN = ROOT / "runs/milestone1-paper-described-adam"
SMOKE_RUN = ROOT / "runs/milestone1-paper-described-adam-smoke"
HISTORICAL_CONFIGS = {"milestone1_paper_described.json", "milestone1_artifact_full.json"}
TOP_KEYS = {"schema_version", "claim", "frozen_date_local", "upstream_revision", "base_commit_note", "supersedes_nothing",
            "dataset", "seeds", "split", "preprocessing", "architecture", "sequence", "training", "attack",
            "evaluation", "smoke_policy", "metrics", "resources", "paper_table4_correspondence"}


class PartitionError(ValueError):
    pass


# ------------------------------------------------------------------ config / optimizer

def read_adam_config(path: Path | str) -> dict:
    path = Path(path)
    if path.name in HISTORICAL_CONFIGS:
        raise ValueError(f"{path.name} is a historical configuration; use the Adam configuration")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg) != TOP_KEYS:
        raise ValueError(f"Unknown or missing configuration keys: {sorted(set(cfg) ^ TOP_KEYS)}")
    t, a, s = cfg["training"], cfg["attack"], cfg["split"]
    if "adamw" in json.dumps(t["optimizer"]).lower() or t["optimizer"] != "torch.optim.Adam":
        raise ValueError("Optimizer must be torch.optim.Adam (not AdamW)")
    if t.get("decoupled_weight_decay") is not False:
        raise ValueError("decoupled_weight_decay must be false")
    expected = {"lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999], "eps": 1e-08, "amsgrad": False, "maximize": False}
    if t["optimizer_kwargs"] != expected:
        raise ValueError("Optimizer kwargs differ from the frozen Adam settings")
    if (t["epochs"], t["batch_size"], t["steps_per_epoch"], t["final_batch_rows"], t["optimizer_steps_total"], t["drop_last"]) != (25, 512, 42, 5, 1050, False):
        raise ValueError("Training budget differs from the frozen settings")
    if cfg["seeds"] != [1, 7, 13, 29] or cfg["dataset"]["sha256"] != DATA_SHA or cfg["dataset"]["classes"] != 27:
        raise ValueError("Seeds or dataset differ from the frozen settings")
    if (s["split_seed"], s["mode"], s["train_rows"], s["validation_rows"], s["test_rows"]) != (2025, "load_saved", 20997, 4500, 4500):
        raise ValueError("Split settings differ from the frozen settings")
    if s["source"].endswith("split_ids.json") is False:
        raise ValueError("Unexpected split source")
    if (a["fgsm"]["epsilon"], a["pgd"]["epsilon"], a["pgd"]["iterations"], a["pgd"]["alpha"], a["pgd"]["random_start"]) != (0.10, 0.10, 7, 0.025, True):
        raise ValueError("Attack settings differ from the frozen settings")
    split_file = ROOT / s["source"]
    if sha(split_file) != s["sha256"]:
        raise ValueError("Saved split hash differs from the configuration")
    return cfg


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    kw = dict(cfg["training"]["optimizer_kwargs"])
    kw["betas"] = tuple(kw["betas"])
    if "decoupled_weight_decay" in inspect.signature(torch.optim.Adam.__init__).parameters:
        kw["decoupled_weight_decay"] = False
    opt = torch.optim.Adam(model.parameters(), **kw)
    assert type(opt) is torch.optim.Adam and not isinstance(opt, torch.optim.AdamW)
    return opt


def derived_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") & (2**63 - 1)


def epoch_batches(n_rows: int, batch_size: int, perm) -> list:
    """Index slices of one epoch; the short final batch is kept (no drop_last)."""
    return [perm[i:i + batch_size] for i in range(0, n_rows, batch_size)]


# ------------------------------------------------------------------ data

def build_data(cfg: dict, root: Path = ROOT) -> dict:
    """Read-only data preparation: saved split, in-memory regeneration cross-check, training-only scaler."""
    split = pv.load_saved_split(root, cfg)
    df, values, labels, classes = pv.load_dataset(root / cfg["dataset"]["path"], cfg)
    if len(df) != cfg["dataset"]["rows"]:
        raise ValueError("Row count differs from the configuration")
    n = len(df)
    for k in ("train", "validation", "test"):
        if split[k].min() < 0 or split[k].max() >= n:
            raise ValueError(f"{k} IDs outside the CSV row range")
        if set(labels[split[k]]) != set(range(27)):
            raise ValueError(f"{k} partition does not cover all classes")
    seed = cfg["split"]["split_seed"]
    idx = np.arange(n)
    tv, te = train_test_split(idx, test_size=4500, random_state=seed, stratify=labels)
    tr, va = train_test_split(tv, test_size=4500, random_state=seed, stratify=labels[tv])
    if not (np.array_equal(tr, split["train"]) and np.array_equal(va, split["validation"]) and np.array_equal(te, split["test"])):
        raise ValueError("In-memory regenerated split differs from the saved split")
    scaler = pv.fit_scaler(values, split["train"])
    x = {k: scaler.transform(values[split[k]]).astype(np.float32) for k in ("train", "validation", "test")}
    for k, arr in x.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite scaled values in {k}")
    return {"values": values, "labels": labels, "classes": classes, "split": split, "scaler": scaler,
            "X": x, "y": {k: labels[split[k]] for k in x}}


def data_range_report(data: dict) -> dict:
    rep = {}
    for k, arr in data["X"].items():
        rep[k] = {"rows": int(len(arr)), "min": float(arr.min()), "max": float(arr.max()),
                  "values_below_0": int((arr < 0).sum()), "values_above_1": int((arr > 1).sum()),
                  "all_finite": bool(np.isfinite(arr).all())}
    rep["constant_training_features"] = int((data["scaler"].data_min_ == data["scaler"].data_max_).sum())
    return rep


def domain_scan(data: dict, eps: float) -> dict:
    """Rows (per partition) that are infeasible for eps: any x < -eps or x > 1+eps."""
    out = {"epsilon": eps}
    for k in ("validation", "test"):
        arr = data["X"][k]
        e = np.float32(eps)
        bad = np.where(((arr - e) > 1.0).any(axis=1) | ((arr + e) < 0.0).any(axis=1))[0]
        out[k] = {"infeasible_rows": int(len(bad)), "row_ids": [int(data["split"][k][i]) for i in bad]}
    return out


def to_sequence(x: np.ndarray, length: int = 50) -> np.ndarray:
    return np.ascontiguousarray(np.repeat(x[:, None, :], length, axis=1).astype(np.float32))


# ------------------------------------------------------------------ smoke selection

def select_smoke_ids(split: dict, labels: np.ndarray, per_class_train: int = 10, per_class_val: int = 5) -> dict:
    train = np.sort(np.asarray(split["train"]))
    val = np.sort(np.asarray(split["validation"]))
    test_ids = set(int(i) for i in split["test"])
    s_train, s_val = [], []
    for c in range(27):
        t = train[labels[train] == c]
        v = val[labels[val] == c]
        if len(t) < per_class_train or len(v) < per_class_val:
            raise PartitionError(f"class {c} has too few rows for the smoke selection")
        s_train.extend(int(i) for i in t[:per_class_train])
        s_val.extend(int(i) for i in v[:per_class_val])
    if not (set(s_train) <= set(int(i) for i in train)):
        raise PartitionError("smoke training IDs are not a subset of the saved training partition")
    if not (set(s_val) <= set(int(i) for i in val)):
        raise PartitionError("smoke validation IDs are not a subset of the saved validation partition")
    if set(s_train) & test_ids or set(s_val) & test_ids:
        raise PartitionError("smoke selection intersects the saved test IDs")
    if set(s_train) & set(s_val):
        raise PartitionError("smoke training and validation IDs intersect")
    if (len(s_train), len(s_val)) != (270, 135):
        raise PartitionError("unexpected smoke selection sizes")
    return {"train_ids": s_train, "validation_ids": s_val,
            "train_ids_sha256": pv.canonical_hash(s_train), "validation_ids_sha256": pv.canonical_hash(s_val),
            "assertions": {"train_subset_of_saved_train": True, "validation_subset_of_saved_validation": True,
                           "no_intersection_with_saved_test": True, "train_validation_disjoint": True}}


# ------------------------------------------------------------------ training

def run_epochs(model, opt, X, y, seed: int, start_epoch: int, epochs: int, batch_size: int,
               seq_len: int, noise_std: float, on_epoch=None) -> None:
    """Training loop. Shuffle and noise come from local generators derived from (seed, epoch, batch)."""
    Xt, yt = torch.from_numpy(X), torch.from_numpy(np.asarray(y, dtype=np.int64))
    for ep in range(start_epoch, epochs):
        model.train()
        g = torch.Generator(device="cpu")
        g.manual_seed(derived_seed("shuffle", seed, ep))
        perm = torch.randperm(len(Xt), generator=g)
        batches = epoch_batches(len(Xt), batch_size, perm)
        total_loss = 0.0
        for b, idx in enumerate(batches):
            ng = torch.Generator(device="cpu")
            ng.manual_seed(derived_seed("noise", seed, ep, b))
            xb = Xt[idx][:, None, :].expand(-1, seq_len, -1)
            xb = xb + torch.randn(xb.shape, generator=ng, dtype=torch.float32) * noise_std
            opt.zero_grad()
            loss = F.cross_entropy(model(xb)["logits"], yt[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
        if on_epoch:
            on_epoch(ep + 1, total_loss / len(Xt), [len(i) for i in batches])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def optimizer_steps(opt) -> set:
    return {int(v["step"].item() if hasattr(v["step"], "item") else v["step"]) for v in opt.state_dict()["state"].values()}


# ------------------------------------------------------------------ evaluation

def evaluate_seed(cfg, model, X_test_seq, y_test, test_ids, folder: Path, ckpt_sha: str, seed: int) -> list:
    a = cfg["attack"]
    bs = a["attack_batch_size"]
    model.eval()
    before = pv.state_hash(model)
    conditions = {}
    t0 = time.perf_counter()
    conditions["Clean"] = X_test_seq
    conditions["FGSM"] = ad.fgsm(model, X_test_seq, y_test, a["fgsm"]["epsilon"], bs)
    conditions["PGD"] = ad.pgd(model, X_test_seq, y_test, a["pgd"]["epsilon"], a["pgd"]["alpha"], a["pgd"]["iterations"], seed, bs)
    attack_seconds = time.perf_counter() - t0
    if pv.state_hash(model) != before:
        raise ValueError("Attack altered model parameters")
    logits, bounds = {}, {}
    (folder / "evaluations").mkdir(exist_ok=True)
    for name, arr in conditions.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: non-finite adversarial inputs")
        d = np.abs(arr - X_test_seq).max(axis=(1, 2))
        if name != "Clean":
            if d.max() > a[name.lower()]["epsilon"] + 1e-6:
                raise ValueError(f"{name}: Linf bound violated ({d.max()})")
            lo = np.maximum(X_test_seq - np.float32(a[name.lower()]["epsilon"]), 0.0)
            hi = np.minimum(X_test_seq + np.float32(a[name.lower()]["epsilon"]), 1.0)
            if (arr < lo).any() or (arr > hi).any():
                raise ValueError(f"{name}: domain intersection violated")
        logits[name] = ad.compute_logits(model, arr, bs)
        if not np.isfinite(logits[name]).all():
            raise ValueError(f"{name}: non-finite logits")
        bounds[name] = d
        np.savez_compressed(folder / "evaluations" / f"{name}.npz", labels=y_test, row_ids=test_ids, logits=logits[name], linf=d)
    if pv.state_hash(model) != before or model.training:
        raise ValueError("Evaluation altered model parameters or mode")

    import pandas as pd
    clean_pred = logits["Clean"].argmax(axis=1)
    clean_ok = clean_pred == y_test
    den = int(clean_ok.sum())
    metrics, class_rows, preds, mats = [], [], [], {}
    for name, lg in logits.items():
        pred = lg.argmax(axis=1)
        num = int((clean_ok & (pred != y_test)).sum())
        metrics.append({
            "seed": seed, "condition": name, "rows": len(y_test),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, labels=np.arange(27), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, pred)),
            "asr_numerator": num, "asr_denominator": den,
            "asr_percent": 100.0 * num / den if den else None,
            "max_linf": float(bounds[name].max()), "checkpoint_sha256": ckpt_sha})
        rep = classification_report(y_test, pred, labels=np.arange(27), output_dict=True, zero_division=0)
        for c in range(27):
            class_rows.append({"seed": seed, "condition": name, "class_id": c, "class_name": cfg["dataset"]["class_names"][c], **rep[str(c)]})
        mats[name] = confusion_matrix(y_test, pred, labels=np.arange(27))
        for r, yt, p, cp in zip(test_ids, y_test, pred, clean_pred):
            preds.append({"condition": name, "row_id": int(r), "label": int(yt), "prediction": int(p),
                          "clean_prediction": int(cp), "checkpoint_sha256": ckpt_sha})
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(folder / "per_class.csv", index=False)
    pd.DataFrame(preds).to_csv(folder / "predictions.csv", index=False)
    np.savez_compressed(folder / "confusion_matrices.npz", **mats)

    # checkpoint reload: fresh model from checkpoint.pt, clean logits on ALL test rows
    ref = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    return metrics, logits["Clean"], ref, attack_seconds


def checkpoint_reload_check(model_factory, folder: Path, X_test_seq, clean_logits, bs: int) -> dict:
    ck = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    fresh = model_factory()
    fresh.load_state_dict(ck["state_dict"], strict=True)
    again = ad.compute_logits(fresh, X_test_seq, bs)
    return {"rows_checked": int(len(X_test_seq)), "rows_total_test": 4500, "conditions_checked": ["Clean"],
            "comparison": "logits from a freshly constructed model loaded from checkpoint.pt vs logits of the evaluated model",
            "max_abs_difference": float(np.abs(again - clean_logits).max()),
            "bitwise_equal": bool(np.array_equal(again, clean_logits)),
            "scope": "Clean logits on all 4,500 test rows; FGSM and PGD conditions are not reloaded."}


# ------------------------------------------------------------------ run-level writes

def write_run_level(run_dir: Path, fp: dict, cfg: dict, infra_env: dict) -> None:
    """First launch only: fingerprint + exact source copies (hashed from the same bytes written)."""
    files = fp["components"]["executed_source"]["files"]
    for rel, expected in files.items():
        data = (ROOT / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{rel} changed between fingerprinting and snapshotting")
        name = "executor_snapshot.py" if rel.endswith("paper_described_adam.py") else f"executed_sources/{Path(rel).name}"
        pv.atomic_bytes(run_dir / name, data)
        if sha(run_dir / name) != expected:
            raise ValueError(f"Snapshot verification failed for {rel}")
    pv.atomic_json(run_dir / "configuration.json", cfg)
    pv.atomic_json(run_dir / "environment_actual.json", {k: v for k, v in infra_env.items() if k != "pip_freeze"})
    pv.atomic_bytes(run_dir / "environment_pip_freeze.txt", infra_env["pip_freeze"].encode())
    pv.atomic_json(run_dir / "fingerprint.json", fp)


def acquire_lock(run_dir: Path) -> Path:
    lock = run_dir / "run.lock"
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        if psutil.pid_exists(old["pid"]) and abs(psutil.Process(old["pid"]).create_time() - old["create_time"]) < 1:
            raise RuntimeError(f"Run already active as PID {old['pid']}")
        lock.rename(run_dir / f"stale-lock-{time.time_ns()}.json")
    with lock.open("x") as stream:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)
    return lock


def build_manifest(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in {"status.json", "manifest.json", "raw.log"} and not p.name.endswith(".tmp")
            and "epochs" not in p.parts}


def verify_seed(folder: Path, cfg: dict, infra=None, data=None) -> None:
    """Read-only audit of a completed seed folder."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        p = folder / name
        if not p.is_file() or sha(p) != digest:
            raise ValueError(f"Corrupted or modified artifact: {p}")
    done = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
    if done["optimizer"]["class"] != "torch.optim.Adam" or done["optimizer_step_counts"] != [1050]:
        raise ValueError("Optimizer class or step count mismatch")
    last = torch.load(folder / "epochs/epoch-25.pt", map_location="cpu", weights_only=True)
    steps = {int(v["step"].item()) for v in last["optimizer"]["state"].values()}
    if last["epoch"] != 25 or steps != {1050}:
        raise ValueError(f"Saved optimizer steps {steps} != 1050")
    import pandas as pd
    pred = pd.read_csv(folder / "predictions.csv")
    split = json.loads((folder / "split_ids.json").read_text(encoding="utf-8"))
    if len(pred) != 13500 or pred.duplicated(["condition", "row_id"]).any():
        raise ValueError("Prediction table size or uniqueness violation")
    for c in ("Clean", "FGSM", "PGD"):
        if pred[pred.condition == c].row_id.tolist() != split["test"]:
            raise ValueError(f"{c}: test row IDs misaligned")


# ------------------------------------------------------------------ seed

def run_seed(cfg, infra, data, seed: int, run_dir: Path, fp: dict, resume_allowed: bool) -> None:
    folder = run_dir / f"seed-{seed}"
    if (folder / "manifest.json").exists():
        verify_seed(folder, cfg)
        print(f"REUSED verified completed seed {seed}", flush=True)
        return
    (folder / "epochs").mkdir(parents=True, exist_ok=True)
    t = cfg["training"]
    seq_len = cfg["sequence"]["length"]
    started = time.perf_counter()
    with (folder / "raw.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(Tee(sys.stdout, stream)), pv.MemoryMonitor() as mem:
        print(f"START seed {seed} pid={os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        seed_everything(seed)
        model = infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu")
        init_hash = pv.state_hash(model)
        opt = build_optimizer(cfg, model)
        fresh = {"initial_state_hash": init_hash, "optimizer_state_entries": len(opt.state_dict()["state"]),
                 "source": "freshly constructed model; no smoke or other checkpoint loaded"}
        if fresh["optimizer_state_entries"] != 0:
            raise ValueError("Optimizer state is not empty at start")
        start_epoch, prev_seconds = 0, 0.0
        files = sorted((folder / "epochs").glob("epoch-*.pt"))
        if files:
            ck = torch.load(files[-1], map_location="cpu", weights_only=True)
            pv.check_checkpoint_fingerprint(ck, fp)
            model.load_state_dict(ck["state_dict"], strict=True)
            opt.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["torch_rng_state"])
            start_epoch, prev_seconds = ck["epoch"], ck["training_seconds"]
            print(f"RESUME seed {seed} from epoch {start_epoch}", flush=True)
        tick = time.perf_counter()
        info = {"batch_sizes_epoch_1": None}

        def on_epoch(n, loss, sizes):
            elapsed = prev_seconds + time.perf_counter() - tick
            info["batch_sizes_epoch_1"] = info["batch_sizes_epoch_1"] or sizes
            pv.atomic_save(folder / "epochs" / f"epoch-{n:02d}.pt", {
                "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "torch_rng_state": torch.get_rng_state(),
                "fingerprint_sha256": fp["combined_sha256"], "epoch": n, "seed": seed, "training_seconds": elapsed, "loss": loss})
            pv.atomic_json(folder / "status.json", {"state": "training", "seed": seed, "epoch_completed": n,
                                                    "epochs_required": t["epochs"], "loss": loss, "training_seconds": elapsed})
            print(f"EPOCH {n}/{t['epochs']} seed={seed} loss={loss:.5f} steps={len(sizes)} elapsed={elapsed:.1f}s", flush=True)

        run_epochs(model, opt, data["X"]["train"], data["y"]["train"], seed, start_epoch, t["epochs"], t["batch_size"],
                   seq_len, cfg["sequence"]["train_noise_std"], on_epoch)
        train_seconds = prev_seconds + time.perf_counter() - tick
        steps = sorted(optimizer_steps(opt))
        if steps != [t["optimizer_steps_total"]]:
            raise ValueError(f"Optimizer step counts {steps} != {t['optimizer_steps_total']}")
        pv.atomic_save(folder / "checkpoint.pt", {"state_dict": model.state_dict(), "epoch": t["epochs"],
                                                  "architecture": cfg["architecture"], "seed": seed, "claim": CLAIM})
        ckpt_sha = sha(folder / "checkpoint.pt")
        sizes = info["batch_sizes_epoch_1"] or [t["batch_size"]] * 41 + [5]
        pv.atomic_json(folder / "training_complete.json", {
            "epochs": t["epochs"], "train_rows": len(data["X"]["train"]), "batch_size": t["batch_size"],
            "steps_per_epoch": len(sizes), "batch_sizes_epoch_1": sizes, "total_optimizer_steps": steps[0],
            "optimizer_step_counts": steps, "dropout": cfg["architecture"]["dropout"],
            "optimizer": {"class": cfg["training"]["optimizer"], "runtime_class": type(opt).__module__ + "." + type(opt).__qualname__,
                          "is_adamw": isinstance(opt, torch.optim.AdamW),
                          "defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}},
            "fresh_initialization": fresh, "checkpoint_rule": t["checkpoint_rule"],
            "fingerprint_sha256": fp["combined_sha256"], "training_seconds": train_seconds})
        X_test_seq = to_sequence(data["X"]["test"], seq_len)
        y_test, test_ids = data["y"]["test"], data["split"]["test"]
        try:
            metrics, clean_logits, _, attack_seconds = evaluate_seed(cfg, model, X_test_seq, y_test, test_ids, folder, ckpt_sha, seed)
        except ad.InfeasibleConstraintError:
            raise
        reload_rec = checkpoint_reload_check(lambda: infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu"),
                                             folder, X_test_seq, clean_logits, cfg["attack"]["attack_batch_size"])
        pv.atomic_json(folder / "verification.json", {
            "epsilon_zero_identity_tested": True, "evaluation_mode": "eval", "evaluation_weights_unchanged": True,
            "same_examples_all_conditions": True, "test_rows_evaluated": int(len(y_test)),
            "checkpoint_sha256": ckpt_sha, "checkpoint_reload": reload_rec})
        pv.atomic_json(folder / "runtime.json", {
            "training_seconds": train_seconds, "attack_seconds": attack_seconds,
            "invocation_seconds": time.perf_counter() - started, "peak_rss_sampled_bytes": mem.peak})
        print("SEED", seed, "METRICS", json.dumps(metrics), flush=True)
    for name in ("split_ids.json", "split_overlap_audit.json", "preprocessing.json", "preprocessing.npz", "data_range_report.json"):
        src = run_dir / name
        if src.exists() and not (folder / name).exists():
            (folder / name).write_bytes(src.read_bytes())
    pv.atomic_json(folder / "manifest.json", build_manifest(folder))
    verify_seed(folder, cfg)
    pv.atomic_json(folder / "status.json", {"state": "complete", "seed": seed, "verified": True})
    print(f"COMPLETED AND VERIFIED SEED {seed}", flush=True)


# ------------------------------------------------------------------ smoke

def run_smoke(cfg, data, infra, smoke_dir: Path, fp: dict) -> None:
    sel = select_smoke_ids(data["split"], data["labels"])
    pv.atomic_json(smoke_dir / "smoke_selection.json", sel)  # before the data is used
    ids_tr, ids_va = np.array(sel["train_ids"]), np.array(sel["validation_ids"])
    scaler = data["scaler"]  # fit on the FULL saved training partition (training-only)
    X_tr = scaler.transform(data["values"][ids_tr]).astype(np.float32)
    X_va = scaler.transform(data["values"][ids_va]).astype(np.float32)
    y_tr, y_va = data["labels"][ids_tr], data["labels"][ids_va]
    eps = cfg["attack"]["fgsm"]["epsilon"]
    feasible = np.where(~(((X_va - np.float32(eps)) > 1.0).any(axis=1) | ((X_va + np.float32(eps)) < 0.0).any(axis=1)))[0]
    if len(feasible) < 10:
        raise ValueError("fewer than 10 feasible smoke validation rows")
    diag = feasible[:10]
    sel["diagnostic_validation_ids"] = [int(ids_va[i]) for i in diag]
    sel["excluded_infeasible_validation_rows"] = int(len(ids_va) - len(feasible))
    pv.atomic_json(smoke_dir / "smoke_selection.json", sel)

    seed_everything(1)
    model = infra.StandardTransformerIDS(**cfg["architecture"]).to("cpu")
    opt = build_optimizer(cfg, model)
    model.train()
    Xs = to_sequence(X_tr)
    loss = F.cross_entropy(model(torch.from_numpy(Xs))["logits"], torch.from_numpy(y_tr.astype(np.int64)))
    opt.zero_grad(); loss.backward(); opt.step()
    ck = smoke_dir / "smoke_chk.pt"
    pv.atomic_save(ck, {"state_dict": model.state_dict()})
    h = pv.state_hash(model)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["state_dict"])
    assert pv.state_hash(model) == h
    xv = to_sequence(X_va[diag]); yv = y_va[diag]
    a = cfg["attack"]
    assert np.array_equal(ad.fgsm(model, xv, yv, 0.0), xv)
    assert np.array_equal(ad.pgd(model, xv, yv, 0.0, 0.025, 7, 1), xv)
    f = ad.fgsm(model, xv, yv, eps, 10)
    p1 = ad.pgd(model, xv, yv, eps, a["pgd"]["alpha"], a["pgd"]["iterations"], 1, 10)
    p2 = ad.pgd(model, xv, yv, eps, a["pgd"]["alpha"], a["pgd"]["iterations"], 1, 10)
    assert np.array_equal(p1, p2) and np.abs(f - xv).max() <= eps + 1e-6 and np.abs(p1 - xv).max() <= eps + 1e-6
    assert pv.state_hash(model) == h and not model.training
    try:
        ad.fgsm(model, np.full((1, 50, 39), 1.5, dtype=np.float32), np.array([0]), eps)
        raise AssertionError("infeasible tensor was accepted")
    except ad.InfeasibleConstraintError:
        pass
    pv.atomic_json(smoke_dir / "smoke_checks_passed.json", {
        "status": "all smoke checks passed", "selection_train_sha256": sel["train_ids_sha256"],
        "selection_validation_sha256": sel["validation_ids_sha256"], "fingerprint_sha256": fp["combined_sha256"],
        "note": "Smoke weights and optimizer state are never used by the full run."})
    print("ALL SMOKE CHECKS PASSED", flush=True)


# ------------------------------------------------------------------ main

def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scan-out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        cfg = read_adam_config(args.config)
        run_dir = (args.run_dir or (SMOKE_RUN if args.smoke else DEFAULT_RUN)).resolve()
        if not args.smoke and _inside(run_dir, SMOKE_RUN.resolve()):
            raise ValueError("The full run may not use the smoke directory")
        seeds = args.seeds or cfg["seeds"]
        if len(set(seeds)) != len(seeds) or any(s not in cfg["seeds"] for s in seeds):
            raise ValueError("Use only the frozen replication seeds")
        fp = pv.compute_fingerprint(cfg, ROOT)
    except Exception as exc:  # fatal validation: nothing has been written
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    try:
        pv.check_resume(run_dir, fp)
        if not args.smoke and not args.resume and not args.verify_only and not args.dry_run:
            for s in seeds:
                sd = run_dir / f"seed-{s}"
                if sd.exists() and any(sd.iterdir()) and not (sd / "manifest.json").exists():
                    raise pv.ResumeIncompatibleError(f"{sd} exists and is incomplete; use --resume")
    except pv.ResumeIncompatibleError as exc:
        print(f"REFUSED (exit 3): {exc}", file=sys.stderr)
        return 3

    try:
        if args.dry_run:
            data = build_data(cfg)
            scan = domain_scan(data, cfg["attack"]["fgsm"]["epsilon"])
            out = {"fingerprint_combined_sha256": fp["combined_sha256"], "data_range": data_range_report(data), "domain_scan": scan}
            print(json.dumps(out, indent=2))
            if args.scan_out:
                pv.atomic_json(args.scan_out, out)
            return 0
        if args.verify_only:
            for s in seeds:
                verify_seed(run_dir / f"seed-{s}", cfg)
            print("All requested seed bundles verified")
            return 0

        run_dir.mkdir(parents=True, exist_ok=True)
        lock = acquire_lock(run_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    try:
        torch.set_num_threads(cfg["resources"]["torch_threads"])
        if not (run_dir / "fingerprint.json").exists():
            write_run_level(run_dir, fp, cfg, pv.environment_record(ROOT, git))
        data = build_data(cfg)
        if args.smoke:
            infra = pv.import_upstream_snapshot(run_dir, ROOT, PIN, DATA_SHA, git, sha)
            run_smoke(cfg, data, infra, run_dir, fp)
            return 0
        infra = pv.import_upstream_snapshot(run_dir, ROOT, PIN, DATA_SHA, git, sha)
        split_raw = data["split"]["raw"]
        if not (run_dir / "split_ids.json").exists():
            (run_dir / "split_ids.json").write_bytes((ROOT / cfg["split"]["source"]).read_bytes())
        elif sha(run_dir / "split_ids.json") != cfg["split"]["sha256"]:
            raise ValueError("Run-directory split_ids.json differs from the saved split")
        pv.atomic_json(run_dir / "split_overlap_audit.json", pv.audit_splits_saved(
            data["values"], data["labels"], data["split"]["train"], data["split"]["validation"], data["split"]["test"]))
        pv.atomic_json(run_dir / "data_range_report.json", data_range_report(data))
        sc = data["scaler"]
        pv.atomic_json(run_dir / "preprocessing.json", {
            "type": "MinMaxScaler", "fit_scope": "training partition only", "train_rows_seen": int(sc.n_samples_seen_),
            "constant_features": int((sc.data_min_ == sc.data_max_).sum()),
            "domain_policy": "scaler fit on training partition only; no clamping of held-out values; attacks use the "
                             "intersection of the epsilon ball with [0,1] (see config attack.domain_policy)",
            "split_row_counts": [len(split_raw["train"]), len(split_raw["validation"]), len(split_raw["test"])]})
        np.savez_compressed(run_dir / "preprocessing.npz", data_min=sc.data_min_, data_max=sc.data_max_, scale=sc.scale_, min=sc.min_)
        for s in seeds:
            run_seed(cfg, infra, data, s, run_dir, fp, args.resume)
        pv.atomic_json(run_dir / "requested_run_complete.json", {"seeds_completed_this_request": seeds, "claim": CLAIM})
        return 0
    except ad.InfeasibleConstraintError as exc:
        print(f"INFEASIBLE (exit 6): {exc}", file=sys.stderr)
        pv.atomic_json(run_dir / "status.json", {"state": "failed", "reason": str(exc)})
        return 6
    except (ValueError, PartitionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
