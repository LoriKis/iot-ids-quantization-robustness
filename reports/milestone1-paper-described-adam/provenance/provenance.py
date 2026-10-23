"""Provenance, fingerprints and the resume-compatibility gate for the corrected Adam run.

atomic_json, atomic_save, state_hash and MemoryMonitor are COPIED from
research/paper_described_baseline.py (historical, untouched). audit_splits_saved is a
re-implementation of its audit_splits that takes the saved index arrays.

compute_fingerprint() is read-only and in-memory: it writes nothing anywhere.
check_resume() must run before any write in a launch; it reads only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

EXECUTED_SOURCE_FILES = [
    "research/paper_described_adam.py",
    "research/adversarial_domain.py",
    "research/provenance.py",
    "research/milestone1.py",
]
EXIT_INCOMPATIBLE = 3


def file_sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path | str, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_bytes(path: Path | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def atomic_save(path: Path | str, value: object) -> None:
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


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, val in model.state_dict().items():
        h.update(key.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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


# ---------------------------------------------------------------- data helpers

def load_dataset(csv_path: Path | str, cfg: dict):
    """Read the CSV; any non-finite raw value is FATAL (no imputation in the corrected run)."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["Label"].unique().tolist())
    if classes != cfg["dataset"]["class_names"]:
        raise ValueError("Class names differ from the configuration")
    from sklearn.preprocessing import LabelEncoder
    labels = LabelEncoder().fit_transform(df["Label"])
    numeric = df.drop(columns="Label").apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        r, c = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"Non-finite raw value at row {r}, column {numeric.columns[c]}; imputation is not part of this protocol")
    if list(numeric.columns) != cfg["dataset"]["feature_names"]:
        raise ValueError("Feature names differ from the configuration")
    return df, values, labels, classes


def load_saved_split(root: Path, cfg: dict) -> dict:
    path = root / cfg["split"]["source"]
    if file_sha(path) != cfg["split"]["sha256"]:
        raise ValueError(f"Saved split hash mismatch for {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    tr, va, te = (np.array(split[k], dtype=np.int64) for k in ("train", "validation", "test"))
    if (len(tr), len(va), len(te)) != (20997, 4500, 4500):
        raise ValueError("Saved split sizes are not 20997/4500/4500")
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise ValueError("Saved split partitions overlap")
    return {"train": tr, "validation": va, "test": te, "raw": split}


def fit_scaler(values: np.ndarray, train_idx: np.ndarray):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(values[train_idx])
    return scaler


def audit_splits_saved(values: np.ndarray, labels: np.ndarray, train_idx, val_idx, test_idx) -> dict:
    rows = [values[i].tobytes() for i in range(len(values))]
    train_set = {rows[i] for i in train_idx}
    val_set = {rows[i] for i in val_idx}
    labeled_train = {(rows[i], int(labels[i])) for i in train_idx}
    t_in_tr = sum(rows[i] in train_set for i in test_idx)
    v_in_tr = sum(rows[i] in train_set for i in val_idx)
    t_in_va = sum(rows[i] in val_set for i in test_idx)
    t_in_tr_l = sum((rows[i], int(labels[i])) in labeled_train for i in test_idx)
    return {
        "train_rows": len(train_idx), "val_rows": len(val_idx), "test_rows": len(test_idx),
        "unique_feature_patterns_total": len(set(rows)),
        "test_rows_matching_train_features": t_in_tr,
        "test_rows_matching_train_features_percent": 100.0 * t_in_tr / len(test_idx),
        "test_rows_matching_train_labeled": t_in_tr_l,
        "test_rows_matching_train_labeled_percent": 100.0 * t_in_tr_l / len(test_idx),
        "val_rows_matching_train_features": v_in_tr,
        "val_rows_matching_train_features_percent": 100.0 * v_in_tr / len(val_idx),
        "test_rows_matching_validation_features": t_in_va,
        "note": "Row IDs are disjoint by construction. Overlap is MEASURED; its origin is not established by this audit.",
    }


# ---------------------------------------------------------------- upstream

def upstream_tree_hash(upstream_root: Path) -> str:
    """sha256 over sorted 'relpath NUL sha256(bytes)' lines for code/** (no __pycache__) plus LICENSE."""
    upstream_root = Path(upstream_root)
    entries = []
    for p in sorted((upstream_root / "code").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            entries.append((p.relative_to(upstream_root).as_posix(), file_sha(p)))
    lic = upstream_root / "LICENSE"
    if lic.is_file():
        entries.append(("LICENSE", file_sha(lic)))
    h = hashlib.sha256()
    for rel, digest in sorted(entries):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def import_upstream_snapshot(run_root: Path, root: Path, pin: str, data_sha: str, git, sha):
    """Verify pinned upstream, copy code into run_root/reference, re-hash the copy, import it."""
    upstream = root / "upstream/PAVE_artifact"
    if git("rev-parse", "HEAD", cwd=upstream) != pin or git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
        raise ValueError("Pinned upstream changed")
    if sha(upstream / "data/balanced_CICIoT.csv") != data_sha:
        raise ValueError("Dataset hash changed")
    reference = run_root / "reference"
    if not reference.exists():
        shutil.copytree(upstream / "code", reference / "code", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(upstream / "LICENSE", reference / "LICENSE")
    if upstream_tree_hash(reference) != upstream_tree_hash(upstream):
        raise ValueError("Upstream snapshot copy differs from the pinned upstream")
    sys.dont_write_bytecode = True
    if str(reference / "code") not in sys.path:
        sys.path.insert(0, str(reference / "code"))
    os.environ["AGT_OUT"] = str(run_root / "upstream_auxiliary")
    infra = importlib.import_module("sota_compare")
    infra.BAL = str(upstream / "data")
    infra.DEV = "cpu"
    return infra


def import_upstream_readonly(root: Path):
    """Import the upstream module directly from the pinned tree (smoke/tests); writes nothing."""
    upstream = root / "upstream/PAVE_artifact"
    sys.dont_write_bytecode = True
    code = str(upstream / "code")
    if code not in sys.path:
        sys.path.insert(0, code)
    os.environ.setdefault("AGT_OUT", tempfile.gettempdir())
    infra = importlib.import_module("sota_compare")
    infra.BAL = str(upstream / "data")
    infra.DEV = "cpu"
    return infra


# ---------------------------------------------------------------- environment

def _version(name: str):
    try:
        return importlib.import_module(name).__version__
    except Exception:  # pragma: no cover - optional packages
        return None


def environment_subset() -> dict:
    import sklearn
    return {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "system": platform.system(), "machine": platform.machine(),
    }


def environment_record(root: Path, git) -> dict:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    return {
        **environment_subset(),
        "executable": sys.executable, "platform": platform.platform(), "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total, "torch_cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(), "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "scipy": _version("scipy"), "psutil": psutil.__version__, "matplotlib": _version("matplotlib"),
        "pytest": _version("pytest"), "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "pip_freeze": freeze, "git_head": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain"),
    }


# ---------------------------------------------------------------- fingerprint + gate

class ResumeIncompatibleError(RuntimeError):
    """Launch refused by the fingerprint gate (exit code 3). Raised before any write."""


def compute_fingerprint(cfg: dict, root: Path, with_scaler: bool = True) -> dict:
    """Read-only, in-memory fingerprint. Writes nothing."""
    root = Path(root)
    files = {rel: file_sha(root / rel) for rel in EXECUTED_SOURCE_FILES}
    executed = {"files": files, "sha256": canonical_hash(sorted(files.items()))}
    components = {
        "config": {"sha256": canonical_hash(cfg)},
        "executed_source": executed,
        "upstream_source": {"sha256": upstream_tree_hash(root / "upstream/PAVE_artifact")},
    }
    data = {"dataset_sha256": file_sha(root / cfg["dataset"]["path"]), "split_sha256": None}
    split = load_saved_split(root, cfg)  # verifies cfg split hash before use
    data["split_sha256"] = cfg["split"]["sha256"]
    if with_scaler:
        _, values, _, _ = load_dataset(root / cfg["dataset"]["path"], cfg)
        scaler = fit_scaler(values, split["train"])
        data["scaler_min_sha256"] = hashlib.sha256(np.ascontiguousarray(scaler.min_).tobytes()).hexdigest()
        data["scaler_scale_sha256"] = hashlib.sha256(np.ascontiguousarray(scaler.scale_).tobytes()).hexdigest()
    data["sha256"] = canonical_hash(data)
    components["data"] = data
    env = environment_subset()
    components["environment"] = {"values": env, "sha256": canonical_hash(env)}
    combined = canonical_hash({k: v["sha256"] for k, v in sorted(components.items())})
    return {"components": components, "combined_sha256": combined}


def _has_content(run_dir: Path) -> bool:
    return run_dir.exists() and any(run_dir.iterdir())


def check_resume(run_dir: Path | str, current: dict) -> None:
    """Read-only gate. Raises ResumeIncompatibleError (-> exit 3) for every refusal."""
    run_dir = Path(run_dir)
    if not _has_content(run_dir):
        return
    fp_path = run_dir / "fingerprint.json"
    if not fp_path.is_file():
        raise ResumeIncompatibleError(f"{run_dir} has content but no fingerprint.json; start a new run directory")
    try:
        stored = json.loads(fp_path.read_text(encoding="utf-8"))
        stored_components = stored["components"]
        differing = [
            f"{name}: stored {stored_components[name]['sha256']} != current {comp['sha256']}"
            for name, comp in current["components"].items()
            if stored_components[name]["sha256"] != comp["sha256"]
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise ResumeIncompatibleError(f"{fp_path} is unreadable or incomplete ({exc!r})") from exc
    if differing:
        raise ResumeIncompatibleError("Incompatible resume; differing components: " + "; ".join(differing))


def check_checkpoint_fingerprint(checkpoint: dict, current: dict) -> None:
    if checkpoint.get("fingerprint_sha256") != current["combined_sha256"]:
        raise ResumeIncompatibleError("Checkpoint fingerprint differs from the current fingerprint")
