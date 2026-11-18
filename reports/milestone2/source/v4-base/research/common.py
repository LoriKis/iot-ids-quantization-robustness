from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "PAVE_artifact"
PIN = "e73911bbdb241f3fb6c5de346139010c4aadefd4"
DATA_SHA = "0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp names also avoid two callers sharing a partial file.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


IDENTITY_NAME = "PACKAGE_IDENTITY.json"
_PACKAGED_STATUS = "PACKAGED: see PACKAGE_IDENTITY.json"
_PAVE_VERIFIED = {}  # in-process cache: resolved upstream path -> True once verify_pave_portable succeeded


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_tree_id(entries: dict) -> str:
    """Root tree object id from {relative posix path: (mode, blob id)}; modes as in `git ls-tree` (100644, 100755)."""
    def build(items):
        children = {}
        files = {}
        for path, (mode, blob) in items.items():
            head, _, rest = path.partition("/")
            if rest:
                children.setdefault(head, {})[rest] = (mode, blob)
            else:
                files[head] = (mode, blob)
        records = []
        for name, (mode, blob) in files.items():
            records.append((name, mode.encode(), bytes.fromhex(blob)))
        for name, sub in children.items():
            records.append((name, b"40000", bytes.fromhex(build(sub))))
        # git orders tree entries by name with a trailing "/" for directories
        dirs = set(children)
        records.sort(key=lambda r: r[0].encode() + (b"/" if r[0] in dirs else b""))
        body = b"".join(mode + b" " + name.encode() + b"\0" + raw for name, mode, raw in records)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()
    return build(entries)


def verify_pave_portable(upstream=None, identity=None, use_cache=True):
    """Stdlib check that the packaged PAVE source equals the identity recorded from the original repository.

    identity["pave"] = {commit, tree_id, files: {relpath: {mode, git_blob_id, sha256}}}. Verifies the pinned commit,
    each file's SHA-256, each git blob id (retrying with CRLF->LF, the source checkout used autocrlf) and the root tree
    object id rebuilt from the blob ids and recorded modes. Raises RuntimeError on any difference.
    """
    upstream = Path(upstream or UPSTREAM)
    key = str(upstream.resolve())
    if use_cache and _PAVE_VERIFIED.get(key):
        return True
    if identity is None:
        identity = json.loads((ROOT / IDENTITY_NAME).read_text(encoding="utf-8"))
    pave = identity["pave"]
    if pave["commit"] != PIN:
        raise RuntimeError(f"recorded PAVE commit {pave['commit']} is not the pin {PIN}")
    entries = {}
    for rel, rec in sorted(pave["files"].items()):
        if rec["mode"] not in ("100644", "100755"):
            raise RuntimeError(f"PAVE file {rel} has unsupported git mode {rec['mode']}")
        path = upstream / rel
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"PAVE file missing or not a regular file: {rel}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != rec["sha256"]:
            raise RuntimeError(f"PAVE file SHA-256 differs from the recorded value: {rel}")
        if git_blob_id(data) != rec["git_blob_id"] and git_blob_id(data.replace(b"\r\n", b"\n")) != rec["git_blob_id"]:
            raise RuntimeError(f"PAVE file git blob id differs from the recorded value: {rel}")
        entries[rel] = (rec["mode"], rec["git_blob_id"])
    if git_tree_id(entries) != pave["tree_id"]:
        raise RuntimeError("PAVE tree id rebuilt from the recorded blobs differs from the recorded tree id")
    present = {p.relative_to(upstream).as_posix() for p in upstream.rglob("*") if p.is_file() and ".git" not in p.parts
               and "__pycache__" not in p.parts}
    extra = sorted(present - set(pave["files"]))
    if extra:
        raise RuntimeError(f"PAVE directory holds files that are not in the recorded identity: {extra[:5]}")
    _PAVE_VERIFIED[key] = True
    return True


def _packaged() -> bool:
    return not (ROOT / ".git").exists() and (ROOT / IDENTITY_NAME).is_file()


def _packaged_git(args, cwd):
    """Fixed answer table for the git-free VM package (DESIGN 7.4). Anything else raises."""
    identity = json.loads((ROOT / IDENTITY_NAME).read_text(encoding="utf-8"))
    cwd = Path(cwd).resolve()
    if cwd == ROOT.resolve():
        if args == ("rev-parse", "HEAD"):
            return identity["source"]["head"]
        if args == ("branch", "--show-current"):
            return identity["source"]["branch"]
        if args in (("status", "--porcelain"), ("status", "--porcelain", "--untracked-files=all")):
            return _PACKAGED_STATUS
    elif cwd == UPSTREAM.resolve():
        if args in (("rev-parse", "HEAD"), ("status", "--porcelain", "--untracked-files=all")):
            verify_pave_portable(identity=identity)
            return PIN if args == ("rev-parse", "HEAD") else ""
    raise RuntimeError("git is not available in the VM package")


def git(*args, cwd=ROOT):
    if _packaged():
        return _packaged_git(args, cwd)
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def verify_upstream():
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != PIN:
        raise RuntimeError(f"PAVE revision must be {PIN}")
    if git("status", "--porcelain", "--untracked-files=all", cwd=UPSTREAM):
        raise RuntimeError("PAVE checkout is dirty; preserve it and inspect manually")
    return sha256(UPSTREAM / "code/models/standard_transformer.py")


def make_model(config):
    # Loading this single file avoids upstream package imports with side effects.
    # Disable bytecode generation to preserve the reference checkout.
    path = UPSTREAM / "code/models/standard_transformer.py"
    spec = importlib.util.spec_from_file_location("pave_standard_reference", path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module.StandardTransformerIDS(**config)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Static Linear islands narrowly exempt their unsupported INT8 kernel;
    # all float training and attack operations retain the strict guard.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import pandas
    import scipy
    import sklearn
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": psutil.disk_usage(str(ROOT)).free,
        "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pandas.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "quantized_engines": torch.backends.quantized.supported_engines,
        "threads": torch.get_num_threads(), "device": "cpu",
        "determinism_policy": "seed all RNGs; strict float deterministic algorithms; static Linear kernel temporarily exempted because torch 2.5.1 quantized_resize_cpu_ is unsupported under the guard",
    }


def provenance():
    dirty = git("status", "--porcelain", "--untracked-files=all")
    sources = {}
    for folder in ("research", "configs", "scripts", "tests"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                sources[p.relative_to(ROOT).as_posix()] = sha256(p)
    return {"revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty_status": dirty, "source_hashes": sources, "source_digest": digest(sources),
            "upstream_revision": PIN, "model_source_hash": verify_upstream()}


def runtime_signature(env):
    """Material runtime versions, excluding volatile RAM/disk snapshots."""
    return {key: env[key] for key in
            ("python", "torch", "numpy", "pandas", "scipy", "sklearn", "device")}


@contextlib.contextmanager
def unit_status(folder, identity, resume=False):
    folder = Path(folder)
    status = folder / "status.json"
    if folder.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {folder}; use --resume")
        if not status.exists():
            raise RuntimeError(f"Incomplete unit without status: {folder}")
        previous = json.loads(status.read_text())
        if previous["identity"] != identity:
            raise RuntimeError(f"Incompatible resume: {folder}")
        if previous["status"] == "completed":
            yield False
            return
        # Never overwrite failed/partial work. Explicit new run is needed.
        raise RuntimeError(f"Unit {folder} is {previous['status']}; use a new run directory")
    folder.mkdir(parents=True)
    start = time.time()
    atomic_json(status, {"status": "running", "identity": identity, "started_unix": start})
    try:
        yield True
    except BaseException as exc:
        atomic_json(status, {"status": "failed", "identity": identity,
                            "elapsed_seconds": time.time() - start, "error": repr(exc)})
        raise
    else:
        atomic_json(status, {"status": "completed", "identity": identity,
                            "elapsed_seconds": time.time() - start})


def state_bytes(model):
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return len(stream.getvalue())
