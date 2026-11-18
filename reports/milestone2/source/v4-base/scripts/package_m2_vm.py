#!/usr/bin/env python3
"""Build the self-contained Milestone 2 VM transfer folder beside the repository (stdlib only; run with `python -B`).

    python -B scripts/package_m2_vm.py [--parent DIR] [--name IOT-network-traffic-DL-M2-VM] [--result-json PATH]

The folder is staged as `<name>.partial-<utc>`, verified, and renamed to `<name>` (or `<name>-v2`, `-v3`, ...) only when
every check passes. Nothing is overwritten and nothing is deleted. Trained weights are not packaged: they are generated
on the VM. The packager copies the working tree (including uncommitted and untracked Milestone 2 files), never `.git`.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # never leave __pycache__ in a staged package
BUILDER_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "IOT-network-traffic-DL-M2-VM"
PAVE_REL = "upstream/PAVE_artifact"
DATASET_REL = PAVE_REL + "/data/balanced_CICIoT.csv"
DATASET_SHA = "0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed"
DATASET_SIZE = 5444531
MANIFEST, IDENTITY, RUNBOOK = "MANIFEST.sha256", "PACKAGE_IDENTITY.json", "VM_RUNBOOK.md"
WEIGHTS_STATEMENT = ("Trained weights are generated on the VM by the M2 training stages; none are packaged and none are "
                     "missing downloads. The model architecture/source is upstream/PAVE_artifact/code/models/standard_transformer.py.")

# ---------------------------------------------------------------------------------------------------- include/exclude
TOP_DIRS = ("research", "tests", "configs", "scripts", "licenses", "notes", "docs", "reports")
TOP_FILES = ("requirements.txt", "requirements-lock-windows.txt", "requirements-linux-cpu.txt", "pytest.ini",
             ".gitattributes", ".gitignore", "README.md", "NOTICE.md", "PROGRESS.md", "provenance.json")
ARTIFACT_DIRS = ("artifacts/m2a-pilot", "artifacts/m2a-integrity", "artifacts/m2b-integrity")
ONLY_SUFFIX = {"research": (".py",), "tests": (".py",)}  # DESIGN 7.2: research/**/*.py and tests/**/*.py
EXCLUDE_DIRS = ("/.git/", "/upstream/PAVE_artifact/.git/", "/.venv/", "/.venv_pave_m1/", "/.kiro/", "/.agents/",
                "/.pytest_cache/", "/work/", "/runs/", "/experiments/", "/output/", "/artifacts/m2a-pilot/models/",
                "/artifacts/m2a-pilot-attempt1-report-bug/", "/artifacts/pilot-estimate/",
                "/artifacts/pilot-verified/", "/artifacts/smoke-verified/")
EXCLUDE_DIRNAMES = ("__pycache__",)
EXCLUDE_BASENAMES = ("*.pyc", "*.pt", "*.pth", "*.ckpt", "*.zip", "*.7z", "*.tar.gz", "*.log", ".env", ".env.*", "*.pem",
                     "*.key", "*.p12", "*.pfx", "credentials.json", "secrets.json", "*.local.json", "*.local.toml",
                     ".DS_Store", "Thumbs.db", "desktop.ini")
EXCLUDE_PATHS = ("/scripts/*.ps1", "/requirements-teammate-mac.txt", "/docs/environment-windows.json")
TEXT_SUFFIXES = (".py", ".json", ".md", ".txt", ".sh", ".ini", ".cfg", ".yml", ".yaml", ".toml", ".sha256")
ACTIVE_PREFIXES = ("research/", "tests/", "scripts/", "configs/", "requirements", "pytest.ini", ".gitattributes",
                   RUNBOOK)
LF_FILES_GLOBS = ("scripts/vm/*.sh", RUNBOOK, "requirements-linux-cpu.txt", MANIFEST)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def excluded(rel: str) -> bool:
    """Anchored exclusion of a repository-relative posix path (applied outside the PAVE subtree only)."""
    anchored = "/" + rel
    if any(anchored.startswith(d) or (anchored + "/").startswith(d) for d in EXCLUDE_DIRS):
        return True
    parts = rel.split("/")
    if any(part in EXCLUDE_DIRNAMES for part in parts[:-1]):
        return True
    if any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in EXCLUDE_BASENAMES):
        return True
    return any(fnmatch.fnmatchcase(anchored, pattern) and anchored.count("/") == pattern.count("/") for pattern in EXCLUDE_PATHS)


def choose_target(parent: Path, name: str) -> Path:
    candidate, index = parent / name, 1
    while candidate.exists() or os.path.lexists(candidate):
        index += 1
        candidate = parent / f"{name}-v{index}"
    return candidate


def list_include(root: Path) -> list:
    """Repository-relative posix paths of the non-PAVE include list, after exclusions."""
    found = set()
    for top in TOP_FILES:
        if (root / top).is_file():
            found.add(top)
    for top in TOP_DIRS + ARTIFACT_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        suffixes = ONLY_SUFFIX.get(top)
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not (here / d).is_symlink()
                           and not excluded((here / d).relative_to(root).as_posix() + "/x")]
            for name in filenames:
                path = here / name
                rel = path.relative_to(root).as_posix()
                if path.is_symlink() or excluded(rel):
                    continue
                if suffixes and not name.endswith(suffixes):
                    continue
                found.add(rel)
    return sorted(found)


def pave_listing(root: Path) -> dict:
    """{relpath: {mode, git_blob_id}} from `git ls-tree -r HEAD`, plus checks that the PAVE tree is clean and complete."""
    pave = root / PAVE_REL
    if run_git(pave, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("PAVE working tree is modified or has untracked files; refusing to package")
    raw = subprocess.check_output(["git", "-C", str(pave), "ls-tree", "-r", "-z", "HEAD"]).decode("utf-8")
    files = {}
    for entry in filter(None, raw.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise SystemExit(f"PAVE entry {path} has unsupported git mode/type {mode} {kind}")
        files[path] = {"mode": mode, "git_blob_id": blob}
    present = {p.relative_to(pave).as_posix() for p in pave.rglob("*") if p.is_file() and ".git" not in p.relative_to(pave).parts}
    if present != set(files):
        raise SystemExit(f"PAVE working tree differs from ls-tree: extra {sorted(present - set(files))[:3]} missing {sorted(set(files) - present)[:3]}")
    for rel in files:
        if (pave / rel).is_symlink():
            raise SystemExit(f"PAVE file is a symlink: {rel}")
    return files


# ------------------------------------------------------------------------------------------------------------ scans
def windows_path_patterns():
    """Built at run time so this file does not match its own scan."""
    drive = "(?<![A-Za-z0-9_])[A-Za-z]" + ":" + "[" + "\\\\" + "/" + "]"
    user = "Users" + "[" + "\\\\" + "/" + "]" + "lo" + "ran"
    return re.compile(drive), re.compile(user)


def is_active(rel: str) -> bool:
    return any(rel.startswith(p) for p in ACTIVE_PREFIXES) and not rel.startswith("upstream/")


def scan_windows_paths(tree: Path, files=None) -> dict:
    drive, user = windows_path_patterns()
    active, historical = [], []
    for rel in (files if files is not None else walk_regular(tree)):
        if rel.startswith("upstream/") or not rel.endswith(TEXT_SUFFIXES) or rel in (IDENTITY, MANIFEST):
            continue
        try:
            text = (tree / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if drive.search(text) or user.search(text):
            (active if is_active(rel) else historical).append(rel)
    return {"active": sorted(active), "historical": sorted(historical)}


def walk_regular(tree: Path, skip_top=("output", ".venv")) -> list:
    out = []
    for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
        here = Path(dirpath)
        if here == tree:
            dirnames[:] = [d for d in dirnames if d not in skip_top]
        for name in filenames:
            out.append((here / name).relative_to(tree).as_posix())
    return sorted(out)


def find_links(tree: Path) -> list:
    links = []
    for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            attrs = getattr(os.lstat(path), "st_file_attributes", 0)
            if os.path.islink(path) or attrs & 0x400:
                links.append(path.relative_to(tree).as_posix())
    return sorted(links)


def find_crlf(tree: Path) -> list:
    targets = set()
    for pattern in LF_FILES_GLOBS:
        targets.update(p.relative_to(tree).as_posix() for p in tree.glob(pattern) if p.is_file())
    return sorted(rel for rel in targets if b"\r" in (tree / rel).read_bytes())


def find_forbidden(tree: Path, exempt_prefix=PAVE_REL + "/") -> list:
    bad = []
    for rel in walk_regular(tree):
        if rel.startswith(exempt_prefix):
            continue
        if excluded(rel) or any(part in (".git", ".venv") for part in rel.split("/")):
            bad.append(rel)
    return bad


def compile_all(tree: Path) -> dict:
    failures, count = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for rel in walk_regular(tree):
            if rel.endswith(".py") and not rel.startswith("upstream/"):
                count += 1
                try:
                    py_compile.compile(str(tree / rel), cfile=str(Path(tmp) / f"{count}.pyc"), doraise=True)
                except py_compile.PyCompileError as exc:
                    failures.append(f"{rel}: {exc.msg.splitlines()[-1]}")
    leftovers = [r for r in walk_regular(tree) if "__pycache__" in r.split("/")]
    return {"compiled": count, "failures": failures, "pycache": leftovers}


def write_manifest(tree: Path) -> int:
    entries = [(rel, sha256_file(tree / rel)) for rel in walk_regular(tree) if rel != MANIFEST]
    entries.sort(key=lambda e: e[0].encode("utf-8"))
    text = "".join(f"{digest}  {rel}\n" for rel, digest in entries)
    (tree / MANIFEST).write_bytes(text.encode("utf-8"))
    return len(entries)


def verify_manifest(tree: Path) -> dict:
    sys.path.insert(0, str(tree / "scripts" / "vm"))
    try:
        import importlib
        mod = importlib.import_module("verify_package")
        importlib.reload(mod)
        return mod.check_manifest(tree)
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------------------------------------------------- identity
def modified_paths(root: Path) -> dict:
    out = {}
    for line in run_git(root, "status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        full = root / path
        out[path] = sha256_file(full) if full.is_file() else None
    return dict(sorted(out.items()))


def build_identity(root: Path, include: list, pave: dict, scans: dict, strict: bool) -> dict:
    identity = {"schema": 1, "builder_version": BUILDER_VERSION, "created_utc": utc_stamp(), "note": WEIGHTS_STATEMENT,
                "source": {"head": "unknown", "branch": "unknown", "modified_or_untracked_at_packaging": {}}}
    if strict:
        identity["source"] = {"head": run_git(root, "rev-parse", "HEAD"), "branch": run_git(root, "branch", "--show-current"),
                              "modified_or_untracked_at_packaging": modified_paths(root)}
    if pave:
        pave_dir = root / PAVE_REL
        files = {rel: {"mode": meta["mode"], "git_blob_id": meta["git_blob_id"], "sha256": sha256_file(pave_dir / rel)}
                 for rel, meta in sorted(pave.items())}
        identity["pave"] = {"commit": run_git(pave_dir, "rev-parse", "HEAD"), "tree_id": run_git(pave_dir, "rev-parse", "HEAD^{tree}"),
                            "files": files, "forbidden_name_exempt": sorted(r for r in files if excluded_basename(r))}
        ds = root / DATASET_REL
        identity["dataset"] = {"path": DATASET_REL, "frozen_sha256": DATASET_SHA, "frozen_size": DATASET_SIZE,
                               "observed_sha256_in_source": sha256_file(ds), "observed_size_in_source": ds.stat().st_size}
    cfgs = {}
    for rel in ("configs/m2b/campaign_v2.json", "configs/m2b/data_v2.json", "configs/m2b/efficiency_v2.json",
                "configs/m2/data.json", "configs/m2/training.json", "configs/m2/quantization.json", "configs/m2/attacks.json"):
        if (root / rel).is_file():
            cfgs[rel] = sha256_file(root / rel)
    identity["config_sha256"] = cfgs
    split_dir = root / "reports" / "milestone2b" / "split_v2"
    freeze = split_dir / "split_freeze.json"
    if freeze.is_file():
        data = json.loads(freeze.read_text(encoding="utf-8"))
        identity["split_v2"] = {"manifest_sha256": data.get("manifest_sha256"), "id_sha256": data.get("id_sha256"),
                                "split_m2_json_sha256": sha256_file(split_dir / "split_m2.json"),
                                "split_freeze_json_sha256": sha256_file(freeze)}
    identity["historical_windows_path_matches"] = scans["historical"]
    return identity


def excluded_basename(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(name, p) for p in EXCLUDE_BASENAMES)


# --------------------------------------------------------------------------------------------------------- build
def stage_copy(root: Path, stage: Path, include: list, pave: dict) -> None:
    for rel in include:
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)
    for rel in pave:
        dest = stage / PAVE_REL / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / PAVE_REL / rel, dest)


def pave_main_ref(text: str) -> str:
    match = re.search(r'^PIN\s*=\s*"([0-9a-f]{40})"', text, re.M)
    return match.group(1) if match else ""


def strict_checks(root: Path, tree: Path, identity: dict, results: dict) -> None:
    """Scans 6-9 of DESIGN 7.5 (dataset, PAVE, split, config hashes). Recorded-before values are read from R."""
    ds = tree / DATASET_REL
    cfg_sha = json.loads((root / "configs/m2/data.json").read_text(encoding="utf-8"))["dataset"]["sha256"]
    prov = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    observed = sha256_file(ds)
    results["dataset"] = {"sha256_in_T": observed, "size_in_T": ds.stat().st_size,
                          "equals_configs_m2_data_json": observed == cfg_sha, "equals_provenance_json": observed == prov["dataset_sha256"],
                          "equals_frozen_constant": observed == DATASET_SHA, "size_ok": ds.stat().st_size == DATASET_SIZE,
                          "recorded_before_copy_in": ["configs/m2/data.json", "provenance.json", "configs/m2/data.json dataset.sha256"]}
    cfg = json.loads((root / "configs/milestone1_artifact_full.json").read_text(encoding="utf-8"))["upstream_source_sha256"]
    bad = [rel for rel, sha in cfg.items() if not (tree / PAVE_REL / rel).is_file() or sha256_file(tree / PAVE_REL / rel) != sha]
    sys.path.insert(0, str(tree / "scripts" / "vm"))
    try:
        import importlib
        vp = importlib.import_module("verify_package")
        importlib.reload(vp)
        pave_problems = vp.check_pave(tree)
        tree_id_T = vp.git_tree_id({r: (m["mode"], m["git_blob_id"]) for r, m in identity["pave"]["files"].items()})
    finally:
        sys.path.pop(0)
    pin = pave_main_ref((root / "research" / "common.py").read_text(encoding="utf-8"))
    results["pave"] = {"upstream_source_sha256_entries": len(cfg), "upstream_source_sha256_mismatches": bad,
                       "recorded_before_copy_in": "configs/milestone1_artifact_full.json",
                       "tree_id_recomputed_in_T": tree_id_T, "tree_id_from_R_pave_git": run_git(root / PAVE_REL, "rev-parse", "HEAD^{tree}"),
                       "derived_from_R_pave_git": True, "verify_package_problems": pave_problems,
                       "head": identity["pave"]["commit"], "common_PIN": pin, "provenance_upstream_commit": prov["upstream_commit"]}
    siblings = json.loads((tree / "configs/m2b/campaign_v2.json").read_text(encoding="utf-8"))["siblings"]
    results["config_hashes"] = {name: sha256_file(tree / meta["path"]) == meta["sha256"] for name, meta in siblings.items()}


def run_tool(tree: Path, args: list) -> tuple:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-B", *args], cwd=tree, capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout + proc.stderr)[-2000:]


def build(root: Path, parent: Path, name: str = DEFAULT_NAME, strict: bool = True) -> dict:
    """Stage, verify and publish the package. strict=False (tests) skips git, PAVE, dataset and split checks."""
    root, parent = Path(root), Path(parent)
    include = list_include(root)
    pave = pave_listing(root) if strict else {}
    stage = parent / f"{name}.partial-{utc_stamp()}"
    stage.mkdir(parents=True)
    stage_copy(root, stage, include, pave)
    runbook_src = root / "docs" / RUNBOOK
    if runbook_src.is_file():
        (stage / RUNBOOK).write_bytes(runbook_src.read_bytes().replace(b"\r\n", b"\n"))
    results = {"files_copied": len(include) + len(pave)}
    if strict:
        sys.path.insert(0, str(root))
        try:
            from research import m2b_state
        finally:
            sys.path.pop(0)
        sets = m2b_state.fingerprint_sets(stage / "research")
        text = (stage / RUNBOOK).read_text(encoding="utf-8")
        (stage / RUNBOOK).write_bytes(m2b_state.replace_runbook_table(text, m2b_state.runbook_table(sets)).encode("utf-8"))
        (stage / "reports" / "milestone2b").mkdir(parents=True, exist_ok=True)
        (stage / "reports" / "milestone2b" / "fingerprint_sets.json").write_bytes(
            (json.dumps(sets, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    scans = scan_windows_paths(stage)
    identity = build_identity(root, include, pave, scans, strict)
    (stage / IDENTITY).write_bytes((json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    results["manifest_entries"] = write_manifest(stage)

    results["windows_paths"] = {"active_hits": scans["active"], "historical_hit_files": len(scans["historical"])}
    results["symlinks"] = find_links(stage)
    results["crlf"] = find_crlf(stage)
    results["forbidden_names"] = find_forbidden(stage)
    results["compile"] = compile_all(stage)
    if strict:
        strict_checks(root, stage, identity, results)
        code, out = run_tool(stage, ["-m", "research.m2_split", "--config", "configs/m2b/data_v2.json", "--audit-only"])
        results["split_v2_audit"] = {"exit": code, "tail": out[-400:]}
        code, out = run_tool(stage, ["-m", "research.m2_split", "--config", "configs/m2/data.json", "--audit-only"])
        results["split_v1_audit"] = {"exit": code, "tail": out[-400:]}
        code, out = run_tool(stage, ["scripts/vm/check_runbook_commands.py"])
        results["runbook_commands"] = {"exit": code, "tail": out[-300:]}
        code, out = run_tool(stage, ["scripts/vm/verify_package.py"])
        results["verify_package"] = {"exit": code, "tail": out[-600:]}
    code, out = run_tool(stage, ["scripts/vm/check_shell.py"]) if (stage / "scripts/vm/check_shell.py").is_file() else (0, "no shell scripts")
    results["shell"] = {"exit": code, "tail": out[-300:]}
    results["manifest_recheck"] = verify_manifest(stage)

    results["failures"] = failures(results, strict)
    results["total_bytes"] = sum((stage / r).stat().st_size for r in walk_regular(stage))
    if results["failures"]:
        results["final_path"] = None
        results["partial_path"] = str(stage)
        return results
    final = choose_target(parent, name)
    while True:
        try:
            os.rename(stage, final)
            break
        except FileExistsError:
            final = choose_target(parent, name)
    results["final_path"] = str(final)
    return results


def failures(r: dict, strict: bool) -> list:
    bad = []
    if r["windows_paths"]["active_hits"]:
        bad.append(f"absolute Windows path in active files: {r['windows_paths']['active_hits'][:5]}")
    for key in ("symlinks", "crlf", "forbidden_names"):
        if r[key]:
            bad.append(f"{key}: {r[key][:5]}")
    if r["compile"]["failures"] or r["compile"]["pycache"]:
        bad.append(f"compile: {r['compile']['failures'][:3]} {r['compile']['pycache'][:3]}")
    if r["shell"]["exit"]:
        bad.append("shell check failed: " + r["shell"]["tail"])
    m = r["manifest_recheck"]
    if m["missing"] or m["mismatched"] or m["unlisted"] or m["symlinks"]:
        bad.append(f"manifest recheck: missing {m['missing'][:3]} mismatched {m['mismatched'][:3]} unlisted {m['unlisted'][:3]}")
    if strict:
        d = r["dataset"]
        if not (d["equals_configs_m2_data_json"] and d["equals_provenance_json"] and d["equals_frozen_constant"] and d["size_ok"]):
            bad.append("dataset identity differs from the recorded values")
        p = r["pave"]
        if p["upstream_source_sha256_mismatches"] or p["verify_package_problems"]:
            bad.append("PAVE source differs from recorded hashes")
        if p["tree_id_recomputed_in_T"] != p["tree_id_from_R_pave_git"]:
            bad.append("PAVE tree id differs from R's PAVE .git")
        if not (p["head"] == p["common_PIN"] == p["provenance_upstream_commit"]):
            bad.append("PAVE HEAD differs from common.PIN or provenance.json")
        if not all(r["config_hashes"].values()):
            bad.append(f"config sibling hashes: {r['config_hashes']}")
        for key in ("split_v2_audit", "split_v1_audit", "runbook_commands", "verify_package"):
            if r[key]["exit"]:
                bad.append(f"{key} failed: {r[key]['tail'][-200:]}")
    return bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parent", default=str(ROOT.parent))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--result-json", default=None)
    args = parser.parse_args(argv)
    result = build(ROOT, Path(args.parent), args.name, strict=True)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
