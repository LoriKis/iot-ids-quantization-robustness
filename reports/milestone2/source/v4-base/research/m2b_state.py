"""Fingerprints and restart state for the M2B campaign (DESIGN 4.4).

Source fingerprints are per unit kind and built from a static AST import closure, so a fix to code that a finished unit
never used does not invalidate it. Orchestration code is recorded for information only. Nothing here trains or reads
model outputs; the module is pure bookkeeping (hashing, JSON, renames).
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import socket
from dataclasses import dataclass
from pathlib import Path

from .m2_split import FreezeBase
from .m2b_io import archive_move, durable_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
SHA_CHECK_LIMIT_BYTES = 64 * 1024 * 1024
UNIT_STATUSES = ("completed", "failed", "blocked", "invalid", "running", "interrupted")
UNIT_KEYS = ("id", "status", "phase", "seed", "kind", "started_utc", "finished_utc", "seconds", "host", "threads",
             "fingerprints_used", "orchestration_sha", "unit_dir", "outputs", "output_sizes", "error", "extra")
TAG_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

# Files that only schedule, aggregate or report. Recorded for information; a change never invalidates a unit.
ORCHESTRATION_FILES = ("m2b_campaign.py", "m2b_plan.py", "m2b_exec.py", "m2b_preflight.py", "m2b_report.py",
                       "m2b_splitreport.py", "m2_campaign.py", "m2_report.py", "m2_estimate.py", "m2_pilot.py",
                       "m2b_state.py")
# Roots of the closures. Component sets are cumulative: train < quant < attack, train < eff.
SOURCE_ROOTS = {
    "source_train": ("m2_training", "m2_split", "common", "m2b_io"),
    "source_quant": ("quantization", "m2_quant"),
    "source_attack": ("m2_attacks", "attacks"),
    "source_eff": ("m2_efficiency",),
}
SOURCE_BASE = {"source_train": None, "source_quant": "source_train", "source_attack": "source_quant",
               "source_eff": "source_quant"}
KIND_SOURCE = {"train": "source_train", "paircheck": "source_train",
               "calib": "source_quant", "ptq": "source_quant", "clean": "source_quant", "qat": "source_quant",
               "fp32_ft": "source_quant", "ft_ptq": "source_quant",
               "attack": "source_attack", "diag": "source_attack", "e4": "source_attack", "step0": "source_attack",
               "efficiency": "source_eff"}
COMMON_COMPONENTS = ("config", "data", "split", "env")
ALL_COMPONENTS = ("source_train", "source_quant", "source_attack", "source_eff") + COMMON_COMPONENTS
RUNBOOK_BEGIN, RUNBOOK_END = "<!-- fingerprint-sets:begin -->", "<!-- fingerprint-sets:end -->"


class ClosureResolutionError(RuntimeError):
    """An import inside research/ could not be resolved statically."""


class FingerprintMismatchError(RuntimeError):
    """Current fingerprints differ from the recorded ones (exit 2, nothing written)."""

    def __init__(self, message, differences=()):
        super().__init__(message)
        self.differences = list(differences)


class UnitCorruptionError(RuntimeError):
    """A completed unit's outputs are missing, truncated or changed (exit 2)."""


class ConfigIntegrityError(RuntimeError):
    """A sibling config or the dataset differs from the hash frozen in the campaign config."""


class FreezeGuardError(RuntimeError):
    """The evaluation freeze is missing or does not match (exit 3)."""


# ------------------------------------------------------------------------------------------------ import closure

def _module_file(package_dir: Path, name: str):
    candidate = package_dir / f"{name}.py"
    return candidate if candidate.is_file() else None


def _imports_of(path: Path, package_dir: Path) -> set:
    """Names of sibling modules (file stems in package_dir) imported anywhere in `path`, including nested imports."""
    package = package_dir.name
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == package and len(parts) > 1:
                    if _module_file(package_dir, parts[1]) is None:
                        raise ClosureResolutionError(f"{path.name}: cannot resolve `import {alias.name}`")
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                raise ClosureResolutionError(f"{path.name}: relative import with level {node.level} (line {node.lineno})")
            if node.level == 1 or (node.level == 0 and node.module and node.module.split(".")[0] == package):
                module_parts = (node.module or "").split(".") if node.module else []
                if node.level == 0:
                    module_parts = module_parts[1:]
                if not module_parts:  # `from . import a, b` / `from research import a`
                    for alias in node.names:
                        if _module_file(package_dir, alias.name) is not None:
                            found.add(alias.name)
                        elif alias.name == "*":
                            raise ClosureResolutionError(f"{path.name}: star import of the package (line {node.lineno})")
                else:
                    if _module_file(package_dir, module_parts[0]) is None:
                        raise ClosureResolutionError(
                            f"{path.name}: cannot resolve import of `{'.'.join(module_parts)}` (line {node.lineno})")
                    found.add(module_parts[0])
    return found


def import_closure(roots, package_dir) -> list:
    """Sorted file names (`x.py`) reachable from the root modules. Static: no import is executed. Cycle safe."""
    package_dir = Path(package_dir)
    seen, queue = set(), list(roots)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = _module_file(package_dir, name)
        if path is None:
            raise ClosureResolutionError(f"module {name!r} not found in {package_dir}")
        seen.add(name)
        queue.extend(sorted(_imports_of(path, package_dir) - seen))
    return sorted(f"{name}.py" for name in seen)


def source_sets(package_dir) -> dict:
    """{component: sorted file names}. Cumulative: quant contains train, attack and eff contain quant."""
    package_dir = Path(package_dir)
    sets = {}
    for component in ("source_train", "source_quant", "source_attack", "source_eff"):
        files = set(import_closure(SOURCE_ROOTS[component], package_dir))
        base = SOURCE_BASE[component]
        if base:
            files |= set(sets[base])
        sets[component] = sorted(files)
    return sets


def file_hashes(package_dir, names) -> dict:
    package_dir = Path(package_dir)
    return {f"{package_dir.name}/{name}": sha256_file(package_dir / name) for name in sorted(names)}


def read_pave_tree_id(root=ROOT):
    path = Path(root) / "PACKAGE_IDENTITY.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("pave", {}).get("tree_id")


def _component_digest(payload, pave_tree_id) -> str:
    text = json.dumps({"payload": payload, "pave_tree_id": pave_tree_id}, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def source_components(package_dir, pave_tree_id=None) -> dict:
    """{"digests": {component: sha}, "files": {component: {relpath: sha}}, "orchestration": {...}, "orchestration_sha"}."""
    package_dir = Path(package_dir)
    sets = source_sets(package_dir)
    files = {c: file_hashes(package_dir, names) for c, names in sets.items()}
    digests = {c: _component_digest(sorted(f.items()), pave_tree_id) for c, f in files.items()}
    orchestration = file_hashes(package_dir, [n for n in ORCHESTRATION_FILES if (package_dir / n).is_file()])
    return {"digests": digests, "files": files, "orchestration": orchestration,
            "orchestration_sha": _component_digest(sorted(orchestration.items()), pave_tree_id)}


def check_orchestration_not_in_closures(package_dir) -> list:
    """Orchestration files that a closure absorbed (must be empty) and the required presence of m2b_io.py."""
    sets = source_sets(package_dir)
    return sorted({n for names in sets.values() for n in names if n in ORCHESTRATION_FILES})


# ------------------------------------------------------------------------------------------ other components

def config_component(root, campaign_path, pave_tree_id=None):
    """Hash campaign config and every sibling; verify each sibling against the hash stored inside the campaign config."""
    root, campaign_path = Path(root), Path(campaign_path)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    files = {campaign_path.relative_to(root).as_posix(): sha256_file(campaign_path)}
    for name, entry in sorted(campaign.get("siblings", {}).items()):
        sibling = root / entry["path"]
        observed = sha256_file(sibling)
        if observed != entry["sha256"]:
            raise ConfigIntegrityError(f"sibling config {entry['path']} ({name}) hashes to {observed}, campaign config "
                                       f"records {entry['sha256']}")
        files[entry["path"]] = observed
    return _component_digest(sorted(files.items()), pave_tree_id), files


def data_component(root, data_config_path, pave_tree_id=None):
    """Dataset SHA-256 (must equal the frozen value) and the PAVE model source SHA-256."""
    root = Path(root)
    cfg = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
    dataset = root / cfg["dataset"]["path"]
    observed = sha256_file(dataset)
    if observed != cfg["dataset"]["sha256"]:
        raise ConfigIntegrityError(f"dataset {dataset.name} hashes to {observed}, frozen value {cfg['dataset']['sha256']}")
    model_src = root / "upstream/PAVE_artifact/code/models/standard_transformer.py"
    detail = {"dataset_sha256": observed, "standard_transformer_sha256": sha256_file(model_src)}
    return _component_digest(detail, pave_tree_id), detail


def split_component(split_dir, pave_tree_id=None):
    split_dir = Path(split_dir)
    freeze = json.loads((split_dir / "split_freeze.json").read_text(encoding="utf-8"))
    detail = {"manifest_sha256": freeze["manifest_sha256"], "id_sha256": freeze["id_sha256"],
              "scaler_minmax_sha256": sha256_file(split_dir / "scaler_minmax.json")}
    return _component_digest(detail, pave_tree_id), detail


def env_snapshot() -> dict:
    import numpy
    import pandas
    import sklearn
    import torch
    return {"python": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            "torch": torch.__version__, "numpy": numpy.__version__, "scikit_learn": sklearn.__version__,
            "pandas": pandas.__version__, "supported_engines": list(torch.backends.quantized.supported_engines),
            "machine": platform.machine()}


def env_warnings_only() -> dict:
    import torch
    return {"torch_threads": torch.get_num_threads(), "processor": platform.processor()}


def compute_fingerprints(root=ROOT, campaign_path=None, env=None, with_data=True) -> dict:
    """All components for the package at `root`. Raises ConfigIntegrityError on frozen-hash mismatch.

    with_data=False skips hashing the dataset CSV (the `data` component is None); the test-phase gate uses it so that
    checks 1 to 3 of DESIGN 4.8 finish before the CSV is touched.
    """
    root = Path(root)
    campaign_path = Path(campaign_path) if campaign_path else root / "configs/m2b/campaign_v2.json"
    pave = read_pave_tree_id(root)
    src = source_components(root / "research", pave)
    config_digest, config_files = config_component(root, campaign_path, pave)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    data_path = root / campaign["siblings"]["data_v2"]["path"]
    data_digest, data_detail = data_component(root, data_path, pave) if with_data else (None, None)
    data_cfg = json.loads(data_path.read_text(encoding="utf-8"))
    split_digest, split_detail = split_component(root / data_cfg["split"]["output_dir"], pave)
    env = env if env is not None else env_snapshot()
    digests = dict(src["digests"], config=config_digest, data=data_digest, split=split_digest,
                   env=_component_digest(env, pave))
    return {"schema": SCHEMA, "pave_tree_id": pave, "components": digests, "orchestration_sha": src["orchestration_sha"],
            "details": {"source_files": src["files"], "orchestration": src["orchestration"], "config": config_files,
                        "data": data_detail, "split": split_detail, "env": env, "warn_only": env_warnings_only()}}


def components_for_kind(kind: str) -> tuple:
    return (KIND_SOURCE[kind],) + COMMON_COMPONENTS


def used_fingerprints(fingerprints: dict, kind: str) -> dict:
    return {c: fingerprints["components"][c] for c in components_for_kind(kind)}


def kinds_using(component: str) -> list:
    """Unit kinds whose `fingerprints_used` contains `component`."""
    if component in COMMON_COMPONENTS:
        return sorted(KIND_SOURCE)
    return sorted(k for k, c in KIND_SOURCE.items() if component in _with_bases(c))


def _with_bases(component):
    chain = []
    while component:
        chain.append(component)
        component = SOURCE_BASE.get(component)
    return chain


def compare_fingerprints(recorded: dict, current: dict) -> list:
    """Differences per component: [{component, recorded, current, files, kinds}]; orchestration is NOT included."""
    out = []
    for component in ALL_COMPONENTS:
        old, new = recorded["components"].get(component), current["components"].get(component)
        if old == new:
            continue
        files = []
        if component.startswith("source_"):
            before = recorded["details"]["source_files"].get(component, {})
            after = current["details"]["source_files"].get(component, {})
            files = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        out.append({"component": component, "recorded": old, "current": new, "files": files,
                    "kinds": kinds_using(component)})
    return out


def orchestration_changed(recorded: dict, current: dict) -> list:
    before, after = recorded["details"]["orchestration"], current["details"]["orchestration"]
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def require_matching(recorded: dict, current: dict, logger=None) -> list:
    """Raise FingerprintMismatchError on any component difference; return the orchestration WARN list otherwise."""
    differences = compare_fingerprints(recorded, current)
    if differences:
        lines = [f"{d['component']}: recorded {d['recorded']} current {d['current']}"
                 + (f"; files changed: {', '.join(d['files'])}" if d["files"] else "")
                 + f"; used by unit kinds: {', '.join(d['kinds'])}" for d in differences]
        raise FingerprintMismatchError("fingerprint mismatch (nothing was written):\n" + "\n".join(lines), differences)
    changed = orchestration_changed(recorded, current)
    if changed and logger is not None:
        logger.warning("WARN orchestration code changed: %s", ", ".join(changed))
    return changed


# ------------------------------------------------------------------------------------------------ state directory

class CampaignState:
    """output/<run>/state/ : fingerprints.json, campaign_state.json, units/<id>.json, locks/."""

    def __init__(self, state_dir):
        self.dir = Path(state_dir)
        self.units_dir = self.dir / "units"
        self.locks_dir = self.dir / "locks"

    @property
    def fingerprints_path(self):
        return self.dir / "fingerprints.json"

    @property
    def state_path(self):
        return self.dir / "campaign_state.json"

    def exists(self) -> bool:
        return self.state_path.is_file()

    def init(self, run_name, kind, seeds, opt_ins=(), fingerprints=None) -> dict:
        """Create state once. Calling again with another kind raises; with new seeds only updates seeds_requested."""
        if kind not in ("real", "smoke"):
            raise ValueError("kind must be 'real' or 'smoke'")
        if self.exists():
            state = self.read_state()
            if state["kind"] != kind:
                raise FingerprintMismatchError(f"state kind is {state['kind']!r}; a {kind!r} run cannot reuse it")
            return self.update_requested(seeds, opt_ins)
        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        if fingerprints is not None:
            durable_write_json(self.fingerprints_path, fingerprints)
        state = {"schema": SCHEMA, "run_name": run_name, "kind": kind, "created_utc": utc_now(),
                 "host": socket.gethostname(), "seeds_requested": sorted(set(seeds)), "opt_ins": sorted(set(opt_ins))}
        durable_write_json(self.state_path, state)
        return state

    def read_state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_requested(self, seeds, opt_ins=()) -> dict:
        """Union of requested seeds and opt-ins. Never touches unit records."""
        state = self.read_state()
        merged = dict(state, seeds_requested=sorted(set(state["seeds_requested"]) | set(seeds)),
                      opt_ins=sorted(set(state["opt_ins"]) | set(opt_ins)))
        if merged != state:
            durable_write_json(self.state_path, merged)
        return merged

    def read_fingerprints(self) -> dict:
        return json.loads(self.fingerprints_path.read_text(encoding="utf-8"))

    def write_fingerprints_once(self, fingerprints) -> None:
        if self.fingerprints_path.exists():
            raise FileExistsError(f"{self.fingerprints_path} exists; fingerprints are written once")
        durable_write_json(self.fingerprints_path, fingerprints)

    def unit_path(self, unit_id: str) -> Path:
        return self.units_dir / f"{unit_id}.json"

    def read_unit(self, unit_id: str):
        path = self.unit_path(unit_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_unit_record(record)
        return record

    def write_unit(self, record: dict) -> None:
        validate_unit_record(record)
        durable_write_json(self.unit_path(record["id"]), record)

    def list_units(self) -> list:
        return [self.read_unit(p.stem) for p in sorted(self.units_dir.glob("*.json"))] if self.units_dir.is_dir() else []


def validate_unit_record(record: dict) -> None:
    missing = [k for k in UNIT_KEYS if k not in record]
    if missing:
        raise UnitCorruptionError(f"unit record {record.get('id')!r} lacks keys {missing}")
    if record["status"] not in UNIT_STATUSES:
        raise UnitCorruptionError(f"unit record {record['id']!r} has unknown status {record['status']!r}")


def make_unit_record(unit_id, kind, phase, seed, status, *, fingerprints, started_utc=None, finished_utc=None,
                     seconds=None, threads=None, unit_dir=None, outputs=None, output_sizes=None, error=None,
                     extra=None) -> dict:
    return {"id": unit_id, "status": status, "phase": phase, "seed": seed, "kind": kind, "started_utc": started_utc,
            "finished_utc": finished_utc, "seconds": seconds, "host": socket.gethostname(), "threads": threads,
            "fingerprints_used": used_fingerprints(fingerprints, kind),
            "orchestration_sha": fingerprints["orchestration_sha"], "unit_dir": unit_dir,
            "outputs": dict(outputs or {}), "output_sizes": dict(output_sizes or {}), "error": error,
            "extra": dict(extra or {})}


def describe_outputs(out_root, relpaths) -> tuple:
    """({relpath: sha256}, {relpath: size}) for files under out_root."""
    out_root = Path(out_root)
    hashes, sizes = {}, {}
    for rel in sorted(relpaths):
        path = out_root / rel
        hashes[rel], sizes[rel] = sha256_file(path), path.stat().st_size
    return hashes, sizes


def verify_unit_completion(record: dict, out_root) -> None:
    """Completed marker plus every output present with the recorded size and (below 64 MB) SHA-256."""
    out_root = Path(out_root)
    if record["status"] != "completed":
        raise UnitCorruptionError(f"{record['id']}: status is {record['status']}, not completed")
    problems = []
    for rel, digest in record["outputs"].items():
        path = out_root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
            continue
        size = path.stat().st_size
        if size != record["output_sizes"].get(rel):
            problems.append(f"{rel}: size {size} != recorded {record['output_sizes'].get(rel)}")
        elif size < SHA_CHECK_LIMIT_BYTES and sha256_file(path) != digest:
            problems.append(f"{rel}: SHA-256 changed")
    if problems:
        raise UnitCorruptionError(f"{record['id']}: " + "; ".join(problems))


def affected_units(state: CampaignState, current: dict) -> list:
    """Unit records whose recorded `fingerprints_used` differ from the current components."""
    affected = []
    for record in state.list_units():
        if any(current["components"].get(c) != v for c, v in record["fingerprints_used"].items()):
            affected.append(record)
    return affected


def archive_and_restart_affected(state: CampaignState, out_root, tag: str, current: dict, now=None) -> list:
    """Move affected units (record, output directory, lock) to archive/<tag>-<utc>/; then record the new fingerprints.

    Renames only. Completed units whose recorded components still match are left in place.
    """
    if not TAG_RE.match(tag):
        raise ValueError("tag must match ^[A-Za-z0-9._-]{1,40}$")
    out_root = Path(out_root)
    archive_root = out_root / "archive"
    label = f"{tag}-{now or utc_now()}"
    moved = []
    for record in affected_units(state, current):
        unit_id = record["id"]
        unit_dir = out_root / record["unit_dir"] if record.get("unit_dir") else None
        moved.append(archive_move(state.unit_path(unit_id), archive_root, label))
        if unit_dir is not None and unit_dir.exists():
            moved.append(archive_move(unit_dir, archive_root, label))
        lock = state.locks_dir / f"{unit_id}.lock"
        if lock.exists():
            moved.append(archive_move(lock, archive_root, label))
    if state.fingerprints_path.exists():
        moved.append(archive_move(state.fingerprints_path, archive_root, label))
    durable_write_json(state.fingerprints_path, current)
    return moved


def archive_incomplete_unit(state: CampaignState, out_root, unit_id: str, unit_dir, now=None):
    """A unit directory without a completed marker is crash debris: rename it into archive/incomplete-<utc>/."""
    record = state.read_unit(unit_id)
    unit_dir = Path(out_root) / unit_dir
    if record is not None and record["status"] == "completed":
        return None
    moved = None
    if unit_dir.exists():
        moved = archive_move(unit_dir, Path(out_root) / "archive", f"incomplete-{now or utc_now()}")
    return moved


# -------------------------------------------------------------------------------------------- fingerprint sets

def fingerprint_sets(package_dir) -> dict:
    """Per component the sorted file list, and per file the unit kinds that use it. Generated, never hand-listed."""
    sets = source_sets(package_dir)
    per_file = {}
    for component, names in sets.items():
        for name in names:
            per_file.setdefault(name, set()).update(kinds_using(component))
    return {"schema": SCHEMA, "components": sets, "kinds_by_file": {n: sorted(k) for n, k in sorted(per_file.items())},
            "orchestration_files": list(ORCHESTRATION_FILES),
            "common_components": {c: sorted(KIND_SOURCE) for c in COMMON_COMPONENTS}}


def runbook_table(sets: dict) -> str:
    lines = ["| file (research/) | unit kinds invalidated if it is edited |", "|---|---|"]
    for name, kinds in sets["kinds_by_file"].items():
        lines.append(f"| `{name}` | {', '.join(kinds)} |")
    return "\n".join(lines)


def replace_runbook_table(text: str, table: str) -> str:
    begin, end = text.find(RUNBOOK_BEGIN), text.find(RUNBOOK_END)
    if begin < 0 or end < begin:
        raise ValueError("runbook lacks the fingerprint-sets markers")
    return text[:begin + len(RUNBOOK_BEGIN)] + "\n" + table + "\n" + text[end:]


def runbook_table_matches(text: str, sets: dict) -> bool:
    return replace_runbook_table(text, runbook_table(sets)) == text


def write_fingerprint_sets(path, package_dir) -> dict:
    sets = fingerprint_sets(package_dir)
    durable_write_json(path, sets)
    return sets


# ---------------------------------------------------------------------------------------------- evaluation freeze

@dataclass(frozen=True)
class EvalFreeze(FreezeBase):
    """A freeze whose file SHA-256 matched the confirmed value. Only verify_eval_freeze creates one."""


def verify_eval_freeze(freeze_path, confirm_sha256: str, **checks) -> EvalFreeze:
    """Checks 1 and 2 of DESIGN 4.8: confirm value is 64 lowercase hex and equals sha256(freeze file).

    Check 3 (checkpoint, INT8 state, calibration, conversion, surrogate-input and fingerprint comparisons against output/)
    is completed by the campaign in FEAT-003; it passes its comparison callable as `checks["compare"]`.
    """
    if not isinstance(confirm_sha256, str) or not SHA_RE.match(confirm_sha256):
        raise FreezeGuardError("--confirm-freeze must be 64 lowercase hex characters")
    freeze_path = Path(freeze_path)
    if not freeze_path.is_file():
        raise FreezeGuardError(f"freeze file not found: {freeze_path}")
    observed = sha256_file(freeze_path)
    if observed != confirm_sha256:
        raise FreezeGuardError(f"--confirm-freeze {confirm_sha256} does not equal sha256({freeze_path.name}) = {observed}")
    data = json.loads(freeze_path.read_text(encoding="utf-8"))
    compare = checks.get("compare")
    if compare is not None:
        compare(data)
    return EvalFreeze(freeze_path, observed, data)
