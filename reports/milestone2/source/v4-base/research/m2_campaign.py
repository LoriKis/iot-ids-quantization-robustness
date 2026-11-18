"""Milestone 2 campaign driver: frozen-config validation and phase guards. Thin by design.

Phases: m2a (the bounded pilot, delegated to research.m2_pilot), m2b_validation and m2b_test. The four-seed campaign
is executed by research.m2b_campaign (configs/m2b/campaign_v2.json); m2b_validation and m2b_test delegate to it.
m2b_test additionally needs --confirm-freeze equal to the SHA-256 of a freeze file written by
`m2b_campaign freeze-eval`, so test-partition access cannot be reached by accident.
Design: docs/MILESTONE2_PROTOCOL.md sections A, H and I; docs/M2B_VM_HANDOFF.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .common import ROOT, sha256
from .m2_split import ConfigError, PartitionGuardError

DEFAULT_CONFIG = ROOT / "configs/m2/campaign.json"
PHASES = ("m2a", "m2b_validation", "m2b_test")
PHASE_PARTITIONS = {"m2a": ("train", "validation"), "m2b_validation": ("train", "validation"),
                    "m2b_test": ("train", "validation", "test")}
SIBLINGS = ("data", "training", "quantization", "attacks", "efficiency", "pilot")
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "status", "siblings", "seeds", "epochs", "batch_size",
            "epsilon_grid", "step0", "matrix", "phases", "post_step0_note"}


class CampaignNotImplementedError(RuntimeError):
    """Kept for import compatibility with M2A callers. Never raised: m2b_validation and m2b_test now delegate."""


# ------------------------------------------------------------------ shared validators
def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_keys(obj, required, where: str, optional=()) -> None:
    if not isinstance(obj, dict):
        raise ConfigError(f"{where} must be an object")
    missing, unknown = set(required) - set(obj), set(obj) - set(required) - set(optional)
    if missing or unknown:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def check_seeds(seeds, where: str) -> None:
    if not isinstance(seeds, list) or not seeds or any(not is_int(s) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ConfigError(f"{where} must be a non-empty list of unique integers")


def check_epsilon_grid(grid, where: str) -> None:
    if (not isinstance(grid, list) or not grid or grid[0] != 0
            or any(isinstance(e, bool) or not isinstance(e, (int, float)) or not 0 <= e <= 1 for e in grid)
            or any(b <= a for a, b in zip(grid, grid[1:]))):
        raise ConfigError(f"{where} must be strictly increasing numbers in [0, 1] starting at 0")


def check_epochs(epochs, where: str, upper: int = 25) -> None:
    if not is_int(epochs) or not 1 <= epochs <= upper:
        raise ConfigError(f"{where} must be an integer in 1..{upper}")


def check_batch_size(batch_size, where: str) -> None:
    if not is_int(batch_size) or not 1 <= batch_size <= 4096:
        raise ConfigError(f"{where} must be an integer in 1..4096")


def guard_partitions(phase: str, partitions) -> None:
    """Refuse any partition a phase may not read (the test partition only in m2b_test)."""
    if phase not in PHASES:
        raise ConfigError(f"unknown phase {phase!r}; allowed: {list(PHASES)}")
    forbidden = sorted(set(partitions) - set(PHASE_PARTITIONS[phase]))
    if forbidden:
        raise PartitionGuardError(f"phase {phase!r} may not read partitions {forbidden}")


# ------------------------------------------------------------------ campaign.json
def sibling_hashes(root: Path = ROOT) -> dict:
    """SHA-256 of the sibling configuration files as they are on disk."""
    return {name: {"path": f"configs/m2/{name}.json", "sha256": sha256(root / "configs/m2" / f"{name}.json")}
            for name in SIBLINGS}


def validate_campaign_config(cfg: dict, root: Path = ROOT) -> dict:
    check_keys(cfg, TOP_KEYS, "campaign.json")
    if cfg["schema_version"] != 1:
        raise ConfigError("unsupported campaign schema_version")
    check_seeds(cfg["seeds"], "seeds")
    check_epsilon_grid(cfg["epsilon_grid"], "epsilon_grid")
    check_epochs(cfg["epochs"], "epochs")
    check_batch_size(cfg["batch_size"], "batch_size")
    check_keys(cfg["siblings"], SIBLINGS, "siblings")
    for name, entry in cfg["siblings"].items():
        check_keys(entry, ("path", "sha256"), f"siblings.{name}")
        path = root / entry["path"]
        if not path.is_file():
            raise ConfigError(f"sibling configuration missing: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ConfigError(f"sibling configuration {entry['path']} changed since campaign.json froze it")
    check_keys(cfg["phases"], PHASES, "phases")
    check_keys(cfg["step0"], ("blocking", "description", "steps", "budget_rule", "freeze"), "step0")
    if cfg["step0"]["blocking"] is not True:
        raise ConfigError("step0 must be blocking")
    training = json.loads((root / "configs/m2/training.json").read_text(encoding="utf-8"))
    attacks = json.loads((root / "configs/m2/attacks.json").read_text(encoding="utf-8"))
    if cfg["seeds"] != training["seeds"] or cfg["epochs"] != training["training"]["epochs"] \
            or cfg["batch_size"] != training["training"]["batch_size"]:
        raise ConfigError("campaign seeds, epochs or batch_size differ from configs/m2/training.json")
    if cfg["epsilon_grid"] != attacks["epsilon_grid"]:
        raise ConfigError("campaign epsilon_grid differs from configs/m2/attacks.json")
    return cfg


def load_campaign_config(path: Path | str = DEFAULT_CONFIG, root: Path = ROOT) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"campaign configuration not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return validate_campaign_config(cfg, root)


def check_freeze(path: Path | str, confirm_freeze: str | None) -> str:
    """m2b_test gate: --confirm-freeze must equal the SHA-256 of the campaign configuration file."""
    actual = sha256(path)
    if not confirm_freeze or confirm_freeze != actual:
        raise PartitionGuardError("phase 'm2b_test' needs --confirm-freeze equal to the SHA-256 of "
                                  f"{Path(path).name} ({actual}); test evaluation is not part of M2A")
    return actual


def run_phase(cfg_path: Path | str, phase: str, confirm_freeze: str | None = None, root: Path = ROOT) -> int:
    if phase not in PHASES:
        raise ConfigError(f"unknown phase {phase!r}; allowed: {list(PHASES)}")
    if phase == "m2b_test" and not (isinstance(confirm_freeze, str) and re.fullmatch(r"[0-9a-f]{64}", confirm_freeze)):
        raise PartitionGuardError("phase 'm2b_test' needs --confirm-freeze equal to the SHA-256 of the freeze file "
                                  "written by `m2b_campaign freeze-eval` (64 lowercase hex characters)")
    cfg = load_campaign_config(cfg_path, root)
    if phase == "m2a":
        from . import m2_pilot
        return m2_pilot.main(["--config", str(root / cfg["siblings"]["pilot"]["path"])])
    from . import m2b_campaign  # lazy: the M2B runner pulls in torch and the executors
    if phase == "m2b_validation":
        return m2b_campaign.main(["run", "--phase", "validation"], root=root)
    code = m2b_campaign.main(["run", "--phase", "test", "--confirm-freeze", confirm_freeze], root=root)
    if code == 3:
        raise PartitionGuardError("phase 'm2b_test' refused: --confirm-freeze did not match a verified freeze (exit 3)")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--confirm-freeze", default=None, help="SHA-256 of campaign.json (required by m2b_test)")
    args = parser.parse_args(argv)
    return run_phase(args.config, args.phase, args.confirm_freeze)


if __name__ == "__main__":
    sys.exit(main())
