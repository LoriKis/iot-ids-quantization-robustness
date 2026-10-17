"""Evaluation-only compatibility entry point for Milestone 1 wrappers."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from .common import ROOT, atomic_json, digest, unit_status, verify_upstream
from .data import build_data, sequences
from .evaluation import evaluate
from .training import load_checkpoint


def main(kind=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Existing corrected run with saved baseline checkpoints")
    p.add_argument("--partition", choices=("validation", "test"), default="validation")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--attack", choices=("clean", "fgsm", "pgd"), default=kind or "clean")
    a = p.parse_args()
    expected = ROOT / ".venv"
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError(f"Invoke the Python interpreter in {expected}")
    verify_upstream()
    root = Path(a.run_dir).resolve()
    snapshot = json.loads((root / "run.json").read_text())
    config = snapshot["config"]
    cfg = {**config, "evaluation_partition": a.partition}
    frozen = json.loads((root / "protocol_selection/frozen.json").read_text())
    seeds = [a.seed] if a.seed is not None else config["seeds"]
    paths = [root / f"train/s{seed}/fp32/checkpoint.pt" for seed in seeds]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Requested seed is not in this run")
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Train the baseline first; evaluation wrappers never train")
    if a.dry_run:
        print(json.dumps({"attack": a.attack, "checkpoints": list(map(str, paths)), "partition": a.partition,
                          "dataset": config["dataset"], "output": "new manual_eval timestamp directory"}, indent=2))
        return
    torch.set_num_threads(config["threads"])
    data = build_data(config, root / "data")
    x, y = sequences(data["x"][a.partition], 50), data["y"][a.partition]
    for seed, path in zip(seeds, paths):
        model, record = load_checkpoint(path)
        if record["compatibility"]["manifest_sha256"] != data["manifest"]["manifest_sha256"]:
            raise ValueError("Checkpoint/data mismatch")
        folder = root / f"manual_eval/{time.time_ns()}_{a.attack}_s{seed}"
        with unit_status(folder, digest({"source": str(path), "attack": a.attack, "partition": a.partition})):
            atomic_json(folder / "source.json", {"checkpoint": str(path), "config": cfg})
            evaluate(model, model, x, y, data["ids"][a.partition], data,
                     {**frozen["selected"], "alpha_fraction": frozen["alpha_fraction"], "random_start": True},
                     config["epsilons"], cfg, folder, "fp32", seed, "fp32_autograd",
                     kinds=() if a.attack == "clean" else (a.attack,))


if __name__ == "__main__":
    main()
