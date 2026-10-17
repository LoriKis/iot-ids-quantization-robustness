"""Artifact-level validation; a zero process exit alone is insufficient."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import sha256


def require(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing/empty artifact: {path}")


def finite(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric: {path}")
    if isinstance(value, dict):
        for k, v in value.items():
            finite(v, f"{path}.{k}")
    if isinstance(value, list):
        for i, v in enumerate(value):
            finite(v, f"{path}[{i}]")


def validate_run(root, phase):
    root = Path(root)
    snapshot = json.loads((root / "run.json").read_text())
    cfg = snapshot["config"]
    expected = json.loads((root / f"expected_{phase}.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    selection = json.loads((root / "protocol_selection/status.json").read_text())
    if selection["status"] != "completed":
        raise ValueError("Attack selection was not completed")
    for name, ids in manifest["full_split_row_ids"].items():
        for other, other_ids in manifest["full_split_row_ids"].items():
            if name != other and set(ids) & set(other_ids):
                raise ValueError("Overlapping split IDs")
    for name in expected["train"]:
        folder = root / "train" / name
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "completed":
            raise ValueError(f"Training unit incomplete: {folder}")
        for artifact in ("checkpoint.pt", "metadata.json", "history.json"):
            require(folder / artifact)
        meta = json.loads((folder / "metadata.json").read_text())
        if sha256(folder / "checkpoint.pt") != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint was modified")
    for kind in ("evaluation", "transfer", "qat"):
        for partition, names in expected.get(kind, {}).items():
            for name in names:
                folder = root / partition / kind / name
                status = json.loads((folder / "status.json").read_text())
                if status["status"] != "completed":
                    raise ValueError(f"Incomplete {folder}")
                require(folder / "metrics.json")
                require(folder / "predictions.npz")
                rows = json.loads((folder / "metrics.json").read_text())
                finite(rows)
                if kind == "transfer":
                    count = 1
                elif kind == "qat":
                    count = 3
                elif name.split("/")[-1] in ("fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear"):
                    count = 1 + 2 * len(cfg["epsilons"])
                else:
                    count = 3
                if len(rows) != count:
                    raise ValueError(f"Expected {count} rows, got {len(rows)}: {folder}")
                expected_seed = int(name.split("/")[0][1:])
                if any(r["seed"] != expected_seed for r in rows):
                    raise ValueError("Incorrect seed in metrics")
                if kind == "evaluation":
                    variant = name.split("/")[-1]
                    if any(r["variant"] != variant or r["dataset"] != "CICIoT" for r in rows):
                        raise ValueError("Incorrect model/dataset in metrics")
                    eps = cfg["epsilons"] if variant in ("fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear") else [cfg["focused_epsilon"]]
                    keys = {(r["attack"], r["epsilon"]) for r in rows}
                    wanted = {("clean", 0.0)} | {(a, e) for a in ("fgsm", "pgd") for e in eps}
                    if keys != wanted or len(keys) != len(rows):
                        raise ValueError("Missing or duplicated attack result rows")
                elif kind == "transfer" and rows[0]["direction"] != name.split("/")[-1]:
                    raise ValueError("Incorrect transfer direction")
                predictions = np.load(folder / "predictions.npz", allow_pickle=False)
                ids = np.asarray(manifest["selected_row_ids"][partition])
                if not np.array_equal(predictions["ids"], ids):
                    raise ValueError("Mismatched evaluation examples")
                for row in rows:
                    if row["n"] != len(ids) or row["partition"] != partition:
                        raise ValueError("Dataset/partition/sample count mismatch")
                    for metric in ("accuracy", "robust_accuracy", "macro_f1"):
                        if not 0 <= row[metric] <= 1:
                            raise ValueError(f"Out-of-range {metric}")
                    if row["asr_denominator"] == 0 and row["attack_success_rate"] is not None:
                        raise ValueError("Undefined ASR must be null")
                if kind in ("evaluation", "qat"):
                    require(folder / "benchmark.json")
                    finite(json.loads((folder / "benchmark.json").read_text()))
                    if "ptq" in name or kind == "qat":
                        require(folder / "state.pt")
                        inv = json.loads((folder / "conversion.json").read_text())
                        if set(inv["requested"]) != {v["name"] for v in inv["converted"]}:
                            raise ValueError("Incomplete quantization conversion")
                        if sha256(folder / "state.pt") != inv["state_sha256"]:
                            raise ValueError("Quantized state hash differs")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--phase", choices=("validation", "test", "both"), default="both")
    args = parser.parse_args()
    validate_run(args.run_dir, args.phase)
    print("All expected units, rows, examples, finite metrics and checkpoint hashes verified")


if __name__ == "__main__":
    main()
