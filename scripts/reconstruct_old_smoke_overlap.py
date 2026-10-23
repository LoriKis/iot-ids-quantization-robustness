"""Reconstruct the row IDs the OLD run_smoke (research/paper_described_baseline.py) selected and
measure their overlap with the saved final split. Result is 'reconstructed, not observed': the
executed B source and smoke artifacts are not held locally. Read-only except the JSON output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
OLD_RUNNER = ROOT / "research/paper_described_baseline.py"
SPLIT = ROOT / "reports/milestone1-paper-described/split_ids.json"
CSV = ROOT / "upstream/PAVE_artifact/data/balanced_CICIoT.csv"
OUT = ROOT / "reports/milestone1-repair/old_smoke_overlap.json"


def reconstruct(labels: np.ndarray, split: dict) -> dict:
    """Old rule: per class, first 10 CSV positions = smoke train, positions 10..14 = smoke validation."""
    tr, va = [], []
    for c in range(27):
        idx = np.where(labels == c)[0]
        tr.extend(int(i) for i in idx[:10])
        va.extend(int(i) for i in idx[10:15])
    assert len(tr) == 270 and len(va) == 135 and not set(tr) & set(va)
    sets = {k: set(split[k]) for k in ("train", "validation", "test")}
    diag = va[:10]  # old attack checks used X_va_seq[:10], i.e. the first 10 smoke validation rows
    result = {
        "smoke_train_ids": tr, "smoke_validation_ids": va,
        "overlap_counts": {
            f"smoke_{a}_in_saved_{b}": len(set(ids) & sets[b])
            for a, ids in (("train", tr), ("validation", va)) for b in ("train", "validation", "test")},
        "distinct_smoke_rows_in_saved_test": len((set(tr) | set(va)) & sets["test"]),
        "smoke_train_rows_in_saved_test_ids": sorted(set(tr) & sets["test"]),
        "smoke_validation_rows_in_saved_test_ids": sorted(set(va) & sets["test"]),
        "attack_check_rows_first_10_smoke_validation": diag,
        "attack_check_rows_in_saved_test": sorted(set(diag) & sets["test"]),
        "per_class_test_overlap": {
            str(c): len({i for i in set(tr) | set(va) if labels[i] == c} & sets["test"]) for c in range(27)},
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    labels = LabelEncoder().fit_transform(pd.read_csv(CSV)["Label"])
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    result = reconstruct(labels, split)
    result.update({
        "status": "reconstructed from the CURRENT run_smoke code, not observed (executed B source and smoke artifacts are not held locally)",
        "old_runner_sha256": hashlib.sha256(OLD_RUNNER.read_bytes()).hexdigest(),
        "old_runner_lines_reproduced": {"run_smoke_starts": 823, "per_class_selection": "842-843 (c_idx[:10] train, c_idx[10:15] validation)",
                                        "one_AdamW_step_on_smoke_train": 874, "checkpoint_save_reload": "878-884",
                                        "epsilon_zero_and_bounds_checks_on_first_10_smoke_validation": "891-901"},
        "uses_of_selected_rows": [
            "one AdamW optimizer step on the 270 smoke-train rows (scaler also fit on those 270 rows)",
            "checkpoint save/reload state-hash equality",
            "epsilon-zero identity, Linf bound and [0,1] clip checks on the first 10 smoke-validation rows",
        ],
        "full_model_training_leakage": "not established: run_seed constructs fresh weights per seed (code-read evidence, not run evidence)",
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("overlap_counts", "distinct_smoke_rows_in_saved_test", "attack_check_rows_in_saved_test")}, indent=2))


if __name__ == "__main__":
    main()
