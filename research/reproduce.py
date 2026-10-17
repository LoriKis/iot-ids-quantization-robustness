"""Run original artifact in an isolated local copy, with output completeness checks.

Preserves original algorithmic behavior, including its limitations. This is
artifact reproduction, not a verified match to a specific paper table.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common import ROOT, UPSTREAM, atomic_json, provenance, unit_status, digest


def check_results(path, seeds):
    df = pd.read_csv(path)
    expected = {(s, model) for s in seeds for model in ("Standard Transformer", "AGT")}
    actual = set(zip(df.seed, df.model))
    if actual != expected or len(df) != len(expected) or set(df.dataset) != {"CICIoT"}:
        raise ValueError(f"Incomplete reproduction rows: expected {expected}, got {actual}")
    cols = ["mcc_clean", "mcc_pgd_unconstrained", "mcc_pgd_projected", "validity_unconstrained"]
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite reproduction metrics")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 13])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    root = Path(a.out).resolve()
    prov = provenance()
    cmd = [sys.executable, "code/revision_experiments.py", "--exp", "proj", "--datasets", "CICIoT",
           "--seeds", *map(str, a.seeds), "--epochs", str(a.epochs), "--out", str(root / "results")]
    if a.dry_run:
        print(json.dumps({"command": cmd, "isolated_working_directory": str(root / "reference"), "claim": __doc__}, indent=2))
        return
    with unit_status(root, digest({"provenance": prov, "epochs": a.epochs, "seeds": a.seeds})):
        reference = root / "reference"
        shutil.copytree(UPSTREAM, reference, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        env = os.environ.copy()
        env["AGT_OUT"] = str(root / "auxiliary_outputs")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (root / "raw.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(cmd, cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Upstream process failed with {completed.returncode}; see raw.log")
        check_results(root / "results/rev_proj.csv", a.seeds)
        atomic_json(root / "provenance.json", {**prov, "command": cmd,
                                               "claim": "artifact behavior only; no exact paper result verified",
                                               "limitations": ["pre-split StandardScaler", "80/20 validation evaluation", "incomplete training seeding", "synthetic sequences", "upstream swallowed errors checked by row validation"]})


if __name__ == "__main__":
    main()
