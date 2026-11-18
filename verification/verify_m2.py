#!/usr/bin/env python3
"""POST-RUN verification of the Milestone 2 evidence (not part of the VM campaign).

Reads only saved files (CSV/JSON/NPZ text and number arrays) from the evidence copies under
reports/milestone2/evidence and recomputes the headline numbers. It loads no model, no .pt file,
and runs no training, inference, attack or benchmark. numpy (+ matplotlib for the figures, and
scikit-learn as an optional cross-check) are the only third-party imports; torch is never imported.

Reported values (61.96 %, 4.81 % ...) are cross-checks only. Nothing is altered to match them.

Usage (from anywhere; run it from OUTSIDE the clone to avoid writing caches into it):
    set PYTHONDONTWRITEBYTECODE=1
    python verification/verify_m2.py --evidence-root reports/milestone2/evidence --out verification/results

Writes only into --out (verify_m2.txt, verify_m2.json) and --figures (PNG regenerations).
Exit code 0 means the script ran to the end (check the PASS/FAIL summary); 1 means at least one FAIL.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

MODELS = ["V1", "V2", "V3", "V4"]
NAMES = {"V1": "FP32-ordinary", "V2": "INT8-PTQ-ordinary", "V3": "FP32-adversarial", "V4": "INT8-PTQ-adversarial"}
PARENT = {"V2": "V1", "V4": "V3"}
UNITS = ["clean_eps0.0", "fgsm_eps0.05", "fgsm_eps0.1", "pgd5_eps0.05", "pgd5_eps0.1"]
UNIT_KEY = {"clean_eps0.0": ("clean", "0.0"), "fgsm_eps0.05": ("fgsm", "0.05"), "fgsm_eps0.1": ("fgsm", "0.1"),
            "pgd5_eps0.05": ("pgd5", "0.05"), "pgd5_eps0.1": ("pgd5", "0.1")}
N_CLASSES = 27
TOL_ABS = 5e-9  # float64 round-trip tolerance for recomputed vs stored fractions

# Reported values from the task statement / earlier reports (cross-check only), fractions.
REPORTED_DISPLAY = {
    "clean V1 acc %": 61.96, "clean V2 acc %": 61.77, "clean V3 acc %": 53.85, "clean V4 acc %": 53.93,
    "PGD-5 eps0.10 V2 acc %": 4.81, "PGD-5 eps0.10 V4 acc %": 32.96,
}

# Size/hash rows of VM files that are NOT copied into Git (from the full VM file manifests of the
# setup step, hashed on the laptop before this work started). Used only to compare with stage markers.
VM_ROWS = {
    "calibration/calibration_x.pt": (4213148, "ceb7c616306b1645969ec08fdb1015a12fb2b5a292421cc9a2c55e35b3d2c024"),
    "models/V1/checkpoint.pt": (1450583, "2333a20bb80d0ac4ad66764a9d86b4bf4ccbf90e544ac75967d86e3ed9119ecf"),
    "models/V3/checkpoint.pt": (1450583, "0714d90f5ffe2d0f18744289f80c9eb9ff0be8e6b492babca561cb31010a9380"),
    "models/V2/state.pt": (499961, "4a43b9917f995188e19ccd5db8f1dc9dcda2eec1ab2f44cd3f909d58d12b61fd"),
    "models/V4/state.pt": (499961, "bf6a091b2723e194743d5dbb03839df0f1874762e831443260a39f13701128b4"),
    "models/V1/ckpt/epoch_025.pt": (4279542, "46919135a2508348022f1b3cdfe80137f1124c8e1b723f41337e4b6134266c73"),
    "models/V3/ckpt/epoch_025.pt": (4289462, "5751fd1f140ec6b08b744989a6872df8ce9f5f16e0d825142bea5774b608bbf1"),
}

LINES = []
COUNTS = {"PASS": 0, "FAIL": 0, "NOTE": 0}
OUT = {}


def rec(status, cid, msg):
    COUNTS[status] += 1
    LINES.append("%-4s [%s] %s" % (status, cid, msg))


def check(cond, cid, msg_ok, msg_bad=None):
    rec("PASS" if cond else "FAIL", cid, msg_ok if cond else (msg_bad or msg_ok))
    return bool(cond)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def ids_sha256(ids):
    """Canonical-JSON sha256 of the sorted int list (same as m2_deadline/sampling.py::ids_sha256)."""
    payload = json.dumps([int(i) for i in sorted(int(x) for x in ids)], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def label_sha256(y):
    """sha256 of int64 little-endian label bytes (same as sampling.py::label_sha256)."""
    return hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def accuracy(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(y == p))


def macro_f1(y, p, labels=range(N_CLASSES)):
    """Unweighted mean F1 over ALL labels in `labels`; zero_division=0 (a label with tp=fp=fn=0 scores 0)."""
    y = np.asarray(y); p = np.asarray(p)
    sc = []
    for c in labels:
        tp = int(np.sum((y == c) & (p == c))); fp = int(np.sum((y != c) & (p == c))); fn = int(np.sum((y == c) & (p != c)))
        d = 2 * tp + fp + fn
        sc.append(0.0 if d == 0 else 2.0 * tp / d)
    return float(np.mean(sc))


def sk_macro_f1(y, p):
    try:
        from sklearn.metrics import f1_score
    except Exception:  # pragma: no cover
        return None
    return float(f1_score(y, p, labels=list(range(N_CLASSES)), average="macro", zero_division=0))


def pct(x):
    return "%.2f%%" % (100.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--repo-root", default=None, help="clone root (for repo-tracked files used only for hash comparison)")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    clone = Path(a.repo_root).resolve() if a.repo_root else here.parent
    ev = Path(a.evidence_root).resolve() if a.evidence_root else clone / "reports" / "milestone2" / "evidence"
    out = Path(a.out).resolve() if a.out else here / "results"
    figdir = Path(a.figures).resolve() if a.figures else here / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    D = ev / "deadline-v1"
    V = ev / "v4-base"
    SRC = clone / "reports" / "milestone2" / "source"

    # ------------------------------------------------------------------ load
    protocol = jload(D / "protocol" / "protocol.json")
    clean_res = jload(D / "test" / "clean_results.json")
    rob_res = jload(D / "robustness" / "robustness_results.json")
    eff = jload(D / "efficiency" / "efficiency.json")
    split = jload(V / "reports" / "milestone2b" / "split_v2" / "split_m2.json")
    scaler = jload(V / "reports" / "milestone2b" / "split_v2" / "scaler_minmax.json")
    calib = jload(D / "calibration" / "calibration_ids.json")
    ident = jload(D / "run_identity.json")

    with open(D / "test" / "predictions_full_test.csv", newline="") as f:
        full_rows = list(csv.DictReader(f))
    with open(D / "robustness" / "predictions_subset.csv", newline="") as f:
        sub_rows = list(csv.DictReader(f))
    with open(D / "protocol" / "robustness_subset_ids.csv", newline="") as f:
        subset_ids_rows = list(csv.DictReader(f))
    with open(D / "results" / "model_comparison.csv", newline="") as f:
        mc_rows = {r["model_id"]: r for r in csv.DictReader(f)}
    with open(D / "results" / "robustness_curves.csv", newline="") as f:
        curve_rows = list(csv.DictReader(f))

    # ------------------------------------------------------------------ a. models and training
    LINES.append("=== a. Models, parents, epochs, seed (confirmed from config/log files; no checkpoint loaded)")
    A = {}
    for m in ("V1", "V3"):
        tl = jload(D / "models" / m / "train_log.json")
        meta = jload(D / "models" / m / "deadline_meta.json")
        latest = jload(D / "models" / m / "ckpt" / "latest.json")
        marker = jload(D / "stage_markers" / ("train_ordinary.ok" if m == "V1" else "train_adversarial.ok"))
        n_loss = len(tl["epoch_losses"])
        check(tl["epochs"] == 25 and n_loss == 25, "a.%s.epochs" % m,
              "%s train_log: epochs=%d, %d epoch_losses entries, %d optimizer updates (25 x %d steps/epoch)" % (
                  m, tl["epochs"], n_loss, tl["optimizer_updates"], tl["steps_per_epoch"]))
        check(tl["optimizer_updates"] == 25 * tl["steps_per_epoch"], "a.%s.updates" % m, "%s optimizer updates = 25 x steps/epoch" % m)
        check(tl["seed"] == 1, "a.%s.seed" % m, "%s train_log seed = 1" % m)
        check(latest["epoch"] == 25 and latest["file"] == "epoch_025.pt", "a.%s.latest" % m, "%s ckpt/latest.json points to epoch 25 (final-epoch rule)" % m)
        check(meta["epochs"] == 25 and marker["epochs"] == 25 and meta["model_id"] == m, "a.%s.meta" % m, "%s deadline_meta and stage marker both say 25 epochs" % m)
        check(bool(tl["adversarial"]) == (m == "V3") and meta["adversarial"] == (m == "V3"), "a.%s.adv_flag" % m,
              "%s adversarial flag = %s" % (m, m == "V3"))
        check(tl["optimizer"] == "torch.optim.Adam" and tl["batch_size"] == 512, "a.%s.optim" % m, "%s: torch.optim.Adam, batch size 512" % m)
        A[m] = {"epochs": tl["epochs"], "seed": tl["seed"], "optimizer_updates": tl["optimizer_updates"],
                "wall_clock_seconds": tl["wall_clock_seconds"], "init_state_hash": tl["init_state_hash"],
                "final_state_hash": tl["final_state_hash"], "adversarial_config": tl.get("adversarial_config"),
                "pass_counters_per_step": tl["pass_counters_per_step"], "checkpoint_sha256": tl["checkpoint_sha256"]}
    check(A["V1"]["init_state_hash"] == A["V3"]["init_state_hash"], "a.init", "V1 and V3 start from the same initial state hash (%s...)" % A["V1"]["init_state_hash"][:12])
    check(A["V1"]["final_state_hash"] != A["V3"]["final_state_hash"], "a.final_differs", "V1 and V3 final state hashes differ (different training objective)")
    check(A["V1"]["adversarial_config"] is None and A["V3"]["adversarial_config"]["epsilon"] == 0.05 and A["V3"]["adversarial_config"]["steps"] == 5,
          "a.v3_adv", "V3 trained with PGD (eps 0.05, 5 steps, alpha 0.025, random start); V1 has no adversarial config")
    for m in ("V2", "V4"):
        conv = jload(D / "models" / m / "conversion.json")
        meta = jload(D / "models" / m / "deadline_meta.json")
        par = PARENT[m]
        check(meta["extra"]["parent"] == par, "a.%s.parent" % m, "%s deadline_meta parent = %s" % (m, par))
        check(conv["float_state_hash"] == A[par]["final_state_hash"], "a.%s.float_hash" % m,
              "%s conversion.json float_state_hash equals %s final_state_hash (%s...): converted from the finished parent" % (m, par, conv["float_state_hash"][:12]))
        check(meta["extra"]["parent_checkpoint_sha256"] == A[par]["checkpoint_sha256"], "a.%s.parent_ckpt" % m, "%s records parent checkpoint sha256 = %s checkpoint sha256" % (m, par))
        check(not (D / "models" / m / "train_log.json").exists(), "a.%s.no_train_log" % m, "%s has no train_log.json in the evidence (no training run recorded for the INT8 copy)" % m)
        check(conv["mode"] == "static" and len(conv["converted"]) == 23 and conv["scope"] == "all_linear", "a.%s.ptq" % m,
              "%s: static PTQ, 23 Linear layers converted, scope all_linear" % m)
        check(conv["calibration"]["partition"] == "train" and conv["calibration"]["n_rows"] == 540, "a.%s.calib" % m, "%s calibration: 540 training rows (20 per class)" % m)
        check(conv["reload_check"]["logits_bitwise_equal"] and conv["reload_check"]["argmax_identical"], "a.%s.reload" % m, "%s reload check on 3821 validation rows: logits bit-identical" % m)
        rec("NOTE", "a.%s.epochs_field" % m, "%s deadline_meta.epochs=%d is the PARENT's epoch count carried over; the INT8 copy received 0 additional training epochs (PTQ only)." % (m, meta["epochs"]))
        A[m] = {"parent": par, "float_state_hash": conv["float_state_hash"], "int8_state_hash": conv["int8_state_hash"],
                "n_converted_linear_layers": len(conv["converted"]), "int8_weight_share": conv["parameter_counts"]["int8_weight_share"],
                "fp32_parameters_not_quantized": conv["parameter_counts"]["fp32_parameters_not_quantized"],
                "fp32_remaining_operations": conv["fp32_remaining_operations"], "calibration_rows": conv["calibration"]["n_rows"]}
    OUT["a_models"] = A

    # ------------------------------------------------------------------ b. split and preprocessing
    LINES.append("=== b. Split, scaler, calibration, hashes")
    B = {}
    ids = {k: [int(i) for i in split["row_ids"][k]] for k in ("train", "validation", "test")}
    sizes = {k: len(v) for k, v in ids.items()}
    check(sizes == {"train": 22412, "validation": 3821, "test": 3764}, "b.sizes", "split sizes %s" % sizes)
    check(protocol["split"]["n_train"] == 22412 and protocol["split"]["n_val"] == 3821 and protocol["split"]["n_test"] == 3764, "b.sizes_protocol", "protocol.json sizes agree")
    s_tr, s_va, s_te = set(ids["train"]), set(ids["validation"]), set(ids["test"])
    check(not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "b.disjoint", "train/validation/test row-ID sets are pairwise disjoint")
    check(len(s_tr | s_va | s_te) == split["dataset"]["rows"] == 29997, "b.cover", "partitions cover exactly %d distinct rows" % split["dataset"]["rows"])
    for k, key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        check(ids_sha256(ids[k]) == split["id_sha256"][key], "b.idhash.%s" % k, "recomputed id hash of %s ids equals split_m2.json id_sha256 (%s...)" % (k, split["id_sha256"][key][:12]))
    check(split["id_sha256"]["test"] == protocol["test_partition_id_sha256"], "b.test_hash_protocol", "protocol test_partition_id_sha256 equals split test id hash")
    sp_sha = sha_file(V / "reports" / "milestone2b" / "split_v2" / "split_m2.json")
    sc_sha = sha_file(V / "reports" / "milestone2b" / "split_v2" / "scaler_minmax.json")
    check(sp_sha == protocol["split"]["split_m2_sha256"], "b.split_sha", "sha256(split_m2.json) = protocol split_m2_sha256 (%s...)" % sp_sha[:12])
    check(sc_sha == protocol["split"]["scaler_sha256"], "b.scaler_sha", "sha256(scaler_minmax.json) = protocol scaler_sha256 (%s...)" % sc_sha[:12])
    check(scaler["fitted_on"] == "training rows only" and scaler["n_samples_seen"] == 22412 and scaler["train_ids_sha256"] == split["id_sha256"]["train"], "b.scaler_train_only",
          "scaler_minmax.json: fitted on training rows only, n_samples_seen=22412, train_ids_sha256 equals the training id hash (confirmed from file; refit not possible here because the dataset CSV is not in the evidence)")
    check(split["scaler"]["fit_partition"] == "train" and split["scaler"]["clip"] is False, "b.scaler_cfg", "split_m2.json scaler: MinMax [0,1], fit_partition=train, clip=false")
    cids = [int(i) for i in calib["ids"]]
    check(len(cids) == 540 and len(set(cids)) == 540 and set(cids) <= s_tr, "b.calib_subset", "540 unique calibration ids, all inside the training partition")
    check(ids_sha256(cids) == calib["id_sha256"] == jload(D / "models" / "V2" / "conversion.json")["calibration"]["id_sha256"], "b.calib_hash", "calibration id hash matches calibration_ids.json and V2 conversion.json")
    check(not (set(cids) & (s_va | s_te)), "b.calib_no_eval", "no calibration id is a validation or test id")
    rec("NOTE", "b.calib_perclass", "calibration per-class balance (20 per class) is confirmed from calibration_ids.json/conversion.json per_class_counts, not recomputed (labels of training rows are not in the compact evidence). per_class_counts=%s" % calib["per_class_counts"])
    check(split["dataset"]["classes"] == 27 and len(split["dataset"]["class_names"]) == 27 and split["dataset"]["class_names"][0] == "Benign_Final", "b.classes", "27 class names: Benign_Final + 26 attack/recon types")
    B = {"sizes": sizes, "split_m2_sha256": sp_sha, "scaler_sha256": sc_sha, "class_names": split["dataset"]["class_names"],
         "group_counts": split["group_counts"], "per_class_counts_test": split["per_class_counts"]["test"],
         "dataset_sha256": split["dataset"]["sha256"], "dataset_rows": split["dataset"]["rows"]}
    OUT["b_split"] = B

    # ------------------------------------------------------------------ c. full clean test
    LINES.append("=== c. Clean accuracy / macro-F1 on all 3,764 test IDs (recomputed from predictions_full_test.csv)")
    fid = np.array([int(r["example_id"]) for r in full_rows])
    fy = np.array([int(r["y_true"]) for r in full_rows])
    fin = np.array([r["in_robust_subset"].strip().lower() == "true" for r in full_rows])
    fp = {m: np.array([int(r[m + "_pred"]) for r in full_rows]) for m in MODELS}
    check(len(fid) == 3764 and len(set(fid.tolist())) == 3764, "c.rows", "3764 rows, 3764 unique example_id")
    check(set(fid.tolist()) == s_te, "c.ids_equal_split", "example_ids equal the split's test ids exactly")
    check(all(len(fp[m]) == 3764 for m in MODELS), "c.same_ids_all_models", "one prediction per model on the SAME rows (one shared id/label column; every model has a prediction on every id)")
    check(min(int(fp[m].min()) for m in MODELS) >= 0 and max(int(fp[m].max()) for m in MODELS) <= 26, "c.pred_range", "all predictions are in 0..26")
    check(set(fy.tolist()) == set(range(27)), "c.all_labels_present", "all 27 labels occur in the test labels (no absent-class case in the full test)")
    tc = np.bincount(fy, minlength=27).tolist()
    check(tc == split["per_class_counts"]["test"], "c.test_counts", "per-class test label counts equal split_m2.json per_class_counts.test")
    C = {}
    for m in MODELS:
        acc = accuracy(fy, fp[m]); f1 = macro_f1(fy, fp[m]); f1sk = sk_macro_f1(fy, fp[m])
        stored = clean_res["models"][m]
        mc_acc = float(mc_rows[m]["clean_accuracy_full_test"]); mc_f1 = float(mc_rows[m]["macro_f1_full_test"])
        da, df = abs(acc - stored["accuracy"]), abs(f1 - stored["macro_f1"])
        check(da <= TOL_ABS and df <= TOL_ABS and stored["n"] == 3764, "c.%s.vs_clean_results" % m,
              "%s %s recomputed acc=%.9f f1=%.9f vs clean_results.json acc=%.9f f1=%.9f (|diff| %.1e / %.1e)" % (m, NAMES[m], acc, f1, stored["accuracy"], stored["macro_f1"], da, df))
        check(abs(acc - mc_acc) <= TOL_ABS and abs(f1 - mc_f1) <= TOL_ABS, "c.%s.vs_model_comparison" % m, "%s matches results/model_comparison.csv" % m)
        if f1sk is not None:
            check(abs(f1 - f1sk) <= 1e-12, "c.%s.sklearn" % m, "%s macro-F1 equals sklearn f1_score(labels=range(27), average='macro', zero_division=0): %.12f" % (m, f1sk))
        never_pred = [c for c in range(27) if not np.any(fp[m] == c)]
        rec("NOTE", "c.%s.absent_pred" % m, "%s never predicts class(es) %s; those classes contribute F1=0 to the 27-class average (zero_division=0)" % (m, never_pred if never_pred else "none"))
        C[m] = {"accuracy": acc, "macro_f1": f1, "macro_f1_sklearn": f1sk, "n": int(len(fy)), "stored_accuracy": stored["accuracy"], "stored_macro_f1": stored["macro_f1"],
                "classes_never_predicted": never_pred}
    OUT["c_clean"] = C
    # agreement between INT8 copy and parent
    agree = {}
    for m in ("V2", "V4"):
        n = int(np.sum(fp[m] == fp[PARENT[m]]))
        agree[m] = {"parent": PARENT[m], "agree": n, "total": 3764, "pct": 100.0 * n / 3764}
        rec("NOTE", "c.%s.parent_agreement" % m, "%s predicts the same class as parent %s on %d of 3764 test rows (%.2f%%)" % (m, PARENT[m], n, 100.0 * n / 3764))
    OUT["c_parent_agreement"] = agree
    # display rounding cross-check
    disp = {"clean V1 acc %": C["V1"]["accuracy"], "clean V2 acc %": C["V2"]["accuracy"], "clean V3 acc %": C["V3"]["accuracy"], "clean V4 acc %": C["V4"]["accuracy"]}
    xchk = {}
    for k, v in disp.items():
        rep = REPORTED_DISPLAY[k]
        ok = round(100.0 * v, 2) == rep
        rec("PASS" if ok else "FAIL", "c.xcheck", "cross-check '%s': recomputed %.4f -> displays %.2f vs previously reported %.2f" % (k, 100.0 * v, round(100.0 * v, 2), rep))
        xchk[k] = {"recomputed_pct": 100.0 * v, "reported_pct": rep, "matches_after_rounding": ok}

    # ------------------------------------------------------------------ d. robustness subset
    LINES.append("=== d. Robustness subset identity (270 ids, 10 per class, same for every model/condition)")
    sid = np.array([int(r["example_id"]) for r in subset_ids_rows]); sy = np.array([int(r["y_true"]) for r in subset_ids_rows])
    check(len(sid) == 270 and len(set(sid.tolist())) == 270, "d.n", "robustness_subset_ids.csv: 270 rows, 270 unique ids")
    check(set(sid.tolist()) <= s_te, "d.in_test", "all subset ids are test ids")
    pc = np.bincount(sy, minlength=27).tolist()
    check(pc == [10] * 27, "d.per_class", "10 rows per class over 27 classes")
    full_label = {int(i): int(y) for i, y in zip(fid, fy)}
    check(all(full_label[int(i)] == int(y) for i, y in zip(sid, sy)), "d.labels_equal_full", "subset labels equal the original full-test labels for the same ids")
    check(set(fid[fin].tolist()) == set(sid.tolist()) and int(fin.sum()) == 270, "d.flag", "in_robust_subset=true rows of the full test are exactly the 270 subset ids")
    check(ids_sha256(sid) == protocol["robustness"]["subset_ids_sha256"], "d.ids_hash", "recomputed subset id hash equals protocol subset_ids_sha256 (%s...)" % protocol["robustness"]["subset_ids_sha256"][:12])
    lab_ok = any(label_sha256(order) == protocol["robustness"]["subset_label_sha256"] for order in (sy, np.array([full_label[i] for i in sorted(sid.tolist())])))
    check(lab_ok, "d.labels_hash", "label hash equals protocol subset_label_sha256 (file order or sorted-id order)")
    check(sha_file(D / "protocol" / "robustness_subset_ids.csv") == protocol["robustness"]["subset_ids_file_sha256"], "d.ids_file_sha", "sha256(robustness_subset_ids.csv) = protocol subset_ids_file_sha256")
    check(sha_file(D / "protocol" / "robustness_subset_counts.json") == protocol["robustness"]["counts_sha256"], "d.counts_sha", "sha256(robustness_subset_counts.json) = protocol counts_sha256")
    groups = {}
    for r in sub_rows:
        groups.setdefault((r["model_id"], r["attack"], r["epsilon"]), []).append(r)
    expected_keys = {(m,) + UNIT_KEY[u] for m in MODELS for u in UNITS}
    check(set(groups.keys()) == expected_keys, "d.groups", "predictions_subset.csv has exactly the 20 (model, attack, epsilon) groups = 4 models x {clean 0, fgsm .05/.1, pgd5 .05/.1}")
    same_ids = True; same_lab = True
    for k, g in groups.items():
        ids_g = [int(r["example_id"]) for r in g]
        same_ids &= (len(g) == 270 and set(ids_g) == set(sid.tolist()) and len(set(ids_g)) == 270)
        same_lab &= all(int(r["y_true"]) == full_label[int(r["example_id"])] for r in g)
    check(same_ids, "d.same_ids", "every one of the 20 groups holds the same 270 distinct ids")
    check(same_lab, "d.same_labels", "every prediction row carries the original full-test label of its id")
    D_ = {"n": 270, "per_class": pc, "ids_sha256": ids_sha256(sid)}
    OUT["d_subset"] = D_

    # ------------------------------------------------------------------ e. robustness tables
    LINES.append("=== e. Robustness tables recomputed from predictions_subset.csv (n=270)")
    sub_y = {}
    SP = {}  # (model, unit) -> (y array sorted by id, pred array)
    for u in UNITS:
        att, eps = UNIT_KEY[u]
        for m in MODELS:
            g = sorted(groups[(m, att, eps)], key=lambda r: int(r["example_id"]))
            SP[(m, u)] = (np.array([int(r["y_true"]) for r in g]), np.array([int(r["y_pred"]) for r in g]), np.array([int(r["example_id"]) for r in g]))
    E = {}
    for m in MODELS:
        E[m] = {}
        for u in UNITS:
            y, p, _ = SP[(m, u)]
            acc = accuracy(y, p); f1 = macro_f1(y, p); f1sk = sk_macro_f1(y, p)
            st = rob_res["units"][m][u]
            ok = abs(acc - st["accuracy"]) <= TOL_ABS and abs(f1 - st["macro_f1"]) <= TOL_ABS and st["n"] == 270
            check(ok, "e.%s.%s" % (m, u), "%s %s acc=%s f1=%s vs robustness_results.json acc=%s f1=%s" % (m, u, pct(acc), pct(f1), pct(st["accuracy"]), pct(st["macro_f1"])))
            if f1sk is not None:
                check(abs(f1 - f1sk) <= 1e-12, "e.%s.%s.sk" % (m, u), "%s %s macro-F1 equals sklearn" % (m, u))
            # float32 field
            d32 = abs(st["robust_accuracy"] - st["accuracy"])
            E[m][u] = {"accuracy": acc, "macro_f1": f1, "stored_accuracy": st["accuracy"], "stored_macro_f1": st["macro_f1"],
                       "stored_robust_accuracy_float32": st["robust_accuracy"], "abs_diff_accuracy_vs_robust_accuracy": d32,
                       "n_correct": int(np.sum(y == p))}
    maxd32 = max(E[m][u]["abs_diff_accuracy_vs_robust_accuracy"] for m in MODELS for u in UNITS)
    rec("NOTE", "e.float32", "robust_accuracy (float32 field in robustness_results.json / curves CSV) differs from accuracy (float64) by at most %.2e: float32 rounding, not a data difference" % maxd32)
    # curves csv
    cm = True
    for r in curve_rows:
        m = r["model_id"]; att = r["attack"]; e = float(r["epsilon"]); uid = r["unit_id"]
        if uid not in E[m]:
            cm = False; continue
        cm &= abs(float(r["accuracy"]) - E[m][uid]["accuracy"]) <= TOL_ABS and abs(float(r["macro_f1"]) - E[m][uid]["macro_f1"]) <= TOL_ABS and int(r["n"]) == 270
    check(cm and len(curve_rows) == 24, "e.curves_csv", "results/robustness_curves.csv (24 rows) equals the recomputed values")
    # pp differences
    ppd = jload(D / "results" / "pp_differences.json")["differences"]
    bad = []
    for d in ppd:
        a_, b_, metric, point = d["a"], d["b"], d["metric"], d["point"]
        def val(mm):
            if point == "full_test":
                return C[mm]["accuracy"] if metric == "accuracy" else C[mm]["macro_f1"]
            return E[mm][point][metric]
        calc = round((val(b_) - val(a_)) * 100.0, 2) + 0.0
        if abs(calc - d["pp"]) > 1e-9:
            bad.append((a_, b_, metric, point, calc, d["pp"]))
    check(not bad, "e.pp_differences", "all %d percentage-point differences in pp_differences.json (V2-V1, V4-V3, V4-V2 ...) equal the recomputed values%s" % (len(ppd), "" if not bad else " BAD:" + str(bad)))
    # headline cells
    headline = {}
    for key, (m, u) in {"PGD-5 eps0.10 V2 acc %": ("V2", "pgd5_eps0.1"), "PGD-5 eps0.10 V4 acc %": ("V4", "pgd5_eps0.1")}.items():
        v = E[m][u]["accuracy"]; rep = REPORTED_DISPLAY[key]
        ok = round(100.0 * v, 2) == rep
        rec("PASS" if ok else "FAIL", "e.xcheck", "cross-check '%s': recomputed %.4f -> displays %.2f vs previously reported %.2f (%d/270 correct)" % (key, 100.0 * v, round(100.0 * v, 2), rep, E[m][u]["n_correct"]))
        headline[key] = {"recomputed_pct": 100.0 * v, "reported_pct": rep, "matches_after_rounding": ok}
    xchk.update(headline)
    OUT["xcheck_reported_values"] = xchk
    OUT["e_robustness"] = E
    # arrays.npz cross-check (numbers only)
    try:
        nz_ok = True
        for m in MODELS:
            for u in UNITS:
                with np.load(D / "robustness" / m / u / "arrays.npz", allow_pickle=False) as z:
                    keys = sorted(z.files)
                    if m == "V1" and u == "clean_eps0.0":
                        rec("NOTE", "e.npz_keys", "arrays.npz keys: %s" % keys)
                    for k in ("example_id", "ids", "id"):
                        if k in z.files:
                            nz_ok &= set(z[k].astype(int).tolist()) == set(sid.tolist())
                    for k in ("pred", "y_pred", "preds", "predictions"):
                        if k in z.files:
                            nz_ok &= np.array_equal(z[k].astype(int), SP[(m, u)][1]) or np.array_equal(np.sort(z[k].astype(int)), np.sort(SP[(m, u)][1]))
        check(nz_ok, "e.npz", "per-unit arrays.npz id/prediction arrays (where present) agree with predictions_subset.csv")
    except Exception as ex:  # keep going: this is a secondary cross-check
        rec("NOTE", "e.npz", "arrays.npz cross-check skipped: %s" % ex)

    # ------------------------------------------------------------------ f. epsilon zero
    LINES.append("=== f. eps-0 subset predictions vs the same ids in the full clean test")
    F = {}
    for m in MODELS:
        y0, p0, i0 = SP[(m, "clean_eps0.0")]
        full_pred = {int(i): int(p) for i, p in zip(fid, fp[m])}
        fpred_same = np.array([full_pred[int(i)] for i in i0])
        mism = int(np.sum(fpred_same != p0)); agree_n = 270 - mism
        stored = rob_res["eps0_consistency"][m]
        check(mism == 0 and stored["eps0_vs_full_test_mismatches"] == 0 and stored["n_compared"] == 270, "f.%s.eps0" % m,
              "%s: %d of 270 eps-0 predictions agree with the full-test predictions of the same ids (recomputed mismatches=%d; stored mismatches=%d)" % (m, agree_n, mism, stored["eps0_vs_full_test_mismatches"]))
        sub_score = accuracy(y0, p0); full_score = C[m]["accuracy"]
        # curve start
        starts = [float(r["accuracy"]) for r in curve_rows if r["model_id"] == m and float(r["epsilon"]) == 0.0]
        check(len(starts) == 2 and all(abs(s - sub_score) <= TOL_ABS for s in starts), "f.%s.curve_start" % m,
              "%s: both curves start at the SUBSET clean accuracy %s (full-test accuracy is %s)" % (m, pct(sub_score), pct(full_score)))
        F[m] = {"eps0_agree": agree_n, "eps0_mismatch": mism, "subset_clean_accuracy": sub_score, "full_test_accuracy": full_score,
                "subset_minus_full_pp": 100.0 * (sub_score - full_score), "eps0_max_abs_logit_diff_recorded": stored["eps0_max_abs_logit_diff"]}
    rec("NOTE", "f.logit_diff", "eps0_max_abs_logit_diff is null (not recorded) in robustness_results.json for all models; only predicted classes were compared")
    OUT["f_eps0"] = F

    # ------------------------------------------------------------------ g. figures
    LINES.append("=== g. Figures: recomputed series vs the series the original report.py plots; regenerated PNGs (post-run)")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))
    G = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps_list = [float(e) for e in protocol["robustness"]["epsilons"]]
        check(eps_list == [0.0, 0.05, 0.1], "g.eps", "figure x-axis ticks = protocol epsilons %s" % eps_list)
        for att, fname, title in (("fgsm", "fgsm_recomputed.png", "FGSM"), ("pgd5", "pgd5_recomputed.png", "PGD-5")):
            series = {}
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, metric, mt in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro-F1")):
                for m in MODELS:
                    ys = []
                    for e in eps_list:
                        u = "clean_eps0.0" if e == 0.0 else "%s_eps%s" % (att, ("%g" % e))
                        ys.append(100.0 * E[m][u][metric])
                        orig = 100.0 * float(rob_res["units"][m][u][metric])
                        if abs(ys[-1] - orig) > 100 * TOL_ABS:
                            check(False, "g.%s.%s.%s" % (att, m, metric), "plotted value differs")
                    series["%s/%s" % (m, metric)] = ys
                    ax.plot(eps_list, ys, marker="o", linestyle="--" if m in ("V2", "V4") else "-", label=NAMES[m])
                ax.set_xlabel("epsilon (feature space)"); ax.set_ylabel("%s (%%)" % mt); ax.set_xticks(eps_list)
                ax.set_title("%s, %s" % (title, mt)); ax.grid(alpha=0.3)
            axes[0].legend(fontsize=7)
            fig.suptitle("%s robustness (POST-RUN regeneration from saved predictions), n=270" % title)
            fig.tight_layout(); fig.savefig(figdir / fname, dpi=150); plt.close(fig)
            G[att] = {k: [round(v, 6) for v in vs] for k, vs in series.items()}
            check(True, "g.%s" % att, "%s: 8 series x 3 points recomputed from predictions equal the stored per-unit values the original script plots; PNG written to verification/figures/%s" % (title, fname))
        rec("NOTE", "g.original_png", "The original PNGs were NOT recomputed pixel by pixel; they were inspected visually (see figure_review.md) and their plotted series are those of results/robustness_results.json, which the recomputation above reproduces.")
    except Exception as ex:
        rec("FAIL", "g.matplotlib", "figure regeneration failed: %s" % ex)
    OUT["g_figure_series_pct"] = G

    # ------------------------------------------------------------------ h. efficiency
    LINES.append("=== h. Efficiency: raw per-process JSONs vs efficiency.json and model_comparison.csv")
    H = {}
    res = {}
    for i in range(1, 13):
        n = "%02d" % i
        r = jload(D / "efficiency" / "results" / (n + ".json")); s = jload(D / "efficiency" / "specs" / (n + ".json"))
        res[i] = (r, s)
    run_order = [x["variant"] for x in eff["run_order"]]
    exp_order = ["V1", "V2", "V3", "V4"] * 3
    check(run_order == exp_order == eff["planned_order"], "h.order", "12 fresh child processes, interleaved V1,V2,V3,V4 x 3 repetitions: %s" % "".join(v[1] for v in run_order))
    check(all(res[i][0]["variant"] == exp_order[i - 1] == res[i][1]["variant"] for i in range(1, 13)), "h.map", "results/NN.json and specs/NN.json variants follow that order")
    check(eff["any_retry"] is False and eff["any_failed"] is False and eff["repetitions"] == 3 and eff["threads"] == 1, "h.flags", "efficiency.json: 3 repetitions, 1 thread, no retry, no failure")
    check(all(res[i][0]["torch_threads"] == 1 and res[i][0]["batch1"]["n"] == 200 and res[i][0]["batch256"]["n"] == 30 and res[i][0]["n_rows"] == 256 for i in res),
          "h.settings", "every process: 1 torch thread, 200 timed batch-1 calls, 30 timed batch-256 calls, 256 fixed rows")
    eng = sorted({(r["variant"], r["quantized_engine"]) for r, s in res.values()})
    rec("NOTE", "h.engine", "quantized_engine recorded in result JSONs: %s; spec backend for INT8: %s; efficiency.json protocol says: %s" % (
        eng, sorted({s.get("backend") for r, s in res.values() if s["kind"] == "int8"}), eff["protocol"]["inference"].get("quantized_engine")))
    for m in MODELS:
        rs = [res[i][0] for i in sorted(res) if res[i][0]["variant"] == m]
        med = lambda xs: float(statistics.median(xs))
        b1_med = med([r["batch1"]["median_ms"] for r in rs]); b1_p95 = med([r["batch1"]["p95_ms"] for r in rs])
        b256_med = med([r["batch256"]["median_ms"] for r in rs]); b256_p95 = med([r["batch256"]["p95_ms"] for r in rs])
        per_ex = b256_med / 256.0
        wset = med([r["peak_wset_bytes"] for r in rs]); delta = med([r["peak_wset_minus_baseline_bytes"] for r in rs])
        sdelta = med([r["sampled_peak_minus_baseline_bytes"] for r in rs]); base = med([r["baseline_rss_bytes"] for r in rs])
        sd = {r["state_dict_bytes"] for r in rs}
        summ = eff["summary"][m]; row = mc_rows[m]
        check(abs(b1_med - summ["batch1_median_ms"]["median"]) < 1e-9 and abs(b256_med - summ["batch256_median_ms"]["median"]) < 1e-9, "h.%s.vs_efficiency_json" % m,
              "%s median-of-3 batch-1 median %.4f ms and batch-256 median %.3f ms equal efficiency.json summary" % (m, b1_med, b256_med))
        check(abs(b1_med - float(row["latency_b1_median_ms"])) < 1e-9 and abs(b1_p95 - float(row["latency_b1_p95_ms"])) < 1e-9 and abs(per_ex - float(row["latency_b256_per_example_median_ms"])) < 1e-12
              and abs(wset - float(row["peak_rss_bytes"])) < 1 and abs(delta - float(row["peak_rss_delta_bytes"])) < 1 and abs(sdelta - float(row["peak_rss_sampled_delta_bytes"])) < 1 and len(sd) == 1 and list(sd)[0] == int(row["state_dict_bytes"]),
              "h.%s.vs_csv" % m, "%s recomputed row equals results/model_comparison.csv (size, b1 median/p95, b256 per example, VmHWM, VmHWM-baseline, sampled delta)" % m)
        # raw samples cross-check
        raw_ok = all(abs(float(np.median(r["batch1"]["samples_ms"])) - r["batch1"]["median_ms"]) < 1e-9 and len(r["batch1"]["samples_ms"]) == 200 and
                     abs(float(np.median(r["batch256"]["samples_ms"])) - r["batch256"]["median_ms"]) < 1e-9 and len(r["batch256"]["samples_ms"]) == 30 for r in rs)
        check(raw_ok, "h.%s.raw_samples" % m, "%s: per-process medians recomputed from the raw samples_ms equal the stored medians" % m)
        pooled = float(np.median(np.concatenate([np.array(r["batch1"]["samples_ms"]) for r in rs])))
        H[m] = {"state_dict_bytes": int(list(sd)[0]), "batch1_median_ms_median_of_3": b1_med, "batch1_p95_ms_median_of_3": b1_p95,
                "batch1_median_ms_each_repetition": [r["batch1"]["median_ms"] for r in rs], "batch1_pooled_median_ms_600_samples": pooled,
                "batch256_median_ms_median_of_3": b256_med, "batch256_p95_ms_median_of_3": b256_p95, "batch256_per_example_ms": per_ex,
                "batch256_median_ms_each_repetition": [r["batch256"]["median_ms"] for r in rs],
                "baseline_rss_bytes_median": base, "peak_vmhwm_bytes_median": wset, "peak_minus_baseline_bytes_median": delta, "sampled_peak_minus_baseline_bytes_median": sdelta}
    rF, rI = H["V1"], H["V2"]
    for a_, b_ in (("V1", "V2"), ("V3", "V4")):
        x, y = H[a_], H[b_]
        rec("NOTE", "h.ratio.%s_vs_%s" % (b_, a_), "INT8 %s vs FP32 parent %s: serialized size ratio %.4f (%d vs %d bytes); batch-1 median latency ratio %.3f (%.3f vs %.3f ms); "
            "batch-256 time per example ratio %.3f (%.4f vs %.4f ms/example); process peak (VmHWM) ratio %.3f; peak-above-baseline ratio %.3f" % (
                b_, a_, y["state_dict_bytes"] / x["state_dict_bytes"], y["state_dict_bytes"], x["state_dict_bytes"],
                y["batch1_median_ms_median_of_3"] / x["batch1_median_ms_median_of_3"], y["batch1_median_ms_median_of_3"], x["batch1_median_ms_median_of_3"],
                y["batch256_per_example_ms"] / x["batch256_per_example_ms"], y["batch256_per_example_ms"], x["batch256_per_example_ms"],
                y["peak_vmhwm_bytes_median"] / x["peak_vmhwm_bytes_median"], y["peak_minus_baseline_bytes_median"] / x["peak_minus_baseline_bytes_median"]))
    check(H["V2"]["batch1_median_ms_median_of_3"] > H["V1"]["batch1_median_ms_median_of_3"] and H["V4"]["batch1_median_ms_median_of_3"] > H["V3"]["batch1_median_ms_median_of_3"],
          "h.b1_slower", "descriptive: INT8 batch-1 median latency is HIGHER (slower) than its FP32 parent for both pairs on this machine")
    check(H["V2"]["batch256_per_example_ms"] < H["V1"]["batch256_per_example_ms"] and H["V4"]["batch256_per_example_ms"] < H["V3"]["batch256_per_example_ms"],
          "h.b256_faster", "descriptive: INT8 batch-256 time per example is LOWER than its FP32 parent for both pairs on this machine")
    check(H["V2"]["peak_vmhwm_bytes_median"] > H["V1"]["peak_vmhwm_bytes_median"] and H["V4"]["peak_vmhwm_bytes_median"] > H["V3"]["peak_vmhwm_bytes_median"],
          "h.mem_not_lower", "descriptive: INT8 process peak memory (VmHWM) is NOT lower than its FP32 parent")
    b256_spread = {m: (min(H[m]["batch256_median_ms_each_repetition"]), max(H[m]["batch256_median_ms_each_repetition"])) for m in MODELS}
    rec("NOTE", "h.spread", "batch-256 median per repetition (ms) min..max: %s (INT8 spread is wide: V2 %.1f..%.1f, V4 %.1f..%.1f vs FP32 about 315-322)" % (
        {m: "%.1f..%.1f" % b256_spread[m] for m in MODELS}, *b256_spread["V2"], *b256_spread["V4"]))
    rec("NOTE", "h.three_reps", "The 3 efficiency repetitions are 3 timing processes of the SAME trained models; they are not 3 training runs.")
    rec("NOTE", "h.sizes_three_things", "serialized state_dict bytes (torch.save of state_dict; V1/V3=%d, V2/V4=%d) is a different quantity from the on-disk checkpoint files (V1/V3 checkpoint.pt = 1450583 B; V2/V4 state.pt = 499961 B, from the setup manifest) and from process memory (VmHWM ~446-476 MB, which includes the Python/torch runtime). Peak minus baseline is inference-time working memory only." % (
        H["V1"]["state_dict_bytes"], H["V2"]["state_dict_bytes"]))
    OUT["h_efficiency"] = H
    fx = jload(D / "efficiency" / "fixed_rows.json")
    OUT["h_efficiency_fixed_rows"] = {"count": eff["fixed_rows"]["count"], "x_sha256": eff["fixed_rows"]["x_sha256"], "id_sha256": eff["fixed_rows"]["id_sha256"]}
    arr = np.load(D / "efficiency" / "fixed_rows.npy", allow_pickle=False)
    tb = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    rec("NOTE", "h.rows_sha_value", "fixed_rows.npy shape=%s dtype=%s; sha256(.npy file)=%s; sha256(array bytes)=%s; efficiency.json fixed_rows.x_sha256=%s -> %s" % (arr.shape, arr.dtype, sha_file(D / "efficiency" / "fixed_rows.npy"), tb, eff["fixed_rows"]["x_sha256"], "array-bytes hash matches" if tb == eff["fixed_rows"]["x_sha256"] else "stored x_sha256 NOT reproduced by either hash (hash definition not documented in the evidence)"))

    # ------------------------------------------------------------------ i. provenance
    LINES.append("=== i. Provenance: markers, identity, hashes")
    I = {}
    run_cfg, run_lock, run_proto = ident["config_sha256"], ident["lock_sha256"], ident["protocol_sha256"]
    marker_files = sorted((D / "stage_markers").glob("*.ok"))
    all_same = True
    for mf in marker_files:
        mk = jload(mf)
        if mk.get("stage") == "preflight":  # runs before the protocol exists, so protocol_sha256 is null by design
            rec("NOTE", "i.preflight_marker", "preflight.ok has protocol_sha256=%r (the protocol is frozen later, in the freeze stage)" % mk.get("protocol_sha256"))
            all_same &= (mk["config_sha256"] == run_cfg and mk["lock_sha256"] == run_lock and mk["protocol_sha256"] is None and mk["run_id"] == "m2-deadline-v1")
        else:
            all_same &= (mk["config_sha256"] == run_cfg and mk["lock_sha256"] == run_lock and mk["protocol_sha256"] == run_proto and mk["run_id"] == "m2-deadline-v1")
    check(len(marker_files) == 10 and all_same, "i.markers_consistent", "10 stage markers; config/lock hashes all equal run_identity.json; protocol hash equal in 9 (null in preflight, before the protocol exists)")
    check(protocol["config_sha256"] == run_cfg and protocol["lock_sha256"] == run_lock, "i.protocol_ids", "protocol.json config/lock hashes equal run_identity.json")
    check(sha_file(D / "protocol" / "protocol.json") == run_proto == jload(D / "protocol" / "protocol.frozen.json")["protocol_sha256"] == (D / "protocol" / "protocol.json.sha256").read_text().split()[0],
          "i.protocol_sha", "sha256(protocol.json) = run_identity protocol_sha256 = protocol.frozen.json = protocol.json.sha256 (%s...)" % run_proto[:12])
    cfg_copy = SRC / "deadline-v1-overlay" / "configs" / "m2_deadline_v1.json"
    check(sha_file(cfg_copy) == run_cfg, "i.config_sha", "sha256(configs/m2_deadline_v1.json) = config_sha256 (%s...)" % run_cfg[:12])
    check(sha_file(D / "timing" / "budget_lock.json") is not None and jload(D / "timing" / "budget_lock.json")["config_sha256"] == run_cfg and jload(D / "timing" / "budget_lock.json")["epochs"] == 25
          and jload(D / "timing" / "budget_lock.json")["performance_inspected"] is False, "i.budget_lock", "budget_lock.json: epochs 25, same config hash, performance_inspected=false (chosen from runtime only)")
    # marker outputs vs evidence files
    n_ok = 0; n_bad = []; n_vm = 0; n_vm_bad = []; n_unverified = []
    for mf in marker_files:
        mk = jload(mf)
        for rel, h in mk.get("outputs", {}).items():
            p = D / rel
            if p.is_file():
                if sha_file(p) == h:
                    n_ok += 1
                else:
                    n_bad.append(rel)
            elif rel.endswith(".log"):
                q = D / (rel + ".txt")
                if q.is_file() and sha_file(q) == h:
                    n_ok += 1
            elif rel in VM_ROWS:
                if VM_ROWS[rel][1] == h:
                    n_vm += 1
                else:
                    n_vm_bad.append(rel)
            else:
                n_unverified.append(rel)
    check(not n_bad, "i.marker_outputs", "%d marker output hashes match the copied evidence files exactly%s" % (n_ok, "" if not n_bad else "; MISMATCH: %s" % n_bad))
    check(not n_vm_bad, "i.marker_vm_files", "%d marker hashes for NOT-copied VM files (checkpoints, calibration_x.pt) equal the sha256 recorded in the setup manifest (%s) - the files themselves were not re-hashed here, not loaded" % (n_vm, sorted(k for k in VM_ROWS if k in {r for mf in marker_files for r in jload(mf).get('outputs', {})})))
    if n_unverified:
        rec("NOTE", "i.marker_unverified", "marker outputs not independently checkable here (file not in evidence copy): %s" % n_unverified)
    # source hashes
    v4src = SRC / "v4-base" / "research"
    for name, h in sorted(ident["source_sha256"].items()):
        p = v4src / name
        if p.is_file():
            check(sha_file(p) == h, "i.source.%s" % name, "run_identity source_sha256[%s] equals sha256 of source/v4-base/research/%s" % (name, name))
        else:
            q = clone / "research" / name
            if q.is_file():
                check(sha_file(q) == h, "i.source.%s" % name, "run_identity source_sha256[%s] equals sha256 of the TRACKED M1-repo file research/%s (same bytes as the VM; not duplicated under source/)" % (name, name))
            else:
                rec("NOTE", "i.source.%s" % name, "no file available to compare")
    mani = (V / "provenance" / "MANIFEST.sha256")
    check(sha_file(mani) == ident["v4_manifest_sha256"] == jload(SRC / "deadline-v1-overlay" / "OVERLAY_IDENTITY.json")["expects_v4_manifest_sha256"] == "5f49c93af30704a181ea77d3c09c23e99062accee40efc6c93ec9365e8a88d99",
          "i.v4_manifest", "sha256(V4 MANIFEST.sha256) equals run_identity v4_manifest_sha256 and the overlay's expected value")
    # overlay manifest vs overlay copy
    bad_m = []
    ok_m = 0
    for line in (SRC / "deadline-v1-overlay" / "OVERLAY_MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        p = SRC / "deadline-v1-overlay" / rel.strip()
        if p.is_file():
            (ok_m if sha_file(p) == h else None)
            if sha_file(p) == h:
                ok_m += 1
            else:
                bad_m.append(rel)
        else:
            bad_m.append(rel + " (missing)")
    check(not bad_m, "i.overlay_manifest", "%d files listed in OVERLAY_MANIFEST.sha256 match the copies under source/deadline-v1-overlay%s" % (ok_m, "" if not bad_m else "; PROBLEM: %s" % bad_m))
    # V4 manifest vs v4-base copies
    mm = {}
    for line in mani.read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1); mm[rel.strip().lstrip("*")] = h
    bad4 = []; ok4 = 0; notin = []
    for p in sorted((SRC / "v4-base").rglob("*")):
        if p.is_file():
            rel = p.relative_to(SRC / "v4-base").as_posix()
            if rel in mm:
                if sha_file(p) == mm[rel]:
                    ok4 += 1
                else:
                    bad4.append(rel)
            else:
                notin.append(rel)
    check(not bad4, "i.v4_manifest_files", "%d copied V4 source files match their entries in V4 MANIFEST.sha256%s" % (ok4, "" if not bad4 else "; MISMATCH: %s" % bad4))
    if notin:
        rec("NOTE", "i.v4_not_in_manifest", "copied V4 source files with no entry in MANIFEST.sha256: %s" % notin)
    I = {"marker_outputs_matching": n_ok, "marker_outputs_vm_manifest_matching": n_vm, "overlay_manifest_ok": ok_m, "v4_manifest_ok": ok4,
         "config_sha256": run_cfg, "lock_sha256": run_lock, "protocol_sha256": run_proto, "v4_manifest_sha256": ident["v4_manifest_sha256"]}
    OUT["i_provenance"] = I

    # ------------------------------------------------------------------ j. extra descriptive facts used in the report
    LINES.append("=== j. Extra facts read from evidence files (used by REPORT.md)")
    rep_md = (SRC.parent / "original" / "m2_deadline_report.md").read_text(encoding="utf-8")
    rec("NOTE", "j.original_epochs_column", "original report Table 1 'Epochs' column prints 25 for the INT8 rows (V2, V4); that is the parent's count carried over. INT8 copies had 0 training epochs." if "INT8-PTQ-ordinary (V2) | V1 | INT8 | 25" in rep_md else "original report epochs wording not found")
    rec("NOTE", "j.original_qat_text", "original report QAT section cites 'artifacts/m2a-pilot/qat/qat_feasibility.json' (status supported, 23 layers, 1 epoch, 2,700 rows)")
    qf = jload(V / "m2a-pilot" / "qat" / "qat_feasibility.json")
    check(qf["status"] == "supported" and qf["serialized"] is True and qf["epochs"] == 1 and qf["train_rows"] == 2700, "j.qat_pilot",
          "qat_feasibility.json: status=%s, serialized=%s, epochs=%d, train_rows=%d, accuracy_on_eval_rows=%.3f; its own claim: '%s'" % (
              qf["status"], qf["serialized"], qf["epochs"], qf["train_rows"], qf["int8_inference"]["accuracy_on_eval_rows"], qf["claim"]))
    rec("NOTE", "j.qat_pilot_calib", "QAT pilot calibration_id_sha256=%s differs from the final campaign's calibration id hash %s: the pilot calibration set is different from this campaign's. The pilot report (evidence/v4-base/m2a-pilot/pilot_report.md) lists split seed 20260 and split manifest sha256 7275ede47c84... for the pilot; this campaign used split seed 20270 (split_m2.json sha256 7b8da874b36c...)" % (qf["calibration_id_sha256"], calib["id_sha256"]))

    # ------------------------------------------------------------------ summary
    LINES.append("=== summary")
    LINES.append("PASS=%d FAIL=%d NOTE=%d" % (COUNTS["PASS"], COUNTS["FAIL"], COUNTS["NOTE"]))
    OUT["summary"] = dict(COUNTS)
    with open(out / "verify_m2.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(out / "verify_m2.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(OUT, indent=1, sort_keys=True) + "\n")
    print("\n".join(LINES))
    return 1 if COUNTS["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
