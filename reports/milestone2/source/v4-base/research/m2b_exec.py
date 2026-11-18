"""Unit executors for the M2B campaign (DESIGN 4.7, 4.9): one function per unit kind.

Every executor takes `(unit, env)` and returns `{"status": "completed"|"invalid", "extra": {...}}`. The scheduler in
research.m2b_campaign owns locks, state records, output hashing and error classification, so executors only compute and
write their files (durably) under env.run_dir / unit_rel_dir(unit). The Env carries paths, configs, a lazy arrays
accessor and a model cache, so tests substitute stub executors or a stub Env without loading data or torch models.

Nothing here is imported by a fingerprint closure root; edits to this file never invalidate finished units.
"""
from __future__ import annotations

import copy
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from . import m2_attacks as ma
from . import m2_efficiency as me
from . import m2_quant as mq
from . import m2_split
from . import m2_training as mt
from .data import stratified_cap
from .m2_split import PartitionGuardError
from .m2b_io import durable_write_json, durable_write_npz, durable_write_torch, sha256_file, utc_now
from .m2b_plan import PARENT
from .paper_described_adam import derived_seed, to_sequence
from .quantization import bpda_model

INT8_MODELS = ("V2", "V4", "FP32-FT-PTQ", "QAT-FT")
E4_SOURCE = {"V1": "fp32_direct", "V2": "bpda_int8", "V3": "fp32_direct", "V4": "bpda_int8"}
RESUMABLE_KINDS = ("train", "fp32_ft", "qat")  # keep <unit dir>/ckpt when an interrupted unit is re-run
CLEAN_BATCH = 256
N_CLASSES = 27
TRAINING_KINDS = ("train", "fp32_ft", "qat")


class UnitBlocked(Exception):
    """The unit cannot produce a result for a recorded reason (QAT limitation, E4 subset too small)."""

    def __init__(self, error: dict, documented: bool = True):
        super().__init__(error.get("message") or error.get("exception") or "blocked")
        self.error, self.documented = error, documented


class PairCheckError(RuntimeError):
    """V1 and V3 of a seed do not share the initial weights or the optimizer update count."""


class CalibrationIntegrityError(RuntimeError):
    """The calibration files differ from the hashes recorded by the seed's calib unit."""


# --------------------------------------------------------------------------------------------------- layout
def unit_rel_dir(unit):
    """Directory of a unit relative to the run output root (None for the shared step-0 decision file)."""
    p, s, k = unit.params, unit.seed, unit.kind
    phase = "validation" if unit.phase == "validation" else "test"
    if k in ("train", "ptq", "fp32_ft", "qat", "ft_ptq"):
        return f"seeds/s{s}/models/{p['model']}"
    if k == "calib":
        return f"seeds/s{s}/calibration"
    if k == "paircheck":
        return f"seeds/s{s}/paircheck"
    if k == "clean":
        return f"{phase}/s{s}/clean/{p['model']}"
    if k == "attack":
        return f"{phase}/s{s}/attacks/{p['model']}/{p['source']}/{p['attack']}/eps{float(p['eps']):g}"
    if k == "diag":
        return f"{phase}/s{s}/diag/{p['model']}"
    if k == "e4":
        return f"step0/e4/s{s}/{p['model']}/{p['source']}/eps{float(p['eps']):g}"
    if k == "efficiency":
        return f"efficiency/s{s}"
    return None  # step0 decision: output is step0/budget_decision.json


def threads_for(kind: str, cfg: dict) -> int:
    t = cfg["threads"]
    return t["training"] if kind in TRAINING_KINDS else t["attack_and_evaluation"]


# --------------------------------------------------------------------------------------------------- Env
class Env:
    """Paths, configs, lazily loaded arrays and a per-process model cache. No file is read until it is needed."""

    def __init__(self, root, run_dir, cfg, *, phase="validation", state=None, freeze=None, stop_flag=None,
                 fingerprints=None, epochs=None, arrays_loader=None, attack_cfg=None, extra=None):
        self.root, self.run_dir, self.cfg = Path(root), Path(run_dir), cfg
        self.phase, self.state, self.freeze = phase, state, freeze
        self.stop_flag = stop_flag or (lambda: False)
        self.fingerprints = fingerprints
        self.epochs = cfg["epochs"] if epochs is None else epochs
        self.extra = dict(extra or {})
        self._arrays_loader, self._attack_cfg = arrays_loader, attack_cfg
        self._arrays, self._cache, self._configs = None, {}, {}

    # ---- configs (lazy)
    def _sibling(self, name):
        return self.root / self.cfg["siblings"][name]["path"]

    @property
    def train_cfg(self):
        return self._configs.setdefault("train", mt.load_config(self._sibling("training")))

    @property
    def quant_cfg(self):
        return self._configs.setdefault("quant", mq.load_config(self._sibling("quantization")))

    @property
    def eff_cfg(self):
        return self._configs.setdefault("eff", me.load_config(self._sibling("efficiency_v2")))

    @property
    def att_cfg(self):
        """validation: the unmodified attacks.json; test: the validated deep copy with evaluation_partition='test'."""
        if self._attack_cfg is None:
            self._attack_cfg = ma.load_config(self._sibling("attacks"))
        return self._attack_cfg

    @property
    def partition(self):
        return "validation" if self.phase == "validation" else "test"

    @property
    def split_dir(self):
        data_cfg = json.loads(self._sibling("data_v2").read_text(encoding="utf-8"))
        return self.root / data_cfg["split"]["output_dir"]

    @property
    def split_seed(self):
        return json.loads((self.split_dir / "split_m2.json").read_text(encoding="utf-8"))["split_seed"]

    # ---- data
    def arrays(self):
        """Validation env: {'train','validation'}. Test env: {'test'} (needs a verified freeze). Loaded once."""
        if self._arrays is None:
            partitions = ("train", "validation") if self.phase == "validation" else ("test",)
            if self._arrays_loader is not None:
                self._arrays = self._arrays_loader(partitions)
            elif self.phase == "validation":
                self._arrays = m2_split.load_arrays(self.split_dir / "split_m2.json", self.split_dir / "scaler_minmax.json",
                                                    partitions)
            else:
                self._arrays = m2_split.load_arrays(self.split_dir / "split_m2.json", self.split_dir / "scaler_minmax.json",
                                                    partitions, purpose="final_test", freeze=self.freeze)
        if self.phase == "validation" and "test" in self._arrays:
            raise PartitionGuardError("a validation run must never hold the test partition")
        return self._arrays

    def partition_arrays(self):
        return self.arrays()[self.partition]

    def attack_rows(self):
        """Rows attacked in this phase (policy 'all' or {'per_class': N} from campaign_v2.json); ids kept for audit."""
        if "attack_rows" not in self._cache:
            part = self.partition_arrays()
            policy = self.cfg["attack_row_policy"][self.partition]
            ids, x, y = np.asarray(part["ids"]), np.asarray(part["x"], dtype=np.float32), np.asarray(part["y"])
            if policy != "all":
                local = stratified_cap(np.arange(len(y)), y, int(policy["per_class"]),
                                       derived_seed("attack-rows", self.split_seed, self.partition))
                ids, x, y = ids[local], x[local], y[local]
            self._cache["attack_rows"] = {"ids": ids, "x": torch.from_numpy(np.ascontiguousarray(x)), "y": y,
                                          "ids_sha256": m2_split.id_hash(ids), "label_sha256": ma._label_hash(y)}
        return self._cache["attack_rows"]

    # ---- paths and models
    def models_dir(self, seed, model):
        return self.run_dir / "seeds" / f"s{seed}" / "models" / model

    def calib_dir(self, seed):
        return self.run_dir / "seeds" / f"s{seed}" / "calibration"

    def fingerprints_used(self, kind):
        from .m2b_state import used_fingerprints
        return used_fingerprints(self.fingerprints, kind) if self.fingerprints else None

    def arch(self):
        return self.train_cfg["architecture"]

    def fp32(self, seed, model):
        key = ("fp32", seed, model)
        if key not in self._cache:
            self._cache[key] = me._load_checkpoint_model(self.models_dir(seed, model) / "checkpoint.pt", self.arch())
        return self._cache[key]

    def calibration(self, seed):
        """Calibration dict from files, after verifying their SHA-256 against the seed's calib unit record."""
        key = ("calibration", seed)
        if key not in self._cache:
            folder = self.calib_dir(seed)
            record = self.state.read_unit(f"s{seed}.val.calib") if self.state is not None else None
            for name in ("calibration_ids.json", "calibration_x.pt"):
                rel = f"seeds/s{seed}/calibration/{name}"
                if record is not None and (record["status"] != "completed" or record["outputs"].get(rel) != sha256_file(folder / name)):
                    raise CalibrationIntegrityError(f"{rel} differs from the hash recorded by s{seed}.val.calib")
            cal = json.loads((folder / "calibration_ids.json").read_text(encoding="utf-8"))
            cal["x"] = torch.load(folder / "calibration_x.pt", map_location="cpu", weights_only=True)
            self._cache[key] = cal
        return self._cache[key]

    def int8_spec(self, seed, model):
        q = self.quant_cfg["ptq"]
        return {"kind": "int8", "state": self.models_dir(seed, model) / "state.pt",
                "float_checkpoint": self.models_dir(seed, PARENT[model]) / "checkpoint.pt",
                "calibration_path": self.calib_dir(seed) / "calibration_x.pt", "scope": q["scope"], "backend": q["backend"],
                "architecture": self.arch()}

    def int8(self, seed, model):
        key = ("int8", seed, model)
        if key not in self._cache:
            torch.backends.quantized.engine = self.quant_cfg["ptq"]["backend"]
            self._cache[key] = me.load_variant(self.int8_spec(seed, model))
        return self._cache[key]

    def conversion(self, seed, model):
        return json.loads((self.models_dir(seed, model) / "conversion.json").read_text(encoding="utf-8"))

    def attack_models(self, seed, model, source):
        """({'fp32','int8','bpda','surrogate'} as the source needs them, model_ids, fp32 parent id)."""
        spec = ma.SOURCE_SPECS[source]
        need = {spec["grad"], spec["target"]}
        is_int8 = model in INT8_MODELS
        fp_id = PARENT[model] if is_int8 else model
        models, ids = {}, {"fp32": fp_id}
        if "fp32" in need:
            models["fp32"] = self.fp32(seed, fp_id)
        if is_int8 and need & {"int8", "bpda"}:
            q = self.int8(seed, model)
            models["int8"], ids["int8"] = q, model
            if "bpda" in need:
                key = ("bpda", seed, model)
                if key not in self._cache:
                    self._cache[key] = bpda_model(q, self.conversion(seed, model)["requested"], self.calibration(seed)["x"][:2])
                models["bpda"], ids["bpda"] = self._cache[key], f"{model}_bpda"
        if "surrogate" in need:
            key = ("surrogate", seed, model)
            if key not in self._cache:
                self._cache[key] = ma.build_surrogate(self.fp32(seed, fp_id), self.calibration(seed)["x"])[0]
            models["surrogate"], ids["surrogate"] = self._cache[key], f"{model}_surrogate"
        return models, ids, fp_id


# --------------------------------------------------------------------------------------------------- helpers
def macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0))


def _predict(model, sequences, batch_size=CLEAN_BATCH):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(b)["logits"].argmax(1) for b in sequences.split(batch_size)]).numpy()


def _blocked(error: Exception, stage=None, module=None, documented=True):
    raise UnitBlocked({"stage": stage, "module": module, "exception": f"{type(error).__name__}: {error}"}, documented) from error


def _unit_dir(env, unit) -> Path:
    path = env.run_dir / unit_rel_dir(unit)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------------------------- executors
def exec_train(unit, env):
    """train_unit on the FULL train partition, 25 epochs, epoch checkpoints in <unit>/ckpt (resumed if present)."""
    out = _unit_dir(env, unit)
    train = env.arrays()["train"]
    result = mt.train_unit(env.train_cfg, train["x"], train["y"], unit.seed, out, adversarial=bool(unit.params["adversarial"]),
                           epochs=env.epochs, threads=threads_for("train", env.cfg), checkpoint_dir=out / "ckpt",
                           keep_last=env.cfg["checkpointing"]["keep_last"], stop_flag=env.stop_flag,
                           fingerprints=env.fingerprints_used("train"))
    log = result["log"]
    return {"status": "completed", "extra": {"init_state_hash": log["init_state_hash"], "state_hash": result["state_hash"],
                                              "epochs": log["epochs"], "optimizer_updates": log["optimizer_updates"],
                                              "resumed_from_epoch": log.get("resumed_from_epoch"),
                                              "wall_clock_seconds": log["wall_clock_seconds"]}}


def exec_calib(unit, env):
    """TRAIN-only calibration rows (20 per class). Written once, by this unit only."""
    out = _unit_dir(env, unit)
    train = env.arrays()["train"]
    per_class = env.quant_cfg["ptq"]["calibration_per_class"]
    cal = mq.calibration_ids({"train": train}, per_class, env.split_seed)
    if not set(cal["ids"]) <= set(int(i) for i in train["ids"]):
        raise PartitionGuardError("calibration ids are not drawn from the train partition")
    mq.save_calibration(cal, out / "calibration_ids.json")
    durable_write_torch(out / "calibration_x.pt", cal["x"])
    rows = np.asarray(train["x"])[cal["local_index"]]
    return {"status": "completed", "extra": {"n": cal["n"], "id_sha256": cal["id_sha256"],
                                              "distinct_vectors": int(len(np.unique(rows, axis=0)))}}


def exec_paircheck(unit, env):
    """V1 and V3 share init_state_hash and the optimizer update count (the matched-pair assumption)."""
    out = _unit_dir(env, unit)
    logs = {m: json.loads((env.models_dir(unit.seed, m) / "train_log.json").read_text(encoding="utf-8")) for m in ("V1", "V3")}
    record = {"seed": unit.seed, "init_state_hash": {m: l["init_state_hash"] for m, l in logs.items()},
              "optimizer_updates": {m: l["optimizer_updates"] for m, l in logs.items()},
              "epochs": {m: l["epochs"] for m, l in logs.items()}}
    record["equal_init"] = len(set(record["init_state_hash"].values())) == 1
    record["equal_updates"] = len(set(record["optimizer_updates"].values())) == 1
    durable_write_json(out / "pair_check.json", record)
    if not (record["equal_init"] and record["equal_updates"]):
        raise PairCheckError(f"seed {unit.seed}: V1 and V3 differ in initial weights or update count: {record}")
    return {"status": "completed", "extra": record}


def exec_ptq(unit, env):
    out = _unit_dir(env, unit)
    model = unit.params["model"]
    val = env.arrays()["validation"]
    eval_x = torch.from_numpy(to_sequence(np.asarray(val["x"])))
    result = mq.ptq_unit(env.fp32(unit.seed, unit.params["parent"]), env.calibration(unit.seed), env.quant_cfg, out,
                         variant=model, eval_x=eval_x)
    return {"status": "completed", "extra": {"int8_state_hash": result["conversion"]["int8_state_hash"]}}


def exec_clean(unit, env):
    """Whole-partition predictions with the model as deployed (real INT8 module for INT8 models)."""
    out = _unit_dir(env, unit)
    model_id, seed = unit.params["model"], unit.seed
    part = env.partition_arrays()
    seq = torch.from_numpy(to_sequence(np.asarray(part["x"], dtype=np.float32)))
    y = np.asarray(part["y"])
    torch.backends.quantized.engine = "fbgemm"
    int8 = model_id in INT8_MODELS
    model = env.int8(seed, model_id) if int8 else env.fp32(seed, model_id)
    pred = _predict(model, seq)
    durable_write_npz(out / "predictions.npz", ids=np.asarray(part["ids"]), y=y, pred=pred)
    result = {"phase": env.phase, "partition": env.partition, "model": model_id, "seed": seed, "n": int(len(y)),
              "accuracy": float((pred == y).mean()), "macro_f1": macro_f1(y, pred),
              "engine": torch.backends.quantized.engine if int8 else None, "batch_size": CLEAN_BATCH,
              "inference_mode": True}
    if int8:
        parent_pred = _predict(env.fp32(seed, PARENT[model_id]), seq)
        result["argmax_agreement_with_parent"] = float((pred == parent_pred).mean())
        result["parent"] = PARENT[model_id]
    durable_write_json(out / "clean_result.json", result)
    return {"status": "completed", "extra": {k: result[k] for k in ("n", "accuracy", "macro_f1")}}


def _evaluate(env, unit, source, attack, eps, rows, out_dir, model_id, notes=None):
    models, model_ids, _ = env.attack_models(unit.seed, model_id, source)
    return ma.evaluate_unit(
        source=source, attack_name=attack, epsilon=eps, x=rows["x"], y=rows["y"], ids=rows["ids"], models=models, cfg=env.att_cfg,
        train_seed=unit.seed, model_id=model_id, evaluation_partition=env.att_cfg["evaluation"]["evaluation_partition"],
        model_ids=model_ids, notes=notes, out_dir=out_dir, expected_ids_sha256=rows["ids_sha256"],
        expected_label_sha256=rows["label_sha256"], raise_on_invalid=False)


def _surrogate_note(env, seed, model_id):
    """Fidelity note recorded by the diag unit (validation phase), used for surrogate sources."""
    path = env.run_dir / "validation" / f"s{seed}" / "diag" / model_id / "gradient_diagnostics.json"
    if path.is_file():
        return [json.loads(path.read_text(encoding="utf-8"))["surrogate_fidelity"]["note"]]
    return None


def exec_attack(unit, env):
    """One (model, source, attack, epsilon) unit on all rows of the phase; durable per-example predictions.npz."""
    out = _unit_dir(env, unit)
    p = unit.params
    rows = env.attack_rows()
    notes = _surrogate_note(env, unit.seed, p["model"]) if "surrogate" in p["source"] else None
    result = _evaluate(env, unit, p["source"], p["attack"], float(p["eps"]), rows, out, p["model"], notes)
    valid = result["row"] is not None and result["checks"]["valid"]
    if result["arrays"] is not None:
        durable_write_npz(out / "predictions.npz", **{k: result["arrays"][k] for k in
                                                       ("ids", "y", "pred_clean", "pred_adv", "robust_correct", "linf")})
    extra = {"valid": bool(valid), "failures": list(result["checks"]["failures"])}
    if result["row"] is not None:
        extra.update(robust_accuracy=result["row"]["robust_accuracy"], attack_seconds=result["row"]["attack_seconds"])
    return {"status": "completed" if valid else "invalid", "extra": extra}


def exec_diag(unit, env):
    """direct_int8_grad diagnostic and surrogate fidelity for one INT8 model (never reported as robustness)."""
    out = _unit_dir(env, unit)
    model_id = unit.params["model"]
    rows = env.attack_rows()
    q, fp_id = env.int8(unit.seed, model_id), PARENT[model_id]
    surrogate, names = ma.build_surrogate(env.fp32(unit.seed, fp_id), env.calibration(unit.seed)["x"])
    fidelity = ma.surrogate_fidelity(surrogate, q, names, rows["x"])
    fidelity.pop("islands")
    durable_write_json(out / "gradient_diagnostics.json", {
        "direct_int8_grad": ma.direct_int8_grad_diagnostic(q, rows["x"], rows["y"]), "surrogate_fidelity": fidelity,
        "n_rows": int(len(rows["y"])), "example_id_sha256": rows["ids_sha256"]})
    return {"status": "completed", "extra": {"surrogate_note": fidelity["note"]}}


def exec_e4(unit, env):
    """One E4 cell: pgd20 and pgd50x5 on the validation E4 subset, paired extra-break count."""
    out = _unit_dir(env, unit)
    p = unit.params
    val = env.arrays()["validation"]
    try:
        sub = ma.e4_subset(val["ids"], val["y"], env.att_cfg, env.split_seed, plumbing_only=bool(env.extra.get("plumbing_only")))
    except ma.SubsetTooSmallError as error:
        raise UnitBlocked({"stage": "e4_subset", "module": None, "exception": f"SubsetTooSmallError: {error}",
                           "message": f"{error}. Stop and ask the user."}, documented=False) from error
    local = sub.pop("local_index")
    rows = {"ids": np.asarray(val["ids"])[local], "y": np.asarray(val["y"])[local],
            "x": torch.from_numpy(np.ascontiguousarray(np.asarray(val["x"], dtype=np.float32)[local]))}
    rows["ids_sha256"], rows["label_sha256"] = m2_split.id_hash(rows["ids"]), ma._label_hash(rows["y"])
    e4 = env.att_cfg["e4"]
    done, bad = {}, []
    for budget in (e4["candidate_budget"], e4["reference_budget"]):
        done[budget] = _evaluate(env, unit, p["source"], budget, float(p["eps"]), rows, out / budget, p["model"])
        if done[budget]["row"] is None or not done[budget]["checks"]["valid"]:
            bad.append(budget)
    if bad:
        durable_write_json(out / "e4_cell.json", {"model": p["model"], "epsilon": float(p["eps"]), "invalid_budgets": bad})
        return {"status": "invalid", "extra": {"invalid_budgets": bad}}
    ref, cand = done[e4["reference_budget"]], done[e4["candidate_budget"]]
    n_extra = ma.paired_n_extra(ref["arrays"]["robust_correct"], cand["arrays"]["robust_correct"])
    durable_write_npz(out / "e4_robust.npz", ids=rows["ids"], reference=ref["arrays"]["robust_correct"],
                      candidate=cand["arrays"]["robust_correct"])
    cell = {"model": p["model"], "source": p["source"], "epsilon": float(p["eps"]), "n_extra": n_extra,
            "n": int(len(rows["ids"])), "seconds_pgd20": cand["row"]["attack_seconds"],
            "seconds_pgd50x5": ref["row"]["attack_seconds"], "subset": sub}
    durable_write_json(out / "e4_cell.json", cell)
    return {"status": "completed", "extra": {"n_extra": n_extra, "n": cell["n"]}}


def exec_step0(unit, env):
    """step0.decision: budget_check over the 12 cells -> step0/budget_decision.json (the attack budget for every unit)."""
    e4 = env.att_cfg["e4"]
    cells, n_subset, seconds = {}, None, []
    for model in e4["models"]:
        for eps in e4["epsilons"]:
            path = env.run_dir / "step0" / "e4" / f"s{unit.seed}" / model / E4_SOURCE[model] / f"eps{float(eps):g}" / "e4_cell.json"
            cell = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if "n_extra" not in cell:
                raise RuntimeError(f"E4 cell {model} eps {eps} has no result (missing or invalid unit)")
            cells[(model, float(eps))] = cell["n_extra"]
            n_subset = cell["n"]
            seconds.append(cell["seconds_pgd50x5"] / cell["seconds_pgd20"])
    plumbing = bool(env.extra.get("plumbing_only"))  # smoke only: arithmetic runs, no adequacy claim
    check = ma.budget_check(cells, n_subset, env.att_cfg, plumbing_only=plumbing)
    if plumbing:
        check = {**check, "decision": env.att_cfg["e4"]["candidate_budget"]}
    hashes = {}
    for m in ("V1", "V3"):
        hashes[m] = sha256_file(env.models_dir(unit.seed, m) / "checkpoint.pt")
    for m in ("V2", "V4"):
        hashes[m] = sha256_file(env.models_dir(unit.seed, m) / "state.pt")
    decision = {"decision": check["decision"], "n_subset": n_subset, "cells": check["cells"],
                "max_n_extra": max(cells.values()), "allowed": check["threshold"], "rule": check["rule"],
                "split_seed": env.split_seed, "model_checkpoint_sha256": hashes,
                "median_seconds_ratio_pgd50x5_over_pgd20": float(np.median(seconds)), "created_utc": utc_now(),
                "plumbing_only": plumbing}
    durable_write_json(env.run_dir / "step0" / "budget_decision.json", decision)
    return {"status": "completed", "outputs": ["step0/budget_decision.json"], "extra": {"decision": check["decision"]}}


# ---- QAT arms: any exception becomes a documented `blocked` unit
def exec_fp32_ft(unit, env):
    out = _unit_dir(env, unit)
    train = env.arrays()["train"]
    try:
        result = mq.fp32_finetune_unit(env.fp32(unit.seed, "V1"), train["x"], train["y"], env.train_cfg, env.quant_cfg, unit.seed,
                                       out_dir=out, checkpoint_dir=out / "ckpt", stop_flag=env.stop_flag,
                                       epochs=env.extra.get("qat_epochs"), fingerprints=env.fingerprints_used("fp32_ft"), threads=threads_for("fp32_ft", env.cfg))
    except (mt.TrainingInterrupted, mt.CheckpointError, FileExistsError):
        raise
    except Exception as error:  # noqa: BLE001 - documented block, other units continue
        _blocked(error, "finetune")
    return {"status": "completed", "extra": {"final_state_hash": result["log"]["final_state_hash"]}}


def exec_qat(unit, env):
    out = _unit_dir(env, unit)
    train = env.arrays()["train"]
    try:
        val = env.arrays()["validation"]
        result = mq.qat_unit(env.fp32(unit.seed, "V1"), env.calibration(unit.seed), train["x"], train["y"], env.train_cfg,
                             env.quant_cfg, unit.seed, out_dir=out, checkpoint_dir=out / "ckpt", stop_flag=env.stop_flag, epochs=env.extra.get("qat_epochs"),
                             eval_x=torch.from_numpy(to_sequence(np.asarray(val["x"]))),
                             fingerprints=env.fingerprints_used("qat"), threads=threads_for("qat", env.cfg))
    except (mt.TrainingInterrupted, mt.CheckpointError, FileExistsError):
        raise
    except mq.QATStageError as error:
        raise UnitBlocked(error.record()) from error
    except Exception as error:  # noqa: BLE001
        _blocked(error, "setup")
    return {"status": "completed", "extra": {"int8_state_hash": result["conversion"]["int8_state_hash"]}}


def exec_ft_ptq(unit, env):
    """FP32-FT -> PTQ with the same calibration ids as V2/V4 (attribution baseline for the QAT arm)."""
    out = _unit_dir(env, unit)
    try:
        val = env.arrays()["validation"]
        result = mq.ptq_unit(env.fp32(unit.seed, "FP32-FT"), env.calibration(unit.seed), env.quant_cfg, out,
                             variant="FP32-FT-PTQ", eval_x=torch.from_numpy(to_sequence(np.asarray(val["x"]))))
    except (mt.TrainingInterrupted, mt.CheckpointError, FileExistsError):
        raise
    except Exception as error:  # noqa: BLE001
        _blocked(error, "ptq")
    return {"status": "completed", "extra": {"int8_state_hash": result["conversion"]["int8_state_hash"]}}


# ---- efficiency
def hardware_info() -> dict:
    """Hardware and software facts of the machine that measured (best effort, Linux files when readable)."""
    def read(path):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    cpuinfo, meminfo = read("/proc/cpuinfo") or "", read("/proc/meminfo") or ""
    model = next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("model name")), platform.processor())
    mem_total = next((l.split(":", 1)[1].strip() for l in meminfo.splitlines() if l.startswith("MemTotal")), None)
    return {"cpu_model": model, "nproc": os.cpu_count(), "mem_total": mem_total, "kernel": platform.release(),
            "platform": platform.platform(), "scaling_governor": (read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "").strip() or None,
            "torch": torch.__version__, "torch_config": torch.__config__.show().splitlines()[:6],
            "quantized_engine": torch.backends.quantized.engine, "supported_engines": list(torch.backends.quantized.supported_engines)}


def exec_efficiency(unit, env):
    """Serialized size, latency (batch 1 and 256) and memory for V1..V4 [+ QAT arms] in fresh child processes."""
    out = _unit_dir(env, unit)
    seed, models = unit.seed, list(unit.params["models"])
    eff_cfg, arch = env.eff_cfg, env.arch()
    val = env.arrays()["validation"]
    rows = me.select_rows(val["ids"], val["x"], eff_cfg["fixed_inputs"]["count"], env.split_seed)
    specs, conversions = {}, {}
    for m in models:
        if m in INT8_MODELS:
            specs[m] = env.int8_spec(seed, m)
            conversions[m] = env.conversion(seed, m)
        else:
            specs[m] = {"kind": "fp32", "checkpoint": env.models_dir(seed, m) / "checkpoint.pt", "architecture": arch}
    total = sum(t.numel() for t in env.fp32(seed, "V1").parameters())
    durable_write_json(out / "hardware.json", {**hardware_info(), "busy_override": bool(env.extra.get("busy_override")),
                                               "variants": models, "threads": eff_cfg["threads"]})
    passes = [eff_cfg["threads"]["primary"]]
    if eff_cfg["threads"]["secondary"] not in passes:
        passes.append(eff_cfg["threads"]["secondary"])
    summary = {}
    for threads in passes:
        record = me.run_efficiency(specs, rows, eff_cfg, out / f"threads{threads}", pilot=False, conversions=conversions,
                                   total_parameters=total, banner="VALIDATION", threads=threads)
        summary[f"threads{threads}"] = {"any_failed": record["any_failed"], "variants": record["variants"]}
    return {"status": "completed", "extra": summary}


EXECUTORS = {"train": exec_train, "calib": exec_calib, "paircheck": exec_paircheck, "ptq": exec_ptq, "clean": exec_clean,
             "e4": exec_e4, "step0": exec_step0, "attack": exec_attack, "diag": exec_diag, "fp32_ft": exec_fp32_ft,
             "qat": exec_qat, "ft_ptq": exec_ft_ptq, "efficiency": exec_efficiency}


def deep_copy_test_attack_cfg(validated_cfg: dict) -> dict:
    """The test-phase attack config: a deep copy of the ALREADY VALIDATED validation config with the partition switched.

    The copy is never passed to m2_attacks.validate_config/load_config (they reject any partition but 'validation').
    """
    cfg = copy.deepcopy(validated_cfg)
    cfg["evaluation"]["evaluation_partition"] = "test"
    return cfg
