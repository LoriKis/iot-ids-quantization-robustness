"""Milestone 2A bounded pilot: one seed, two epochs, small train/validation subsets, validation data only.

Trains V1 (ordinary FP32) and V3 (PGD adversarial training), derives V2 and V4 by INT8 PTQ with conversion and
reload checks, evaluates clean accuracy, runs FGSM and pgd20 over the frozen epsilon grid for every attack source,
the E4 budget plumbing check, the efficiency harness (reduced repetitions) and the QAT feasibility check.
Phase 'm2a' reads only the train and validation partitions; the test partition is never loaded or evaluated.
The output directory (artifacts/m2a-pilot) must not exist: the pilot refuses to overwrite.
Pilot results are non-scientific development checks and carry no accuracy or robustness conclusion.
Design: docs/MILESTONE2_PROTOCOL.md section G; configuration: configs/m2/pilot.json.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import m2_attacks as ma
from . import m2_efficiency as me
from . import m2_estimate as est
from . import m2_quant as mq
from . import m2_report
from . import m2_split
from . import m2_training as mt
from .common import ROOT, atomic_json, atomic_torch, environment, provenance, sha256, unit_status
from .data import stratified_cap
from .m2_campaign import (check_batch_size, check_epochs, check_epsilon_grid, check_keys, check_seeds, guard_partitions,
                          is_int)
from .m2_split import ConfigError, PartitionGuardError, id_hash
from .paper_described_adam import derived_seed, to_sequence
from .provenance import MemoryMonitor
from .quantization import bpda_model

log = logging.getLogger("m2_pilot")

DEFAULT_CONFIG = ROOT / "configs/m2/pilot.json"
BANNER = est.BANNER
FRAMING = m2_report.FRAMING
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "phase", "seeds", "epochs", "batch_size", "allow_partitions",
            "output_dir", "inputs", "siblings", "subsets", "attack", "e4_plumbing", "efficiency", "qat_feasibility",
            "threads", "banner"}
PILOT_MAX_EPOCHS = 2
EXPECTED_ROWS = {"train": 2700, "validation": 540, "attack": 270, "calibration": 540}


# ------------------------------------------------------------------ configuration
def validate_pilot_config(cfg: dict) -> dict:
    """ConfigError for any malformed value; PartitionGuardError if the test partition is requested."""
    check_keys(cfg, TOP_KEYS, "pilot.json")
    if cfg["schema_version"] != 1:
        raise ConfigError("unsupported pilot schema_version")
    if cfg["phase"] != "m2a":
        raise ConfigError("the pilot runs in phase 'm2a' only")
    check_seeds(cfg["seeds"], "seeds")
    if cfg["seeds"] != [1]:
        raise ConfigError("the pilot uses seed 1 only")
    check_epochs(cfg["epochs"], "epochs", upper=PILOT_MAX_EPOCHS)
    check_batch_size(cfg["batch_size"], "batch_size")
    if not isinstance(cfg["allow_partitions"], list):
        raise ConfigError("allow_partitions must be a list")
    guard_partitions(cfg["phase"], cfg["allow_partitions"])  # PartitionGuardError for 'test'
    if sorted(cfg["allow_partitions"]) != ["train", "validation"]:
        raise ConfigError("allow_partitions must be exactly train and validation")
    check_keys(cfg["inputs"], ("split", "scaler", "integrity_before"), "inputs")
    check_keys(cfg["siblings"], ("data", "training", "quantization", "attacks", "efficiency"), "siblings")
    sub = cfg["subsets"]
    check_keys(sub, ("seed_name", "seed_rule", "train_per_class", "validation_per_class", "attack_per_class",
                     "calibration_per_class", "expected_rows"), "subsets")
    for key in ("train_per_class", "validation_per_class", "attack_per_class", "calibration_per_class"):
        if not is_int(sub[key]) or sub[key] < 1:
            raise ConfigError(f"subsets.{key} must be a positive integer")
    if sub["attack_per_class"] > sub["validation_per_class"]:
        raise ConfigError("attack_per_class cannot exceed validation_per_class")
    check_keys(sub["expected_rows"], tuple(EXPECTED_ROWS), "subsets.expected_rows")
    check_keys(cfg["attack"], ("budgets", "epsilon_grid", "sources", "diagnostic"), "attack")
    check_epsilon_grid(cfg["attack"]["epsilon_grid"], "attack.epsilon_grid")
    if not cfg["attack"]["budgets"] or set(cfg["attack"]["budgets"]) - {"fgsm", "pgd20"}:
        raise ConfigError("pilot attack budgets must be a non-empty subset of fgsm, pgd20")
    if set(cfg["attack"]["sources"]) != set(ma.SOURCES) or cfg["attack"]["diagnostic"] != ma.DIAGNOSTIC_SOURCE:
        raise ConfigError("pilot attack sources must be the six frozen sources plus the direct_int8_grad diagnostic")
    check_keys(cfg["e4_plumbing"], ("enabled", "subset", "per_class", "expected_rows", "candidate_budget",
                                    "reference_budget", "reason"), "e4_plumbing")
    if not is_int(cfg["e4_plumbing"]["per_class"]) or cfg["e4_plumbing"]["per_class"] < 1 \
            or cfg["e4_plumbing"]["per_class"] > sub["attack_per_class"]:
        raise ConfigError("e4_plumbing.per_class must be an integer in 1..attack_per_class")
    check_keys(cfg["efficiency"], ("enabled", "mode"), "efficiency")
    if cfg["efficiency"]["mode"] != "pilot":
        raise ConfigError("the pilot runs the efficiency harness in 'pilot' mode")
    check_keys(cfg["qat_feasibility"], ("enabled",), "qat_feasibility")
    check_keys(cfg["threads"], ("training", "attack_and_evaluation"), "threads")
    if any(not is_int(v) or not 1 <= v <= 16 for v in cfg["threads"].values()):
        raise ConfigError("thread counts must be integers in 1..16")
    if cfg["banner"] != BANNER:
        raise ConfigError("banner must be the frozen non-scientific pilot banner")
    return cfg


def load_pilot_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"pilot configuration not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return validate_pilot_config(cfg)


# ------------------------------------------------------------------ helpers
def make_subset(part: dict, per_class: int, seed: int) -> dict:
    """Stratified subset of a loaded partition by local row position (stratified_cap returns sorted positions)."""
    y = np.asarray(part["y"])
    local = stratified_cap(np.arange(len(y)), y, per_class, seed)
    return {"ids": np.asarray(part["ids"])[local], "x": np.asarray(part["x"])[local], "y": y[local], "local": local}


def describe(subset: dict, n_classes: int = 27) -> dict:
    return {"n": int(len(subset["ids"])), "id_sha256": id_hash(subset["ids"]), "ids": [int(i) for i in subset["ids"]],
            "per_class_counts": np.bincount(subset["y"], minlength=n_classes).tolist()}


def predict(model, sequences: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(b)["logits"].argmax(1) for b in sequences.split(batch_size)])


class Phases:
    def __init__(self):
        self.seconds = {}

    @contextlib.contextmanager
    def __call__(self, name):
        log.info("phase %s: start", name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = time.perf_counter() - start
            log.info("phase %s: %.1f s", name, self.seconds[name])


def _stamp(record: dict) -> dict:
    return {"banner": BANNER, "framing": FRAMING, **record}


def _unit_dir(out: Path, model_id: str, source: str, attack: str, epsilon: float) -> Path:
    return out / "attacks" / model_id / source / attack / f"eps{epsilon:g}"


# ------------------------------------------------------------------ pilot
def run_pilot(config_path: Path | str = DEFAULT_CONFIG, out_dir: Path | str | None = None, root: Path = ROOT) -> dict:
    config_path = Path(config_path)
    cfg = load_pilot_config(config_path)
    out = Path(out_dir) if out_dir is not None else root / cfg["output_dir"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite the existing pilot output {out}")
    sib = {k: root / v for k, v in cfg["siblings"].items()}
    train_cfg, quant_cfg = mt.load_config(sib["training"]), mq.load_config(sib["quantization"])
    att_cfg, eff_cfg = ma.load_config(sib["attacks"]), me.load_config(sib["efficiency"])
    if cfg["attack"]["epsilon_grid"] != att_cfg["epsilon_grid"]:
        raise ConfigError("pilot epsilon_grid differs from configs/m2/attacks.json")
    if cfg["batch_size"] != train_cfg["training"]["batch_size"] or not set(cfg["seeds"]) <= set(train_cfg["seeds"]):
        raise ConfigError("pilot batch_size or seeds differ from configs/m2/training.json")
    before = root / cfg["inputs"]["integrity_before"]
    if not before.is_file():
        raise FileNotFoundError(f"integrity snapshot missing: {before}; run scripts/check_m2_integrity.py --write first")
    identity = {"phase": cfg["phase"], "seeds": cfg["seeds"], "epochs": cfg["epochs"], "config_sha256": sha256(config_path)}
    seed = cfg["seeds"][0]
    phases = Phases()
    with unit_status(out, identity) as execute:  # creates `out`, writes status.json (completed or failed)
        assert execute
        handler = logging.FileHandler(out / "m2.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            with MemoryMonitor() as monitor:
                result = _run(cfg, out, root, seed, phases, train_cfg, quant_cfg, att_cfg, eff_cfg, before, monitor)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    return result


def _run(cfg, out, root, seed, phases, train_cfg, quant_cfg, att_cfg, eff_cfg, before, monitor) -> dict:
    started = time.perf_counter()
    (out / "PILOT_NOTICE.txt").write_text(f"{BANNER}\n\n{FRAMING}\n", encoding="utf-8")
    prov = provenance()
    threads = cfg["threads"]
    sub_cfg = cfg["subsets"]

    # ---- data: train and validation only
    with phases("load_data"):
        split_path, scaler_path = root / cfg["inputs"]["split"], root / cfg["inputs"]["scaler"]
        split_doc = json.loads(split_path.read_text(encoding="utf-8"))
        split_seed = split_doc["split_seed"]
        arrays = m2_split.load_arrays(split_path, scaler_path, tuple(cfg["allow_partitions"]))
        if set(arrays) != {"train", "validation"}:
            raise PartitionGuardError(f"loaded partitions {sorted(arrays)}")
        name = sub_cfg["seed_name"]
        train = make_subset(arrays["train"], sub_cfg["train_per_class"], derived_seed(name, split_seed, "train"))
        val = make_subset(arrays["validation"], sub_cfg["validation_per_class"], derived_seed(name, split_seed, "validation"))
        att = make_subset({"ids": val["ids"], "x": val["x"], "y": val["y"]}, sub_cfg["attack_per_class"],
                          derived_seed(name, split_seed, "attack"))
        plumbing = make_subset({"ids": att["ids"], "x": att["x"], "y": att["y"]}, cfg["e4_plumbing"]["per_class"],
                               derived_seed(name, split_seed, "e4-plumbing"))
        cal = mq.calibration_ids({"train": {"ids": train["ids"], "x": train["x"], "y": train["y"]}},
                                 sub_cfg["calibration_per_class"], split_seed)
        for key, subset in (("train", train), ("validation", val), ("attack", att)):
            if len(subset["ids"]) != sub_cfg["expected_rows"][key]:
                raise RuntimeError(f"{key} subset has {len(subset['ids'])} rows, expected {sub_cfg['expected_rows'][key]}")
        if cal["n"] != sub_cfg["expected_rows"]["calibration"] or len(plumbing["ids"]) != cfg["e4_plumbing"]["expected_rows"]:
            raise RuntimeError("calibration or E4 plumbing subset has an unexpected size")
        if not set(val["ids"].tolist()) <= set(arrays["validation"]["ids"].tolist()) \
                or not set(train["ids"].tolist()) <= set(arrays["train"]["ids"].tolist()):
            raise RuntimeError("pilot subsets are not drawn from the train and validation partitions")
        e4_campaign = ma.e4_subset(arrays["validation"]["ids"], arrays["validation"]["y"], att_cfg, split_seed, plumbing_only=True)
        e4_campaign.pop("local_index")
        mq.save_calibration(cal, out / "calibration_ids.json")
        atomic_torch(out / "calibration_x.pt", cal["x"])
        atomic_json(out / "subsets.json", _stamp({
            "split_seed": split_seed, "split_manifest_sha256": split_doc["manifest_sha256"],
            "split_id_sha256": split_doc["id_sha256"], "subset_seed_rule": sub_cfg["seed_rule"],
            "train": describe(train), "validation": describe(val), "attack": describe(att),
            "calibration": {k: cal[k] for k in ("n", "id_sha256", "per_class_counts", "ids", "seed_rule", "seed")},
            "e4_plumbing": describe(plumbing), "campaign_e4_subset": e4_campaign}))
    val_seq = torch.from_numpy(to_sequence(val["x"]))
    x_att, y_att, ids_att = torch.from_numpy(att["x"]).float(), att["y"], att["ids"]
    models, convs, ptq_seconds, train_logs = {}, {}, {}, {}

    # ---- training: V1 ordinary, V3 adversarial, one loop
    for variant, adversarial in (("V1", False), ("V3", True)):
        with phases(f"train_{variant}"):
            unit = mt.train_unit(train_cfg, train["x"], train["y"], seed, out / "models" / variant, adversarial=adversarial,
                                 epochs=cfg["epochs"], threads=threads["training"])
            models[variant], train_logs[variant] = unit["model"].eval(), unit["log"]
    if train_logs["V1"]["init_state_hash"] != train_logs["V3"]["init_state_hash"]:
        raise RuntimeError("V1 and V3 do not share the same initial weights")
    if train_logs["V1"]["optimizer_updates"] != train_logs["V3"]["optimizer_updates"]:
        raise RuntimeError("V1 and V3 optimizer update counts differ")

    # ---- PTQ: V2 from V1, V4 from V3, same calibration rows
    torch.set_num_threads(threads["training"])
    for variant, parent in (("V2", "V1"), ("V4", "V3")):
        with phases(f"ptq_{variant}"):
            t0 = time.perf_counter()
            unit = mq.ptq_unit(models[parent], cal, quant_cfg, out / "models" / variant, variant=variant, eval_x=val_seq)
            ptq_seconds[variant] = time.perf_counter() - t0
            models[variant], convs[variant] = unit["model"].eval(), unit["conversion"]

    # ---- clean evaluation on the validation subset
    with phases("clean_eval"):
        preds = {v: predict(m, val_seq) for v, m in models.items()}
        yv = torch.from_numpy(val["y"])
        atomic_json(out / "clean_eval.json", _stamp({
            "partition": "validation (pilot subset)", "n_rows": int(len(yv)),
            "models": {v: {"accuracy": float(p.eq(yv).float().mean()), "n": int(len(yv))} for v, p in preds.items()},
            "int8_agreement": {"V2_vs_V1": float(preds["V2"].eq(preds["V1"]).float().mean()),
                               "V4_vs_V3": float(preds["V4"].eq(preds["V3"]).float().mean())}}))

    # ---- adversarial evaluation: FGSM and pgd20, full grid, every source
    torch.set_num_threads(threads["attack_and_evaluation"])
    ids_hash, label_hash = id_hash(ids_att), ma._label_hash(y_att)
    rows, union_rows, invalid, diagnostics, pair_models = [], [], [], {}, {}
    example = torch.from_numpy(to_sequence(att["x"][:2]))
    with phases("attacks"):
        for fp_id, q_id in (("V1", "V2"), ("V3", "V4")):
            fp, q = models[fp_id], models[q_id]
            bpda = bpda_model(q, convs[q_id]["requested"], example)
            surrogate, names = ma.build_surrogate(fp, cal["x"])
            fidelity = ma.surrogate_fidelity(surrogate, q, names, x_att)
            fidelity.pop("islands")
            diagnostics[q_id] = {"direct_int8_grad": ma.direct_int8_grad_diagnostic(q, x_att, y_att),
                                 "surrogate_fidelity": fidelity}
            pair_models[(fp_id, q_id)] = {"fp32": fp, "int8": q, "bpda": bpda, "surrogate": surrogate}
            model_ids = {"fp32": fp_id, "int8": q_id, "bpda": f"{q_id}_bpda", "surrogate": f"{q_id}_surrogate"}
            for attack_name in cfg["attack"]["budgets"]:
                for epsilon in cfg["attack"]["epsilon_grid"]:
                    results = []
                    for source in ma.SOURCES:
                        model_id = fp_id if source == "fp32_direct" else q_id
                        result = ma.evaluate_unit(
                            source=source, attack_name=attack_name, epsilon=epsilon, x=x_att, y=y_att, ids=ids_att,
                            models=pair_models[(fp_id, q_id)], cfg=att_cfg, train_seed=seed, model_id=model_id,
                            model_ids=model_ids, notes=[fidelity["note"]] if "surrogate" in source else None,
                            out_dir=_unit_dir(out, model_id, source, attack_name, epsilon),
                            expected_ids_sha256=ids_hash, expected_label_sha256=label_hash, raise_on_invalid=False)
                        if result["row"] is None or not result["checks"]["valid"]:
                            invalid.append({"model_id": model_id, "source": source, "attack": attack_name, "epsilon": epsilon,
                                            "failures": result["checks"]["failures"]})
                            log.error("invalid unit %s", invalid[-1])
                        if result["row"] is not None:
                            rows.append({**result["row"], "model_id": model_id})
                            if source != "fp32_direct" and result["checks"]["valid"]:
                                results.append(result)
                    if epsilon > 0 and len(results) == len(ma.SOURCES) - 1:
                        union_rows.append({**ma.union_row(results), "model_id": q_id})
        atomic_json(out / "attack_results.json", _stamp({"attack_note": ma.CLAIM_FRAMING, "rows": rows,
                                                         "union_rows": union_rows, "invalid_units": invalid}))
        m2_report.attack_results_csv(rows, out / "attack_results.csv")
        atomic_json(out / "gradient_diagnostics.json", _stamp(diagnostics))

    # ---- E4 plumbing (code path and cost only)
    cell_costs, cells = [], {}
    if cfg["e4_plumbing"]["enabled"]:
        with phases("e4_plumbing"):
            e4 = att_cfg["e4"]
            x_p, y_p, ids_p = torch.from_numpy(plumbing["x"]).float(), plumbing["y"], plumbing["ids"]
            for variant, source, pair in (("V1", "fp32_direct", ("V1", "V2")), ("V2", "bpda_int8", ("V1", "V2")),
                                          ("V3", "fp32_direct", ("V3", "V4")), ("V4", "bpda_int8", ("V3", "V4"))):
                for epsilon in e4["epsilons"]:
                    done = {}
                    for budget in (e4["candidate_budget"], e4["reference_budget"]):
                        done[budget] = ma.evaluate_unit(
                            source=source, attack_name=budget, epsilon=epsilon, x=x_p, y=y_p, ids=ids_p,
                            models=pair_models[pair], cfg=att_cfg, train_seed=seed, model_id=variant,
                            model_ids={"fp32": pair[0], "int8": pair[1], "bpda": f"{pair[1]}_bpda", "surrogate": f"{pair[1]}_surrogate"},
                            out_dir=_unit_dir(out, f"e4/{variant}", source, budget, epsilon), raise_on_invalid=False)
                    if any(r["row"] is None or not r["checks"]["valid"] for r in done.values()):
                        invalid.append({"model_id": variant, "source": source, "attack": "e4", "epsilon": epsilon})
                        continue
                    n_extra = ma.paired_n_extra(done[e4["reference_budget"]]["arrays"]["robust_correct"],
                                                done[e4["candidate_budget"]]["arrays"]["robust_correct"])
                    cells[(variant, float(epsilon))] = n_extra
                    cell_costs.append({"model": variant, "epsilon": float(epsilon), "n_extra": n_extra, "n": int(len(ids_p)),
                                       "seconds_pgd20": done[e4["candidate_budget"]]["row"]["attack_seconds"],
                                       "seconds_pgd50x5": done[e4["reference_budget"]]["row"]["attack_seconds"]})
            check = ma.budget_check(cells, len(plumbing["ids"]), att_cfg, plumbing_only=True)
            atomic_json(out / "attack_budget_check.json", _stamp({
                **check, "cell_costs": cell_costs, "plumbing_subset": describe(plumbing),
                "attacks_status": {k: v["status"] for k, v in att_cfg["attacks"].items()},
                "reason": cfg["e4_plumbing"]["reason"]}))
    if invalid:
        atomic_json(out / "invalid_units.json", _stamp({"invalid_units": invalid}))

    # ---- efficiency (reduced repetitions, 1 thread, fresh process per measurement)
    if cfg["efficiency"]["enabled"]:
        pair_models.clear()  # release attack models before the memory-sensitive timing children start
        gc.collect()
        with phases("efficiency"):
            fixed = me.select_rows(val["ids"], val["x"], eff_cfg["fixed_inputs"]["count"], split_seed)
            arch, p = train_cfg["architecture"], quant_cfg["ptq"]
            specs = {
                "V1": {"kind": "fp32", "checkpoint": out / "models/V1/checkpoint.pt", "architecture": arch},
                "V3": {"kind": "fp32", "checkpoint": out / "models/V3/checkpoint.pt", "architecture": arch}}
            for variant, parent in (("V2", "V1"), ("V4", "V3")):
                specs[variant] = {"kind": "int8", "state": out / "models" / variant / "state.pt",
                                  "float_checkpoint": out / "models" / parent / "checkpoint.pt",
                                  "calibration_path": out / "calibration_x.pt", "scope": p["scope"], "backend": p["backend"],
                                  "architecture": arch}
            ordered = {v: specs[v] for v in eff_cfg["variants"]}
            total = sum(t.numel() for t in models["V1"].parameters())
            me.run_efficiency(ordered, fixed, eff_cfg, out / "efficiency", pilot=True, conversions=convs,
                              total_parameters=total, banner=BANNER)

    # ---- QAT feasibility
    if cfg["qat_feasibility"]["enabled"]:
        with phases("qat_feasibility"):
            torch.set_num_threads(threads["training"])
            qat = mq.qat_feasibility(models["V1"], cal, train["x"], train["y"], train_cfg, quant_cfg, seed, val_seq,
                                     out / "qat", eval_y=val["y"])
            log.info("QAT feasibility: %s", qat["status"])

    # ---- runtime record, report
    ratio = est.training_cost_ratio(train_logs["V1"], train_logs["V3"])
    runtime = _stamp({
        "seed": seed, "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "threads": threads,
        "phases": phases.seconds, "total_seconds": time.perf_counter() - started, "ptq_seconds": ptq_seconds,
        "at_cost": ratio, "first_step_hook_counts_real_model": train_logs["V3"]["first_step_hook_counts"],
        "campaign_e4_subset": {k: e4_campaign[k] for k in ("n_subset", "rule", "id_sha256", "per_class_counts")},
        "n_invalid_units": len(invalid), "peak_rss_bytes": int(monitor.peak), "min_available_ram_bytes": int(monitor.min_available),
        "provenance": {k: prov[k] for k in ("revision", "branch", "source_digest", "upstream_revision", "model_source_hash")},
        "environment": environment(), "integrity_before_sha256": sha256(before),
        "integrity_after": "run scripts/check_m2_integrity.py --compare ... --write ... after the pilot (separate command)",
        "test_partition": "not loaded and not evaluated"})
    atomic_json(out / "pilot_runtime.json", runtime)
    (out / "pilot_report.md").write_text(m2_report.render_pilot_report(out), encoding="utf-8")
    log.info("pilot finished in %.1f s with %d invalid units", runtime["total_seconds"], len(invalid))
    return runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    runtime = run_pilot(args.config)
    print(json.dumps({"total_seconds": runtime["total_seconds"], "phases": runtime["phases"],
                      "n_invalid_units": runtime["n_invalid_units"]}, indent=2))
    return 1 if runtime["n_invalid_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
