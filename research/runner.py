"""Run bounded local experiments. No remote write or compute capabilities."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.ao import quantization as aq

from .attacks import attack
from .common import (ROOT, atomic_json, atomic_torch, digest, environment,
                     provenance, runtime_signature, seed_all, sha256, unit_status)
from .data import build_data, sequences, stratified_cap
from .evaluation import evaluate, metrics, transfer
from .quantization import bpda_model, inventory, load_int8, prepare_static, ptq
from .training import predictions, train


def validate_config(config):
    if config["dataset"] != "CICIoT" or config["model"] != "StandardTransformerIDS":
        raise ValueError("Only the pinned CICIoT multiclass Standard Transformer is supported")
    if config["tier"] == "full" and (len(config["seeds"]) != 4 or len(set(config["seeds"])) != 4):
        raise ValueError("Full comparisons require four distinct explicit training seeds")
    if config["tier"] == "full" and any(v is not None for v in config["per_class"].values()):
        raise ValueError("The full protocol must use every fixed-split row")
    if config["sequence_length"] != 50:
        raise ValueError("PAVE Standard Transformer sequence length must be 50")
    if config["quantization"]["mode"] not in ("static", "dynamic"):
        raise ValueError("Unsupported quantization mode")
    if config["training"]["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Invalid training budget")
    if config["quantization"]["scopes"] != ["classifier", "ffn", "attention", "all_linear"]:
        raise ValueError("All four requested ablations must be explicit")
    if 0 not in config["epsilons"]:
        raise ValueError("Include epsilon zero for sanity checks")


def calibration(data, config):
    flat, y = data["x"]["train"], data["y"]["train"].numpy()
    ix = stratified_cap(np.arange(len(y)), y, config["quantization"]["calibration_per_class"], config["split_seed"] + 99)
    return sequences(flat[ix], config["sequence_length"]), data["ids"]["train"][ix].tolist()


def save_quantized(folder, model, conversion, compatibility, calibration_ids,
                   float_model, scope, calibration_x):
    atomic_torch(folder / "state.pt", {"state_dict": model.state_dict(), "compatibility": compatibility})
    reloaded, _ = load_int8(float_model, scope, calibration_x, folder / "state.pt", compatibility,
                           mode="dynamic" if conversion["mode"] == "dynamic" else "static",
                           backend=conversion["backend"])
    with torch.no_grad():
        torch.testing.assert_close(model(calibration_x[:2])["logits"], reloaded(calibration_x[:2])["logits"], rtol=0, atol=0)
    atomic_json(folder / "conversion.json", {**conversion, "calibration_partition": "train",
                                             "calibration_row_ids": calibration_ids,
                                             "checkpoint_roundtrip_max_abs_error": 0.0,
                                             "state_sha256": sha256(folder / "state.pt")})


def select_attack(config, data, baseline, quantized, conversion, root, identity, resume):
    folder = root / "protocol_selection"
    with unit_status(folder, identity, resume) as execute:
        if execute:
            vy = data["y"]["validation"]
            ix = stratified_cap(np.arange(len(vy)), vy.numpy(), config["selection_per_class"], config["split_seed"] + 333)
            x = sequences(data["x"]["validation"][ix], config["sequence_length"])
            y = vy[ix]
            surrogate = bpda_model(quantized, conversion["requested"], x[:2])
            trials = []
            for candidate in config["attack_candidates"]:
                spec = {"kind": "pgd", "epsilon": config["focused_epsilon"],
                        "alpha": config["focused_epsilon"] * config["alpha_fraction"],
                        "random_start": True, **candidate}
                outcomes = []
                for name, model, grad_model in (("fp32", baseline, baseline), ("int8", quantized, surrogate)):
                    adv = torch.cat([attack(grad_model, xb, yb, **spec, lower=data["lower"], upper=data["upper"], seed=91000 + j)
                                     for j, (xb, yb) in enumerate(zip(x.split(config["batch_size"]), y.split(config["batch_size"])))] )
                    pred = predictions(model, adv, config["batch_size"]).argmax(1)
                    outcomes.append({"target": name, "adversarial_accuracy": float(pred.eq(y).float().mean())})
                trials.append({"candidate": candidate, "outcomes": outcomes,
                               "mean_adversarial_accuracy": float(np.mean([v["adversarial_accuracy"] for v in outcomes]))})
            # Lowest validation accuracy; ties favor a stronger computation budget.
            selected = min(trials, key=lambda r: (r["mean_adversarial_accuracy"], -r["candidate"]["steps"], -r["candidate"]["restarts"]))
            frozen = {"selection_partition": "validation", "training_seed": config["seeds"][0],
                      "validation_row_ids": data["ids"]["validation"][ix].tolist(),
                      "trials": trials, "selected": selected["candidate"],
                      "alpha_fraction": config["alpha_fraction"], "focused_epsilon": config["focused_epsilon"],
                      "threat_model": "untargeted L_inf; standardized numeric feature units; all 39 features mutable at each synthetic timestep; training feature bounds extended to include clean point",
                      "scope": "feature-space vulnerability; no packet/protocol realizability claim",
                      "identity": identity}
            atomic_json(folder / "frozen.json", frozen)
    frozen = json.loads((folder / "frozen.json").read_text())
    if frozen["identity"] != identity:
        raise ValueError("Frozen protocol is incompatible")
    return {**frozen["selected"], "random_start": True, "alpha_fraction": frozen["alpha_fraction"]}


def qat_feasibility(model, calibration_x, config, folder, compatibility, ids, x, y, data, seed, attacks):
    """A small actual converted QAT deployment pilot, never a full QAT study."""
    q, names = prepare_static(model, "all_linear", calibration_x[:2], qat=True,
                              backend=config["quantization"]["backend"])
    q.train()
    # BN does not occur in this architecture. Dropout stays in training mode;
    # observer/fake quantization is used only for training/calibration.
    opt = torch.optim.Adam(q.parameters(), lr=config["qat_learning_rate"])
    seed_all(seed + 800000)
    training_y = data["y"]["train"]
    start = time.perf_counter()
    for _ in range(config["qat_epochs"]):
        for flat, target in zip(data["x"]["train"].split(config["batch_size"]), training_y.split(config["batch_size"])):
            seq = sequences(flat, config["sequence_length"])
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(q(seq)["logits"], target)
            loss.backward()
            opt.step()
    q.eval()
    q.apply(aq.disable_observer)
    aq.convert(q, inplace=True)
    conversion = inventory(q, names, "qat_static")
    save_quantized(folder, q, conversion, compatibility, ids, model, "all_linear", calibration_x)
    bpda = bpda_model(q, names, x[:2])
    rows = evaluate(q, bpda, x, y, data["ids"][config["evaluation_partition"]], data,
                    attacks, [config["focused_epsilon"]], config, folder,
                    "qat_int8_feasibility", seed, "bpda_qat")
    atomic_json(folder / "feasibility.json", {"status": "actual converted INT8 inference verified",
                                              "claim": "focused one-seed feasibility pilot only; not full QAT scientific result",
                                              "fine_tuning_seconds": time.perf_counter() - start,
                                              "qat_epochs": config["qat_epochs"],
                                              "training_rows": len(training_y)})
    return rows


def expected_units(config, phases):
    names = ["fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear"]
    names += [f"ptq_{scope}" for scope in config["quantization"]["scopes"] if scope != "all_linear"]
    expected = {"train": [f"s{s}/{v}" for s in config["seeds"] for v in ("fp32", "at_fp32")],
                "evaluation": {p: [f"s{s}/{v}" for s in config["seeds"] for v in names] for p in phases},
                "transfer": {p: [f"s{s}/{name}" for s in config["seeds"] for name in (
                    "fp32_to_int8", "int8_to_fp32", "at_fp32_to_int8", "fp32_to_at_int8")] for p in phases}}
    if config["qat"]:
        expected["qat"] = {p: [f"s{config['seeds'][0]}/qat_int8_feasibility"] for p in phases}
    return expected


def run(config_path, run_dir, phase, resume):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.backends.quantized.engine = config["quantization"]["backend"]
    root = Path(run_dir).resolve()
    prov = provenance()
    env = environment()
    identity = digest({"config": config, "source_digest": prov["source_digest"],
                       "runtime": runtime_signature(env)})
    snapshot = root / "run.json"
    if root.exists():
        if not resume or not snapshot.exists():
            raise FileExistsError(f"Use a fresh run directory or compatible --resume: {root}")
        if json.loads(snapshot.read_text())["identity"] != identity:
            raise ValueError("Configuration, source, or environment changed; use a new run directory")
    else:
        root.mkdir(parents=True)
        atomic_json(snapshot, {"identity": identity, "config": config, "provenance": prov,
                               "environment": env, "created_unix": time.time()})
    if phase == "both" and config["tier"] == "full":
        raise ValueError("Full test evaluation requires a separate invocation after validation and protocol freeze")
    if phase == "test" and not (root / "protocol_selection/status.json").exists():
        raise RuntimeError("Run --phase validation first; test is excluded from protocol selection")
    data = build_data(config, root / "data")
    compatibility = {"run_identity": identity, "manifest_sha256": data["manifest"]["manifest_sha256"],
                     "preprocessing_sha256": data["manifest"]["preprocessing_sha256"],
                     "split_sha256": data["manifest"]["split_sha256"]}
    model_config = {"input_dim": 39, "d_model": 96, "n_heads": 4, "n_layers": 3,
                    "d_ff": 384, "n_classes": 27, "max_seq_len": 50, "dropout": 0.1}
    cal, cal_ids = calibration(data, config)
    cal_meta = {"row_ids": cal_ids, "partition": "train", "row_hash": digest(cal_ids)}
    atomic_json(root / "data/calibration.json", cal_meta)
    phases = ["validation", "test"] if phase == "both" else [phase]
    atomic_json(root / f"expected_{phase}.json", expected_units(config, phases))
    for seed in config["seeds"]:
        fp, fp_meta = train(config, data, model_config, seed, "fp32", root / f"train/s{seed}/fp32", compatibility, resume)
        q, conversion = ptq(fp, "all_linear", cal, config["quantization"]["mode"], config["quantization"]["backend"])
        attacks = select_attack(config, data, fp, q, conversion, root, identity, resume or seed != config["seeds"][0])
        at, at_meta = train(config, data, model_config, seed, "at_fp32", root / f"train/s{seed}/at_fp32", compatibility, resume)
        atq, at_conversion = ptq(at, "all_linear", cal, config["quantization"]["mode"], config["quantization"]["backend"])
        variants = [("fp32", fp, None, fp_meta), ("ptq_all_linear", q, conversion, fp_meta),
                    ("at_fp32", at, None, at_meta), ("at_ptq_all_linear", atq, at_conversion, at_meta)]
        for scope in config["quantization"]["scopes"]:
            if scope != "all_linear":
                ablation, inv = ptq(fp, scope, cal, config["quantization"]["mode"], config["quantization"]["backend"])
                variants.append((f"ptq_{scope}", ablation, inv, fp_meta))
        for partition in phases:
            cfg = {**config, "evaluation_partition": partition}
            x = sequences(data["x"][partition], config["sequence_length"])
            y = data["y"][partition]
            for name, model, inv, checkpoint_meta in variants:
                folder = root / f"{partition}/evaluation/s{seed}/{name}"
                with unit_status(folder, digest({"base": compatibility, "seed": seed, "name": name, "partition": partition}), resume) as execute:
                    if execute:
                        grad_model = bpda_model(model, inv["requested"], x[:2]) if inv else model
                        if inv:
                            float_model = at if name.startswith("at_") else fp
                            scope = name.replace("at_", "", 1).removeprefix("ptq_")
                            save_quantized(folder, model, inv, compatibility, cal_ids, float_model, scope, cal)
                        atomic_json(folder / "checkpoint_source.json", checkpoint_meta)
                        eps = config["epsilons"] if name in ("fp32", "ptq_all_linear", "at_fp32", "at_ptq_all_linear") else [config["focused_epsilon"]]
                        evaluate(model, grad_model, x, y, data["ids"][partition], data, attacks, eps, cfg,
                                 folder, name, seed, "bpda_actual_int8_forward" if inv else "fp32_autograd")
            qbpda = bpda_model(q, conversion["requested"], x[:2])
            directions = [("fp32_to_int8", fp, q, "FP32 PGD transfer"),
                          ("int8_to_fp32", qbpda, fp, "actual INT8 forward / BPDA gradients -> FP32 transfer"),
                          ("at_fp32_to_int8", at, q, "AT-FP32 PGD -> ordinary INT8 transfer"),
                          ("fp32_to_at_int8", fp, atq, "FP32 PGD -> AT-INT8 transfer")]
            transfer_spec = {k: v for k, v in attacks.items() if k != "alpha_fraction"}
            transfer_spec.update(kind="pgd", epsilon=config["focused_epsilon"], alpha=config["focused_epsilon"] * attacks["alpha_fraction"])
            for name, source, target, source_kind in directions:
                folder = root / f"{partition}/transfer/s{seed}/{name}"
                with unit_status(folder, digest({"base": compatibility, "seed": seed, "transfer": name, "partition": partition}), resume) as execute:
                    if execute:
                        transfer(source, target, x, y, data["ids"][partition], data, transfer_spec, cfg, folder, name, seed, source_kind)
            if config["qat"] and seed == config["seeds"][0]:
                folder = root / f"{partition}/qat/s{seed}/qat_int8_feasibility"
                with unit_status(folder, digest({"base": compatibility, "qat": seed, "partition": partition}), resume) as execute:
                    if execute:
                        qat_feasibility(fp, cal, cfg, folder, compatibility, cal_ids, x, y, data, seed, attacks)
    from .validate import validate_run
    validate_run(root, phase)
    atomic_json(root / f"completed_{phase}.json", {"status": "completed", "identity": identity, "phases": phases})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("validation", "test", "both"), default="validation")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible completed units only; partial units are never overwritten")
    args = parser.parse_args()
    root = Path(args.run_dir)
    # A fresh log on every invocation; prior logs are preserved.
    log_dir = ROOT / "work" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"runner-{time.time_ns()}.log"
    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log
        def write(self, value):
            self.original.write(value)
            self.log.write(value)
        def flush(self):
            self.original.flush()
            self.log.flush()
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            print(f"raw log: {log_path}", flush=True)
            try:
                result = run(args.config, args.run_dir, args.phase, args.resume)
                print(f"Validated completed run: {result}", flush=True)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                if (root / "run.json").is_file():
                    stream.flush()
                    destination = root / "logs"
                    destination.mkdir(exist_ok=True)
                    shutil.copyfile(log_path, destination / log_path.name)


if __name__ == "__main__":
    main()
