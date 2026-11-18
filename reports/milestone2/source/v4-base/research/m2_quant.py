"""Milestone 2A INT8 post-training quantization unit, conversion verification and QAT feasibility check.

Reuses research.quantization (eager static PTQ on the 23 LinearIsland modules) and research.runner.save_quantized;
adds the M2 calibration adapter (train partition only), the checks that keep a fake-quantized or unconverted model
from being labelled INT8, the conversion.json evidence, bitwise reload verification and a bounded QAT check.
The converted model is hybrid: INT8 Linear islands, FP32 everything else.
Design: docs/MILESTONE2_PROTOCOL.md section C; configuration: configs/m2/quantization.json.
"""
from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.ao import quantization as aq
from torch.ao.nn.quantized import DeQuantize, Quantize
from torch.ao.nn.quantized import Linear as QuantizedLinear
from torch.profiler import ProfilerActivity, profile

from .common import ROOT, atomic_json
from .data import stratified_cap
from .m2_attacks import model_state_hash
from .m2_split import ConfigError, id_hash
from .m2_training import (CheckpointError, TrainingInterrupted, load_epoch_checkpoint, restore_rng,
                          save_epoch_checkpoint)
from .m2b_io import durable_write_json, durable_write_torch
from .paper_described_adam import build_optimizer, derived_seed, epoch_batches, to_sequence
from .quantization import LinearIsland, inventory, load_int8, prepare_static, ptq, selected_names
from .runner import save_quantized

DEFAULT_CONFIG = ROOT / "configs/m2/quantization.json"
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "ptq", "serialization", "qat_feasibility", "campaign_qat"}
CLAIM = ("INT8 here means the 23 nn.Linear islands run as torch.ao.nn.quantized.Linear with qint8 per-channel packed "
         "weights; attention matmuls, softmax, LayerNorm, residual adds, ReLU, positional encoding, pooling and the "
         "scorer sigmoid stay FP32 (hybrid model, not integer-only).")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"quantization configuration not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg) != TOP_KEYS:
        raise ConfigError(f"unknown or missing quantization configuration keys: {sorted(set(cfg) ^ TOP_KEYS)}")
    p = cfg["ptq"]
    if (p["scope"], p["mode"], p["backend"]) != ("all_linear", "static", "fbgemm"):
        raise ConfigError("PTQ workflow must be all_linear / static / fbgemm")
    if (p["weight_dtype"], p["weight_qscheme"], p["activation_dtype"]) != ("qint8", "per_channel_affine", "quint8"):
        raise ConfigError("PTQ must use per-channel qint8 weights and quint8 activations")
    if p["calibration_partition"] != "train" or not isinstance(p["calibration_per_class"], int) or p["calibration_per_class"] < 1:
        raise ConfigError("calibration must come from the train partition with a positive per-class count")
    if cfg["qat_feasibility"]["epochs"] < 1 or cfg["campaign_qat"]["finetune_epochs_k"] < 1:
        raise ConfigError("QAT epochs must be positive")
    return cfg


# ------------------------------------------------------------------ calibration
def calibration_ids(arrays: dict, per_class: int, split_seed: int, *, seed_name: str = "calibration",
                    length: int = 50) -> dict:
    """Calibration rows from the TRAIN partition only (`arrays` as returned by m2_split.load_arrays).

    Rows are drawn with data.stratified_cap and seed derived_seed('calibration', split_seed); the result is
    shared by V2, V4, the QAT feasibility run and the surrogate. Returns ids, hash, per-class counts and the
    noise-free repeated sequences.
    """
    train = arrays["train"]
    ids, y = np.asarray(train["ids"]), np.asarray(train["y"])
    seed = derived_seed(seed_name, split_seed)
    local = stratified_cap(np.arange(len(y)), y, per_class, seed)
    selected = ids[local]
    return {"partition": "train", "per_class": per_class, "seed_rule": f"derived_seed('{seed_name}', split_seed)",
            "split_seed": split_seed, "seed": seed, "ids": [int(i) for i in selected], "local_index": local,
            "n": int(len(local)), "id_sha256": id_hash(selected),
            "per_class_counts": np.bincount(y[local], minlength=int(y.max()) + 1).tolist(),
            "x": torch.from_numpy(to_sequence(np.asarray(train["x"])[local], length))}


def save_calibration(calibration: dict, path: Path | str) -> None:
    """Write the calibration ID list and hash once (refuses to overwrite)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    atomic_json(path, {k: calibration[k] for k in ("partition", "per_class", "seed_rule", "split_seed", "seed", "n",
                                                    "id_sha256", "per_class_counts", "ids")})


# ------------------------------------------------------------------ verification
def verify_identity_wrapper(model, example, *, scope: str = "all_linear", backend: str = "fbgemm") -> float:
    """The island wrapper reproduces the FP32 logits with atol=0 before observers run (returns the max error)."""
    prepared, _ = prepare_static(model, scope, example, backend=backend)
    with torch.inference_mode():
        err = float((model.eval()(example)["logits"] - prepared(example)["logits"]).abs().max())
    if err != 0.0:
        raise RuntimeError(f"island wrapper does not reproduce FP32 logits exactly (max error {err})")
    return err


def profile_quantized_ops(model, example) -> dict:
    """Run one forward under torch.profiler and list the quantized operators that executed."""
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            model(example)
    rows = [(e.key, e.count) for e in prof.key_averages() if "quantiz" in e.key]
    return {"operators": {k: c for k, c in sorted(rows)},
            "quantized_linear_calls": sum(c for k, c in rows if k == "quantized::linear")}


def verify_converted(q, requested, *, float_model=None, example=None, expected_islands: int | None = None,
                     profile_ops: bool = True) -> dict:
    """Reject anything that is not a genuinely converted INT8 model; return the structure evidence.

    Requires every requested island to be torch.ao.nn.quantized.Linear with qint8 per_channel_affine packed
    weights, and no FakeQuantize, observer, stub or float nn.Linear module anywhere. With `example`, a profiled
    forward must execute quantized::linear. A model failing any check is never labelled INT8 (RuntimeError).
    """
    problems, packed = [], []
    for name in requested:
        island = q.get_submodule(name)
        if not isinstance(island, LinearIsland):
            problems.append(f"{name}: not a LinearIsland ({type(island).__name__})")
            continue
        lin = island.linear
        if type(lin) is not QuantizedLinear:
            problems.append(f"{name}: linear is {type(lin).__module__}.{type(lin).__name__}, not torch.ao.nn.quantized.Linear")
            continue
        if type(island.quant) is not Quantize or type(island.dequant) is not DeQuantize:
            problems.append(f"{name}: quant/dequant stubs were not converted")
        weight = lin.weight()
        if not weight.is_quantized or weight.dtype != torch.qint8 or weight.qscheme() != torch.per_channel_affine:
            problems.append(f"{name}: weight is {weight.dtype}/{weight.qscheme() if weight.is_quantized else 'float'}, "
                            "not qint8 per_channel_affine")
        packed.append({"name": name, "class": f"{type(lin).__module__}.{type(lin).__name__}",
                       "weight_dtype": str(weight.dtype), "weight_qscheme": str(weight.qscheme()),
                       "weight_shape": list(weight.shape), "n_weight_scales": int(weight.q_per_channel_scales().numel()),
                       "input_scale": float(island.quant.scale), "output_scale": float(lin.scale),
                       "output_zero_point": int(lin.zero_point)})
    for name, module in q.named_modules():
        if isinstance(module, (aq.FakeQuantizeBase, aq.ObserverBase)):
            problems.append(f"{name or '<root>'}: {type(module).__name__} left in the converted model")
        elif isinstance(module, (aq.QuantStub, aq.DeQuantStub)):
            problems.append(f"{name}: unconverted {type(module).__name__}")
        elif isinstance(module, nn.Linear):
            problems.append(f"{name}: float {type(module).__module__}.{type(module).__name__} left in the converted model")
    if float_model is not None and sorted(requested) != sorted(selected_names(float_model, "all_linear")):
        problems.append("requested islands differ from the FP32 model's Linear modules")
    if expected_islands is not None and len(requested) != expected_islands:
        problems.append(f"{len(requested)} islands, expected {expected_islands}")
    profiled = None
    if example is not None and profile_ops and not problems:
        profiled = profile_quantized_ops(q, example)
        if profiled["quantized_linear_calls"] < len(requested):
            problems.append(f"profiler saw {profiled['quantized_linear_calls']} quantized::linear calls for "
                            f"{len(requested)} islands")
    if problems:
        raise RuntimeError("converted model rejected, not labelled INT8: " + "; ".join(problems))
    return {"n_islands": len(requested), "packed_weights": packed,
            "weight_dtype_inventory": {"torch.qint8": len(packed)}, "weight_qscheme_inventory": {"torch.per_channel_affine": len(packed)},
            "no_fake_quant_observer_or_float_linear": True, "profiler": profiled,
            "quantize_modules": sum(isinstance(m, Quantize) for m in q.modules()),
            "dequantize_modules": sum(isinstance(m, DeQuantize) for m in q.modules())}


def parameter_counts(float_model, requested) -> dict:
    total = sum(p.numel() for p in float_model.parameters())
    int8_weights = bias = anomaly = 0
    for name in requested:
        lin = float_model.get_submodule(name)
        int8_weights += lin.weight.numel()
        bias += lin.bias.numel() if lin.bias is not None else 0
        anomaly += lin.weight.numel() if name.startswith("anomaly_scorer.") else 0
    return {"total_parameters": total, "int8_weight_elements": int8_weights,
            "fp32_bias_elements_in_quantized_linear": bias,
            "fp32_parameters_not_quantized": total - int8_weights, "fp32_other_parameters": total - int8_weights - bias,
            "int8_weight_share": int8_weights / total,
            "anomaly_scorer_int8_weight_elements": anomaly,
            "classification_path_int8_weight_elements": int8_weights - anomaly}


def build_conversion(q, inv, float_model, structure, ptq_cfg, calibration) -> dict:
    requested = inv["requested"]
    anomaly = sum(n.startswith("anomaly_scorer.") for n in requested)
    return {**inv, "scope": ptq_cfg["scope"], "claim": CLAIM, "int8_label_verified": True,
            "fp32_remaining_operations": ptq_cfg["fp32_remaining_operations"],
            "parameter_counts": parameter_counts(float_model, requested),
            "quant_dequant_boundaries": {"quantize": structure["quantize_modules"], "dequantize": structure["dequantize_modules"],
                                         "total": structure["quantize_modules"] + structure["dequantize_modules"]},
            "classification_path_islands": len(requested) - anomaly, "anomaly_scorer_islands": anomaly,
            "structure_checks": structure,
            "calibration": {"partition": "train", "n_rows": calibration["n"], "id_sha256": calibration["id_sha256"],
                            "per_class": calibration["per_class"], "seed_rule": calibration["seed_rule"]}}


def reload_check(q, float_model, scope, calibration_x, state_path, compatibility, eval_x, *,
                 backend: str = "fbgemm", batch_size: int = 256) -> dict:
    """Reload the saved INT8 state with quantization.load_int8: logits must be bitwise equal, argmax identical."""
    reloaded, _ = load_int8(float_model, scope, calibration_x, state_path, compatibility, "static", backend)
    mismatched, rows = 0, 0
    with torch.inference_mode():
        for batch in eval_x.split(batch_size):
            a, b = q(batch)["logits"], reloaded(batch)["logits"]
            if not torch.equal(a, b) or not torch.equal(a.argmax(1), b.argmax(1)):
                mismatched += 1
            rows += len(batch)
    if mismatched:
        raise RuntimeError(f"reloaded INT8 model differs from the saved one in {mismatched} batches")
    return {"rows": rows, "logits_bitwise_equal": True, "argmax_identical": True,
            "reloaded_state_hash": model_state_hash(reloaded), "saved_state_hash": model_state_hash(q)}


# ------------------------------------------------------------------ PTQ unit
def ptq_unit(model, calibration: dict, cfg: dict, out_dir: Path | str | None = None, *, variant: str = "V2",
             expected_islands: int | None = None, eval_x=None) -> dict:
    """INT8 PTQ of one FP32 model with the frozen workflow and the shared calibration rows.

    Returns {'model', 'conversion'}. With out_dir the packed state and conversion.json are written
    (runner.save_quantized) and a bitwise reload check over `eval_x` (default: the calibration rows) is added.
    """
    p = cfg["ptq"]
    expected = p["expected_quantized_linear"] if expected_islands is None else expected_islands
    model = model.cpu().eval()
    example = calibration["x"]
    identity_error = verify_identity_wrapper(model, example[:2], scope=p["scope"], backend=p["backend"])
    q, inv = ptq(model, p["scope"], example, p["mode"], p["backend"])
    inv["architecture_wrapper_fp32_max_abs_error"] = identity_error
    structure = verify_converted(q, inv["requested"], float_model=model, example=example[:4], expected_islands=expected)
    conversion = build_conversion(q, inv, model, structure, p, calibration)
    conversion["variant"] = variant
    conversion["float_state_hash"] = model_state_hash(model)
    conversion["int8_state_hash"] = model_state_hash(q)
    result = {"model": q, "conversion": conversion}
    if out_dir is not None:
        out_dir = Path(out_dir)
        if (out_dir / "state.pt").exists() or (out_dir / "conversion.json").exists():
            raise FileExistsError(f"refusing to overwrite quantized output in {out_dir}")
        compatibility = {"variant": variant, "float_state_hash": conversion["float_state_hash"],
                         "calibration_id_sha256": calibration["id_sha256"], "scope": p["scope"],
                         "mode": p["mode"], "backend": p["backend"]}
        save_quantized(out_dir, q, conversion, compatibility, calibration["ids"], model, p["scope"], example)
        reload = reload_check(q, model, p["scope"], example, out_dir / "state.pt", compatibility,
                              example if eval_x is None else eval_x, backend=p["backend"])
        saved = json.loads((out_dir / "conversion.json").read_text(encoding="utf-8"))
        saved["reload_check"] = reload
        atomic_json(out_dir / "conversion.json", saved)
        conversion["reload_check"] = reload
    return result


# ------------------------------------------------------------------ QAT feasibility
class _LastModule:
    """Remembers the most recent module entered by a forward pass (to name a failing module)."""

    def __init__(self, model):
        self.name = None
        self.handles = [m.register_forward_pre_hook(self._hook(n)) for n, m in model.named_modules() if n]

    def _hook(self, name):
        def hook(module, args):
            self.name = name
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def _qat_finetune(q, X, y, train_cfg, qat_cfg, seed, epochs):
    cfg = copy.deepcopy(train_cfg)
    cfg["training"]["optimizer_kwargs"].update(lr=qat_cfg["learning_rate"], weight_decay=qat_cfg["weight_decay"])
    opt = build_optimizer(cfg, q)
    length, std, bs = cfg["sequence"]["length"], cfg["sequence"]["train_noise_std"], qat_cfg["batch_size"]
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    losses = []
    for ep in range(epochs):
        q.train()
        g = torch.Generator(device="cpu").manual_seed(derived_seed("shuffle", seed, ep))
        perm = torch.randperm(len(Xt), generator=g)
        total = 0.0
        for b, idx in enumerate(epoch_batches(len(Xt), bs, perm)):
            ng = torch.Generator(device="cpu").manual_seed(derived_seed("noise", seed, ep, b))
            rows = Xt[idx]
            noise = torch.randn((len(idx), length, rows.shape[1]), generator=ng, dtype=torch.float32) * std
            opt.zero_grad()
            loss = F.cross_entropy(q(rows[:, None, :].expand(-1, length, -1) + noise)["logits"], yt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite QAT loss at epoch {ep + 1}, batch {b}")
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        losses.append(total / len(Xt))
    return losses


def qat_feasibility(model, calibration: dict, train_x, train_y, train_cfg: dict, cfg: dict, seed: int, eval_x,
                    out_dir: Path | str | None = None, *, eval_y=None, epochs: int | None = None,
                    expected_islands: int | None = None) -> dict:
    """Bounded QAT feasibility check (optional V5): prepare, fine-tune, freeze observers, convert, run real INT8.

    Never raises for a QAT limitation: any failure is recorded as status 'unsupported' with the exact exception,
    stage and module. `train_x` are MinMax-scaled training rows (N, 39); `eval_x` are (M, 50, 39) validation sequences.
    """
    qat_cfg = cfg["qat_feasibility"]
    epochs = qat_cfg["epochs"] if epochs is None else epochs
    expected = cfg["ptq"]["expected_quantized_linear"] if expected_islands is None else expected_islands
    record = {"status": None, "error": None, "module": None, "stage": None, "where": None, "packed_dtypes": None,
              "epochs": epochs, "train_rows": int(len(train_x)), "learning_rate": qat_cfg["learning_rate"],
              "claim": "bounded feasibility check only; not a QAT result and not comparable with PTQ without the matched FP32-FT controls",
              "backend": qat_cfg["backend"], "calibration_id_sha256": calibration["id_sha256"]}
    stage, tracker, q = "prepare", None, None
    rng_state = torch.get_rng_state().clone()
    started = time.perf_counter()
    try:
        torch.manual_seed(derived_seed("qat-ft", seed))
        model = model.cpu().eval()
        q, names = prepare_static(model, qat_cfg["scope"], calibration["x"][:2], qat=True, backend=qat_cfg["backend"])
        tracker = _LastModule(q)
        stage = "finetune"
        record["epoch_losses"] = _qat_finetune(q, train_x, train_y, train_cfg, qat_cfg, seed, epochs)
        stage = "convert"
        q.eval()
        q.apply(aq.disable_observer)
        aq.convert(q, inplace=True)
        tracker.close()
        tracker = None
        stage = "inventory"
        inv = inventory(q, names, "qat_static")
        structure = verify_converted(q, names, float_model=model, example=eval_x[:4], expected_islands=expected)
        record["packed_dtypes"] = {"weight_dtype": structure["weight_dtype_inventory"],
                                   "weight_qscheme": structure["weight_qscheme_inventory"],
                                   "per_island": [{k: p[k] for k in ("name", "weight_dtype", "weight_qscheme", "weight_shape")}
                                                  for p in structure["packed_weights"]]}
        record["quantized_ops_profiled"] = structure["profiler"]
        stage = "inference"
        tracker = _LastModule(q)
        with torch.inference_mode():
            logits = torch.cat([q(b)["logits"] for b in eval_x.split(256)])
            parent = torch.cat([model(b)["logits"] for b in eval_x.split(256)])
        tracker.close()
        tracker = None
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite INT8 logits after QAT conversion")
        record["int8_inference"] = {"rows": int(len(logits)), "logits_finite": True,
                                    "argmax_agreement_with_fp32_parent": float(logits.argmax(1).eq(parent.argmax(1)).float().mean())}
        if eval_y is not None:
            record["int8_inference"]["accuracy_on_eval_rows"] = float(logits.argmax(1).eq(torch.as_tensor(np.asarray(eval_y))).float().mean())
        if out_dir is not None:
            conversion = build_conversion(q, inv, model, structure, cfg["ptq"], calibration)
            conversion["variant"] = "V5_qat_feasibility"
            save_quantized(Path(out_dir) / "qat_int8", q, conversion,
                           {"variant": "V5_qat_feasibility", "float_state_hash": model_state_hash(model),
                            "calibration_id_sha256": calibration["id_sha256"]},
                           calibration["ids"], model, qat_cfg["scope"], calibration["x"])
            record["serialized"] = True
        record["status"] = "supported"
    except Exception as error:  # noqa: BLE001 - every limitation must be recorded exactly, not raised
        frames = traceback.extract_tb(error.__traceback__)
        message = str(error)
        module = tracker.name if tracker is not None and stage in ("finetune", "inference") else None
        if module is None and "was not converted:" in message:
            module = message.rsplit(":", 1)[1].strip()
        record.update(status="unsupported", error=f"{type(error).__name__}: {message}", stage=stage, module=module,
                      where=f"{Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name}" if frames else None)
    finally:
        if tracker is not None:
            tracker.close()
        torch.set_rng_state(rng_state)
    record["seconds"] = time.perf_counter() - started
    if record["status"] == "supported":
        record["stage"] = "inference"
    if out_dir is not None:
        path = Path(out_dir) / "qat_feasibility.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        atomic_json(path, record)
    return record


# ------------------------------------------------------------------ campaign QAT arms (M2B, DESIGN 4.9)
class QATStageError(RuntimeError):
    """QAT failed at a named stage; the campaign records {stage, module, exception} and blocks the unit."""

    def __init__(self, stage, module, original):
        super().__init__(f"QAT failed at stage {stage!r}" + (f" in module {module!r}" if module else "")
                         + f": {type(original).__name__}: {original}")
        self.stage, self.module, self.original = stage, module, original

    def record(self) -> dict:
        return {"stage": self.stage, "module": self.module,
                "exception": f"{type(self.original).__name__}: {self.original}"}


def finetune_settings(cfg: dict, epochs: int | None = None) -> dict:
    """Matched fine-tuning settings of both QAT-campaign arms: k epochs, Adam lr/wd, batch 512 (quantization.json)."""
    c, f = cfg["campaign_qat"], cfg["qat_feasibility"]
    return {"epochs": c["finetune_epochs_k"] if epochs is None else epochs, "learning_rate": c["learning_rate"],
            "weight_decay": c["weight_decay"], "batch_size": f["batch_size"]}


def _finetune(model, X, y, train_cfg, ft, seed, epochs, *, checkpoint_dir=None, stop_flag=None, fingerprints=None,
              threads=None, variant="FT", on_epoch=None) -> dict:
    """k epochs of Adam fine-tuning with the data order and noise streams of training (derived_seed shuffle/noise).

    Identical for the FP32-FT control and the QAT arm (fake quantization is only a property of `model`). With
    checkpoint_dir an epoch-boundary checkpoint (model, optimizer, torch/numpy/python RNG) is written after every epoch
    and an existing one is resumed; stop_flag() is polled after each checkpoint and raises TrainingInterrupted.
    """
    cfg = copy.deepcopy(train_cfg)
    cfg["training"]["optimizer_kwargs"].update(lr=ft["learning_rate"], weight_decay=ft["weight_decay"])
    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    run_threads = torch.get_num_threads()
    try:
        torch.manual_seed(derived_seed("qat-ft", seed))
        opt = build_optimizer(cfg, model)
        init_hash = model_state_hash(model)
        length, std, bs = cfg["sequence"]["length"], cfg["sequence"]["train_noise_std"], ft["batch_size"]
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        losses, updates, start, resumed_from, prior_wall = [], 0, 0, None, 0.0
        if checkpoint_dir is not None:
            payload = load_epoch_checkpoint(checkpoint_dir, threads=run_threads, fingerprints=fingerprints)
            if payload is not None:
                if payload["init_state_hash"] != init_hash or payload["seed"] != seed:
                    raise CheckpointError(f"{variant}: checkpoint belongs to a different parent or seed")
                model.load_state_dict(payload["model_state"])
                opt.load_state_dict(payload["optimizer_state"])
                start, updates, losses = payload["epoch_done"], payload["updates"], list(payload["epoch_losses"])
                prior_wall, resumed_from = payload["extras"].get("wall_seconds", 0.0), start
                restore_rng(payload["rng"])  # last, so nothing consumed randomness before the first resumed batch
        started = time.perf_counter()
        for ep in range(start, epochs):
            model.train()
            g = torch.Generator(device="cpu").manual_seed(derived_seed("shuffle", seed, ep))
            perm = torch.randperm(len(Xt), generator=g)
            total = 0.0
            for b, idx in enumerate(epoch_batches(len(Xt), bs, perm)):
                ng = torch.Generator(device="cpu").manual_seed(derived_seed("noise", seed, ep, b))
                rows = Xt[idx]
                noise = torch.randn((len(idx), length, rows.shape[1]), generator=ng, dtype=torch.float32) * std
                opt.zero_grad()
                loss = F.cross_entropy(model(rows[:, None, :].expand(-1, length, -1) + noise)["logits"], yt[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {variant} loss at epoch {ep + 1}, batch {b}")
                loss.backward()
                opt.step()
                updates += 1
                total += float(loss.item()) * len(idx)
            losses.append(total / len(Xt))
            if on_epoch is not None:
                on_epoch(ep + 1, losses[-1])
            if checkpoint_dir is not None:
                save_epoch_checkpoint(checkpoint_dir, epoch=ep + 1, model=model, opt=opt, updates=updates,
                                      epoch_losses=losses, step_seconds=[], attack_seconds=[], init_state_hash=init_hash,
                                      seed=seed, adversarial=False, threads=run_threads, fingerprints=fingerprints,
                                      extras={"variant": variant, "wall_seconds": prior_wall + time.perf_counter() - started})
                if ep + 1 < epochs and stop_flag is not None and stop_flag():
                    raise TrainingInterrupted(f"{variant}: stopped after epoch {ep + 1} of {epochs}")
        log = {"variant": variant, "epochs": epochs, "epoch_losses": losses, "optimizer_updates": updates,
               "learning_rate": ft["learning_rate"], "weight_decay": ft["weight_decay"], "batch_size": bs,
               "seed": seed, "init_state_hash": init_hash, "torch_threads": run_threads,
               "wall_clock_seconds": prior_wall + time.perf_counter() - started}
        if resumed_from is not None:
            log["resumed_from_epoch"] = resumed_from
        return log
    finally:
        torch.set_num_threads(previous_threads)


def fp32_finetune_unit(parent, X, y, train_cfg: dict, cfg: dict, seed: int, epochs: int | None = None,
                       out_dir: Path | str | None = None, checkpoint_dir: Path | str | None = None, *, stop_flag=None,
                       fingerprints=None, threads=None) -> dict:
    """Matched control: k epochs of FP32 fine-tuning of `parent` with the QAT arm's settings and no fake quantization."""
    ft = finetune_settings(cfg, epochs)
    if out_dir is not None and ((Path(out_dir) / "checkpoint.pt").exists() or (Path(out_dir) / "train_log.json").exists()):
        raise FileExistsError(f"refusing to overwrite fine-tuning output in {out_dir}")
    model = copy.deepcopy(parent).cpu()
    log = _finetune(model, X, y, train_cfg, ft, seed, ft["epochs"], checkpoint_dir=checkpoint_dir, stop_flag=stop_flag,
                    fingerprints=fingerprints, threads=threads, variant="FP32-FT")
    model.eval()
    log["final_state_hash"] = model_state_hash(model)
    log["parent_state_hash"] = model_state_hash(parent)
    if out_dir is not None:
        out_dir = Path(out_dir)
        durable_write_torch(out_dir / "checkpoint.pt", {"state_dict": model.state_dict(), "epoch": ft["epochs"],
                                                        "architecture": train_cfg["architecture"], "seed": seed,
                                                        "variant": "FP32-FT"})
        durable_write_json(out_dir / "train_log.json", log)
    return {"model": model, "log": log}


def qat_unit(parent, calibration: dict, X, y, train_cfg: dict, cfg: dict, seed: int, epochs: int | None = None,
             out_dir: Path | str | None = None, checkpoint_dir: Path | str | None = None, *, eval_x=None,
             expected_islands: int | None = None, stop_flag=None, fingerprints=None, threads=None) -> dict:
    """QAT arm: prepare(qat), k fine-tuning epochs, freeze observers, convert, verify_converted, bitwise reload check.

    Unlike qat_feasibility no exception is swallowed: a failure raises QATStageError naming stage, module and exception
    (the campaign turns it into a documented `blocked` unit). Interruption and refusal-to-overwrite errors pass through.
    """
    ft = finetune_settings(cfg, epochs)
    qcfg = cfg["qat_feasibility"]
    expected = cfg["ptq"]["expected_quantized_linear"] if expected_islands is None else expected_islands
    if out_dir is not None and (Path(out_dir) / "state.pt").exists():
        raise FileExistsError(f"refusing to overwrite quantized output in {out_dir}")
    stage, tracker = "prepare", None
    try:
        parent = parent.cpu().eval()
        q, names = prepare_static(parent, qcfg["scope"], calibration["x"][:2], qat=True, backend=qcfg["backend"])
        tracker = _LastModule(q)
        stage = "finetune"
        log = _finetune(q, X, y, train_cfg, ft, seed, ft["epochs"], checkpoint_dir=checkpoint_dir, stop_flag=stop_flag,
                        fingerprints=fingerprints, threads=threads, variant="QAT-FT")
        tracker.close()
        tracker = None
        stage = "convert"
        q.eval()
        q.apply(aq.disable_observer)
        aq.convert(q, inplace=True)
        stage = "verify_converted"
        example = calibration["x"] if eval_x is None else eval_x
        inv = inventory(q, names, "qat_static")
        structure = verify_converted(q, names, float_model=parent, example=example[:4], expected_islands=expected)
        conversion = build_conversion(q, inv, parent, structure, cfg["ptq"], calibration)
        conversion.update(variant="QAT-FT", float_state_hash=model_state_hash(parent), int8_state_hash=model_state_hash(q),
                          qat_training=log)
        result = {"model": q, "conversion": conversion, "log": log}
        if out_dir is not None:
            stage = "reload_check"
            out_dir = Path(out_dir)
            compatibility = {"variant": "QAT-FT", "float_state_hash": conversion["float_state_hash"],
                             "calibration_id_sha256": calibration["id_sha256"], "scope": cfg["ptq"]["scope"],
                             "mode": cfg["ptq"]["mode"], "backend": cfg["ptq"]["backend"]}
            save_quantized(out_dir, q, conversion, compatibility, calibration["ids"], parent, cfg["ptq"]["scope"],
                           calibration["x"])
            reload = reload_check(q, parent, cfg["ptq"]["scope"], calibration["x"], out_dir / "state.pt", compatibility,
                                  example, backend=cfg["ptq"]["backend"])
            saved = json.loads((out_dir / "conversion.json").read_text(encoding="utf-8"))
            saved["reload_check"] = reload
            durable_write_json(out_dir / "conversion.json", saved)
            durable_write_json(out_dir / "train_log.json", log)
            conversion["reload_check"] = reload
        return result
    except (TrainingInterrupted, CheckpointError, FileExistsError):
        raise
    except Exception as error:
        module = tracker.name if tracker is not None else None
        if module is None and "was not converted:" in str(error):
            module = str(error).rsplit(":", 1)[1].strip()
        raise QATStageError(stage, module, error) from error
    finally:
        if tracker is not None:
            tracker.close()