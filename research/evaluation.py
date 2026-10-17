from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch

from .attacks import attack, logits
from .common import atomic_json, state_bytes
from .training import predictions


def metrics(y, pred, clean, labels):
    y, pred, clean = np.asarray(y), np.asarray(pred), np.asarray(clean)
    eligible = clean == y
    denominator = int(eligible.sum())
    successes = int((eligible & (pred != y)).sum())
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "robust_accuracy": float(np.mean(eligible & (pred == y))),
            "macro_f1": float(f1_score(y, pred, labels=np.arange(len(labels)), average="macro", zero_division=0)),
            "attack_success_rate": successes / denominator if denominator else None,
            "asr_numerator": successes, "asr_denominator": denominator,
            "asr_definition": "target clean-correct examples misclassified by this attack / target clean-correct examples",
            "classwise": classification_report(y, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(labels))).tolist()}


def benchmark(model, example, config):
    model.eval()
    rows = []
    for size in config["batch_sizes"]:
        x = example[:1].repeat(size, 1, 1).contiguous()
        # Full upstream forward, including anomaly head and attention outputs;
        # input creation, data loading and attack generation are excluded.
        with torch.inference_mode():
            for _ in range(config["warmup"]):
                logits(model, x)
            timings = []
            for _ in range(config["repetitions"]):
                start = time.perf_counter_ns()
                logits(model, x)
                timings.append((time.perf_counter_ns() - start) / 1e6)
        rows.append({"batch_size": size, "median_ms": float(np.median(timings)),
                     "p95_ms": float(np.percentile(timings, 95)),
                     "throughput_examples_per_second": size * len(timings) * 1000 / sum(timings),
                     "warmup": config["warmup"], "repetitions": config["repetitions"],
                     "threads": torch.get_num_threads(), "device": "cpu",
                     "scope": "complete upstream model forward; preallocated repeated input; includes output finiteness check",
                     "raw_latency_ms": timings})
    return {"serialized_state_bytes": state_bytes(model), "measurements": rows,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "memory_note": "Process RSS includes other loaded models and allocator caches; no model memory speedup inferred. Isolated per-model peak memory not measured."}


def evaluate(model, surrogate, x, y, ids, data, attack_config, epsilons, config,
             folder, name, seed, mode, focused=False, kinds=("fgsm", "pgd")):
    folder = Path(folder)
    rows = []
    started = time.perf_counter()
    clean = predictions(model, x, config["batch_size"]).argmax(1).numpy()
    raw = {"ids": np.asarray(ids), "y": y.numpy(), "clean_prediction": clean}
    common = {"variant": name, "seed": seed, "partition": config["evaluation_partition"],
              "dataset": "CICIoT", "attack_gradient": mode}
    rows.append({**common, "attack": "clean", "epsilon": 0.0,
                 **metrics(y, clean, clean, data["labels"]), "attack_seconds": 0.0,
                 "max_linf": 0.0})
    # Check agreement on the entire evaluation set, not only a calibration probe.
    a = predictions(model, x, config["batch_size"])
    b = predictions(surrogate, x, config["batch_size"])
    max_error = float((a - b).abs().max())
    if mode.startswith("bpda") and max_error != 0:
        raise AssertionError("BPDA forward does not match actual INT8 target")
    gradient_probe = x[:2].clone().requires_grad_(True)
    grad = torch.autograd.grad(logits(surrogate, gradient_probe).square().sum(), gradient_probe)[0]
    if not torch.isfinite(grad).all():
        raise FloatingPointError("Invalid surrogate input gradients")
    atomic_json(folder / "gradient_diagnostics.json", {
        "forward_max_abs_error": max_error, "probe_gradient_abs_max": float(grad.abs().max()),
        "probe_gradient_nonzero_fraction": float(grad.ne(0).float().mean()),
        "claim": "exact target forward, dequantized Linear derivative approximation" if mode.startswith("bpda") else "FP32 autograd",
        "limitations": "Nonzero gradients and PGD/FGSM comparisons do not prove absence of gradient masking; BPDA is approximate."})
    for kind in kinds:
        for epsilon in epsilons:
            spec = {**attack_config, "kind": kind, "epsilon": epsilon}
            if kind == "fgsm":
                spec.update(steps=1, restarts=1, random_start=False, alpha=epsilon)
            else:
                spec["alpha"] = epsilon * attack_config["alpha_fraction"]
            spec.pop("alpha_fraction", None)
            tick = time.perf_counter()
            adv_batches = []
            for j, (xb, yb) in enumerate(zip(x.split(config["batch_size"]), y.split(config["batch_size"]))):
                adv_batches.append(attack(surrogate, xb, yb, **spec,
                                          lower=data["lower"], upper=data["upper"],
                                          seed=seed + 500000 + j))
            adv = torch.cat(adv_batches)
            attack_seconds = time.perf_counter() - tick
            if epsilon == 0:
                torch.testing.assert_close(adv, x, atol=0, rtol=0)
            norms = (adv - x).flatten(1).abs().amax(1).numpy()
            if max(norms) > epsilon + 2e-6:
                raise AssertionError("Epsilon budget exceeded")
            pred = predictions(model, adv, config["batch_size"]).argmax(1).numpy()
            if epsilon == 0 and not np.array_equal(pred, clean):
                raise AssertionError("Epsilon-zero predictions differ")
            raw[f"{kind}_{epsilon}_prediction"] = pred
            raw[f"{kind}_{epsilon}_linf"] = norms
            row = {**common, "attack": kind, "epsilon": epsilon, "settings": spec,
                   **metrics(y, pred, clean, data["labels"]),
                   "attack_seconds": attack_seconds, "max_linf": float(max(norms))}
            rows.append(row)
            print(f"eval {name} s{seed} {kind} eps={epsilon}: acc={row['accuracy']:.4f} ({attack_seconds:.2f}s)", flush=True)
    np.savez_compressed(folder / "predictions.npz", **raw)
    atomic_json(folder / "metrics.json", rows)
    atomic_json(folder / "benchmark.json", benchmark(model, x, config["benchmark"]))
    atomic_json(folder / "timing.json", {"evaluation_seconds": time.perf_counter() - started})
    return rows


def transfer(source, target, x, y, ids, data, spec, config, folder, name, seed, source_kind):
    folder = Path(folder)
    clean_target = predictions(target, x, config["batch_size"]).argmax(1).numpy()
    clean_source = predictions(source, x, config["batch_size"]).argmax(1).numpy()
    tick = time.perf_counter()
    adv = torch.cat([attack(source, xb, yb, **spec, lower=data["lower"], upper=data["upper"],
                            seed=seed + 700000 + j)
                     for j, (xb, yb) in enumerate(zip(x.split(config["batch_size"]), y.split(config["batch_size"])))] )
    pred = predictions(target, adv, config["batch_size"]).argmax(1).numpy()
    source_pred = predictions(source, adv, config["batch_size"]).argmax(1).numpy()
    both_clean = (clean_target == y.numpy()) & (clean_source == y.numpy())
    denom = int(both_clean.sum())
    norms = (adv - x).flatten(1).abs().amax(1).numpy()
    row = {"direction": name, "seed": seed, "partition": config["evaluation_partition"],
           "source_attack": source_kind, "settings": spec,
           **metrics(y, pred, clean_target, data["labels"]),
           "source_attack_successes": int(((source_pred != y.numpy()) & (clean_source == y.numpy())).sum()),
           "both_clean_correct_denominator": denom,
           "both_clean_correct_transfer_asr": float(((pred != y.numpy()) & both_clean).sum() / denom) if denom else None,
           "attack_seconds": time.perf_counter() - tick, "max_linf": float(norms.max())}
    atomic_json(folder / "metrics.json", [row])
    np.savez_compressed(folder / "predictions.npz", ids=ids, y=y.numpy(), prediction=pred,
                        source_prediction=source_pred, target_clean=clean_target, source_clean=clean_source, linf=norms)
    return row
