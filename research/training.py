from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .attacks import attack, logits
from .common import atomic_json, atomic_torch, digest, make_model, seed_all, sha256, unit_status
from .data import sequences


def predictions(model, x, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([logits(model, batch).cpu() for batch in x.split(batch_size)])


def load_checkpoint(path, expected=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    if expected is not None and record["compatibility"] != expected:
        raise ValueError("Incompatible checkpoint metadata")
    model = make_model(record["model_config"])
    model.load_state_dict(record["state_dict"], strict=True)
    model.eval()
    return model, record


def train(config, data, model_config, seed, variant, folder, compatibility, resume=False):
    folder = Path(folder)
    identity = digest({"compatibility": compatibility, "variant": variant, "seed": seed})
    with unit_status(folder, identity, resume) as execute:
        if execute:
            seed_all(seed)
            model = make_model(model_config).cpu()
            opt = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                   weight_decay=config["training"]["weight_decay"])
            # Constant schedule, explicitly recorded; no implicit scheduler.
            generator = torch.Generator().manual_seed(seed + 1000)
            loader = DataLoader(TensorDataset(data["x"]["train"], data["y"]["train"]),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=generator, num_workers=0)
            val_x = sequences(data["x"]["validation"], config["sequence_length"])
            val_y = data["y"]["validation"]
            best_score, best_state, best_epoch = -float("inf"), None, None
            started = time.perf_counter()
            log = []
            for epoch in range(config["training"]["epochs"]):
                epoch_start = time.perf_counter()
                model.train()
                total_loss, updates = 0.0, 0
                for batch_id, (flat, y) in enumerate(loader):
                    x = sequences(flat, config["sequence_length"], config["train_noise_std"], generator)
                    opt.zero_grad(set_to_none=True)
                    if variant == "at_fp32":
                        spec = config["adversarial_training"]
                        adv = attack(model, x, y, lower=data["lower"], upper=data["upper"],
                                     seed=seed + epoch * 100000 + batch_id, **spec["attack"])
                        mix = spec["adversarial_fraction"]
                        loss = (1 - mix) * F.cross_entropy(logits(model, x), y) + mix * F.cross_entropy(logits(model, adv), y)
                    else:
                        loss = F.cross_entropy(logits(model, x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"], error_if_nonfinite=True)
                    opt.step()
                    total_loss += float(loss.detach())
                    updates += 1
                clean = predictions(model, val_x, config["batch_size"]).argmax(1).numpy()
                clean_f1 = f1_score(val_y, clean, labels=np.arange(len(data["labels"])), average="macro", zero_division=0)
                score, robust_f1 = clean_f1, None
                if variant == "at_fp32":
                    adv_batches = []
                    for batch_id, (x, y) in enumerate(zip(val_x.split(config["batch_size"]), val_y.split(config["batch_size"]))):
                        adv_batches.append(attack(model, x, y, lower=data["lower"], upper=data["upper"],
                                                  seed=seed + 200000 + batch_id,
                                                  **config["adversarial_training"]["attack"]))
                    pred = predictions(model, torch.cat(adv_batches), config["batch_size"]).argmax(1).numpy()
                    robust_f1 = f1_score(val_y, pred, labels=np.arange(len(data["labels"])), average="macro", zero_division=0)
                    score = (clean_f1 + robust_f1) / 2
                if score > best_score:
                    best_score, best_state, best_epoch = float(score), copy.deepcopy(model.state_dict()), epoch + 1
                row = {"epoch": epoch + 1, "updates": updates, "loss": total_loss / updates,
                       "validation_clean_macro_f1": float(clean_f1), "validation_robust_macro_f1": robust_f1,
                       "selection_score": float(score), "seconds": time.perf_counter() - epoch_start}
                log.append(row)
                print(f"train {variant} seed={seed} {row}", flush=True)
                atomic_json(folder / "history.json", log)
            model.load_state_dict(best_state)
            model.eval()
            metadata = {"format_version": 1, "model_config": model_config,
                        "compatibility": compatibility, "seed": seed, "variant": variant,
                        "selection_epoch": best_epoch, "selection_score": best_score,
                        "selection_criterion": "validation clean macro-F1" if variant == "fp32" else "mean validation clean/PGD macro-F1",
                        "training_seconds": time.perf_counter() - started,
                        "training_rows": len(data["y"]["train"]), "epochs": len(log),
                        "total_updates": sum(r["updates"] for r in log),
                        "optimizer": "Adam", "schedule": "constant", "config": config,
                        "resume_support": "completed units only; no mid-training resume"}
            atomic_torch(folder / "checkpoint.pt", {**metadata, "state_dict": best_state})
            loaded, _ = load_checkpoint(folder / "checkpoint.pt", compatibility)
            probe = val_x[:2]
            torch.testing.assert_close(predictions(model, probe, 2), predictions(loaded, probe, 2), rtol=0, atol=0)
            atomic_json(folder / "metadata.json", {**metadata, "checkpoint_sha256": sha256(folder / "checkpoint.pt"),
                                                  "checkpoint_roundtrip_max_abs_error": 0.0})
    model, record = load_checkpoint(folder / "checkpoint.pt", compatibility)
    metadata = json.loads((folder / "metadata.json").read_text())
    if sha256(folder / "checkpoint.pt") != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return model, metadata
