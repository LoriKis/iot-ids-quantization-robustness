"""Milestone 2A training loop: V1 ordinary FP32 and V3 PGD adversarial training through ONE loop.

Reproduces paper_described_adam.run_epochs (batching, shuffle, noise, Adam step) so that
train_unit(adversarial=False) is bit-identical to it; adversarial=True adds the protocol-D step.
Pairing: same seed initialisation, shuffle order, noise tensors, optimizer and update count for V1 and V3.
Dropout masks cannot be paired (V3 does two train-mode forwards per update). Attack seeds come from
derived_seed("atk-train", seed, epoch, batch) and a local generator inside attacks.attack, never from the
global, shuffle or noise streams.
Design: docs/MILESTONE2_PROTOCOL.md sections B and D.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import provenance as pv
from .common import ROOT, atomic_json, atomic_torch, make_model, sha256
from .m2_attacks import RowTiedModel, checked_attack
from .m2b_io import archive_move, durable_write_json, durable_write_torch, sha256_file
from .m2_split import ConfigError
from .paper_described_adam import build_optimizer, derived_seed, epoch_batches, optimizer_steps, seed_everything

DEFAULT_CONFIG = ROOT / "configs/m2/training.json"
TOP_KEYS = {"schema_version", "frozen_date_local", "claim", "source_experiment", "upstream_revision", "seeds",
            "architecture", "sequence", "training", "resources", "derived_seed_names", "adversarial"}
CLAIM = "M2 training unit (Experiment C recipe on the M2 split); not an exact reproduction of Table 4."


class TrainingDivergedError(RuntimeError):
    """The training loss became non-finite."""


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"training configuration not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg) != TOP_KEYS:
        raise ConfigError(f"unknown or missing training configuration keys: {sorted(set(cfg) ^ TOP_KEYS)}")
    t = cfg["training"]
    if t["optimizer"] != "torch.optim.Adam" or t.get("decoupled_weight_decay") is not False:
        raise ConfigError("optimizer must be torch.optim.Adam (not AdamW)")
    adv = cfg["adversarial"]
    if adv["kind"] != "pgd" or adv["attack_model_mode"] != "eval" or adv["perturbation_structure"] != "row_tied":
        raise ConfigError("adversarial block must be eval-mode row-tied PGD")
    if not math.isclose(adv["clean_weight"] + adv["adversarial_weight"], 1.0):
        raise ConfigError("clean_weight + adversarial_weight must equal 1")
    return cfg


def attack_pass_counts(cfg: dict, adversarial: bool) -> dict:
    """Per-step pass counters derived from the attacks.attack control flow (protocol D, rev 3)."""
    if not adversarial:
        return {"attack_grad_passes_per_step": 0, "attack_scoring_forwards_per_step": 0,
                "train_forward_backward_per_step": 1}
    a = cfg["adversarial"]
    # gradients: one at the clean point (FGSM candidate) + `steps` per restart
    # scoring: clean baseline + FGSM point + per restart (start point + `steps` iterates)
    return {"attack_grad_passes_per_step": 1 + a["restarts"] * a["steps"],
            "attack_scoring_forwards_per_step": 2 + a["restarts"] * (1 + a["steps"]),
            "train_forward_backward_per_step": 2}


def weighted_loss(logits_clean, logits_adv, y, clean_weight: float, adversarial_weight: float):
    return clean_weight * F.cross_entropy(logits_clean, y) + adversarial_weight * F.cross_entropy(logits_adv, y)


class _ForwardCounter:
    """Counts forward calls on `model` split by the module's training flag at call time."""

    def __init__(self, model):
        self.attack_side = 0   # eval-mode calls: attack gradient and scoring forwards
        self.train_side = 0    # train-mode calls: training forwards
        self._handle = model.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if module.training:
            self.train_side += 1
        else:
            self.attack_side += 1

    def close(self):
        self._handle.remove()


class TrainingInterrupted(RuntimeError):
    """stop_flag was set; the last completed epoch is checkpointed and the unit can be resumed."""


class CheckpointError(RuntimeError):
    """A checkpoint is missing, torn, corrupt (SHA mismatch) or belongs to a different run (fingerprints, threads)."""


CHECKPOINT_SCHEMA = 1


def _checkpoint_file(ckpt_dir: Path, epoch: int) -> Path:
    return Path(ckpt_dir) / f"epoch_{epoch:03d}.pt"


def save_epoch_checkpoint(ckpt_dir, *, epoch: int, model, opt, updates: int, epoch_losses, step_seconds, attack_seconds,
                          init_state_hash: str, seed: int, adversarial: bool, threads: int, fingerprints=None,
                          extras=None, keep_last: int = 2) -> Path:
    """Durably write ckpt/epoch_NNN.pt, then ckpt/latest.json LAST, then move older epoch files to ckpt/old/ (rename)."""
    ckpt_dir = Path(ckpt_dir)
    path = _checkpoint_file(ckpt_dir, epoch)
    payload = {
        "schema": CHECKPOINT_SCHEMA, "epoch_done": epoch, "updates": updates, "epoch_losses": list(epoch_losses),
        "step_seconds": list(step_seconds), "attack_seconds": list(attack_seconds),
        "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
        "init_state_hash": init_state_hash, "seed": seed, "adversarial": adversarial, "threads": threads,
        "fingerprints": dict(fingerprints or {}), "extras": dict(extras or {}),
    }
    durable_write_torch(path, payload)
    durable_write_json(ckpt_dir / "latest.json", {"epoch": epoch, "file": path.name, "sha256": sha256_file(path)})
    for old in sorted(ckpt_dir.glob("epoch_*.pt")):
        if int(old.stem.split("_")[1]) <= epoch - keep_last:
            archive_move(old, ckpt_dir, "old")
    return path


def load_epoch_checkpoint(ckpt_dir, *, threads: int, fingerprints=None):
    """Return the payload named by ckpt/latest.json, or None if there is no checkpoint.

    Verifies the file SHA-256, schema, thread count and (when both sides are given) the fingerprints. A torn *.tmp file
    is ignored and moved aside. Does not touch any RNG or model.
    """
    ckpt_dir = Path(ckpt_dir)
    for tmp in sorted(ckpt_dir.glob("*.tmp")) if ckpt_dir.is_dir() else []:
        archive_move(tmp, ckpt_dir, "old")
    latest = ckpt_dir / "latest.json"
    if not latest.is_file():
        return None
    record = json.loads(latest.read_text(encoding="utf-8"))
    path = ckpt_dir / record["file"]
    if not path.is_file():
        raise CheckpointError(f"latest.json names a missing checkpoint file: {path}")
    if sha256_file(path) != record["sha256"]:
        raise CheckpointError(f"checkpoint SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)  # own file, SHA-256 verified above
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"unsupported checkpoint schema {payload.get('schema')}")
    if payload["epoch_done"] != record["epoch"]:
        raise CheckpointError("checkpoint epoch differs from latest.json")
    if payload["threads"] != threads:
        raise CheckpointError(f"checkpoint was written with {payload['threads']} torch threads, this run uses {threads}; "
                              "training is thread-count dependent")
    if fingerprints is not None and payload["fingerprints"]:
        differ = sorted(k for k in set(fingerprints) | set(payload["fingerprints"])
                        if fingerprints.get(k) != payload["fingerprints"].get(k))
        if differ:
            raise CheckpointError(f"checkpoint fingerprints differ from the current run: {differ}")
    return payload


def restore_rng(rng: dict) -> None:
    torch.set_rng_state(rng["torch"].to("cpu", torch.uint8))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def _run_epoch(ep: int, env: dict) -> dict:
    """One epoch. Statement order is the original train_unit loop body (bit-identity with run_epochs)."""
    model, opt, Xt, yt = env["model"], env["opt"], env["Xt"], env["yt"]
    seed, adversarial, attack_model, counters = env["seed"], env["adversarial"], env["attack_model"], env["counters"]
    batch_size, seq_len, noise_std, adv = env["batch_size"], env["seq_len"], env["noise_std"], env["adv"]
    n_rows = len(Xt)
    model.train()
    g = torch.Generator(device="cpu")
    g.manual_seed(derived_seed("shuffle", seed, ep))
    perm = torch.randperm(n_rows, generator=g)
    batches = epoch_batches(n_rows, batch_size, perm)
    batch_sizes = [len(i) for i in batches]
    total_loss, updates, hook_counts = 0.0, 0, None
    step_seconds, attack_seconds = [], []
    for b, idx in enumerate(batches):
        tick = time.perf_counter()
        counter = _ForwardCounter(model) if (ep == 0 and b == 0) else None
        ng = torch.Generator(device="cpu")
        ng.manual_seed(derived_seed("noise", seed, ep, b))
        rows, yb = Xt[idx], yt[idx]
        shape = (len(idx), seq_len, rows.shape[1])
        noise = torch.randn(shape, generator=ng, dtype=torch.float32) * noise_std
        seq_clean = rows[:, None, :].expand(-1, seq_len, -1) + noise
        if adversarial:
            t0 = time.perf_counter()
            x_adv = checked_attack(attack_model, rows, yb, kind=adv["kind"], epsilon=adv["epsilon"],
                                   steps=adv["steps"], alpha=adv["alpha"], restarts=adv["restarts"],
                                   random_start=adv["random_start"],
                                   seed=derived_seed("atk-train", seed, ep, b))
            attack_seconds.append(time.perf_counter() - t0)
            seq_adv = x_adv[:, None, :].expand(-1, seq_len, -1) + noise
            opt.zero_grad()
            loss = weighted_loss(model(seq_clean)["logits"], model(seq_adv)["logits"], yb,
                                 adv["clean_weight"], adv["adversarial_weight"])
        else:
            opt.zero_grad()
            loss = F.cross_entropy(model(seq_clean)["logits"], yb)
        if not torch.isfinite(loss):
            raise TrainingDivergedError(f"non-finite loss at epoch {ep + 1}, batch {b}, seed {seed}")
        loss.backward()
        opt.step()
        updates += 1
        total_loss += float(loss.item()) * len(idx)
        step_seconds.append(time.perf_counter() - tick)
        if counter is not None:
            counter.close()
            hook_counts = {"attack_side_forwards": counter.attack_side, "train_side_forwards": counter.train_side,
                           "expected_attack_side": counters["attack_grad_passes_per_step"] + counters["attack_scoring_forwards_per_step"],
                           "expected_train_side": counters["train_forward_backward_per_step"]}
            hook_counts["matches_logged_counters"] = (
                hook_counts["attack_side_forwards"] == hook_counts["expected_attack_side"]
                and hook_counts["train_side_forwards"] == hook_counts["expected_train_side"])
            if not hook_counts["matches_logged_counters"]:
                raise RuntimeError(f"forward-hook counts {hook_counts} differ from the logged pass counters")
    return {"total_loss": total_loss, "updates": updates, "step_seconds": step_seconds,
            "attack_seconds": attack_seconds, "hook_counts": hook_counts, "batch_sizes": batch_sizes}


def train_unit(cfg: dict, X: np.ndarray, y, seed: int, out_dir: Path | str | None = None, *,
               adversarial: bool = False, epochs: int | None = None, model_factory=None, threads: int | None = None,
               checkpoint_dir: Path | str | None = None, keep_last: int = 2, stop_flag=None, epoch_fn=None,
               fingerprints: dict | None = None) -> dict:
    """Train one unit and return {"model", "log", "init_state_hash", "state_hash"}.

    X is the (N, 39) MinMax-scaled training matrix; rows are repeated to the sequence length inside the loop.
    Writes checkpoint.pt (final epoch only) and train_log.json to out_dir if given (never overwrites).

    Optional restart support (defaults leave behaviour bit-identical): with checkpoint_dir a durable epoch-boundary
    checkpoint (model, optimizer, torch/numpy/python RNG, counters) is written after every epoch and an existing one
    is resumed; stop_flag() is polled after each checkpoint and raises TrainingInterrupted; epoch_fn(ep, env) replaces
    the epoch body (test seam).
    """
    t, seq = cfg["training"], cfg["sequence"]
    epochs = t["epochs"] if epochs is None else epochs
    batch_size, seq_len, noise_std = t["batch_size"], seq["length"], seq["train_noise_std"]
    adv = cfg["adversarial"]
    if out_dir is not None:
        out_dir = Path(out_dir)
        if (out_dir / "checkpoint.pt").exists() or (out_dir / "train_log.json").exists():
            raise FileExistsError(f"refusing to overwrite training output in {out_dir}")
    previous_threads = torch.get_num_threads()
    run_threads = cfg["resources"]["torch_threads"] if threads is None else threads
    torch.set_num_threads(run_threads)
    try:
        seed_everything(seed)
        model = model_factory() if model_factory else make_model(cfg["architecture"])
        model = model.to("cpu")
        init_hash = pv.state_hash(model)
        opt = build_optimizer(cfg, model)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        n_rows = len(Xt)
        attack_model = RowTiedModel(model, seq_len) if adversarial else None
        counters = attack_pass_counts(cfg, adversarial)
        step_seconds, attack_seconds, epoch_losses, updates = [], [], [], 0
        hook_counts = None
        batch_sizes_epoch_1 = None
        start_epoch, prior_wall, resumed_from = 0, 0.0, None
        if checkpoint_dir is not None:
            payload = load_epoch_checkpoint(checkpoint_dir, threads=run_threads, fingerprints=fingerprints)
            if payload is not None:
                # Resume order: threads, seed, build (above), verify init hash, load states, restore RNG LAST.
                if payload["init_state_hash"] != init_hash:
                    raise CheckpointError("fresh model init_state_hash differs from the checkpoint (different seed or architecture)")
                if payload["seed"] != seed or payload["adversarial"] != adversarial:
                    raise CheckpointError("checkpoint belongs to a different seed or training variant")
                model.load_state_dict(payload["model_state"])
                opt.load_state_dict(payload["optimizer_state"])
                start_epoch, updates = payload["epoch_done"], payload["updates"]
                epoch_losses, step_seconds = list(payload["epoch_losses"]), list(payload["step_seconds"])
                attack_seconds = list(payload["attack_seconds"])
                hook_counts = payload["extras"].get("hook_counts")
                batch_sizes_epoch_1 = payload["extras"].get("batch_sizes_epoch_1")
                prior_wall = payload["extras"].get("wall_seconds", 0.0)
                resumed_from = start_epoch
                restore_rng(payload["rng"])
        env = {"model": model, "opt": opt, "Xt": Xt, "yt": yt, "seed": seed, "adversarial": adversarial,
               "attack_model": attack_model, "counters": counters, "batch_size": batch_size, "seq_len": seq_len,
               "noise_std": noise_std, "adv": adv}
        run_epoch = epoch_fn or _run_epoch
        started = time.perf_counter()
        for ep in range(start_epoch, epochs):
            result = run_epoch(ep, env)
            batch_sizes_epoch_1 = batch_sizes_epoch_1 or result["batch_sizes"]
            hook_counts = result["hook_counts"] or hook_counts
            updates += result["updates"]
            step_seconds.extend(result["step_seconds"])
            attack_seconds.extend(result["attack_seconds"])
            epoch_losses.append(result["total_loss"] / n_rows)
            if checkpoint_dir is not None:
                save_epoch_checkpoint(
                    checkpoint_dir, epoch=ep + 1, model=model, opt=opt, updates=updates, epoch_losses=epoch_losses,
                    step_seconds=step_seconds, attack_seconds=attack_seconds, init_state_hash=init_hash, seed=seed,
                    adversarial=adversarial, threads=run_threads, fingerprints=fingerprints, keep_last=keep_last,
                    extras={"hook_counts": hook_counts, "batch_sizes_epoch_1": batch_sizes_epoch_1,
                            "wall_seconds": prior_wall + time.perf_counter() - started})
                if ep + 1 < epochs and stop_flag is not None and stop_flag():
                    raise TrainingInterrupted(f"stopped after epoch {ep + 1} of {epochs}; resume from {checkpoint_dir}")
        wall_clock = prior_wall + time.perf_counter() - started
        steps = sorted(optimizer_steps(opt))
        if epoch_fn is None and steps != [updates]:
            raise RuntimeError(f"optimizer step counts {steps} != update count {updates}")
        model.eval()
        final_hash = pv.state_hash(model)
        log = {
            "claim": CLAIM, "variant": "V3_adversarial_fp32" if adversarial else "V1_ordinary_fp32",
            "adversarial": adversarial, "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "train_rows": n_rows, "steps_per_epoch": len(batch_sizes_epoch_1 or []), "batch_sizes_epoch_1": batch_sizes_epoch_1,
            "optimizer_updates": updates, "optimizer": "torch.optim.Adam", "torch_threads": run_threads,
            "init_state_hash": init_hash, "final_state_hash": final_hash, "epoch_losses": epoch_losses,
            "wall_clock_seconds": wall_clock, "step_seconds": step_seconds, "attack_seconds": attack_seconds,
            "pass_counters_per_step": counters, "first_step_hook_counts": hook_counts,
            "adversarial_config": adv if adversarial else None,
            "dropout_pairing_note": "dropout masks are not paired between V1 and V3 (V3 performs two train-mode forwards per update)",
        }
        if resumed_from is not None:
            log["resumed_from_epoch"] = resumed_from
        if out_dir is not None:
            atomic_torch(out_dir / "checkpoint.pt", {"state_dict": model.state_dict(), "epoch": epochs,
                                                     "architecture": cfg["architecture"], "seed": seed,
                                                     "variant": log["variant"], "claim": CLAIM})
            log["checkpoint_sha256"] = sha256(out_dir / "checkpoint.pt")
            atomic_json(out_dir / "train_log.json", log)
        return {"model": model, "log": log, "init_state_hash": init_hash, "state_hash": final_hash}
    finally:
        torch.set_num_threads(previous_threads)
