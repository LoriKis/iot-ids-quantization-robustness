import numpy as np
import pytest
import torch
from research.paper_described_baseline import (
    DEFAULT_CONFIG,
    fgsm_attack,
    import_reference,
    pgd_attack,
    prepare_data,
    read_config,
    state_hash,
)


def test_paper_described_configuration():
    cfg = read_config(DEFAULT_CONFIG)
    assert cfg["seeds"] == [1, 7, 13, 29]
    assert cfg["split_seed"] == 2025
    assert cfg["training"]["epochs"] == 25
    assert cfg["training"]["batch_size"] == 512
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["training"]["weight_decay"] == 0.0001
    assert cfg["evaluation"]["test_rows"] == 4500
    assert cfg["evaluation"]["fgsm"]["epsilon"] == 0.10
    assert cfg["evaluation"]["pgd"]["epsilon"] == 0.10
    assert cfg["evaluation"]["pgd"]["alpha"] == 0.025
    assert cfg["evaluation"]["pgd"]["iterations"] == 7


def test_paper_described_data_and_training_only_scaling(tmp_path):
    cfg = read_config(DEFAULT_CONFIG)
    infra = import_reference(tmp_path, cfg)
    bundle = prepare_data(infra, cfg, tmp_path)

    # Split row counts
    assert len(bundle["train_idx"]) == 20997
    assert len(bundle["val_idx"]) == 4500
    assert len(bundle["test_idx"]) == 4500
    assert not (set(bundle["train_idx"]) & set(bundle["val_idx"]))
    assert not (set(bundle["train_idx"]) & set(bundle["test_idx"]))
    assert not (set(bundle["val_idx"]) & set(bundle["test_idx"]))

    # MinMaxScaler fitted only on train
    assert bundle["scaler"].n_samples_seen_ == 20997
    assert bundle["X_tr_norm"].min() >= -1e-6 and bundle["X_tr_norm"].max() <= 1.0 + 1e-6
    # Test values in [0, 1]
    assert bundle["X_te_norm"].min() >= -1e-6 and bundle["X_te_norm"].max() <= 1.0 + 1e-6


def test_paper_attacks_bounds_and_invariance(tmp_path):
    cfg = read_config(DEFAULT_CONFIG)
    infra = import_reference(tmp_path, cfg)
    model = infra.StandardTransformerIDS(
        input_dim=3, d_model=8, n_heads=2, n_layers=1, d_ff=16, n_classes=2, max_seq_len=4, dropout=0.1
    ).eval()
    before = state_hash(model)

    x = np.random.default_rng(42).uniform(0.1, 0.9, size=(4, 4, 3)).astype(np.float32)
    y = np.array([0, 1, 0, 1])

    # Epsilon zero identity
    np.testing.assert_array_equal(fgsm_attack(model, x, y, eps=0.0, batch_size=2), x)
    np.testing.assert_array_equal(pgd_attack(model, x, y, seed=1, eps=0.0, alpha=0.025, iters=3, batch_size=2), x)

    # Attacks at eps=0.10
    fg = fgsm_attack(model, x, y, eps=0.10, batch_size=2)
    pg = pgd_attack(model, x, y, seed=1, eps=0.10, alpha=0.025, iters=7, batch_size=2)

    # Linf bound checks
    assert np.max(np.abs(fg - x)) <= 0.10001
    assert np.max(np.abs(pg - x)) <= 0.10001

    # Domain clipping checks
    assert fg.min() >= -1e-6 and fg.max() <= 1.0 + 1e-6
    assert pg.min() >= -1e-6 and pg.max() <= 1.0 + 1e-6

    # Model mode and parameter invariance
    assert not model.training
    assert state_hash(model) == before
