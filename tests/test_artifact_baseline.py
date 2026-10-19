import copy
import numpy as np
import pytest
import torch
from research.artifact_baseline import DEFAULT_CONFIG, import_reference, instrument_training, read_config, state_hash


def test_upstream_training_adapter_and_interrupted_resume_are_identical(tmp_path):
    torch.set_num_threads(2)
    infra = import_reference(tmp_path, read_config(DEFAULT_CONFIG))
    cfg = dict(input_dim=3, d_model=8, n_heads=2, n_layers=1, d_ff=16, n_classes=2, max_seq_len=3, dropout=.1)
    x = np.random.default_rng(12).normal(size=(8, 3, 3)).astype('float32')
    y = np.arange(8) % 2
    torch.manual_seed(81)
    original = infra.StandardTransformerIDS(**cfg)
    initial = copy.deepcopy(original.state_dict())
    torch.manual_seed(99)
    infra.train_seq(original, x, y, epochs=2, bs=4)
    expected = state_hash(original)
    adapted = infra.StandardTransformerIDS(**cfg)
    adapted.load_state_dict(initial)
    torch.manual_seed(99)
    epochs = []
    trainer, source = instrument_training(infra, None, lambda epoch, model, opt: epochs.append(epoch))
    trainer(adapted, x, y, epochs=2, bs=4)
    assert epochs == [1, 2] and state_hash(adapted) == expected
    interrupted = infra.StandardTransformerIDS(**cfg)
    interrupted.load_state_dict(initial)
    snapshot = {}
    def stop(epoch, model, opt):
        snapshot.update(epoch=epoch, state_dict=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), torch_rng=torch.get_rng_state().clone())
        raise InterruptedError('test interruption after completed epoch')
    torch.manual_seed(99)
    trainer, _ = instrument_training(infra, None, stop)
    with pytest.raises(InterruptedError):
        trainer(interrupted, x, y, epochs=2, bs=4)
    reloaded = infra.StandardTransformerIDS(**cfg)
    reloaded.load_state_dict(snapshot['state_dict'])
    trainer, _ = instrument_training(infra, snapshot, lambda *args: None)
    trainer(reloaded, x, y, epochs=2, bs=4)
    assert state_hash(reloaded) == expected


def test_real_upstream_attacks_zero_epsilon_and_parameter_invariance(tmp_path):
    torch.set_num_threads(2)
    infra = import_reference(tmp_path, read_config(DEFAULT_CONFIG))
    torch.manual_seed(2)
    model = infra.StandardTransformerIDS(input_dim=3, d_model=8, n_heads=2, n_layers=1, d_ff=16, n_classes=2, max_seq_len=3).eval()
    x = np.random.default_rng(3).normal(size=(4, 3, 3)).astype('float32')
    y = np.array([0, 1, 0, 1])
    before = state_hash(model)
    np.testing.assert_array_equal(infra.seq_fgsm(model, x, y, 0, bs=2), x)
    np.testing.assert_array_equal(infra.seq_pgd(model, x, y, 0, .01, 10, bs=2), x)
    for attacked in [infra.seq_fgsm(model, x, y, .1, bs=2), infra.seq_pgd(model, x, y, .1, .01, 10, bs=2)]:
        assert np.isfinite(attacked).all()
        assert np.max(np.abs(attacked-x)) <= .10001
        assert np.any(attacked != x)
    assert state_hash(model) == before and not model.training
