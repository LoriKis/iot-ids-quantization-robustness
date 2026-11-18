"""VM-only integration checks on a tiny real model (1 layer, d_model 16). They train, quantize and attack, so they are
marked vm_only and must never run on the development machine (the local guard blocks them anyway):
VM stage 4 runs them as part of the regression tests. Status on this machine: written, NOT executed.
"""
import copy

import numpy as np
import pytest
import torch
from m2_helpers import TINY_ARCH, TINY_ISLANDS, split_arrays, tiny_model

from research import m2_attacks as ma
from research import m2_quant as mq
from research import m2_training as mt
from research.common import make_model
from research.m2_training import TrainingInterrupted
from research.paper_described_adam import to_sequence

pytestmark = pytest.mark.vm_only


@pytest.fixture(scope="module")
def train_cfg():
    cfg = mt.load_config()
    cfg = copy.deepcopy(cfg)
    cfg["architecture"] = dict(TINY_ARCH)
    cfg["training"]["batch_size"] = 64
    return cfg


@pytest.fixture(scope="module")
def arrays():
    return split_arrays(per_class=4)


def tiny_factory():
    return make_model(TINY_ARCH)


def test_resume_equals_an_uninterrupted_run(train_cfg, arrays, tmp_path):
    x, y = arrays["train"]["x"], arrays["train"]["y"]
    straight = mt.train_unit(train_cfg, x, y, 3, tmp_path / "a", epochs=3, threads=1, checkpoint_dir=tmp_path / "a" / "ckpt")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] == 1                       # stop after the first epoch checkpoint
    with pytest.raises(TrainingInterrupted):
        mt.train_unit(train_cfg, x, y, 3, tmp_path / "b", epochs=3, threads=1, checkpoint_dir=tmp_path / "b" / "ckpt",
                      stop_flag=stop)
    resumed = mt.train_unit(train_cfg, x, y, 3, tmp_path / "b", epochs=3, threads=1, checkpoint_dir=tmp_path / "b" / "ckpt")
    assert resumed["log"]["resumed_from_epoch"] == 1
    assert resumed["state_hash"] == straight["state_hash"]           # equal at equal thread count


def calibration(arrays):
    return mq.calibration_ids(arrays, 4, 20260)


def test_ptq_of_the_tiny_model(arrays):
    cfg = mq.load_config()
    out = mq.ptq_unit(tiny_model(), calibration(arrays), cfg, expected_islands=TINY_ISLANDS)
    assert out["conversion"]["int8_label_verified"]


def test_one_fp32_ft_and_one_qat_epoch_with_a_resumable_checkpoint(train_cfg, arrays, tmp_path):
    cfg = mq.load_config()
    parent, cal = tiny_model(), calibration(arrays)
    x, y = arrays["train"]["x"], arrays["train"]["y"]
    ft = mq.fp32_finetune_unit(parent, x, y, train_cfg, cfg, 3, epochs=1, out_dir=tmp_path / "ft",
                               checkpoint_dir=tmp_path / "ft" / "ckpt", threads=1)
    assert (tmp_path / "ft" / "checkpoint.pt").is_file() and len(ft["log"]["epoch_losses"]) == 1
    eval_x = torch.from_numpy(to_sequence(arrays["validation"]["x"][:16]))
    qat = mq.qat_unit(parent, cal, x, y, train_cfg, cfg, 3, epochs=1, out_dir=tmp_path / "qat",
                      checkpoint_dir=tmp_path / "qat" / "ckpt", eval_x=eval_x, expected_islands=TINY_ISLANDS, threads=1)
    assert qat["conversion"]["reload_check"]["logits_bitwise_equal"]
    assert (tmp_path / "qat" / "state.pt").is_file()


def test_qat_failure_names_the_stage(train_cfg, arrays):
    cfg = mq.load_config()
    with pytest.raises(mq.QATStageError) as error:
        mq.qat_unit(tiny_model(), calibration(arrays), arrays["train"]["x"], arrays["train"]["y"], train_cfg, cfg, 3, epochs=1,
                    expected_islands=TINY_ISLANDS + 1, threads=1)
    assert error.value.stage == "verify_converted" and "islands" in error.value.record()["exception"]


def test_one_pgd_unit_on_eight_rows(arrays, tmp_path):
    acfg = ma.load_config()
    x = torch.from_numpy(arrays["validation"]["x"][:8])
    y, ids = arrays["validation"]["y"][:8], arrays["validation"]["ids"][:8]
    result = ma.evaluate_unit(source="fp32_direct", attack_name="pgd20", epsilon=0.05, x=x, y=y, ids=ids,
                              models={"fp32": tiny_model()}, cfg=acfg, train_seed=1, model_id="V1", out_dir=tmp_path,
                              raise_on_invalid=False)
    assert result["checks"]["valid"] and len(result["arrays"]["pred_adv"]) == 8
    assert np.isfinite(result["row"]["robust_accuracy"])
