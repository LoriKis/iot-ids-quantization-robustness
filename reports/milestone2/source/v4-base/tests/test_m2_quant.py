"""Milestone 2A INT8 PTQ and QAT-feasibility tests (tiny models; real-model checks are marked slow)."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.ao import quantization as aq

from m2_helpers import TINY_ISLANDS, split_arrays, tiny_model
from research import m2_quant as mq
from research.common import ROOT, make_model
from research.m2_attacks import model_state_hash
from research.m2_split import ConfigError, id_hash
from research.provenance import state_hash
from research.quantization import LinearIsland, load_int8, prepare_static, ptq

TRAIN_CFG = json.loads((ROOT / "configs/m2/training.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg():
    return mq.load_config()


@pytest.fixture(scope="module")
def arrays():
    return split_arrays()


@pytest.fixture(scope="module")
def cal(arrays, cfg):
    return mq.calibration_ids(arrays, cfg["ptq"]["calibration_per_class"], 20260)


@pytest.fixture
def model():
    return tiny_model()


def test_config_frozen_values(cfg):
    p = cfg["ptq"]
    assert (p["scope"], p["mode"], p["backend"]) == ("all_linear", "static", "fbgemm")
    assert (p["weight_dtype"], p["weight_qscheme"], p["activation_dtype"]) == ("qint8", "per_channel_affine", "quint8")
    assert p["calibration_per_class"] == 20 and p["calibration_partition"] == "train" and p["expected_quantized_linear"] == 23
    assert p["calibration_seed_rule"] == "derived_seed('calibration', split_seed)"
    k = cfg["campaign_qat"]
    assert k["finetune_epochs_k"] == 3 and k["learning_rate"] == 1e-4
    assert {"FP32-FT", "FP32-FT->PTQ"} <= set(k["arms"]) and "FP32-FT->PTQ" in k["attribution_rule"]
    with pytest.raises(ConfigError):
        mq.load_config(ROOT / "does-not-exist.json")
    with pytest.raises(ConfigError):
        mq.load_config(ROOT / "configs/m2/attacks.json")  # wrong keys


def test_calibration_is_train_only_stratified_and_deterministic(arrays, cfg):
    cal = mq.calibration_ids(arrays, 20, 20260)
    assert cal["n"] == 540 and cal["x"].shape == (540, 50, 39) and cal["partition"] == "train"
    train_ids, val_ids = set(arrays["train"]["ids"].tolist()), set(arrays["validation"]["ids"].tolist())
    assert set(cal["ids"]) <= train_ids and not set(cal["ids"]) & val_ids
    assert cal["per_class_counts"] == [20] * 27
    assert cal["id_sha256"] == id_hash(cal["ids"])
    again = mq.calibration_ids(arrays, 20, 20260)
    assert again["ids"] == cal["ids"] and torch.equal(again["x"], cal["x"])
    assert mq.calibration_ids(arrays, 20, 20261)["ids"] != cal["ids"]
    # validation rows are never touched: scrambling them changes nothing
    scrambled = {"train": arrays["train"], "validation": {**arrays["validation"], "x": arrays["validation"]["x"] * 0 + 9}}
    assert mq.calibration_ids(scrambled, 20, 20260)["ids"] == cal["ids"]
    # rows are noise-free repeats of the stored feature rows
    first = np.flatnonzero(arrays["train"]["ids"] == cal["ids"][0])[0]
    assert torch.equal(cal["x"][0, 7], torch.from_numpy(arrays["train"]["x"][first]))
    assert torch.equal(cal["x"][0, 0], cal["x"][0, 49])


def test_save_calibration_once(tmp_path, cal):
    path = tmp_path / "calibration_ids.json"
    mq.save_calibration(cal, path)
    saved = json.loads(path.read_text())
    assert saved["ids"] == cal["ids"] and saved["id_sha256"] == cal["id_sha256"] and saved["partition"] == "train"
    with pytest.raises(FileExistsError):
        mq.save_calibration(cal, path)


def test_identity_wrapper_is_exact(model, cal):
    assert mq.verify_identity_wrapper(model, cal["x"][:4]) == 0.0


def test_ptq_unit_real_converted_modules(model, cal, cfg):
    out = mq.ptq_unit(model, cal, cfg, expected_islands=TINY_ISLANDS)
    q, conv = out["model"], out["conversion"]
    islands = [m for m in q.modules() if isinstance(m, LinearIsland)]
    assert len(islands) == TINY_ISLANDS
    for island in islands:
        assert type(island.linear).__module__ == "torch.ao.nn.quantized.modules.linear"
        w = island.linear.weight()
        assert w.dtype == torch.qint8 and w.qscheme() == torch.per_channel_affine
    assert not [m for m in q.modules() if isinstance(m, (aq.FakeQuantizeBase, aq.ObserverBase, nn.Linear))]
    assert conv["int8_label_verified"] is True and conv["backend"] == "fbgemm" and conv["mode"] == "static"
    assert conv["architecture_wrapper_fp32_max_abs_error"] == 0.0
    assert conv["structure_checks"]["weight_dtype_inventory"] == {"torch.qint8": TINY_ISLANDS}
    assert conv["structure_checks"]["profiler"]["quantized_linear_calls"] >= TINY_ISLANDS
    assert "quantized::linear" in conv["structure_checks"]["profiler"]["operators"]
    assert conv["quant_dequant_boundaries"] == {"quantize": TINY_ISLANDS, "dequantize": TINY_ISLANDS, "total": 2 * TINY_ISLANDS}
    counts = conv["parameter_counts"]
    assert counts["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert counts["int8_weight_elements"] + counts["fp32_parameters_not_quantized"] == counts["total_parameters"]
    assert counts["anomaly_scorer_int8_weight_elements"] > 0
    assert conv["classification_path_islands"] + conv["anomaly_scorer_islands"] == TINY_ISLANDS
    listed = " | ".join(conv["fp32_remaining_operations"]).lower()
    for op in ("softmax", "layernorm", "residual", "relu", "positional", "pooling", "sigmoid", "matmul"):
        assert op in listed
    assert conv["float_state_hash"] == model_state_hash(model)
    with torch.inference_mode():
        assert torch.isfinite(q(cal["x"][:8])["logits"]).all()


def test_unconverted_and_fake_quant_models_are_rejected(model, cal, cfg):
    x = cal["x"]
    from research.quantization import selected_names
    names = selected_names(model, "all_linear")
    q, inv = ptq(model, "all_linear", x)
    mq.verify_converted(q, inv["requested"], float_model=model, example=x[:2], expected_islands=TINY_ISLANDS)

    prepared, names = prepare_static(model, "all_linear", x[:2])  # observers inserted, never converted
    with pytest.raises(RuntimeError, match="not labelled INT8"):
        mq.verify_converted(prepared, names)
    qat, names = prepare_static(model, "all_linear", x[:2], qat=True)  # fake-quantized FP32
    with pytest.raises(RuntimeError, match="not labelled INT8"):
        mq.verify_converted(qat, names)
    with pytest.raises(RuntimeError, match="not labelled INT8"):
        mq.verify_converted(model, names)  # the plain FP32 model

    leaky = copy.deepcopy(q)
    leaky.add_module("stray", nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="float"):
        mq.verify_converted(leaky, inv["requested"])
    kept = copy.deepcopy(q)
    kept.add_module("stray_observer", aq.MinMaxObserver())
    with pytest.raises(RuntimeError, match="Observer"):
        mq.verify_converted(kept, inv["requested"])

    with pytest.raises(RuntimeError, match="islands"):
        mq.verify_converted(q, inv["requested"], expected_islands=23)
    partial, inv_partial = ptq(model, "classifier", x)  # converting only some Linears is not the frozen all_linear scope
    with pytest.raises(RuntimeError, match="differ"):
        mq.verify_converted(partial, inv_partial["requested"], float_model=model)


def test_same_calibration_ids_for_v2_and_v4(arrays, cfg, tmp_path):
    first, second = tiny_model(5), tiny_model(6)  # stand-ins for V1 and V3
    cal = mq.calibration_ids(arrays, 20, 20260)
    v2 = mq.ptq_unit(first, cal, cfg, tmp_path / "v2", variant="V2", expected_islands=TINY_ISLANDS)
    v4 = mq.ptq_unit(second, cal, cfg, tmp_path / "v4", variant="V4", expected_islands=TINY_ISLANDS)
    assert v2["conversion"]["calibration"] == v4["conversion"]["calibration"]
    ids2 = json.loads((tmp_path / "v2/conversion.json").read_text())["calibration_row_ids"]
    ids4 = json.loads((tmp_path / "v4/conversion.json").read_text())["calibration_row_ids"]
    assert ids2 == ids4 == cal["ids"]
    assert v2["conversion"]["float_state_hash"] != v4["conversion"]["float_state_hash"]


def test_serialized_int8_reloads_bitwise(arrays, cfg, tmp_path, model, cal):
    eval_x = torch.from_numpy(np.repeat(arrays["validation"]["x"][:90, None, :], 50, axis=1))
    out = mq.ptq_unit(model, cal, cfg, tmp_path / "v2", expected_islands=TINY_ISLANDS, eval_x=eval_x)
    saved = json.loads((tmp_path / "v2/conversion.json").read_text())
    assert saved["reload_check"]["rows"] == 90 and saved["reload_check"]["logits_bitwise_equal"] is True
    assert saved["reload_check"]["argmax_identical"] is True
    assert saved["checkpoint_roundtrip_max_abs_error"] == 0.0 and saved["calibration_partition"] == "train"
    assert saved["fp32_remaining_operations"] and saved["parameter_counts"]["int8_weight_elements"] > 0
    compat = {"variant": "V2", "float_state_hash": out["conversion"]["float_state_hash"],
              "calibration_id_sha256": cal["id_sha256"], "scope": "all_linear", "mode": "static", "backend": "fbgemm"}
    reloaded, _ = load_int8(model, "all_linear", cal["x"], tmp_path / "v2/state.pt", compat)
    with torch.inference_mode():
        a, b = out["model"](eval_x)["logits"], reloaded(eval_x)["logits"]
    assert torch.equal(a, b) and torch.equal(a.argmax(1), b.argmax(1))
    assert model_state_hash(reloaded) == model_state_hash(out["model"])
    with pytest.raises(ValueError, match="Incompatible"):
        load_int8(model, "all_linear", cal["x"], tmp_path / "v2/state.pt", {**compat, "variant": "other"})
    with pytest.raises(FileExistsError):
        mq.ptq_unit(model, cal, cfg, tmp_path / "v2", expected_islands=TINY_ISLANDS)


def test_reload_mismatch_is_fatal(arrays, cfg, tmp_path, model, cal):
    out = mq.ptq_unit(model, cal, cfg, tmp_path / "v2", expected_islands=TINY_ISLANDS)
    other = copy.deepcopy(out["model"])
    with torch.no_grad():
        other.get_submodule("embedding").quant.scale.mul_(1.5)
    compat = {"variant": "V2", "float_state_hash": out["conversion"]["float_state_hash"],
              "calibration_id_sha256": cal["id_sha256"], "scope": "all_linear", "mode": "static", "backend": "fbgemm"}
    with pytest.raises(RuntimeError, match="differs"):
        mq.reload_check(other, model, "all_linear", cal["x"], tmp_path / "v2/state.pt", compat, cal["x"][:64])


def test_model_state_hash_handles_int8_and_matches_float_hash(model, cal):
    assert model_state_hash(model) == state_hash(model)
    q, _ = ptq(model, "all_linear", cal["x"])
    with pytest.raises(Exception):
        state_hash(q)  # the provenance hash cannot hash packed INT8 state
    h = model_state_hash(q)
    assert h == model_state_hash(copy.deepcopy(q))
    q2, _ = ptq(tiny_model(9), "all_linear", cal["x"])
    assert model_state_hash(q2) != h


def _original_model_state_hash(model):
    """SHA-256 over the state_dict; unlike provenance.state_hash it also handles packed INT8 state.

    Plain float tensors are hashed exactly as provenance.state_hash does (key, then raw bytes).
    Quantized tensors contribute integer values, scales and zero points; tuples (packed params) are
    walked; dtypes and other scalars contribute their repr.
    """
    h = hashlib.sha256()

    def feed(key, val):
        if isinstance(val, torch.Tensor):
            t = val.detach().cpu()
            if t.is_quantized:
                h.update(f"{key}|quantized|{t.qscheme()}".encode())
                h.update(t.int_repr().contiguous().numpy().tobytes())
                if t.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                    h.update(t.q_per_channel_scales().numpy().tobytes())
                    h.update(t.q_per_channel_zero_points().numpy().tobytes())
                    h.update(str(t.q_per_channel_axis()).encode())
                else:
                    h.update(repr((t.q_scale(), t.q_zero_point())).encode())
            else:
                h.update(key.encode())
                h.update(t.contiguous().numpy().tobytes())
        elif isinstance(val, (tuple, list)):
            for i, item in enumerate(val):
                feed(f"{key}[{i}]", item)
        else:
            h.update(f"{key}|{val!r}".encode())

    for key, val in model.state_dict().items():
        feed(key, val)
    return h.hexdigest()


def _fresh_observers():
    torch.manual_seed(0)
    return nn.ModuleDict({
        "minmax": aq.MinMaxObserver(),
        "perch": aq.PerChannelMinMaxObserver(ch_axis=0),
        "fq": aq.FakeQuantize(observer=aq.MinMaxObserver),
        "fq_perch": aq.FakeQuantize(observer=aq.PerChannelMinMaxObserver, ch_axis=0, qscheme=torch.per_channel_symmetric),
    })


def test_model_state_hash_leaves_fresh_qat_observers_usable(model, cal):
    mods = _fresh_observers()
    x = torch.randn(4, 3)
    fresh = {"minmax": mods["minmax"].min_val.clone(), "perch_shape": tuple(mods["perch"].min_val.shape)}
    model_state_hash(mods)  # must not export the live observer buffers to NumPy
    for name in mods:
        mods[name](x)  # first forward resizes the per-channel min_val/max_val
    assert not torch.equal(mods["minmax"].min_val, fresh["minmax"]) and torch.isfinite(mods["minmax"].min_val).all()
    assert fresh["perch_shape"] == (0,) and tuple(mods["perch"].min_val.shape) == (4,)
    assert tuple(mods["perch"].max_val.shape) == (4,) and torch.isfinite(mods["perch"].max_val).all()
    assert tuple(mods["fq_perch"].activation_post_process.min_val.shape) == (4,)
    assert torch.equal(mods["minmax"].max_val, x.max().reshape(()))
    qat, _ = prepare_static(model, "all_linear", cal["x"][:2], qat=True)
    model_state_hash(qat)
    qat(cal["x"][:2])  # first forward after hashing must succeed


def test_model_state_hash_matches_original_implementation_on_disposable_copies(model, cal):
    q, _ = ptq(model, "all_linear", cal["x"])
    qat, _ = prepare_static(model, "all_linear", cal["x"][:2], qat=True)
    for obj in (model, q, qat, _fresh_observers()):
        expected = _original_model_state_hash(copy.deepcopy(obj))
        assert model_state_hash(copy.deepcopy(obj)) == expected
        assert model_state_hash(obj) == expected  # hashing the live object gives the same digest


def test_model_state_hash_pinned_digests():
    lin = nn.Linear(3, 2)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7)
        lin.bias.copy_(torch.tensor([0.5, -1.25]))
    qt = torch.quantize_per_channel(torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2,
                                    torch.tensor([0.5, 0.25], dtype=torch.float64), torch.tensor([0, 1]), 0, torch.qint8)
    holder = nn.Module()
    holder.register_buffer("w", qt)
    assert model_state_hash(lin) == "324759f0fa8e7ba1641ac2cedbdb0d37b7c1b02ec14a832007985d865f33e3cf"
    assert model_state_hash(holder) == "75cf56b502e24ba03894f42325f025691cbcb6b53ee54632e1a3a0e6f3b3fd5d"


def test_qat_feasibility_supported_on_tiny_model(arrays, cfg, tmp_path, model, cal):
    eval_x = torch.from_numpy(np.repeat(arrays["validation"]["x"][:60, None, :], 50, axis=1))
    rng_before = torch.get_rng_state().clone()
    rec = mq.qat_feasibility(model, cal, arrays["train"]["x"][:300], arrays["train"]["y"][:300], TRAIN_CFG, cfg, 1,
                             eval_x, tmp_path, eval_y=arrays["validation"]["y"][:60], expected_islands=TINY_ISLANDS)
    assert torch.equal(rng_before, torch.get_rng_state())
    saved = json.loads((tmp_path / "qat_feasibility.json").read_text())
    assert saved["status"] == "supported" and saved["error"] is None
    assert saved["packed_dtypes"]["weight_dtype"] == {"torch.qint8": TINY_ISLANDS}
    assert saved["int8_inference"]["logits_finite"] is True and saved["epochs"] == 1
    assert saved["quantized_ops_profiled"]["quantized_linear_calls"] >= TINY_ISLANDS
    assert saved["serialized"] is True and (tmp_path / "qat_int8/state.pt").exists()
    assert rec["status"] == "supported"
    with pytest.raises(FileExistsError):
        mq.qat_feasibility(model, cal, arrays["train"]["x"][:100], arrays["train"]["y"][:100], TRAIN_CFG, cfg, 1,
                           eval_x, tmp_path, expected_islands=TINY_ISLANDS)


def test_qat_feasibility_records_unsupported_with_exact_evidence(arrays, cfg, tmp_path, model, cal, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("convert is not supported for this operator")

    monkeypatch.setattr(aq, "convert", boom)
    eval_x = cal["x"][:16]
    rec = mq.qat_feasibility(model, cal, arrays["train"]["x"][:200], arrays["train"]["y"][:200], TRAIN_CFG, cfg, 1,
                             eval_x, tmp_path, expected_islands=TINY_ISLANDS)
    saved = json.loads((tmp_path / "qat_feasibility.json").read_text())
    assert saved["status"] == "unsupported" and rec["status"] == "unsupported"
    assert saved["error"] == "RuntimeError: convert is not supported for this operator"
    assert saved["stage"] == "convert" and saved["where"] and "boom" in saved["where"]
    assert saved["packed_dtypes"] is None


def test_qat_feasibility_names_module_that_fails_in_forward(arrays, cfg, tmp_path, model, cal, monkeypatch):
    original = nn.ReLU.forward

    def broken(self, x):  # only the training-mode fine-tune pass fails; the eval-mode identity check passes
        if self.training:
            raise RuntimeError("relu exploded")
        return original(self, x)

    monkeypatch.setattr(nn.ReLU, "forward", broken)
    rec = mq.qat_feasibility(model, cal, arrays["train"]["x"][:100], arrays["train"]["y"][:100], TRAIN_CFG, cfg, 1,
                             cal["x"][:8], tmp_path, expected_islands=TINY_ISLANDS)
    monkeypatch.setattr(nn.ReLU, "forward", original)
    assert rec["status"] == "unsupported" and rec["stage"] == "finetune" and "relu exploded" in rec["error"]
    assert rec["module"]  # the last module entered before the failure is named


# ---------------------------------------------------------------- real model (slow)
SPLIT = ROOT / "reports/milestone2a/split/split_m2.json"
SCALER = ROOT / "reports/milestone2a/split/scaler_minmax.json"


@pytest.fixture(scope="module")
def real_arrays():
    if not SPLIT.is_file():
        pytest.skip("frozen split not present")
    from research import m2_split
    return m2_split.load_arrays(SPLIT, SCALER)


@pytest.mark.slow
def test_real_model_ptq_calibration_and_qat_feasibility(real_arrays, cfg, tmp_path):
    split = json.loads(SPLIT.read_text())
    cal = mq.calibration_ids(real_arrays, cfg["ptq"]["calibration_per_class"], split["split_seed"])
    assert cal["n"] == 540 and cal["per_class_counts"] == [20] * 27 and cal["partition"] == "train"
    assert set(cal["ids"]) <= set(real_arrays["train"]["ids"].tolist())
    assert not set(cal["ids"]) & set(real_arrays["validation"]["ids"].tolist())
    assert cal["x"].shape == (540, 50, 39)

    torch.manual_seed(3)
    model = make_model(TRAIN_CFG["architecture"]).eval()
    assert sum(p.numel() for p in model.parameters()) == 351_612
    eval_x = torch.from_numpy(np.repeat(real_arrays["validation"]["x"][:64, None, :], 50, axis=1))
    out = mq.ptq_unit(model, cal, cfg, tmp_path / "v2", eval_x=eval_x)
    conv = out["conversion"]
    assert conv["structure_checks"]["n_islands"] == 23 and conv["classification_path_islands"] == 21
    assert conv["anomaly_scorer_islands"] == 2
    assert conv["structure_checks"]["profiler"]["quantized_linear_calls"] == 23
    assert conv["quant_dequant_boundaries"]["total"] == 46
    assert conv["parameter_counts"]["total_parameters"] == 351_612
    assert conv["structure_checks"]["weight_dtype_inventory"] == {"torch.qint8": 23}
    assert json.loads((tmp_path / "v2/conversion.json").read_text())["reload_check"]["logits_bitwise_equal"] is True

    # QAT feasibility on the real architecture: record the outcome exactly, whichever way it goes
    rec = mq.qat_feasibility(model, cal, real_arrays["train"]["x"][:256], real_arrays["train"]["y"][:256], TRAIN_CFG, cfg,
                             1, eval_x, tmp_path / "qat", eval_y=real_arrays["validation"]["y"][:64])
    saved = json.loads((tmp_path / "qat/qat_feasibility.json").read_text())
    assert saved["status"] in ("supported", "unsupported") and saved["status"] == rec["status"]
    if saved["status"] == "supported":
        assert saved["packed_dtypes"]["weight_dtype"] == {"torch.qint8": 23}
        assert saved["int8_inference"]["logits_finite"] is True
    else:
        assert saved["error"] and saved["stage"]
