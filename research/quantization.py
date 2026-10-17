"""Actual CPU INT8 Linear islands and exact-forward, approximate-gradient BPDA."""
from __future__ import annotations

import copy
import warnings

import torch
from torch import nn
from torch.ao import quantization as aq
from torch.ao.nn.quantized import Linear as StaticLinear
from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear


class LinearIsland(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.quant = aq.QuantStub()
        self.linear = linear
        self.dequant = aq.DeQuantStub()

    def forward(self, x):
        # PyTorch 2.5.1 rejects quantized_resize_cpu_ under the deterministic
        # guard (and emits a C++ warning on EVERY call with warn_only=True).
        # Exempt only this actual static kernel; restore the caller's policy
        # immediately, including on failure. This is a single-process runner.
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        exempt = isinstance(self.linear, StaticLinear) and enabled
        if exempt:
            torch.use_deterministic_algorithms(False)
        try:
            return self.dequant(self.linear(self.quant(x)))
        finally:
            if exempt:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)


def selected_names(model, scope):
    if scope not in ("classifier", "ffn", "attention", "all_linear"):
        raise ValueError(f"Unknown quantization scope {scope}")
    return [name for name, module in model.named_modules() if type(module) is nn.Linear and (
        scope == "all_linear" or
        (scope == "classifier" and name.startswith("classification_head.")) or
        (scope == "ffn" and ".ff." in name) or
        (scope == "attention" and ".attn.W_" in name))]


def replace(model, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


def prepare_static(model, scope, example, *, qat=False, backend="fbgemm"):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"Unsupported quantization engine: {backend}")
    torch.backends.quantized.engine = backend
    warnings.warn("Static INT8 kernels are narrowly exempted from PyTorch's unsupported determinism guard; float operations retain the requested policy.", UserWarning)
    converted = copy.deepcopy(model).cpu().eval()
    names = selected_names(converted, scope)
    if not names:
        raise RuntimeError("Quantization request selected no Linear modules")
    for name in names:
        replace(converted, name, LinearIsland(converted.get_submodule(name)))
    with torch.inference_mode():
        expected, observed = model.cpu().eval()(example)["logits"], converted(example)["logits"]
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    converted.qconfig = None
    for name in names:
        converted.get_submodule(name).qconfig = (
            aq.get_default_qat_qconfig(backend) if qat else aq.get_default_qconfig(backend))
    if qat:
        converted.train()
        aq.prepare_qat(converted, inplace=True)
    else:
        aq.prepare(converted, inplace=True)
    return converted, names


def inventory(model, requested, mode):
    actual = []
    for name in requested:
        module = model.get_submodule(name)
        qlinear = module.linear if isinstance(module, LinearIsland) else module
        if not isinstance(qlinear, (StaticLinear, DynamicLinear)):
            raise RuntimeError(f"Requested INT8 module was not converted: {name}")
        weight = qlinear.weight()
        if not weight.is_quantized or weight.dtype != torch.qint8:
            raise RuntimeError(f"Non-INT8 weight: {name}")
        actual.append({"name": name, "class": f"{type(qlinear).__module__}.{type(qlinear).__name__}",
                       "weight_dtype": str(weight.dtype), "weight_shape": list(weight.shape),
                       "weight_qscheme": str(weight.qscheme()),
                       "activation": "quint8 static calibrated" if mode != "dynamic" else "dynamic runtime quantization"})
    return {"mode": mode, "requested": requested, "converted": actual,
            "backend": torch.backends.quantized.engine,
            "fp32_operators": ["attention QK/AV matmul", "softmax", "LayerNorm", "residual add", "pooling", "PE", "unselected linears"],
            "architecture_wrapper_fp32_max_abs_error": 0.0}


def ptq(model, scope, calibration, mode="static", backend="fbgemm"):
    if mode == "dynamic":
        torch.backends.quantized.engine = backend
        names = selected_names(model, scope)
        q = aq.quantize_dynamic(copy.deepcopy(model).cpu().eval(), set(names), dtype=torch.qint8)
    elif mode == "static":
        q, names = prepare_static(model, scope, calibration[:2], backend=backend)
        with torch.inference_mode():
            for batch in calibration.split(32):
                q(batch)
        q.eval()
        aq.convert(q, inplace=True)
    else:
        raise ValueError(f"Unsupported PTQ mode {mode}")
    return q, inventory(q, names, mode)


def load_int8(float_architecture, scope, calibration, checkpoint, compatibility,
              mode="static", backend="fbgemm"):
    """Reconstruct converted modules before loading their packed INT8 state."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["compatibility"] != compatibility:
        raise ValueError("Incompatible quantized checkpoint")
    q, inv = ptq(float_architecture, scope, calibration, mode, backend)
    q.load_state_dict(state["state_dict"], strict=True)
    q.eval()
    return q, inv


class _BPDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight):
        ctx.save_for_backward(weight)
        # Actual target kernel in forward; no FP32-copy approximation here.
        return target(x.detach())

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        return grad.matmul(weight), None, None


class BPDALinear(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        linear = target.linear if isinstance(target, LinearIsland) else target
        self.register_buffer("weight", linear.weight().dequantize())

    def forward(self, x):
        return _BPDA.apply(x, self.target, self.weight)


def bpda_model(target, requested, example):
    model = copy.deepcopy(target).eval()
    for name in requested:
        replace(model, name, BPDALinear(model.get_submodule(name)))
    with torch.no_grad():
        a, b = target(example)["logits"], model(example)["logits"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    return model
