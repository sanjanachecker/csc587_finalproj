"""
src/calibrate.py

Activation-range calibration for fake-quantized models.

How it works
------------
1. Register a forward pre-hook on every QConv2d and QLinear in the model.
   The hook records the min and max of the *input* tensor (i.e. the true
   FP32 activation arriving at that layer, before fake-quant runs).
2. Run a single forward pass over `n_images` stratified calibration images
   with the model in eval mode and torch.no_grad().
3. After the pass, compute scale / zero_point from the observed [min, max]
   and write them into each layer's act_fq via set_calibration().
4. Remove all hooks.

Why input hooks instead of output hooks?
  We want to calibrate the *activation quantizer*, which sits at the
  entrance of each QConv2d/QLinear and quantizes the incoming feature map.
  Hooking the input gives us exactly the tensor that act_fq will see at
  inference time.

Why min/max calibration (not percentile)?
  Simple and sufficient for EuroSAT (small, balanced, low-dynamic-range
  imagery). If you observe large accuracy drops due to outliers, swap in
  the percentile collector below.

Public API
----------
    calibrate(model, dataloader, n_images=256, device='cpu')
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

from fake_quant import FakeQuantize, QConv2d, QLinear


# ---------------------------------------------------------------------------
# Scale / zero-point helpers (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

def _compute_scale_zp(x_min: float, x_max: float, n_bits: int = 8):
    """Asymmetric affine scale and zero-point from observed [x_min, x_max]."""
    q_min, q_max = 0, 2 ** n_bits - 1
    # Guard against degenerate constant-activation layers
    if x_max == x_min:
        x_max = x_min + 1e-6
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = int(round(-x_min / scale))
    zero_point = max(q_min, min(q_max, zero_point))
    return scale, zero_point


# ---------------------------------------------------------------------------
# Range collectors
# ---------------------------------------------------------------------------

class _MinMaxCollector:
    """Accumulates the global min and max seen across all calibration batches."""

    def __init__(self) -> None:
        self.min_val = float("inf")
        self.max_val = -float("inf")

    def update(self, x: torch.Tensor) -> None:
        self.min_val = min(self.min_val, float(x.min()))
        self.max_val = max(self.max_val, float(x.max()))

    def result(self):
        return self.min_val, self.max_val


class _PercentileCollector:
    """
    Accumulates a flat sample of values and returns a clipped [p_low, p_high]
    range.  More robust to outliers than min/max, at the cost of memory.

    Not used by default — swap in calibrate() if you see outlier sensitivity.
    """

    def __init__(self, p_low: float = 0.001, p_high: float = 0.999) -> None:
        self.p_low = p_low
        self.p_high = p_high
        self._buf: List[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        # Store a flat subsample to keep memory bounded
        flat = x.detach().float().flatten()
        if flat.numel() > 4096:
            idx  = torch.randperm(flat.numel())[:4096]
            flat = flat[idx]
        self._buf.append(flat.cpu())

    def result(self):
        all_vals = torch.cat(self._buf)
        lo = float(torch.quantile(all_vals, self.p_low))
        hi = float(torch.quantile(all_vals, self.p_high))
        return lo, hi


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def calibrate(
    model: nn.Module,
    dataloader: DataLoader,
    n_images: int = 256,
    device: str | torch.device = "cpu",
    use_percentile: bool = False,
) -> None:
    """
    Calibrate activation quantizers for all QConv2d and QLinear layers.

    Parameters
    ----------
    model : nn.Module
        A model whose Conv2d/Linear layers have already been replaced with
        QConv2d/QLinear by quantize.py.  Must be in eval mode before calling.
    dataloader : DataLoader
        Yields (images, labels) batches.  Calibration stops after n_images
        images have been seen; it does not need to be a special calibration
        loader — the training or val loader is fine.
    n_images : int
        Number of images to run through the model.  256 is sufficient for
        EuroSAT; increase if accuracy is sensitive.
    device : str | torch.device
        Where to run inference.
    use_percentile : bool
        If True, use PercentileCollector instead of MinMaxCollector.
    """

    model.eval()
    model.to(device)

    # ------------------------------------------------------------------ #
    # 1. Find all quantizable layers and attach hooks                     #
    # ------------------------------------------------------------------ #

    # Map from module -> collector
    collectors: Dict[nn.Module, _MinMaxCollector | _PercentileCollector] = {}
    hooks = []

    def _make_hook(layer: nn.Module):
        """Return a pre-hook closure that updates this layer's collector."""
        def hook(module, args):
            # args is a tuple; the activation is always the first element
            x = args[0]
            collectors[layer].update(x.detach().float())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (QConv2d, QLinear)):
            collectors[module] = (
                _PercentileCollector() if use_percentile else _MinMaxCollector()
            )
            # register_forward_pre_hook fires *before* the module's forward,
            # so we see the true FP32 activation before act_fq touches it.
            h = module.register_forward_pre_hook(_make_hook(module))
            hooks.append(h)

    if not collectors:
        raise RuntimeError(
            "calibrate(): no QConv2d or QLinear layers found in the model. "
            "Run quantize.py first to replace Conv2d/Linear layers."
        )

    print(f"Calibrating {len(collectors)} layers over {n_images} images …")

    # ------------------------------------------------------------------ #
    # 2. Forward pass over calibration data                               #
    # ------------------------------------------------------------------ #

    # Temporarily disable fake-quant so the activations reaching each hook
    # are true FP32 values, not already-quantized values from the previous
    # layer.  (Quantization error compounds if you calibrate on quantized
    # activations.)
    _set_quantization_enabled(model, enabled=False)

    images_seen = 0
    with torch.no_grad():
        for images, _ in dataloader:
            if images_seen >= n_images:
                break
            # Only take what we need from this batch
            remaining = n_images - images_seen
            images = images[:remaining].to(device)

            model(images)
            images_seen += images.shape[0]

    print(f"  Calibration forward pass complete ({images_seen} images).")

    # ------------------------------------------------------------------ #
    # 3. Remove hooks                                                     #
    # ------------------------------------------------------------------ #

    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------ #
    # 4. Write scale / zero_point into each layer's act_fq               #
    # ------------------------------------------------------------------ #

    _set_quantization_enabled(model, enabled=True)   # re-enable

    n_calibrated = 0
    for module, collector in collectors.items():
        x_min, x_max = collector.result()
        scale, zp = _compute_scale_zp(x_min, x_max)
        module.act_fq.set_calibration(scale, zp)
        n_calibrated += 1

    print(f"  Wrote scale/zero_point to {n_calibrated} activation quantizers.")
    print("Calibration complete.\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_quantization_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle fake-quant on all QConv2d and QLinear layers in the model."""
    for module in model.modules():
        if isinstance(module, (QConv2d, QLinear)):
            if enabled:
                module.enable_quantization()
            else:
                module.disable_quantization()


def print_calibration_summary(model: nn.Module) -> None:
    """
    Print a table of calibrated scale / zero_point for every quantized layer.
    Useful for debugging and for the report.
    """
    header = f"{'Layer':<55} {'W scale':>10} {'W zp':>6} {'A scale':>10} {'A zp':>6}"
    print(header)
    print("-" * len(header))

    for name, module in model.named_modules():
        if isinstance(module, (QConv2d, QLinear)):
            wfq = module.weight_fq
            afq = module.act_fq
            w_scale = f"{wfq.scale.item():.6f}" if wfq.calibrated else "—"
            w_zp = f"{wfq.zero_point.item()}"  if wfq.calibrated else "—"
            a_scale = f"{afq.scale.item():.6f}"   if afq.calibrated else "NOT SET"
            a_zp = f"{afq.zero_point.item()}"  if afq.calibrated else "NOT SET"
            print(f"{name:<55} {w_scale:>10} {w_zp:>6} {a_scale:>10} {a_zp:>6}")