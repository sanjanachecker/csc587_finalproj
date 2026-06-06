"""
src/quantize.py

Convert a trained FP32 model to a fake-quantized model ready for calibration.

Public API
----------
    quantize_model(model, n_bits=8, fold_bn=False, inplace=False)
        Walk the module tree and replace every Conv2d / Linear with
        QConv2d / QLinear.  Optionally fold BatchNorm into preceding Conv2d.

    restore_model(model)
        Undo quantization: replace QConv2d / QLinear back with the
        original Conv2d / Linear they wrap.  Useful for sensitivity analysis
        baselines.

Design notes
------------
* We recurse over named_children() (one level at a time) rather than
  named_modules() (all descendants flat) so that we can call setattr()
  on the correct *parent* module when swapping a layer.
* BatchNorm folding is off by default.  Turn it on with fold_bn=True once
  the basic pipeline is verified — it tightens the activation range seen
  by each layer's quantizer and typically recovers 0.2–0.5 pp accuracy.
* The classifier / final Linear is quantized by default.  If you want to
  keep it in FP32 (a common mixed-precision heuristic), pass
  skip_layer_names=['classifier'] to quantize_model().
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn

from fake_quant import QConv2d, QLinear


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Return a new Conv2d whose weight and bias absorb the BatchNorm parameters.

    The folded equivalence is:

        BN(Conv(x)) ≈ FoldedConv(x)

    where FoldedConv.weight = conv.weight * (γ / σ)
          FoldedConv.bias   = (conv.bias - μ) * (γ / σ) + β

    References
    ----------
    Krishnamoorthi (2018), Section 3.1.
    """
    assert bn.track_running_stats, "BN must track running stats for folding"

    # BN parameters
    gamma = bn.weight.data                          # γ  (scale)
    beta  = bn.bias.data                            # β  (shift)
    mu    = bn.running_mean                         # μ
    var   = bn.running_var                          # σ²
    eps   = bn.eps

    std   = torch.sqrt(var + eps)                   # σ
    scale = gamma / std                             # γ / σ, shape (C_out,)

    # Fold into conv weight: each output channel scaled by scale[c]
    folded_weight = conv.weight.data * scale.view(-1, 1, 1, 1)

    # Fold into conv bias
    conv_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(mu)
    folded_bias = (conv_bias - mu) * scale + beta

    # Build the new conv (same geometry, bias always present after folding)
    folded_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True,
    )
    folded_conv.weight = nn.Parameter(folded_weight)
    folded_conv.bias   = nn.Parameter(folded_bias)
    return folded_conv


def _fold_bn_in_module(module: nn.Module) -> nn.Module:
    """
    Walk a module's *direct* children and fold any (Conv2d, BatchNorm2d)
    consecutive pair into a single Conv2d.  Returns the module in-place.

    Handles the common patterns in EfficientNet and MobileNetV2:
        Sequential(Conv2d, BatchNorm2d, ...)
        Sequential(Conv2d, BatchNorm2d, ReLU, ...)
    """
    children = list(module.named_children())
    i = 0
    while i < len(children) - 1:
        name_a, layer_a = children[i]
        name_b, layer_b = children[i + 1]
        if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
            folded = fold_bn_into_conv(layer_a, layer_b)
            setattr(module, name_a, folded)
            # Replace BN with Identity so the rest of the graph is unchanged
            setattr(module, name_b, nn.Identity())
            # Update local list so the next iteration is correct
            children[i]     = (name_a, folded)
            children[i + 1] = (name_b, nn.Identity())
        i += 1
    return module


# ---------------------------------------------------------------------------
# Recursive module-tree replacement
# ---------------------------------------------------------------------------

def _replace_layers(
    module:           nn.Module,
    n_bits:           int,
    skip_layer_names: List[str],
    current_path:     str = "",
) -> None:
    """
    Recursively walk `module`, replacing Conv2d → QConv2d and
    Linear → QLinear in-place on the parent.

    Parameters
    ----------
    module : nn.Module
        The module whose *children* we are inspecting (not the child itself).
    n_bits : int
        Quantization bit-width passed to QConv2d / QLinear.
    skip_layer_names : list[str]
        If any element is a substring of the full dotted path to a layer,
        that layer is left in FP32.  E.g. ['classifier'] skips the head.
    current_path : str
        Dot-separated path built up during recursion (for skip matching).
    """
    for child_name, child_module in list(module.named_children()):
        full_path = f"{current_path}.{child_name}" if current_path else child_name

        # --- skip check ---------------------------------------------------
        if any(skip in full_path for skip in skip_layer_names):
            continue

        # --- Conv2d → QConv2d ---------------------------------------------
        if isinstance(child_module, nn.Conv2d):
            qconv = QConv2d.from_conv2d(child_module, n_bits=n_bits)
            setattr(module, child_name, qconv)

        # --- Linear → QLinear ---------------------------------------------
        elif isinstance(child_module, nn.Linear):
            qlinear = QLinear.from_linear(child_module, n_bits=n_bits)
            setattr(module, child_name, qlinear)

        # --- container: recurse -------------------------------------------
        else:
            _replace_layers(child_module, n_bits, skip_layer_names, full_path)


# ---------------------------------------------------------------------------
# Public: quantize_model
# ---------------------------------------------------------------------------

def quantize_model(
    model:            nn.Module,
    n_bits:           int = 8,
    fold_bn:          bool = False,
    inplace:          bool = False,
    skip_layer_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Convert a trained FP32 model to a fake-quantized model.

    Steps
    -----
    1. (Optional) Deep-copy the model so the FP32 original is untouched.
    2. (Optional) Fold BatchNorm into preceding Conv2d layers.
    3. Replace every Conv2d with QConv2d and every Linear with QLinear.
       Weight quantizers are calibrated immediately from the copied weights.
       Activation quantizers are left un-calibrated — call calibrate() next.

    Parameters
    ----------
    model : nn.Module
        Trained FP32 model in eval mode.
    n_bits : int
        Bit-width for quantization. Default 8.
    fold_bn : bool
        If True, fold BatchNorm2d into preceding Conv2d before quantizing.
        Recommended once the basic pipeline is verified; improves accuracy
        by ~0.2–0.5 pp by tightening activation ranges.
    inplace : bool
        If False (default), deep-copy the model first so the FP32 original
        is preserved for comparison.  If True, modify in-place (saves memory
        but destroys the FP32 model).
    skip_layer_names : list[str] | None
        Layer name substrings to leave in FP32.  Pass ['classifier'] to keep
        the final head in FP32 for a simple mixed-precision baseline.

    Returns
    -------
    nn.Module
        The quantized model with all activation quantizers un-calibrated.
        Call calibrate.calibrate() before running inference.

    Example
    -------
    >>> from models import build_model
    >>> from quantize import quantize_model
    >>> from calibrate import calibrate
    >>>
    >>> fp32_model = build_model('efficientnet_b0')
    >>> fp32_model.load_state_dict(torch.load('results/checkpoints/efficientnet_b0_fp32.pt'))
    >>> fp32_model.eval()
    >>>
    >>> qmodel = quantize_model(fp32_model, fold_bn=True)
    >>> calibrate(qmodel, calib_loader, n_images=256, device='cpu')
    >>> # qmodel is now ready for evaluation
    """
    if skip_layer_names is None:
        skip_layer_names = []

    # Step 1: copy
    qmodel = model if inplace else copy.deepcopy(model)
    qmodel.eval()

    # Step 2: BN folding (walk every submodule)
    if fold_bn:
        print("Folding BatchNorm into Conv2d …")
        n_folded = 0
        for submodule in qmodel.modules():
            before = sum(isinstance(c, nn.BatchNorm2d) for c in submodule.children())
            _fold_bn_in_module(submodule)
            after  = sum(isinstance(c, nn.BatchNorm2d) for c in submodule.children())
            n_folded += (before - after)
        print(f"  Folded {n_folded} BatchNorm layer(s).\n")

    # Step 3: replace Conv2d / Linear
    _replace_layers(qmodel, n_bits, skip_layer_names)

    # Summary
    n_qconv   = sum(1 for m in qmodel.modules() if isinstance(m, QConv2d))
    n_qlinear = sum(1 for m in qmodel.modules() if isinstance(m, QLinear))
    n_skipped = len(skip_layer_names)
    print(
        f"quantize_model: replaced {n_qconv} Conv2d → QConv2d, "
        f"{n_qlinear} Linear → QLinear"
        + (f" (skipped patterns: {skip_layer_names})" if n_skipped else "")
    )
    print("Activation quantizers are NOT yet calibrated — run calibrate() next.\n")

    return qmodel


# ---------------------------------------------------------------------------
# Public: restore_model
# ---------------------------------------------------------------------------

def restore_model(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Undo quantization: replace QConv2d / QLinear with the original
    Conv2d / Linear they wrap.

    Used by sensitivity.py to get a clean FP32 baseline inside the
    layer-wise loop without reloading the checkpoint each time.

    Parameters
    ----------
    model : nn.Module
        A quantized model (output of quantize_model).
    inplace : bool
        If False, deep-copy first.

    Returns
    -------
    nn.Module
        FP32 model (BN layers will be Identity if fold_bn was used).
    """
    rmodel = model if inplace else copy.deepcopy(model)

    def _restore(module: nn.Module) -> None:
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, QConv2d):
                setattr(module, child_name, child_module.conv)
            elif isinstance(child_module, QLinear):
                setattr(module, child_name, child_module.linear)
            else:
                _restore(child_module)

    _restore(rmodel)
    return rmodel


# ---------------------------------------------------------------------------
# Public: layer inventory
# ---------------------------------------------------------------------------

def list_quantized_layers(model: nn.Module) -> List[tuple[str, nn.Module]]:
    """
    Return a list of (dotted_name, module) for every QConv2d and QLinear
    in the model.  Useful for sensitivity.py iteration.
    """
    result = []
    for name, module in model.named_modules():
        if isinstance(module, (QConv2d, QLinear)):
            result.append((name, module))
    return result


def print_quantization_summary(model: nn.Module) -> None:
    """Print a layer inventory with calibration status."""
    layers = list_quantized_layers(model)
    if not layers:
        print("No quantized layers found.")
        return

    print(f"{'Layer':<55} {'Type':<10} {'W cal':>6} {'A cal':>6}")
    print("-" * 80)
    for name, module in layers:
        kind  = "QConv2d" if isinstance(module, QConv2d) else "QLinear"
        w_cal = "✓" if module.weight_fq.calibrated else "✗"
        a_cal = "✓" if module.act_fq.calibrated    else "✗"
        print(f"{name:<55} {kind:<10} {w_cal:>6} {a_cal:>6}")
    print(f"\nTotal: {len(layers)} quantized layers")