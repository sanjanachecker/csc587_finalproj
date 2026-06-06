"""
src/sensitivity.py

Layer-wise sensitivity analysis for a fake-quantized model.

For each quantizable layer in the model:
    1. Start from a fully FP32 model (all quantization disabled)
    2. Enable quantization on that one layer only
    3. Evaluate accuracy on the validation set
    4. Record the accuracy drop vs the FP32 baseline

The result is a ranked list of layers by sensitivity, used to build
a mixed-precision configuration that keeps the most sensitive layers
in FP32 and quantizes the rest.

Public API
----------
    run_sensitivity(model, val_loader, device, n_images=None)
        Returns a list of dicts sorted by accuracy drop (most sensitive first).

    build_mixed_precision(model, sensitivity_results, top_k=5)
        Returns a model with the top_k most sensitive layers left in FP32
        and all others quantized.

    save_sensitivity(results, path)
    print_sensitivity(results)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
from pathlib import Path
_SRC = Path(__file__).parent.parent   # src/
sys.path.insert(0, str(_SRC / 'models'))
sys.path.insert(0, str(_SRC / 'quantization'))
sys.path.insert(0, str(_SRC / 'evaluation'))

from fake_quant import QConv2d, QLinear
from quantize import list_quantized_layers


def _quick_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    n_images: Optional[int] = None,
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            if n_images and total >= n_images:
                break
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def _set_all(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.weight_fq.enabled = enabled
            m.act_fq.enabled = enabled


def run_sensitivity(
    model: nn.Module,
    val_loader: DataLoader,
    device: str | torch.device = "cpu",
    n_images: Optional[int] = None,
) -> list[dict]:
    """
    Run layer-wise sensitivity analysis.

    Parameters
    ----------
    model : nn.Module
        A fully calibrated fake-quantized model.
    val_loader : DataLoader
        Validation loader. Use n_images to limit evaluation per layer
        for speed (500-1000 is enough to rank layers reliably).
    device : str
        Device to run inference on.
    n_images : int | None
        If set, evaluate on only this many images per layer (faster).
        Full val set gives the most accurate ranking.

    Returns
    -------
    list[dict]
        Sorted by accuracy drop descending (most sensitive first).
        Each dict has: layer_name, layer_type, fp32_acc, quant_acc, drop.
    """
    model.eval()
    model.to(device)
    layers = list_quantized_layers(model)
    n = len(layers)

    # FP32 baseline: all quantization off
    _set_all(model, False)
    fp32_acc = _quick_accuracy(model, val_loader, device, n_images)
    print(f"FP32 baseline accuracy: {fp32_acc*100:.2f}%")
    print(f"Running sensitivity over {n} layers ...\n")

    results = []
    for i, (name, module) in enumerate(layers):
        # Enable only this one layer
        module.weight_fq.enabled = True
        module.act_fq.enabled = True

        acc = _quick_accuracy(model, val_loader, device, n_images)
        drop = fp32_acc - acc

        kind = "QConv2d" if isinstance(module, QConv2d) else "QLinear"
        results.append({
            "layer_name": name,
            "layer_type": kind,
            "fp32_acc": round(fp32_acc, 6),
            "quant_acc": round(acc, 6),
            "drop": round(drop, 6),
        })

        print(f"[{i+1:3d}/{n}] {name:<50} drop={drop*100:+.2f}%")

        # Disable again before moving to the next layer
        module.weight_fq.enabled = False
        module.act_fq.enabled    = False

    # Re-enable everything
    _set_all(model, True)

    results.sort(key=lambda x: x["drop"], reverse=True)
    return results


def build_mixed_precision(
    model: nn.Module,
    sensitivity_results: list[dict],
    top_k: int = 5,
) -> nn.Module:
    """
    Build a mixed-precision model: top_k most sensitive layers stay FP32,
    all others are quantized.

    Parameters
    ----------
    model : nn.Module
        A fully calibrated fake-quantized model (all layers enabled).
    sensitivity_results : list[dict]
        Output of run_sensitivity(), sorted most-sensitive first.
    top_k : int
        Number of layers to leave in FP32.

    Returns
    -------
    nn.Module
        The same model with top_k layers' quantization disabled.
    """
    fp32_layer_names = {r["layer_name"] for r in sensitivity_results[:top_k]}

    _set_all(model, True)

    for name, module in list_quantized_layers(model):
        if name in fp32_layer_names:
            module.weight_fq.enabled = False
            module.act_fq.enabled = False

    kept = len(fp32_layer_names)
    total = len(sensitivity_results)
    print(f"Mixed precision: {kept} layers FP32, {total - kept} layers INT8.")
    print(f"FP32 layers: {sorted(fp32_layer_names)}\n")
    return model


def save_sensitivity(results: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")


def print_sensitivity(results: list[dict], top_k: int = 20) -> None:
    print(f"\n{'Rank':<5} {'Layer':<50} {'Type':<10} {'Drop':>8}")
    print("-" * 78)
    for i, r in enumerate(results[:top_k]):
        bar = "█" * min(40, int(abs(r["drop"]) * 400))
        print(
            f"{i+1:<5} {r['layer_name']:<50} {r['layer_type']:<10} "
            f"{r['drop']*100:>+7.2f}%  {bar}"
        )
    if len(results) > top_k:
        remaining = sum(1 for r in results[top_k:] if r["drop"] > 0.001)
        print(f"  ... {len(results) - top_k} more layers "
              f"({remaining} with drop > 0.1%)")