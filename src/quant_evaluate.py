"""
src/evaluate.py

Evaluate any model configuration on the EuroSAT test set.
Produces accuracy, macro-F1, per-class F1, and confusion matrix.

Usage
-----
    from evaluate import evaluate_model, print_results, save_results

    results = evaluate_model(model, test_loader, device='cpu')
    print_results(results)
    save_results(results, 'results/metrics/efficientnet_b0_fakequant.json')
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import CLASSES


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    label: str = "",
    move_model: bool = True,
) -> dict:
    """
    Evaluate a model on a dataloader.

    Parameters
    ----------
    move_model : bool
        If True (default), call model.to(device) before inference.
        Set to False for PyTorch static/dynamic PTQ models: their weights
        are stored as QuantizedCPU tensors and calling .to('cpu') silently
        remaps them to the plain CPU dispatch key, breaking quantized kernels.
        PTQ baseline models are already on CPU after construction.
    """
    model.eval()
    if move_model:
        model.to(device)

    all_preds, all_labels = [], []

    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    elapsed = time.time() - start

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None).tolist()
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "label": label,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "n_samples": len(all_labels),
        "elapsed_sec": elapsed,
    }


def print_results(results: dict) -> None:
    label = results.get("label", "model")
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy:   {results['accuracy']*100:.2f}%")
    print(f"  Macro-F1:   {results['macro_f1']*100:.2f}%")
    print(f"  Samples:    {results['n_samples']}")
    print(f"  Time:       {results['elapsed_sec']:.1f}s")
    print(f"\n  Per-class F1:")
    for cls, f1 in zip(CLASSES, results["per_class_f1"]):
        bar = "█" * int(f1 * 20)
        print(f"    {cls:<25} {f1*100:5.1f}%  {bar}")


def save_results(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")


def compare_results(results_list: list[dict]) -> None:
    """Print a side-by-side comparison table of multiple configurations."""
    print(f"\n{'Configuration':<30} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 52)
    for r in results_list:
        print(f"{r['label']:<30} {r['accuracy']*100:>9.2f}%  {r['macro_f1']*100:>9.2f}%")

    # Per-class delta vs first result (assumed to be FP32 baseline)
    if len(results_list) > 1:
        baseline = results_list[0]
        print(f"\n  Per-class F1 drop vs '{baseline['label']}':")
        print(f"  {'Class':<25}", end="")
        for r in results_list[1:]:
            print(f"  {r['label']:>15}", end="")
        print()
        print("  " + "-" * (25 + 17 * (len(results_list) - 1)))
        for i, cls in enumerate(CLASSES):
            print(f"  {cls:<25}", end="")
            for r in results_list[1:]:
                delta = r["per_class_f1"][i] - baseline["per_class_f1"][i]
                print(f"  {delta*100:>+14.1f}%", end="")
            print()