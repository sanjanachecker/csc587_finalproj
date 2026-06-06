"""Evaluation utilities: accuracy, macro-F1, per-class F1, confusion matrix."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report)

from dataset import CLASSES


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on loader, return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def compute_metrics(y_true, y_pred):
    """Return a dict of all the metrics we care about."""
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro"),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1":   {CLASSES[i]: f
                           for i, f in enumerate(
                               f1_score(y_true, y_pred, average=None,
                                        labels=list(range(len(CLASSES)))))},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def print_report(y_true, y_pred):
    """Pretty-print classification report."""
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))


def evaluate_model(model, loader, device):
    """Convenience: run + compute + return metrics dict."""
    y_true, y_pred = get_predictions(model, loader, device)
    return y_true, y_pred, compute_metrics(y_true, y_pred)