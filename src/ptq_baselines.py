"""
src/ptq_baselines.py

PyTorch built-in PTQ wrappers for comparison against our custom FakeQuantize.

Two approaches:
1. Dynamic PTQ  -- weights quantized ahead of time, activations quantized
                   on-the-fly at runtime. No calibration pass needed.
2. Static PTQ   -- weights AND activations quantized ahead of time using a
                   calibration pass. Scoped to Linear layers only on Apple
                   Silicon due to Conv2d backend limitations (see note below).

IMPORTANT: PyTorch's quantization backend only runs on CPU.
           Both functions move the model to CPU before quantizing.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ARM/Apple Silicon backend. fbgemm is x86-only.
torch.backends.quantized.engine = 'qnnpack'


def apply_dynamic_ptq(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic PTQ.
    """
    print("Applying PyTorch dynamic PTQ ...")
    qmodel = copy.deepcopy(model).cpu()
    qmodel.eval()
    qmodel = torch.quantization.quantize_dynamic(
        qmodel,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    print("Dynamic PTQ complete.\n")
    return qmodel


def apply_static_ptq(
    model: nn.Module,
    calib_loader: DataLoader,
    n_images: int = 256,
) -> nn.Module:
    
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import get_default_qconfig_mapping

    print("Applying PyTorch static PTQ (FX graph mode) ...")
    qmodel = copy.deepcopy(model).cpu()
    qmodel.eval()

    qconfig_mapping = get_default_qconfig_mapping('qnnpack')

    example_inputs = (torch.zeros(1, 3, 64, 64),)

    from torch.ao.quantization.quantize_fx import fuse_fx
    qmodel = fuse_fx(qmodel)

    prepared = prepare_fx(qmodel, qconfig_mapping, example_inputs)

    # calibration pass
    print(f"  Running calibration pass over {n_images} images ...")
    images_seen = 0
    with torch.no_grad():
        for images, _ in calib_loader:
            if images_seen >= n_images:
                break
            images = images[:n_images - images_seen].cpu()
            prepared(images)
            images_seen += images.shape[0]
    print(f"  Calibration complete ({images_seen} images).")

    converted = convert_fx(prepared)
    print("Static PTQ complete.\n")
    return converted


def get_model_size_mb(model: nn.Module, path: str = "/tmp/_tmp_model.pt") -> float:
    """Save model to disk and return size in MB."""
    torch.save(model.state_dict(), path)
    import os
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb