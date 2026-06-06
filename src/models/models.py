"""Model definitions for EuroSAT classification.

Both backbones are initialized from ImageNet pretrained weights and given a
fresh 10-way classification head for EuroSAT.
"""
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10


def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # Classifier is Sequential(Dropout, Linear); replace the final Linear.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name in ("efficientnet_b0", "effnet", "effnet_b0"):
        return build_efficientnet_b0(pretrained)
    if name in ("mobilenet_v2", "mobilenetv2", "mnv2"):
        return build_mobilenet_v2(pretrained)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)