# Results

## Quantization — EfficientNet-B0

| Configuration | Accuracy | Macro-F1 | Inference |
|---|---:|---:|---:|
| FP32 (baseline) | 98.74% | 98.69% | 5.6s |
| FakeQuant INT8 | 92.33% | 91.99% | 6.4s |
| PyTorch Dynamic | 98.74% | 98.69% | 66.1s |
| PyTorch Static (FX) | 87.56% | 87.56% | 5.1s |

**Per-class F1 drop vs FP32**

| Class | FakeQuant | Dynamic | Static |
|---|---:|---:|---:|
| AnnualCrop | −11.7 | 0.0 | −20.4 |
| Forest | −9.7 | 0.0 | −26.6 |
| HerbaceousVegetation | −4.1 | 0.0 | −6.6 |
| Highway | −1.0 | 0.0 | −1.2 |
| Industrial | −0.6 | 0.0 | −3.1 |
| Pasture | −20.1 | 0.0 | −19.7 |
| PermanentCrop | −4.0 | 0.0 | −8.0 |
| Residential | −0.7 | 0.0 | −1.7 |
| River | −5.3 | 0.0 | −6.5 |
| SeaLake | −9.7 | 0.0 | −17.4 |

---

## Quantization — MobileNetV2

| Configuration | Accuracy | Macro-F1 | Inference |
|---|---:|---:|---:|
| FP32 (baseline) | 96.81% | 96.78% | 5.9s |
| FakeQuant INT8 | 96.85% | 96.82% | 6.1s |
| PyTorch Dynamic | 96.81% | 96.78% | 37.5s |
| PyTorch Static (FX) | 11.11%* | 2.00%* | 5.6s |

*\* qnnpack backend incompatibility with MobileNetV2's depthwise-conv + residual-add topology on Apple Silicon. Model outputs a constant tensor for all inputs.*

**Per-class F1 drop vs FP32**

| Class | FakeQuant | Dynamic |
|---|---:|---:|
| AnnualCrop | −0.5 | 0.0 |
| Forest | +0.3 | 0.0 |
| HerbaceousVegetation | +0.2 | 0.0 |
| Highway | +0.2 | 0.0 |
| Industrial | −0.2 | 0.0 |
| Pasture | +0.5 | 0.0 |
| PermanentCrop | −0.2 | 0.0 |
| Residential | 0.0 | 0.0 |
| River | −0.2 | 0.0 |
| SeaLake | +0.3 | 0.0 |

---

## Sensitivity Analysis — EfficientNet-B0

Evaluated on 1,000 validation images. Each layer enabled individually; all others in FP32.

| Rank | Layer | Type | Accuracy Drop |
|---:|---|---|---:|
| 1 | features.1.0.block.0.0 | QConv2d | −3.81% |
| 2 | features.1.0.block.2.0 | QConv2d | −0.20% |
| 3 | features.2.0.block.1.0 | QConv2d | −0.20% |
| 4–82 | all remaining layers | — | 0.00% |

**Mixed precision:** top-5 most sensitive layers kept in FP32, remaining 77 in INT8.

| Configuration | Accuracy | Macro-F1 |
|---|---:|---:|
| Full INT8 | 92.33% | 91.99% |
| Mixed Precision (top-5 FP32) | 98.74% | 98.71% |

Protecting just 5 layers (6% of the network) fully recovers FP32 accuracy. The first depthwise block (`features.1.0.block.0.0`) accounts for nearly all of the accuracy loss on its own.
