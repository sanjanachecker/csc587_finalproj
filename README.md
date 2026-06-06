# CSC 587 Final Project: Post-Training Quantization for Satellite Land Cover Classification

**Authors:** Sanjana Checker, Sofija Dimitrijevic  
**Course:** CSC 587 — Deep Learning  
**Quarter:** Spring 2026

## Overview

This project investigates post-training quantization (PTQ) for satellite land cover classification on the EuroSAT RGB dataset. We implement affine INT8 quantization from scratch in a custom `FakeQuantize` module, compare it against PyTorch's built-in PTQ, and use it as the substrate for a layer-wise sensitivity analysis that produces a mixed-precision configuration.

**Contributions:**

1. A from-scratch `FakeQuantize` module (affine INT8, fixed scale/zero-point from one-pass calibration) wrapped in `QConv2d` and `QLinear` layers.
2. A four-way comparison of FP32, FakeQuant INT8, PyTorch dynamic PTQ, and PyTorch static PTQ across two architectures (EfficientNet-B0, MobileNetV2).
3. A layer-wise sensitivity analysis producing a mixed-precision configuration that fully recovers FP32 accuracy at near-INT8 cost.
4. A per-class degradation analysis linking quantization sensitivity to the semantic structure of EuroSAT's ten land cover classes.

See [`results/RESULTS.md`](results/RESULTS.md) for all metrics.

## Repository Structure

```
csc587_finalproj/
├── data/
│   ├── raw/                    # EuroSAT RGB (not committed)
│   └── processed/              # train/val/test split CSVs
├── results/
│   ├── checkpoints/            # FP32 weights (.pt, not committed)
│   ├── metrics/                # JSON metrics per configuration
│   └── RESULTS.md              # Summary tables
├── src/
│   ├── models/
│   │   ├── dataset.py          # EuroSAT dataset and dataloader
│   │   ├── make_splits.py      # Stratified 80/10/10 split CSVs
│   │   ├── models.py           # EfficientNet-B0 and MobileNetV2 wrappers
│   │   ├── train.py            # FP32 training loop
│   │   └── evaluate.py         # Accuracy, macro-F1, per-class F1
│   ├── quantization/
│   │   ├── fake_quant.py       # FakeQuantize, QConv2d, QLinear
│   │   ├── calibrate.py        # Activation range calibration via hooks
│   │   ├── quantize.py         # FP32 → INT8 module-tree replacement
│   │   └── custom_quant.py     # End-to-end quantization script
│   └── evaluation/
│       ├── ptq_baselines.py        # PyTorch dynamic and static PTQ (FX mode)
│       ├── quant_evaluate.py       # Evaluation harness
│       ├── sensitivity.py          # Layer-wise sensitivity analysis
│       ├── print_quant_results_eff.py   # Run all configs for EfficientNet-B0
│       ├── print_quant_results_mnv2.py  # Run all configs for MobileNetV2
│       └── run-sensitivity_analysis.py  # Sensitivity + mixed-precision pipeline
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`.

### Download EuroSAT RGB

```bash
mkdir -p data/raw data/processed
cd data/raw
curl -L -o EuroSAT_RGB.zip https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
unzip -q EuroSAT_RGB.zip
cd ../..
```

After unzipping, `data/raw/2750/` contains ten class subfolders with ~27,000 `.jpg` patches at 64×64.

### Generate splits

```bash
python src/models/make_splits.py
```

Produces `data/processed/{train,val,test}.csv` with a deterministic stratified 80/10/10 split (seed 42).

## Reproducing the Results

```bash
# Train FP32 baselines
python src/models/train.py --model efficientnet_b0 --epochs 40
python src/models/train.py --model mobilenet_v2 --epochs 40

# Evaluate all quantization configurations
python src/evaluation/print_quant_results_eff.py
python src/evaluation/print_quant_results_mnv2.py

# Layer sensitivity + mixed-precision (EfficientNet-B0)
python src/evaluation/run-sensitivity_analysis.py
```

All scripts must be run from the project root (`csc587_finalproj/`).

## Division of work
Sanjana:
- Quantizations and evaluation

Sofija:
- Model training
- Generating final results visualizations

## References

- Helber, P., et al. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE J-STARS*, 12(7), 2217–2226.
- Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.
- Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. *arXiv:1806.08342*.
- Nagel, M., et al. (2021). A white paper on neural network quantization. *arXiv:2106.08295*.
- Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
