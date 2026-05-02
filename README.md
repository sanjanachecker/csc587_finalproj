# CSC 587 Final Project: Post-Training Quantization for Satellite Land Cover Classification

**Authors:** Sanjana Checker, Sofija Dimitrijevic
**Course:** CSC 587 — Deep Learning
**Quarter:** Spring 2026

## Overview

This project investigates post-training quantization (PTQ) for satellite land cover classification on the EuroSAT RGB dataset. We implement affine INT8 quantization from scratch in a custom `FakeQuantize` module, compare it head-to-head against PyTorch's built-in PTQ, and use it as the substrate for a layer-wise sensitivity analysis that produces a mixed-precision configuration.

The headline contributions are:

1. A from-scratch `FakeQuantize` module (affine quantization, fixed scale and zero-point from one-pass calibration) wrapped in `QConv2d` and `QLinear` layers.
2. A four-way comparison of FP32, PyTorch dynamic PTQ, PyTorch static PTQ, and our FakeQuantize across two architectures (EfficientNet-B0, MobileNetV2).
3. A layer-wise sensitivity analysis enabled by our custom module, used to construct a mixed-precision configuration that recovers most of the FP32 accuracy at near-INT8 cost.
4. A per-class degradation analysis tying quantization sensitivity to the semantic structure of EuroSAT's ten land cover classes.

## Repository Structure

```
csc587_finalproj/
├── data/
│   ├── raw/                    # EuroSAT RGB (downloaded, not committed)
│   └── processed/              # train/val/test split CSVs
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fp32_baselines.ipynb
│   ├── 03_fakequant_unit_tests.ipynb
│   ├── 04_quantization_comparison.ipynb
│   └── 05_layer_sensitivity.ipynb
├── results/
│   ├── checkpoints/            # FP32 model weights (.pt, not committed)
│   ├── figures/                # Final plots for the report
│   ├── logs/                   # Training logs
│   └── metrics/                # JSON/CSV metrics per configuration
├── src/
│   ├── dataset.py              # EuroSAT dataset and dataloader
│   ├── make_splits.py          # Generate stratified 80/10/10 split CSVs
│   ├── models.py               # EfficientNet-B0 and MobileNetV2 wrappers
│   ├── train.py                # FP32 training loop
│   ├── fake_quant.py           # Custom FakeQuantize, QConv2d, QLinear
│   ├── calibrate.py            # Activation range calibration via hooks
│   ├── quantize.py             # Module-tree replacement for converting FP32 -> INT8
│   ├── ptq_baselines.py        # PyTorch dynamic and static PTQ wrappers
│   ├── sensitivity.py          # Layer-wise sensitivity analysis
│   ├── benchmark.py            # CPU latency and model size measurement
│   └── evaluate.py             # Accuracy, macro-F1, per-class F1, confusion matrix
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- A virtual environment (`.venv/` is gitignored)
- ~500MB free disk for the EuroSAT dataset
- Colab or a machine with GPU access (CPU is fine for benchmarking, slow for training)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `jupyter`.

### Download EuroSAT RGB

```bash
mkdir -p data/raw data/processed
cd data/raw
curl -L -o EuroSAT_RGB.zip https://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip -q EuroSAT_RGB.zip
cd ../..
```

After unzipping, `data/raw/2750/` contains ten class subfolders with ~27,000 total `.jpg` patches at 64×64.

If `madm.dfki.de` is slow, the Zenodo mirror works: `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip`.

### Generate splits

```bash
python src/make_splits.py
```

Produces `data/processed/{train,val,test}.csv` with a deterministic stratified 80/10/10 split (seed 42). Train: ~21,600. Val: ~2,700. Test: ~2,700.

## Project Plan

The project runs over six project weeks (quarter weeks 5–10). Each week has explicit deliverables.

### Week 1 — Data setup and pipeline

**Goal:** Working dataloader, sanity-checked class balance, and the scaffolding for the training loop.

- [ ] Install dependencies and verify GPU access in Colab.
- [ ] Download EuroSAT RGB to `data/raw/`.
- [ ] Implement `src/make_splits.py` to produce stratified CSV splits.
- [ ] Implement `src/dataset.py` with `EuroSATRGB` Dataset class and `get_dataloaders` helper.
- [ ] Notebook `01_data_exploration.ipynb`: visualize a batch, compute per-class counts, sanity-check the splits.

**Milestone:** Working dataloader, batch visualization, and class-balance stats committed.

### Week 2 — FP32 baselines

**Goal:** Two trained FP32 classifiers with full classification metrics.

- [ ] Implement `src/models.py` with EfficientNet-B0 and MobileNetV2 wrappers (ImageNet pretrained, 10-way classification head).
- [ ] Implement `src/train.py` (cross-entropy loss, Adam optimizer, 30–50 epochs, early stopping on validation accuracy).
- [ ] Implement `src/evaluate.py` (accuracy, macro-F1, per-class F1, confusion matrix).
- [ ] Train both architectures, save checkpoints to `results/checkpoints/`.
- [ ] Notebook `02_fp32_baselines.ipynb`: full evaluation report with confusion matrices.

**Milestone:** Two FP32 baselines with all classification metrics reported.

**Tuning notes:** Initial learning rate around 1e-3 with a step or cosine schedule. Watch for overfitting — EuroSAT is small, so early stopping matters. Class weighting only if any class drops below ~2,500 train samples.

### Week 3 — Custom FakeQuantize implementation

**Goal:** Working from-scratch quantization module with passing unit tests.

- [ ] Implement `src/fake_quant.py`:
  - `FakeQuantize` module: forward pass applies `q = round(x / s) + z` and back, no parameter learning.
  - `QConv2d` and `QLinear` wrappers that quantize both weights and activations.
  - Fixed `scale` and `zero_point` set externally (no learnable params; this is PTQ, not QAT).
- [ ] Implement `src/calibrate.py`: forward hooks that record per-tensor activation min/max during a calibration pass over 256 stratified images.
- [ ] Implement `src/quantize.py`: walk a trained FP32 model's module tree and replace `Conv2d`/`Linear` with their quantized counterparts using the calibrated scales.
- [ ] Notebook `03_fakequant_unit_tests.ipynb`: verify against `torch.quantize_per_tensor` on synthetic tensors. Round-trip error should be near-zero up to floating-point precision.

**Milestone:** Custom FakeQuantize module with unit tests passing.

**Implementation notes:** Watch out for batch-norm folding (you can either skip it for the first pass and just live with worse calibration, or fold BN into preceding Conv layers as a post-training step). For activations, make sure your hooks handle modules with multiple outputs correctly — wrapping every module is overkill, only wrap `Conv2d` and `Linear`.

### Week 4 — Apply quantization, run all PTQ configurations

**Goal:** Three INT8 configurations evaluated on both architectures.

- [ ] Implement `src/ptq_baselines.py` wrapping `torch.quantization.quantize_dynamic` and the static PTQ workflow.
- [ ] Apply our FakeQuantize as a forward-pass simulator on both FP32 baselines.
- [ ] Apply PyTorch dynamic and static PTQ on both architectures.
- [ ] Evaluate all configurations on the test set; save metrics to `results/metrics/`.
- [ ] Begin per-class degradation analysis (ΔF1 per class for each configuration).
- [ ] Notebook `04_quantization_comparison.ipynb`: side-by-side comparison.

**Milestone:** Three INT8 configurations × two architectures = six quantized models evaluated.

**Sanity check:** Our FakeQuantize accuracy should land within ~1–2 percentage points of PyTorch's static PTQ. If it doesn't, the calibration hooks are buggy or the module-tree replacement missed a layer.

### Week 5 — Layer-wise sensitivity and benchmarking

**Goal:** Layer-sensitivity plot, mixed-precision configuration, full tradeoff table.

- [ ] Implement `src/sensitivity.py`: for each quantizable layer in the strongest baseline, quantize that layer alone and measure accuracy drop.
- [ ] Build a mixed-precision configuration: keep top-K most sensitive layers in FP32, quantize the rest.
- [ ] Evaluate the mixed-precision model against full INT8 and FP32.
- [ ] Implement `src/benchmark.py`: CPU latency over 1,000 images with warmup, model size on disk.
- [ ] Run benchmarks on all configurations; record latency mean/std and disk size.
- [ ] Begin the final report.
- [ ] Notebook `05_layer_sensitivity.ipynb`: sensitivity bar plot, mixed-precision results.

**Milestone:** Layer-sensitivity plot, mixed-precision result, full accuracy-latency-size tradeoff table.

**Watch out:** Latency measurements on Colab CPUs are noisy. Run with `num_threads=1` and at least 1,000 iterations after a 100-iteration warmup. Report median or mean ± std.

### Week 6 — Visualizations, report, submit

**Goal:** Final report submitted with all required visualizations.

- [ ] Confusion matrices for FP32 and INT8 side by side, both architectures.
- [ ] Per-class ΔF1 heatmap across all (architecture, configuration) pairs.
- [ ] Layer-sensitivity plot (already produced in week 5; polish for the report).
- [ ] Misclassification grid for the most-degraded class under INT8.
- [ ] Pareto plot: accuracy vs. model size, accuracy vs. latency, all configurations labeled.
- [ ] Write up Method, Results, and Discussion sections.
- [ ] Cross-review between authors. Submit.

**Milestone:** Final code and report submitted.

## Reproducing the Results

Once everything is in place, the full pipeline runs as:

```bash
# Data
python src/make_splits.py

# Train FP32 baselines
python src/train.py --model efficientnet_b0 --epochs 40
python src/train.py --model mobilenet_v2 --epochs 40

# Quantize and evaluate
python src/quantize.py --model efficientnet_b0 --method fakequant
python src/quantize.py --model efficientnet_b0 --method ptq_dynamic
python src/quantize.py --model efficientnet_b0 --method ptq_static
python src/quantize.py --model mobilenet_v2 --method fakequant
python src/quantize.py --model mobilenet_v2 --method ptq_dynamic
python src/quantize.py --model mobilenet_v2 --method ptq_static

# Layer sensitivity (on the stronger of the two baselines)
python src/sensitivity.py --model efficientnet_b0

# Benchmarks
python src/benchmark.py --all

# Evaluate everything and dump metrics
python src/evaluate.py --all
```

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE J-STARS*, 12(7), 2217–2226.
- Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.
- Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. *arXiv:1806.08342*.
- Nagel, M., et al. (2021). A white paper on neural network quantization. *arXiv:2106.08295*.
- Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
- Zhao, S., et al. (2023). Land use and land cover classification meets deep learning: A review. *Sensors*, 23(21), 8966.