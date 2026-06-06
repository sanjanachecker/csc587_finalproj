import sys
from pathlib import Path
_SRC = Path(__file__).parent.parent   # src/
sys.path.insert(0, str(_SRC / 'models'))
sys.path.insert(0, str(_SRC / 'quantization'))
sys.path.insert(0, str(_SRC / 'evaluation'))

import torch
from models import build_model
from dataset import get_dataloaders
from quantize import quantize_model
from calibrate import calibrate
from sensitivity import run_sensitivity, build_mixed_precision, print_sensitivity, save_sensitivity
from quant_evaluate import evaluate_model, compare_results

device = "mps"

_, val_loader, test_loader = get_dataloaders(
    processed_dir='data/processed',
    data_root='data/raw',
    batch_size=32,
    num_workers=0,
)

# Load and quantize EfficientNet-B0
fp32 = build_model('efficientnet_b0')
ckpt = torch.load('results/checkpoints/efficientnet_b0_fp32.pt', map_location=device)
fp32.load_state_dict(ckpt['model_state'])
fp32.eval()

qmodel = quantize_model(fp32)
calibrate(qmodel, val_loader, n_images=256, device=device)

# Run sensitivity — use n_images=1000 for speed, full val for final results
results = run_sensitivity(qmodel, val_loader, device=device, n_images=1000)
print_sensitivity(results, top_k=15)
save_sensitivity(results, 'results/metrics/sensitivity/efficientnet_b0_sensitivity.json')

# Build mixed-precision model keeping top 5 most sensitive layers in FP32
build_mixed_precision(qmodel, results, top_k=5)
r_mp = evaluate_model(qmodel, test_loader, device=device, label='Mixed Precision (top-5 FP32)')

# Full INT8 for comparison
from sensitivity import _set_all
_set_all(qmodel, True)
r_int8 = evaluate_model(qmodel, test_loader, device=device, label='Full INT8')

compare_results([r_int8, r_mp])