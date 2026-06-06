import torch
from models import build_model
from dataset import get_dataloaders
from quantize import quantize_model
from calibrate import calibrate
from ptq_baselines import apply_dynamic_ptq, apply_static_ptq
from quant_evaluate import evaluate_model, print_results, compare_results, save_results

_, val_loader, test_loader = get_dataloaders(
    processed_dir='data/processed',
    data_root='data/raw',
    batch_size=32,
    num_workers=0,
)

# Load FP32
fp32 = build_model('efficientnet_b0')
ckpt = torch.load('results/checkpoints/efficientnet_b0_fp32.pt', map_location='cpu')
fp32.load_state_dict(ckpt['model_state'])
fp32.eval()

# FP32 baseline
r_fp32 = evaluate_model(fp32, test_loader, device='mps', label='FP32')

# FakeQuant
qmodel = quantize_model(fp32)
calibrate(qmodel, val_loader, n_images=256, device='mps')
r_fq = evaluate_model(qmodel, test_loader, device='mps', label='FakeQuant INT8')

# Dynamic PTQ (CPU only)
fp32.cpu()
dyn = apply_dynamic_ptq(fp32)
r_dyn = evaluate_model(dyn, test_loader, device='cpu', label='PyTorch Dynamic', move_model=False)

# Static PTQ (CPU only)
sta = apply_static_ptq(fp32, val_loader, n_images=256)
r_sta = evaluate_model(sta, test_loader, device='cpu', label='PyTorch Static', move_model=False)

# Print everything
for r in [r_fp32, r_fq, r_dyn, r_sta]:
    print_results(r)
    save_results(r, f"results/metrics/efficientnet_b0_{r['label'].replace(' ', '_')}.json")

compare_results([r_fp32, r_fq, r_dyn, r_sta])