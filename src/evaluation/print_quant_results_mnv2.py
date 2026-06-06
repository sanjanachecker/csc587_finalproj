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
from ptq_baselines import apply_dynamic_ptq, apply_static_ptq
from quant_evaluate import evaluate_model, print_results, compare_results, save_results

MODEL_NAME  = 'mobilenet_v2'
CKPT_PATH   = 'results/checkpoints/mobilenet_v2_fp32.pt'
OUT_PREFIX  = 'results/metrics/mobilenet_v2'
DEVICE      = 'mps'

_, val_loader, test_loader = get_dataloaders(
    processed_dir='data/processed',
    data_root='data/raw',
    batch_size=32,
    num_workers=0,
)

fp32 = build_model(MODEL_NAME)
ckpt = torch.load(CKPT_PATH, map_location='cpu')
fp32.load_state_dict(ckpt['model_state'])
fp32.eval()
r_fp32 = evaluate_model(fp32, test_loader, device=DEVICE, label='FP32')

qmodel = quantize_model(fp32)
calibrate(qmodel, val_loader, n_images=256, device=DEVICE)
r_fq = evaluate_model(qmodel, test_loader, device=DEVICE, label='FakeQuant INT8')

fp32.cpu()
dyn = apply_dynamic_ptq(fp32)
r_dyn = evaluate_model(dyn, test_loader, device='cpu', label='PyTorch Dynamic', move_model=False)

sta = apply_static_ptq(fp32, val_loader, n_images=256)
r_sta = evaluate_model(sta, test_loader, device='cpu', label='PyTorch Static', move_model=False)

for r in [r_fp32, r_fq, r_dyn, r_sta]:
    print_results(r)
    tag = r['label'].replace(' ', '_')
    save_results(r, f"{OUT_PREFIX}_{tag}.json")

compare_results([r_fp32, r_fq, r_dyn, r_sta])
