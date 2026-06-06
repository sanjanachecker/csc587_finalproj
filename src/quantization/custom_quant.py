import sys
from pathlib import Path
_SRC = Path(__file__).parent.parent   # src/
sys.path.insert(0, str(_SRC / 'models'))
sys.path.insert(0, str(_SRC / 'quantization'))

import torch
from models import build_model
from quantize import quantize_model, print_quantization_summary
from calibrate import calibrate, print_calibration_summary
from dataset import get_dataloaders

_, val_loader, _ = get_dataloaders(
    processed_dir='data/processed',
    data_root='data/raw',
    batch_size=32,
    num_workers=0,
)

device = "mps"

fp32 = build_model('efficientnet_b0')

# unwrap checkpt
ckpt = torch.load('results/checkpoints/efficientnet_b0_fp32.pt', map_location=device)
fp32.load_state_dict(ckpt['model_state'])

fp32.eval()

qmodel = quantize_model(fp32)
print_quantization_summary(qmodel)

calibrate(qmodel, val_loader, n_images=256, device=device)
print_calibration_summary(qmodel)