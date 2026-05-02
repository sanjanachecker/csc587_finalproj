"""EuroSAT RGB dataset with split CSV-based indexing."""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# ImageNet stats (matches EfficientNet-B0 pretrained encoder)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class EuroSATRGB(Dataset):
    def __init__(self, split_csv, data_root="data/raw", transform=None):
        self.data_root = Path(data_root)
        self.samples = []
        with open(split_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(processed_dir="data/processed", batch_size=64, num_workers=2):
    train_ds = EuroSATRGB(f"{processed_dir}/train.csv", transform=build_transforms(True))
    val_ds   = EuroSATRGB(f"{processed_dir}/val.csv",   transform=build_transforms(False))
    test_ds  = EuroSATRGB(f"{processed_dir}/test.csv",  transform=build_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader