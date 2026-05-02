"""Generate stratified 80/10/10 train/val/test splits for EuroSAT RGB."""
import argparse
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

def main(raw_dir: Path, out_dir: Path, seed: int = 42):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for img in sorted(cls_dir.glob("*.jpg")):
            paths.append(str(img.relative_to(raw_dir.parent)))
            labels.append(idx)

    print(f"Found {len(paths)} images across {len(CLASSES)} classes")

    # 80 / 10 / 10 stratified
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=seed
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    for split_name, P, Y in [("train", p_train, y_train),
                              ("val", p_val, y_val),
                              ("test", p_test, y_test)]:
        out_csv = out_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "class_name"])
            for path, lbl in zip(P, Y):
                w.writerow([path, lbl, CLASSES[lbl]])
        print(f"  {split_name}: {len(P):>5d} -> {out_csv}")

    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/2750",
                    help="Folder containing the 10 class subfolders")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(Path(args.raw_dir), Path(args.out_dir), args.seed)