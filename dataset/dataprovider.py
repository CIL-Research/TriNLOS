import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_data import Renderdataset

warnings.filterwarnings("ignore")


def collect_triplets(root: str):
    triplets = []
    for dirpath, _, files in os.walk(root):
        video_file = next((f for f in files if f.startswith("video")), None)
        if video_file is None:
            continue
        img_path = next(
            (os.path.join(dirpath, f) for f in files if f.startswith("conf") and f.endswith(".hdr")),
            None,
        )
        dep_path = next(
            (os.path.join(dirpath, f) for f in files if f.startswith("dep") and f.endswith(".hdr")),
            None,
        )
        triplets.append([os.path.join(dirpath, video_file), img_path, dep_path])
    return triplets


def save_triplet_csv(triplets, csv_path: str):
    if not triplets:
        print(f"[SKIP] {csv_path}: no data")
        return
    nlos_files, img_files, dep_files = zip(*triplets)
    df = pd.DataFrame({
        "nlos_path": nlos_files,
        "img_path": img_files,
        "dep_path": dep_files,
    })
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] {csv_path}: {len(df)} rows saved")


def main():
    parser = argparse.ArgumentParser(description="Build dataset .pth files from raw data")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory containing train/, seen/, unseen/ subdirectories")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Directory to write CSV and .pth files")
    parser.add_argument("--down-value", type=int, default=2,
                        help="Spatial downsampling factor for image and depth targets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = {"train": True, "seen": False, "unseen": False}
    for split, is_train in splits.items():
        split_root = os.path.join(args.data_root, split)
        csv_path = os.path.join(args.output_dir, f"{split}.csv")
        pth_path = os.path.join(args.output_dir, f"{split}.pth")

        triplets = collect_triplets(split_root)
        save_triplet_csv(triplets, csv_path)

        if not triplets:
            continue

        df = pd.read_csv(csv_path)
        dataset = Renderdataset(df, args.down_value, train=is_train)
        torch.save(dataset, pth_path)
        print(f"[OK] {pth_path}: {len(dataset)} samples saved")


if __name__ == "__main__":
    main()
