import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from model import Elct


def average_spapooling(tensor: torch.Tensor) -> torch.Tensor:
    return F.avg_pool3d(tensor, kernel_size=(1, 2, 2), stride=(1, 2, 2))


def load_mat(path: str) -> torch.Tensor:
    mat = scipy.io.loadmat(path)
    if "final_meas" in mat:
        array = mat["final_meas"]
        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2)  # (1, T, H, W)
    else:
        array = mat["data"]
        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.permute(0, 3, 2, 1)
    return tensor


def percentile_scale(tensor: torch.Tensor, pct: float = 99.99) -> torch.Tensor:
    arr = tensor.numpy()
    p = np.percentile(arr, pct)
    if p > 0:
        return torch.tensor(np.clip(arr / p, 0.0, 1.0), dtype=torch.float32)
    arr = arr - arr.min()
    return torch.tensor(arr / (arr.max() + 1e-8), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data",  type=str, default="data/real_data")
    parser.add_argument("--output-dir", type=str, default="output/real_data_elct")
    parser.add_argument("--time-bins",  type=int, default=512)
    parser.add_argument("--spatial",    type=int, default=128)
    parser.add_argument("--bin-len",    type=float, default=0.0096)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    elct = Elct(
        fixed_shape=(args.time_bins, args.spatial, args.spatial),
        bin_len=args.bin_len,
        device=str(device),
    ).to(device)
    elct.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(args.real_data) if f.endswith(".mat"))
    if not mat_files:
        print(f"No .mat files found in {args.real_data}")
        return

    with torch.no_grad():
        for fname in mat_files:
            stem = Path(fname).stem
            tensor = load_mat(os.path.join(args.real_data, fname))  # (1, T, H, W)
            tensor = percentile_scale(tensor)
            tensor = average_spapooling(tensor.unsqueeze(0)).to(device)  # (1, 1, T, S, S)

            vol = elct(tensor)                  # (1, 1, D, H, W)
            frontal = vol[0, 0].max(dim=0)[0].cpu().numpy()
            frontal = frontal / (frontal.max() + 1e-8)

            plt.imsave(os.path.join(args.output_dir, f"{stem}.png"), frontal, cmap="gray")
            print(f"[{stem}] done")


if __name__ == "__main__":
    main()
