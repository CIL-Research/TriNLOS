import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import Pipeline


def average_spapooling(tensor: torch.Tensor) -> torch.Tensor:
    return F.avg_pool3d(tensor, kernel_size=(1, 2, 2), stride=(1, 2, 2))


def load_mat(path: str) -> torch.Tensor:
    mat = scipy.io.loadmat(path)
    if "final_meas" in mat:
        array = mat["final_meas"]
        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2)  # (1, T, H, W)
        tensor = average_spapooling(tensor)
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
    mx = arr.max()
    return torch.tensor(arr / (mx + 1e-8), dtype=torch.float32)


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = Pipeline(bin_len=0.0096)
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, dict):
        state = obj.get("model_state", obj.get("state_dict", obj))
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
        if "cached_psf_fft" not in k
    }
    model.load_state_dict(state, strict=False)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=str, default="data/real_data")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/model_state.pth")
    parser.add_argument("--output-dir", type=str, default="output/real_data")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(args.real_data) if f.endswith(".mat"))
    if not mat_files:
        print(f"No .mat files found in {args.real_data}")
        return

    with torch.no_grad():
        for fname in mat_files:
            stem = Path(fname).stem
            tensor = load_mat(os.path.join(args.real_data, fname))
            if 'teaser' in fname:
                tensor = percentile_scale(tensor)
            tensor = tensor.unsqueeze(0).to(device)  # (1, 1, T, H, W)

            pred_img, pred_dep, _ = model(tensor)

            img_out = pred_img.clamp(0.0, 1.0)[0, 0].cpu().numpy()
            dep_out = pred_dep.clamp(0.0, 1.0)[0, 0].cpu().numpy()

            plt.imsave(os.path.join(args.output_dir, f"{stem}_img.png"), img_out, cmap="gray")
            plt.imsave(os.path.join(args.output_dir, f"{stem}_dep.png"), dep_out, cmap="gray")
            print(f"[{stem}] done")


if __name__ == "__main__":
    main()
