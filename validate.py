import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data import Renderdataset
from model import Pipeline


def seed_everything(seed: int = 7777) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_spapooling(tensor: torch.Tensor, down_factor: int = 2) -> torch.Tensor:
    return F.avg_pool3d(
        tensor,
        kernel_size=(1, down_factor, down_factor),
        stride=(1, down_factor, down_factor),
    )


def rmse_t(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def mad_t(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def laplacian_sharpness(batch_b1hw: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=batch_b1hw.device,
        dtype=batch_b1hw.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(batch_b1hw, kernel, padding=1)
    return lap.flatten(1).var(dim=1, unbiased=False)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if "cached_psf_fft" in k:
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_model_state(model_state_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(model_state_path, map_location="cpu")

    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            state_dict = obj["model_state"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        else:
            state_dict = obj
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")

    return clean_state_dict(state_dict)


def build_model(device: torch.device) -> nn.Module:
    model = Pipeline()
    return model.to(device)


def load_torch_dataset(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 8
    return DataLoader(**loader_kwargs)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Optional[str] = None,
    progress_desc: str = "Validation",
    beta: float = 1.0,
) -> Dict[str, float]:
    criterion_img = nn.L1Loss(reduction="mean")
    criterion_dep = nn.L1Loss(reduction="mean")

    loss_total_list: List[float] = []
    loss_img_list: List[float] = []
    loss_dep_list: List[float] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    img_rmse_vals: List[float] = []
    dep_rmse_vals: List[float] = []
    dep_mad_vals: List[float] = []
    grad_diff_vals: List[float] = []
    sharp_pred_vals: List[float] = []
    sharp_gt_vals: List[float] = []
    sharp_gap_vals: List[float] = []
    sample_idx = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for nlos_ar, img, dep in tqdm(val_loader, desc=progress_desc, leave=False):
            nlos_ar = nlos_ar.to(device)
            img = img.to(device)
            dep = dep.to(device)

            pred_img, pred_dep, _ = model(average_spapooling(nlos_ar))

            loss_img = criterion_img(pred_img, img)
            loss_dep = criterion_dep(pred_dep, dep)
            loss_total = loss_img + beta * loss_dep

            loss_total_list.append(loss_total.item())
            loss_img_list.append(loss_img.item())
            loss_dep_list.append(loss_dep.item())

            device_dtype = pred_img.dtype
            sobel_x = torch.tensor(
                [[[-1.0, 0.0, 1.0],
                  [-2.0, 0.0, 2.0],
                  [-1.0, 0.0, 1.0]]],
                device=device,
                dtype=device_dtype,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[[-1.0, -2.0, -1.0],
                  [0.0, 0.0, 0.0],
                  [1.0, 2.0, 1.0]]],
                device=device,
                dtype=device_dtype,
            ).view(1, 1, 3, 3)

            gx_pred = F.conv2d(pred_img, sobel_x, padding=1)
            gy_pred = F.conv2d(pred_img, sobel_y, padding=1)
            gx_gt = F.conv2d(img, sobel_x, padding=1)
            gy_gt = F.conv2d(img, sobel_y, padding=1)

            grad_mag_pred = torch.sqrt(gx_pred * gx_pred + gy_pred * gy_pred + 1e-12)
            grad_mag_gt = torch.sqrt(gx_gt * gx_gt + gy_gt * gy_gt + 1e-12)
            grad_diff_batch = torch.mean(
                torch.abs(grad_mag_pred - grad_mag_gt),
                dim=[1, 2, 3],
            ).detach().cpu().numpy()
            grad_diff_vals.extend(grad_diff_batch.tolist())

            pred_img_c = pred_img.clamp(0.0, 1.0).detach().cpu()
            pred_dep_c = pred_dep.clamp(0.0, 1.0).detach().cpu()
            img_c = img.detach().cpu()
            dep_c = dep.detach().cpu()

            sharp_pred_batch = laplacian_sharpness(pred_img_c).numpy()
            sharp_gt_batch = laplacian_sharpness(img_c).numpy()
            sharp_gap_batch = np.abs(sharp_pred_batch - sharp_gt_batch)

            sharp_pred_vals.extend(sharp_pred_batch.tolist())
            sharp_gt_vals.extend(sharp_gt_batch.tolist())
            sharp_gap_vals.extend(sharp_gap_batch.tolist())

            batch_size = pred_img_c.shape[0]
            for i in range(batch_size):
                gt_img = img_c[i, 0].numpy()
                pr_img = pred_img_c[i, 0].numpy()

                psnr_vals.append(psnr_fn(gt_img, pr_img, data_range=1.0))
                ssim_vals.append(ssim_fn(gt_img, pr_img, data_range=1.0))
                img_rmse_vals.append(rmse_t(img_c[i, 0], pred_img_c[i, 0]))
                dep_rmse_vals.append(rmse_t(dep_c[i, 0], pred_dep_c[i, 0]))
                dep_mad_vals.append(mad_t(dep_c[i, 0], pred_dep_c[i, 0]))

                if save_dir is not None:
                    sample_idx += 1
                    prefix = f"sample_{sample_idx:06d}"
                    plt.imsave(os.path.join(save_dir, f"{prefix}_img_target.png"), gt_img, cmap="gray")
                    plt.imsave(os.path.join(save_dir, f"{prefix}_img_pred.png"), pr_img, cmap="gray")
                    plt.imsave(os.path.join(save_dir, f"{prefix}_dep_target.png"), dep_c[i, 0].numpy(), cmap="gray")
                    plt.imsave(os.path.join(save_dir, f"{prefix}_dep_pred.png"), pred_dep_c[i, 0].numpy(), cmap="gray")

    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "loss_total": safe_mean(loss_total_list),
        "loss_img": safe_mean(loss_img_list),
        "loss_dep": safe_mean(loss_dep_list),
        "psnr": safe_mean(psnr_vals),
        "ssim": safe_mean(ssim_vals),
        "img_rmse": safe_mean(img_rmse_vals),
        "dep_rmse": safe_mean(dep_rmse_vals),
        "dep_mad": safe_mean(dep_mad_vals),
        "sharpness_pred": safe_mean(sharp_pred_vals),
        "sharpness_gt": safe_mean(sharp_gt_vals),
        "sharpness_gap": safe_mean(sharp_gap_vals),
        "grad_diff": safe_mean(grad_diff_vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLOSTAR model with fixed model_state.pth")

    parser.add_argument(
        "--val-data",
        type=str,
        nargs="+",
        default=["dataset/seen.pth", "dataset/unseen.pth"],
        help="List of validation dataset pth files. Example: --val-data seen.pth unseen.pth",
    )
    parser.add_argument(
        "--val-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional names for validation datasets. If omitted, file stems are used.",
    )
    parser.add_argument(
        "--model-state-pth",
        type=str,
        default="checkpoint/model_state.pth",
        help="Path to the saved model_state.pth",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--base-root", type=str, default="data")
    parser.add_argument(
        "--output-csv-dir",
        type=str,
        default="eval_summaries",
        help="Directory to save per-dataset CSV summaries",
    )
    parser.add_argument(
        "--save-image-root",
        type=str,
        default="eval_images",
        help="Root directory to save evaluation images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    model = build_model( device=device)
    state_dict = load_model_state(args.model_state_pth)
    model.load_state_dict(state_dict, strict=False)

    os.makedirs(args.output_csv_dir, exist_ok=True)
    os.makedirs(args.save_image_root, exist_ok=True)

    if args.val_names is not None and len(args.val_names) != len(args.val_data):
        raise ValueError("--val-names length must match --val-data length")

    rows: List[Dict[str, float]] = []

    for i, val_path in enumerate(args.val_data):
        val_name = args.val_names[i] if args.val_names is not None else Path(val_path).stem

        val_dataset = load_torch_dataset(val_path)
        val_loader = make_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        variant_image_dir = os.path.join(args.save_image_root, val_name)
        metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            save_dir=variant_image_dir,
            progress_desc=f"Val {val_name}",
            beta=args.beta,
        )

        row = {
            "dataset": val_name,
            "val_data": val_path,
            "model_state_pth": args.model_state_pth,
            "image_dir": variant_image_dir,
            **metrics,
        }
        rows.append(row)

        print(
            f"[{val_name}] "
            f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"IMG_RMSE={metrics['img_rmse']:.4f} DEP_RMSE={metrics['dep_rmse']:.4f} "
            f"DEP_MAD={metrics['dep_mad']:.4f} SHARP={metrics['sharpness_pred']:.6f} "
            f"SHARP_GAP={metrics['sharpness_gap']:.6f} GRAD_DIFF={metrics['grad_diff']:.6f}"
        )

        out_csv = os.path.join(args.output_csv_dir, f"eval_summary_{val_name}.csv")
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        print(f"Saved per-dataset summary to: {out_csv}")

    if not rows:
        print("No dataset was evaluated. Check paths.")
        return

    combined_csv = os.path.join(args.output_csv_dir, "eval_summary_all.csv")
    df = pd.DataFrame(rows)
    df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined summary to: {combined_csv}")


if __name__ == "__main__":
    main()