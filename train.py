import argparse
import gc
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Pipeline
from load_data import average_spapooling

matplotlib.use("Agg")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(x):
    b, c, d, h, w = x.shape
    flat = x.reshape(b, c, -1)
    mn   = flat.min(2, keepdim=True)[0]
    flat = flat - mn
    mx   = flat.max(2, keepdim=True)[0].clamp(min=1e-15)
    return (flat / mx).view(b, c, d, h, w)


def spad_noise(
    tau,
    photon_range=(500.0, 5000.0),
    bg_range=(0.0, 0.03),
    gauss_range=(0.001, 0.04),
    eps=1e-8,
):
    assert tau.dim() == 5 and tau.size(1) == 1
    B, device, dtype = tau.size(0), tau.device, tau.dtype
    tau = tau.clamp(min=0.0) + eps
    tau = (tau + torch.randn_like(tau) *
           torch.empty(B, 1, 1, 1, 1, device=device, dtype=dtype).uniform_(*gauss_range)
           ).clamp(min=0.0)
    scale = torch.empty(B, 1, 1, 1, 1, device=device, dtype=dtype).uniform_(*photon_range)
    bg    = torch.empty(B, 1, 1, 1, 1, device=device, dtype=dtype).uniform_(*bg_range)
    lam   = (scale * (tau + bg)).clamp(min=eps)
    return (torch.poisson(lam) / scale).clamp(0)


def build_optimizer(model, lr, weight_decay):
    no_decay_keys = ("bias", "norm", "scale", "alpha", "beta")
    pg_decay, pg_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("makevol") or any(k in name for k in no_decay_keys):
            pg_no_decay.append(p)
        else:
            pg_decay.append(p)
    return torch.optim.AdamW([
        {"params": pg_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": pg_decay,    "lr": lr, "weight_decay": weight_decay},
    ])


def make_loader(dataset, batch_size, num_workers, shuffle):
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 16
    return DataLoader(dataset, shuffle=shuffle, drop_last=shuffle, **kwargs)


def train_epoch(model, loader, optimizer, criterion_img, criterion_dep,
                device, accum_steps, epoch, num_epochs):
    model.train()
    t_loss = t_img = t_dep = 0.0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True)

    for i, (nlos, img, dep) in enumerate(bar):
        nlos = nlos.to(device)
        img  = img.to(device)
        dep  = dep.to(device)

        nlos = normalize(average_spapooling(spad_noise(nlos)))
        pred_img, pred_dep, _ = model(nlos)

        l_img = criterion_img(pred_img, img)
        l_dep = criterion_dep(pred_dep, dep)
        loss  = (l_img + l_dep) / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        t_loss += (l_img + l_dep).item()
        t_img  += l_img.item()
        t_dep  += l_dep.item()
        bar.set_postfix(loss=f"{(l_img + l_dep).item():.4f}")

    n = len(loader)
    return t_loss / n, t_img / n, t_dep / n


def val_epoch(model, loader, criterion_img, criterion_dep,
              device, epoch, num_epochs, save_dir):
    model.eval()
    v_loss = 0.0
    psnr_vals, ssim_vals, rmse_vals = [], [], []
    os.makedirs(save_dir, exist_ok=True)
    idx = 0

    bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=True)
    with torch.no_grad():
        for nlos, img, dep in bar:
            nlos = nlos.to(device)
            img  = img.to(device)
            dep  = dep.to(device)

            pred_img, pred_dep, _ = model(average_spapooling(nlos))

            l_img = criterion_img(pred_img, img)
            l_dep = criterion_dep(pred_dep, dep)
            v_loss += (l_img + l_dep).item()

            pr = pred_img.clamp(0, 1).cpu()
            gt = img.cpu()
            pd = pred_dep.clamp(0, 1).cpu()

            for b in range(pr.shape[0]):
                idx += 1
                gt_np = gt[b, 0].numpy()
                pr_np = pr[b, 0].numpy()
                psnr_vals.append(psnr_fn(gt_np, pr_np, data_range=1.0))
                ssim_vals.append(ssim_fn(gt_np, pr_np, data_range=1.0))
                rmse_vals.append(
                    float(torch.sqrt(torch.mean((gt[b, 0] - pr[b, 0]) ** 2)))
                )
                plt.imsave(os.path.join(save_dir, f"{idx:05d}_pred.png"), pr_np, cmap="gray")
                plt.imsave(os.path.join(save_dir, f"{idx:05d}_gt.png"),   gt_np, cmap="gray")
                plt.imsave(os.path.join(save_dir, f"{idx:05d}_dep.png"),  pd[b, 0].numpy(), cmap="gray")

    return (
        v_loss / len(loader),
        {
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "rmse": float(np.mean(rmse_vals)),
        },
    )


def save_plot(values, label, path):
    plt.figure(figsize=(8, 5))
    plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    cfg = load_config(parser.parse_args().config)

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    val_names = cfg.get("val_names") or [
        os.path.splitext(os.path.basename(p))[0] for p in cfg["val_data"]
    ]

    train_dataset = torch.load(cfg["train_data"], weights_only=False)
    val_datasets  = [torch.load(p, weights_only=False) for p in cfg["val_data"]]

    train_loader = make_loader(train_dataset, cfg["batch_size"], cfg["num_workers"], shuffle=True)
    val_loaders  = [make_loader(ds, cfg["batch_size"], cfg["num_workers"], shuffle=False)
                    for ds in val_datasets]

    model     = Pipeline().to(device)
    optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_min"])

    criterion_img = nn.L1Loss()
    criterion_dep = nn.L1Loss()

    ckpt_path   = os.path.join(cfg["checkpoint_dir"], "checkpoint.pth")
    start_epoch = 0
    history     = {"train": []}
    for name in val_names:
        history[name] = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = {k: v for k, v in ckpt["model_state"].items()
                 if "cached_psf_fft" not in k}
        model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device)
        start_epoch = ckpt.get("epoch", 0) + 1
        history     = ckpt.get("history", history)
        print(f"Resumed from epoch {start_epoch - 1}  lr={scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(start_epoch, cfg["epochs"]):
        t_loss, _, _ = train_epoch(
            model, train_loader, optimizer,
            criterion_img, criterion_dep,
            device, cfg["accum_steps"], epoch, cfg["epochs"],
        )
        history["train"].append(t_loss)
        scheduler.step()

        for name, loader in zip(val_names, val_loaders):
            save_dir = os.path.join(cfg["checkpoint_dir"], f"{name}_epoch{epoch:03d}")
            v_loss, metrics = val_epoch(
                model, loader, criterion_img, criterion_dep,
                device, epoch, cfg["epochs"], save_dir,
            )
            history[name].append(v_loss)

            with open(os.path.join(cfg["checkpoint_dir"],
                                   f"metrics_{name}_epoch{epoch:03d}.txt"), "w") as f:
                f.write(f"loss: {v_loss:.6f}\n")
                f.write(f"psnr: {metrics['psnr']:.4f}\n")
                f.write(f"ssim: {metrics['ssim']:.4f}\n")
                f.write(f"rmse: {metrics['rmse']:.6f}\n")

            print(f"[{name}] epoch={epoch}  loss={v_loss:.4f}  "
                  f"psnr={metrics['psnr']:.2f}  ssim={metrics['ssim']:.4f}")

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{cfg['epochs']}  train={t_loss:.4f}  lr={lr_now:.2e}")

        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history":         history,
        }, ckpt_path)

        save_plot(history["train"], "train",
                  os.path.join(cfg["checkpoint_dir"], "loss_train.png"))
        for name in val_names:
            save_plot(history[name], name,
                      os.path.join(cfg["checkpoint_dir"], f"loss_{name}.png"))

        gc.collect()
        torch.cuda.empty_cache()

    torch.save({"model_state": model.state_dict()},
               os.path.join(cfg["checkpoint_dir"], "model_state.pth"))
    print("Training complete.")


if __name__ == "__main__":
    main()
