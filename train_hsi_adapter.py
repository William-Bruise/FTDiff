import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import yaml

from data.hsi_dataset import HyperspectralFolderDataset
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from models_hsi_adapter import build_hsi_adapter_model


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def extract(arr, t, shape, device):
    vals = torch.as_tensor(arr, dtype=torch.float32, device=device)[t]
    while len(vals.shape) < len(shape):
        vals = vals[..., None]
    return vals


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model_cfg = load_yaml(args.model_config)
    diffusion_cfg = load_yaml(args.diffusion_config)

    base_model = create_model(**model_cfg).to(device)
    adapter_model = build_hsi_adapter_model(
        core_model=base_model,
        hsi_channels=args.hsi_channels,
        adapter_hidden_channels=args.adapter_hidden_channels,
        freeze_core=True,
    ).to(device)

    dataset = HyperspectralFolderDataset(
        root=args.data_root,
        image_size=args.image_size,
        channels=args.hsi_channels,
    )

    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 HSI samples for train/val split.")

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sampler = create_sampler(**diffusion_cfg)

    optim = AdamW(adapter_model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        adapter_model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x0 in pbar:
            x0 = x0.to(device, non_blocking=True)
            b = x0.shape[0]
            t = torch.randint(0, sampler.num_timesteps, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = extract(sampler.sqrt_alphas_cumprod, t, x0.shape, device) * x0 + \
                extract(sampler.sqrt_one_minus_alphas_cumprod, t, x0.shape, device) * noise

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = adapter_model(x_t, sampler._scale_timesteps(t))
                if pred.shape[1] == 2 * x0.shape[1]:
                    pred, _ = torch.chunk(pred, 2, dim=1)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(adapter_model.trainable_parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            global_step += 1
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{sum(train_losses) / len(train_losses):.4f}"})

        # validation
        adapter_model.eval()
        val_losses = []
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                b = x0.shape[0]
                t = torch.randint(0, sampler.num_timesteps, (b,), device=device)
                noise = torch.randn_like(x0)
                x_t = extract(sampler.sqrt_alphas_cumprod, t, x0.shape, device) * x0 + \
                    extract(sampler.sqrt_one_minus_alphas_cumprod, t, x0.shape, device) * noise
                pred = adapter_model(x_t, sampler._scale_timesteps(t))
                if pred.shape[1] == 2 * x0.shape[1]:
                    pred, _ = torch.chunk(pred, 2, dim=1)
                val_losses.append(F.mse_loss(pred, noise).item())

        mean_train = sum(train_losses) / max(1, len(train_losses))
        mean_val = sum(val_losses) / max(1, len(val_losses))
        print(f"[Epoch {epoch}] train={mean_train:.6f} val={mean_val:.6f}")

        last_ckpt = out_dir / "hsi_adapter_last.pt"
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "adapter_state_dict": adapter_model.state_dict(),
            "train_loss": mean_train,
            "val_loss": mean_val,
            "args": vars(args),
        }, last_ckpt)

        if mean_val < best_val:
            best_val = mean_val
            best_ckpt = out_dir / "hsi_adapter_best.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "adapter_state_dict": adapter_model.state_dict(),
                "train_loss": mean_train,
                "val_loss": mean_val,
                "args": vars(args),
            }, best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--data_root", type=str, default="./data/hsi/cave")
    parser.add_argument("--save_dir", type=str, default="./models/hsi_adapter")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--hsi_channels", type=int, default=31)
    parser.add_argument("--adapter_hidden_channels", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
