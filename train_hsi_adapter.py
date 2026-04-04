import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)




def get_timestep_bounds(epoch: int, epochs: int, num_timesteps: int, args):
    progress = ((epoch - 1) / max(1, epochs - 1)) ** args.t_curriculum_power
    t_min = int(args.t_min_ratio * (num_timesteps - 1))
    t_max_ratio = args.t_max_start_ratio + (args.t_max_end_ratio - args.t_max_start_ratio) * progress
    t_max = int(t_max_ratio * (num_timesteps - 1))
    t_max = max(t_min + 1, min(num_timesteps, t_max + 1))
    return t_min, t_max

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    effective_num_workers = max(0, int(args.num_workers))
    effective_pin_memory = bool(args.pin_memory and device.type == "cuda")
    if device.type != "cuda" and args.pin_memory:
        print("[WARN] pin_memory=True is only useful with CUDA. Setting pin_memory=False.")
    if effective_num_workers > 0 and sys.version_info >= (3, 12):
        print(
            f"[WARN] Python {sys.version.split()[0]} with DataLoader workers can be unstable in some envs. "
            "Falling back to num_workers=0. You can force workers via --allow_worker_fallback."
        )
        if not args.allow_worker_fallback:
            effective_num_workers = 0

    model_cfg = load_yaml(args.model_config)
    diffusion_cfg = load_yaml(args.diffusion_config)

    # Default to gradient checkpointing for memory safety in HSI adapter training.
    model_cfg["use_checkpoint"] = not args.disable_checkpoint

    base_model = create_model(**model_cfg).to(device)
    adapter_model = build_hsi_adapter_model(
        core_model=base_model,
        hsi_channels=args.hsi_channels,
        adapter_hidden_channels=args.adapter_hidden_channels,
        adapter_num_blocks=args.adapter_num_blocks,
        freeze_core=True,
        core_peft=args.core_peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_conv2d_target=args.lora_conv2d_target,
        lora_enable_conv1d=args.lora_enable_conv1d,
    ).to(device)

    dataset = HyperspectralFolderDataset(
        root=args.data_root,
        image_size=args.image_size,
        channels=args.hsi_channels,
        random_crop_size=args.random_crop_size,
        repeats_per_scene=args.repeats_per_scene,
        use_grid_patches=args.use_grid_patches,
        grid_patch_size=args.grid_patch_size,
        rotation_aug=args.rotation_aug,
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
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
    )

    sampler = create_sampler(**diffusion_cfg)

    optim = AdamW(adapter_model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and not amp_enabled:
        print("[WARN] AMP requested but CUDA is unavailable. AMP disabled.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))
    scheduler = build_scheduler(
        optim,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
        min_lr_scale=args.min_lr_scale,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_file
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "epoch", "global_step", "iter", "loss", "val_loss", "lr", "t_min", "t_max"])

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        adapter_model.train()
        train_losses = []
        optim.zero_grad(set_to_none=True)

        t_min, t_max = get_timestep_bounds(epoch, args.epochs, sampler.num_timesteps, args)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [t:{t_min}-{t_max-1}]")
        for iter_idx, x0 in enumerate(pbar):
            x0 = x0.to(device, non_blocking=True)
            b = x0.shape[0]
            t = torch.randint(t_min, t_max, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = extract(sampler.sqrt_alphas_cumprod, t, x0.shape, device) * x0 + \
                extract(sampler.sqrt_one_minus_alphas_cumprod, t, x0.shape, device) * noise

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = adapter_model(x_t, sampler._scale_timesteps(t))
                if pred.shape[1] == 2 * x0.shape[1]:
                    pred, _ = torch.chunk(pred, 2, dim=1)
                loss = F.mse_loss(pred, noise)
                loss_for_backward = loss / args.grad_accum_steps

            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = ((iter_idx + 1) % args.grad_accum_steps == 0) or ((iter_idx + 1) == len(train_loader))
            if should_step:
                if args.grad_clip > 0:
                    if amp_enabled:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(adapter_model.trainable_parameters(), args.grad_clip)

                if amp_enabled:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            train_losses.append(loss.item())
            if (iter_idx + 1) % args.log_interval == 0 or (iter_idx + 1) == len(train_loader):
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "train_step",
                        epoch,
                        global_step,
                        iter_idx + 1,
                        float(loss.item()),
                        "",
                        float(optim.param_groups[0]["lr"]),
                        t_min,
                        t_max - 1,
                    ])
            pbar.set_postfix({
                "loss": f"{sum(train_losses) / len(train_losses):.4f}",
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
            })

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
        print(f"[Epoch {epoch}] train={mean_train:.6f} val={mean_val:.6f} lr={optim.param_groups[0]['lr']:.3e}")
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                epoch,
                global_step,
                len(train_loader),
                float(mean_train),
                float(mean_val),
                float(optim.param_groups[0]["lr"]),
                t_min,
                t_max - 1,
            ])

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
    parser.add_argument("--model_config", type=str, default="configs/imagenet_model_config.yaml")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--data_root", type=str, default="./data/hsi/cave")
    parser.add_argument("--save_dir", type=str, default="./models/hsi_adapter")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--hsi_channels", type=int, default=31)
    parser.add_argument("--adapter_hidden_channels", type=int, default=256)
    parser.add_argument("--adapter_num_blocks", type=int, default=4)
    parser.add_argument("--core_peft", type=str, default="none", choices=["none", "lora"])
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_conv2d_target", type=str, default="1x1", choices=["1x1", "all"])
    parser.add_argument("--lora_enable_conv1d", action="store_true")
    parser.add_argument("--disable_checkpoint", action="store_true")

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_worker_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When false, auto-fallback num_workers to 0 on Python>=3.12 for stability.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_crop_size", type=int, default=128)
    parser.add_argument("--use_grid_patches", action="store_true")
    parser.add_argument("--grid_patch_size", type=int, default=128)
    parser.add_argument("--rotation_aug", action="store_true")
    parser.add_argument("--repeats_per_scene", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_scale", type=float, default=0.1)
    parser.add_argument("--t_min_ratio", type=float, default=0.0)
    parser.add_argument("--t_max_start_ratio", type=float, default=1.0)
    parser.add_argument("--t_max_end_ratio", type=float, default=1.0)
    parser.add_argument("--t_curriculum_power", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log_file", type=str, default="train_log.csv")
    parser.add_argument("--log_interval", type=int, default=20)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
