import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.unet import create_model
from models_hsi_adapter import build_hsi_adapter_model
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def to_rgb_preview(hsi_tensor: torch.Tensor) -> torch.Tensor:
    c = hsi_tensor.shape[1]
    ridx = int(0.8 * (c - 1))
    gidx = int(0.5 * (c - 1))
    bidx = int(0.2 * (c - 1))
    return torch.stack(
        [hsi_tensor[:, ridx, :, :], hsi_tensor[:, gidx, :, :], hsi_tensor[:, bidx, :, :]], dim=1
    )


def to_vis_image(tensor: torch.Tensor) -> torch.Tensor:
    vis = to_rgb_preview(tensor) if tensor.shape[1] != 3 else tensor
    return torch.clamp((vis + 1.0) * 0.5, 0.0, 1.0)


def _uncond_identity_condition(x_t, measurement, noisy_measurement, x_prev, x_0_hat):
    return x_t, torch.tensor(0.0, device=x_t.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--hsi_channels", type=int, default=31)
    parser.add_argument("--adapter_hidden_channels", type=int, default=256)
    parser.add_argument("--adapter_num_blocks", type=int, default=4)
    parser.add_argument("--core_peft", type=str, default="none", choices=["none", "lora"])
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_conv2d_target", type=str, default="1x1", choices=["1x1", "all"])
    parser.add_argument("--lora_enable_conv1d", action="store_true")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results_hsi_uncond")
    args = parser.parse_args()

    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    ckpt = torch.load(args.adapter_ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt_args, dict) and len(ckpt_args) > 0:
        args.hsi_channels = int(ckpt_args.get("hsi_channels", args.hsi_channels))
        args.adapter_hidden_channels = int(ckpt_args.get("adapter_hidden_channels", args.adapter_hidden_channels))
        args.adapter_num_blocks = int(ckpt_args.get("adapter_num_blocks", args.adapter_num_blocks))
        args.core_peft = ckpt_args.get("core_peft", args.core_peft)
        args.lora_rank = int(ckpt_args.get("lora_rank", args.lora_rank))
        args.lora_alpha = float(ckpt_args.get("lora_alpha", args.lora_alpha))
        args.lora_conv2d_target = ckpt_args.get("lora_conv2d_target", args.lora_conv2d_target)
        args.lora_enable_conv1d = bool(ckpt_args.get("lora_enable_conv1d", args.lora_enable_conv1d))

    model_cfg = load_yaml(args.model_config)
    diffusion_cfg = load_yaml(args.diffusion_config)
    base_model = create_model(**model_cfg).to(device)
    model = build_hsi_adapter_model(
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
    state = ckpt["adapter_state_dict"] if "adapter_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    sampler = create_sampler(**diffusion_cfg)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=_uncond_identity_condition)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "preview"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "npy"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "progress"), exist_ok=True)

    for i in range(args.num_samples):
        x_start = torch.randn((1, args.hsi_channels, args.image_size, args.image_size), device=device)
        measurement = torch.zeros_like(x_start)
        sample = sample_fn(
            x_start=x_start,
            measurement=measurement,
            record=True,
            save_root=args.save_dir,
        )
        out_np = sample.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        np.save(os.path.join(args.save_dir, "npy", f"{i:05d}.npy"), out_np)

        vis = to_vis_image(sample).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        plt.imsave(os.path.join(args.save_dir, "preview", f"{i:05d}.png"), vis)

    logger.info(f"Unconditional HSI samples saved to {args.save_dir}")


if __name__ == "__main__":
    main()
