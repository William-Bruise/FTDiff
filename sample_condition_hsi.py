from functools import partial
import os
import argparse
import csv
import subprocess
import sys
from typing import Optional
import yaml

import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.hsi_dataset import HyperspectralFolderDataset
from models_hsi_adapter import build_hsi_adapter_model
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config




def build_hsi_inpainting_mask(ref_img: torch.Tensor, mask_opt: dict) -> torch.Tensor:
    """Generate 1-channel spatial mask and broadcast along HSI channels."""
    mask_type = mask_opt.get("mask_type", "random")
    if mask_type != "random":
        raise NotImplementedError("Only random mask_type is supported for HSI in this script.")

    pmin, pmax = mask_opt.get("mask_prob_range", (0.3, 0.7))
    prob = float(torch.empty(1).uniform_(float(pmin), float(pmax)).item())

    b, c, h, w = ref_img.shape
    spatial_mask = (torch.rand((b, 1, h, w), device=ref_img.device) < prob).float()
    return spatial_mask

def to_rgb_preview(hsi_tensor: torch.Tensor) -> torch.Tensor:
    """Simple band selection for visualization (R,G,B ~= high, mid, low wavelength bands)."""
    c = hsi_tensor.shape[1]
    ridx = int(0.8 * (c - 1))
    gidx = int(0.5 * (c - 1))
    bidx = int(0.2 * (c - 1))
    rgb = torch.stack([
        hsi_tensor[:, ridx, :, :],
        hsi_tensor[:, gidx, :, :],
        hsi_tensor[:, bidx, :, :],
    ], dim=1)
    return rgb


def to_vis_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert model-space tensor to visualization image in [0,1] without per-image min-max stretching.
    Assumes training/data range is approximately [-1, 1].
    """
    if tensor.shape[1] == 1:
        vis = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] == 3:
        vis = tensor
    else:
        vis = to_rgb_preview(tensor)
    return torch.clamp((vis + 1.0) * 0.5, 0.0, 1.0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_u = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0)
    tgt_u = torch.clamp((target + 1.0) * 0.5, 0.0, 1.0)
    mse = torch.mean((pred_u - tgt_u) ** 2).item()
    return 10.0 * np.log10(1.0 / max(mse, eps))


def compute_ssim_global(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Global SSIM approximation (channel-wise, no local window), averaged across channels.
    This avoids extra dependencies while providing a stable structural metric.
    """
    pred_u = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0).squeeze(0)
    tgt_u = torch.clamp((target + 1.0) * 0.5, 0.0, 1.0).squeeze(0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    vals = []
    for c in range(pred_u.shape[0]):
        x = pred_u[c]
        y = tgt_u[c]
        mux = x.mean()
        muy = y.mean()
        sigx = ((x - mux) ** 2).mean()
        sigy = ((y - muy) ** 2).mean()
        sigxy = ((x - mux) * (y - muy)).mean()
        num = (2 * mux * muy + C1) * (2 * sigxy + C2)
        den = (mux ** 2 + muy ** 2 + C1) * (sigx + sigy + C2)
        vals.append((num / (den + eps)).item())
    return float(np.mean(vals))


def ensure_icvl_dataset(root: str, local_zip: str = None, fallback_dataset: Optional[str] = "ehu"):
    root_dir = os.path.abspath(root)
    has_hsi = False
    if os.path.isdir(root_dir):
        for base, _, files in os.walk(root_dir):
            if any(f.endswith(".mat") or f.endswith(".npy") for f in files):
                has_hsi = True
                break
    if has_hsi:
        return

    cmd = [
        sys.executable,
        "scripts/download_hsi_dataset.py",
        "--dataset", "icvl",
        "--output", root_dir,
        "--only_mat",
    ]
    if local_zip:
        cmd.extend(["--local_zip", local_zip])
    print(f"[AutoDownload] ICVL dataset not found at {root_dir}, downloading...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if fallback_dataset not in ("ehu", "cave"):
            raise RuntimeError(
                "ICVL auto-download failed and fallback dataset is disabled. "
                "Use --icvl_local_zip <zip_path> or set --download_fallback_dataset ehu/cave."
            ) from e
        print(
            f"[AutoDownload][WARN] ICVL download failed ({e}). "
            f"Falling back to --dataset {fallback_dataset} at {root_dir}."
        )
        fallback_cmd = [
            sys.executable,
            "scripts/download_hsi_dataset.py",
            "--dataset", fallback_dataset,
            "--output", root_dir,
        ]
        if fallback_dataset == "ehu":
            fallback_cmd.append("--only_mat")
        subprocess.run(fallback_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--hsi_channels', type=int, default=31)
    parser.add_argument('--adapter_hidden_channels', type=int, default=256)
    parser.add_argument('--adapter_num_blocks', type=int, default=4)
    parser.add_argument('--core_peft', type=str, default='none', choices=['none', 'lora'])
    parser.add_argument('--lora_rank', type=int, default=1)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_conv2d_target', type=str, default='1x1', choices=['1x1', 'all'])
    parser.add_argument('--lora_enable_conv1d', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results_hsi')
    parser.add_argument('--use_ckpt_model_args', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_root_override', type=str, default=None)
    parser.add_argument('--auto_download_icvl', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--icvl_local_zip', type=str, default=None)
    parser.add_argument('--download_fallback_dataset', type=str, default='ehu', choices=['ehu', 'cave', 'none'])
    args = parser.parse_args()

    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    ckpt = torch.load(args.adapter_ckpt, map_location='cpu')
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if args.use_ckpt_model_args and isinstance(ckpt_args, dict) and len(ckpt_args) > 0:
        # Keep architecture aligned with training-time checkpoint args.
        args.hsi_channels = int(ckpt_args.get("hsi_channels", args.hsi_channels))
        args.adapter_hidden_channels = int(ckpt_args.get("adapter_hidden_channels", args.adapter_hidden_channels))
        args.adapter_num_blocks = int(ckpt_args.get("adapter_num_blocks", args.adapter_num_blocks))
        args.core_peft = ckpt_args.get("core_peft", args.core_peft)
        args.lora_rank = int(ckpt_args.get("lora_rank", args.lora_rank))
        args.lora_alpha = float(ckpt_args.get("lora_alpha", args.lora_alpha))
        args.lora_conv2d_target = ckpt_args.get("lora_conv2d_target", args.lora_conv2d_target)
        args.lora_enable_conv1d = bool(ckpt_args.get("lora_enable_conv1d", args.lora_enable_conv1d))
        logger.info(
            "Loaded model architecture args from checkpoint: "
            f"hsi_channels={args.hsi_channels}, hidden={args.adapter_hidden_channels}, "
            f"blocks={args.adapter_num_blocks}, core_peft={args.core_peft}."
        )

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    if args.data_root_override:
        task_config["data"]["root"] = args.data_root_override
    if args.auto_download_icvl:
        fb = None if args.download_fallback_dataset == "none" else args.download_fallback_dataset
        ensure_icvl_dataset(task_config["data"]["root"], local_zip=args.icvl_local_zip, fallback_dataset=fb)

    base_model = create_model(**model_config).to(device)
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

    state = ckpt['adapter_state_dict'] if 'adapter_state_dict' in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load adapter checkpoint strictly. "
            "This is usually caused by architecture mismatch between training and sampling "
            "(e.g., adapter_num_blocks/hidden/core_peft differs). "
            "Try enabling --use_ckpt_model_args (default on) or matching sampling args to checkpoint args.\n"
            f"Original error: {e}"
        )
    model.eval()

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    metrics_csv = os.path.join(out_path, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr", "ssim_global"])

    data_config = task_config['data']
    dataset = HyperspectralFolderDataset(
        root=data_config['root'],
        image_size=data_config.get('image_size', 256),
        channels=args.hsi_channels,
    )

    psnr_list = []
    ssim_list = []
    for i in range(len(dataset)):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = dataset[i].unsqueeze(0).to(device)

        if measure_config['operator']['name'] == 'inpainting':
            mask = build_hsi_inpainting_mask(ref_img, measure_config['mask_opt'])
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
        psnr_val = compute_psnr(sample, ref_img)
        ssim_val = compute_ssim_global(sample, ref_img)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, psnr_val, ssim_val])

        vis_y = to_vis_image(y_n)
        vis_ref = to_vis_image(ref_img)
        vis_rec = to_vis_image(sample)

        plt.imsave(
            os.path.join(out_path, 'input', fname),
            vis_y.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        )
        plt.imsave(
            os.path.join(out_path, 'label', fname),
            vis_ref.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        )
        plt.imsave(
            os.path.join(out_path, 'recon', fname),
            vis_rec.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        )

    if len(psnr_list) > 0:
        logger.info(
            f"[Metrics] mean PSNR={float(np.mean(psnr_list)):.4f} dB, "
            f"mean SSIM(global)={float(np.mean(ssim_list)):.4f}. "
            f"Per-image metrics saved to {metrics_csv}"
        )


if __name__ == '__main__':
    main()
