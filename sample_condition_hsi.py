from functools import partial
import os
import argparse
import yaml

import torch
import matplotlib.pyplot as plt

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

    data_config = task_config['data']
    dataset = HyperspectralFolderDataset(
        root=data_config['root'],
        image_size=data_config.get('image_size', 256),
        channels=args.hsi_channels,
    )

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


if __name__ == '__main__':
    main()
