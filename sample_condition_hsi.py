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
from util.img_utils import clear_color
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--hsi_channels', type=int, default=31)
    parser.add_argument('--adapter_hidden_channels', type=int, default=128)
    parser.add_argument('--adapter_num_blocks', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results_hsi')
    args = parser.parse_args()

    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

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
    ).to(device)

    ckpt = torch.load(args.adapter_ckpt, map_location='cpu')
    state = ckpt['adapter_state_dict'] if 'adapter_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
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

        vis_y = y_n
        if vis_y.shape[1] != 3:
            if vis_y.shape[1] == 1:
                vis_y = vis_y.repeat(1, 3, 1, 1)
            else:
                vis_y = to_rgb_preview(vis_y)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(vis_y))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(to_rgb_preview(ref_img)))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(to_rgb_preview(sample)))


if __name__ == '__main__':
    main()
