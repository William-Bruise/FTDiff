import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.hsi_dataset import _ensure_hwc, _load_mat_cube


def resize_cube(cube_hwc: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(cube_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def normalize_per_band_01(cube_hwc: np.ndarray) -> np.ndarray:
    flat = cube_hwc.reshape(-1, cube_hwc.shape[-1])
    cmin = flat.min(axis=0, keepdims=True)
    cmax = flat.max(axis=0, keepdims=True)
    denom = np.maximum(cmax - cmin, 1e-6)
    flat = (flat - cmin) / denom
    return flat.reshape(cube_hwc.shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./data/samples/icvl")
    parser.add_argument("--size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mats = sorted(in_root.rglob("*.mat"))
    if len(mats) == 0:
        raise RuntimeError(f"No .mat files found under {in_root}")

    kept = 0
    for idx, p in enumerate(mats):
        if args.max_samples > 0 and kept >= args.max_samples:
            break
        cube = _load_mat_cube(p)
        if cube is None:
            continue
        cube = _ensure_hwc(cube).astype(np.float32)
        cube = resize_cube(cube, args.size)
        cube = normalize_per_band_01(cube)
        np.save(out_root / f"{kept:05d}.npy", cube)
        kept += 1
        if (kept % 20) == 0:
            print(f"[Prepare] processed {kept} samples...")

    if kept == 0:
        raise RuntimeError("No valid HSI cubes were processed from input_root.")
    print(f"[OK] prepared {kept} samples at {out_root} (size={args.size}, per-band normalized to [0,1]).")


if __name__ == "__main__":
    main()
