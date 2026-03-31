import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
import scipy.io as sio
try:
    import h5py
except Exception:
    h5py = None

def _ensure_hwc(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")
    if cube.shape[0] <= 64 and cube.shape[1] > 64 and cube.shape[2] > 64:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def _load_mat_cube(path: Path):
    try:
        mat = sio.loadmat(path)
        for _, value in mat.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return value
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError(f"{path} requires h5py for MATLAB v7.3 files.")
        with h5py.File(path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 3:
                    if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
                        arr = np.transpose(arr, (1, 2, 0))
                    elif arr.shape[0] > 64 and arr.shape[1] > 64 and arr.shape[2] <= 64:
                        pass
                    else:
                        arr = np.transpose(arr, (2, 1, 0))
                    return arr
    return None


def resize_cube(cube_hwc: np.ndarray, size: int) -> np.ndarray:
    h, w, _ = cube_hwc.shape
    zh = float(size) / float(max(1, h))
    zw = float(size) / float(max(1, w))
    # zoom over H/W only, keep spectral channels unchanged.
    return zoom(cube_hwc.astype(np.float32), zoom=(zh, zw, 1.0), order=1)


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
