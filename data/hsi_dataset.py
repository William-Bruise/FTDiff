from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _to_tensor_hsi(cube: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
    # cube: H, W, C
    cube = cube.astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube(H,W,C), got shape={cube.shape}")

    cmin, cmax = cube.min(), cube.max()
    if cmax > cmin:
        cube = (cube - cmin) / (cmax - cmin)
    cube = cube * 2.0 - 1.0

    tensor = torch.from_numpy(cube).permute(2, 0, 1).contiguous()  # C,H,W
    if target_size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def _load_mat_cube(path: Path) -> Optional[np.ndarray]:
    mat = sio.loadmat(path)
    for _, value in mat.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value
    return None


def _try_stack_png_bands(scene_dir: Path) -> Optional[np.ndarray]:
    pngs = sorted(scene_dir.glob("*.png"))
    if len(pngs) < 8:
        return None

    bands = []
    for p in pngs:
        img = Image.open(p).convert("L")
        bands.append(np.array(img, dtype=np.float32) / 255.0)
    cube = np.stack(bands, axis=-1)
    return cube


class HyperspectralFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, channels: Optional[int] = None):
        self.root = Path(root)
        self.target_size = (image_size, image_size)
        self.channels = channels

        self.npy_files = sorted(self.root.glob("**/*.npy"))
        self.mat_files = sorted(self.root.glob("**/*.mat"))

        self.scene_dirs = []
        if not self.npy_files and not self.mat_files:
            # fallback: scene folder with per-band png files
            for d in sorted(self.root.glob("**/*")):
                if d.is_dir() and any(d.glob("*.png")):
                    self.scene_dirs.append(d)

        self.length = len(self.npy_files) + len(self.mat_files) + len(self.scene_dirs)
        if self.length == 0:
            raise RuntimeError(
                f"No hyperspectral samples found in {self.root}. "
                "Expected .npy cubes, .mat cubes, or scene folders with per-band .png files."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < len(self.npy_files):
            cube = np.load(self.npy_files[idx])
        elif idx < len(self.npy_files) + len(self.mat_files):
            mat_idx = idx - len(self.npy_files)
            cube = _load_mat_cube(self.mat_files[mat_idx])
            if cube is None:
                raise RuntimeError(f"No 3D cube found in {self.mat_files[mat_idx]}")
        else:
            dir_idx = idx - len(self.npy_files) - len(self.mat_files)
            cube = _try_stack_png_bands(self.scene_dirs[dir_idx])
            if cube is None:
                raise RuntimeError(f"Could not parse png-band scene in {self.scene_dirs[dir_idx]}")

        if cube.shape[0] < cube.shape[-1] and cube.shape[1] < cube.shape[-1]:
            # probably C,H,W -> H,W,C
            cube = np.transpose(cube, (1, 2, 0))

        if self.channels is not None and cube.shape[-1] != self.channels:
            if cube.shape[-1] > self.channels:
                cube = cube[..., : self.channels]
            else:
                pad = np.repeat(cube[..., -1:], self.channels - cube.shape[-1], axis=-1)
                cube = np.concatenate([cube, pad], axis=-1)

        return _to_tensor_hsi(cube, target_size=self.target_size)
