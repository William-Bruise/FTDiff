from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
try:
    import h5py
except Exception:
    h5py = None


def _normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32)
    # per-band normalization to [0,1], then to [-1,1]
    flat = cube.reshape(-1, cube.shape[-1])
    cmin = flat.min(axis=0, keepdims=True)
    cmax = flat.max(axis=0, keepdims=True)
    denom = np.maximum(cmax - cmin, 1e-6)
    flat = (flat - cmin) / denom
    cube = flat.reshape(cube.shape)
    return cube * 2.0 - 1.0


def _to_tensor_hsi(cube: np.ndarray, target_size: Optional[Tuple[int, int]]) -> torch.Tensor:
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
    try:
        mat = sio.loadmat(path)
        for _, value in mat.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return value
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError(
                f"{path} looks like MATLAB v7.3 (HDF5), but h5py is unavailable. Please install h5py."
            )
        with h5py.File(path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 3:
                    # HDF5 mat often stores Fortran-order dims, map to HWC.
                    if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
                        arr = np.transpose(arr, (1, 2, 0))
                    elif arr.shape[0] > 64 and arr.shape[1] > 64 and arr.shape[2] <= 64:
                        pass
                    else:
                        arr = np.transpose(arr, (2, 1, 0))
                    return arr
    return None


def _try_stack_png_bands(scene_dir: Path) -> Optional[np.ndarray]:
    pngs = sorted(scene_dir.glob("*.png"))
    if len(pngs) < 8:
        return None

    bands = []
    for p in pngs:
        img = Image.open(p).convert("L")
        bands.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(bands, axis=-1)


def _ensure_hwc(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")
    if cube.shape[0] <= 64 and cube.shape[1] > 64 and cube.shape[2] > 64:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def _random_crop(cube: np.ndarray, crop_size: int) -> np.ndarray:
    h, w, _ = cube.shape
    if crop_size <= 0 or (h <= crop_size and w <= crop_size):
        return cube
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    return cube[top : top + ch, left : left + cw, :]


class HyperspectralFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        channels: Optional[int] = None,
        random_crop_size: int = 256,
        repeats_per_scene: int = 1,
        use_grid_patches: bool = False,
        grid_patch_size: int = 128,
        rotation_aug: bool = False,
    ):
        self.root = Path(root)
        self.target_size = (image_size, image_size) if image_size > 0 else None
        self.channels = channels
        self.random_crop_size = random_crop_size
        self.repeats_per_scene = max(1, int(repeats_per_scene))
        self.use_grid_patches = use_grid_patches
        self.grid_patch_size = int(grid_patch_size)
        self.rotation_aug = rotation_aug

        npy_files = sorted(self.root.glob("**/*.npy"))
        mat_files = sorted(self.root.glob("**/*.mat"))

        scene_refs = [("npy", p) for p in npy_files] + [("mat", p) for p in mat_files]

        if not scene_refs:
            for d in sorted(self.root.glob("**/*")):
                if d.is_dir() and any(d.glob("*.png")):
                    scene_refs.append(("pngdir", d))

        if not scene_refs:
            raise RuntimeError(
                f"No hyperspectral samples found in {self.root}. "
                "Expected .npy cubes, .mat cubes, or scene folders with per-band .png files."
            )

        self.scene_refs = scene_refs
        self.sample_map = self._build_sample_map()

    def _build_sample_map(self):
        sample_map = []
        rot_list = [0, 1, 2, 3] if self.rotation_aug else [0]

        if not self.use_grid_patches:
            for scene_idx in range(len(self.scene_refs)):
                for _ in range(self.repeats_per_scene):
                    for rot_k in rot_list:
                        sample_map.append((scene_idx, None, None, rot_k))
            return sample_map

        for scene_idx in range(len(self.scene_refs)):
            cube = self._load_scene_cube(scene_idx)
            h, w, _ = cube.shape
            ps = self.grid_patch_size
            nh = max(1, h // ps)
            nw = max(1, w // ps)
            for iy in range(nh):
                for ix in range(nw):
                    top = min(iy * ps, max(0, h - ps))
                    left = min(ix * ps, max(0, w - ps))
                    for _ in range(self.repeats_per_scene):
                        for rot_k in rot_list:
                            sample_map.append((scene_idx, top, left, rot_k))
        return sample_map

    def __len__(self) -> int:
        return len(self.sample_map)

    def _load_scene_cube(self, scene_idx: int) -> np.ndarray:
        kind, path = self.scene_refs[scene_idx]
        if kind == "npy":
            cube = np.load(path)
        elif kind == "mat":
            cube = _load_mat_cube(path)
            if cube is None:
                raise RuntimeError(f"No 3D cube found in {path}")
        else:
            cube = _try_stack_png_bands(path)
            if cube is None:
                raise RuntimeError(f"Could not parse png-band scene in {path}")

        cube = _ensure_hwc(cube)

        if self.channels is not None and cube.shape[-1] != self.channels:
            if cube.shape[-1] > self.channels:
                cube = cube[..., : self.channels]
            else:
                pad = np.repeat(cube[..., -1:], self.channels - cube.shape[-1], axis=-1)
                cube = np.concatenate([cube, pad], axis=-1)

        return cube

    def __getitem__(self, idx: int) -> torch.Tensor:
        scene_idx, top, left, rot_k = self.sample_map[idx]
        cube = self._load_scene_cube(scene_idx)

        if top is not None and left is not None:
            ps = self.grid_patch_size
            cube = cube[top : top + ps, left : left + ps, :]
        else:
            cube = _random_crop(cube, self.random_crop_size)

        if rot_k > 0:
            cube = np.rot90(cube, k=rot_k, axes=(0, 1)).copy()

        cube = _normalize_cube(cube)
        return _to_tensor_hsi(cube, target_size=self.target_size)


def load_hsi_cube_from_path(path: str, channels: Optional[int] = None) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HSI sample path does not exist: {p}")

    if p.is_file():
        if p.suffix.lower() == ".npy":
            cube = np.load(p)
        elif p.suffix.lower() == ".mat":
            cube = _load_mat_cube(p)
            if cube is None:
                raise RuntimeError(f"No 3D cube found in {p}")
        else:
            raise ValueError(f"Unsupported file type for single HSI sample: {p.suffix}")
    elif p.is_dir():
        cube = _try_stack_png_bands(p)
        if cube is None:
            raise RuntimeError(f"Could not parse png-band scene in {p}")
    else:
        raise ValueError(f"Unsupported path type for single HSI sample: {p}")

    cube = _ensure_hwc(cube)
    if channels is not None and cube.shape[-1] != channels:
        if cube.shape[-1] > channels:
            cube = cube[..., :channels]
        else:
            pad = np.repeat(cube[..., -1:], channels - cube.shape[-1], axis=-1)
            cube = np.concatenate([cube, pad], axis=-1)
    return cube


class SingleHSIOverfitDataset(Dataset):
    """Overfit dataset that repeats one HSI scene for memorization sanity checks."""

    def __init__(
        self,
        sample_path: str,
        image_size: int = 256,
        channels: Optional[int] = None,
        repeats: int = 1024,
    ):
        self.target_size = (image_size, image_size) if image_size > 0 else None
        self.repeats = max(1, int(repeats))
        cube = load_hsi_cube_from_path(sample_path, channels=channels)
        cube = _normalize_cube(cube)
        self.tensor = _to_tensor_hsi(cube, target_size=self.target_size)

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor.clone()
