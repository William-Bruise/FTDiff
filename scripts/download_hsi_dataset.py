import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


EHU_DATASETS = {
    "indian_pines": "https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
    "salinas": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "pavia_u": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "ksc": "https://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat",
}

ICVL_SHAREPOINT_SOURCES = [
    "https://bgu365.sharepoint.com/sites/ICVL/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FICVL%2FShared%20Documents%2FDatasets%2FHS&download=1",
]

CAVE_ZIP_SOURCES = [
    "https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip",
]


def _download(url: str, out_file: Path):
    print(f"[Download] {url} -> {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(out_file))


def _extract(zip_path: Path, dst: Path):
    print(f"[Extract] {zip_path} -> {dst}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def _has_hsi_files(output: Path) -> bool:
    if not output.exists():
        return False
    return any(output.rglob("*.mat")) or any(output.rglob("*.npy"))


def _download_ehu(output: Path, cache_dir: Path):
    output.mkdir(parents=True, exist_ok=True)
    for name, url in EHU_DATASETS.items():
        out_file = output / f"{name}.mat"
        if out_file.exists():
            print(f"[Skip] exists: {out_file}")
            continue
        cache_file = cache_dir / f"{name}.mat"
        _download(url, cache_file)
        shutil.copy2(cache_file, out_file)


def _download_zip_dataset(output: Path, cache_dir: Path, source_urls, dataset_name: str):
    output.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    errors = []

    for idx, url in enumerate(source_urls):
        zip_file = cache_dir / f"{dataset_name}_{idx}.zip"
        try:
            _download(url, zip_file)
            _extract(zip_file, output)
            succeeded += 1
        except Exception as e:
            print(f"[WARN] failed source: {url}\n       {e}")
            errors.append((url, str(e)))

    if succeeded == 0:
        msg = "\n".join([f"- {u}: {e}" for u, e in errors])
        raise RuntimeError(
            f"Failed to download {dataset_name}. No source succeeded.\n"
            "Please pass a reachable URL with --source_urls or use --local_zip.\n"
            f"Errors:\n{msg}"
        )

    print(f"[OK] {dataset_name} downloaded from {succeeded}/{len(source_urls)} sources.")


def _keep_only_mat(output: Path):
    for p in output.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".mat":
            p.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cave", choices=["cave", "ehu", "icvl"])
    parser.add_argument("--output", type=str, default="./data/hsi/cave")
    parser.add_argument("--cache_dir", type=str, default="./data/hsi/.cache")
    parser.add_argument(
        "--source_urls",
        type=str,
        nargs="+",
        default=None,
        help="Custom dataset URLs. Used for cave and icvl.",
    )
    parser.add_argument(
        "--local_zip",
        type=str,
        default=None,
        help="Path to a manually downloaded zip file (e.g., ICVL SharePoint zip).",
    )
    parser.add_argument("--only_mat", action="store_true", help="After extraction, keep only .mat files.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset files already exist.")
    parser.add_argument("--clean_cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if (not args.force) and _has_hsi_files(out_dir):
        print(f"[Skip] Existing dataset detected in {out_dir}. Use --force to re-download.")
        return

    if args.dataset == "ehu":
        _download_ehu(out_dir, cache_dir)

    elif args.dataset == "cave":
        urls = args.source_urls if args.source_urls else CAVE_ZIP_SOURCES
        _download_zip_dataset(out_dir, cache_dir, urls, dataset_name="cave")

    else:  # icvl
        if args.local_zip:
            _extract(Path(args.local_zip), out_dir)
        else:
            urls = args.source_urls if args.source_urls else ICVL_SHAREPOINT_SOURCES
            _download_zip_dataset(out_dir, cache_dir, urls, dataset_name="icvl")

    if args.only_mat:
        _keep_only_mat(out_dir)

    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"[OK] HSI dataset downloaded: {args.dataset} -> {out_dir}")


if __name__ == "__main__":
    main()
