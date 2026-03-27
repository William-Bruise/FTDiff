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

ICVL_ZIP_SOURCES = [
    "https://huggingface.co/datasets/anonymous-icvl/hsi-icvl/resolve/main/icvl_hsi_train.zip",
    "https://huggingface.co/datasets/anonymous-icvl/hsi-icvl/resolve/main/icvl_hsi_val.zip",
]

CAVE_ZIP_SOURCES = [
    "https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip",
    "https://huggingface.co/datasets/pinecone/cave-multispectral/resolve/main/complete_ms_data.zip",
]


def _download(url: str, out_file: Path):
    print(f"[Download] {url} -> {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(out_file))


def _extract(zip_path: Path, dst: Path):
    print(f"[Extract] {zip_path} -> {dst}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


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
            "Please pass a reachable URL with --source_urls.\n"
            f"Errors:\n{msg}"
        )

    print(f"[OK] {dataset_name} downloaded from {succeeded}/{len(source_urls)} sources.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="icvl", choices=["icvl", "cave", "ehu"])
    parser.add_argument("--output", type=str, default="./data/hsi/icvl")
    parser.add_argument("--cache_dir", type=str, default="./data/hsi/.cache")
    parser.add_argument(
        "--source_urls",
        type=str,
        nargs="+",
        default=None,
        help="Custom dataset URLs. If omitted, built-in mirrors are used.",
    )
    parser.add_argument("--clean_cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ehu":
        _download_ehu(out_dir, cache_dir)
    elif args.dataset == "cave":
        urls = args.source_urls if args.source_urls else CAVE_ZIP_SOURCES
        _download_zip_dataset(out_dir, cache_dir, urls, dataset_name="cave")
    else:
        urls = args.source_urls if args.source_urls else ICVL_ZIP_SOURCES
        _download_zip_dataset(out_dir, cache_dir, urls, dataset_name="icvl")

    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"[OK] HSI dataset downloaded: {args.dataset} -> {out_dir}")


if __name__ == "__main__":
    main()
