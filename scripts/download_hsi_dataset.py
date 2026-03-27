import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


EHU_DATASETS = {
    "indian_pines": "https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
    "salinas": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "pavia_u": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "ksc": "https://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat",
}

ICVL_HF_REPO_ID = "danaroth/icvl"
ICVL_HF_REPO_URL = "https://huggingface.co/datasets/danaroth/icvl"

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


def _download_icvl_from_hf(output: Path, cache_dir: Path):
    output.mkdir(parents=True, exist_ok=True)

    # Preferred: huggingface_hub snapshot API (handles LFS correctly).
    try:
        from huggingface_hub import snapshot_download

        print(f"[HF] snapshot_download dataset repo: {ICVL_HF_REPO_ID}")
        snapshot_download(
            repo_id=ICVL_HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(output),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return
    except Exception as e:
        print(f"[WARN] huggingface_hub snapshot failed: {e}")

    # Fallback: git clone. Requires git-lfs for real large files.
    repo_dir = cache_dir / "icvl_repo"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", ICVL_HF_REPO_URL, str(repo_dir)], check=True)

    # Pull LFS files if possible.
    try:
        subprocess.run(["git", "-C", str(repo_dir), "lfs", "pull"], check=True)
    except Exception as e:
        print(f"[WARN] git lfs pull failed: {e}")

    for item in repo_dir.iterdir():
        if item.name == ".git":
            continue
        target = output / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)



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
        help="Custom dataset URLs. Used for cave; for icvl this is optional zip fallback.",
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
        if args.source_urls:
            _download_zip_dataset(out_dir, cache_dir, args.source_urls, dataset_name="icvl")
        else:
            _download_icvl_from_hf(out_dir, cache_dir)

    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"[OK] HSI dataset downloaded: {args.dataset} -> {out_dir}")


if __name__ == "__main__":
    main()
