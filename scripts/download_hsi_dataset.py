import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def _download(url: str, out_file: Path):
    print(f"[Download] {url} -> {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(out_file))


def _extract(zip_path: Path, dst: Path):
    print(f"[Extract] {zip_path} -> {dst}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/hsi/cave")
    parser.add_argument("--cache_dir", type=str, default="./data/hsi/.cache")
    parser.add_argument(
        "--source_urls",
        type=str,
        nargs="+",
        default=[
            "https://www.cs.columbia.edu/CAVE/databases/multispectral/complete_ms_data.zip",
            "https://huggingface.co/datasets/pinecone/cave-multispectral/resolve/main/complete_ms_data.zip",
        ],
        help="Candidate URLs for CAVE multispectral dataset zip."
    )
    parser.add_argument("--clean_cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cache_dir = Path(args.cache_dir)
    cache_zip = cache_dir / "cave_ms.zip"

    success = False
    last_err = None
    for url in args.source_urls:
        try:
            _download(url, cache_zip)
            _extract(cache_zip, out_dir)
            success = True
            break
        except Exception as e:
            print(f"[WARN] failed from source: {url}\n       {e}")
            last_err = e

    if not success:
        raise RuntimeError(
            "Failed to download CAVE dataset from all sources. "
            "Please provide a working URL via --source_urls. "
            f"Last error: {last_err}"
        )

    if args.clean_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print("[OK] HSI dataset downloaded.")


if __name__ == "__main__":
    main()
