"""
Cross-Platform Python Model Asset Downloader for Abhedya SoC / Trinetra AI
Downloads:
1. tokenizer.json (~9 MB)
2. xnnpack/config.json (~1 KB)
3. Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte (~750 MB - 1 GB)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import urllib.request

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HF_REPO = "experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch"
BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

FILES = [
    ("tokenizer.json", "tokenizer.json"),
    ("xnnpack/config.json", "xnnpack/config.json"),
    ("xnnpack/Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte", "xnnpack/Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte"),
]


def download_with_progress(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        sz_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Already exists: {dest_path.name} ({sz_mb:.2f} MB)")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"  [>] Downloading {dest_path.name} from {url}...")
    start_time = time.time()

    def report_progress(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            speed = mb_down / max(0.1, time.time() - start_time)
            sys.stdout.write(f"\r      {percent:5.1f}% [{mb_down:6.1f} / {mb_tot:6.1f} MB] ({speed:4.1f} MB/s)")
            sys.stdout.flush()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Abhedya-SoC Downloader)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_dest, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", -1))
            block_size = 128 * 1024
            count = 0
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                count += 1
                report_progress(count, block_size, total_size)
        sys.stdout.write("\n")
        temp_dest.replace(dest_path)
        sz_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Downloaded {dest_path.name} ({sz_mb:.2f} MB)")
    except Exception as e:
        sys.stdout.write("\n")
        if temp_dest.exists():
            temp_dest.unlink(missing_ok=True)
        print(f"  [ERR] Failed to download {dest_path.name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download ExecuTorch Llama 3.2 1B model assets")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory (default: models/llama)")
    parser.add_argument("--tokenizer-only", action="store_true", help="Download only tokenizer and config without the 1GB PTE")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dest_dir = Path(args.dest) if args.dest else project_root / "models" / "llama"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Abhedya SoC / Trinetra AI -- ExecuTorch Model Asset Downloader")
    print(f"  Target Directory: {dest_dir}")
    print(f"  Source: huggingface.co/{HF_REPO}")
    print("=" * 65)

    files_to_download = FILES[:2] if args.tokenizer_only else FILES

    for remote_rel, local_rel in files_to_download:
        url = f"{BASE_URL}/{remote_rel}"
        target_path = dest_dir / local_rel
        try:
            download_with_progress(url, target_path)
        except Exception as e:
            print(f"Error downloading {remote_rel}: {e}")
            if args.tokenizer_only:
                sys.exit(1)

    # Also mirror PTE to root of dest_dir if present in xnnpack/
    pte_xnnpack = dest_dir / "xnnpack" / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte"
    pte_root = dest_dir / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte"
    if pte_xnnpack.exists() and not pte_root.exists():
        try:
            os.link(pte_xnnpack, pte_root)
        except Exception:
            pass

    print("\n[OK] Asset download sequence complete.")


if __name__ == "__main__":
    main()
