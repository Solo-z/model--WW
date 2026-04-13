#!/usr/bin/env python3
"""
Download and prepare Lakh MIDI Dataset

Downloads the full Lakh MIDI dataset (~32GB, 176k MIDI files)
and prepares it for training.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


LAKH_URLS = {
    "lmd_full": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
    "lmd_matched": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
    "lmd_aligned": "http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz",
}

LAKH_SIZES = {
    "lmd_full": "32GB",
    "lmd_matched": "6GB",
    "lmd_aligned": "3GB",
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_tar_gz(tar_path: Path, output_dir: Path) -> None:
    """Extract a tar.gz file with progress."""
    print(f"Extracting {tar_path} to {output_dir}...")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, output_dir)


def count_midi_files(directory: Path) -> int:
    """Count MIDI files in directory."""
    count = 0
    for ext in [".mid", ".midi", ".MID", ".MIDI"]:
        count += len(list(directory.rglob(f"*{ext}")))
    return count


def main():
    parser = argparse.ArgumentParser(description="Download Lakh MIDI Dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/lakh_midi",
        help="Output directory for MIDI files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmd_full",
        choices=["lmd_full", "lmd_matched", "lmd_aligned"],
        help="Which dataset variant to download",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded tar.gz file",
    )
    parser.add_argument(
        "--use-wget",
        action="store_true",
        help="Use wget instead of Python requests (faster, requires wget)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = LAKH_URLS[args.dataset]
    archive_name = f"{args.dataset}.tar.gz"
    archive_path = output_dir / archive_name
    
    print("=" * 60)
    print("Lakh MIDI Dataset Downloader")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Size: ~{LAKH_SIZES[args.dataset]}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Download
    if not archive_path.exists():
        print(f"\nDownloading from {url}...")
        
        if args.use_wget:
            subprocess.run(
                ["wget", "-c", "-O", str(archive_path), url],
                check=True,
            )
        else:
            download_file(url, archive_path)
        
        print(f"Downloaded to {archive_path}")
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Extract
    print("\nExtracting archive...")
    extract_tar_gz(archive_path, output_dir)
    
    # Count files
    midi_count = count_midi_files(output_dir)
    print(f"\nExtracted {midi_count} MIDI files")
    
    # Cleanup
    if not args.keep_archive:
        print(f"\nRemoving archive {archive_path}...")
        archive_path.unlink()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"MIDI files are in: {output_dir}")
    print("=" * 60)
    
    # Print next steps
    print("""
Next steps:
    1. Preprocess the dataset:
       python -c "from modelw.dataset import LakhMIDIDataset, DatasetConfig; \\
                  from modelw.tokenizer import MIDITokenizer; \\
                  tok = MIDITokenizer(); \\
                  ds = LakhMIDIDataset(DatasetConfig(data_dir='%s'), tok)"
    
    2. Start training:
       python -m modelw.trainer --data_dir=%s --model_size=base
""" % (output_dir, output_dir))


if __name__ == "__main__":
    main()

