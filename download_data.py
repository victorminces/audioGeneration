"""Download a public audio dataset into data/raw/ for DDSP training.

Usage:
    python download_data.py --dataset librispeech --max-files 500
"""
import argparse
import os
import random
import shutil
import tarfile
import tempfile

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

DATASETS = {
    "librispeech": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "ext": (".flac", ".wav"),
        "label": "LibriSpeech test-clean — 346 MB — English speech, multiple speakers",
    },
    "speech_commands": {
        "url": "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
        "ext": (".wav",),
        "label": "Google Speech Commands v0.01 — 1.4 GB — spoken words",
    },
    "nsynth": {
        "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
        "ext": (".wav",),
        "label": "NSynth test — 1.4 GB — musical instrument notes",
    },
}


def download_dataset(name, max_files):
    cfg = DATASETS[name]
    url, exts = cfg["url"], cfg["ext"]
    os.makedirs(DATA_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = os.path.join(tmpdir, "dataset.tar.gz")

        print(f"Downloading {cfg['label']}...")
        r = requests.get(url, stream=True, timeout=300)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded // 1_000_000} / {total // 1_000_000} MB",
                          end="", flush=True)
        print()

        print("Extracting...")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        found = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    found.append(os.path.join(root, fname))

        if max_files and max_files < len(found):
            found = random.sample(found, max_files)

        existing = len([f for f in os.listdir(DATA_DIR)
                        if f.lower().endswith((".wav", ".flac"))])
        for i, src in enumerate(found):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(DATA_DIR, f"ds_{existing + i:06d}{ext}")
            shutil.copy2(src, dst)
            print(f"\r  copying {i + 1}/{len(found)}", end="", flush=True)
        print()

    print(f"Done — {len(found)} files added to {DATA_DIR}")
    return len(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="librispeech")
    parser.add_argument("--max-files", type=int, default=500,
                        help="Randomly subsample to this many files (0 = keep all)")
    args = parser.parse_args()
    download_dataset(args.dataset, args.max_files)


if __name__ == "__main__":
    main()
