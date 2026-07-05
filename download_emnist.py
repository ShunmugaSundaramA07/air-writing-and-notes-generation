"""
Download EMNIST ByClass split and populate the dataset/ folder with
~500 samples per class for A-Z (uppercase) and 0-9.
EMNIST ByClass mapping: 0-9 -> digits, 10-35 -> A-Z, 36-61 -> a-z
We only need 0-35 (digits + uppercase).
"""

import os
import sys
import gzip
import struct
import urllib.request
import numpy as np
import cv2

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
SAMPLES_PER_CLASS = 500

# EMNIST ByClass URLs (from NIST)
BASE_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip/"
FILES = {
    "train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
}

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "_emnist_cache")

# EMNIST ByClass class mapping: 0-9 = digits, 10-35 = uppercase A-Z
# We want folders: '0','1',...'9','A','B',...'Z'
CLASS_MAP = {}
for i in range(10):
    CLASS_MAP[i] = str(i)
for i in range(26):
    CLASS_MAP[10 + i] = chr(ord('A') + i)


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  [CACHED] {os.path.basename(dest)}")
        return
    print(f"  [DOWNLOADING] {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  [DONE] {os.path.basename(dest)}")


def read_idx_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def fix_emnist_orientation(img):
    """EMNIST images need to be transposed and flipped horizontally."""
    return np.fliplr(np.transpose(img))


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Download EMNIST files
    print("[1/4] Downloading EMNIST ByClass dataset...")
    paths = {}
    for key, fname in FILES.items():
        dest = os.path.join(DOWNLOAD_DIR, fname)
        download_file(BASE_URL + fname, dest)
        paths[key] = dest

    # Read data
    print("[2/4] Reading EMNIST data...")
    images = read_idx_images(paths["train_images"])
    labels = read_idx_labels(paths["train_labels"])
    print(f"  Total training samples: {len(labels)}")

    # Count existing samples per class
    print("[3/4] Counting existing samples...")
    existing_counts = {}
    for class_idx, folder_name in CLASS_MAP.items():
        folder = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(folder, exist_ok=True)
        count = len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        existing_counts[folder_name] = count
        print(f"  {folder_name}: {count} existing samples")

    # Save EMNIST samples
    print(f"[4/4] Saving up to {SAMPLES_PER_CLASS} EMNIST samples per class...")
    saved_counts = {name: 0 for name in CLASS_MAP.values()}
    needed = {name: max(0, SAMPLES_PER_CLASS - existing_counts.get(name, 0))
              for name in CLASS_MAP.values()}

    for i in range(len(labels)):
        class_idx = int(labels[i])
        if class_idx not in CLASS_MAP:
            continue  # skip lowercase (36-61)

        folder_name = CLASS_MAP[class_idx]
        if saved_counts[folder_name] >= needed[folder_name]:
            continue

        # Fix orientation and save
        img = fix_emnist_orientation(images[i])
        folder = os.path.join(DATASET_DIR, folder_name)
        base_count = existing_counts.get(folder_name, 0)
        out_idx = base_count + saved_counts[folder_name] + 1
        out_path = os.path.join(folder, f"emnist_{out_idx:05d}.png")
        cv2.imwrite(out_path, img)
        saved_counts[folder_name] += 1

        # Check if all classes are done
        if all(saved_counts[n] >= needed[n] for n in CLASS_MAP.values()):
            break

    # Summary
    print("\n=== Dataset Summary ===")
    total = 0
    for class_idx in sorted(CLASS_MAP.keys()):
        name = CLASS_MAP[class_idx]
        folder = os.path.join(DATASET_DIR, name)
        count = len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        total += count
        added = saved_counts[name]
        print(f"  {name}: {count} total ({added} new from EMNIST)")
    print(f"\n  TOTAL: {total} samples across {len(CLASS_MAP)} classes")
    print("[DONE] Dataset is ready for training!")


if __name__ == "__main__":
    main()
