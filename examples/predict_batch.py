"""
predict_batch.py

Example: batch classify herbarium images using Malon.

Downloads a stratified sample from the released GBIF predictions dataset,
runs batch inference using a DataLoader for parallel image prefetching,
and sorts images into class subdirectories — mirroring a typical triage
pipeline: discard Class 0, review Class 1, pass Class 2 downstream.

This script is intended as a starting point — modify it for your use case.
It can be easily repurposed to operate on a local folder of images rather
than downloaded URLs; see comments in the example block below.

See README.md for setup instructions.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import urllib.request, urllib.error
import socket
import shutil
import time
import sys

Image.MAX_IMAGE_PIXELS = None  # Herbarium scans are large; suppress DecompressionBombWarning
socket.setdefaulttimeout(10)  # seconds; prevents hung downloads from stalling the script
# ==============================================================================
# NOTICE: This script is a starting point. Copy and modify it for your pipeline.
# Downloads a small stratified sample from data/gbif_predictions.csv —
# images are verified out-of-training-distribution (GBIF hit-rate analysis
# dataset). To run on already-downloaded images, see the comment in the
# example block below. See README.md for setup and usage guidance.
# ==============================================================================

# --- Configuration ------------------------------------------------------------
# Path to Malon model weights.
# Default assumes you are running from the repository root.
# Change to an absolute path if needed, e.g.:
#   MODEL_PATH = "/home/user/models/model_infer.pt"
MODEL_PATH = Path("model_infer.pt")
WEIGHTS_URL = "https://github.com/CapPow/Malon/releases/download/v1.0.0/model_infer.pt"

# Path to the released GBIF predictions CSV (used to pull example image URLs).
GBIF_CSV = Path("data/gbif_predictions.csv")

# Number of images to sample per class (0, 1, 2) -> SAMPLE_N * 3 total images.
# Class 1 (atypical) has the smallest pool (~550 records); keep SAMPLE_N well
# below that if increasing. Class 0 and 2 pools are much larger.
SAMPLE_N = 8  # -> 24 images total, 3 batches of 8

# Inference batch size.
# 8 is conservative and runs comfortably on CPU or modest GPU.
# Push to 32+ on a dedicated GPU for substantially higher throughput.
# See README.md for benchmark timings.
BATCH_SIZE = 8

# DataLoader worker processes for parallel image loading and preprocessing.
# 2 works reliably on Linux, macOS, and Windows with Python 3.11/3.12.
# Set to 0 if you encounter multiprocessing errors (e.g. Python 3.13+).
# For large datasets, scaling toward (CPU cores - 1) improves throughput.
NUM_WORKERS = 2

# Downloaded images land here before sorting.
DOWNLOAD_DIR = Path(__file__).parent / "downloads" / "batch"

# Sorted outputs: images are moved into class subdirectories after inference.
OUTPUT_DIR = Path(__file__).parent / "downloads" / "batch"
# ------------------------------------------------------------------------------


class HerbariumDataset(Dataset):
    """
    Minimal Dataset wrapping a list of image paths for use with DataLoader.
    Returns (tensor, path_string) pairs so results can be matched back to files.
    Swap image_paths for any list of Path objects to repurpose for your data.
    """

    def __init__(self, image_paths, transform):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


class HerbariumClassifier:
    """
    Three-class herbarium image classifier for CV utility assessment.

    Classes
    -------
    0 -- Not useful: field photos, closed packets, non-specimen content
    1 -- Atypical: wood sections, fragments, non-standard preparations
    2 -- Typical: standard pressed specimens suitable for CV applications
    """

    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint
        self.class_names = checkpoint["class_names"]

        self.model = models.swin_v2_t(weights=None)
        self.model.head = torch.nn.Linear(
            self.model.head.in_features, checkpoint["num_classes"]
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            # Note: inference time scales with input resolution. Pre-scaling
            # large images before this pipeline substantially reduces wall-clock
            # time. See README.md for benchmark details.
            transforms.Resize((checkpoint["img_size"], checkpoint["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["norm_mean"], std=checkpoint["norm_std"]),
        ])

        print(f"Loaded: {checkpoint['arch']} | Test acc: {checkpoint.get('test_acc', 'N/A'):.3f} | Device: {self.device}")

    def classify_batch(self, image_paths, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        """
        Classify a list of images using DataLoader for parallel prefetching.

        Parameters
        ----------
        image_paths : list of str or Path
        batch_size : int
        num_workers : int -- parallel workers for image loading (0 = main thread)

        Returns
        -------
        list of dicts with keys: class_id, class_name, confidence, probabilities
            Results are returned in the same order as image_paths.
        """
        dataset = HerbariumDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            shuffle=False,  # preserve order so results align with image_paths
        )

        results = []
        with torch.no_grad():
            for batch_tensors, _ in loader:
                batch_tensors = batch_tensors.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(batch_tensors)
                probs = F.softmax(logits, dim=1).cpu().numpy()

                for prob in probs:
                    class_id = int(prob.argmax())
                    results.append({
                        "class_id": class_id,
                        "class_name": self.class_names[class_id],
                        "confidence": float(prob[class_id]),
                        "probabilities": prob.tolist(),
                    })

        return results


def download_image(url, dest_path):
    """Download a single image from a URL. Raises on failure."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from image host (access may be restricted): {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def download_stratified_sample(df, sample_n, download_dir):
    """
    Download sample_n images per class, retrying on failure until the pool
    is exhausted. Returns a list of (dest_path, true_class_id) tuples for
    successfully downloaded images.
    """
    downloaded = []

    for class_id in [0, 1, 2]:
        pool = df[df["class_id"] == class_id].sample(frac=1).reset_index(drop=True)
        retrieved = []
        idx = 0

        while len(retrieved) < sample_n and idx < len(pool):
            row = pool.iloc[idx]
            gbif_id = row["gbifID"]
            url = row["image_url"]
            dest = download_dir / f"{gbif_id}.jpg"
            idx += 1

            try:
                download_image(url, dest)
                retrieved.append((dest, class_id))
                time.sleep(0.5)  # Be courteous to institution image servers
            except RuntimeError as e:
                print(f"  Skipping {gbif_id}: {e}")

        if len(retrieved) < sample_n:
            print(f"  Warning: only retrieved {len(retrieved)}/{sample_n} "
                  f"images for class {class_id} -- pool may have access restrictions.")

        downloaded.extend(retrieved)

    return downloaded

def fetch_weights(model_path=MODEL_PATH, url=WEIGHTS_URL):
    """Download model weights if not present at model_path."""
    if not Path(model_path).exists():
        print(f"Model weights not found at {model_path}.")
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

# ==============================================================================
# Example: stratified download, batch inference, and class-based sorting.
#
# To run on already-downloaded images instead, replace the download block with:
#   image_paths = list(Path("your/image/folder").glob("*.jpg"))
#   downloaded = [(p, None) for p in image_paths]
# ==============================================================================

if __name__ == "__main__":
    print("""
+------------------------------------------------------------------------------+
|  MALON -- BATCH INFERENCE EXAMPLE                                            |
|  This script is a starting point. Copy and modify it for your own pipeline.  |
|  Downloads a stratified sample from data/gbif_predictions.csv -- images are  |
|  verified out-of-training-distribution (GBIF hit-rate analysis dataset).     |
|  To run on already-downloaded images, see the comment in the example block.  |
|  See README.md for setup and usage guidance.                                 |
+------------------------------------------------------------------------------+
""")

    fetch_weights()  # Retrieve the model weights.
    classifier = HerbariumClassifier()

    # --- Download stratified sample ---
    print(f"\nDownloading {SAMPLE_N} images per class ({SAMPLE_N * 3} total) "
          f"from data/gbif_predictions.csv...")
    df = pd.read_csv(GBIF_CSV)
    downloaded = download_stratified_sample(df, SAMPLE_N, DOWNLOAD_DIR)
    image_paths = [p for p, _ in downloaded]

    if not image_paths:
        print("No images downloaded. Check your internet connection and try again.")
        sys.exit(1)

    print(f"\nDownload complete: {len(image_paths)} images ready for inference.")

    # --- Batch inference (timed) ---
    print(f"\nRunning batch inference (batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...")
    t0 = time.perf_counter()
    results = classifier.classify_batch(image_paths, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    elapsed = time.perf_counter() - t0

    print(f"Inference complete: {len(results)} images in {elapsed:.2f}s "
          f"({elapsed / len(results) * 1000:.1f} ms/image)")

    # --- Move images to class subdirectories ---
    # This block mirrors a typical triage pipeline:
    #   class_0 -- discard or quarantine
    #   class_1 -- manual review
    #   class_2 -- pass downstream
    # Modify to copy instead of move, log to CSV, filter by confidence, etc.
    print("\nSorting images into class subdirectories...")
    class_dirs = {i: OUTPUT_DIR / f"class_{i}" for i in [0, 1, 2]}
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for path, result in zip(image_paths, results):
        dest = class_dirs[result["class_id"]] / path.name
        shutil.move(str(path), dest)

    # --- Summary ---
    print("\n--- Inference Summary ---")
    print(f"  {'Predicted class':<22} {'N':>4}  {'Mean confidence':>16}")
    print(f"  {'-'*46}")
    for class_id in [0, 1, 2]:
        class_results = [r for r in results if r["class_id"] == class_id]
        n = len(class_results)
        if n > 0:
            mean_conf = sum(r["confidence"] for r in class_results) / n
            name = next((r["class_name"] for r in class_results), "")
            print(f"  class_{class_id} ({name:<12}) {n:>4}  {mean_conf:>16.3f}")
    print(f"\n  Images sorted to: {OUTPUT_DIR.resolve()}")
