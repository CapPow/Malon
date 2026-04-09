"""
predict_single.py

Example: classify a single herbarium image using Malon.

This script is intended as a starting point — modify it for your use case.
It downloads one example image from the released GBIF predictions dataset
and classifies it using the Malon model weights.

See README.md for setup instructions.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
import urllib.request, urllib.error
import socket
import sys

Image.MAX_IMAGE_PIXELS = None  # Herbarium images are large; this warning is expected
socket.setdefaulttimeout(10)  # seconds; prevents hung downloads from stalling the script
# ==============================================================================
# NOTICE: This script is a minimal working example intended to be modified.
# It is not a general-purpose CLI tool. Edit the variables below and the
# example block at the bottom to suit your use case.
# ==============================================================================

# --- Configuration ------------------------------------------------------------
# Path to Malon model weights.
# Default assumes you are running from the repository root.
# Change to an absolute path if needed, e.g.:
#   MODEL_PATH = "/home/user/models/model_infer.pt"
MODEL_PATH = Path("model_infer.pt")
WEIGHTS_URL = "https://github.com/CapPow/Malon/releases/download/v1.0.0/model_infer.pt"

# Path to the released GBIF predictions CSV (used to pull an example image URL).
GBIF_CSV = Path("data/gbif_predictions.csv")

# Downloaded example image will be saved here (relative to this script).
DOWNLOAD_DIR = Path(__file__).parent / "downloads"
# ------------------------------------------------------------------------------


class HerbariumClassifier:
    """
    Three-class herbarium image classifier for CV utility assessment.

    Classes
    -------
    0 — Not useful: field photos, closed packets, non-specimen content
    1 — Atypical: wood sections, fragments, non-standard preparations
    2 — Typical: standard pressed specimens suitable for CV applications
    """

    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
            )

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
            transforms.Resize((checkpoint["img_size"], checkpoint["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["norm_mean"], std=checkpoint["norm_std"]),
        ])

        print(f"Loaded: {checkpoint['arch']} | Test acc: {checkpoint.get('test_acc', 'N/A'):.3f} | Device: {self.device}")

    def classify_image(self, image_path):
        """
        Classify a single herbarium image.

        Parameters
        ----------
        image_path : str or Path

        Returns
        -------
        dict with keys: class_id, class_name, confidence, probabilities
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_id = int(probs.argmax())
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": float(probs[class_id]),
            "probabilities": probs.tolist(),
        }


# --- Class descriptions printed with results ----------------------------------
CLASS_DESCRIPTIONS = {
    0: "Not useful for CV (field photo, closed packet, non-specimen content)",
    1: "Atypical — potentially informative but non-standard preparation",
    2: "Typical pressed specimen — suitable for CV applications",
}


def download_image(url, dest_path):
    """Download a single image from a URL. Raises on failure."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from image host (access may be restricted): {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def fetch_weights(model_path=MODEL_PATH, url=WEIGHTS_URL):
    """Download model weights if not present at model_path."""
    if not Path(model_path).exists():
        print(f"Model weights not found at {model_path}.")
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

# ==============================================================================
# Example: classify one image pulled from the released GBIF predictions CSV.
# Modify this block to point at your own image or URL list.
# ==============================================================================

if __name__ == "__main__":
    print("""
+------------------------------------------------------------------------------+
|  MALON -- SINGLE IMAGE INFERENCE EXAMPLE                                     |
|  This script is a starting point. Copy and modify it for your own pipeline.  |
|  Downloads one random image from data/gbif_predictions.csv -- images are     |
|  verified out-of-training-distribution (GBIF hit-rate analysis dataset).     |
|  See README.md for setup and usage guidance.                                 |
+------------------------------------------------------------------------------+
""")

    fetch_weights()  # Retrieve the model weights.
    # Load the GBIF predictions CSV and randomly sample one example.
    # A random record is selected each run to demonstrate variation across institutions.
    # --- Modify here to select a specific record or use your own image path ---
    df = pd.read_csv(GBIF_CSV)
    #df = df[df["class_id"] == 0]  # Option to isolate sampling to a specific class
    # --------------------------------------------------------------------------

    classifier = HerbariumClassifier()

    # Attempt download, cycling through random candidates on failure.
    # Image URLs are institution-hosted and may be temporarily unavailable —
    # re-running will draw a new random record from a different provider.
    dest = None
    for attempt in range(10):
        example_row = df.sample(1).iloc[0]
        gbif_id = example_row["gbifID"]
        url = example_row["image_url"]
        dest = DOWNLOAD_DIR / f"{gbif_id}.jpg"

        print(f"Example image — gbifID: {gbif_id}")
        print(f"URL: {url}")
        print(f"Downloading to: {dest}")

        try:
            download_image(url, dest)
            break
        except RuntimeError as e:
            print(f"  Download failed (attempt {attempt + 1}/10): {e}")
            print("  Image source may be temporarily unavailable. Trying another record...\n")
            dest = None

    if dest is None:
        print("Could not download an example image after 10 attempts. "
              "Check your internet connection or try again later.")
        sys.exit(1)

    print("\nRunning inference...")
    result = classifier.classify_image(dest)

    print("\n--- Result ---")
    print(f"  Class ID    : {result['class_id']}")
    print(f"  Class name  : {result['class_name']}")
    print(f"  Confidence  : {result['confidence']:.3f}")
    print(f"  Description : {CLASS_DESCRIPTIONS[result['class_id']]}")
    print(f"  Probabilities — Class 0: {result['probabilities'][0]:.3f} | "
          f"Class 1: {result['probabilities'][1]:.3f} | "
          f"Class 2: {result['probabilities'][2]:.3f}")
