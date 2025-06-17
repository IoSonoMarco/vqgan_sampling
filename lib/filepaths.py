import os
from pathlib import Path

DATASET_IMAGE_TOKENS = os.path.join(Path(__file__).resolve().parents[1], "assets", "image_tokens.npy")
DATASET_IMAGE_METADATA = os.path.join(Path(__file__).resolve().parents[1], "assets", "metadata.tsv")