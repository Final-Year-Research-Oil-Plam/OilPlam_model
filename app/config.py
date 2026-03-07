"""
Application configuration. Model paths are relative to the app package directory.
"""
from pathlib import Path
import os

# Load .env from workspace root (parent of app/) so CLOUDINARY_NAME etc. are set
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
except ImportError:
    pass

# Base directory of the app package (this file's parent)
APP_DIR = Path(__file__).resolve().parent

# YOLO .pt models in app folder
MODEL_CLS_PATH = APP_DIR / "palm_cls_v1.0_20.pt"
MODEL_DET_PATH = APP_DIR / "palm_det_v1_50.pt"

# Optional: explicit list for iteration
MODEL_PATHS = {
    "classification": MODEL_CLS_PATH,
    "detection": MODEL_DET_PATH,
}

# API settings
API_TITLE = "Palm API"
API_VERSION = "1.0.0"

# Output for ripe ROI hex dump (same dir as app)
API_TXT_PATH = APP_DIR / "api.txt"

# Detection: minimum confidence (0–1). Boxes below this are discarded.
DETECTION_CONFIDENCE = 0.5

# Cloudinary (optional – for /palm/detect-from-url with public_id)
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_NAME", "").strip() or None
