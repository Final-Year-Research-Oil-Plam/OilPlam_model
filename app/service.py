"""
Model loading, detection, and classification.
Detection: palm_det model. Classification: palm_cls model (run when object detected).
ROI hex export (check_range_2 style) for ripe/unripe class.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

# Support both run from app/ and from workspace (app as package)
try:
    from app.config import MODEL_DET_PATH, MODEL_CLS_PATH, API_TXT_PATH, DETECTION_CONFIDENCE
    from app.day_from_hex import run as run_day_from_hex
except ModuleNotFoundError:
    from config import MODEL_DET_PATH, MODEL_CLS_PATH, API_TXT_PATH, DETECTION_CONFIDENCE
    from day_from_hex import run as run_day_from_hex

_det_model = None
_cls_model = None


def get_detection_model():
    """Load and cache the detection model."""
    global _det_model
    if _det_model is None:
        path = Path(MODEL_DET_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Detection model not found: {path}")
        _det_model = YOLO(str(path))
    return _det_model


def get_classification_model():
    """Load and cache the classification model (same as classigy_inf.py)."""
    global _cls_model
    if _cls_model is None:
        path = Path(MODEL_CLS_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Classification model not found: {path}")
        _cls_model = YOLO(str(path))
    return _cls_model


def run_detection(source_path: str):
    """Run detection model on an image path and return raw YOLO results."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    model = get_detection_model()
    results = model(source_path, save=False, show=False, conf=DETECTION_CONFIDENCE)
    return results


def run_classification(image_path: str) -> dict:
    """
    Run classification on image (as in classigy_inf.py).
    Returns {"class": str, "confidence": float}.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    model = get_classification_model()
    results = model.predict(source=image_path, imgsz=64, device="cpu")
    probs = results[0].probs
    cls_id = probs.top1
    conf = float(probs.top1conf)
    cls_name = model.names[cls_id]
    return {"class": cls_name, "confidence": conf}


def _rgb_to_hex(rgb):
    """BGR or RGB tuple to #rrggbb (expects (r,g,b) or (b,g,r) from cv2)."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _validate_box_coords(x1: float, y1: float, x2: float, y2: float) -> None:
    """Raise ValueError if box is invalid (no area, or non-finite)."""
    for v in (x1, y1, x2, y2):
        if not (v == v and abs(v) != float("inf")):  # finite check
            raise ValueError(f"Invalid box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box has no area: x1={x1}, y1={y1}, x2={x2}, y2={y2}")


def _crop_roi(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float = 0.8,
):
    """Scale down the box from center, crop and return ROI (BGR). Raises on invalid box/image."""
    _validate_box_coords(x1, y1, x2, y2)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h_img, w_img = img.shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    x1_s = max(0, int(cx - w / 2))
    y1_s = max(0, int(cy - h / 2))
    x2_s = min(w_img, int(cx + w / 2))
    y2_s = min(h_img, int(cy + h / 2))
    if x2_s <= x1_s or y2_s <= y1_s:
        raise ValueError("Scaled ROI has no area")
    return img[y1_s:y2_s, x1_s:x2_s]


def get_roi_image_bytes(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float = 0.8,
    fmt: str = "png",
) -> tuple[bytes, str]:
    """
    Crop ROI using the same scaled box as extract_roi_hex_and_save.
    Returns (image_bytes, content_type) e.g. (png_bytes, "image/png").
    """
    roi = _crop_roi(image_path, x1, y1, x2, y2, scale)
    if fmt.lower() in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", roi)
        return (buf.tobytes(), "image/jpeg")
    ok, buf = cv2.imencode(".png", roi)
    if not ok:
        raise ValueError("Failed to encode ROI as PNG")
    return (buf.tobytes(), "image/png")


def show_roi_popup(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float = 0.8,
    window_name: str = "ROI (detect-with-date)",
    wait_ms: int = 2000,
) -> None:
    """Best-effort ROI preview for local debugging; silently ignores UI errors."""
    try:
        roi = _crop_roi(image_path, x1, y1, x2, y2, scale)
        cv2.imshow(window_name, roi)
        cv2.waitKey(wait_ms if wait_ms > 0 else 0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def extract_roi_hex_and_save(
    image_path: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float = 0.8,
    output_path: Path = None,
) -> str:
    """Crop ROI, convert pixels to hex lines, and save them to a text file."""
    if output_path is None:
        output_path = API_TXT_PATH
    roi = _crop_roi(image_path, x1, y1, x2, y2, scale)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    hex_values = []
    for row in roi_rgb:
        for pixel in row:
            hex_values.append(_rgb_to_hex(pixel))
    num_pixels = len(hex_values)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # One #RRGGBB value per line so day_from_hex can process the file directly.
    with open(output_path, "w") as f:
        for hx in hex_values:
            f.write(hx + "\n")
    print("[date flow] ROI extracted: scale=%s, pixels=%d, hex lines written to %s" % (scale, num_pixels, output_path))
    return str(output_path)


def get_final_date_from_hex_file(hex_file_path: str) -> str:
    """Run day-from-hex logic and return only final date (or None on error)."""
    print("[date flow] day_from_hex input:", hex_file_path)
    result = run_day_from_hex(hex_file_path)
    final = result.get("final_date") if result.get("error") is None else None
    if result.get("error"):
        print("[date flow] day_from_hex error:", result.get("error"))
    return final
