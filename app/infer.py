"""
Inference: run detection and return success / no-detection message.
When object detected, run classification and include class + coordinates.
If class is "ripe" or "unripe", scale down the detected area and save ROI hex to api.txt (check_range_2 style).
"""
import logging
from pathlib import Path

try:
    from app.service import (
        run_detection,
        run_classification,
        extract_roi_hex_and_save,
        get_final_date_from_hex_file,
    )
except ModuleNotFoundError:
    from service import (
        run_detection,
        run_classification,
        extract_roi_hex_and_save,
        get_final_date_from_hex_file,
    )

MSG_DETECTED = "Object detected."
MSG_NOT_DETECTED = "No object detected."
# Classes that trigger day-class (ROI hex + date) when always_run_date is False
DATE_CLASSES = {"ripe", "unripe"}
# Same scale is used when extracting ROI hex from the first detected box.
ROI_SCALE_DOWN = 0.8

logger = logging.getLogger(__name__)


def _boxes_to_coords(result):
    """Extract list of {x1, y1, x2, y2} from YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    coords = []
    for box in result.boxes:
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        coords.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return coords


def run(image_path: str, always_run_date: bool = False) -> dict:
    """
    Run detection on the image at image_path. Return coordinates of all detected boxes.
    If any object detected, run classification.
    Date (ROI hex + day_from_hex):
      - If always_run_date=True: always run for first box (no class filter).
      - If always_run_date=False: only run when class is "ripe" or "unripe".
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 1) Run object detection and normalize output to a simple coordinate list.
    results = run_detection(image_path)
    if not results:
        raise ValueError("Detection returned no results")
    result = results[0]
    count = len(result.boxes) if result.boxes is not None else 0
    coordinates = _boxes_to_coords(result)

    if count > 0:
        # Print detection confidence per box (object detection model)
        if result.boxes is not None and result.boxes.conf is not None:
            confs = [float(c) for c in result.boxes.conf]
            print("[object detection confidence]", confs)
        # 2) Run whole-image maturity classification once at least one object exists.
        cls_out = run_classification(image_path)
        class_name = cls_out.get("class")
        confidence = cls_out.get("confidence")
        out = {
            "success": True,
            "message": MSG_DETECTED,
            "count": count,
            "coordinates": coordinates,
            "class": class_name,
            "confidence": confidence,
            "api_txt_path": None,
            "final_date": None,
        }
        # 3) Decide whether to run date estimation from ROI hex values.
        run_date = always_run_date or (class_name and class_name.lower() in DATE_CLASSES)
        if run_date and coordinates:
            try:
                # Current rule: use only the first detected box for date extraction.
                box = coordinates[0]
                if always_run_date:
                    print("[date flow] always_run_date -> extracting ROI for first box, then computing date")
                else:
                    print("[date flow] class=ripe/unripe -> extracting ROI for first box, then computing date")
                print("[date flow] box (x1,y1,x2,y2):", box["x1"], box["y1"], box["x2"], box["y2"], "scale=", ROI_SCALE_DOWN)
                path_str = extract_roi_hex_and_save(
                    image_path,
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    scale=ROI_SCALE_DOWN,
                )
                out["api_txt_path"] = path_str
                print("[date flow] hex file saved ->", path_str)
                print("[date flow] running date-from-hex analysis...")
                final_date = get_final_date_from_hex_file(path_str)
                if final_date is not None:
                    out["final_date"] = final_date
                    print("[date flow] identified date:", final_date)
                else:
                    print("[date flow] date identification returned None (see [date flow] / [date] logs above)")
            except Exception as e:
                logger.warning("Ripe ROI hex or day_from_hex failed: %s", e, exc_info=True)
        print("[final result]", out)
        return out
    no_detect = {
        "success": False,
        "message": MSG_NOT_DETECTED,
        "count": 0,
        "coordinates": [],
        "class": None,
        "confidence": None,
        "api_txt_path": None,
        "final_date": None,
    }
    print("[final result]", no_detect)
    return no_detect
