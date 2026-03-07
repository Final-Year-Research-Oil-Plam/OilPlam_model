# Palm Model Full Workflow (Supervisor-Friendly)

This document explains the full project end-to-end:

- How each file is used
- How object detection and classification run
- How image URL input is handled
- How the color-code day logic works
- How this connects to a Roboflow training workflow

## 1. Project purpose

This API takes an oil palm bunch image and returns:

- Whether an object is detected
- Detected box coordinates
- Classification label and confidence
- Optional day-class result (`2d`, `4d`, `12d`-`16d`) from ROI color analysis

## 2. Folder and file roles

### Root files

- `requirements.txt`
  - Python dependencies (`ultralytics`, `opencv-python`, `fastapi`, `httpx`, `python-dotenv`, etc.).

- `.env` (not in git usually)
  - Environment variables, especially `CLOUDINARY_NAME` for URL flow using `public_id`.

- `workflow.md`
  - This explanation document.

### `app/` files

- `app/main.py`
  - FastAPI entrypoint.
  - Creates app object and includes routers.

- `app/config.py`
  - Central settings.
  - Defines model file paths, confidence threshold, API metadata, and output text path (`api.txt`).

- `app/routers/router.py`
  - HTTP endpoints.
  - Validates input, downloads image (for URL flow), creates temporary files, calls inference, returns JSON.

- `app/infer.py`
  - High-level inference orchestrator.
  - Runs detection first, then classification, then optional date logic.

- `app/service.py`
  - Model and image utilities.
  - Loads YOLO models, runs detection/classification, crops ROI, writes hex color file, calls day logic.

- `app/day_from_hex.py`
  - Color-code day decision engine.
  - Converts ROI pixel hex values into weighted class scores and outputs final day class.

- `app/api.txt`
  - Generated intermediate file.
  - Contains one pixel color per line as `#RRGGBB` from ROI.

- `app/palm_det_v1_50.pt`
  - Trained object detection model weights.

- `app/palm_cls_v1.0_20.pt`
  - Trained classification model weights.

- `app/__init__.py`
  - Package marker.

## 3. API endpoints and what they do

### Health/info endpoints

- `GET /`
  - Basic message.

- `GET /health`
  - Service health check.

- `GET /api/v1/info`
  - Returns API title and version.

- `GET /palm/models`
  - Returns configured model file paths.

### Inference endpoints

- `POST /palm/detect`
  - Input: uploaded image file.
  - Runs normal detection + classification pipeline.
  - Runs date logic only if class is `ripe` or `unripe`.

- `POST /palm/detect-with-date`
  - Input: uploaded image file.
  - Same as above, but forces date logic (`always_run_date=True`) when detection exists.

- `POST /palm/detect-from-url`
  - Input: JSON with either `image_url` or `public_id`.
  - Downloads image first, then runs same pipeline as `/palm/detect`.

## 4. End-to-end runtime workflow

The system always follows this core order:

1. Validate input.
2. Ensure there is a local image file path (upload temp file or URL download temp file).
3. Run object detection.
4. If detection exists, run classification.
5. If date logic condition passes, crop first detection ROI and export colors to hex text.
6. Run day classification from hex file.
7. Return structured JSON response.
8. Delete temporary image file.

## 5. Detailed workflow: image URL case

Endpoint: `POST /palm/detect-from-url` in `app/routers/router.py`.

1. Request validation:
   - Reject if both `image_url` and `public_id` are sent.
   - Reject if neither is sent.

2. Build URL:
   - If `public_id` is sent, build Cloudinary URL using `CLOUDINARY_NAME`.

3. Download image:
   - `_fetch_image_bytes()` fetches content using `httpx`.
   - Validates HTTP 200.
   - Chooses file suffix from content-type.

4. Temp file:
   - Writes bytes into `NamedTemporaryFile`.

5. Inference call:
   - Calls `run_infer(tmp_path)` from `app/infer.py`.

6. Response mapping:
   - Returns only standard output keys.

7. Cleanup:
   - Temp file removed in `finally`.

## 6. Detection + classification logic

Main function: `run()` in `app/infer.py`.

1. `run_detection(image_path)` from `app/service.py`:
   - Uses YOLO detection model (`palm_det_v1_50.pt`).
   - Confidence threshold from `DETECTION_CONFIDENCE` in `app/config.py`.

2. Convert boxes:
   - `_boxes_to_coords()` converts YOLO box objects to plain `{x1,y1,x2,y2}` list.

3. If at least one box:
   - Run `run_classification(image_path)` using `palm_cls_v1.0_20.pt`.
   - Returns top class and confidence.

4. Decide date logic:
   - If `always_run_date=True`, date logic runs.
   - Else date logic runs only when class is in `{"ripe", "unripe"}`.

5. If no boxes:
   - Return `success: false` and no class/date outputs.

## 7. ROI and color extraction logic

Implemented in `app/service.py`.

1. First detected box is selected.
2. `_crop_roi()` scales box by `0.8` from center.
3. ROI is converted BGR to RGB.
4. Each pixel becomes a hex string (`#RRGGBB`).
5. Hex values are written line-by-line to `app/api.txt`.

Why this step exists:

- The day logic module works on color distributions from ROI pixels, not directly on the original image.

## 8. Color-code day logic explained

Implemented in `app/day_from_hex.py`.

### Input

- File with many lines like `#a1b2c3`.

### Stage A: Convert hex to RGB

- `hex_to_rgb()` converts each line to integer `(R,G,B)`.

### Stage B: Compute channel coverage

For each class (`2d`, `4d`, `12d`, `16d`) and each channel (`R`, `G`, `B`):

- Count how many pixels are inside that class channel range.
- Coverage percent formula:

```text
coverage = (pixels in channel range / total pixels) * 100
```

### Stage C: Compute weighted class scores

Each class has weights per channel.

Formula:

```text
raw_score = (wR*covR + wG*covG + wB*covB) / 3
normalized_score = (raw_score / sum_of_all_raw_scores) * 100
```

### Stage D: Final day decision

1. Find winning class from normalized scores.
2. If winner is `12d`, map score to one of `12d`, `13d`, `14d`, `15d`, `16d`.
3. If winner is not `12d`, apply special rules using:
   - Gap between rounded `2d` and `4d`
   - Constraint that `4d` only valid when `50 <= score_4d <= 70`

Output:

- `final_date`
- `winning_class`
- `scores`

## 9. Roboflow training workflow (how to present)

The `.pt` files indicate YOLO model artifacts that are typically produced from a Roboflow + Ultralytics pipeline. A clear supervisor explanation is:

1. Collected and labeled images in Roboflow.
2. Created two tasks:
   - Object detection dataset (bounding boxes).
   - Classification dataset (ripe/unripe/other labels as required).
3. Applied preprocessing/augmentation in Roboflow.
4. Exported datasets in YOLO format.
5. Trained with Ultralytics YOLO.
6. Saved best weights as:
   - `palm_det_v1_50.pt` (detection)
   - `palm_cls_v1.0_20.pt` (classification)
7. Integrated weights into FastAPI inference service.

If your supervisor asks for evidence, provide:

- Roboflow project links/screenshots
- Label class list
- Train/val/test split
- Metrics (mAP for detection, accuracy/F1 for classification)
- Training config used (epochs, image size, model variant)

## 10. Response schema (what client receives)

Standard response keys:

- `success`: boolean
- `message`: status text
- `count`: number of detected boxes
- `coordinates`: list of detected boxes
- `class`: classification label or `null`
- `confidence`: classification confidence or `null`
- `api_txt_path`: saved hex file path or `null`
- `final_date`: day class or `null`

## 11. Common failure points and fixes

- Model file missing:
  - Ensure both `.pt` files exist in `app/`.

- URL download fails:
  - Verify image is public and returns HTTP 200.

- Cloudinary path fails:
  - Set `CLOUDINARY_NAME` in `.env`.

- No detections:
  - Check image quality/angle and detection threshold.

- No final date:
  - Date logic may be skipped by class condition or ROI hex may be invalid.

## 12. Quick demo script for supervisor

Run API:

```bash
uvicorn app.main:app --reload
```

Test URL endpoint:

```bash
curl -X POST http://127.0.0.1:8000/palm/detect-from-url \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/sample.jpg"}'
```

Show that output includes:

- Detection result
- Class and confidence
- Optional final day

## 13. One-line architecture summary

FastAPI endpoint receives image input, YOLO detects objects, YOLO classifies maturity, ROI pixels are converted to hex, weighted RGB range logic determines day class, and the API returns a single structured JSON result.
