"""HTTP routes for health checks and palm detection workflows."""

import tempfile
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Body, File, UploadFile

try:
    # Normal package imports when app is started from project root.
    from app.config import (
        CLOUDINARY_CLOUD_NAME,
        MODEL_PATHS,
        MODEL_CLS_PATH,
        MODEL_DET_PATH,
    )
    from app.infer import run as run_infer
    from app.service import show_roi_popup
except ModuleNotFoundError:
    # Fallback for direct script-style execution from inside app/.
    from config import (
        CLOUDINARY_CLOUD_NAME,
        MODEL_PATHS,
        MODEL_CLS_PATH,
        MODEL_DET_PATH,
    )
    from infer import run as run_infer
    from service import show_roi_popup

# Health routes
health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health():
    return {"status": "ok"}


@health_router.get("/")
def root():
    return {"message": "Palm API"}


# Palm / models routes
palm_router = APIRouter(prefix="/palm", tags=["palm"])


@palm_router.get("/models")
def list_models():
    """Return configured .pt model paths (as strings)."""
    return {
        "classification": str(MODEL_CLS_PATH),
        "detection": str(MODEL_DET_PATH),
        "models": {k: str(v) for k, v in MODEL_PATHS.items()},
    }


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Default suffix when Content-Type is unknown
DEFAULT_IMAGE_SUFFIX = ".jpg"
CONTENT_TYPE_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _cloudinary_url(public_id: str) -> str:
    """Build Cloudinary delivery URL using cloud name from .env and the given public_id."""
    if not CLOUDINARY_CLOUD_NAME:
        raise ValueError("Cloudinary not configured (set CLOUDINARY_NAME in .env)")
    pid = (public_id or "").strip()
    if not pid:
        raise ValueError("public_id is required when using Cloudinary")
    return f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/image/upload/{pid}"


async def _fetch_image_bytes(url: str, timeout: float = 30.0) -> tuple[bytes, str]:
    """
    Fetch image from URL. Returns (content_bytes, suffix for temp file).
    Raises ValueError on non-200 or non-image response.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch image: HTTP {resp.status_code}")
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    suffix = CONTENT_TYPE_TO_SUFFIX.get(content_type)
    if not suffix:
        for ct, ext in CONTENT_TYPE_TO_SUFFIX.items():
            if ct in content_type:
                suffix = ext
                break
    if not suffix:
        suffix = DEFAULT_IMAGE_SUFFIX
    return resp.content, suffix


def _detect_error_response(message: str) -> dict:
    """Return a consistent error payload for /detect."""
    return {
        "success": False,
        "message": message,
        "count": 0,
        "coordinates": [],
        "class": None,
        "confidence": None,
        "api_txt_path": None,
        "final_date": None,
    }


# Scale for ROI popup (same as infer.ROI_SCALE_DOWN)
_ROI_SCALE_FOR_DATE = 0.8


@palm_router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accept an image file, run object detection. Returns success message if any object
    detected, otherwise no-detection message.
    """
    if not file.filename or not file.filename.strip():
        return _detect_error_response("No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return _detect_error_response(
            f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    tmp_path = None
    try:
        content = await file.read()
        # The inference pipeline expects a path, so store upload in a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        out = run_infer(tmp_path)
        return {
            "success": out["success"],
            "message": out["message"],
            "count": out["count"],
            "coordinates": out.get("coordinates", []),
            "class": out.get("class"),
            "confidence": out.get("confidence"),
            "api_txt_path": out.get("api_txt_path"),
            "final_date": out.get("final_date"),
        }
    except FileNotFoundError as e:
        return _detect_error_response(str(e))
    except ValueError as e:
        return _detect_error_response(str(e))
    except Exception as e:
        return _detect_error_response(f"Inference failed: {e!s}")
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


@palm_router.post("/detect-with-date")
async def detect_with_date(file: UploadFile = File(...)):
    """
    Same as POST /palm/detect, but always runs date identification on the first
    detected box (ROI hex + day_from_hex). Returns final_date and api_txt_path
    even when the classification is not "ripe".
    """
    if not file.filename or not file.filename.strip():
        return _detect_error_response("No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return _detect_error_response(
            f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    tmp_path = None
    try:
        content = await file.read()
        # Keep same temp-file approach used by /detect for consistency.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        out = run_infer(tmp_path, always_run_date=True)
        if out.get("count", 0) > 0 and out.get("coordinates"):
            try:
                # Visual ROI popup is best-effort: failure should not break API response.
                box = out["coordinates"][0]
                show_roi_popup(
                    tmp_path,
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    scale=_ROI_SCALE_FOR_DATE,
                    wait_ms=2000,
                )
            except Exception:
                pass
        return {
            "success": out["success"],
            "message": out["message"],
            "count": out["count"],
            "coordinates": out.get("coordinates", []),
            "class": out.get("class"),
            "confidence": out.get("confidence"),
            "api_txt_path": out.get("api_txt_path"),
            "final_date": out.get("final_date"),
        }
    except FileNotFoundError as e:
        return _detect_error_response(str(e))
    except ValueError as e:
        return _detect_error_response(str(e))
    except Exception as e:
        return _detect_error_response(f"Inference failed: {e!s}")
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


@palm_router.post("/detect-from-url")
async def detect_from_url(
    image_url: Optional[str] = Body(None, description="Direct URL of the image"),
    public_id: Optional[str] = Body(
        None,
        description="Cloudinary public_id (e.g. 'folder/filename'). Cloud name is read from CLOUDINARY_NAME in .env",
    ),
):
    """
    Fetch an image from a URL or from Cloudinary, then run the same detection pipeline as
    POST /palm/detect.

    **Option 1 – direct URL:** set `image_url` to any public image URL.

    **Option 2 – Cloudinary public_id:** set `public_id` (e.g. `oil-palm-bunches/fv3ab7eprfvjq2kyoo5z`).
    The cloud name is read from `CLOUDINARY_NAME` in the server `.env`.
    """
    if image_url and public_id:
        return _detect_error_response("Provide only one of: image_url, public_id.")
    if not image_url and not public_id:
        return _detect_error_response("Provide either image_url or public_id.")

    if public_id:
        try:
            image_url = _cloudinary_url(public_id)
        except ValueError as e:
            return _detect_error_response(str(e))

    url = (image_url or "").strip()
    if not url:
        return _detect_error_response("image_url is empty.")
    if not url.startswith(("http://", "https://")):
        return _detect_error_response("image_url must be http or https.")

    tmp_path = None
    try:
        # Download remote image first, then run the same local-file inference flow.
        content, suffix = await _fetch_image_bytes(url)
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            suffix = DEFAULT_IMAGE_SUFFIX
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        out = run_infer(tmp_path)
        return {
            "success": out["success"],
            "message": out["message"],
            "count": out["count"],
            "coordinates": out.get("coordinates", []),
            "class": out.get("class"),
            "confidence": out.get("confidence"),
            "api_txt_path": out.get("api_txt_path"),
            "final_date": out.get("final_date"),
        }
    except ValueError as e:
        return _detect_error_response(str(e))
    except FileNotFoundError as e:
        return _detect_error_response(str(e))
    except Exception as e:
        return _detect_error_response(f"Inference failed: {e!s}")
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
