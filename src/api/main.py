"""
src/api/main.py

FastAPI backend application for the Sentry surface-defect inspection system.
Provides REST endpoints for service health verification and autonomous
concrete crack inspection from uploaded photographs.

Infrastructure additions (do not affect verdict logic):
- JSON-structured logging to stdout (request_id, verdict, processing_time_ms).
  Image bytes, tokens, and credentials are NEVER logged.
- Per-IP rate limiting via RATE_LIMIT_PER_MINUTE env var (default 20 req/min).
- Optional S3 persistence via S3_BUCKET env var. Uses the default AWS credential
  chain (EC2 IAM role). Failures only warn; verdict is never changed.
"""

import base64
import collections
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent.planner import inspect
from src.agent.trace import make_json_safe

# ---------------------------------------------------------------------------
# JSON-structured logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for stdout ingestion."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra fields passed via LogRecord extras
        for key in ("request_id", "verdict", "processing_time_ms", "remote_ip", "path"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger("sentry.api")

# ---------------------------------------------------------------------------
# Per-IP rate limiter (sliding-window, in-memory)
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))


class _RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._lock = threading.Lock()
        # ip -> deque of epoch timestamps (float)
        self._windows: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque()
        )

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            dq = self._windows[ip]
            # Evict timestamps older than 60 s
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


_rate_limiter = _RateLimiter(RATE_LIMIT_PER_MINUTE)


async def _rate_limit_check(request: Request) -> None:
    """FastAPI dependency: raises 429 when the per-IP limit is exceeded."""
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min per IP). Retry after 60 s.",
        )

# ---------------------------------------------------------------------------
# S3 persistence helper
# ---------------------------------------------------------------------------

S3_BUCKET: Optional[str] = os.environ.get("S3_BUCKET") or None


def _upload_to_s3(
    request_id: str,
    result_payload: Dict[str, Any],
    overlay_png_bytes: Optional[bytes],
) -> None:
    """
    Saves the inspection result JSON and overlay PNG to S3 under a
    request-scoped prefix.  Uses the default boto3 credential chain
    (EC2 IAM role, ~/.aws, env vars).  Credentials are NEVER logged.

    This function MUST NOT raise; all errors are warnings only.
    """
    if not S3_BUCKET:
        return
    try:
        import boto3  # lazy import — not required in non-S3 deployments
        s3 = boto3.client("s3")  # uses instance profile / env credential chain
        prefix = f"inspections/{request_id}"

        # Save verdict + metrics JSON (mask bytes are already base64 in payload)
        json_key = f"{prefix}/result.json"
        # Strip mask bytes from S3 copy to keep objects small; they are
        # already returned to the caller over HTTP.
        s3_payload = {k: v for k, v in result_payload.items()
                      if k not in ("mask_base64", "mask_primary_base64")}
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=json_key,
            Body=json.dumps(s3_payload).encode(),
            ContentType="application/json",
        )

        # Save overlay PNG if available
        if overlay_png_bytes:
            png_key = f"{prefix}/overlay.png"
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=png_key,
                Body=overlay_png_bytes,
                ContentType="image/png",
            )

        logger.info(
            "S3 upload complete",
            extra={"request_id": request_id, "path": f"s3://{S3_BUCKET}/{prefix}"},
        )
    except Exception as exc:  # noqa: BLE001
        # S3 failures must NEVER change the verdict
        logger.warning(
            "S3 upload failed (non-fatal): %s",
            type(exc).__name__,
            extra={"request_id": request_id},
        )


# Configuration constants
MAX_FILE_SIZE_BYTES: int = int(os.environ.get("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)
SUPPORTED_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # Fallback for some clients
)

app = FastAPI(
    title="Sentry — Concrete Surface-Defect Inspection API",
    description=(
        "Autonomous perception-decision-action inspection service for concrete cracks. "
        "Strictly operates on OpenCV 5.x and returns one of four discrete verdicts: "
        "'defect_found', 'no_defect', 'recapture_requested', 'human_review'."
    ),
    version="1.0.0",
)

# Enable CORS for web dashboards and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request-ID injection + JSON access logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _logging_middleware(request: Request, call_next: Callable) -> Any:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    ip = request.client.host if request.client else "unknown"
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "remote_ip": ip,
            "path": request.url.path,
            "processing_time_ms": elapsed_ms,
        },
    )
    response.headers["X-Request-Id"] = request_id
    return response



@app.get(
    "/health",
    summary="Service Health Check",
    response_description="Active service status and OpenCV runtime version.",
)
def get_health() -> Dict[str, Any]:
    """
    Returns the operational status of the inspection service along with
    the actively loaded OpenCV version.
    """
    return {
        "status": "healthy",
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "app_name": "Sentry Inspection API",
        "version": "1.0.0",
    }


def _encode_mask_to_base64(mask: Optional[np.ndarray]) -> Optional[str]:
    """
    Safely encodes a 2D uint8 binary mask array into a base64-encoded PNG string.
    Ensures that raw NumPy arrays are never serialized directly into JSON payloads.
    """
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return None

    try:
        # Ensure mask is 2D uint8
        binary_mask = (mask > 0).astype(np.uint8) * 255
        success, encoded_buf = cv2.imencode(".png", binary_mask)
        if not success:
            return None
        b64_str = base64.b64encode(encoded_buf.tobytes()).decode("utf-8")
        return b64_str
    except Exception:
        return None


@app.post(
    "/inspect",
    summary="Inspect Concrete Surface Photograph",
    response_description="Inspection verdict, confidence, metrics, audited trace, and masks.",
    dependencies=[Depends(_rate_limit_check)],
)
async def inspect_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Uploaded concrete photograph file"),
    roi: Optional[str] = Form(
        None,
        description="Optional ROI polygon points as JSON string, e.g. '[[x1, y1], [x2, y2], [x3, y3], ...]'",
    ),
    scale_ref: Optional[str] = Form(
        None,
        description=(
            'Optional scale reference as JSON: {"p1": [x1, y1], "p2": [x2, y2], "real_mm": <float>}. '
            "Points must be inside image bounds, distinct, and real_mm must be > 0."
        ),
    ),
    auto_roi: bool = Form(
        True,
        description="Enable automatic surface Region of Interest detection (default: True)",
    ),
) -> JSONResponse:
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_start = time.perf_counter()
    """
    Uploads a concrete surface photograph for autonomous inspection.

    Performs rigorous file validation, image decoding, optional ROI polygon validation,
    runs the perception-decision-action agent loop, and returns a structured JSON payload
    with base64 mask encoding.
    """
    # 1. Validate filename and extension
    filename = file.filename or "unknown"
    has_valid_ext = any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)

    if not has_valid_ext:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": (
                    f"Unsupported file format for '{filename}'. Supported extensions are: "
                    f"{', '.join(SUPPORTED_EXTENSIONS)}."
                ),
            },
        )

    # 2. Read file contents and check payload size
    try:
        file_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": f"Failed to read uploaded file payload: {str(exc)}",
            },
        )

    if len(file_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": "Uploaded file is empty (0 bytes).",
            },
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": (
                    f"Uploaded file size ({len(file_bytes)} bytes) exceeds the maximum allowed limit "
                    f"of {MAX_FILE_SIZE_BYTES} bytes ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."
                ),
            },
        )

    # 3. Decode image with OpenCV
    try:
        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": f"Corrupted image data could not be decoded: {str(exc)}",
            },
        )

    if img is None or img.size == 0 or img.ndim not in (2, 3):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "verdict": "recapture_requested",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": "Failed to decode valid image array from the uploaded file.",
            },
        )

    height, width = img.shape[:2]

    # 4. Validate optional ROI polygon
    roi_mask = None
    if roi is not None and roi.strip() != "":
        try:
            roi_points = json.loads(roi)
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "status": "error",
                    "verdict": "human_review",
                    "confidence": None,
                    "mask_base64": None,
                    "mask_primary_base64": None,
                    "metrics": {},
                    "trace": [],
                    "error": f"Invalid ROI polygon format: Must be valid JSON array of points. Details: {str(exc)}",
                },
            )

        if not isinstance(roi_points, list):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "status": "error",
                    "verdict": "human_review",
                    "confidence": None,
                    "mask_base64": None,
                    "mask_primary_base64": None,
                    "metrics": {},
                    "trace": [],
                    "error": "Invalid ROI: Polygon must be a list of [x, y] coordinates.",
                },
            )

        if len(roi_points) < 3:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "status": "error",
                    "verdict": "human_review",
                    "confidence": None,
                    "mask_base64": None,
                    "mask_primary_base64": None,
                    "metrics": {},
                    "trace": [],
                    "error": f"Invalid ROI polygon: Must contain at least 3 points, got {len(roi_points)}.",
                },
            )

        parsed_pts = []
        for idx, pt in enumerate(roi_points):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    content={
                        "status": "error",
                        "verdict": "human_review",
                        "confidence": None,
                        "mask_base64": None,
                        "mask_primary_base64": None,
                        "metrics": {},
                        "trace": [],
                        "error": f"Invalid ROI point at index {idx}: Expected [x, y] pair, got {pt}.",
                    },
                )
            x, y = pt[0], pt[1]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    content={
                        "status": "error",
                        "verdict": "human_review",
                        "confidence": None,
                        "mask_base64": None,
                        "mask_primary_base64": None,
                        "metrics": {},
                        "trace": [],
                        "error": f"Invalid ROI coordinates at index {idx}: Expected numeric values, got [{x}, {y}].",
                    },
                )
            if x < 0 or x > width or y < 0 or y > height:
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    content={
                        "status": "error",
                        "verdict": "human_review",
                        "confidence": None,
                        "mask_base64": None,
                        "mask_primary_base64": None,
                        "metrics": {},
                        "trace": [],
                        "error": (
                            f"Invalid ROI point [{x}, {y}] at index {idx}: Point is out of bounds for image dimensions "
                            f"(width={width}, height={height})."
                        ),
                    },
                )
            parsed_pts.append([int(round(x)), int(round(y))])

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        poly_pts = np.array(parsed_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(roi_mask, [poly_pts], 255)

    # 4b. Validate optional scale reference and compute mm_per_px
    mm_per_px: Optional[float] = None
    if scale_ref is not None and scale_ref.strip() != "":
        try:
            scale_data = json.loads(scale_ref)
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "status": "error",
                    "verdict": "human_review",
                    "confidence": None,
                    "mask_base64": None,
                    "mask_primary_base64": None,
                    "metrics": {},
                    "trace": [],
                    "error": f"Invalid scale_ref format: must be valid JSON. Details: {str(exc)}",
                },
            )

        _scale_error: Optional[str] = None
        try:
            p1 = scale_data["p1"]
            p2 = scale_data["p2"]
            real_mm = float(scale_data["real_mm"])
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            _scale_error = f'scale_ref must have keys "p1": [x,y], "p2": [x,y], "real_mm": float. Details: {str(exc)}'

        if _scale_error is None and (x1 < 0 or x1 > width or y1 < 0 or y1 > height):
            _scale_error = (
                f"scale_ref p1=[{x1}, {y1}] is outside image bounds "
                f"(width={width}, height={height})."
            )
        if _scale_error is None and (x2 < 0 or x2 > width or y2 < 0 or y2 > height):
            _scale_error = (
                f"scale_ref p2=[{x2}, {y2}] is outside image bounds "
                f"(width={width}, height={height})."
            )
        if _scale_error is None and real_mm <= 0:
            _scale_error = f"scale_ref real_mm must be > 0, got {real_mm}."

        if _scale_error is None:
            pixel_dist = float(np.hypot(x2 - x1, y2 - y1))
            if pixel_dist <= 0:
                _scale_error = "scale_ref points are coincident (pixel distance = 0)."
            else:
                mm_per_px = real_mm / pixel_dist

        if _scale_error is not None:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "status": "error",
                    "verdict": "human_review",
                    "confidence": None,
                    "mask_base64": None,
                    "mask_primary_base64": None,
                    "metrics": {},
                    "trace": [],
                    "error": _scale_error,
                },
            )

    # 5. Execute autonomous agent planner
    try:
        if roi_mask is not None:
            result = inspect(img, roi_mask=roi_mask, auto_roi=False, mm_per_px=mm_per_px)
        else:
            result = inspect(img, auto_roi=auto_roi, mm_per_px=mm_per_px)
    except Exception as exc:

        # Internal planner failure: never return no_defect
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "verdict": "human_review",
                "confidence": None,
                "mask_base64": None,
                "mask_primary_base64": None,
                "metrics": {},
                "trace": [],
                "error": f"Unexpected internal error during inspection: {str(exc)}",
            },
        )

    # 6. Extract results and encode masks
    verdict = result["verdict"]
    mask_b64 = _encode_mask_to_base64(result.get("mask"))
    mask_primary_b64 = _encode_mask_to_base64(result.get("mask_primary"))
    roi_mask_b64 = _encode_mask_to_base64(result.get("roi_mask"))
    metrics_safe = make_json_safe(result.get("metrics", {}))
    trace_safe = make_json_safe(result.get("trace", []))
    roi_safe = make_json_safe(result.get("roi_info"))
    err = result.get("error")

    confidence = None
    primary_crack_payload: Optional[Dict[str, Any]] = None
    if isinstance(metrics_safe, dict):
        crack_det = metrics_safe.get("crack_detection")
        if isinstance(crack_det, dict):
            confidence = crack_det.get("confidence")
            primary_crack_payload = crack_det.get("primary_crack")

    processing_ms = round((time.perf_counter() - t_start) * 1000, 1)

    response_payload: Dict[str, Any] = {
        "status": "success" if err is None else "completed_with_notes",
        "verdict": verdict,
        "confidence": confidence,
        "mm_per_px": mm_per_px,
        "primary_crack": primary_crack_payload,
        "mask_base64": mask_b64,
        "mask_primary_base64": mask_primary_b64,
        "roi_mask_base64": roi_mask_b64,
        "roi": roi_safe,
        "metrics": metrics_safe,
        "trace": trace_safe,
        "error": err,
        "request_id": request_id,
        "processing_time_ms": processing_ms,
    }

    # 7. Structured verdict log (no image bytes, no credentials)
    logger.info(
        "inspection_complete",
        extra={
            "request_id": request_id,
            "verdict": verdict,
            "processing_time_ms": processing_ms,
        },
    )

    # 8. Optional S3 persistence (non-fatal)
    if S3_BUCKET:
        # Build a small overlay PNG (primary mask in red over grey thumb) for S3
        overlay_png: Optional[bytes] = None
        primary_mask = result.get("mask_primary")
        if primary_mask is not None and isinstance(primary_mask, np.ndarray):
            try:
                thumb = cv2.cvtColor(
                    cv2.resize(img, (320, 240)), cv2.COLOR_BGR2GRAY
                )
                overlay_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                mask_resized = cv2.resize(primary_mask, (320, 240))
                overlay_bgr[mask_resized > 0] = (0, 0, 200)
                _, buf = cv2.imencode(".png", overlay_bgr)
                overlay_png = buf.tobytes()
            except Exception:
                overlay_png = None

        _upload_to_s3(request_id, response_payload, overlay_png)

    return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)
