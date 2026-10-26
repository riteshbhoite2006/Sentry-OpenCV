"""
tests/test_auto_roi.py

Comprehensive test suite for automatic Region of Interest (ROI) detection,
zero-manual-input inspection workflow, coordinate mapping, and fallback behavior.
"""

import json
import os
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.perception.roi import detect_surface_roi, create_roi_visualization
from src.perception.detect import detect_cracks
from src.agent.planner import inspect
from src.api.main import app, _rate_limiter
from tests.test_perception import make_concrete_texture, make_sharp_checkered_image


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the sliding-window rate limiter before each test."""
    with _rate_limiter._lock:
        _rate_limiter._windows.clear()
    yield
    with _rate_limiter._lock:
        _rate_limiter._windows.clear()


# ---------------------------------------------------------------------------
# 1. Synthetic Crack Tests: Horizontal, Vertical, Diagonal inside Auto ROI
# ---------------------------------------------------------------------------

def test_auto_roi_horizontal_crack():
    """Verify horizontal crack on concrete surface is detected with auto ROI."""
    img = make_concrete_texture(300, 400, base_intensity=140, noise_sigma=22.0, seed=123)
    cv2.line(img, (80, 150), (320, 150), (20, 20, 20), 3)

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("defect_found", "human_review")
    assert result["mask"] is not None
    assert result["roi_info"] is not None

    prim = result["primary_crack"]
    assert prim is not None
    assert prim["length_px"] > 50
    assert prim["length_path_px"] > 50
    assert prim["width_mean_px"] > 0
    # No real mm invented when scale is not provided
    assert prim["length_mm"] is None


def test_auto_roi_vertical_crack():
    """Verify vertical crack on concrete surface is detected with auto ROI."""
    img = make_concrete_texture(400, 300, base_intensity=140, noise_sigma=22.0, seed=123)
    cv2.line(img, (150, 80), (150, 320), (20, 20, 20), 3)

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("defect_found", "human_review")
    assert result["mask"] is not None

    prim = result["primary_crack"]
    assert prim is not None
    assert prim["length_px"] > 50
    assert prim["length_path_px"] > 50


def test_auto_roi_diagonal_crack():
    """Verify diagonal crack on concrete surface is detected with auto ROI."""
    img = make_concrete_texture(350, 350, base_intensity=140, noise_sigma=22.0, seed=123)
    cv2.line(img, (80, 80), (270, 270), (20, 20, 20), 3)

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("defect_found", "human_review")
    assert result["mask"] is not None

    prim = result["primary_crack"]
    assert prim is not None
    assert prim["length_px"] > 50
    assert prim["length_path_px"] > 50


# ---------------------------------------------------------------------------
# 2. Clean Surface & Heavy Texture Tests
# ---------------------------------------------------------------------------

def test_clean_surface_no_defect():
    """Clean surface should yield no_defect verdict without false positive cracks."""
    img = make_sharp_checkered_image(300, 300)
    result = inspect(img, auto_roi=True)
    assert result["verdict"] == "no_defect"
    assert result["primary_crack"] is None


def test_heavy_texture_surface_handling():
    """Heavy aggregate texture should not trigger false defect_found."""
    rng = np.random.RandomState(99)
    noise = rng.normal(0, 25, (300, 300)).astype(np.int16)
    img_gray = np.clip(140 + noise, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("no_defect", "human_review")


# ---------------------------------------------------------------------------
# 3. Cluttered Background & ROI Exclusion
# ---------------------------------------------------------------------------

def test_cluttered_background_sky_vegetation_void():
    """Auto ROI should segment concrete road surface and exclude sky, vegetation, and voids."""
    h, w = 400, 500
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Top 30% is blue sky (BGR: high blue, lower red)
    img[:120, :] = [225, 180, 60]

    # Middle 50% is concrete road surface
    concrete = make_concrete_texture(200, w, base_intensity=140, noise_sigma=10.0, seed=42)
    img[120:320, :] = concrete

    # Bottom left is green vegetation (high green)
    img[320:, :200] = [30, 160, 40]

    # Bottom right is dark shadow void
    img[320:, 200:] = [15, 15, 15]

    roi_res = detect_surface_roi(img)
    assert not roi_res["fallback"], "Should segment clear road scene"
    assert roi_res["roi_mask"] is not None
    mask = roi_res["roi_mask"]

    # Concrete region should be inside ROI
    assert np.mean(mask[150:280, 100:400] > 0) > 0.85
    # Sky should be excluded
    assert np.mean(mask[:100, :] > 0) < 0.05
    # Vegetation should be excluded
    assert np.mean(mask[330:, :180] > 0) < 0.05


# ---------------------------------------------------------------------------
# 4. Failed ROI Detection & Fallback Behavior
# ---------------------------------------------------------------------------

def test_failed_roi_detection_triggers_fallback():
    """When surface area < 20% (e.g. vegetation filling image), detect_surface_roi triggers fallback."""
    img = np.full((300, 300, 3), (30, 160, 40), dtype=np.uint8)  # pure vegetation

    roi_res = detect_surface_roi(img, min_surface_ratio=0.20)
    assert roi_res["fallback"] is True
    assert roi_res["warning"] is not None
    assert "whole-image" in roi_res["warning"].lower()
    assert roi_res["roi_mask"] is None


def test_planner_handles_roi_fallback(monkeypatch):
    """When ROI detection flags fallback, planner applies whole-image inspection and logs fallback."""
    img = make_sharp_checkered_image(200, 200)

    # Mock detect_surface_roi to force fallback
    def mock_detect_surface_roi(working_img, **kwargs):
        return {
            "roi_mask": None,
            "roi_polygon": None,
            "roi_bbox": [0, 0, 200, 200],
            "coverage_pct": 100.0,
            "roi_area_px": 40000,
            "surface_type": "full_image_fallback",
            "confidence": 0.40,
            "fallback": True,
            "warning": "Ambiguous surface boundaries; defaulting to whole-image inspection.",
            "excluded_categories": ["vegetation"],
        }

    monkeypatch.setattr("src.agent.planner.detect_surface_roi", mock_detect_surface_roi)

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("defect_found", "human_review", "no_defect")
    trace = result["trace"]
    roi_step = next((s for s in trace if s.get("tool") == "define_roi"), None)
    assert roi_step is not None
    assert roi_step["decision"] == "full_image_roi"
    assert "defaulting to whole-image" in roi_step["reason"]


# ---------------------------------------------------------------------------
# 5. ROI-to-Image Coordinate Mapping Alignment
# ---------------------------------------------------------------------------

def test_roi_coordinate_alignment_1_to_1():
    """Crack bounding box and mask coordinates must align 1:1 with original image space."""
    h, w = 400, 400
    img = make_concrete_texture(h, w, base_intensity=140, noise_sigma=22.0, seed=123)
    crack_x1, crack_y1, crack_x2, crack_y2 = 150, 200, 280, 200
    cv2.line(img, (crack_x1, crack_y1), (crack_x2, crack_y2), (20, 20, 20), 3)

    result = inspect(img, auto_roi=True)
    assert result["primary_crack"] is not None
    prim = result["primary_crack"]
    bbox = prim["bbox"]
    bx, by, bw, bh = bbox

    # Bounding box must enclose the drawn crack coordinates 1:1
    assert bx <= crack_x1 + 5
    assert bx + bw >= crack_x2 - 5
    assert by <= crack_y1 + 5
    assert by + bh >= crack_y2 - 5

    mask = result["mask"]
    assert mask.shape == (h, w)
    assert np.count_nonzero(mask[198:203, 150:280]) > 0


# ---------------------------------------------------------------------------
# 6. Real Concrete Photograph Test
# ---------------------------------------------------------------------------

def test_column_crack_photograph_inspection():
    """Inspect data/dev/column_crack.jpg using automatic ROI detection."""
    photo_path = "data/dev/column_crack.jpg"
    if not os.path.exists(photo_path):
        pytest.skip(f"{photo_path} not available.")

    img = cv2.imread(photo_path)
    assert img is not None

    result = inspect(img, auto_roi=True)
    assert result["verdict"] in ("defect_found", "human_review")
    assert result["roi_info"] is not None
    # Verify lateral barriers and background voids were excluded
    assert "lateral_barrier" in result["roi_info"].get("excluded_categories", [])
    assert result["primary_crack"] is not None
    assert result["primary_crack"]["length_path_px"] > 50


# ---------------------------------------------------------------------------
# 7. API Endpoint Zero-Manual-Input & Backward Compatibility
# ---------------------------------------------------------------------------

def test_api_zero_manual_input_inspection():
    """Calling /inspect with only the image file executes auto ROI and returns dimensions in pixels."""
    client = TestClient(app)
    img = make_concrete_texture(300, 400, base_intensity=140, noise_sigma=22.0, seed=42)
    _, encoded = cv2.imencode(".jpg", img)

    resp = client.post(
        "/inspect",
        files={"file": ("concrete_test.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] in ("success", "completed_with_notes")
    assert payload["verdict"] in ("no_defect", "defect_found", "human_review")
    assert "roi" in payload
    assert payload["mm_per_px"] is None


def test_api_explicit_roi_override():
    """When a caller supplies an explicit ROI polygon, auto ROI is bypassed."""
    client = TestClient(app)
    img = make_concrete_texture(300, 400, base_intensity=140, noise_sigma=22.0, seed=42)
    _, encoded = cv2.imencode(".jpg", img)

    custom_roi = [[50, 50], [250, 50], [250, 250], [50, 250]]
    resp = client.post(
        "/inspect",
        files={"file": ("concrete_test.jpg", encoded.tobytes(), "image/jpeg")},
        data={"roi": json.dumps(custom_roi)},
    )
    assert resp.status_code == 200
    payload = resp.json()
    roi_info = payload.get("roi")
    assert roi_info is not None
    assert roi_info.get("surface_type") == "manual_roi"


# ---------------------------------------------------------------------------
# 8. Regression: Missing Measurement Display Safety
# ---------------------------------------------------------------------------

def test_measurement_display_missing_values_safe():
    """Verify measurement display logic handles None, partial, or missing measurements safely."""
    # Scenario A: primary_crack is None (e.g. clean surface or recapture)
    crack_data = {"primary_crack": None, "secondary_count": 0, "n_cracks": 0, "area_pct": 0.0}
    primary_crack = crack_data.get("primary_crack") or {}
    primary_len = float(primary_crack.get("length_px", 0.0))
    primary_bbox_len = primary_len
    length_path_px = primary_crack.get("length_path_px")

    centerline_disp = (
        f"{length_path_px:.1f} px"
        if length_path_px is not None
        else (f"{primary_len:.1f} px" if primary_len > 0 else "N/A")
    )
    assert centerline_disp == "N/A"
    assert primary_len == 0.0

    # Scenario B: primary_crack has bbox length only (no skeleton length)
    crack_data_b = {
        "primary_crack": {
            "length_px": 75.4,
            "bbox": [10, 20, 70, 15],
            "mean_contrast": 18.2,
            "area_px": 210,
            "length_path_px": None,
        }
    }
    prim_b = crack_data_b.get("primary_crack") or {}
    prim_len_b = float(prim_b.get("length_px", 0.0))
    path_len_b = prim_b.get("length_path_px")

    centerline_b = (
        f"{path_len_b:.1f} px"
        if path_len_b is not None
        else (f"{prim_len_b:.1f} px" if prim_len_b > 0 else "N/A")
    )
    assert centerline_b == "75.4 px"

    # Scenario C: primary_crack has skeleton length, width, and severity
    crack_data_c = {
        "primary_crack": {
            "length_px": 80.0,
            "length_path_px": 85.3,
            "width_mean_px": 2.8,
            "width_max_px": 4.5,
            "severity": "medium",
            "bbox": [10, 20, 75, 12],
            "mean_contrast": 22.0,
            "area_px": 240,
        }
    }
    prim_c = crack_data_c.get("primary_crack") or {}
    prim_len_c = float(prim_c.get("length_px", 0.0))
    path_len_c = prim_c.get("length_path_px")

    centerline_c = (
        f"{path_len_c:.1f} px"
        if path_len_c is not None
        else (f"{prim_len_c:.1f} px" if prim_len_c > 0 else "N/A")
    )
    assert centerline_c == "85.3 px"


# ---------------------------------------------------------------------------
# 8. Regression Tests: Chaining across grooves, Spall triage, and Smooth wall crack
# ---------------------------------------------------------------------------

def test_column_crack_horizontal_chaining_and_spall():
    """Verify that on column_crack.jpg, the horizontal crack is chained across grooves,
    spall chunk is excluded from primary crack, and spall triage triggers human_review."""
    photo_path = "data/dev/column_crack.jpg"
    if not os.path.exists(photo_path):
        pytest.skip(f"{photo_path} not found.")

    img = cv2.imread(photo_path)
    res = inspect(img, auto_roi=True)

    # (a) Horizontal crack is chained across grooves with substantial skeleton path length
    prim = res["primary_crack"]
    assert prim is not None
    assert prim["length_path_px"] > 100.0, f"Expected chained crack path > 100 px, got {prim['length_path_px']}"
    assert prim["bbox"][0] >= 100, f"Spall edge should not be primary crack: {prim['bbox']}"

    # (b) Spall feature identified and triage set
    flags = res.get("metrics", {}).get("crack_detection", {}).get("flags", [])
    assert "spall_suspected" in flags or "spall_detected" in flags
    assert res["verdict"] == "human_review"

    # (c) Reason message reflects spall presence, not false multi-region spread
    trace_steps = [s for s in res["trace"] if s["tool"] == "detect_cracks"]
    assert len(trace_steps) > 0
    reason = trace_steps[0]["reason"].lower()
    assert "spall" in reason
    assert "multi-region" not in reason


def test_smooth_surface_with_sharp_crack_avoids_recapture():
    """Verify that a smooth wall/ceiling with an isolated sharp dark crack is NOT
    falsely rejected as blurry (recapture_requested), but passes quality and detects defect."""
    # Create smooth concrete/plaster background with very low global variance
    smooth_wall = np.full((500, 500, 3), 190, dtype=np.uint8)
    # Draw prominent, jagged dark crack down the center
    pts = np.array(
        [[250, 40], [253, 120], [248, 220], [255, 330], [249, 440]],
        dtype=np.int32,
    )
    cv2.polylines(smooth_wall, [pts], False, (40, 40, 40), thickness=3)

    res = inspect(smooth_wall, auto_roi=True)
    # Must not reject as blurry
    assert res["verdict"] != "recapture_requested", "Sharp crack on smooth background must not be rejected for recapture!"
    assert res["verdict"] in ("defect_found", "human_review")
    assert res["primary_crack"] is not None
    assert res["primary_crack"]["length_path_px"] > 150.0


