"""
tests/test_api.py

Integration and unit tests for the Sentry FastAPI service (src/api/main.py).
Tests the /health endpoint, multipart image upload validation, image decoding,
response schemas, base64 mask generation, JSON safety, and error handling.
"""

import base64
import json
from typing import Any, Dict
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from tests.test_perception import (
    make_blank_image,
    make_crack_image,
    make_sharp_checkered_image,
)

client = TestClient(app)


def _encode_to_png_bytes(img: np.ndarray) -> bytes:
    """Encodes an image array into PNG byte sequence."""
    success, buf = cv2.imencode(".png", img)
    assert success, "Failed to encode image to PNG format"
    return buf.tobytes()


# ============================================================================
# Tests: /health Endpoint
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_status_and_opencv_version(self):
        """Health check must return 200, healthy status, and OpenCV 5.x version."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "opencv_version" in data
        assert str(data["opencv_version"]).startswith("5.")
        assert data["app_name"] == "Sentry Inspection API"
        assert data["version"] == "1.0.0"


# ============================================================================
# Tests: /inspect Endpoint
# ============================================================================

class TestInspectEndpoint:
    """Tests for POST /inspect."""

    def test_inspect_valid_crack_image(self):
        """A valid image with a prominent crack must return 200, 'defect_found', and a base64 mask."""
        crack_img = make_crack_image(200, 200, crack_darkness=15, thickness=3, seed=42)
        png_bytes = _encode_to_png_bytes(crack_img)

        response = client.post(
            "/inspect",
            files={"file": ("concrete_crack.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["status"] == "success"
        assert payload["verdict"] == "defect_found"
        assert isinstance(payload["confidence"], float)
        assert payload["confidence"] > 0.70
        assert payload["error"] is None

        # Verify mask is base64 encoded and decodes to non-empty mask
        mask_b64 = payload["mask_base64"]
        assert mask_b64 is not None
        assert isinstance(mask_b64, str)
        decoded_mask = cv2.imdecode(
            np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        assert decoded_mask is not None
        assert np.count_nonzero(decoded_mask) > 0

        # Verify trace structure
        assert isinstance(payload["trace"], list)
        assert len(payload["trace"]) >= 2
        for step in payload["trace"]:
            assert "step" in step
            assert "tool" in step
            assert "decision" in step
            assert "reason" in step
            assert "timestamp" in step

    def test_inspect_valid_clean_image_no_defect(self):
        """A valid clean concrete surface must return 200 and 'no_defect'."""
        clean_img = make_sharp_checkered_image(200, 200)
        png_bytes = _encode_to_png_bytes(clean_img)

        response = client.post(
            "/inspect",
            files={"file": ("clean_slab.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["status"] == "success"
        assert payload["verdict"] == "no_defect"
        assert payload["confidence"] is not None
        assert payload["confidence"] < 0.35

    def test_inspect_blurry_image_requests_recapture(self):
        """A blurry/blank image that cannot be enhanced must return 'recapture_requested'."""
        blank_img = make_blank_image(200, 200, value=128)
        png_bytes = _encode_to_png_bytes(blank_img)

        response = client.post(
            "/inspect",
            files={"file": ("blurry_surface.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["verdict"] == "recapture_requested"
        assert payload["mask_base64"] is None

    def test_unsupported_file_extension_rejected(self):
        """Files with non-image extensions (.txt, .exe, .pdf) must be rejected with 400 Bad Request."""
        response = client.post(
            "/inspect",
            files={"file": ("report.txt", b"structural inspection notes", "text/plain")},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "error"
        assert payload["verdict"] == "recapture_requested"
        assert "Unsupported file format" in payload["error"]

    def test_empty_file_rejected(self):
        """Zero-byte uploaded files must be rejected with 400 Bad Request."""
        response = client.post(
            "/inspect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "error"
        assert payload["verdict"] == "recapture_requested"
        assert "empty" in payload["error"].lower()

    def test_corrupted_image_data_rejected(self):
        """Files that cannot be decoded into an image array must return 400 Bad Request."""
        corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupted-junk-binary-data"
        response = client.post(
            "/inspect",
            files={"file": ("damaged.png", corrupt_bytes, "image/png")},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "error"
        assert payload["verdict"] == "recapture_requested"
        assert "decode" in payload["error"].lower()

    def test_oversized_file_rejected(self, monkeypatch):
        """Uploading a payload exceeding the file size limit must return 413 Payload Too Large."""
        # Temporarily lower the size limit for deterministic testing
        monkeypatch.setattr("src.api.main.MAX_FILE_SIZE_BYTES", 1024)  # 1 KB limit

        large_payload = b"X" * 2048  # 2 KB
        response = client.post(
            "/inspect",
            files={"file": ("large_image.png", large_payload, "image/png")},
        )
        assert response.status_code == 413
        payload = response.json()
        assert payload["status"] == "error"
        assert "exceeds the maximum allowed limit" in payload["error"]

    def test_internal_pipeline_crash_never_returns_no_defect(self, monkeypatch):
        """If an internal exception occurs during inspection, return 500 and human_review."""
        def mock_crash(img, **kwargs):
            raise RuntimeError("Simulated segmentation failure in internal model")

        monkeypatch.setattr("src.api.main.inspect", mock_crash)

        crack_img = make_crack_image(200, 200)
        png_bytes = _encode_to_png_bytes(crack_img)

        response = client.post(
            "/inspect",
            files={"file": ("error_test.png", png_bytes, "image/png")},
        )
        assert response.status_code == 500
        payload = response.json()
        assert payload["status"] == "error"
        assert payload["verdict"] == "human_review"
        assert payload["verdict"] != "no_defect"
        assert "Simulated segmentation failure" in payload["error"]

    def test_api_response_is_strictly_json_safe(self):
        """Verifies that the entire response payload is strictly native Python types and serializable."""
        crack_img = make_crack_image(200, 200, seed=77)
        png_bytes = _encode_to_png_bytes(crack_img)

        response = client.post(
            "/inspect",
            files={"file": ("json_safety_test.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

        data = response.json()
        # json.dumps must execute without error (no non-serializable NumPy types)
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

        # Confirm no raw array string representation in metrics or trace
        assert "ndarray" not in serialized

    def test_invalid_polygon_two_points_returns_422(self):
        """Invalid polygon with fewer than 3 points must return HTTP 422 Unprocessable Entity."""
        clean_img = make_sharp_checkered_image(200, 200)
        png_bytes = _encode_to_png_bytes(clean_img)

        invalid_roi = json.dumps([[10, 10], [50, 50]])
        response = client.post(
            "/inspect",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"roi": invalid_roi},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["status"] == "error"
        assert "at least 3 points" in payload["error"].lower()

    def test_invalid_polygon_out_of_bounds_returns_422(self):
        """Invalid polygon with points outside image bounds must return HTTP 422 Unprocessable Entity."""
        clean_img = make_sharp_checkered_image(200, 200)
        png_bytes = _encode_to_png_bytes(clean_img)

        out_of_bounds_roi = json.dumps([[10, 10], [350, 50], [50, 150]])
        response = client.post(
            "/inspect",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"roi": out_of_bounds_roi},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["status"] == "error"
        assert "out of bounds" in payload["error"].lower()

    def test_valid_roi_polygon_accepted_and_applied(self):
        """A valid ROI polygon in multipart form data must return 200 and filter results to the ROI."""
        crack_img = make_crack_image(200, 200, crack_darkness=15, thickness=3, seed=42)
        png_bytes = _encode_to_png_bytes(crack_img)

        valid_roi = json.dumps([[10, 10], [190, 10], [190, 190], [10, 190]])
        response = client.post(
            "/inspect",
            files={"file": ("test_roi.png", png_bytes, "image/png")},
            data={"roi": valid_roi},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert "mask_primary_base64" in payload
        assert payload["metrics"]["crack_detection"]["roi_area_px"] > 0


# ============================================================================
# Tests: scale_ref Parameter on /inspect
# ============================================================================

class TestScaleRefValidation:
    """Tests for scale_ref validation and mm calibration via POST /inspect."""

    def _png_bytes(self) -> bytes:
        img = make_crack_image(200, 200, crack_darkness=15, thickness=3, seed=42)
        success, buf = cv2.imencode(".png", img)
        assert success
        return buf.tobytes()

    def test_coincident_points_returns_422(self):
        """scale_ref with p1 == p2 (zero pixel distance) must return 422."""
        bad_ref = json.dumps({"p1": [50, 50], "p2": [50, 50], "real_mm": 100.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422
        payload = response.json()
        assert "coincident" in payload["error"].lower() or "distance" in payload["error"].lower()

    def test_p1_out_of_bounds_returns_422(self):
        """scale_ref with p1 outside image bounds must return 422."""
        bad_ref = json.dumps({"p1": [-10, 50], "p2": [100, 50], "real_mm": 100.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422
        assert "bounds" in response.json()["error"].lower()

    def test_p2_out_of_bounds_returns_422(self):
        """scale_ref with p2 outside image bounds must return 422."""
        bad_ref = json.dumps({"p1": [10, 10], "p2": [9999, 9999], "real_mm": 100.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422
        assert "bounds" in response.json()["error"].lower()

    def test_real_mm_zero_returns_422(self):
        """scale_ref with real_mm of zero must return 422."""
        bad_ref = json.dumps({"p1": [10, 10], "p2": [100, 10], "real_mm": 0.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422
        assert "real_mm" in response.json()["error"].lower()

    def test_real_mm_negative_returns_422(self):
        """scale_ref with negative real_mm must return 422."""
        bad_ref = json.dumps({"p1": [10, 10], "p2": [100, 10], "real_mm": -5.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422

    def test_invalid_json_scale_ref_returns_422(self):
        """Malformed JSON in scale_ref must return 422."""
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": "{not valid json}"})
        assert response.status_code == 422

    def test_missing_key_returns_422(self):
        """scale_ref missing 'real_mm' key must return 422."""
        bad_ref = json.dumps({"p1": [10, 10], "p2": [100, 10]})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": bad_ref})
        assert response.status_code == 422

    def test_valid_scale_ref_returns_200_with_mm_per_px(self):
        """A valid scale_ref must return 200 with mm_per_px in the response payload."""
        # 100 px apart, 50 mm -> 0.5 mm/px
        valid_ref = json.dumps({"p1": [10, 100], "p2": [110, 100], "real_mm": 50.0})
        response = client.post("/inspect", files={"file": ("img.png", self._png_bytes(), "image/png")}, data={"scale_ref": valid_ref})
        assert response.status_code == 200
        payload = response.json()
        assert payload.get("mm_per_px") is not None
        assert abs(payload["mm_per_px"] - 0.5) < 0.001
