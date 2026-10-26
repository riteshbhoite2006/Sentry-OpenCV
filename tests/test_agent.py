"""
tests/test_agent.py

Unit and integration tests for the Sentry agent module:
- src/agent/trace.py (make_json_safe, TraceStep, InspectionTrace)
- src/agent/planner.py (inspect, decision sequencing, confidence triage)

Validates the perception -> decision -> action loop, strict 6-step ceiling,
audited JSON trace structure, confidence boundary triage, and safe failure handling.
"""

import json
from typing import Any, Dict
import cv2
import numpy as np
import pytest

from src.agent.planner import (
    CONFIDENCE_HUMAN_REVIEW_UPPER_BOUND,
    CONFIDENCE_NO_DEFECT_UPPER_BOUND,
    MAX_AGENT_STEPS,
    inspect,
)
from src.agent.trace import InspectionTrace, TraceStep, make_json_safe
from tests.test_perception import (
    make_blank_image,
    make_concrete_texture,
    make_crack_image,
    make_sharp_checkered_image,
)


# ============================================================================
# Tests: Trace Helpers & JSON Serialization (trace.py)
# ============================================================================

class TestTraceHelpers:
    """Tests for make_json_safe, TraceStep, and InspectionTrace."""

    def test_make_json_safe_primitives(self):
        """Standard NumPy numbers and booleans must convert to native Python types."""
        assert isinstance(make_json_safe(np.int64(42)), int)
        assert isinstance(make_json_safe(np.float32(3.14)), float)
        assert isinstance(make_json_safe(np.bool_(True)), bool)
        assert make_json_safe(None) is None
        assert make_json_safe("text") == "text"

    def test_make_json_safe_excludes_large_image_arrays(self):
        """Large NumPy arrays (images/masks) must summarize to metadata rather than full dumps."""
        large_arr = np.zeros((100, 100), dtype=np.uint8)
        large_arr[10:20, 10:20] = 255
        safe = make_json_safe(large_arr)

        assert isinstance(safe, dict)
        assert safe["shape"] == [100, 100]
        assert safe["dtype"] == "uint8"
        assert safe["non_zero_count"] == 100

    def test_make_json_safe_small_collections(self):
        """Small arrays or lists of bounding boxes should serialize cleanly to lists."""
        bbox = np.array([10, 20, 30, 40], dtype=np.int32)
        safe = make_json_safe(bbox)
        assert safe == [10, 20, 30, 40]

    def test_trace_step_structure_and_json(self):
        """A TraceStep must expose all required fields and serialize to valid JSON."""
        trace = InspectionTrace(max_steps=6)
        step = trace.add_step(
            tool="check_quality",
            inputs_and_metrics={"blur_score": np.float64(142.5), "ok": np.bool_(True)},
            decision="quality_passed",
            reason="Image quality passed.",
        )

        assert step.step == 1
        assert step.tool == "check_quality"
        assert isinstance(step.inputs_and_metrics["blur_score"], float)
        assert step.decision == "quality_passed"
        assert "timestamp" in step.to_dict()

        # Ensure valid JSON string serialization
        json_str = trace.to_json()
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["step"] == 1

    def test_trace_max_steps_limit_enforced(self):
        """Attempting to exceed max_steps must raise a RuntimeError."""
        trace = InspectionTrace(max_steps=3)
        for i in range(3):
            trace.add_step(tool="tool", inputs_and_metrics={}, decision="ok", reason=f"step {i+1}")

        assert len(trace) == 3
        with pytest.raises(RuntimeError, match="step limit exceeded"):
            trace.add_step(tool="tool", inputs_and_metrics={}, decision="overflow", reason="step 4")


# ============================================================================
# Tests: Agent Planner Execution (inspect)
# ============================================================================

class TestAgentPlanner:
    """Tests for the inspect function in src/agent/planner.py."""

    def test_clear_synthetic_crack_detected(self):
        """A sharp image with an obvious concrete crack must yield 'defect_found'."""
        crack_img = make_crack_image(200, 200, crack_darkness=20, thickness=3, seed=12)
        res = inspect(crack_img)

        assert res["verdict"] == "defect_found"
        assert res["mask"] is not None
        assert np.count_nonzero(res["mask"]) > 0
        assert res["metrics"]["crack_detection"]["confidence"] > CONFIDENCE_HUMAN_REVIEW_UPPER_BOUND
        assert res["error"] is None
        assert 2 <= len(res["trace"]) <= MAX_AGENT_STEPS

        # Check trace structure
        tools_called = [step["tool"] for step in res["trace"]]
        assert "check_quality" in tools_called
        assert "detect_cracks" in tools_called

    def test_unblemished_concrete_yields_no_defect(self):
        """A textured but defect-free concrete photograph must yield 'no_defect'."""
        clean_img = make_sharp_checkered_image(200, 200)
        res = inspect(clean_img)

        assert res["verdict"] == "no_defect"
        assert res["metrics"]["crack_detection"]["confidence"] < CONFIDENCE_NO_DEFECT_UPPER_BOUND
        assert res["error"] is None
        assert len(res["trace"]) <= MAX_AGENT_STEPS

    def test_poor_quality_requests_recapture_after_enhancement_fails(self):
        """A pitch-black or uniform blank image cannot be recovered by CLAHE and must request recapture."""
        black_img = make_blank_image(200, 200, value=5)
        res = inspect(black_img)

        assert res["verdict"] == "recapture_requested"
        assert res["mask"] is None
        assert res["error"] is None
        # Should record: check_quality -> enhance_image -> check_quality
        assert len(res["trace"]) == 3
        assert res["trace"][0]["decision"] == "quality_failed"
        assert res["trace"][1]["decision"] == "image_enhanced"
        assert res["trace"][2]["decision"] == "recapture_requested"

    def test_quality_recovery_after_enhancement(self, monkeypatch):
        """
        Tests the recovery branch where an image initially fails quality check,
        is enhanced, passes quality reassessment, and proceeds to crack detection.
        """
        original_check_quality = __import__("src.perception.quality", fromlist=["check_quality"]).check_quality
        call_count = {"count": 0}

        def mock_check_quality(img, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # First call fails on contrast
                return {
                    "ok": False,
                    "blur_score": 150.0,
                    "brightness": 120.0,
                    "contrast": 15.0,
                    "issues": ["Low contrast"],
                }
            # Second call (after enhancement) passes
            return {
                "ok": True,
                "blur_score": 180.0,
                "brightness": 125.0,
                "contrast": 35.0,
                "issues": [],
            }

        monkeypatch.setattr("src.agent.planner.check_quality", mock_check_quality)

        test_img = make_sharp_checkered_image(200, 200)
        res = inspect(test_img)

        # Expected flow: check_quality (failed) -> enhance_image -> check_quality (passed) -> define_roi -> detect_cracks -> measure_primary_crack
        assert len(res["trace"]) == 6
        assert res["trace"][0]["decision"] == "quality_failed"
        assert res["trace"][1]["decision"] == "image_enhanced"
        assert res["trace"][2]["decision"] == "quality_passed_after_enhancement"
        assert res["trace"][3]["tool"] == "define_roi"
        assert res["trace"][4]["tool"] == "detect_cracks"
        assert res["trace"][5]["tool"] == "measure_primary_crack"
        assert res["verdict"] in ("no_defect", "human_review", "defect_found")


    @pytest.mark.parametrize(
        "mock_conf, expected_verdict",
        [
            (0.00, "no_defect"),
            (0.34, "no_defect"),
            (0.35, "human_review"),   # Exact lower boundary for human review
            (0.50, "human_review"),   # Mid-range ambiguous
            (0.70, "human_review"),   # Exact upper boundary for human review
            (0.71, "defect_found"),   # Just above upper threshold
            (0.95, "defect_found"),
        ],
    )
    def test_confidence_threshold_boundaries(self, monkeypatch, mock_conf: float, expected_verdict: str):
        """Deterministic boundary tests for all three triage regions and exact edges."""
        def mock_detect(img, **kwargs):
            return {
                "mask": np.zeros((10, 10), dtype=np.uint8),
                "crack_count": 1 if mock_conf >= 0.35 else 0,
                "estimated_pixel_length": 50.0 if mock_conf >= 0.35 else 0.0,
                "area_percentage": 1.5 if mock_conf >= 0.35 else 0.0,
                "confidence": mock_conf,
                "measurements": {
                    "crack_count": 1 if mock_conf >= 0.35 else 0,
                    "total_crack_pixels": 150,
                    "area_percentage": 1.5,
                    "estimated_pixel_length": 50.0,
                    "mean_crack_width_px": 3.0,
                    "max_crack_length_px": 50.0,
                    "bounding_boxes": [[0, 0, 10, 10]],
                },
            }

        monkeypatch.setattr("src.agent.planner.detect_cracks", mock_detect)

        valid_img = make_sharp_checkered_image(200, 200)
        res = inspect(valid_img)

        assert res["verdict"] == expected_verdict
        assert res["metrics"]["crack_detection"]["confidence"] == pytest.approx(mock_conf)

    def test_trace_structure_and_json_safety(self):
        """Verifies that all trace steps adhere to the specification and serialize cleanly."""
        img = make_crack_image(200, 200, seed=99)
        res = inspect(img)

        trace = res["trace"]
        assert isinstance(trace, list)
        assert len(trace) > 0

        for idx, step in enumerate(trace, start=1):
            assert step["step"] == idx
            assert isinstance(step["tool"], str)
            assert isinstance(step["inputs_and_metrics"], dict)
            assert isinstance(step["decision"], str)
            assert isinstance(step["reason"], str)
            assert isinstance(step["timestamp"], str)

        # Confirm JSON serialization succeeds without raising errors
        dumped = json.dumps(res["trace"])
        assert isinstance(dumped, str)

        # Confirm full image or mask is not in trace steps
        assert "shape" not in str(trace) or "255" not in str(trace)

    def test_six_step_maximum_never_exceeded(self):
        """Ensures that no code branch ever logs more than MAX_AGENT_STEPS (6)."""
        images_to_test = [
            make_crack_image(200, 200),
            make_blank_image(200, 200, value=0),
            make_sharp_checkered_image(200, 200),
        ]
        for img in images_to_test:
            res = inspect(img)
            assert len(res["trace"]) <= MAX_AGENT_STEPS

    def test_invalid_input_handling_never_returns_no_defect(self):
        """Invalid inputs (None, wrong types, empty arrays) must request recapture with error."""
        invalid_inputs = [
            None,
            "not_an_image",
            np.array([], dtype=np.uint8),
            np.zeros((10, 10, 10, 10), dtype=np.uint8),
        ]

        for invalid in invalid_inputs:
            res = inspect(invalid)
            assert res["verdict"] != "no_defect"
            assert res["verdict"] == "recapture_requested"
            assert res["error"] is not None
            assert len(res["trace"]) == 1
            assert res["trace"][0]["tool"] == "input_validation"

    def test_unexpected_runtime_exception_never_returns_no_defect(self, monkeypatch):
        """If an unexpected exception occurs, agent must refer to human_review, not no_defect."""
        def crash_detect(img, **kwargs):
            raise RuntimeError("Unexpected internal crash in detection library")

        monkeypatch.setattr("src.agent.planner.detect_cracks", crash_detect)

        valid_img = make_sharp_checkered_image(200, 200)
        res = inspect(valid_img)

        # Must never return no_defect on error
        assert res["verdict"] != "no_defect"
        assert res["verdict"] == "human_review"
        assert res["error"] is not None
        assert "internal crash" in res["error"].lower()
        # The crash step must be recorded in trace
        assert any("interrupted by unexpected error" in s["reason"] for s in res["trace"])

    def test_two_cracks_opposite_corners_without_roi_yields_human_review(self):
        """
        Synthetic: no ROI, two cracks in opposite corners -> verdict human_review with the "ROI needed" reason.
        When component centroids span more than 60% of image width and height, agent escalates to human_review.
        """
        h, w = 300, 300
        img = make_concrete_texture(h, w, noise_sigma=25.0)

        # Crack 1: Top-left corner (near (45, 45)) -> length ~ 70 px
        cv2.line(img, (20, 20), (70, 70), (10, 10, 10), 3)

        # Crack 2: Bottom-right corner (near (255, 255)) -> length ~ 70 px
        cv2.line(img, (230, 230), (280, 280), (10, 10, 10), 3)

        # Inspect without ROI
        res = inspect(img)

        # Both components span >60% width and >60% height
        assert res["verdict"] == "human_review"
        assert res["error"] is None

        # Confirm trace records "multiple separate regions found, ROI needed"
        detect_step = next(s for s in res["trace"] if s["tool"] == "detect_cracks")
        assert "multiple separate regions found, ROI needed" in detect_step["reason"]
        assert detect_step["inputs_and_metrics"]["multi_region_spread"] is True

        # Now test with ROI isolating only the first crack
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[10:100, 10:100] = 255
        res_roi = inspect(img, roi_mask=roi_mask)

        roi_detect_step = next(s for s in res_roi["trace"] if s["tool"] == "detect_cracks")
        assert "multiple separate regions found, ROI needed" not in roi_detect_step["reason"]
        assert res_roi["metrics"]["crack_detection"]["secondary_count"] == 0

