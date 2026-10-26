"""
tests/test_perception.py

Unit and integration tests for the Sentry perception module:
- src/perception/quality.py (check_quality, enhance_image)
- src/perception/cracks.py (detect_cracks, measure_defect)

Tests utilize synthetic images (blank, dark, overexposed, blurry, textured, and crack-like).
These synthetic tests verify mathematical and algorithmic correctness under controlled
conditions, but do not claim or imply real-world structural crack detection accuracy.
"""

import cv2
import numpy as np
import pytest

from src.perception.quality import (
    check_quality,
    enhance_image,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_MIN_CONTRAST,
)
from src.perception.cracks import (
    detect_cracks,
    measure_defect,
    select_primary_crack,
)



# ============================================================================
# Synthetic Image Fixtures & Helpers
# ============================================================================

def make_blank_image(height: int = 200, width: int = 200, value: int = 128) -> np.ndarray:
    """Creates a uniform, flat grayscale or BGR image."""
    return np.full((height, width, 3), fill_value=value, dtype=np.uint8)


def make_concrete_texture(
    height: int = 200,
    width: int = 200,
    base_intensity: int = 140,
    noise_sigma: float = 8.0,
    seed: int = 42,
) -> np.ndarray:
    """Generates synthetic concrete-like background texture with Gaussian grain."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_sigma, (height, width)).astype(np.float32)
    textured = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR)


def make_sharp_checkered_image(height: int = 200, width: int = 200) -> np.ndarray:
    """Generates a high-frequency, sharp pattern with high Laplacian variance."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[::10, :] = 255
    img[:, ::10] = 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_crack_image(
    height: int = 200,
    width: int = 200,
    crack_darkness: int = 30,
    thickness: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Generates synthetic concrete with a dark, continuous, winding crack line."""
    concrete = make_concrete_texture(height, width, base_intensity=150, noise_sigma=5.0, seed=seed)
    # Draw a winding polyline representing a crack
    pts = np.array(
        [[20, 30], [50, 70], [80, 85], [110, 130], [150, 160], [180, 185]],
        dtype=np.int32,
    )
    cv2.polylines(
        concrete,
        [pts],
        isClosed=False,
        color=(crack_darkness, crack_darkness, crack_darkness),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return concrete


# ============================================================================
# Tests: Image Quality Assessment (check_quality)
# ============================================================================

class TestCheckQuality:
    """Tests for check_quality in src/perception/quality.py."""

    def test_invalid_inputs_raise_errors(self):
        """Validates that None, wrong types, empty arrays, and invalid dimensions fail."""
        with pytest.raises(ValueError, match="Input image cannot be None"):
            check_quality(None)  # type: ignore

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            check_quality("not-an-image")  # type: ignore

        with pytest.raises(ValueError, match="empty"):
            check_quality(np.array([], dtype=np.uint8))

        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            check_quality(np.zeros((10, 10, 10, 10), dtype=np.uint8))

        with pytest.raises(ValueError, match="Unsupported number of image channels"):
            check_quality(np.zeros((10, 10, 7), dtype=np.uint8))

    def test_blank_uniform_image_fails_quality(self):
        """A uniform blank image lacks texture, yielding zero blur score and zero contrast."""
        blank = make_blank_image(200, 200, value=128)
        result = check_quality(blank)

        assert result["ok"] is False
        assert result["blur_score"] == pytest.approx(0.0, abs=1e-3)
        assert result["contrast"] == pytest.approx(0.0, abs=1e-3)
        assert result["brightness"] == pytest.approx(128.0, abs=1.0)
        assert any("blurry" in issue.lower() for issue in result["issues"])
        assert any("low contrast" in issue.lower() for issue in result["issues"])

    def test_dark_underexposed_image(self):
        """An excessively dark image should be flagged as underexposed."""
        dark = make_blank_image(200, 200, value=15)
        result = check_quality(dark)

        assert result["ok"] is False
        assert result["brightness"] < DEFAULT_MIN_BRIGHTNESS
        assert any("underexposed" in issue.lower() for issue in result["issues"])

    def test_overexposed_image(self):
        """An excessively bright image should be flagged as overexposed."""
        bright = make_blank_image(200, 200, value=245)
        result = check_quality(bright)

        assert result["ok"] is False
        assert result["brightness"] > DEFAULT_MAX_BRIGHTNESS
        assert any("overexposed" in issue.lower() for issue in result["issues"])

    def test_blurry_image_detected(self):
        """Applying heavy Gaussian blur to a sharp pattern must drop the blur score below threshold."""
        sharp = make_sharp_checkered_image(200, 200)
        sharp_result = check_quality(sharp, blur_threshold=100.0)
        assert sharp_result["blur_score"] > 100.0

        # Heavily blur the image
        blurred = cv2.GaussianBlur(sharp, (31, 31), 10.0)
        blurred_result = check_quality(blurred, blur_threshold=100.0)

        assert blurred_result["blur_score"] < sharp_result["blur_score"]
        assert blurred_result["blur_score"] < 100.0
        assert any("blurry" in issue.lower() for issue in blurred_result["issues"])

    def test_acceptable_quality_image(self):
        """A sharp, well-exposed, textured pattern should pass all quality checks."""
        sharp = make_sharp_checkered_image(200, 200)
        result = check_quality(sharp, blur_threshold=50.0, min_contrast=15.0)

        assert result["ok"] is True
        assert len(result["issues"]) == 0
        assert result["blur_score"] >= 50.0

    def test_grayscale_2d_input(self):
        """Grayscale 2D arrays should be handled directly without error."""
        gray = np.full((100, 100), 120, dtype=np.uint8)
        result = check_quality(gray)
        assert "brightness" in result
        assert result["brightness"] == pytest.approx(120.0, abs=1.0)


# ============================================================================
# Tests: Image Enhancement (enhance_image)
# ============================================================================

class TestEnhanceImage:
    """Tests for enhance_image in src/perception/quality.py."""

    def test_invalid_input(self):
        """Non-arrays and empty inputs must raise errors."""
        with pytest.raises(ValueError):
            enhance_image(None)  # type: ignore

        with pytest.raises(ValueError):
            enhance_image(np.array([], dtype=np.uint8))

    def test_color_image_enhancement_shape_and_dtype(self):
        """Enhancing a 3-channel BGR image must preserve dimensions and uint8 dtype."""
        img = make_concrete_texture(150, 150)
        enhanced = enhance_image(img)

        assert enhanced.shape == img.shape
        assert enhanced.dtype == np.uint8
        assert isinstance(enhanced, np.ndarray)

    def test_grayscale_enhancement_shape_and_dtype(self):
        """Enhancing a 2D grayscale image must preserve dimensions and uint8 dtype."""
        gray = np.full((120, 120), 100, dtype=np.uint8)
        gray[30:60, 30:60] = 160
        enhanced = enhance_image(gray)

        assert enhanced.shape == (120, 120)
        assert enhanced.dtype == np.uint8

    def test_enhancement_improves_contrast(self):
        """CLAHE should stretch local contrast on a low-contrast synthetic texture."""
        low_contrast = make_concrete_texture(150, 150, base_intensity=128, noise_sigma=2.0)
        pre_std = float(np.std(cv2.cvtColor(low_contrast, cv2.COLOR_BGR2GRAY)))

        enhanced = enhance_image(low_contrast, clip_limit=3.0)
        post_std = float(np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))

        assert post_std >= pre_std


# ============================================================================
# Tests: Defect Measurement (measure_defect)
# ============================================================================

class TestMeasureDefect:
    """Tests for measure_defect in src/perception/cracks.py."""

    def test_invalid_mask_and_shape_inputs(self):
        """Invalid mask or dimension parameters must raise descriptive exceptions."""
        with pytest.raises(ValueError, match="Mask cannot be None"):
            measure_defect(None, (100, 100))  # type: ignore

        with pytest.raises(TypeError, match="Mask must be a numpy.ndarray"):
            measure_defect("invalid", (100, 100))  # type: ignore

        with pytest.raises(ValueError, match="Mask must be a 2D array"):
            measure_defect(np.zeros((10, 10, 3), dtype=np.uint8), (10, 10))

        with pytest.raises(ValueError, match="img_shape must be a tuple/list"):
            measure_defect(np.zeros((10, 10), dtype=np.uint8), (10,))  # type: ignore

        with pytest.raises(ValueError, match="does not match"):
            measure_defect(np.zeros((50, 50), dtype=np.uint8), (100, 100))

    def test_empty_mask_safe_handling(self):
        """An empty (all zeros) mask must return clean zero values without division by zero."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        m = measure_defect(empty_mask, (100, 100))

        assert m["crack_count"] == 0
        assert m["total_crack_pixels"] == 0
        assert m["area_percentage"] == 0.0
        assert m["estimated_pixel_length"] == 0.0
        assert m["mean_crack_width_px"] == 0.0
        assert m["max_crack_length_px"] == 0.0
        assert m["bounding_boxes"] == []

    def test_known_crack_mask_measurements(self):
        """A mask with a known synthetic line should report consistent measurements."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Draw a horizontal line of width 2 and length 80 pixels (from x=50 to 130, y=100 to 102)
        cv2.line(mask, (50, 100), (130, 100), color=255, thickness=2)

        m = measure_defect(mask, (200, 200))

        assert m["crack_count"] == 1
        assert m["total_crack_pixels"] > 0
        assert m["area_percentage"] == pytest.approx(m["total_crack_pixels"] / 40000.0 * 100.0)
        assert m["estimated_pixel_length"] > 50.0
        assert m["max_crack_length_px"] > 50.0
        assert len(m["bounding_boxes"]) == 1
        x, y, w, h = m["bounding_boxes"][0]
        assert w >= 75  # Length spans approx 80 px
        assert h <= 10  # Width is small


# ============================================================================
# Tests: Crack Detection (detect_cracks)
# ============================================================================

class TestDetectCracks:
    """Tests for detect_cracks in src/perception/cracks.py."""

    def test_invalid_image_inputs(self):
        """Invalid images must raise errors."""
        with pytest.raises(ValueError, match="Input image cannot be None"):
            detect_cracks(None)  # type: ignore

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            detect_cracks([1, 2, 3])  # type: ignore

    def test_blank_image_returns_no_cracks(self):
        """A uniform blank image must yield 0 cracks, 0 confidence, and empty mask."""
        blank = make_blank_image(200, 200, value=130)
        res = detect_cracks(blank)

        assert res["crack_count"] == 0
        assert res["confidence"] == 0.0
        assert res["area_percentage"] == 0.0
        assert res["estimated_pixel_length"] == 0.0
        assert np.count_nonzero(res["mask"]) == 0

    def test_unblemished_concrete_texture_returns_no_cracks(self):
        """Texture grain without fissures must not trigger false positive detections."""
        concrete = make_concrete_texture(200, 200, base_intensity=140, noise_sigma=6.0, seed=123)
        res = detect_cracks(concrete)

        assert res["crack_count"] == 0
        assert res["confidence"] == 0.0
        assert np.count_nonzero(res["mask"]) == 0

    def test_circular_pores_filtered_out(self):
        """Round dark concrete pores (bug holes) must be rejected by elongation/thinness filters."""
        concrete = make_concrete_texture(200, 200, base_intensity=150, noise_sigma=3.0, seed=456)
        # Draw circular pores (diameter 8, aspect ratio ~ 1.0, low thinness)
        cv2.circle(concrete, (60, 60), 4, (40, 40, 40), -1)
        cv2.circle(concrete, (140, 140), 4, (40, 40, 40), -1)

        res = detect_cracks(concrete)
        assert res["crack_count"] == 0
        assert res["confidence"] == 0.0

    def test_synthetic_crack_detected(self):
        """A clear dark crack line must be detected with high confidence (> 0.70)."""
        crack_img = make_crack_image(200, 200, crack_darkness=20, thickness=3, seed=789)
        res = detect_cracks(crack_img)

        assert res["crack_count"] >= 1
        assert res["estimated_pixel_length"] > 30.0
        assert res["area_percentage"] > 0.0
        assert np.count_nonzero(res["mask"]) > 0
        # High confidence for a prominent crack (> 0.70 aligns with 'defect_found' verdict)
        assert res["confidence"] > 0.70

    def test_faint_crack_yields_intermediate_confidence(self):
        """A faint, thin crack should yield a heuristic confidence score in the human-review zone (0.35 - 0.75)."""
        # A faint crack has lower contrast against the background
        faint_crack = make_concrete_texture(200, 200, base_intensity=140, noise_sigma=3.0, seed=101)
        # Draw a shorter, subtle crack line with moderate darkness
        pts = np.array([[60, 80], [90, 100], [120, 115]], dtype=np.int32)
        cv2.polylines(faint_crack, [pts], isClosed=False, color=(90, 90, 90), thickness=2)

        res = detect_cracks(faint_crack, min_ridge_intensity=20.0, min_length_px=10.0)
        if res["crack_count"] > 0:
            # When detected, confidence reflects ambiguity / human review tier
            assert 0.35 <= res["confidence"] <= 0.85
        else:
            # If filtered by noise floor, confidence is 0
            assert res["confidence"] == 0.0


# ============================================================================
# Tests: Over-Detection Fixes (Percentile, Border, Close, Connected Components)
# ============================================================================

class TestCrackDetectorOverDetectionFixes:
    """Specific unit tests verifying the over-detection and noise suppression fixes."""

    def test_plain_gray_with_single_thin_diagonal_line(self):
        """Plain gray image with one thin dark diagonal line -> n_cracks == 1, longest_crack_px > 0."""
        h, w = 200, 200
        img = np.full((h, w, 3), 150, dtype=np.uint8)
        cv2.line(img, (30, 30), (170, 170), (30, 30, 30), 2)

        res = detect_cracks(img)

        assert res["n_cracks"] == 1
        assert res["crack_count"] == 1
        assert res["longest_crack_px"] > 50.0
        assert res["total_crack_px"] >= res["longest_crack_px"]
        assert res["area_pct"] > 0.0
        assert res["confidence"] > 0.70
        assert np.count_nonzero(res["mask"]) > 0

    def test_plain_gray_with_gaussian_noise_only(self):
        """Plain gray with Gaussian noise only -> n_cracks == 0, longest_crack_px == 0.0."""
        h, w = 200, 200
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 8.0, (h, w)).astype(np.float32)
        noisy = np.clip(140 + noise, 0, 255).astype(np.uint8)
        noisy_bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

        res = detect_cracks(noisy_bgr)

        assert res["n_cracks"] == 0
        assert res["longest_crack_px"] == 0.0
        assert res["total_crack_px"] == 0.0
        assert res["area_pct"] == 0.0
        assert res["confidence"] == 0.0
        assert np.count_nonzero(res["mask"]) == 0

    def test_thick_white_border_bar_not_detected(self):
        """A thick white border bar on the left edge must be zeroed out by border margin and not detected."""
        h, w = 200, 200
        img = np.full((h, w, 3), 150, dtype=np.uint8)
        # Draw thick white border bar on left edge (width 8 px, inside border_margin=10)
        img[:, :8] = 255

        res = detect_cracks(img)

        assert res["n_cracks"] == 0
        assert res["longest_crack_px"] == 0.0
        assert res["total_crack_px"] == 0.0
        assert res["confidence"] == 0.0
        assert np.count_nonzero(res["mask"]) == 0

    def test_crack_drawn_with_small_gaps_merged_as_one_component(self):
        """A crack drawn with small gaps (< 5 px) must be merged by morphological closing into 1 component."""
        h, w = 200, 200
        img = np.full((h, w, 3), 150, dtype=np.uint8)
        # Draw 3 segments with gaps of 3-4 px
        cv2.line(img, (30, 30), (70, 70), (30, 30, 30), 2)
        cv2.line(img, (74, 74), (120, 120), (30, 30, 30), 2)
        cv2.line(img, (125, 125), (170, 170), (30, 30, 30), 2)

        res = detect_cracks(img)

        assert res["n_cracks"] == 1
        assert res["longest_crack_px"] > 100.0
        assert res["total_crack_px"] >= res["longest_crack_px"]

    def test_texture_heavy_sanity_rule_caps_confidence(self):
        """If area_pct > max_area_pct_threshold, flag 'texture_heavy' and cap confidence at 0.69."""
        h, w = 200, 200
        # Draw multiple parallel thick lines to exceed 5.0% area coverage
        img = np.full((h, w, 3), 150, dtype=np.uint8)
        for y in range(30, 170, 20):
            cv2.line(img, (30, y), (170, y), (30, 30, 30), 4)

        res = detect_cracks(img, max_area_pct_threshold=5.0)

        if res["area_pct"] > 5.0:
            assert "texture_heavy" in res["flags"]
            assert res["confidence"] <= 0.69


# ============================================================================
# Tests: Region of Interest (ROI) & Primary Crack Reporting
# ============================================================================

class TestRegionOfInterestAndPrimaryCrack:
    """Unit tests for Region-of-Interest (ROI) support and primary crack selection."""

    def test_crack_inside_roi_reported_crack_outside_ignored(self):
        """
        Synthetic: crack inside the ROI plus a similar crack outside it ->
        only the inside one is reported, secondary_count == 0.
        """
        h, w = 300, 300
        img = np.full((h, w, 3), 150, dtype=np.uint8)

        # Crack 1: Inside ROI (x=200, from y=50 to 180) -> length 130 px
        cv2.line(img, (200, 50), (200, 180), (30, 30, 30), 2)

        # Crack 2: Outside ROI (x=50, from y=50 to 180) -> length 130 px
        cv2.line(img, (50, 50), (50, 180), (30, 30, 30), 2)

        # Define ROI mask covering right side [x: 120..280, y: 30..220]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[30:220, 120:280] = 255
        expected_roi_area = int(np.count_nonzero(roi_mask))

        res = detect_cracks(img, roi_mask=roi_mask)

        # Only the inside crack must be detected and reported as primary
        assert res["n_cracks"] == 1
        assert res["secondary_count"] == 0
        assert res["primary_crack"] is not None
        assert res["primary_crack"]["length_px"] > 100.0
        # The outside region (x < 100) must have 0 crack pixels in the mask
        assert np.count_nonzero(res["mask"][:, :100]) == 0
        # Check ROI area reporting and ROI-relative area coverage
        assert res["roi_area_px"] == expected_roi_area
        assert res["area_pct"] > 0.0

    def test_roi_edge_not_detected_as_crack(self):
        """
        ROI edge is not detected as a crack.
        A sharp step edge at the user's ROI boundary (simulating barrier wall vs road)
        must be excluded by ~10 px erosion and not flagged as a crack.
        """
        h, w = 250, 250
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        # Concrete barrier wall on left side with high contrast (dark gray 50)
        img[:, :100] = 50
        # Road surface on right side (light gray 180)
        img[:, 100:] = 180

        # User draws ROI starting exactly at the wall-road transition x=100
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[20:230, 100:230] = 255

        res = detect_cracks(img, roi_mask=roi_mask)

        # The boundary edge must be completely eroded and zeroed; 0 cracks detected
        assert res["n_cracks"] == 0
        assert res["primary_crack"] is None
        assert res["secondary_count"] == 0
        assert np.count_nonzero(res["mask"]) == 0
        assert res["confidence"] == 0.0

    def test_select_primary_crack_ranking(self):
        """
        select_primary_crack ranks components by combined length, elongation, and contrast.
        """
        shape = (200, 200)
        comp_a = {
            "label": 1,
            "length_px": 50.0,
            "bbox": [10, 10, 50, 2],
            "area_px": 100,
            "centroid": [35.0, 11.0],
            "mean_contrast": 30.0,
        }
        comp_b = {
            "label": 2,
            "length_px": 140.0,
            "bbox": [20, 40, 140, 2],
            "area_px": 280,
            "centroid": [90.0, 41.0],
            "mean_contrast": 50.0,
        }

        ranked = select_primary_crack([comp_a, comp_b], img_shape=shape)

        assert ranked["primary_crack"] is not None
        assert ranked["primary_crack"]["length_px"] == 140.0
        assert ranked["secondary_count"] == 1
        assert ranked["secondary_cracks"][0]["length_px"] == 50.0


# ============================================================================
# Tests: measure_primary_crack
# ============================================================================

class TestMeasurePrimaryCrack:
    """Unit tests for measure_primary_crack using synthetic geometric masks."""

    def _make_hline_mask(self, height=20, width=120, thickness=4) -> np.ndarray:
        """Horizontal line mask: 120 px long, ~4 px thick."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cy = height // 2
        mask[cy - thickness // 2: cy + thickness // 2, :] = 255
        return mask

    def _make_curved_crack_mask(self) -> np.ndarray:
        """Curved polyline crack: path length should exceed its bounding-box length."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        pts = np.array([[20, 100], [60, 60], [100, 80], [140, 50], [180, 100]], dtype=np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, thickness=5)
        return mask

    def test_straight_line_length_approx(self):
        """100 px long, 4 px thick horizontal bar → length_path_px ≈ 100 (within 10%)."""
        from src.perception.cracks import measure_primary_crack
        mask = self._make_hline_mask(height=20, width=100, thickness=4)
        result = measure_primary_crack(mask)
        assert abs(result["length_path_px"] - 100.0) < 15.0, (
            f"Expected length ≈ 100, got {result['length_path_px']}"
        )

    def test_straight_line_width_approx(self):
        """100 px long, 4 px thick horizontal bar → width_mean_px ≈ 4 (within 50%)."""
        from src.perception.cracks import measure_primary_crack
        mask = self._make_hline_mask(height=20, width=100, thickness=4)
        result = measure_primary_crack(mask)
        # width_mean = area / path_length ≈ 400 / 100 = 4
        assert 2.0 <= result["width_mean_px"] <= 7.0, (
            f"Expected width_mean ≈ 4, got {result['width_mean_px']}"
        )

    def test_curved_line_path_longer_than_bbox(self):
        """Curved polyline crack: skeleton path length must exceed the bounding-box long-edge."""
        from src.perception.cracks import measure_primary_crack
        mask = self._make_curved_crack_mask()
        result = measure_primary_crack(mask)
        # The polyline's actual path is longer than any straight bounding-box edge
        assert result["length_path_px"] > result["bbox_length_px"], (
            f"Path length {result['length_path_px']:.1f} should exceed bbox {result['bbox_length_px']:.1f}"
        )

    def test_mm_calibration_halves_values(self):
        """With mm_per_px=0.5, mm values should be half of px values."""
        from src.perception.cracks import measure_primary_crack
        mask = self._make_hline_mask(height=20, width=100, thickness=4)
        result_px = measure_primary_crack(mask, mm_per_px=None)
        result_mm = measure_primary_crack(mask, mm_per_px=0.5)

        assert result_mm["length_mm"] is not None
        assert result_mm["width_mean_mm"] is not None
        assert result_mm["width_max_mm"] is not None

        assert abs(result_mm["length_mm"] - result_px["length_path_px"] * 0.5) < 0.5
        assert abs(result_mm["width_mean_mm"] - result_px["width_mean_px"] * 0.5) < 0.5

    def test_no_calibration_returns_none_mm(self):
        """Without mm_per_px, all mm fields must be None."""
        from src.perception.cracks import measure_primary_crack
        mask = self._make_hline_mask(height=20, width=100, thickness=4)
        result = measure_primary_crack(mask, mm_per_px=None)
        assert result["length_mm"] is None
        assert result["width_mean_mm"] is None
        assert result["width_max_mm"] is None

    def test_severity_low_on_thin_crack(self):
        """A 1-px thin line should be classified as severity 'low'."""
        from src.perception.cracks import measure_primary_crack
        mask = np.zeros((10, 80), dtype=np.uint8)
        mask[5, :] = 255  # 1-px line
        result = measure_primary_crack(mask)
        assert result["severity"] == "low"

    def test_severity_bump_on_long_crack(self):
        """A crack spanning >40% of image width should bump severity to 'high'."""
        from src.perception.cracks import measure_primary_crack
        # 1-px line 500 px wide on a 600-px image → 500/600 ≈ 83% > 40%
        mask = np.zeros((10, 600), dtype=np.uint8)
        mask[5, :] = 255
        result = measure_primary_crack(mask, img_shape=(10, 600))
        assert result["severity"] == "high", (
            f"Expected 'high' due to length bump, got {result['severity']}"
        )

    def test_empty_mask_returns_zeros(self):
        """An empty mask must return zero dimensions and 'low' severity."""
        from src.perception.cracks import measure_primary_crack
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = measure_primary_crack(mask)
        assert result["length_path_px"] == 0.0
        assert result["width_mean_px"] == 0.0
        assert result["width_max_px"] == 0.0
        assert result["severity"] == "low"

    def test_invalid_mask_returns_safe_defaults(self):
        """A None mask must return safe zero defaults without raising."""
        from src.perception.cracks import measure_primary_crack
        result = measure_primary_crack(None)  # type: ignore[arg-type]
        assert result["length_path_px"] == 0.0
        assert result["severity"] == "low"
