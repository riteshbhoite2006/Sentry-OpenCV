"""
src/perception/roi.py

Automatic Region of Interest (ROI) detection module for the Sentry concrete inspection system.
Identifies relevant concrete or road surfaces in uploaded photographs, excluding irrelevant
surroundings (sky, vegetation, barrier walls, deep shadows, peripheral borders).
Provides safe whole-image fallback with transparent diagnostics when boundaries are ambiguous.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def detect_surface_roi(
    img: np.ndarray,
    min_surface_ratio: float = 0.20,
    max_surface_ratio: float = 0.92,
) -> Dict[str, Any]:
    """
    Automatically detects the primary concrete or pavement surface in an image.

    Excludes:
    - Sky / highlight blowout (blue hue in HSV or extreme uniform highlight V >= 245)
    - Vegetation / foliage (green hue in HSV: 35-85 deg, saturation >= 35)
    - Deep shadow voids / background recess (luminance < 25)
    - Distant dark boundaries / non-structural peripheral frames

    Fallback:
    If a reliable surface boundary cannot be identified (e.g. uniform macro concrete,
    noisy scene, or candidate surface < min_surface_ratio), falls back to whole-image
    inspection with a visible warning to prevent missing critical defects.

    Parameters
    ----------
    img : np.ndarray
        Input 2D (grayscale) or 3D (BGR) image array.
    min_surface_ratio : float
        Minimum fraction of image area required for a candidate surface (default: 0.20).
    max_surface_ratio : float
        Upper fraction above which the image is classified as a full/macro surface (default: 0.92).

    Returns
    -------
    dict
        - 'roi_mask': Optional[np.ndarray], uint8 binary mask (255=surface, 0=excluded) or None on full fallback
        - 'roi_polygon': Optional[List[List[int]]], Polygon vertices [[x, y], ...] outlining the detected ROI
        - 'roi_bbox': Optional[List[int]], [x, y, w, h] bounding box
        - 'coverage_pct': float, Percentage of image covered by the ROI (0-100)
        - 'roi_area_px': int, Pixel area of the active ROI
        - 'surface_type': str ('segmented_surface', 'uniform_surface', 'full_image_fallback')
        - 'confidence': float, Confidence in the surface segmentation [0.0 .. 1.0]
        - 'fallback': bool, True if whole-image fallback was triggered
        - 'warning': Optional[str], Diagnostic warning if fallback occurred
        - 'excluded_categories': List[str], Surrounding features detected and excluded
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        return {
            "roi_mask": None,
            "roi_polygon": None,
            "roi_bbox": None,
            "coverage_pct": 100.0,
            "roi_area_px": 0,
            "surface_type": "full_image_fallback",
            "confidence": 0.0,
            "fallback": True,
            "warning": "Invalid image data provided for ROI detection.",
            "excluded_categories": [],
        }

    h, w = img.shape[:2]
    total_area = h * w

    # Standardize color representation
    if img.ndim == 2:
        gray = img.copy()
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    excluded_categories: List[str] = []

    # 1. Detect Sky / Atmospheric Blowout
    # Blue sky: H in [90..135], S >= 25, V >= 50
    blue_sky = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 50)
    # Bright overexposed blowout with low local texture: V >= 248, low gradient
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    low_texture = np.abs(lap) < 15.0
    bright_blowout = (gray >= 248) & low_texture
    m_sky = blue_sky | bright_blowout
    if np.count_nonzero(m_sky) > 0.03 * total_area:
        excluded_categories.append("sky_or_highlight")

    # 2. Detect Vegetation / Foliage
    # Green hues in HSV: H in [35..85], S >= 35, V >= 30
    m_veg = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 30)
    if np.count_nonzero(m_veg) > 0.02 * total_area:
        excluded_categories.append("vegetation")

    # 3. Detect Deep Shadow Voids / Background Cavities
    # Dark voids: luminance < 25
    m_void = (gray < 25)
    if np.count_nonzero(m_void) > 0.03 * total_area:
        excluded_categories.append("dark_background_void")

    # Composite exclusion mask
    m_exclude = m_sky | m_veg | m_void

    # 4. Detect Lateral High-Contrast Obstacles (e.g. dark roadside barrier walls)
    cand_rough = ~m_exclude
    if np.any(cand_rough):
        cand_gray = gray[cand_rough]
        if cand_gray.std() > 38.0 and cand_gray.size > 200:
            try:
                otsu_thresh, _ = cv2.threshold(cand_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                left_strip = gray[:, : max(int(w * 0.25), 1)]
                right_strip = gray[:, min(int(w * 0.75), w - 1) :]
                center_strip = gray[:, int(w * 0.25) : int(w * 0.75)]
                
                if left_strip.mean() < otsu_thresh - 15 and center_strip.mean() > otsu_thresh:
                    dark_cutoff = 35 if left_strip.std() > 40.0 else otsu_thresh
                    wall_mask = np.zeros((h, w), dtype=bool)
                    wall_mask[:, : int(w * 0.25)] = (gray[:, : int(w * 0.25)] <= dark_cutoff)
                    m_exclude = m_exclude | wall_mask
                    if "lateral_barrier" not in excluded_categories:
                        excluded_categories.append("lateral_barrier")
                elif right_strip.mean() < otsu_thresh - 15 and center_strip.mean() > otsu_thresh:
                    dark_cutoff = 35 if right_strip.std() > 40.0 else otsu_thresh
                    wall_mask = np.zeros((h, w), dtype=bool)
                    wall_mask[:, int(w * 0.75) :] = (gray[:, int(w * 0.75) :] <= dark_cutoff)
                    m_exclude = m_exclude | wall_mask
                    if "lateral_barrier" not in excluded_categories:
                        excluded_categories.append("lateral_barrier")
            except Exception:
                pass

    # 5. Morphological Regularization of Candidate Surface
    cand_binary = (~m_exclude).astype(np.uint8) * 255

    # Close internal holes (cracks, surface texture, aggregate particles)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cand_closed = cv2.morphologyEx(cand_binary, cv2.MORPH_CLOSE, close_kernel)
    # Open to detach thin noise tendrils
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cand_clean = cv2.morphologyEx(cand_closed, cv2.MORPH_OPEN, open_kernel)

    # 6. Connected Component Analysis & Dominant Surface Extraction
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_clean, connectivity=8)

    best_label = 0
    max_comp_area = 0

    for i in range(1, num_labels):
        comp_a = int(stats[i, cv2.CC_STAT_AREA])
        if comp_a > max_comp_area:
            max_comp_area = comp_a
            best_label = i

    coverage_ratio = max_comp_area / float(total_area)

    # 7. Decision and Fallback Evaluation
    # Case A: Nearly the entire image is concrete surface (e.g. close-up shot)
    if coverage_ratio >= max_surface_ratio or len(excluded_categories) == 0:
        full_poly = [[0, 0], [w, 0], [w, h], [0, h]]
        return {
            "roi_mask": None,
            "roi_polygon": full_poly,
            "roi_bbox": [0, 0, w, h],
            "coverage_pct": 100.0,
            "roi_area_px": total_area,
            "surface_type": "uniform_surface",
            "confidence": 0.95,
            "fallback": False,
            "warning": None,
            "excluded_categories": [],
        }

    # Case B: Surface region too small or fragmented (< min_surface_ratio) -> Fallback
    if coverage_ratio < min_surface_ratio or best_label == 0:
        return {
            "roi_mask": None,
            "roi_polygon": None,
            "roi_bbox": [0, 0, w, h],
            "coverage_pct": 100.0,
            "roi_area_px": total_area,
            "surface_type": "full_image_fallback",
            "confidence": 0.40,
            "fallback": True,
            "warning": (
                f"Automatic ROI detection found insufficient contiguous surface "
                f"({coverage_ratio * 100.0:.1f}% < {min_surface_ratio * 100.0:.0f}% threshold). "
                "Defaulting to whole-image inspection to prevent missing defects."
            ),
            "excluded_categories": excluded_categories,
        }

    # Case C: Distinct, reliable surface region segmented
    roi_mask = (labels == best_label).astype(np.uint8) * 255

    # Fill internal holes in the surface mask to ensure cracks inside the surface are not excluded
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "roi_mask": None,
            "roi_polygon": None,
            "roi_bbox": [0, 0, w, h],
            "coverage_pct": 100.0,
            "roi_area_px": total_area,
            "surface_type": "full_image_fallback",
            "confidence": 0.50,
            "fallback": True,
            "warning": "Failed to extract clean contour boundary for candidate surface. Fallback to whole image.",
            "excluded_categories": excluded_categories,
        }

    dominant_cnt = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_mask, [dominant_cnt], -1, 255, thickness=cv2.FILLED)

    final_area = int(np.count_nonzero(filled_mask))
    final_coverage_pct = round((final_area / float(total_area)) * 100.0, 1)

    # Simplify contour for clean polygon display and API transmission
    peri = cv2.arcLength(dominant_cnt, True)
    epsilon = max(0.008 * peri, 2.0)
    approx_poly = cv2.approxPolyDP(dominant_cnt, epsilon, True)
    poly_pts: List[List[int]] = [
        [int(pt[0][0]), int(pt[0][1])] for pt in approx_poly
    ]

    bx, by, bw, bh = cv2.boundingRect(dominant_cnt)
    roi_bbox = [int(bx), int(by), int(bw), int(bh)]

    surface_type = "road_surface" if ("sky_or_highlight" in excluded_categories) else "concrete_structure"

    hull = cv2.convexHull(dominant_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = (final_area / max(hull_area, 1.0)) if hull_area > 0 else 0.5
    conf = float(np.clip(0.65 + 0.30 * solidity, 0.0, 0.98))

    return {
        "roi_mask": filled_mask,
        "roi_polygon": poly_pts,
        "roi_bbox": roi_bbox,
        "coverage_pct": final_coverage_pct,
        "roi_area_px": final_area,
        "surface_type": surface_type,
        "confidence": conf,
        "fallback": False,
        "warning": None,
        "excluded_categories": excluded_categories,
    }


def create_roi_visualization(
    img: np.ndarray,
    roi_result: Dict[str, Any],
) -> np.ndarray:
    """
    Renders an informative visual overlay of the automatically detected surface ROI.

    - Shaded/tinted excluded background regions.
    - Vibrant cyan/green border along the active surface boundary.
    - Information badge indicating surface classification and coverage %.
    - If fallback active, displays a boundary indicator and fallback notice.
    """
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vis = rgb.copy()
    h, w = vis.shape[:2]

    fallback = roi_result.get("fallback", False)
    roi_mask = roi_result.get("roi_mask")
    roi_polygon = roi_result.get("roi_polygon")
    surface_type = roi_result.get("surface_type", "surface")
    coverage_pct = roi_result.get("coverage_pct", 100.0)

    if fallback or roi_mask is None:
        cv2.rectangle(vis, (4, 4), (w - 5, h - 5), (234, 179, 8), thickness=2)
        label_text = f"Whole-Image Surface ({coverage_pct:.1f}% coverage)"
        cv2.putText(vis, label_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (234, 179, 8), 2)
        return vis

    # Shade the excluded areas with a dark semi-transparent tint
    excluded_mask = (roi_mask == 0)
    if np.any(excluded_mask):
        shaded = vis.copy()
        shaded[excluded_mask] = (shaded[excluded_mask] * 0.35).astype(np.uint8)
        vis = shaded

    # Draw cyan boundary polygon around the detected surface ROI
    if roi_polygon and len(roi_polygon) >= 3:
        pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(6, 182, 212), thickness=3)

    type_display = surface_type.replace("_", " ").title()
    label = f"Auto ROI: {type_display} ({coverage_pct:.1f}%)"
    cv2.putText(vis, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (6, 182, 212), 2)

    return vis
