"""
src/perception/cracks.py

Concrete crack detection and defect measurement module for the Sentry inspection system.
Implements black-hat morphological ridge extraction, percentile thresholding,
border masking, elliptical fragment merging, connected-components filtering
(area, length ratio, elongation), longest-crack confidence estimation, and
skeleton-based primary crack dimension measurement.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.perception.quality import _validate_and_to_gray

# Baseline default parameters for crack perception
DEFAULT_STRUCTURING_KERNEL_SIZE: int = 17
DEFAULT_PERCENTILE: float = 97.0
DEFAULT_PERCENTILE_LOW: float = 93.0
DEFAULT_BORDER_MARGIN: int = 10
DEFAULT_CLOSE_KERNEL_SIZE: int = 7
DEFAULT_MIN_COMPONENT_AREA: int = 60
DEFAULT_MIN_LENGTH_RATIO: float = 0.08
DEFAULT_MAX_ELONGATION_RATIO: float = 12.0
DEFAULT_MAX_AREA_PCT_THRESHOLD: float = 5.0
DEFAULT_MIN_RIDGE_INTENSITY: float = 20.0
DEFAULT_ROI_EROSION: int = 10
DEFAULT_GAP_MAX_PX: float = 28.0
DEFAULT_MAX_ANGLE_DEG: float = 40.0
DEFAULT_SPALL_MIN_WIDTH: float = 5.5

# Named severity thresholds for measure_primary_crack
DEFAULT_SEVERITY_LOW_WIDTH: float = 3.0
DEFAULT_SEVERITY_MEDIUM_WIDTH: float = 10.0
DEFAULT_SEVERITY_LENGTH_RATIO_BUMP: float = 0.4



def measure_defect(mask: np.ndarray, img_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Measures geometric dimensions of detected defects from a binary crack mask.

    Parameters
    ----------
    mask : np.ndarray
        Single-channel 2D binary mask array (uint8, with foreground pixels > 0).
    img_shape : tuple of int
        Shape of the reference image (height, width, ...).

    Returns
    -------
    dict
        Dictionary containing measurements (crack counts, pixel lengths, bounding boxes).
    """
    if mask is None:
        raise ValueError("Mask cannot be None.")

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Mask must be a numpy.ndarray, got {type(mask).__name__}.")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, received ndim={mask.ndim}.")

    if not isinstance(img_shape, (tuple, list)) or len(img_shape) < 2:
        raise ValueError(
            f"img_shape must be a tuple/list of at least 2 dimensions, got {img_shape}."
        )

    height, width = int(img_shape[0]), int(img_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: height={height}, width={width}.")

    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {(height, width)}."
        )

    total_pixels = height * width
    total_crack_pixels = int(np.count_nonzero(mask))

    # Safe handling of empty masks
    if total_crack_pixels == 0:
        return {
            "n_cracks": 0,
            "crack_count": 0,
            "longest_crack_px": 0.0,
            "total_crack_px": 0.0,
            "estimated_pixel_length": 0.0,
            "total_crack_pixels": 0,
            "area_pct": 0.0,
            "area_percentage": 0.0,
            "mean_crack_width_px": 0.0,
            "max_crack_length_px": 0.0,
            "bounding_boxes": [],
        }

    binary_mask = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    bounding_boxes: List[List[int]] = []
    component_lengths: List[float] = []

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp_len = float(max(w, h))

        bounding_boxes.append([x, y, w, h])
        component_lengths.append(comp_len)

    n_cracks = len(component_lengths)
    longest_crack_px = float(max(component_lengths, default=0.0))
    total_crack_px = float(sum(component_lengths))
    area_pct = float((total_crack_pixels / total_pixels) * 100.0)

    mean_crack_width_px = (
        float(total_crack_pixels / max(total_crack_px, 1.0))
        if total_crack_px > 0
        else 0.0
    )

    return {
        "n_cracks": n_cracks,
        "crack_count": n_cracks,
        "longest_crack_px": longest_crack_px,
        "total_crack_px": total_crack_px,
        "estimated_pixel_length": total_crack_px,
        "total_crack_pixels": total_crack_pixels,
        "area_pct": area_pct,
        "area_percentage": area_pct,
        "mean_crack_width_px": mean_crack_width_px,
        "max_crack_length_px": longest_crack_px,
        "bounding_boxes": bounding_boxes,
    }


def measure_primary_crack(
    mask: np.ndarray,
    img_shape: Optional[Tuple[int, ...]] = None,
    mm_per_px: Optional[float] = None,
    severity_low_width: float = DEFAULT_SEVERITY_LOW_WIDTH,
    severity_medium_width: float = DEFAULT_SEVERITY_MEDIUM_WIDTH,
    severity_length_ratio_bump: float = DEFAULT_SEVERITY_LENGTH_RATIO_BUMP,
) -> Dict[str, Any]:
    """
    Computes morphological skeleton-based dimension measurements for the primary crack component.

    Uses skimage.morphology.skeletonize for 1-px skeleton extraction. Diagonal
    neighbour steps contribute sqrt(2) to path length, orthogonal steps contribute 1.0.
    Width is derived from a cv2.distanceTransform evaluated at skeleton pixels.

    Parameters
    ----------
    mask : np.ndarray
        2D uint8 binary mask of the primary crack component (foreground > 0).
    img_shape : Optional[Tuple[int, ...]]
        Reference image shape (height, width[, ...]) used for severity length bump.
        If None, the length-ratio severity bump is skipped.
    mm_per_px : Optional[float]
        Calibration factor (millimetres per pixel). When provided, length_mm,
        width_mean_mm, and width_max_mm are computed; otherwise they are None.
    severity_low_width : float
        width_max_px threshold below which severity is 'low' (default: 3.0 px).
    severity_medium_width : float
        width_max_px threshold below which severity is 'medium' (default: 10.0 px).
    severity_length_ratio_bump : float
        If length_path_px / image_width exceeds this ratio, severity is bumped to
        'high' regardless of width (default: 0.4).

    Returns
    -------
    dict
        - length_path_px: float, weighted skeleton path length (diagonal = sqrt(2)).
        - width_mean_px:  float, area / length_path_px (mean cross-section).
        - width_max_px:   float, 2 * max distanceTransform value over skeleton.
        - bbox_length_px: float, bounding-box long-edge length (reference).
        - length_mm, width_mean_mm, width_max_mm: float or None (requires mm_per_px).
        - severity: 'low' | 'medium' | 'high'.
    """
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return {
            "length_path_px": 0.0,
            "width_mean_px": 0.0,
            "width_max_px": 0.0,
            "bbox_length_px": 0.0,
            "length_mm": None,
            "width_mean_mm": None,
            "width_max_mm": None,
            "severity": "low",
        }

    binary = (mask > 0).astype(np.uint8)
    area_px = int(np.count_nonzero(binary))

    if area_px == 0:
        return {
            "length_path_px": 0.0,
            "width_mean_px": 0.0,
            "width_max_px": 0.0,
            "bbox_length_px": 0.0,
            "length_mm": None,
            "width_mean_mm": None,
            "width_max_mm": None,
            "severity": "low",
        }

    # --- bounding-box length (reference value) ---
    ys, xs = np.nonzero(binary)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_length_px = float(max(bbox_h, bbox_w))

    # --- 1-px skeleton path length with diagonal weighting ---
    skel_bool = skeletonize(binary > 0)  # bool array
    skel = skel_bool.astype(np.uint8)

    length_path_px = 0.0
    if np.any(skel_bool):
        # Forward-neighbour offsets (avoids double-counting undirected edges)
        # (dy, dx, weight)
        forward_offsets = [
            (0, 1, 1.0),           # right
            (1, -1, float(np.sqrt(2))),  # down-left
            (1, 0, 1.0),           # down
            (1, 1, float(np.sqrt(2))),   # down-right
        ]
        rows, cols = np.where(skel_bool)
        px_set = set(zip(rows.tolist(), cols.tolist()))
        for r, c in px_set:
            for dy, dx, w in forward_offsets:
                nr, nc = r + dy, c + dx
                if (nr, nc) in px_set:
                    length_path_px += w

    # --- distance-transform max width over skeleton pixels ---
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if np.any(skel_bool):
        width_max_px = float(2.0 * dist[skel_bool].max())
    else:
        width_max_px = 0.0

    # --- mean width ---
    width_mean_px = float(area_px / max(length_path_px, 1.0)) if length_path_px > 0 else 0.0

    # --- severity classification ---
    if width_max_px < severity_low_width:
        severity = "low"
    elif width_max_px < severity_medium_width:
        severity = "medium"
    else:
        severity = "high"

    # Length-ratio bump: if crack spans more than severity_length_ratio_bump of image width
    if img_shape is not None and len(img_shape) >= 2:
        img_width = int(img_shape[1])
        if img_width > 0 and length_path_px > severity_length_ratio_bump * img_width:
            severity = "high"

    # --- calibrated mm values ---
    if mm_per_px is not None and mm_per_px > 0:
        length_mm: Optional[float] = float(length_path_px * mm_per_px)
        width_mean_mm: Optional[float] = float(width_mean_px * mm_per_px)
        width_max_mm: Optional[float] = float(width_max_px * mm_per_px)
    else:
        length_mm = None
        width_mean_mm = None
        width_max_mm = None

    return {
        "length_path_px": round(length_path_px, 4),
        "width_mean_px": round(width_mean_px, 4),
        "width_max_px": round(width_max_px, 4),
        "bbox_length_px": round(bbox_length_px, 4),
        "length_mm": round(length_mm, 4) if length_mm is not None else None,
        "width_mean_mm": round(width_mean_mm, 4) if width_mean_mm is not None else None,
        "width_max_mm": round(width_max_mm, 4) if width_max_mm is not None else None,
        "severity": severity,
    }


def select_primary_crack(
    components: List[Dict[str, Any]],
    img_shape: Tuple[int, int],
    labels: Optional[np.ndarray] = None,
    blackhat: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Ranks kept crack components by a score combining length, elongation, and mean black-hat contrast.
    Returns the primary component plus a list of secondary ones, along with individual and combined masks.

    Score formulation:
        thinness = area_px / max(length_px, 1.0)
        elongation_ratio = length_px / max(thinness, 1.0)
        contrast_multiplier = 1.0 + (mean_contrast / 50.0)
        score = length_px * contrast_multiplier * np.log1p(elongation_ratio)

    Parameters
    ----------
    components : list of dict
        Candidate components passing geometric filters.
    img_shape : tuple of int
        Image dimensions (height, width).
    labels : np.ndarray, optional
        Connected components label array.
    blackhat : np.ndarray, optional
        Morphological black-hat response map.

    Returns
    -------
    dict
        - 'primary_crack': Optional[dict] with keys {'length_px', 'bbox', 'mean_contrast', 'area_px', 'centroid'}
        - 'secondary_cracks': list of dict
        - 'secondary_count': int
        - 'mask_primary': np.ndarray (uint8)
        - 'mask_all': np.ndarray (uint8)
    """
    height, width = img_shape[:2]
    mask_primary = np.zeros((height, width), dtype=np.uint8)
    mask_all = np.zeros((height, width), dtype=np.uint8)

    if not components:
        return {
            "primary_crack": None,
            "secondary_cracks": [],
            "secondary_count": 0,
            "mask_primary": mask_primary,
            "mask_all": mask_all,
        }

    scored_components: List[Dict[str, Any]] = []
    for comp in components:
        comp_copy = dict(comp)
        length_px = float(comp_copy.get("length_px", 0.0))
        area_px = float(comp_copy.get("area_px", 0.0))
        mean_contrast = float(comp_copy.get("mean_contrast", 0.0))
        skel_len = float(comp_copy.get("skel_len", length_px))

        contrast_multiplier = 1.0 + (mean_contrast / 50.0)
        score = max(length_px, skel_len) * contrast_multiplier
        comp_copy["score"] = score
        scored_components.append(comp_copy)

    # Rank kept components descending by composite score
    scored_components.sort(key=lambda c: c["score"], reverse=True)
    primary = scored_components[0]
    secondaries = scored_components[1:]

    primary_crack = {
        "length_px": float(primary["length_px"]),
        "bbox": list(primary["bbox"]),
        "mean_contrast": float(primary["mean_contrast"]),
        "area_px": int(primary["area_px"]),
        "centroid": list(primary["centroid"]),
    }

    if "mask" in primary and primary["mask"] is not None:
        mask_primary[primary["mask"] > 0] = 255
        for comp in scored_components:
            if "mask" in comp and comp["mask"] is not None:
                mask_all[comp["mask"] > 0] = 255
    elif labels is not None:
        primary_label = primary.get("label")
        if primary_label is not None:
            mask_primary[labels == primary_label] = 255
        for comp in scored_components:
            lbl = comp.get("label")
            if lbl is not None:
                mask_all[labels == lbl] = 255
    else:
        bx, by, bw, bh = primary_crack["bbox"]
        mask_primary[by : by + bh, bx : bx + bw] = 255
        for comp in scored_components:
            cbx, cby, cbw, cbh = comp["bbox"]
            mask_all[cby : cby + cbh, cbx : cbx + cbw] = 255

    return {
        "primary_crack": primary_crack,
        "secondary_cracks": secondaries,
        "secondary_count": len(secondaries),
        "mask_primary": mask_primary,
        "mask_all": mask_all,
    }


def _calculate_primary_crack_confidence(
    primary_crack: Optional[Dict[str, Any]],
    n_cracks: int,
    area_pct: float,
    img_diag: float,
    min_length_ratio: float,
    min_ridge_intensity: float,
    max_area_pct_threshold: float,
) -> Tuple[float, List[str]]:
    """
    Computes a heuristic confidence score in [0.0, 1.0] derived primarily from
    the PRIMARY crack component, its local contrast, and the ROI-relative area coverage.

    Includes sanity rule: if area_pct exceeds max_area_pct_threshold (default 5.0%),
    flags as 'texture_heavy' and caps confidence at 0.69 (enforcing human_review).
    """
    flags: List[str] = []
    if primary_crack is None or n_cracks == 0:
        return 0.0, flags

    primary_len = float(primary_crack.get("length_px", 0.0))
    if primary_len <= 0.0:
        return 0.0, flags

    min_len = min_length_ratio * img_diag
    target_prominent_len = 0.25 * img_diag

    # 1. Primary Crack Signal (scales 0.0 at min_len to 1.0 at target_prominent_len)
    s_primary = float(
        np.clip(
            (primary_len - min_len) / max(target_prominent_len - min_len, 1.0),
            0.0,
            1.0,
        )
    )

    # 2. Local Contrast Signal on Primary Crack Ridge Pixels
    mean_contrast = float(primary_crack.get("mean_contrast", 0.0))
    s_contrast = float(np.clip((mean_contrast - min_ridge_intensity) / 45.0, 0.0, 1.0))

    # 3. Area Coverage Signal (relative to active inspection ROI area)
    s_area = float(np.clip(area_pct / 1.5, 0.0, 1.0))

    # Base confidence starts at 0.35 once a verified component passes geometric filters
    raw_confidence = 0.35 + 0.65 * (0.55 * s_primary + 0.30 * s_contrast + 0.15 * s_area)
    confidence = float(np.clip(raw_confidence, 0.0, 1.0))

    # Sanity rule: heavy texture using ROI-relative area coverage
    if area_pct > max_area_pct_threshold:
        flags.append("texture_heavy")
        confidence = min(confidence, 0.69)

    return confidence, flags


def _calculate_longest_crack_confidence(
    n_cracks: int,
    longest_crack_px: float,
    area_pct: float,
    img_diag: float,
    min_length_ratio: float,
    blackhat: np.ndarray,
    mask: np.ndarray,
    min_ridge_intensity: float,
    max_area_pct_threshold: float,
) -> Tuple[float, List[str]]:
    """Legacy compatibility helper delegating to primary crack confidence calculation."""
    crack_pixels = mask > 0
    mean_val = float(np.mean(blackhat[crack_pixels])) if np.any(crack_pixels) else 0.0
    dummy_primary = {
        "length_px": longest_crack_px,
        "mean_contrast": mean_val,
    }
    return _calculate_primary_crack_confidence(
        primary_crack=dummy_primary,
        n_cracks=n_cracks,
        area_pct=area_pct,
        img_diag=img_diag,
        min_length_ratio=min_length_ratio,
        min_ridge_intensity=min_ridge_intensity,
        max_area_pct_threshold=max_area_pct_threshold,
    )


def detect_cracks(
    img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    roi_erosion: int = DEFAULT_ROI_EROSION,
    percentile: float = DEFAULT_PERCENTILE,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    border_margin: int = DEFAULT_BORDER_MARGIN,
    close_kernel_size: int = DEFAULT_CLOSE_KERNEL_SIZE,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_elongation_ratio: float = DEFAULT_MAX_ELONGATION_RATIO,
    max_area_pct_threshold: float = DEFAULT_MAX_AREA_PCT_THRESHOLD,
    min_ridge_intensity: float = DEFAULT_MIN_RIDGE_INTENSITY,
    gap_max_px: Optional[float] = None,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    spall_min_width: float = DEFAULT_SPALL_MIN_WIDTH,
    structuring_kernel_size: int = DEFAULT_STRUCTURING_KERNEL_SIZE,
    mm_per_px: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Detects concrete surface cracks using black-hat morphology, percentile thresholding,
    optional region-of-interest (ROI) masking with boundary erosion, fragment closing,
    connected-components filtering, and primary-crack selection.

    Parameters
    ----------
    img : np.ndarray
        Input image as a 2D grayscale or 3D BGR array.
    roi_mask : Optional[np.ndarray]
        Binary mask (uint8, same HxW, 255 = inspect) defining the region of interest.
    roi_erosion : int
        Erosion distance in pixels applied to ROI boundary to prevent false crack detection at edges (default: 10).
    percentile : float
        Percentile of black-hat response used for thresholding (default: 97.0).
    border_margin : int
        Border strip width in pixels zeroed out on all sides (default: 10).
    close_kernel_size : int
        Size of elliptical kernel for merging crack fragments (default: 7).
    min_component_area : int
        Minimum area in pixels for a valid crack component (default: 60).
    min_length_ratio : float
        Minimum length ratio relative to image diagonal (default: 0.08).
    max_elongation_ratio : float
        Maximum area / length ratio for thinness (default: 12.0).
    max_area_pct_threshold : float
        Maximum area percentage before flagging 'texture_heavy' and capping confidence at 0.69 (default: 5.0).
    min_ridge_intensity : float
        Noise floor intensity threshold for dark ridges (default: 20.0).
    structuring_kernel_size : int
        Black-hat rectangular structuring element size (default: 17).

    Returns
    -------
    dict
        Dictionary containing:
        - 'mask': np.ndarray, Binary uint8 crack mask of all kept cracks (alias for mask_all).
        - 'mask_primary': np.ndarray, Binary mask containing only the primary crack.
        - 'mask_all': np.ndarray, Binary mask containing all kept crack components.
        - 'primary_crack': Optional[dict], Attributes of primary crack {length_px, bbox, mean_contrast, area_px, centroid}.
        - 'secondary_cracks': list[dict], Secondary candidate crack components.
        - 'secondary_count': int, Number of secondary candidates.
        - 'n_cracks': int, Total count of confirmed crack components.
        - 'longest_crack_px': float, Length of the longest crack component.
        - 'total_crack_px': float, Summed length of all validated crack components.
        - 'area_pct': float, Percentage of ROI area occupied by cracks.
        - 'roi_area_px': int, Total pixel area of the inspection ROI.
        - 'confidence': float, Confidence score derived from primary crack and ROI area ratio.
        - 'flags': list[str], Diagnostic flags (e.g. ['texture_heavy']).
        - 'crack_count': int, (alias for n_cracks).
        - 'estimated_pixel_length': float, (alias for total_crack_px).
        - 'area_percentage': float, (alias for area_pct).
        - 'measurements': dict, Detailed measurements dictionary.
        - 'components': list[dict], All kept candidate components.
    """
    _, gray = _validate_and_to_gray(img)
    height, width = gray.shape
    img_diag = float(np.sqrt(height**2 + width**2))
    min_length_px = min_length_ratio * img_diag

    # Support legacy kwargs if provided
    if "min_length_px" in kwargs and kwargs["min_length_px"] is not None:
        min_length_px = float(kwargs["min_length_px"])
    if "min_contour_area" in kwargs and kwargs["min_contour_area"] is not None:
        min_component_area = int(kwargs["min_contour_area"])

    # 1. ROI validation and ~10 px boundary erosion
    effective_roi: Optional[np.ndarray] = None
    if roi_mask is not None:
        if not isinstance(roi_mask, np.ndarray):
            raise TypeError(f"roi_mask must be a numpy.ndarray, got {type(roi_mask).__name__}.")
        if roi_mask.shape[:2] != (height, width):
            raise ValueError(
                f"roi_mask shape {roi_mask.shape[:2]} does not match image shape {(height, width)}."
            )
        roi_bin = (roi_mask > 0).astype(np.uint8) * 255
        roi_area_px = int(np.count_nonzero(roi_bin))
        if roi_erosion > 0 and roi_area_px > 0:
            kernel_ero = np.ones((3, 3), np.uint8)
            effective_roi = cv2.erode(roi_bin, kernel_ero, iterations=roi_erosion)
        else:
            effective_roi = roi_bin.copy()
    else:
        roi_area_px = height * width

    # Check if effective ROI is completely empty
    if effective_roi is not None and np.count_nonzero(effective_roi) == 0:
        empty_mask = np.zeros_like(gray, dtype=np.uint8)
        measurements = measure_defect(empty_mask, gray.shape)
        measurements["roi_area_px"] = roi_area_px
        measurements["primary_crack"] = None
        measurements["secondary_count"] = 0
        return {
            "mask": empty_mask,
            "mask_primary": empty_mask,
            "mask_all": empty_mask,
            "primary_crack": None,
            "secondary_cracks": [],
            "secondary_count": 0,
            "n_cracks": 0,
            "longest_crack_px": 0.0,
            "total_crack_px": 0.0,
            "area_pct": 0.0,
            "roi_area_px": roi_area_px,
            "confidence": 0.0,
            "flags": [],
            "crack_count": 0,
            "estimated_pixel_length": 0.0,
            "area_percentage": 0.0,
            "measurements": measurements,
            "components": [],
        }

    # 2. Subtle Gaussian pre-filtering
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Black-Hat Morphological Transform (Closing(img) - img)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (structuring_kernel_size, structuring_kernel_size)
    )
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    # 4. Zero black-hat response outside ROI before thresholding
    if effective_roi is not None:
        blackhat[effective_roi == 0] = 0

    # Noise floor validation inside active inspection region
    if effective_roi is not None:
        roi_vals = blackhat[effective_roi > 0]
        if roi_vals.size == 0 or float(roi_vals.max()) < min_ridge_intensity:
            empty_mask = np.zeros_like(gray, dtype=np.uint8)
            measurements = measure_defect(empty_mask, gray.shape)
            measurements["roi_area_px"] = roi_area_px
            measurements["primary_crack"] = None
            measurements["secondary_count"] = 0
            return {
                "mask": empty_mask,
                "mask_primary": empty_mask,
                "mask_all": empty_mask,
                "primary_crack": None,
                "secondary_cracks": [],
                "secondary_count": 0,
                "n_cracks": 0,
                "longest_crack_px": 0.0,
                "total_crack_px": 0.0,
                "area_pct": 0.0,
                "roi_area_px": roi_area_px,
                "confidence": 0.0,
                "flags": [],
                "crack_count": 0,
                "estimated_pixel_length": 0.0,
                "area_percentage": 0.0,
                "measurements": measurements,
                "components": [],
            }
        p_val_high = float(np.percentile(roi_vals, percentile))
        p_val_low = float(np.percentile(roi_vals, percentile_low))
    else:
        if float(blackhat.max()) < min_ridge_intensity:
            empty_mask = np.zeros_like(gray, dtype=np.uint8)
            measurements = measure_defect(empty_mask, gray.shape)
            measurements["roi_area_px"] = roi_area_px
            measurements["primary_crack"] = None
            measurements["secondary_count"] = 0
            return {
                "mask": empty_mask,
                "mask_primary": empty_mask,
                "mask_all": empty_mask,
                "primary_crack": None,
                "secondary_cracks": [],
                "secondary_count": 0,
                "n_cracks": 0,
                "longest_crack_px": 0.0,
                "total_crack_px": 0.0,
                "area_pct": 0.0,
                "roi_area_px": roi_area_px,
                "confidence": 0.0,
                "flags": [],
                "crack_count": 0,
                "estimated_pixel_length": 0.0,
                "area_percentage": 0.0,
                "measurements": measurements,
                "components": [],
            }
        p_val_high = float(np.percentile(blackhat, percentile))
        p_val_low = float(np.percentile(blackhat, percentile_low))

    # 5. Hysteresis Thresholding on Black-Hat Response
    effective_thresh_high = max(p_val_high, min_ridge_intensity)
    effective_thresh_low = max(p_val_low, max(min_ridge_intensity * 0.75, 15.0))
    _, bin_high = cv2.threshold(blackhat, effective_thresh_high, 255, cv2.THRESH_BINARY)
    _, bin_low = cv2.threshold(blackhat, effective_thresh_low, 255, cv2.THRESH_BINARY)
    if effective_roi is not None:
        bin_high[effective_roi == 0] = 0
        bin_low[effective_roi == 0] = 0

    # 6. Border Handling: Zero out border strip on all sides
    if border_margin > 0 and height > 2 * border_margin and width > 2 * border_margin:
        for b in (bin_high, bin_low):
            b[:border_margin, :] = 0
            b[-border_margin:, :] = 0
            b[:, :border_margin] = 0
            b[:, -border_margin:] = 0

    # 7. Hysteresis Component Recovery: Keep bin_low components touching bin_high
    num_low, labels_low, _, _ = cv2.connectedComponentsWithStats(bin_low, connectivity=8)
    binary = np.zeros_like(gray, dtype=np.uint8)
    for lbl in range(1, num_low):
        comp_m = (labels_low == lbl)
        if np.any(bin_high[comp_m] > 0):
            binary[comp_m] = 255
    if np.count_nonzero(binary) == 0:
        binary = bin_high

    # 7b. Merge Crack Fragments with Elliptical Morphological Close
    if close_kernel_size > 1:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        if effective_roi is not None:
            binary[effective_roi == 0] = 0

    # Re-apply border mask to ensure clean margins after closing
    if border_margin > 0 and height > 2 * border_margin and width > 2 * border_margin:
        binary[:border_margin, :] = 0
        binary[-border_margin:, :] = 0
        binary[:, :border_margin] = 0
        binary[:, -border_margin:] = 0

    # 8. Connected Components with Stats (connectivity=8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidate_components: List[Dict[str, Any]] = []
    spall_detected = False

    for i in range(1, num_labels):
        comp_area = float(stats[i, cv2.CC_STAT_AREA])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp_len = float(max(bw, bh))
        elongation = comp_area / max(comp_len, 1.0)
        cx = float(centroids[i, 0])
        cy = float(centroids[i, 1])

        comp_mask = (labels == i)
        mean_contrast = float(np.mean(blackhat[comp_mask])) if np.any(comp_mask) else 0.0

        # Spall detection: large high-contrast non-linear pocket (area > 150, elongation >= max_elongation_ratio, mean_contrast > 35)
        if comp_area > 150 and elongation >= max_elongation_ratio and mean_contrast > 35.0:
            spall_detected = True
            continue

        # Retain component only if ALL three geometric criteria pass
        if (
            comp_area >= min_component_area
            and comp_len >= min_length_px
            and elongation < max_elongation_ratio
        ):
            mean_contrast = float(np.mean(blackhat[comp_mask])) if np.any(comp_mask) else 0.0
            candidate_components.append({
                "label": i,
                "length_px": comp_len,
                "bbox": [bx, by, bw, bh],
                "area_px": int(comp_area),
                "centroid": [cx, cy],
                "elongation": elongation,
                "mean_contrast": mean_contrast,
                "mask": comp_mask,
            })

    # Chaining components with Union-Find along dominant orientation
    effective_gap = gap_max_px if gap_max_px is not None else max(0.03 * img_diag, DEFAULT_GAP_MAX_PX)
    if len(candidate_components) > 1:
        n_c = len(candidate_components)
        parent = list(range(n_c))
        def _find(x: int) -> int:
            if parent[x] != x:
                parent[x] = _find(parent[x])
            return parent[x]
        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for u in range(n_c):
            for v in range(u + 1, n_c):
                c1, c2 = candidate_components[u], candidate_components[v]
                b1, b2 = c1["bbox"], c2["bbox"]
                dx = max(0, b1[0] - (b2[0] + b2[2]), b2[0] - (b1[0] + b1[2]))
                dy = max(0, b1[1] - (b2[1] + b2[3]), b2[1] - (b1[1] + b1[3]))
                if np.hypot(dx, dy) <= effective_gap:
                    pts1 = np.argwhere(c1["mask"])
                    pts2 = np.argwhere(c2["mask"])
                    if len(pts1) > 100:
                        pts1 = pts1[::max(1, len(pts1) // 100)]
                    if len(pts2) > 100:
                        pts2 = pts2[::max(1, len(pts2) // 100)]
                    diff = pts1[:, None, :] - pts2[None, :, :]
                    min_dist = float(np.sqrt(np.min(np.sum(diff**2, axis=-1))))
                    if min_dist <= effective_gap:
                        _union(u, v)

        groups: Dict[int, List[int]] = {}
        for u in range(n_c):
            groups.setdefault(_find(u), []).append(u)

        if len(groups) < n_c:
            chained_components: List[Dict[str, Any]] = []
            for root, idxs in groups.items():
                if len(idxs) == 1:
                    chained_components.append(candidate_components[idxs[0]])
                else:
                    merged_mask = np.zeros_like(gray, dtype=bool)
                    tot_area = 0
                    for idx in idxs:
                        merged_mask |= candidate_components[idx]["mask"]
                        tot_area += candidate_components[idx]["area_px"]
                    pts = np.argwhere(merged_mask)[:, [1, 0]]
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    comp_len = float(max(bw, bh))
                    mean_contrast = float(np.mean(blackhat[merged_mask]))
                    skel = skeletonize(merged_mask)
                    skel_len = float(np.count_nonzero(skel))
                    chained_components.append({
                        "label": candidate_components[idxs[0]]["label"],
                        "length_px": comp_len,
                        "skel_len": skel_len,
                        "bbox": [bx, by, bw, bh],
                        "area_px": tot_area,
                        "centroid": [float(bx + bw / 2.0), float(by + bh / 2.0)],
                        "elongation": tot_area / max(comp_len, 1.0),
                        "mean_contrast": mean_contrast,
                        "mask": merged_mask,
                    })
            candidate_components = chained_components

    # 9. Select primary crack and partition secondary candidates
    primary_res = select_primary_crack(
        candidate_components,
        img_shape=(height, width),
        labels=labels,
        blackhat=blackhat,
    )
    primary_crack = primary_res["primary_crack"]
    secondary_cracks = primary_res["secondary_cracks"]
    secondary_count = primary_res["secondary_count"]
    mask_primary = primary_res["mask_primary"]
    mask_all = primary_res["mask_all"]

    # 9b. Measure primary crack dimensions (skeleton path, width, severity, mm calibration)
    primary_measurements: Optional[Dict[str, Any]] = None
    if primary_crack is not None and np.count_nonzero(mask_primary) > 0:
        primary_measurements = measure_primary_crack(
            mask=mask_primary,
            img_shape=gray.shape,
            mm_per_px=mm_per_px,
        )
        # Merge dimension fields into primary_crack dict for downstream consumers
        primary_crack = {**primary_crack, **primary_measurements}

    # 10. Metrics & Confidence Computation
    total_crack_pixels = int(np.count_nonzero(mask_all))
    area_pct = float((total_crack_pixels / max(roi_area_px, 1)) * 100.0)

    n_cracks = len(candidate_components)
    longest_crack_px = float(max([c["length_px"] for c in candidate_components], default=0.0))
    total_crack_px = float(sum([c["length_px"] for c in candidate_components]))

    confidence, flags = _calculate_primary_crack_confidence(
        primary_crack=primary_crack,
        n_cracks=n_cracks,
        area_pct=area_pct,
        img_diag=img_diag,
        min_length_ratio=min_length_ratio,
        min_ridge_intensity=min_ridge_intensity,
        max_area_pct_threshold=max_area_pct_threshold,
    )

    if spall_detected:
        flags.append("spall_suspected")
        confidence = min(confidence, 0.69)

    measurements = {
        "n_cracks": n_cracks,
        "crack_count": n_cracks,
        "longest_crack_px": longest_crack_px,
        "total_crack_px": total_crack_px,
        "estimated_pixel_length": total_crack_px,
        "total_crack_pixels": total_crack_pixels,
        "area_pct": area_pct,
        "area_percentage": area_pct,
        "roi_area_px": roi_area_px,
        "mean_crack_width_px": (
            float(total_crack_pixels / max(total_crack_px, 1.0))
            if total_crack_px > 0
            else 0.0
        ),
        "max_crack_length_px": longest_crack_px,
        "bounding_boxes": [c["bbox"] for c in candidate_components],
        "primary_crack": primary_crack,
        "secondary_count": secondary_count,
    }

    return {
        "mask": mask_all,
        "mask_primary": mask_primary,
        "mask_all": mask_all,
        "primary_crack": primary_crack,
        "secondary_cracks": secondary_cracks,
        "secondary_count": secondary_count,
        "n_cracks": n_cracks,
        "longest_crack_px": longest_crack_px,
        "total_crack_px": total_crack_px,
        "area_pct": area_pct,
        "roi_area_px": roi_area_px,
        "confidence": confidence,
        "flags": flags,
        "crack_count": n_cracks,
        "estimated_pixel_length": total_crack_px,
        "area_percentage": area_pct,
        "measurements": measurements,
        "components": candidate_components,
    }

