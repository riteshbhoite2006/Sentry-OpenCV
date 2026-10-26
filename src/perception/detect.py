"""
src/perception/detect.py

Crack detection module alias and entrypoint for Sentry.
Re-exports detect_cracks, measure_defect, measure_primary_crack,
select_primary_crack, and configuration defaults from src.perception.cracks.
"""

from src.perception.cracks import (
    DEFAULT_BORDER_MARGIN,
    DEFAULT_CLOSE_KERNEL_SIZE,
    DEFAULT_MAX_AREA_PCT_THRESHOLD,
    DEFAULT_MAX_ELONGATION_RATIO,
    DEFAULT_MIN_COMPONENT_AREA,
    DEFAULT_MIN_LENGTH_RATIO,
    DEFAULT_MIN_RIDGE_INTENSITY,
    DEFAULT_PERCENTILE,
    DEFAULT_ROI_EROSION,
    DEFAULT_SEVERITY_LENGTH_RATIO_BUMP,
    DEFAULT_SEVERITY_LOW_WIDTH,
    DEFAULT_SEVERITY_MEDIUM_WIDTH,
    DEFAULT_STRUCTURING_KERNEL_SIZE,
    detect_cracks,
    measure_defect,
    measure_primary_crack,
    select_primary_crack,
)
from src.perception.roi import create_roi_visualization, detect_surface_roi

__all__ = [
    "detect_cracks",
    "measure_defect",
    "measure_primary_crack",
    "select_primary_crack",
    "detect_surface_roi",
    "create_roi_visualization",
    "DEFAULT_PERCENTILE",
    "DEFAULT_BORDER_MARGIN",
    "DEFAULT_CLOSE_KERNEL_SIZE",
    "DEFAULT_MIN_COMPONENT_AREA",
    "DEFAULT_MIN_LENGTH_RATIO",
    "DEFAULT_MAX_ELONGATION_RATIO",
    "DEFAULT_MAX_AREA_PCT_THRESHOLD",
    "DEFAULT_MIN_RIDGE_INTENSITY",
    "DEFAULT_STRUCTURING_KERNEL_SIZE",
    "DEFAULT_ROI_EROSION",
    "DEFAULT_SEVERITY_LOW_WIDTH",
    "DEFAULT_SEVERITY_MEDIUM_WIDTH",
    "DEFAULT_SEVERITY_LENGTH_RATIO_BUMP",
]
