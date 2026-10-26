"""
src/agent/planner.py

Rule-based inspection agent planner implementing the perception -> decision -> action loop
for concrete surface crack inspection in the Sentry system.

Enforces:
1. Strict decision sequencing (Quality -> Enhancement -> Reassessment -> Detection -> Triage)
2. Four discrete verdicts: 'defect_found', 'no_defect', 'recapture_requested', 'human_review'
3. Comprehensive JSON-compatible step-by-step tracing (max 6 steps)
4. Safe error handling (never returns 'no_defect' upon failure)
"""

from typing import Any, Dict, Optional
import numpy as np

from src.agent.trace import InspectionTrace, make_json_safe
from src.perception.cracks import detect_cracks
from src.perception.quality import check_quality, enhance_image
from src.perception.roi import detect_surface_roi

# Operational confidence thresholds
CONFIDENCE_NO_DEFECT_UPPER_BOUND: float = 0.35
CONFIDENCE_HUMAN_REVIEW_UPPER_BOUND: float = 0.70
MAX_AGENT_STEPS: int = 6


def inspect(
    img: Any,
    roi_mask: Optional[np.ndarray] = None,
    auto_roi: bool = True,
    mm_per_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes the autonomous inspection agent loop on a concrete photograph.

    Sequence:
    1. Assess initial image quality.
    2. If quality fails, attempt conservative enhancement and reassess.
    3. If quality remains unacceptable, return 'recapture_requested'.
    4. Define region of interest (ROI): either user-specified or full image.
    5. When acceptable, execute crack detection (with boundary erosion and primary crack extraction).
    6. Measure primary crack dimensions via skeleton path and distance transform.
    7. Evaluate spatial spread and confidence score:
       - Multi-region spread (>60% W and H with >= 2 candidates and no ROI): 'human_review'
       - < 0.35: 'no_defect'
       - 0.35 to 0.70 inclusive: 'human_review'
       - > 0.70: 'defect_found'
    8. Maintain a maximum of 6 audited trace steps.

    Parameters
    ----------
    img : Any
        Input image candidate (expected to be a 2D or 3D numpy.ndarray).
    roi_mask : Optional[np.ndarray]
        Optional binary mask (uint8, same HxW, 255 = inspect) defining the region of interest.
    mm_per_px : Optional[float]
        Scale calibration factor (millimetres per pixel). When provided, the
        measure_primary_crack step reports calibrated length_mm / width_mm values.

    Returns
    -------
    dict
        Structured result containing:
        - 'verdict': str ('defect_found', 'no_defect', 'recapture_requested', 'human_review')
        - 'mask': Optional[np.ndarray] (Binary crack mask or None)
        - 'mask_primary': Optional[np.ndarray] (Primary crack mask or None)
        - 'mask_all': Optional[np.ndarray] (All confirmed crack components mask or None)
        - 'primary_crack': Optional[dict] (Primary crack attributes or None)
        - 'secondary_count': int
        - 'roi_area_px': int
        - 'metrics': dict (Inspection and quality metrics, JSON-safe)
        - 'trace': list[dict] (Ordered list of audited trace step dicts)
        - 'error': Optional[str] (Error description if an issue occurred)
    """
    trace = InspectionTrace(max_steps=MAX_AGENT_STEPS)
    metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # Step 0: Input Validation
    # ------------------------------------------------------------------------
    if (
        img is None
        or not isinstance(img, np.ndarray)
        or img.size == 0
        or img.ndim not in (2, 3)
    ):
        issue_desc = (
            "Input image is None."
            if img is None
            else f"Input must be a non-empty 2D or 3D numpy.ndarray, got {type(img).__name__}."
        )
        trace.add_step(
            tool="input_validation",
            inputs_and_metrics={
                "received_type": str(type(img).__name__),
                "is_ndarray": isinstance(img, np.ndarray),
            },
            decision="recapture_requested",
            reason=f"Input rejected: {issue_desc}. Recapture is required.",
        )
        return {
            "verdict": "recapture_requested",
            "mask": None,
            "mask_primary": None,
            "mask_all": None,
            "primary_crack": None,
            "secondary_count": 0,
            "roi_area_px": 0,
            "roi_mask": None,
            "roi_info": None,
            "metrics": {},
            "trace": trace.to_list(),
            "error": issue_desc,
        }

    working_img = img

    try:
        # --------------------------------------------------------------------
        # Step 1: Initial Quality Assessment
        # --------------------------------------------------------------------
        q1 = check_quality(working_img)
        metrics["initial_quality"] = make_json_safe(q1)

        if q1["ok"]:
            trace.add_step(
                tool="check_quality",
                inputs_and_metrics={
                    "blur_score": q1["blur_score"],
                    "brightness": q1["brightness"],
                    "contrast": q1["contrast"],
                    "quality_ok": True,
                },
                decision="quality_passed",
                reason="Input image quality metrics satisfy all operational thresholds.",
            )
        else:
            trace.add_step(
                tool="check_quality",
                inputs_and_metrics={
                    "blur_score": q1["blur_score"],
                    "brightness": q1["brightness"],
                    "contrast": q1["contrast"],
                    "quality_ok": False,
                    "issues": q1["issues"],
                },
                decision="quality_failed",
                reason=f"Quality check failed: {'; '.join(q1['issues'])}. Attempting enhancement.",
            )

            # ----------------------------------------------------------------
            # Step 2: Attempt Enhancement
            # ----------------------------------------------------------------
            enhanced = enhance_image(working_img)
            trace.add_step(
                tool="enhance_image",
                inputs_and_metrics={"enhancement_method": "CLAHE"},
                decision="image_enhanced",
                reason="Applied conservative CLAHE enhancement to improve luminance/contrast.",
            )

            # ----------------------------------------------------------------
            # Step 3: Reassess Enhanced Image Quality
            # ----------------------------------------------------------------
            q2 = check_quality(enhanced)
            metrics["enhanced_quality"] = make_json_safe(q2)

            if q2["ok"]:
                trace.add_step(
                    tool="check_quality",
                    inputs_and_metrics={
                        "blur_score": q2["blur_score"],
                        "brightness": q2["brightness"],
                        "contrast": q2["contrast"],
                        "quality_ok": True,
                    },
                    decision="quality_passed_after_enhancement",
                    reason="Enhanced image successfully meets quality criteria. Proceeding to detection.",
                )
                working_img = enhanced
            else:
                trace.add_step(
                    tool="check_quality",
                    inputs_and_metrics={
                        "blur_score": q2["blur_score"],
                        "brightness": q2["brightness"],
                        "contrast": q2["contrast"],
                        "quality_ok": False,
                        "issues": q2["issues"],
                    },
                    decision="recapture_requested",
                    reason=f"Quality remains deficient after enhancement: {'; '.join(q2['issues'])}. Aborting inspection.",
                )
                return {
                    "verdict": "recapture_requested",
                    "mask": None,
                    "mask_primary": None,
                    "mask_all": None,
                    "primary_crack": None,
                    "secondary_count": 0,
                    "roi_area_px": 0,
                    "roi_mask": None,
                    "roi_info": None,
                    "metrics": metrics,
                    "trace": trace.to_list(),
                    "error": None,
                }

        # --------------------------------------------------------------------
        # Step: Define Region of Interest (ROI)
        # --------------------------------------------------------------------
        effective_roi_mask = roi_mask
        auto_roi_info = None

        if effective_roi_mask is not None:
            # Explicit caller-provided ROI (e.g. from existing test suite or custom polygon)
            roi_provided = True
            roi_method = "manual_roi"
            roi_area_val = int(np.count_nonzero(effective_roi_mask > 0))
            roi_decision = "roi_applied"
            roi_reason = f"Explicit Region-of-Interest active: inspecting within bounded region of {roi_area_val:,} pixels."
        elif auto_roi:
            # Fully automated surface ROI detection
            auto_roi_info = detect_surface_roi(working_img)
            if auto_roi_info.get("roi_mask") is not None and not auto_roi_info.get("fallback", False):
                effective_roi_mask = auto_roi_info["roi_mask"]
                roi_provided = True
                roi_method = "auto_surface_detection"
                roi_area_val = int(auto_roi_info["roi_area_px"])
                surf_type = auto_roi_info.get("surface_type", "surface")
                cov_pct = auto_roi_info.get("coverage_pct", 100.0)
                roi_decision = "roi_applied"
                roi_reason = f"Automatic surface ROI active ({surf_type}, {cov_pct:.1f}% coverage, {roi_area_val:,} px)."
            elif auto_roi_info.get("fallback", False):
                roi_provided = False
                roi_method = "auto_fallback_whole_image"
                roi_area_val = int(working_img.shape[0] * working_img.shape[1])
                warn = auto_roi_info.get("warning") or "No distinct surface boundaries detected; whole-image inspection applied."
                roi_decision = "full_image_roi"
                roi_reason = warn
            else:
                roi_provided = False
                roi_method = "uniform_surface_full_image"
                roi_area_val = int(working_img.shape[0] * working_img.shape[1])
                roi_decision = "full_image_roi"
                roi_reason = "Full concrete surface identified: inspecting entire image frame."
        else:
            roi_provided = False
            roi_method = "full_image"
            roi_area_val = int(working_img.shape[0] * working_img.shape[1])
            roi_decision = "full_image_roi"
            roi_reason = "No ROI provided: full image surface inspection active."

        trace.add_step(
            tool="define_roi",
            inputs_and_metrics={
                "roi_provided": roi_provided,
                "roi_method": roi_method,
                "roi_area_px": roi_area_val,
                "surface_type": auto_roi_info.get("surface_type") if auto_roi_info else ("manual" if roi_provided else "full_image"),
                "coverage_pct": auto_roi_info.get("coverage_pct", 100.0) if auto_roi_info else 100.0,
                "fallback": auto_roi_info.get("fallback", False) if auto_roi_info else False,
            },
            decision=roi_decision,
            reason=roi_reason,
        )

        if auto_roi_info is not None:
            roi_summary = {k: v for k, v in auto_roi_info.items() if k != "roi_mask"}
            metrics["roi"] = make_json_safe(roi_summary)
        elif effective_roi_mask is not None:
            metrics["roi"] = {
                "roi_area_px": roi_area_val,
                "surface_type": "manual_roi",
                "coverage_pct": round((roi_area_val / float(working_img.shape[0] * working_img.shape[1])) * 100.0, 1),
                "fallback": False,
                "warning": None,
            }
        else:
            metrics["roi"] = {
                "roi_area_px": roi_area_val,
                "surface_type": "full_image",
                "coverage_pct": 100.0,
                "fallback": False,
                "warning": None,
            }

        # --------------------------------------------------------------------
        # Step: Crack Detection & Triage
        # --------------------------------------------------------------------
        detection = detect_cracks(working_img, roi_mask=effective_roi_mask, mm_per_px=mm_per_px)
        confidence = float(detection["confidence"])
        mask = detection["mask"]
        components = detection.get("components", [])
        h_img, w_img = working_img.shape[:2]

        # Multi-region spread check: if no ROI and candidate centroids span >60% W and H
        multi_region_spread = False
        if effective_roi_mask is None and len(components) >= 2:
            cx_vals = [c["centroid"][0] for c in components]
            cy_vals = [c["centroid"][1] for c in components]
            span_x = max(cx_vals) - min(cx_vals)
            span_y = max(cy_vals) - min(cy_vals)
            if span_x > 0.60 * w_img and span_y > 0.60 * h_img:
                multi_region_spread = True

        metrics["crack_detection"] = {
            "n_cracks": detection.get("n_cracks", detection.get("crack_count", 0)),
            "crack_count": detection.get("crack_count", detection.get("n_cracks", 0)),
            "longest_crack_px": detection.get("longest_crack_px", 0.0),
            "total_crack_px": detection.get("total_crack_px", detection.get("estimated_pixel_length", 0.0)),
            "estimated_pixel_length": detection.get("total_crack_px", detection.get("estimated_pixel_length", 0.0)),
            "area_pct": detection.get("area_pct", detection.get("area_percentage", 0.0)),
            "area_percentage": detection.get("area_pct", detection.get("area_percentage", 0.0)),
            "roi_area_px": detection.get("roi_area_px", roi_area_val),
            "primary_crack": make_json_safe(detection.get("primary_crack")),
            "secondary_count": detection.get("secondary_count", 0),
            "confidence": confidence,
            "flags": detection.get("flags", []),
            "measurements": make_json_safe(detection.get("measurements", {})),
        }

        # Triaging verdicts based on spall presence, multi-region spread, and confidence thresholds
        flags = detection.get("flags", [])
        if "spall_suspected" in flags or "spall_detected" in flags:
            verdict = "human_review"
            decision_msg = "human_review"
            reason_msg = (
                f"Spalled concrete feature identified (Score: {confidence:.2f}); human engineering review required."
            )
        elif multi_region_spread:
            verdict = "human_review"
            decision_msg = "human_review"
            reason_msg = f"multiple separate regions found, ROI needed ({len(components)} separate crack components detected across image)"
        elif confidence < CONFIDENCE_NO_DEFECT_UPPER_BOUND:
            verdict = "no_defect"
            decision_msg = "no_defect"
            reason_msg = (
                f"Confidence score {confidence:.2f} is below defect threshold "
                f"{CONFIDENCE_NO_DEFECT_UPPER_BOUND:.2f}. No confirmed surface cracks detected."
            )
        elif confidence <= CONFIDENCE_HUMAN_REVIEW_UPPER_BOUND:
            verdict = "human_review"
            decision_msg = "human_review"
            reason_msg = (
                f"Features detected with intermediate confidence (Score: {confidence:.2f}). "
                "Referred to human inspector."
            )
        else:
            verdict = "defect_found"
            decision_msg = "defect_found"
            reason_msg = (
                f"Confidence score {confidence:.2f} exceeds threshold "
                f"{CONFIDENCE_HUMAN_REVIEW_UPPER_BOUND:.2f}. Concrete surface crack confirmed."
            )

        trace.add_step(
            tool="detect_cracks",
            inputs_and_metrics={
                "confidence": confidence,
                "n_cracks": detection.get("n_cracks", detection.get("crack_count", 0)),
                "primary_crack": make_json_safe(detection.get("primary_crack")),
                "secondary_count": detection.get("secondary_count", 0),
                "longest_crack_px": detection.get("longest_crack_px", 0.0),
                "total_crack_px": detection.get("total_crack_px", 0.0),
                "area_pct": detection.get("area_pct", 0.0),
                "roi_area_px": detection.get("roi_area_px", roi_area_val),
                "flags": detection.get("flags", []),
                "multi_region_spread": multi_region_spread,
            },
            decision=decision_msg,
            reason=reason_msg,
        )

        # ----------------------------------------------------------------
        # Step: Measure primary crack dimensions (only if crack found)
        # ----------------------------------------------------------------
        primary_crack_result = detection.get("primary_crack")
        if primary_crack_result is not None and verdict != "no_defect":
            trace.add_step(
                tool="measure_primary_crack",
                inputs_and_metrics={
                    "length_path_px": primary_crack_result.get("length_path_px"),
                    "width_mean_px": primary_crack_result.get("width_mean_px"),
                    "width_max_px": primary_crack_result.get("width_max_px"),
                    "bbox_length_px": primary_crack_result.get("bbox_length_px"),
                    "severity": primary_crack_result.get("severity"),
                    "mm_per_px": mm_per_px,
                    "length_mm": primary_crack_result.get("length_mm"),
                    "width_mean_mm": primary_crack_result.get("width_mean_mm"),
                    "width_max_mm": primary_crack_result.get("width_max_mm"),
                },
                decision="primary_crack_measured",
                reason=(
                    f"Primary crack skeleton path length: {primary_crack_result.get('length_path_px', 0.0):.1f} px; "
                    f"max width: {primary_crack_result.get('width_max_px', 0.0):.1f} px; "
                    f"severity: {primary_crack_result.get('severity', 'unknown')}."
                    + (
                        f" Calibrated: length={primary_crack_result.get('length_mm'):.1f} mm, "
                        f"width={primary_crack_result.get('width_max_mm'):.1f} mm."
                        if mm_per_px is not None
                        else " No scale reference — pixel measurements only."
                    )
                ),
            )
        elif verdict == "no_defect" or primary_crack_result is None:
            trace.add_step(
                tool="measure_primary_crack",
                inputs_and_metrics={"crack_present": False},
                decision="no_crack_to_measure",
                reason="No primary crack component detected; dimension measurement skipped.",
            )

        return {
            "verdict": verdict,
            "mask": mask,
            "mask_primary": detection.get("mask_primary"),
            "mask_all": detection.get("mask_all"),
            "primary_crack": detection.get("primary_crack"),
            "secondary_count": detection.get("secondary_count", 0),
            "roi_area_px": detection.get("roi_area_px", roi_area_val),
            "roi_mask": effective_roi_mask,
            "roi_info": metrics.get("roi"),
            "metrics": metrics,
            "trace": trace.to_list(),
            "error": None,
        }


    except Exception as exc:
        # Never falsely return 'no_defect' upon failure; preserve trace and refer to human review
        trace.add_step(
            tool="agent_planner",
            inputs_and_metrics={"exception_type": type(exc).__name__},
            decision="human_review",
            reason=f"Inspection interrupted by unexpected error: {str(exc)}",
        )
        return {
            "verdict": "human_review",
            "mask": None,
            "mask_primary": None,
            "mask_all": None,
            "primary_crack": None,
            "secondary_count": 0,
            "roi_area_px": 0,
            "roi_mask": None,
            "roi_info": None,
            "metrics": metrics,
            "trace": trace.to_list(),
            "error": f"Internal inspection error: {str(exc)}",
        }
