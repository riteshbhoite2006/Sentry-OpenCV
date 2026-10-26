"""
app.py

Streamlit interactive dashboard for the Sentry concrete surface-defect inspection system.
Communicates with the Sentry FastAPI backend (/inspect, /health) to visualize
inspection verdicts, visual overlays, metrics, audited traces, and human reviews.
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import requests
import streamlit as st

from src.perception.roi import create_roi_visualization

# Configure Streamlit page
st.set_page_config(
    page_title="Sentry | Concrete Crack Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .verdict-box {
        padding: 18px 24px;
        border-radius: 10px;
        margin: 15px 0px 25px 0px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .verdict-defect-found {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
    }
    .verdict-no-defect {
        background-color: rgba(34, 197, 94, 0.15);
        border: 2px solid #22c55e;
        color: #4ade80;
    }
    .verdict-recapture {
        background-color: rgba(234, 179, 8, 0.15);
        border: 2px solid #eab308;
        color: #fde047;
    }
    .verdict-human-review {
        background-color: rgba(249, 115, 22, 0.15);
        border: 2px solid #f97316;
        color: #fb923c;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .step-badge {
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def decode_base64_mask(mask_b64: Optional[str]) -> Optional[np.ndarray]:
    """Decodes a base64 PNG string back into a 2D uint8 OpenCV mask array."""
    if not mask_b64:
        return None
    try:
        mask_bytes = base64.b64decode(mask_b64)
        np_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        return mask
    except Exception:
        return None


def create_annotated_overlay(
    original_bytes: bytes,
    mask_primary: Optional[np.ndarray],
    mask_all: Optional[np.ndarray],
    roi_polygon: Optional[List[List[int]]] = None,
    show_all: bool = False,
) -> Optional[np.ndarray]:
    """
    Renders an overlay highlighting crack pixels and the ROI outline on top of the original photograph.
    Default (show_all=False): highlights only the primary crack in bright red.
    If show_all=True: highlights secondary cracks in bright amber and primary crack in bright red.
    Draws the green ROI boundary outline if an ROI polygon is provided.
    """
    try:
        np_arr = np.frombuffer(original_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        has_overlay = False

        if show_all:
            # Color secondary candidates in bright amber/orange [255, 140, 0]
            if mask_all is not None and np.count_nonzero(mask_all) > 0:
                m_all = mask_all
                if m_all.shape[:2] != rgb.shape[:2]:
                    m_all = cv2.resize(m_all, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay[m_all > 0] = [255, 140, 0]
                has_overlay = True

            # Color primary crack in bright red [255, 40, 40]
            if mask_primary is not None and np.count_nonzero(mask_primary) > 0:
                m_prim = mask_primary
                if m_prim.shape[:2] != rgb.shape[:2]:
                    m_prim = cv2.resize(m_prim, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay[m_prim > 0] = [255, 40, 40]
                has_overlay = True
        else:
            # Default: only primary crack is highlighted in bright red [255, 40, 40]
            target_mask = (
                mask_primary
                if (mask_primary is not None and np.count_nonzero(mask_primary) > 0)
                else mask_all
            )
            if target_mask is not None and np.count_nonzero(target_mask) > 0:
                if target_mask.shape[:2] != rgb.shape[:2]:
                    target_mask = cv2.resize(
                        target_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                overlay[target_mask > 0] = [255, 40, 40]
                has_overlay = True

        blended = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0) if has_overlay else rgb.copy()

        # Draw ROI outline in vibrant green if specified
        if roi_polygon is not None and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(blended, [pts], isClosed=True, color=(34, 197, 94), thickness=3)

        return blended
    except Exception:
        return None



# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.title("⚙️ Configuration")
default_api_url = os.getenv("SENTRY_API_URL", "http://localhost:8000")
api_base_url = st.sidebar.text_input("Backend API Base URL", value=default_api_url)

# Health Check
st.sidebar.subheader("API Connectivity")
try:
    health_resp = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=2.5)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(
            f"Connected to API\n\nOpenCV: {health_data.get('opencv_version', 'N/A')}"
        )
    else:
        st.sidebar.warning(f"API returned status {health_resp.status_code}")
except Exception as e:
    st.sidebar.error(f"Backend offline at {api_base_url}")
    st.sidebar.caption(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("About Sentry")
st.sidebar.markdown(
    """
    **Sentry** is an autonomous perception-decision-action inspection aid for **concrete cracks only**.

    **Four Discrete Verdicts:**
    - `defect_found`: Confirmed surface crack.
    - `no_defect`: No crack detected.
    - `recapture_requested`: Image quality deficient.
    - `human_review`: Borderline / uncertain feature.

    *Responsible Engineering Notice: Image-based aid only; not a substitute for a licensed engineer.*
    """
)

# ============================================================================
# Main Dashboard
# ============================================================================

st.title("Sentry — Concrete Surface Crack Inspection")
st.markdown(
    "Upload a high-resolution or mobile photograph of a concrete surface to perform "
    "autonomous image quality evaluation, crack detection, and step-by-step audited triage."
)

col_upload, col_action = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Select a concrete surface photograph",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Upload an uncropped image of cast concrete or concrete pavement.",
    )

roi_polygon = None
show_all_candidates = False
enable_auto_roi = True
manual_roi_polygon = None
scale_ref_json: Optional[str] = None

if uploaded_file is not None:
    file_bytes_raw = uploaded_file.getvalue()
    np_raw = np.frombuffer(file_bytes_raw, dtype=np.uint8)
    img_dec = cv2.imdecode(np_raw, cv2.IMREAD_COLOR)
    img_h, img_w = img_dec.shape[:2] if img_dec is not None else (600, 600)

    st.markdown("#### 🎯 Inspection & Display Controls")
    c_roi_chk, c_cand_chk = st.columns(2)
    with c_roi_chk:
        enable_auto_roi = st.checkbox(
            "Automatic Surface ROI Detection",
            value=True,
            help="Automatically segments road or concrete surface and excludes sky, vegetation, lateral barriers, and background voids. Falls back to whole image if ambiguous.",
        )
    with c_cand_chk:
        show_all_candidates = st.checkbox(
            "Show all candidates in ROI",
            value=False,
            help="When enabled, highlights secondary candidates in amber in addition to primary crack. Off by default: only the primary crack is highlighted.",
        )

    # Optional manual overrides collapsed in expander — zero manual input required in normal workflow
    with st.expander("🛠️ Advanced Settings (Optional Manual Overrides)", expanded=False):
        st.caption("The normal workflow requires zero manual inputs. Use these controls only to test explicit polygons or calibrated physical rulers.")
        override_manual_roi = st.checkbox("Manual ROI Box Override", value=False)
        if override_manual_roi:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                roi_xmin = st.number_input("X Min", min_value=0, max_value=img_w, value=0, step=10)
            with r2:
                roi_ymin = st.number_input("Y Min", min_value=0, max_value=img_h, value=0, step=10)
            with r3:
                roi_xmax = st.number_input("X Max", min_value=0, max_value=img_w, value=img_w, step=10)
            with r4:
                roi_ymax = st.number_input("Y Max", min_value=0, max_value=img_h, value=img_h, step=10)
            if roi_xmin < roi_xmax and roi_ymin < roi_ymax:
                manual_roi_polygon = [
                    [int(roi_xmin), int(roi_ymin)],
                    [int(roi_xmax), int(roi_ymin)],
                    [int(roi_xmax), int(roi_ymax)],
                    [int(roi_xmin), int(roi_ymax)],
                ]
                box_area = (roi_xmax - roi_xmin) * (roi_ymax - roi_ymin)
                st.caption(f"📐 Active Manual ROI: [{roi_xmin}, {roi_ymin}] to [{roi_xmax}, {roi_ymax}] ({box_area:,} px)")
            else:
                st.error("Invalid ROI: X Min must be less than X Max, and Y Min must be less than Y Max.")

        override_scale = st.checkbox("Manual Scale Reference Marker", value=False)
        if override_scale:
            st.caption("Enter two points whose real-world physical distance is known:")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            with sc1:
                sx1 = st.number_input("X₁ (px)", min_value=0, max_value=img_w, value=0, step=1, key="sx1")
            with sc2:
                sy1 = st.number_input("Y₁ (px)", min_value=0, max_value=img_h, value=0, step=1, key="sy1")
            with sc3:
                sx2 = st.number_input("X₂ (px)", min_value=0, max_value=img_w, value=min(100, img_w), step=1, key="sx2")
            with sc4:
                sy2 = st.number_input("Y₂ (px)", min_value=0, max_value=img_h, value=0, step=1, key="sy2")
            with sc5:
                real_mm_val = st.number_input("Real Length (mm)", min_value=0.1, value=100.0, step=0.1, key="real_mm")
            if sx1 != sx2 or sy1 != sy2:
                scale_ref_json = json.dumps({"p1": [sx1, sy1], "p2": [sx2, sy2], "real_mm": real_mm_val})
                st.caption(f"📐 Scale Reference: ({sx1}, {sy1}) → ({sx2}, {sy2}) = {real_mm_val:.1f} mm")

with col_action:
    st.write("")
    st.write("")
    inspect_clicked = st.button(
        "🔍 Run Inspection",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

# Execution upon button click
if inspect_clicked and uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "image/jpeg")}
    data = {
        "auto_roi": "true" if enable_auto_roi else "false",
    }
    if manual_roi_polygon is not None:
        data["roi"] = json.dumps(manual_roi_polygon)
    if scale_ref_json is not None:
        data["scale_ref"] = scale_ref_json

    with st.spinner("Executing autonomous agent perception -> decision -> action loop..."):
        try:
            resp = requests.post(
                f"{api_base_url.rstrip('/')}/inspect",
                files=files,
                data=data,
                timeout=30.0,
            )
            if resp.status_code == 200:
                st.session_state["result"] = resp.json()
                st.session_state["image_bytes"] = file_bytes
                st.session_state["filename"] = uploaded_file.name
                st.session_state["show_all_candidates"] = show_all_candidates
                st.session_state["scale_active"] = scale_ref_json is not None
            else:
                try:
                    err_payload = resp.json()
                    st.error(f"API Error ({resp.status_code}): {err_payload.get('error', resp.text)}")
                except Exception:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"Failed to communicate with API server at {api_base_url}: {exc}")

# Display Results if available
if "result" in st.session_state and "image_bytes" in st.session_state:
    result = st.session_state["result"]
    verdict = result.get("verdict", "unknown")
    confidence = result.get("confidence")
    metrics = result.get("metrics", {})
    trace = result.get("trace", [])
    mask_b64 = result.get("mask_base64")
    mask_prim_b64 = result.get("mask_primary_base64")
    error_msg = result.get("error")
    active_roi = st.session_state.get("roi_polygon")
    active_show_all = st.session_state.get("show_all_candidates", False)
    scale_was_active = st.session_state.get("scale_active", False)
    mm_per_px_result = result.get("mm_per_px")

    # Render Prominent Verdict Banner
    if verdict == "defect_found":
        st.markdown(
            f"""
            <div class="verdict-box verdict-defect-found">
                <h2 style="margin:0 0 6px 0;">🚨 Verdict: DEFECT FOUND</h2>
                <p style="margin:0; font-size:1.05rem;">
                    Confirmed concrete crack identified with high confidence
                    (Score: <b>{f'{confidence:.2f}' if confidence is not None else 'N/A'}</b>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict == "no_defect":
        st.markdown(
            f"""
            <div class="verdict-box verdict-no-defect">
                <h2 style="margin:0 0 6px 0;">✅ Verdict: NO DEFECT</h2>
                <p style="margin:0; font-size:1.05rem;">
                    Concrete surface inspection completed. No crack features confirmed
                    (Score: <b>{f'{confidence:.2f}' if confidence is not None else '0.00'}</b> &lt; 0.35).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict == "recapture_requested":
        st.markdown(
            """
            <div class="verdict-box verdict-recapture">
                <h2 style="margin:0 0 6px 0;">⚠️ Verdict: RECAPTURE REQUESTED</h2>
                <p style="margin:0; font-size:1.05rem;">
                    Image quality (blur, lighting, or contrast) was unacceptable, and enhancement failed
                    to meet operational standards. Please capture a clearer, properly illuminated photograph.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict == "human_review":
        rev_flags = result.get("metrics", {}).get("crack_detection", {}).get("flags", [])
        if "spall_suspected" in rev_flags or "spall_detected" in rev_flags:
            review_reason = "Spalled concrete feature identified. Escalated to human inspector for verification."
        elif result.get("secondary_count", 0) >= 1 and result.get("roi") is None:
            review_reason = "Multiple separate crack regions detected. Escalated to human inspector for verification."
        else:
            review_reason = "Features detected with intermediate/ambiguous confidence. Escalated to human inspector for verification."

        st.markdown(
            f"""
            <div class="verdict-box verdict-human-review">
                <h2 style="margin:0 0 6px 0;">👤 Verdict: HUMAN REVIEW REQUIRED</h2>
                <p style="margin:0; font-size:1.05rem;">
                    {review_reason} (Score: <b>{f'{confidence:.2f}' if confidence is not None else 'N/A'}</b>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Verdict: {verdict}")

    if error_msg:
        st.warning(f"Inspection Notes / Errors: {error_msg}")

    # Tabs for Inspection Details
    tab_vis, tab_metrics, tab_trace, tab_review = st.tabs(
        ["🖼️ Visual Inspection", "📊 Metrics & Analysis", "📜 Audited Agent Trace", "✍️ Human Review Queue"]
    )

    mask_all = decode_base64_mask(mask_b64)
    mask_primary = decode_base64_mask(mask_prim_b64)
    roi_mask = decode_base64_mask(result.get("roi_mask_base64"))
    roi_info = result.get("roi") or {}

    # 1. Visual Analysis Tab
    with tab_vis:
        if roi_info.get("warning"):
            st.warning(f"⚠️ ROI Detection Notice: {roi_info['warning']}")
        elif roi_info.get("surface_type"):
            cov = roi_info.get("coverage_pct", 100.0)
            st.info(f"🎯 Active Surface ROI: **{roi_info.get('surface_type').replace('_', ' ').title()}** ({cov:.1f}% coverage, {roi_info.get('roi_area_px', 0):,} px)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader("1. Original Image")
            st.image(st.session_state["image_bytes"], use_container_width=True)

        with c2:
            st.subheader("2. Surface ROI Overlay")
            np_arr_vis = np.frombuffer(st.session_state["image_bytes"], dtype=np.uint8)
            img_dec_vis = cv2.imdecode(np_arr_vis, cv2.IMREAD_COLOR)
            if img_dec_vis is not None:
                roi_data_copy = dict(roi_info)
                roi_data_copy["roi_mask"] = roi_mask
                roi_vis = create_roi_visualization(img_dec_vis, roi_data_copy)
                st.image(roi_vis, use_container_width=True)
            else:
                st.info("Surface ROI overlay unavailable.")

        with c3:
            if active_show_all:
                st.subheader("3. All Candidates Mask")
                disp_mask = mask_all
            else:
                st.subheader("3. Primary Crack Mask")
                disp_mask = (
                    mask_primary
                    if (mask_primary is not None and np.count_nonzero(mask_primary) > 0)
                    else mask_all
                )

            if disp_mask is not None and np.count_nonzero(disp_mask) > 0:
                st.image(disp_mask, use_container_width=True, clamp=True)
            else:
                st.info("No crack mask available (clean surface or recapture).")

        with c4:
            st.subheader("4. Annotated Overlay")
            roi_poly_overlay = roi_info.get("roi_polygon")
            overlay = create_annotated_overlay(
                st.session_state["image_bytes"],
                mask_primary=mask_primary,
                mask_all=mask_all,
                roi_polygon=roi_poly_overlay,
                show_all=active_show_all,
            )
            if overlay is not None:
                st.image(overlay, use_container_width=True)
            else:
                st.info("Overlay unavailable.")

    # 2. Metrics Tab
    with tab_metrics:
        st.subheader("Crack & Defect Measurements")

        # Scale notice banner - Requirement 3
        if mm_per_px_result is not None:
            st.success(f"📏 Calibrated Physical Scale Active: {mm_per_px_result:.4f} mm/px (from verified reference)")
        else:
            st.info("ℹ️ Real-world scale unavailable; measurements are in pixels.")

        crack_data = metrics.get("crack_detection", {})
        roi_meta = metrics.get("roi", {}) or {}
        measurements = crack_data.get("measurements", {})
        flags = crack_data.get("flags", [])
        primary_crack = crack_data.get("primary_crack") or {}
        primary_len = float(primary_crack.get("length_px", 0.0))
        primary_bbox_len = primary_len
        secondary_count = int(crack_data.get("secondary_count", 0))
        roi_area_px = int(crack_data.get("roi_area_px", roi_meta.get("roi_area_px", 0)))
        area_pct_val = float(crack_data.get("area_pct", 0.0))
        total_candidates = int(crack_data.get("n_cracks", crack_data.get("crack_count", 0)))

        # Skeleton-based dimension values from primary_crack
        length_path_px = primary_crack.get("length_path_px")
        width_mean_px = primary_crack.get("width_mean_px")
        width_max_px = primary_crack.get("width_max_px")
        severity_val = primary_crack.get("severity")
        length_mm = primary_crack.get("length_mm")
        width_mean_mm = primary_crack.get("width_mean_mm")
        width_max_mm = primary_crack.get("width_max_mm")

        if "texture_heavy" in flags:
            st.warning("⚠️ Texture-heavy surface: detection reliability is low.")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(
                "Confidence Score",
                f"{confidence:.2f}" if confidence is not None else "N/A",
                help="Heuristic confidence score calculated from primary crack length, contrast, and ROI area ratio.",
            )
        with m2:
            centerline_metric = (
                f"{length_path_px:.1f} px"
                if length_path_px is not None
                else (f"{primary_len:.1f} px" if primary_len > 0 else "N/A")
            )
            st.metric(
                "Centerline Path Length",
                centerline_metric,
                help="Estimated crack centerline length from skeletonized mask (px). Avoids bounding-box diagonal overestimation.",
            )
        with m3:
            st.metric(
                "Surface Defect Area %",
                f"{area_pct_val:.3f} %",
                help=f"Detected crack area percentage relative to inspected surface ROI ({roi_area_px:,} px).",
            )
        with m4:
            st.metric(
                "Detected Components",
                total_candidates,
                help=f"Total confirmed crack components ({secondary_count} secondary candidates).",
            )
        with m5:
            if severity_val:
                severity_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(severity_val.lower(), "⚪")
                severity_disp = f"{severity_color} {severity_val.upper()}"
            else:
                severity_disp = "N/A"
            st.metric(
                "Severity Heuristic",
                severity_disp,
                help="Documented project-defined heuristic classification based on pixel width/length. NOT an engineering safety assessment.",
            )

        # Primary crack skeleton dimension metrics
        if length_path_px is not None:
            st.markdown("---")
            st.subheader("Primary Crack Dimensions (Centerline & Width Profile)")
            d1, d2, d3, d4 = st.columns(4)

            def _fmt_dim(px_val, mm_val):
                if px_val is None:
                    return "N/A"
                if mm_per_px_result is not None and mm_val is not None:
                    return f"{mm_val:.2f} mm ({px_val:.1f} px)"
                return f"{px_val:.1f} px"

            with d1:
                st.metric(
                    "Centerline Length",
                    _fmt_dim(length_path_px, length_mm),
                    help="Morphological skeleton path length (diagonal steps counted as √2).",
                )
            with d2:
                st.metric(
                    "Mean Crack Width",
                    _fmt_dim(width_mean_px, width_mean_mm),
                    help="Mean crack cross-section width (area_px / skeleton_path_length).",
                )
            with d3:
                st.metric(
                    "Max Crack Width",
                    _fmt_dim(width_max_px, width_max_mm),
                    help="Maximum width: 2 × peak Euclidean distance transform value on crack skeleton.",
                )
            with d4:
                st.metric(
                    "Bounding Box Length",
                    f"{primary_len:.1f} px" if primary_len > 0 else "N/A",
                    help="Reference bounding-box diagonal length (px).",
                )

            st.caption(
                "🔬 **Measurement Methodology**: Centerline length is estimated directly from a 1-pixel-wide skeletonized mask. "
                "Mean width is calculated by dividing total component pixel area by centerline path length; maximum width "
                "is derived from Euclidean distance transform along the skeleton. "
                "Severity classifications are heuristic project-defined triage categories and do NOT constitute an engineering safety assessment."
            )

        if primary_crack:
            st.markdown("---")
            length_attr_disp = (
                f"{length_path_px:.1f} px"
                if length_path_px is not None
                else (f"{primary_len:.1f} px" if primary_len > 0 else "N/A")
            )
            contrast_val = primary_crack.get("mean_contrast")
            contrast_disp = f"{contrast_val:.1f}" if contrast_val is not None else "N/A"
            area_val = primary_crack.get("area_px")
            area_disp = f"{area_val} px" if area_val is not None else "N/A"
            bbox_disp = primary_crack.get("bbox") if primary_crack.get("bbox") is not None else "N/A"
            st.markdown(
                f"**Primary Crack Attributes:** "
                f"Length: `{length_attr_disp}` | "
                f"Bounding Box: `{bbox_disp}` | "
                f"Mean Black-Hat Contrast: `{contrast_disp}` | "
                f"Area: `{area_disp}`"
            )

        st.markdown("---")
        st.subheader("Image Quality Assessment")
        init_q = metrics.get("initial_quality", {})
        enh_q = metrics.get("enhanced_quality", {})

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.metric("Blur Score (Laplacian Var)", f"{init_q.get('blur_score', 0.0):.1f}")
        with qc2:
            st.metric("Brightness Intensity", f"{init_q.get('brightness', 0.0):.1f} / 255")
        with qc3:
            st.metric("Contrast (Std Dev)", f"{init_q.get('contrast', 0.0):.1f}")

        if init_q.get("issues"):
            st.warning("Initial Quality Issues: " + "; ".join(init_q["issues"]))

        if enh_q:
            st.info(
                f"Enhanced Quality: Blur={enh_q.get('blur_score', 0.0):.1f}, "
                f"Contrast={enh_q.get('contrast', 0.0):.1f}. "
                f"Issues: {'; '.join(enh_q.get('issues', [])) or 'None'}"
            )

    # 3. Audited Trace Tab
    with tab_trace:
        st.subheader("Audited Perception-Decision-Action Trace")
        st.caption("Complete, chronological, tamper-evident log of agent execution steps.")

        if not trace:
            st.info("No trace recorded.")
        else:
            for s in trace:
                step_num = s.get("step", 0)
                tool = s.get("tool", "unknown")
                dec = s.get("decision", "unknown")
                reason = s.get("reason", "")
                ts = s.get("timestamp", "")
                inp_metrics = s.get("inputs_and_metrics", {})

                with st.expander(f"Step {step_num}: {tool} ➔ **{dec}** ({ts})", expanded=True):
                    st.markdown(f"**Reason:** {reason}")
                    if inp_metrics:
                        st.json(inp_metrics)

    # 4. Human Review Queue Tab
    with tab_review:
        st.subheader("Human Inspector Verification Queue")
        if verdict == "human_review":
            st.warning(
                "⚠️ This inspection yielded an ambiguous confidence score or required escalation. "
                "A licensed civil or structural engineer should record their determination below."
            )
        else:
            st.info(f"The automated verdict is '{verdict}'. You may optionally record an inspector audit review below.")

        with st.form("human_review_form"):
            reviewer_name = st.text_input("Reviewer Name / Employee ID", value="Inspector 1")
            determination = st.selectbox(
                "Manual Determination",
                [
                    "Confirm Defect Present (True Positive)",
                    "Confirm Surface Clean / False Alarm (False Positive)",
                    "Request High-Resolution Reinspection On Site",
                    "Schedule Physical Destructive / Ultrasonic Testing",
                ],
            )
            review_notes = st.text_area("Engineering Remarks / Observations", height=100)
            submitted = st.form_submit_button("Submit Review Sign-Off", type="primary")

            if submitted:
                review_record = {
                    "filename": st.session_state.get("filename", "unknown"),
                    "automated_verdict": verdict,
                    "confidence": confidence,
                    "reviewer": reviewer_name,
                    "determination": determination,
                    "notes": review_notes,
                }
                if "reviews_history" not in st.session_state:
                    st.session_state["reviews_history"] = []
                st.session_state["reviews_history"].append(review_record)
                st.success(f"Review successfully submitted by {reviewer_name}!")

        if "reviews_history" in st.session_state and st.session_state["reviews_history"]:
            st.markdown("#### Completed Review Log")
            for r in reversed(st.session_state["reviews_history"]):
                st.markdown(
                    f"- **{r['determination']}** by *{r['reviewer']}* on `{r['filename']}`: {r['notes']}"
                )
