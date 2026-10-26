# Decision Log — Sentry (OpenCV AI Competition 2026)

This document tracks architectural and technical decisions made during the design, development, and deployment of the Sentry surface-defect inspection system.

Each entry records the date, context, decision, rationale, alternatives considered, and validation evidence.

---

## Decision 1: Scope Restricted to Concrete Cracks Only

- **Date:** 2026-10-24
- **Decision:** Restrict inspection scope exclusively to surface cracks on concrete structures. Spalling, rebar corrosion, efflorescence, delamination, and non-concrete surfaces are explicitly out of scope for the core perception pipeline.
- **Reason:** Surface defect inspection requires calibrated feature extraction and robust domain boundaries. Narrowing the scope to concrete cracks ensures high precision in edge/ridge morphology, enables reliable heuristic and machine-learning thresholding, and prevents ambiguity in quality and defect scoring.
- **Alternatives Considered:**
  - *Multi-defect concrete inspection (cracks, spalls, stains, rebar exposure):* Rejected due to excessive model complexity, lack of reliable unified annotation for early milestones, and danger of diluted accuracy.
  - *General masonry/asphalt crack detection:* Rejected due to high variance in texture, aggregate size, and reflectance profiles across uncurated materials.
- **Evidence / Validation Status:** Established as project baseline requirement.

---

## Decision 2: OpenCV 5.x Mandate (No OpenCV 4.x Fallback)

- **Date:** 2026-10-24
- **Decision:** Require OpenCV 5.x as the foundational computer vision runtime. The installed `cv2.__version__` must strictly begin with `5.`. Never silently fall back to OpenCV 4.x.
- **Reason:** The competition specifically mandates OpenCV 5.x capabilities and forward-looking API standards. Ensuring strict version gating at environment startup eliminates compatibility regressions and guarantees reproducible behavior.
- **Alternatives Considered:**
  - *Dual-compatibility fallback (OpenCV 4.x / 5.x):* Rejected because the competition rules require OpenCV 5.x, and silent fallbacks mask environment misconfigurations.
- **Evidence / Validation Status:** Verified via `tests/check_env.py` using `opencv-python-headless==5.0.0.93`, yielding runtime version `5.0.0` on Python 3.13.0 (Windows x86_64).

---

## Decision 3: Deterministic Rule-Based Agent Loop First

- **Date:** 2026-10-24
- **Decision:** Implement a deterministic, rule-based perception → decision → action loop with strict heuristics and maximum step caps (≤ 6 steps) before introducing any stochastic or LLM-driven components.
- **Reason:** Industrial inspection systems require auditability, low latency, and deterministic reproducibility. A rule-based agent ensures every decision branch (quality assessment → enhancement → crack detection → confidence triage) is fully traceable, predictable, and verifiable before exploring external model calls.
- **Alternatives Considered:**
  - *LLM / VLM-based agent directly orchestrating perception tools:* Rejected for MVP due to high token cost, non-deterministic latency, potential hallucination of defect severity, and dependency on internet/API connectivity.
- **Evidence / Validation Status:** Architecture designed with four discrete verdicts (`defect_found`, `no_defect`, `recapture_requested`, `human_review`) and step-by-step JSON tracing. Planned for implementation in subsequent agent task.

---

## Decision 4: Amazon Bedrock as an Optional Future Enhancement

- **Date:** 2026-10-24
- **Decision:** Amazon Bedrock multimodal inference (e.g., Claude 3.5 Sonnet / Haiku for inspection narrative generation or second-opinion triage on `human_review` cases) is designated as an optional enhancement for future phases, strictly after local and core AWS milestones are validated.
- **Reason:** Core defect verification must run autonomously without reliance on billable external model endpoints. Adding Bedrock early creates unnecessary billing and external dependency risks during active core algorithm development.
- **Alternatives Considered:**
  - *Integrating Bedrock into initial agent decision loop:* Rejected to maintain zero cloud spend during local prototyping and keep the perception core lightweight.
- **Evidence / Validation Status:** Deferred. No Bedrock resources configured or billed during Phase 1.

---

## Decision 5: COOL Benchmark on AWS Graviton Designated Optional

- **Date:** 2026-10-24
- **Decision:** Benchmarking against the OpenCV Computer Vision Optimization Library (COOL) on AWS Graviton instances is cataloged as an optional future benchmark rather than a hard blocking dependency.
- **Reason:** Local development, core detection precision, and robust agentic triage take priority. Graviton deployment and hardware-specific SIMD/NEON benchmarking can be evaluated once the Dockerized baseline is finalized.
- **Alternatives Considered:**
  - *Mandating Graviton-only local emulators:* Rejected as impractical on standard x86 developer workstations.
- **Evidence / Validation Status:** Deferred to future cloud benchmarking phase.

---

## Decision 6: Local Development and Automated Testing Before AWS Deployment

- **Date:** 2026-10-24
- **Decision:** Develop, test, and validate the complete perception pipeline, agent loop, FastAPI backend, and Streamlit frontend locally before deploying to AWS EC2, S3, and CloudWatch.
- **Reason:** Local iteration delivers immediate feedback, zero cloud compute costs, and rapid unit/integration test cycles. Containerization and cloud deployment are subsequent progression steps once local tests pass.
- **Alternatives Considered:**
  - *Cloud-first / remote-only development:* Rejected due to slow iteration speeds, debugging friction, and unnecessary operational overhead.
- **Evidence / Validation Status:** Initial environment verification script `tests/check_env.py` and local repo structure established and verified locally.

---

## Decision 7: Environment & Python Version Alignment

- **Date:** 2026-10-24
- **Decision:** Utilize inspected package versions (`opencv-python-headless==5.0.0.93`, `numpy==2.5.4`, `fastapi==0.142.2`, `uvicorn==0.54.0`, `streamlit==1.65.0`, `boto3==1.43.112`, `pytest==9.1.1`, `pandas==3.0.2`, `scikit-learn==1.8.0`). While the host currently runs Python 3.13 with verified OpenCV 5.0.0 availability, the deployment container will target Python 3.11/3.10 with the same pinned wheels to preserve competition environment compliance.
- **Reason:** The brief specifies Python 3.10 or 3.11 subject to verified package compatibility. Testing confirmed that OpenCV 5.0.0.93 wheels are functional and available across modern Python releases.
- **Alternatives Considered:**
  - *Unpinned dependency installations:* Rejected to prevent drifting sub-dependencies from breaking builds.
- **Evidence / Validation Status:** Full import verification script passed with 0 exit code on 2026-10-24.

---

## Decision 8: Classical Morphological Baseline for Perception Module

- **Date:** 2026-10-24
- **Decision:** Implement classical computer vision methods for initial perception: Laplacian variance for blur estimation, mean/std for exposure/contrast, CLAHE in LAB space for conservative enhancement, and a black-hat morphological ridge extractor combined with noise-floor-gated Otsu thresholding and aspect-ratio/thinness contour filtering for crack detection.
- **Reason:** Surface cracks on concrete present as narrow, dark structural indentations across varying matrix textures. Black-hat morphology isolates dark features narrower than the structuring element without requiring heavy neural networks. Geometric contour filtering (aspect ratio and thinness ratio) effectively rejects round concrete pores (bug holes) and isolated sensor noise while keeping CPU/memory overhead minimal and execution deterministic.
- **Alternatives Considered:**
  - *Deep learning segmentation (e.g. U-Net / SegFormer):* Deferred to later phases due to zero labeled training data currently available, high compute overhead, and need for a deterministic, verifiable baseline first.
  - *Canny edge detection alone:* Rejected because Canny responds heavily to concrete aggregate texture and high-frequency speckle noise, causing excessive false edges.
- **Evidence / Validation Status:** Implemented in `src/perception/quality.py` and `src/perception/cracks.py`. Validated with 20 unit tests in `tests/test_perception.py` passing in 0.89s under OpenCV 5.0.0.

---

## Decision 9: Deterministic Agent Sequencer and Audited JSON Trace Logging

- **Date:** 2026-10-24
- **Decision:** Implement the perception → decision → action loop in `src/agent/planner.py` using a strict sequential rule set: initial quality assessment -> optional CLAHE enhancement & reassessment -> crack detection -> threshold triage (`confidence < 0.35` -> `no_defect`, `0.35 <= confidence <= 0.70` -> `human_review`, `confidence > 0.70` -> `defect_found`). Enforce an audited `InspectionTrace` capped at 6 steps. Convert NumPy datatypes to JSON-safe primitives via `make_json_safe` and exclude raw pixel matrices from traces.
- **Reason:** Industrial structural inspection demands high auditability, reproducible triage, and predictable response times. Capping execution at 6 steps prevents circular enhancement loops. Keeping image matrices separate from traces keeps JSON logs lightweight for CloudWatch and API transmission while retaining full audit details (step, tool, metrics, decision, reason, ISO timestamp).
- **Evidence / Validation Status:** Validated via 20 unit tests in `tests/test_agent.py` covering synthetic cracks, unblemished surfaces, blurry images, enhancement recovery, exact confidence boundary edges, trace JSON serialization, step caps, and exception handling.

---

## Decision 10: FastAPI REST Service & Streamlit Decoupling

- **Date:** 2026-10-24
- **Decision:** Expose the perception-decision-action loop via FastAPI (`src/api/main.py`) with `GET /health` and `POST /inspect`. Decouple the user interface (`app.py`) so it consumes the backend over HTTP without duplicating any inspection heuristics. Encode crack masks as base64 PNG strings (`mask_base64`) rather than raw NumPy arrays, cap multipart uploads at 20MB, and enforce strict format validation. Provide an interactive human review queue in Streamlit for verdicts marked `human_review`.
- **Reason:** Separating the API from the frontend enables multi-client deployment (Streamlit web dashboard, automated batch evaluators, CLI tools, and future AWS EC2 microservices). Transmitting masks as base64 PNG preserves lossless pixel boundaries while preventing JSON serialization errors. Keeping the UI purely presentational ensures decision logic remains strictly unified in `src/agent/planner.py`.
- **Evidence / Validation Status:** Validated via 10 integration and unit tests in `tests/test_api.py` covering `/health`, valid image uploads, clean image responses, blurry images, unsupported extensions, empty files, corrupted streams, oversized payloads, internal exception fallbacks, and JSON safety. Full suite (50 tests) passing in 1.93s.

---

## Decision 11: Crack Detector Over-Detection Mitigation & Robust Morphological Filtering

- **Date:** 2026-10-24
- **Decision:** Refactor the crack detection pipeline in `src/perception/cracks.py` (and aliased at `src/perception/detect.py`) to eliminate texture over-detection, fragment splitting, and border artifacts observed on real-world concrete photos. Replace Otsu thresholding with percentile thresholding, zero border strips, merge micro-gaps with an elliptical close, filter connected components using area, diagonal length ratio, and elongation (`area / length`), calculate confidence from the longest crack and area coverage, and apply a 5.0% area sanity cap (`texture_heavy`).
- **Parameters, Defaults, and Rationale:**
  1. `percentile = 97.0`: Isolates the top 3% darkest ridge responses in black-hat space, preventing Otsu from forcibly partitioning defect-free concrete texture into foreground.
  2. `border_margin = 10` (px): Clears edge boundaries to prevent false positives from frame borders, vignettes, or border alignment marks.
  3. `close_kernel_size = 7`: Elliptical closing kernel ($7 \times 7$) that bridges micro-fissure gaps so a continuous crack is recognized as a single coherent component rather than dozens of fragments.
  4. `min_component_area = 60` (px): Rejects small surface pits, aggregate cavities, and sensor noise specks.
  5. `min_length_ratio = 0.08`: Mandates that component bounding-box length spans at least 8% of the image diagonal ($\ge 0.08 \times \sqrt{H^2 + W^2}$), filtering out localized short scratches.
  6. `max_elongation_ratio = 12.0`: Requires component `area / length < 12.0`, ensuring narrow fissure geometry and rejecting circular bug holes or wide spall craters.
  7. `max_area_pct_threshold = 5.0` (%): If crack coverage exceeds 5.0% of the image, flags `'texture_heavy'` and caps confidence at 0.69, routing ambiguous scenes to `human_review` instead of `defect_found`.
  8. `min_ridge_intensity = 20.0`: Establishes a floor below which flat or low-contrast black-hat responses are treated as defect-free.
- **Alternatives Considered:**
  - *Increasing Otsu threshold offsets:* Rejected because Otsu remains fundamentally sensitive to background texture distributions.
  - *Deep learning segmentation:* Retained as future phase; classical morphological filtering provides immediate, verifiable, deterministic CPU execution without requiring labeled training datasets.
---

## Decision 12: Region of Interest (ROI) Filtering, Boundary Erosion, and Primary-Crack Triage

- **Date:** 2026-10-24
- **Decision:** Implement optional Region-of-Interest (ROI) support and primary-crack triage across the perception module, agent loop, REST API, and Streamlit dashboard. In `detect_cracks`, support `roi_mask`, apply morphological boundary erosion before zeroing outer black-hat responses, calculate percentile thresholds strictly over the eroded ROI, and compute `area_pct` relative to `roi_area_px`. Implement `select_primary_crack` to isolate the dominant crack from secondary features. In `src/agent/planner.py`, introduce the `"define_roi"` trace step and enforce a multi-region spread check (`> 60%` width and height spread with no ROI) escalating to `human_review`.
- **Parameters, Defaults, and Rationale:**
  1. `roi_erosion = 10` (px): A ~10-pixel morphological erosion kernel applied to the input `roi_mask` boundaries. When inspecting road pavements adjacent to high-contrast concrete barrier walls or curbs, drawing an ROI right up to the wall boundary would cause step edges to be falsely detected as cracks. Eroding the ROI boundary guarantees that outer contrast transitions are zeroed out before thresholding and fragment closing.
  2. `roi_area_px`: The active inspection pixel area (either the non-zero pixel count of `roi_mask`, or total image area $H \times W$ when no ROI is provided).
  3. `area_pct` (ROI-relative): Computed as `(total_crack_pixels / max(roi_area_px, 1)) * 100.0`. Ensures that defect density accurately reflects the designated road surface rather than being diluted by non-inspected wall/barrier area.
  4. `select_primary_crack` Scoring: Ranks kept components using a composite structural metric:
     $$\text{score} = \text{length\_px} \times \left(1.0 + \frac{\text{mean\_contrast}}{50.0}\right) \times \ln\left(1 + \frac{\text{length\_px}}{\max(\text{thinness}, 1.0)}\right)$$
     where $\text{thinness} = \text{area\_px} / \max(\text{length\_px}, 1.0)$. Prioritizes long, high-contrast, narrow structural fissures while cleanly separating secondary micro-cracks (`secondary_count`).
  5. Multi-Region Centroid Spread Threshold (`0.60` or 60%): When `roi_mask is None` and $\ge 2$ candidates are detected with centroid bounding spans $\Delta x > 0.60 \times W$ and $\Delta y > 0.60 \times H$, the scene is determined to contain disjoint candidate regions (e.g. wall feature plus road crack). The agent assigns verdict `human_review` with reason `"multiple separate regions found, ROI needed"`.
  6. API Form-Data Polygon Validation: `POST /inspect` validates user polygons to ensure $\ge 3$ coordinate pairs $[x, y]$ within image bounds $[0, W] \times [0, H]$, returning HTTP 422 if invalid.
- **Alternatives Considered:**
  - *Zeroing black-hat without boundary erosion:* Rejected because high-contrast border transitions between barrier walls and road pavement immediately trigger spurious edge contours.
  - *Using raw contour perimeter for primary crack selection:* Rejected because noisy fragmented edges inflate perimeter; oriented bounding length combined with thinness and contrast provides robust defect characterization.
- **Evidence / Validation Status:** Validated with pytest unit and integration tests across perception, agent, and API modules. Synthetic test cases confirm that cracks outside the ROI are rejected, boundary edges do not generate cracks, multi-region corner cracks trigger human review with ROI needed, and invalid API polygons yield HTTP 422.

---

## Decision 13: Skeleton-Based Primary Crack Dimension Measurement and Scale Calibration

- **Date:** 2026-10-24
- **Decision:** Implement `measure_primary_crack` using `skimage.morphology.skeletonize` for 1-px skeleton extraction, weighted path-length summation (diagonal steps = √2), and `cv2.distanceTransform` for maximum half-width estimation. Provide optional `mm_per_px` calibration passthrough from `POST /inspect` `scale_ref`.
- **Reason:** Bounding-box length (the existing `length_px`) systematically overestimates curved crack path length and cannot distinguish long, thin structural cracks from short, wide spall features. Skeleton path length captures crack morphology more accurately. The distance transform provides a physically meaningful maximum width at the widest cross-section. Named severity thresholds give actionable classification without false claims of structural danger.
- **Key Parameter Choices:**
  1. **Skeleton path length**: `skimage.morphology.skeletonize(mask > 0)` → 1-px connected skeleton. Forward-neighbour pairs `(dr, dc)` ∈ `{(0,1), (1,0), (1,-1), (1,1)}` each contribute `1.0` (orthogonal) or `√2` (diagonal). Avoids double-counting edges in an undirected graph traversal.
  2. **Maximum width**: `width_max_px = 2 × max(distanceTransform[skeleton_pixels])`. The distance transform gives the radius to the nearest background pixel; multiplying by 2 gives the local crack diameter at the widest point over the skeleton.
  3. **Mean width**: `width_mean_px = component_area_px / max(length_path_px, 1.0)`. Provides a stable aggregate cross-section estimate.
  4. **Severity thresholds** (named constants, not magic numbers):
     - `DEFAULT_SEVERITY_LOW_WIDTH = 3.0 px`: Hair-line cracks below 3 px max width.
     - `DEFAULT_SEVERITY_MEDIUM_WIDTH = 10.0 px`: Moderate cracks 3–10 px max width.
     - `HIGH` otherwise, or bumped from any base class if `length_path_px > 0.40 × image_width` (`DEFAULT_SEVERITY_LENGTH_RATIO_BUMP = 0.4`).
  5. **Scale calibration**: `POST /inspect` accepts optional `scale_ref` JSON `{"p1": [x1, y1], "p2": [x2, y2], "real_mm": float}`. API validates points inside image bounds, distinctness (pixel distance > 0), and `real_mm > 0`, returning HTTP 422 on failure. `mm_per_px = real_mm / pixel_dist` is then passed through `inspect()` → `detect_cracks()` → `measure_primary_crack()`. Length and width mm values are returned alongside px values. A disclaimer note warns that perspective distortion applies.
- **Constraints and Non-Claims:**
  - Pixel measurements are not physical measurements unless calibration is provided. Even with calibration, perspective projection distorts distances away from the reference plane; this limitation is disclosed in the UI and API documentation.
  - Severity thresholds are engineering guidance heuristics, not structural safety determinations. Sentry never claims a detected crack is structurally dangerous.
- **Alternatives Considered:**
  - *Contour perimeter length:* Rejected because jagged fragmented contours inflate measured lengths, and contour connectivity is not guaranteed after morphological operations.
  - *Bounding ellipse major axis:* Rejected because elongated but curved cracks do not fit well to ellipses; the skeleton path captures tortuous morphology.
  - *Canny edge-based width estimation:* Rejected as fragile to lighting variation; distance transform on the already-filtered component mask is more stable.
- **Evidence / Validation Status:** Prototype validated via CLI script (results: 100×4 bar → `length_path_px ≈ 96.4`, `width_mean ≈ 4.15`; curved 45° line → `length_path_px > bbox_length_px`; `mm_per_px=0.5` → mm values = px values × 0.5). Confirmed by pytest `TestMeasurePrimaryCrack` (9 synthetic tests) and `TestScaleRefValidation` (8 API tests), verified alongside the 62 previously passing tests.

---

## Decision 14: Automatic Surface Region-of-Interest (ROI) Detection and Zero-Manual-Input Inspection

- **Date:** 2026-10-26
- **Decision:** Implement autonomous Region-of-Interest (ROI) surface detection and a zero-manual-input inspection workflow across the perception engine, agent planner, REST API, and Streamlit dashboard. Remove manual ROI coordinate inputs ($X_{\min}, Y_{\min}, X_{\max}, Y_{\max}$) and manual scale coordinate entries ($X_1, Y_1, X_2, Y_2, \text{real\_mm}$) from the normal operational workflow. Users simply upload a photograph and click **Run Inspection**; Sentry autonomously validates quality, segments the concrete/road surface, inspects for crack fissures, extracts skeleton-based centerline and width profiles in pixels, and delivers audited verdicts.
- **Surface Segmentation Architecture:**
  1. **Background & Surrounding Feature Rejection:**
     - *Sky / Atmospheric Blowout:* Blue sky segmented in HSV space ($H \in [90, 135], S \ge 25, V \ge 50$); bright overexposure highlights detected via high luminance ($V \ge 248$) combined with low Laplacian texture variance ($|\nabla^2 I| < 15.0$).
     - *Vegetation / Foliage:* Foliage detected via HSV green gamut ($H \in [35, 85], S \ge 35, V \ge 30$).
     - *Deep Shadow Voids & Cavities:* Non-surface darkness ($V < 25$) excluded to eliminate dark scene voids adjacent to structural columns or underpasses.
     - *Lateral High-Contrast Obstacles:* Peripheral vertical strips ($< 25\%$ or $> 75\%$ width) with significant multimodal luminance shifts evaluated via Otsu thresholding to segment dark roadside barrier walls or column edge drop-offs.
  2. **Morphological Regularization & Dominant Component Extraction:**
     - Candidate surface binary mask undergoes morphological closing ($15 \times 15$ kernel) to bridge internal fissures and surface aggregate texture, followed by opening ($9 \times 9$ kernel) to prune noise tendrils.
     - Connected component analysis selects the dominant contiguous surface component.
  3. **Triage Branches & Transparent Fallback:**
     - *Case A (Uniform Surface):* If coverage exceeds $92\%$ or zero background categories are excluded (e.g. macro close-up of a slab), the system applies whole-image inspection without artificial boundaries (`surface_type: "uniform_surface"`).
     - *Case B (Fallback Guarantee):* If the dominant surface covers $< 20\%$ of the image area or boundaries are excessively fragmented, Sentry defaults to whole-image inspection with an explicit diagnostic warning (`fallback: True`, `warning: "Automatic ROI detection found insufficient contiguous surface (...); defaulting to whole-image inspection to prevent missing defects."`). Defects are never silently excluded.
     - *Case C (Segmented Surface):* When a clear concrete/road surface is identified ($20\% \le \text{coverage} < 92\%$), internal holes are closed so cracks within the surface remain inspectable, the boundary contour is simplified with Douglas-Peucker polygon approximation for API/UI visualization, and inspection is restricted to the active surface.
- **Zero-Manual-Scale & Measurement Presentation Policy:**
  - Real-world millimetre or centimetre measurements are NEVER fabricated or guessed from pixel dimensions alone. Without verified physical calibration markers, dimensions are strictly presented in pixels (`px`).
  - The dashboard and API explicitly communicate: *"Real-world scale unavailable; measurements are in pixels."*
  - Bounding-box diagonal length is deprecated as the primary length metric; centerline length is estimated from a 1-pixel morphological skeleton with diagonal step weighting ($\sqrt{2}$), accompanied by mean width ($\text{area} / \text{path\_length}$) and maximum width ($2 \times \text{peak distance transform}$).
  - Defect area percentage is computed relative to the active inspected surface ROI.
  - Severity categories (`low`, `medium`, `high`) are documented as project-defined triage heuristics and explicitly disclaimed as NOT constituting structural engineering safety assessments.
- **Coordinate Space Integrity:**
  - All masks (`roi_mask`, `mask_primary`, `mask_all`) and bounding boxes are maintained 1:1 in original image space $(H, W)$. No sub-image coordinate translations or offsets are required, eliminating coordinate drift.
- **Constraints and Limitations:**
  - *Perspective Distortions:* Surface tilt, incline, and distance gradients distort pixel density away from the lens. Without orthorectification or ground-truth calibration targets, pixel widths cannot be linearly converted to physical crack widths.
  - *Heuristic Bounds:* Complex scenes with extreme moving shadows, dense debris, or non-concrete objects (vehicles, construction equipment) may trigger the transparent fallback branch, ensuring no potentially critical fissures are masked.
- **Evidence / Validation Status:** Verified with 12 new synthetic and real-world test cases in `tests/test_auto_roi.py` (horizontal, vertical, diagonal cracks, clean surface, cluttered scenes with sky/vegetation/voids, fallback triggers, 1:1 coordinate alignment, real column photograph `column_crack.jpg`, and zero-manual-input API backward compatibility). Full suite of 104 tests passing cleanly.
