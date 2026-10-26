"""
src/perception/quality.py

Image quality assessment and conservative enhancement for the Sentry inspection system.
Implements blur detection via Laplacian variance, exposure (brightness) checking,
contrast assessment, and CLAHE-based contrast enhancement.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Default quality evaluation thresholds
DEFAULT_BLUR_THRESHOLD: float = 100.0
DEFAULT_MIN_BRIGHTNESS: float = 40.0
DEFAULT_MAX_BRIGHTNESS: float = 220.0
DEFAULT_MIN_CONTRAST: float = 20.0


def _validate_and_to_gray(img: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Validates input image array and extracts a uint8 grayscale representation.

    Parameters
    ----------
    img : Any
        Input image object expected to be a non-empty numpy.ndarray.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (validated_original_img, uint8_grayscale_img).

    Raises
    ------
    TypeError
        If img is not a numpy.ndarray.
    ValueError
        If img is None, empty, has invalid dimensions, or invalid channel count.
    """
    if img is None:
        raise ValueError("Input image cannot be None.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Input image must be a numpy.ndarray, got {type(img).__name__}.")

    if img.size == 0:
        raise ValueError("Input image is empty (size 0).")

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input image must have 2 or 3 dimensions. Received shape: {img.shape}."
        )

    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img.shape}.")

    # Convert to standard uint8 if necessary
    processed_img = img
    if processed_img.dtype != np.uint8:
        if np.issubdtype(processed_img.dtype, np.floating):
            # Scale float [0.0, 1.0] to [0, 255] if values are within unit range
            if processed_img.max() <= 1.0 and processed_img.min() >= 0.0:
                processed_img = (processed_img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                processed_img = processed_img.clip(0, 255).astype(np.uint8)
        else:
            processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

    # Derive grayscale channel
    if processed_img.ndim == 2:
        gray = processed_img
    else:
        channels = processed_img.shape[2]
        if channels == 1:
            gray = processed_img[:, :, 0]
        elif channels == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(processed_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}.")

    return processed_img, gray


def check_quality(
    img: np.ndarray,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
) -> Dict[str, Any]:
    """
    Assesses image optical quality metrics: blur, brightness, and contrast.

    Parameters
    ----------
    img : np.ndarray
        Input image as a 2D grayscale or 3D color numpy array.
    blur_threshold : float
        Minimum acceptable Laplacian variance score. Scores below this are flagged blurry.
    min_brightness : float
        Minimum acceptable average intensity (0-255). Below this is flagged underexposed.
    max_brightness : float
        Maximum acceptable average intensity (0-255). Above this is flagged overexposed.
    min_contrast : float
        Minimum acceptable standard deviation of intensity (0-255).

    Returns
    -------
    dict
        Dictionary containing:
        - 'ok': bool, True if all quality criteria pass without issue.
        - 'blur_score': float, Laplacian variance score.
        - 'brightness': float, Mean grayscale pixel intensity.
        - 'contrast': float, Standard deviation of grayscale pixel intensity.
        - 'issues': list[str], Explanatory list of detected quality deficiencies.
    """
    _, gray = _validate_and_to_gray(img)

    # 1. Blur Score: Laplacian variance (with localized patch assessment for smooth surfaces)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    global_blur_score = float(laplacian.var())
    blur_score = global_blur_score

    if global_blur_score < blur_threshold:
        # Check if localized regions contain sharp structural features (e.g. sharp crack on smooth concrete)
        th_h = max(gray.shape[0] // 6, 16)
        th_w = max(gray.shape[1] // 6, 16)
        max_local_var = 0.0
        for r in range(0, gray.shape[0] - th_h + 1, th_h):
            for c in range(0, gray.shape[1] - th_w + 1, th_w):
                tile = gray[r : r + th_h, c : c + th_w]
                tile_var = float(cv2.Laplacian(tile, cv2.CV_64F).var())
                if tile_var > max_local_var:
                    max_local_var = tile_var
        if max_local_var >= max(blur_threshold * 1.2, 100.0):
            blur_score = max_local_var

    # 2. Brightness: Mean intensity
    brightness = float(np.mean(gray))

    # 3. Contrast: Standard deviation of intensity (RMS contrast)
    contrast = float(np.std(gray))
    if contrast < min_contrast:
        # Check if localized regions contain sharp contrast defects (e.g. dark crack on smooth painted surface)
        th_h = max(gray.shape[0] // 6, 16)
        th_w = max(gray.shape[1] // 6, 16)
        max_local_std = 0.0
        for r in range(0, gray.shape[0] - th_h + 1, th_h):
            for c in range(0, gray.shape[1] - th_w + 1, th_w):
                tile_std = float(np.std(gray[r : r + th_h, c : c + th_w]))
                if tile_std > max_local_std:
                    max_local_std = tile_std
        if max_local_std >= min_contrast and (int(gray.max()) - int(gray.min())) >= 50:
            contrast = max_local_std

    # 4. Identify issues
    issues: List[str] = []

    if blur_score < blur_threshold:
        issues.append(
            f"Image is too blurry: Laplacian variance is {blur_score:.2f} "
            f"(minimum threshold: {blur_threshold:.2f})."
        )

    if brightness < min_brightness:
        issues.append(
            f"Image is underexposed / too dark: average brightness is {brightness:.2f} "
            f"(minimum threshold: {min_brightness:.2f})."
        )
    elif brightness > max_brightness:
        issues.append(
            f"Image is overexposed / too bright: average brightness is {brightness:.2f} "
            f"(maximum threshold: {max_brightness:.2f})."
        )

    if contrast < min_contrast:
        issues.append(
            f"Image has low contrast: standard deviation is {contrast:.2f} "
            f"(minimum threshold: {min_contrast:.2f})."
        )

    return {
        "ok": len(issues) == 0,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "issues": issues,
    }


def enhance_image(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies conservative contrast and feature enhancement to an image.
    Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance
    channel to avoid creating harsh artificial artifacts or saturating noise.

    Parameters
    ----------
    img : np.ndarray
        Input image as a 2D grayscale or 3D BGR color numpy array.
    clip_limit : float
        Threshold for contrast limiting in CLAHE (default: 2.0).
    tile_grid_size : tuple[int, int]
        Size of grid for histogram equalization (default: (8, 8)).

    Returns
    -------
    np.ndarray
        Enhanced image with identical dimensions, channels, and uint8 dtype.
    """
    validated_img, _ = _validate_and_to_gray(img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if validated_img.ndim == 2:
        # Grayscale image
        enhanced = clahe.apply(validated_img)
    elif validated_img.shape[2] == 1:
        # Single-channel 3D image
        enhanced_2d = clahe.apply(validated_img[:, :, 0])
        enhanced = np.expand_dims(enhanced_2d, axis=2)
    elif validated_img.shape[2] == 3:
        # 3-channel color image (assumed BGR format)
        # Convert to LAB to isolate luminance from chromaticity
        lab = cv2.cvtColor(validated_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = clahe.apply(l_channel)
        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    elif validated_img.shape[2] == 4:
        # 4-channel BGRA image: preserve alpha
        bgr = validated_img[:, :, :3]
        alpha = validated_img[:, :, 3]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = clahe.apply(l_channel)
        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.merge((enhanced_bgr[:, :, 0], enhanced_bgr[:, :, 1], enhanced_bgr[:, :, 2], alpha))
    else:
        raise ValueError(f"Unsupported channel count: {validated_img.shape[2]}")

    return enhanced
