"""
preprocessing.py — Image preparation for contact angle analysis.

Converts raw BGR camera frames or loaded images into a normalised
grayscale form that maximises edge contrast while suppressing noise.

Pipeline
--------
1. Convert to grayscale (if needed).
2. CLAHE — Contrast Limited Adaptive Histogram Equalisation.
   Boosts local contrast without blowing out highlights.
3. Gaussian blur — suppresses pixel noise before Canny edge detection.

Captive bubble images are vertically flipped so the substrate appears
at the bottom (sessile-like orientation), enabling the same analysis
pipeline for both modes.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to uint8 grayscale."""
    if image is None:
        raise ValueError("image is None")
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unexpected image shape: {image.shape}")


def preprocess(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    blur_ksize: int = 5,
    clahe_tile: tuple = (8, 8),
) -> np.ndarray:
    """
    Prepare an image for Canny edge detection.

    Args:
        image      : BGR or grayscale image.
        clahe_clip : CLAHE clip limit (higher = stronger contrast boost;
                     0 disables CLAHE).
        blur_ksize : Gaussian blur kernel size (must be odd; 1 = no blur).
        clahe_tile : CLAHE tile grid size.

    Returns:
        Preprocessed uint8 grayscale image of the same spatial size.
    """
    gray = to_gray(image)

    if clahe_clip > 0:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=clahe_tile,
        )
        gray = clahe.apply(gray)

    # Ensure kernel size is a positive odd integer
    ksize = max(1, int(blur_ksize))
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 1:
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    return gray


def flip_for_captive_bubble(image: np.ndarray) -> np.ndarray:
    """
    Flip a captive-bubble image vertically so the substrate is at the bottom.

    Captive bubble images typically have the flat substrate at the TOP
    (the needle/capillary punctures from below). Flipping makes the
    geometry equivalent to a sessile drop, allowing the same detection
    and fitting code to be used for both modes.

    Args:
        image : BGR or grayscale image in original camera orientation.

    Returns:
        Vertically flipped copy of the image.
    """
    return cv2.flip(image, 0)
