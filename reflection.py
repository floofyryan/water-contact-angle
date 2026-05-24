"""
Reflection masking for reflective substrates (PDMS, glass, polished metal).

On reflective surfaces, the sessile drop produces a mirror image below
the baseline. This causes two problems:

  1. Baseline detection: Hough may find the BOTTOM of the reflection
     (a strong horizontal edge) instead of the actual substrate line.
  2. Edge contamination: if baseline_y is set slightly too low, reflected
     drop edges leak into the drop region and distort the angle fit.

This module adds:
  - detect_reflection_depth(): finds how far below the baseline the
    reflection extends by looking for the strongest horizontal gradient row.
  - suppress_reflection(): blurs the reflection band in the raw gray image
    so Canny ignores it.
  - robust_baseline_reflective(): baseline detector that handles surfaces
    where the reflection produces a second strong horizontal feature.

Integration
-----------
Pass suppress_reflection=True to ContactAngleAnalyzer, or call the
functions here directly before passing the image to the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def detect_reflection_depth(
    gray: np.ndarray,
    baseline_y: int,
    max_search_px: int = 150,
    smoothing: int = 5,
) -> int:
    """
    Estimate how far below the baseline the reflection extends.

    Strategy: the bottom of the reflected drop produces a horizontal
    band of strong gradient (a second "baseline-like" feature).
    We search for the strongest gradient row in the region
    [baseline_y, baseline_y + max_search_px].

    Args:
        gray          : Grayscale image (already oriented).
        baseline_y    : Row of the real baseline.
        max_search_px : How far below baseline to search.
        smoothing     : Median filter half-width to reduce noise.

    Returns:
        depth in pixels (distance from baseline_y to reflection floor).
        Returns 0 if no clear reflection is found.
    """
    h = gray.shape[0]
    y_end = min(h, baseline_y + max_search_px)
    region = gray[baseline_y:y_end, :].astype(np.float32)

    if region.shape[0] < 5:
        return 0

    # Row-averaged horizontal gradient magnitude
    grad_y = np.abs(np.diff(region, axis=0))
    row_grad = grad_y.mean(axis=1)

    # Smooth to avoid noise spikes
    if smoothing > 1 and len(row_grad) > smoothing:
        row_grad = np.convolve(
            row_grad,
            np.ones(smoothing) / smoothing,
            mode="same"
        )

    # Peak gradient row below the baseline = bottom of reflection
    peak_row = int(np.argmax(row_grad))

    # Only return a depth if the peak is significantly above background
    if row_grad[peak_row] < row_grad.mean() * 1.5:
        return 0   # no clear reflection floor found

    # peak_row is the index into the diff array, corresponding to the
    # transition between rows peak_row and peak_row+1 in `region`.
    # The depth from baseline_y to that boundary is therefore peak_row+1.
    return peak_row + 1


def suppress_reflection(
    gray: np.ndarray,
    baseline_y: int,
    reflection_depth: Optional[int] = None,
    blur_ksize: int = 21,
    max_search_px: int = 150,
) -> np.ndarray:
    """
    Blur the reflection band so Canny ignores it.

    The real drop edges above the baseline are untouched. Only the band
    [baseline_y, baseline_y + reflection_depth] is blurred.

    Args:
        gray             : Preprocessed grayscale image.
        baseline_y       : Row of the real baseline.
        reflection_depth : Depth of reflection in pixels. If None,
                           auto-detected via detect_reflection_depth().
        blur_ksize       : Kernel size for the suppression blur (odd).
        max_search_px    : Search range for auto-detection.

    Returns:
        Modified grayscale image with reflection band blurred.
    """
    if reflection_depth is None:
        reflection_depth = detect_reflection_depth(
            gray, baseline_y, max_search_px=max_search_px
        )

    if reflection_depth <= 0:
        return gray   # nothing to suppress

    result = gray.copy()
    h = gray.shape[0]
    y_start = baseline_y
    y_end = min(h, baseline_y + reflection_depth)

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    region = result[y_start:y_end, :]
    result[y_start:y_end, :] = cv2.GaussianBlur(
        region, (blur_ksize, blur_ksize), 0
    )
    return result


def robust_baseline_reflective(
    gray: np.ndarray,
    search_fraction: float = 0.25,
    hough_threshold: Optional[int] = None,
    min_line_gap: int = 20,
) -> int:
    """
    Baseline detection for reflective substrates.

    On PDMS/glass, two strong horizontal lines often appear:
      - The actual substrate surface (higher in the image = smaller y)
      - The bottom of the reflection (lower in the image = larger y)

    This function detects ALL near-horizontal lines in the search region,
    then returns the TOPMOST one (smallest y), which corresponds to the
    real substrate surface. The reflection floor is always below it.

    Args:
        gray             : Preprocessed grayscale image (already oriented).
        search_fraction  : Fraction of image height to search from bottom.
        hough_threshold  : Hough vote threshold (auto if None).
        min_line_gap     : Minimum gap between two candidate lines (px)
                           to consider them distinct features.

    Returns:
        Row index of the detected baseline.
    """
    h, w = gray.shape
    y_start = int(h * (1 - search_fraction))
    region = gray[y_start:, :]

    threshold = hough_threshold or max(w // 6, 30)
    edges = cv2.Canny(region, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=w // 5,
        maxLineGap=30,
    )

    if lines is None:
        # Fallback: gradient-based
        # +1: argmax of diff[i] marks the boundary between rows i and i+1
        grad_y = np.abs(np.diff(gray[y_start:].astype(np.float32), axis=0))
        return y_start + int(np.argmax(grad_y.mean(axis=1))) + 1

    # Collect near-horizontal line y-positions
    y_vals = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 8:
            y_vals.append(int((y1 + y2) / 2 + y_start))

    if not y_vals:
        return y_start

    y_vals = sorted(set(y_vals))

    # Cluster close y-values and take the topmost cluster centroid
    # (real baseline is always above its reflection)
    clusters = [[y_vals[0]]]
    for y in y_vals[1:]:
        if y - clusters[-1][-1] <= min_line_gap:
            clusters[-1].append(y)
        else:
            clusters.append([y])

    # Return centroid of the topmost cluster
    topmost = clusters[0]
    return int(np.mean(topmost))
