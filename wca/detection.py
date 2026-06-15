"""
detection.py — Baseline detection, edge detection, and contour extraction.

These functions form the middle layer of the analysis pipeline:

    preprocessing.py  →  detection.py  →  fitting.py

Functions
---------
detect_baseline_auto()   : Find the substrate row from a preprocessed image.
detect_edges()           : Run Canny edge detection, masked to the ROI above baseline.
extract_contour_points() : Turn a binary edge image into (xs, ys) point arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ── Baseline auto-detection ───────────────────────────────────────────────────

def detect_baseline_auto(
    gray: np.ndarray,
    search_fraction: float = 0.20,
    hough_threshold: Optional[int] = None,
) -> int:
    """
    Estimate the substrate baseline row from a preprocessed grayscale image.

    Strategy
    --------
    1. Search the bottom ``search_fraction`` of the image.
    2. Try Hough line segments — if enough near-horizontal lines are found,
       use their median y-coordinate.
    3. Fallback: find the row with the strongest mean horizontal gradient
       (sharpest dark-to-light transition) in the search zone.

    Args:
        gray             : Preprocessed grayscale image (oriented).
        search_fraction  : Fraction of image height to search from bottom.
        hough_threshold  : Hough vote threshold (auto-set if None).

    Returns:
        Row index (int) of the estimated baseline.
    """
    h, w = gray.shape
    y_start = int(h * (1.0 - search_fraction))
    region  = gray[y_start:, :]

    threshold = hough_threshold or max(w // 8, 20)
    edges_h   = cv2.Canny(region, 50, 150)
    lines     = cv2.HoughLinesP(
        edges_h, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=w // 8,
        maxLineGap=25,
    )

    if lines is not None:
        y_vals = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:   # roughly horizontal
                y_vals.append(int((y1 + y2) / 2 + y_start))
        if len(y_vals) >= 3:
            return int(np.median(y_vals))

    # Gradient-based fallback
    grad = np.abs(np.diff(gray[y_start:].astype(np.float32), axis=0))
    if grad.shape[0] == 0:
        return y_start
    peak = int(np.argmax(grad.mean(axis=1)))
    return y_start + peak + 1   # +1: diff boundary is between rows peak and peak+1


# ── Edge detection ────────────────────────────────────────────────────────────

def detect_edges(
    gray: np.ndarray,
    baseline_y: int,
    canny_low: int = 30,
    canny_high: int = 100,
    baseline_margin: int = 5,
    suppress_reflections: bool = True,
) -> np.ndarray:
    """
    Run Canny edge detection, restricted to the drop/bubble region above baseline.

    Pixels at or below (baseline_y − baseline_margin) are zeroed so that
    substrate texture and reflections below the line do not contaminate the fit.

    suppress_reflections removes edges that border bright specular reflections
    (the white circle visible inside some drops under direct lighting).  It
    dilates any pixel brighter than 220/255 by 9 px and zeros those edges.

    Args:
        gray                  : Preprocessed grayscale image (oriented).
        baseline_y            : Row index of the substrate baseline.
        canny_low             : Canny lower hysteresis threshold.
        canny_high            : Canny upper hysteresis threshold.
        baseline_margin       : Extra rows to exclude below baseline_y.
        suppress_reflections  : Remove edges adjacent to bright reflection zones.

    Returns:
        Binary uint8 edge image (255 = edge, 0 = background).
    """
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Mask out the substrate region and everything below it
    cutoff = max(0, baseline_y - baseline_margin)
    edges[cutoff:, :] = 0

    if suppress_reflections:
        # Bright specular reflections inside the drop create a spurious inner
        # ring of edges.  We remove edges bordering bright zones — but ONLY
        # small bright blobs (the specular highlight), NOT a large bright
        # background (back-lit images), which would otherwise erase the drop
        # outline itself.
        bright = (gray > 220).astype(np.uint8)
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        img_area = gray.shape[0] * gray.shape[1]
        small_bright = np.zeros_like(bright)
        for lbl in range(1, n_lbl):
            area = stats[lbl, cv2.CC_STAT_AREA]
            # Specular highlights are small; a back-lit background is large.
            if area < 0.08 * img_area:
                small_bright[labels == lbl] = 1
        if small_bright.any():
            bright_zone = cv2.dilate(small_bright, np.ones((9, 9), np.uint8))
            edges[bright_zone > 0] = 0

    return edges


# ── Contour point extraction ──────────────────────────────────────────────────

def extract_contour_points(
    edges: np.ndarray,
    min_points: int = 10,
    baseline_y: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (x, y) edge coordinates, selecting only the drop/bubble contour.

    Strategy:
    1. Find all contours in the edge image.
    2. Among contours whose lowest point reaches within 30 px of baseline_y,
       keep the widest one (largest x-span) — this is the outer drop boundary.
    3. If no contour reaches the baseline, fall back to the largest by length.
    4. If findContours returns nothing, fall back to all non-zero pixels.

    This two-stage selection reliably discards inner specular-reflection circles
    (they float above the baseline and are narrower than the outer boundary).

    Args:
        edges      : Binary edge image (uint8, values 0 or 255).
        min_points : Raise ValueError if fewer points are found.
        baseline_y : Baseline row — used to prefer contours that reach it.

    Returns:
        (xs, ys) : float64 arrays of x and y coordinates.

    Raises:
        ValueError if fewer than min_points are found.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if contours:
        if baseline_y is not None:
            # Prefer the contour whose bottom row is closest to the baseline
            # and whose horizontal span is widest (the outer drop profile).
            def _score(cnt):
                pts = cnt.reshape(-1, 2)
                bottom = pts[:, 1].max()
                span   = pts[:, 0].max() - pts[:, 0].min()
                # Contours that reach within 30 px of baseline_y get full span;
                # others are penalised by how far they are from the baseline.
                reach_bonus = span if abs(bottom - baseline_y) < 30 else 0
                return reach_bonus + len(cnt) * 0.01
            cnt = max(contours, key=_score)
        else:
            cnt = max(contours, key=len)

        pts = cnt.reshape(-1, 2)
        xs  = pts[:, 0].astype(np.float64)
        ys  = pts[:, 1].astype(np.float64)
    else:
        # Last-resort fallback: every non-zero pixel
        ys_px, xs_px = np.where(edges > 0)
        if len(xs_px) == 0:
            raise ValueError("No edge pixels found.")
        xs = xs_px.astype(np.float64)
        ys = ys_px.astype(np.float64)

    if len(xs) < min_points:
        raise ValueError(
            f"Too few edge points: {len(xs)} found, {min_points} required. "
            "Try adjusting Canny thresholds or baseline position."
        )

    return xs, ys
