"""
segmentation.py — Isolate the bubble/drop from background clutter.

Standard Canny edge detection picks up everything: substrate texture,
reflections, out-of-focus bubbles, debris.  This module uses a
threshold + morphology approach to find the ONE blob that is the
bubble/drop of interest, scored by multiple criteria:

  • Circularity     (bubbles/drops are round)
  • Area            (real bubble dominates the ROI)
  • Area ratio      (reject blobs that are tiny relative to the largest)
  • Proximity       (closest to the centre of the ROI horizontally)
  • Baseline dist   (bubble must be reasonably close to substrate)
  • Edge margin     (reject blobs whose centroid is near the image edge)
  • Intensity var   (real bubble has high variance: bright glare + dark ring)

Usage
-----
    from contact_angle.segmentation import segment_bubble

    xs, ys = segment_bubble(gray_oriented, baseline_y)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def segment_bubble(
    gray: np.ndarray,
    baseline_y: int,
    baseline_margin: int = 15,
    close_ksize: int = 15,
    close_iters: int = 3,
    min_circularity: float = 0.5,
    min_area_fraction: float = 0.005,
    min_area_ratio: float = 0.05,
    edge_margin_px: int = 20,
    invert: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolate the bubble (or drop) and return its outer contour.

    Works on the oriented (possibly flipped) grayscale image. The ROI is
    restricted to rows above the baseline so substrate texture is ignored.

    Parameters
    ----------
    gray              : Grayscale oriented image.
    baseline_y        : Row of the baseline in this image.
    baseline_margin   : Extra rows to exclude near the baseline.
    close_ksize       : Morphological closing kernel size (fills glare holes).
    close_iters       : Closing iterations.
    min_circularity   : Reject blobs with 4π·A/P² below this.
    min_area_fraction : Reject blobs smaller than this fraction of the ROI.
    min_area_ratio    : Reject blobs smaller than this fraction of the
                        LARGEST blob found (catches tiny artefacts).
                        Default 0.05 = must be ≥ 5% of the largest blob.
    edge_margin_px    : Reject blobs whose centroid is within this many
                        pixels of the left/right image edge.
                        Catches edge-hugging artefacts and holder shadows.
    invert            : True if feature is DARK on LIGHT background.

    Returns
    -------
    xs, ys : float64 arrays of contour point coordinates.

    Raises
    ------
    ValueError if no suitable blob is found.
    """
    h, w = gray.shape
    y_end = max(0, baseline_y - baseline_margin)
    roi   = gray[:y_end, :]

    if roi.shape[0] < 10:
        raise ValueError("ROI above baseline is too small. Check baseline_y.")

    # ── Threshold ──────────────────────────────────────────────────────────────
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(roi, 0, 255, flag | cv2.THRESH_OTSU)

    # ── Morphological closing: fill interior glare / gaps ─────────────────────
    if close_ksize > 0:
        ks = close_ksize if close_ksize % 2 == 1 else close_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                   iterations=close_iters)

    # ── Find all external contours ─────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No blobs found after thresholding.")

    roi_area = roi.shape[0] * roi.shape[1]
    roi_cx   = w / 2

    # ── Pre-filter: circularity, area fraction, edge margin ───────────────────
    max_area = max(cv2.contourArea(c) for c in contours)

    candidates = []
    rejected   = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / peri ** 2) if peri > 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            rejected.append((cnt, "zero_moment"))
            continue
        cx_b = M["m10"] / M["m00"]
        cy_b = M["m01"] / M["m00"]

        # Hard filters
        if circ < min_circularity:
            rejected.append((cnt, f"circularity={circ:.2f}"))
            continue
        if area < min_area_fraction * roi_area:
            rejected.append((cnt, f"too_small_abs={area:.0f}"))
            continue
        if area < min_area_ratio * max_area:
            rejected.append((cnt, f"too_small_rel={area/max_area:.2%}"))
            continue
        if cx_b < edge_margin_px or cx_b > (w - edge_margin_px):
            rejected.append((cnt, f"edge_cx={cx_b:.0f}"))
            continue

        # Intensity variance inside blob (main bubble has high variance).
        # Restrict sampling to the ROI to avoid contamination from below-
        # baseline pixels if morphological closing expanded the contour.
        mask_b = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_b, [cnt], -1, 255, -1)
        pixels = gray[:y_end, :][mask_b[:y_end, :] > 0]
        inten_var = float(pixels.std()) if len(pixels) > 0 else 0.0

        # Solidity: real bubbles are convex and solid
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / hull_area if hull_area > 0 else 0.0

        candidates.append({
            "cnt":       cnt,
            "area":      area,
            "circ":      circ,
            "solidity":  solidity,
            "cx":        cx_b,
            "cy":        cy_b,
            "inten_var": inten_var,
        })

    if not candidates:
        n_rej = len(rejected)
        reasons = "; ".join(set(r for _, r in rejected[:5]))
        raise ValueError(
            f"No blob passed all filters ({n_rej} rejected: {reasons}). "
            "Try: lower min_circularity, lower min_area_ratio, or increase close_ksize."
        )

    # ── Score remaining candidates ────────────────────────────────────────────
    best_score   = -1.0
    best_contour = None

    max_cand_area = max(c["area"] for c in candidates)
    max_inten_var = max(c["inten_var"] for c in candidates) or 1.0

    for c in candidates:
        # Each score component in [0, 1]
        score_circ     = c["circ"]
        score_solidity = c["solidity"]
        score_area     = c["area"] / max_cand_area
        score_x        = 1.0 - abs(c["cx"] - roi_cx) / (w / 2)
        score_var      = c["inten_var"] / max_inten_var
        # Baseline proximity: favour blobs whose bottom edge is near baseline
        ys_cnt = c["cnt"][:, 0, 1]
        score_bl = min(1.0, ys_cnt.max() / y_end) if y_end > 0 else 0.5

        # Weighted composite — area and circularity matter most.
        # Denominator is computed from the actual weights so it stays
        # correct if any weight is ever changed.
        _W = 3.0 + 2.0 + 1.5 + 1.0 + 1.0 + 0.5   # = 9.0
        score = (
            score_area     * 3.0 +
            score_circ     * 2.0 +
            score_solidity * 1.5 +
            score_x        * 1.0 +
            score_bl       * 1.0 +
            score_var      * 0.5
        ) / _W

        if score > best_score:
            best_score   = score
            best_contour = c["cnt"]

    pts = best_contour.reshape(-1, 2)
    return pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)


def debug_segmentation(
    gray: np.ndarray,
    baseline_y: int,
    **kwargs,
) -> np.ndarray:
    """
    Return a BGR diagnostic image showing all blobs colour-coded:
      Green  = passes all filters (candidate)
      Red    = rejected (circularity, area, or edge margin)
      Orange = passes shape filters but rejected by area ratio

    Circularity score and rejection reason are annotated on each blob.
    """
    h, w = gray.shape
    y_end = max(0, baseline_y - kwargs.get("baseline_margin", 15))

    if y_end < 10:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "ROI too small — check baseline_y",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return vis

    roi   = gray[:y_end, :]
    flag = cv2.THRESH_BINARY_INV if kwargs.get("invert", True) else cv2.THRESH_BINARY
    _, binary = cv2.threshold(roi, 0, 255, flag | cv2.THRESH_OTSU)

    ks = kwargs.get("close_ksize", 15)
    if ks % 2 == 0: ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                               iterations=kwargs.get("close_iters", 3))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, baseline_y), (w - 1, baseline_y), (0, 255, 255), 1)

    if not contours:
        return vis

    min_circ     = kwargs.get("min_circularity", 0.5)
    min_area_f   = kwargs.get("min_area_fraction", 0.005)
    min_area_r   = kwargs.get("min_area_ratio", 0.05)
    edge_margin  = kwargs.get("edge_margin_px", 20)
    roi_area     = y_end * w
    max_area     = max(cv2.contourArea(c) for c in contours)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / peri ** 2) if peri > 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])

        # Determine status
        if circ < min_circ:
            col, label = (0, 0, 180), f"circ={circ:.2f}"
        elif area < min_area_f * roi_area:
            col, label = (0, 0, 180), f"tiny_abs"
        elif area < min_area_r * max_area:
            col, label = (0, 140, 255), f"tiny_rel={area/max_area:.1%}"
        elif cx_ < edge_margin or cx_ > (w - edge_margin):
            col, label = (0, 0, 180), f"edge"
        else:
            col, label = (0, 200, 0), f"circ={circ:.2f}"

        cv2.drawContours(vis, [cnt], -1, col, 2)
        cv2.putText(vis, label, (cx_ - 30, cy_),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    return vis
