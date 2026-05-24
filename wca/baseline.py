"""
baseline.py — Uneven / tilted baseline handling.

Real substrates are rarely perfectly horizontal. A 1–2° tilt introduces
a systematic error of ~1–3° in the contact angle. This module:

  1. Detects the substrate baseline as a set of (x, y) points along its
     full width using Hough segments or column-wise gradient peaks.

  2. Fits a polynomial y = f(x) to those points (degree 1 = line,
     degree 2 = parabola for slightly curved holders).

  3. Returns a PolyBaseline object that gives:
       • y_at(x)              local baseline height
       • tilt_deg_at(x)       local tilt angle (°)
       • correct_angle(θ, x)  subtract local tilt from a measured angle
       • sample_points()      evenly spaced (x, y) pairs for visualisation

  4. A helper straighten_image() rotates the image so the fitted
     baseline becomes horizontal — the simplest way to integrate with
     the rest of the pipeline.

Usage — Option A (straighten and proceed normally)
---------------------------------------------------
    from contact_angle.baseline import detect_baseline_poly, straighten_image

    poly_bl = detect_baseline_poly(preprocessed, search_fraction=0.15)
    straight_img, straight_bl_y = straighten_image(oriented_img, poly_bl)
    # Now analyse straight_img with baseline_y = straight_bl_y as usual

Usage — Option B (correct angle post-hoc)
------------------------------------------
    poly_bl = detect_baseline_poly(preprocessed)
    result  = analyzer.analyze(image, baseline_y=int(poly_bl.y_at(w/2)))
    x_left  = result['x_left']
    theta_corrected = poly_bl.correct_angle(result['theta_left'], x_left)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


class PolyBaseline:
    """
    A polynomial baseline y = f(x) fitted to the substrate edge.

    All coordinates are in the oriented (possibly flipped) image space.
    """

    def __init__(self, coeffs: np.ndarray, x_range: Tuple[float, float]):
        self._poly  = np.poly1d(coeffs)
        self._dpoly = self._poly.deriv()
        self.x_range = x_range         # (x_min, x_max) where the fit is valid
        self.degree  = len(coeffs) - 1

    # ── Evaluation ────────────────────────────────────────────────────────────

    def y_at(self, x: float) -> float:
        """Baseline y-coordinate at a given x (pixels)."""
        return float(self._poly(x))

    def tilt_deg_at(self, x: float) -> float:
        """
        Local tilt of the baseline (degrees from horizontal) at x.
        Positive = baseline slopes downward to the right (y increases with x).
        """
        return float(math.degrees(math.atan(float(self._dpoly(x)))))

    def correct_angle(
        self,
        theta_measured: float,
        x_contact: float,
        side: str = 'left',
    ) -> float:
        """
        Return the true contact angle corrected for local baseline tilt.

        On a surface tilted by angle α (positive = right side is lower in
        y-down image coordinates, i.e. baseline slopes downward to the right):
          θ_true_left  = θ_measured_left  − tilt(x_L)
          θ_true_right = θ_measured_right + tilt(x_R)

        Args:
            theta_measured : Raw measured contact angle (degrees).
            x_contact      : x-coordinate of the contact point.
            side           : 'left' (subtracts tilt) or 'right' (adds tilt).

        Returns:
            Corrected contact angle (degrees), clamped to [0°, 180°].
        """
        tilt = self.tilt_deg_at(x_contact)
        corrected = theta_measured - tilt if side == 'left' else theta_measured + tilt
        return float(np.clip(corrected, 0.0, 180.0))

    def correct_pair(
        self,
        theta_left: Optional[float],
        theta_right: Optional[float],
        x_left: Optional[float],
        x_right: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Correct both contact angles for baseline tilt simultaneously.

        On a surface tilted by angle α (positive = right side lower):
          • The advancing side (lower) reads too high  → subtract α
          • The receding side (higher) reads too low   → add α

        Equivalently, for left and right sides at their respective x positions:
          θ_L_true = θ_L − tilt(x_L)
          θ_R_true = θ_R + tilt(x_R)

        Sign convention: tilt_deg_at(x) = atan(dy/dx) in image coordinates
        (y increases downward). A positive tilt means the baseline slopes
        downward to the right, so the right side of the substrate is lower.
        This matches the physical interpretation: the right contact point is
        on a lower part of the substrate, causing it to read higher than the
        true angle, hence we ADD the tilt to correct it.

        Returns (theta_left_corrected, theta_right_corrected).
        """
        tl, tr = None, None
        if theta_left is not None and x_left is not None:
            t = self.tilt_deg_at(x_left)
            tl = float(np.clip(theta_left - t, 0, 180))
        if theta_right is not None and x_right is not None:
            t = self.tilt_deg_at(x_right)
            tr = float(np.clip(theta_right + t, 0, 180))
        return tl, tr

    def sample_points(self, n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return n evenly-spaced (x, y) pairs along the fitted baseline."""
        xs = np.linspace(self.x_range[0], self.x_range[1], n)
        ys = self._poly(xs)
        return xs, ys

    def mean_y(self) -> float:
        """Average baseline y across the image width (useful as a scalar estimate)."""
        xs, ys = self.sample_points()
        return float(ys.mean())

    def max_deviation_px(self) -> float:
        """Maximum deviation from the mean y across the image width."""
        xs, ys = self.sample_points()
        return float(np.max(np.abs(ys - ys.mean())))

    def __repr__(self) -> str:
        tilt = self.tilt_deg_at((self.x_range[0] + self.x_range[1]) / 2)
        return (f"PolyBaseline(degree={self.degree}, "
                f"tilt≈{tilt:.2f}°, max_dev={self.max_deviation_px():.1f}px)")


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_baseline_poly(
    gray: np.ndarray,
    search_fraction: float = 0.20,
    degree: int = 1,
    n_columns: int = 40,
    column_search_px: int = 30,
    hough_threshold: Optional[int] = None,
    min_points: int = 6,
) -> PolyBaseline:
    """
    Detect the substrate edge as a polynomial y = f(x).

    Two strategies are tried in order:

    1. Hough line segments — finds straight baseline segments, uses their
       (x, y) endpoints as control points for the polynomial fit.

    2. Column-wise gradient peaks — for each of `n_columns` vertical strips,
       find the row with the strongest downward intensity gradient within
       the search zone. Robust when Hough finds nothing.

    Args:
        gray             : Pre-processed grayscale image (oriented).
        search_fraction  : Fraction of image height to search from the bottom.
        degree           : Polynomial degree (1 = line, 2 = parabola).
        n_columns        : Number of vertical strips for column-wise strategy.
        column_search_px : Height of search window per column (pixels).
        hough_threshold  : Minimum Hough votes (auto if None).
        min_points       : Minimum control points needed for reliable fit.

    Returns:
        PolyBaseline fitted to the detected substrate edge.
    """
    h, w = gray.shape
    y_start = int(h * (1 - search_fraction))
    region  = gray[y_start:, :]

    control_pts: list = []   # list of (x, y) in FULL image coords

    # ── Strategy 1: Hough ─────────────────────────────────────────────────────
    threshold = hough_threshold or max(w // 8, 20)
    edges     = cv2.Canny(region, 50, 150)
    lines     = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=threshold,
        minLineLength=w // 8,
        maxLineGap=25,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:   # roughly horizontal
                control_pts.append((x1, y1 + y_start))
                control_pts.append((x2, y2 + y_start))

    # ── Strategy 2: column-wise gradient peaks ────────────────────────────────
    if len(control_pts) < min_points:
        col_width = max(1, w // n_columns)
        for col in range(n_columns):
            x_lo = col * col_width
            x_hi = min(w, x_lo + col_width)
            strip = gray[y_start:, x_lo:x_hi]
            grad  = np.abs(np.diff(strip.astype(np.float32), axis=0)).mean(axis=1)

            if len(grad) < 2:
                continue

            peak = int(np.argmax(grad))
            x_mid = (x_lo + x_hi) / 2
            # +1: argmax of diff[i] marks the boundary between rows i and i+1
            control_pts.append((x_mid, peak + 1 + y_start))

    if len(control_pts) < min_points:
        # Last resort: flat baseline from the single strongest gradient row
        grad_all = np.abs(np.diff(gray[y_start:].astype(np.float32), axis=0)).mean(axis=1)
        flat_y   = int(np.argmax(grad_all)) + 1 + y_start
        xs_flat  = np.array([0.0, float(w - 1)])
        ys_flat  = np.array([float(flat_y), float(flat_y)])
        coeffs   = np.polyfit(xs_flat, ys_flat, 1)
        return PolyBaseline(coeffs, (0.0, float(w - 1)))

    cp = np.array(control_pts)
    xs_cp = cp[:, 0];  ys_cp = cp[:, 1]

    # Remove outliers: points more than 2 std from the median y
    med = np.median(ys_cp)
    std = max(ys_cp.std(), 1.0)
    ok  = np.abs(ys_cp - med) < 2.5 * std
    xs_cp, ys_cp = xs_cp[ok], ys_cp[ok]

    if len(xs_cp) < 2:
        xs_cp = np.array([0.0, float(w - 1)])
        ys_cp = np.array([float(med), float(med)])

    actual_degree = min(degree, len(xs_cp) - 1)
    coeffs = np.polyfit(xs_cp, ys_cp, actual_degree)
    return PolyBaseline(coeffs, (float(xs_cp.min()), float(xs_cp.max())))


# ── Image straightening ───────────────────────────────────────────────────────

def straighten_image(
    image: np.ndarray,
    poly_baseline: PolyBaseline,
    target_y: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Rotate the image so the fitted baseline becomes horizontal.

    Only meaningful when the baseline is approximately linear (degree=1).
    For higher-degree baselines, rotation only corrects the average tilt;
    use correct_pair() for per-contact-point correction instead.

    Args:
        image        : BGR or grayscale image (oriented).
        poly_baseline: PolyBaseline from detect_baseline_poly().
        target_y     : Row to put the straightened baseline at.
                       Defaults to the mean y of the fitted baseline.

    Returns:
        (straightened_image, baseline_y_in_straightened_image)
    """
    h, w = image.shape[:2]
    cx   = w / 2

    # Rotation angle = negative of the mean tilt (to level the baseline)
    tilt_deg = poly_baseline.tilt_deg_at(cx)
    angle    = -tilt_deg

    # Rotate around image centre
    M     = cv2.getRotationMatrix2D((cx, h / 2), angle, 1.0)
    flags = cv2.INTER_LINEAR
    if len(image.shape) == 2:
        border = (int(image.mean()),)   # 1-tuple keeps type consistent with 3-channel branch
        straight = cv2.warpAffine(image, M, (w, h),
                                   flags=flags, borderValue=border)
    else:
        straight = cv2.warpAffine(image, M, (w, h),
                                   flags=flags,
                                   borderValue=tuple(int(c) for c in image.mean(axis=(0,1))))

    # New baseline y: the mean y of the original baseline after rotation
    bl_y_before = poly_baseline.mean_y()
    # After rotation by `angle` around (cx, h/2):
    #   y_new ≈ y_old (for small angles and points near cx)
    # More precisely, transform the centre-x baseline point:
    pt = np.array([[[cx, bl_y_before]]], dtype=np.float32)
    pt_rot = cv2.transform(pt, M)
    new_bl_y = int(round(float(pt_rot[0, 0, 1])))

    if target_y is not None:
        # Shift the image vertically so baseline lands at target_y
        shift_y = target_y - new_bl_y
        T = np.float32([[1, 0, 0], [0, 1, shift_y]])
        straight = cv2.warpAffine(straight, T, (w, h))
        new_bl_y = target_y

    return straight, new_bl_y


# ─── Captive-bubble specific baseline detection ───────────────────────────────

def detect_baseline_captive_bubble(
    gray_flipped: np.ndarray,
    search_centre: int,
    search_half_width: int = 40,
    n_columns: int = 40,
    min_points: int = 8,
    degree: int = 1,
    outlier_sigma: float = 2.0,
) -> PolyBaseline:
    """
    Detect a potentially tilted baseline for captive bubble images.

    Designed for the FLIPPED image (substrate now at bottom of frame).
    The baseline is the bright→dark transition going downward in the flipped
    image (water/bubble region above, dark substrate below).

    Rather than searching the whole image, it searches a narrow band around
    ``search_centre`` ± ``search_half_width``.  Use the output of
    ``find_substrate_boundary()`` as ``search_centre``.

    Strategy
    --------
    For each of ``n_columns`` vertical strips, find the row with the strongest
    downward intensity gradient (bright→dark) within the search band.
    Fit a polynomial y = f(x) of the given ``degree`` to those column peaks,
    with outlier rejection.

    Parameters
    ----------
    gray_flipped      : Grayscale image in flipped (sessile-like) orientation.
    search_centre     : Approximate baseline row from coarse detection.
    search_half_width : ± rows around search_centre to scan.
    n_columns         : Number of vertical strips.
    min_points        : Minimum valid column peaks for a reliable fit.
    degree            : Polynomial degree (1 = line, 2 = gentle curve).
    outlier_sigma     : Reject column peaks more than this many σ from median.

    Returns
    -------
    PolyBaseline fitted to the detected substrate edge.
    """
    h, w = gray_flipped.shape
    y_lo = max(0, search_centre - search_half_width)
    y_hi = min(h, search_centre + search_half_width)

    col_width = max(1, w // n_columns)
    xs_ctrl, ys_ctrl = [], []

    for col in range(n_columns):
        x_lo = col * col_width
        x_hi = min(w, x_lo + col_width)
        strip = gray_flipped[y_lo:y_hi, x_lo:x_hi].astype(float)

        if strip.shape[0] < 3:
            continue

        row_means = strip.mean(axis=1)
        # Bright→dark: look for the most negative gradient (sharpest drop)
        grad = np.diff(row_means)
        peak = int(np.argmin(grad))   # most negative = sharpest bright→dark
        xs_ctrl.append((x_lo + x_hi) / 2.0)
        # +1: argmin of diff[i] marks the boundary between rows i and i+1
        ys_ctrl.append(float(peak + 1 + y_lo))

    if len(xs_ctrl) < min_points:
        # Fallback: flat baseline at search_centre
        coeffs = np.polyfit([0, float(w - 1)],
                             [float(search_centre)] * 2, 1)
        return PolyBaseline(coeffs, (0.0, float(w - 1)))

    xs_a = np.array(xs_ctrl)
    ys_a = np.array(ys_ctrl)

    # Outlier rejection
    med = np.median(ys_a)
    std = max(float(ys_a.std()), 1.0)
    ok  = np.abs(ys_a - med) < outlier_sigma * std
    xs_a, ys_a = xs_a[ok], ys_a[ok]

    if len(xs_a) < 2:
        coeffs = np.polyfit([0, float(w - 1)],
                             [float(search_centre)] * 2, 1)
        return PolyBaseline(coeffs, (0.0, float(w - 1)))

    actual_degree = min(degree, len(xs_a) - 1)
    coeffs = np.polyfit(xs_a, ys_a, actual_degree)
    return PolyBaseline(coeffs, (float(xs_a.min()), float(xs_a.max())))


def find_substrate_boundary(
    gray_flipped: np.ndarray,
    search_lo_frac: float = 0.3,
    search_hi_frac: float = 0.8,
) -> int:
    """
    Coarse detection of the substrate boundary in a flipped captive-bubble image.

    Finds the brightest→darkest row transition (most negative row-mean
    gradient) in the fraction [search_lo_frac, search_hi_frac] of the image.

    Returns the row index of that transition.
    """
    h = gray_flipped.shape[0]
    lo = int(h * search_lo_frac)
    hi = int(h * search_hi_frac)
    row_means = gray_flipped[lo:hi, :].mean(axis=1).astype(float)
    grad      = np.diff(row_means)
    peak      = int(np.argmin(grad))
    # +1: argmin of diff[i] marks the boundary between rows i and i+1
    return lo + peak + 1
