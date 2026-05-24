"""
Contact angle calculation from contour edge points.

Method: PCA-based tangent estimation
-------------------------------------
Instead of fitting y = f(x) and computing slope (which breaks near 90° where
the curve is nearly vertical), we fit a line to the near-contact edge points
using Principal Component Analysis (SVD). The first principal component gives
the best-fit line direction independent of slope magnitude.

Coordinate system
-----------------
Images use OpenCV convention: y increases downward.
baseline_y = row index of the substrate line.
Drop occupies rows ABOVE baseline (smaller y values).

Angle formulas
--------------
Let (dx_t, dy_t) be the unit tangent vector at the contact point, oriented
so it points UPWARD INTO THE DROP (dy_t < 0 in image coords).

In y-up world, this is (dx_t, -dy_t), so -dy_t > 0.

LEFT contact point — measure angle from +x CCW to tangent:
    θ_L = atan2(−dy_t, +dx_t)

RIGHT contact point — measure angle from −x CW to tangent (equivalent to):
    θ_R = atan2(−dy_t, −dx_t)

Both formulas use arctan2 → both work correctly for all angles 0°–180°,
including the previously problematic near-90° case.

Verification
    θ = 45°:  L tangent (image) = (+1,−1)/√2  → atan2(+1,+1)=45° ✓
              R tangent (image) = (−1,−1)/√2  → atan2(+1,+1)=45° ✓
    θ = 90°:  both tangents = (0,−1)           → atan2(+1, 0)=90° ✓
    θ = 135°: L tangent (image) = (−1,−1)/√2  → atan2(+1,−1)=135° ✓
              R tangent (image) = (+1,−1)/√2  → atan2(+1,−1)=135° ✓
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple


# ─── PCA line fit ─────────────────────────────────────────────────────────────

def _pca_direction(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a line to (xs, ys) using SVD. Returns the direction of the
    first principal component, the centroid, and the R² analogue
    (fraction of total variance explained by the first PC).

    Unlike polyfit, this is symmetric in x and y — it minimises
    orthogonal residuals, not vertical ones — so it is equally
    accurate for slopes near 0, ±∞, or anything in between.
    """
    pts = np.column_stack([xs, ys]).astype(float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # SVD: columns of Vt are the PC directions
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]          # unit vector along first PC

    # Fraction of variance explained (our "R²")
    r2 = float(s[0] ** 2 / (s[0] ** 2 + s[1] ** 2)) if len(s) > 1 and (s[0]**2+s[1]**2) > 1e-12 else 1.0

    return direction, centroid, r2


# ─── Single-side fitting ──────────────────────────────────────────────────────

def _fit_one_side(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    window_px: float,
    side: str,
) -> Dict[str, Any]:
    """
    Fit the contour points near the baseline on one side and return
    the contact angle and contact point location.
    """
    empty = {k: None for k in ('theta', 'x_contact', 'direction', 'r2', 'n_points')}

    # Midpoint of the contour's x-extent — more robust than median for
    # asymmetric drops where the median can skew to one side.
    x_center = (xs.min() + xs.max()) / 2.0
    near      = np.abs(ys - baseline_y) <= window_px
    half      = (xs <= x_center) if side == 'left' else (xs > x_center)
    mask      = near & half

    xs_s, ys_s = xs[mask], ys[mask]
    if len(xs_s) < 5:
        return empty

    direction, centroid, r2 = _pca_direction(xs_s, ys_s)
    dx_t, dy_t = direction

    # Orient so the tangent points UPWARD INTO THE DROP (dy_t < 0 in image coords)
    if dy_t > 0:
        dx_t, dy_t = -dx_t, -dy_t

    # Contact point: where the line (centroid + t·direction) meets y = baseline_y
    # centroid[1] + t·dy_t = baseline_y  →  t = (baseline_y − centroid[1]) / dy_t
    if abs(dy_t) > 1e-6:
        t_c = (baseline_y - centroid[1]) / dy_t
        x_contact = float(centroid[0] + t_c * dx_t)
    else:
        # Tangent is horizontal — contact point is directly below centroid
        x_contact = float(centroid[0])

    # Contact angle (see module docstring for derivation)
    if side == 'left':
        theta = float(np.degrees(np.arctan2(-dy_t,  dx_t)))
    else:
        theta = float(np.degrees(np.arctan2(-dy_t, -dx_t)))

    # arctan2 returns values in (−180°, 180°]; clamp to [0°, 180°]
    theta = float(np.clip(theta, 0.0, 180.0))

    return {
        'theta':     theta,
        'x_contact': x_contact,
        'direction': (dx_t, dy_t),
        'r2':        r2,
        'n_points':  int(len(xs_s)),
    }


# ─── Public interface ─────────────────────────────────────────────────────────

def compute_contact_angles(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    window_px: float = 25.0,
) -> Dict[str, Any]:
    """
    Compute left and right contact angles from contour edge points.

    Uses PCA-based tangent estimation near each contact point. Works
    correctly for all contact angles 0°–180°, including near 90°.

    Args:
        xs, ys     : Contour point coordinates (image space, y-down).
        baseline_y : Row index of the substrate baseline.
        window_px  : Half-height (px) of the fitting region at each
                     contact point. Tune alongside Canny thresholds:
                       • too small → noisy fit (few points)
                       • too large → pulls in mid-drop curvature

    Returns dict
    ------------
    theta_left, theta_right : Contact angles in degrees (None if failed).
    theta_mean              : Mean of valid angles.
    asymmetry               : |θ_L − θ_R| (None if either side failed).
    x_left, x_right         : Contact point x-coordinates (pixels).
    direction_left/right    : Unit tangent vectors (dx, dy) in image coords.
    r2_left, r2_right       : PCA variance explained (1.0 = perfect line).
    n_left, n_right         : Number of points used per side.
    baseline_y              : As provided.
    """
    left  = _fit_one_side(xs, ys, baseline_y, window_px, 'left')
    right = _fit_one_side(xs, ys, baseline_y, window_px, 'right')

    valid = [v for v in (left['theta'], right['theta']) if v is not None]
    asymmetry = (
        abs(left['theta'] - right['theta'])
        if left['theta'] is not None and right['theta'] is not None
        else None
    )

    return {
        'theta_left':       left['theta'],
        'theta_right':      right['theta'],
        'theta_mean':       float(np.mean(valid)) if valid else None,
        'asymmetry':        asymmetry,
        'x_left':           left['x_contact'],
        'x_right':          right['x_contact'],
        'direction_left':   left['direction'],
        'direction_right':  right['direction'],
        'r2_left':          left['r2'],
        'r2_right':         right['r2'],
        'n_left':           left['n_points'],
        'n_right':          right['n_points'],
        'baseline_y':       float(baseline_y),
    }


# ── Circle-fit contact angle ───────────────────────────────────────────────────
# Used for captive bubble and sessile drop geometries where the full profile
# is approximately spherical.  More accurate than local-tangent methods when
# the contact-line region itself is noisy or nearly horizontal.

def fit_circle_algebraic(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Algebraic least-squares circle fit (Coope 1993).

    Minimises sum of squared algebraic distances: x²+y²+Dx+Ey+F = 0.
    Works with as few as 4 points; accurate for hundreds.

    Returns
    -------
    cx, cy : Circle centre (pixels).
    r      : Radius (pixels).
    rms    : RMS radial residual (pixels) — fit quality indicator.
             < 5 px is excellent; > 20 px suggests the profile is not circular.
    """
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x ** 2 + y ** 2)
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = coeffs
    cx = -D / 2
    cy = -E / 2
    r  = float(np.sqrt(max(cx ** 2 + cy ** 2 - F, 0.0)))
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms   = float(np.sqrt(np.mean((radii - r) ** 2)))
    return float(cx), float(cy), r, rms


def fit_circle_contact_angle(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    captive_bubble: bool = False,
) -> Dict[str, Any]:
    """
    Compute contact angle by fitting a circle to the full drop/bubble profile.

    Preferred over local-tangent methods when:
      • The contact-line region is nearly horizontal (captive bubble, θ > 120°)
      • There are few clean edge points near the contact line
      • The feature is approximately spherical

    Angle convention
    ----------------
    sessile drop (captive_bubble=False):
        cos(θ) = (baseline_y − cy) / r
        θ < 90° = hydrophilic;  θ > 90° = hydrophobic

    captive bubble (captive_bubble=True):
        Pass the FLIPPED image (flip_for_captive_bubble applied first).
        The code automatically converts to the correct angle through the
        liquid (water) phase: cos(θ_CB) = (cy_orig − bl_orig) / r
        which equals 180° − (sessile-equivalent angle in flipped image).

    Returns
    -------
    Same key set as compute_contact_angles() for drop-in compatibility:
    theta_left, theta_right, theta_mean, asymmetry,
    x_left, x_right, direction_left, direction_right,
    r2, cx, cy, radius, rms_px, baseline_y, captive_bubble.
    """
    import math as _math

    cx, cy, r, rms = fit_circle_algebraic(xs, ys)

    # Check that the circle intersects the baseline
    # dy = cy - baseline_y; negative when centre is above baseline (y-down)
    dy = cy - baseline_y
    if abs(dy) >= r:
        return {k: None for k in
                ('theta_left', 'theta_right', 'theta_mean', 'asymmetry',
                 'x_left', 'x_right', 'direction_left', 'direction_right',
                 'r2_left', 'r2_right', 'cx', 'cy', 'radius', 'rms_px',
                 'baseline_y', 'method')}

    # Spherical-cap formula (y-down image coords):
    #   cos(θ) = (baseline_y − cy) / r = -dy / r
    theta = float(_math.degrees(_math.acos(
        max(-1.0, min(1.0, -dy / r))
    )))

    half_chord = float(_math.sqrt(max(0.0, r ** 2 - dy ** 2)))
    xl = cx - half_chord
    xr = cx + half_chord

    # Tangent direction at each contact point, pointing INTO the liquid phase.
    # For sessile: liquid = drop interior (upward from baseline).
    # For captive bubble (flipped): liquid = water (downward from baseline
    #   in flipped image = the exterior of the bubble).
    def _tangent(x_contact):
        norm = np.array([cx - x_contact, cy - baseline_y]) / r
        tang = np.array([-norm[1], norm[0]])
        if tang[1] > 0:
            tang = -tang   # point upward into drop (sessile default)
        if captive_bubble:
            tang = -tang   # flip to point into water (CB convention)
        return (float(tang[0]), float(tang[1]))

    dir_l = _tangent(xl)
    dir_r = _tangent(xr)

    r2 = float(max(0.0, 1.0 - (rms / r) ** 2)) if r > 0 else 0.0

    return {
        'theta_left':      theta,
        'theta_right':     theta,
        'theta_mean':      theta,
        'asymmetry':       0.0,
        'x_left':          float(xl),
        'x_right':         float(xr),
        'direction_left':  dir_l,
        'direction_right': dir_r,
        'r2_left':         r2,
        'r2_right':        r2,
        'n_left':          len(xs),
        'n_right':         len(xs),
        'cx':              float(cx),
        'cy':              float(cy),
        'radius':          float(r),
        'rms_px':          float(rms),
        'baseline_y':      float(baseline_y),
        'captive_bubble':  captive_bubble,
        'method':          'circle_fit',
    }


def fit_circle_free_arc(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    exclude_near_baseline_px: float = 22.0,
    sigma: float = 2.5,
    max_iter: int = 12,
    arc_fraction: float = 0.5,
    captive_bubble: bool = False,
) -> Dict[str, Any]:
    """
    Circle fit using only the free bubble arc — the portion of the contour
    not flattened against the substrate.

    This is the recommended method for captive bubble images. It handles:
      - The flat cap where the bubble presses against the substrate
        (excluded by ``exclude_near_baseline_px``)
      - Substrate-holder edges that contaminate the upper contour
        (excluded by angular arc trimming)
      - Background clutter that survived segmentation
        (removed by iterative sigma-clipping)

    Steps
    -----
    1. Remove points within ``exclude_near_baseline_px`` of the baseline.
    2. Fit a rough circle; identify the upper arc (angles ≈ −90° ± arc_fraction·90°)
       and remove it (that region is where substrate contact flattens the bubble).
    3. Iterative sigma-clip: refit, drop points with radial residual
       > ``sigma`` × MAD, repeat until convergence.
    4. Compute contact angle from the final circle geometry.

    Returns
    -------
    Same key set as ``fit_circle_contact_angle()``.
    Adds ``quality_pct`` = rms / r × 100; values < 3% are considered good.
    """
    import math as _math

    x, y = xs.copy().astype(np.float64), ys.copy().astype(np.float64)

    # 1. Trim flat cap
    keep = np.abs(y - baseline_y) > exclude_near_baseline_px
    x, y = x[keep], y[keep]
    if len(x) < 10:
        raise ValueError(
            f"Too few points after baseline cap trim ({len(x)} remain). "
            "Try reducing exclude_near_baseline_px."
        )

    # 2. Remove upper arc (substrate contact region)
    cx0, cy0, r0, _ = fit_circle_algebraic(x, y)
    angles = np.degrees(np.arctan2(y - cy0, x - cx0))
    half_band = arc_fraction * 90.0
    upper = (angles > -(90.0 + half_band)) & (angles < -(90.0 - half_band))
    x, y  = x[~upper], y[~upper]
    if len(x) < 10:
        raise ValueError("Too few points after upper-arc removal.")

    # 3. Iterative sigma-clip
    _clipping_truncated = False
    for _ in range(max_iter):
        cx, cy, r, rms = fit_circle_algebraic(x, y)
        rads = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        res  = np.abs(rads - r)
        mad  = float(np.median(res))
        ok   = res <= sigma * max(mad, 0.5)
        if ok.sum() == len(x):
            break   # converged
        if ok.sum() < 10:
            # Clipping would leave too few points; stop but warn
            _clipping_truncated = True
            break
        x, y = x[ok], y[ok]
    if _clipping_truncated:
        import warnings
        warnings.warn(
            "fit_circle_free_arc: sigma-clipping stopped early (would leave "
            f"<10 points). Fit uses {len(x)} points and may be affected by outliers.",
            RuntimeWarning, stacklevel=2,
        )

    cx, cy, r, rms = fit_circle_algebraic(x, y)
    quality_pct = float(rms / r * 100) if r > 0 else 999.0

    # 4. Contact angle — spherical-cap formula (y-down):
    #   cos(θ) = (baseline_y − cy) / r = -dy / r
    dy = cy - baseline_y
    if abs(dy) >= r:
        return {k: None for k in (
            'theta_left', 'theta_right', 'theta_mean', 'asymmetry',
            'x_left', 'x_right', 'direction_left', 'direction_right',
            'r2_left', 'r2_right', 'n_left', 'n_right',
            'cx', 'cy', 'radius', 'rms_px', 'quality_pct',
            'baseline_y', 'captive_bubble', 'method',
        )}

    theta = float(_math.degrees(_math.acos(
        max(-1.0, min(1.0, -dy / r))
    )))

    half_chord = float(_math.sqrt(max(0.0, r ** 2 - dy ** 2)))
    xl = cx - half_chord
    xr = cx + half_chord

    def _tangent(x_contact):
        norm = np.array([cx - x_contact, cy - baseline_y]) / r
        tang = np.array([-norm[1], norm[0]])
        if tang[1] > 0:
            tang = -tang        # point upward into drop/bubble
        if captive_bubble:
            tang = -tang        # flip into water for CB
        return (float(tang[0]), float(tang[1]))

    r2 = float(max(0.0, 1.0 - (rms / r) ** 2)) if r > 0 else 0.0

    return {
        'theta_left':      theta,
        'theta_right':     theta,
        'theta_mean':      theta,
        'asymmetry':       0.0,
        'x_left':          float(xl),
        'x_right':         float(xr),
        'direction_left':  _tangent(xl),
        'direction_right': _tangent(xr),
        'r2_left':         r2,
        'r2_right':        r2,
        'n_left':          len(x),
        'n_right':         len(x),
        'cx':              float(cx),
        'cy':              float(cy),
        'radius':          float(r),
        'rms_px':          float(rms),
        'quality_pct':     quality_pct,
        'baseline_y':      float(baseline_y),
        'captive_bubble':  captive_bubble,
        'method':          'circle_fit_free_arc',
    }
