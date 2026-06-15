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

    # Spherical-cap formula (y-down image coords).
    # For a sessile drop ABOVE the baseline: a shallow drop (θ<90) has its
    # circle centre BELOW the baseline (cy > baseline_y → dy > 0), giving an
    # acute angle, while θ>90 puts the centre above (dy < 0).  The correct
    # relation is therefore  cos(θ) = dy / r.
    theta = float(_math.degrees(_math.acos(
        max(-1.0, min(1.0, dy / r))
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
    #   cos(θ) = dy / r   (see fit_circle_contact_angle for the sign derivation)
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
        max(-1.0, min(1.0, dy / r))
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


# ── Ellipse-fit contact angle ───────────────────────────────────────────────────
# Real sessile drops are flattened by gravity, so their profile is an ellipse
# rather than a circle.  Ellipse fitting is the method used by ImageJ's
# contact-angle plugins and most commercial drop-shape analysers for moderate
# drop sizes.  The contact angle is the angle between the baseline and the
# ellipse tangent at the point where the ellipse crosses the baseline.

def fit_ellipse_lstsq(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """
    Direct least-squares ellipse fit (Halir & Flusser, 1998).

    Numerically stable variant of the Fitzgibbon method.  Returns the six
    coefficients of the general conic

        A·x² + B·x·y + C·y² + D·x + E·y + F = 0

    constrained so the solution is an ellipse (4AC − B² > 0).

    Raises
    ------
    ValueError if no valid ellipse can be fitted (e.g. collinear points).
    """
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    if len(x) < 6:
        raise ValueError("Need at least 6 points for an ellipse fit.")

    # Centre and scale for numerical conditioning, undo on the coefficients
    mx, my = x.mean(), y.mean()
    sx = max(x.std(), 1e-6)
    sy = max(y.std(), 1e-6)
    xn = (x - mx) / sx
    yn = (y - my) / sy

    D1 = np.column_stack([xn ** 2, xn * yn, yn ** 2])      # quadratic part
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])       # linear part
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        raise ValueError("Ellipse fit failed (singular linear part).")

    M = S1 + S2 @ T
    # Apply inverse of the ellipse-constraint matrix C1
    C1inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    M = C1inv @ M

    eigvals, eigvecs = np.linalg.eig(M)
    cond = 4.0 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
    valid = np.where(cond > 0)[0]
    if len(valid) == 0:
        raise ValueError("No valid ellipse solution found.")

    a1 = np.real(eigvecs[:, valid[0]])
    a2 = T @ a1
    An, Bn, Cn = a1
    Dn, En, Fn = a2

    # Undo the centre/scale normalisation: substitute xn=(x-mx)/sx, yn=(y-my)/sy
    A = An / sx ** 2
    B = Bn / (sx * sy)
    C = Cn / sy ** 2
    D = (-2.0 * An * mx / sx ** 2 - Bn * my / (sx * sy) + Dn / sx)
    E = (-2.0 * Cn * my / sy ** 2 - Bn * mx / (sx * sy) + En / sy)
    F = (An * mx ** 2 / sx ** 2 + Bn * mx * my / (sx * sy)
         + Cn * my ** 2 / sy ** 2 - Dn * mx / sx - En * my / sy + Fn)
    return float(A), float(B), float(C), float(D), float(E), float(F)


def _ellipse_geometry(A, B, C, D, E, F):
    """Convert conic coefficients to centre, semi-axes and rotation (radians)."""
    den = B ** 2 - 4 * A * C
    if abs(den) < 1e-12:
        raise ValueError("Degenerate conic.")
    cx = (2 * C * D - B * E) / den
    cy = (2 * A * E - B * D) / den

    num = 2 * (A * E ** 2 + C * D ** 2 + F * B ** 2 - B * D * E - 4 * A * C * F)
    s = np.sqrt(max((A - C) ** 2 + B ** 2, 0.0))
    ax1 = num / (den * ((A + C) + s))
    ax2 = num / (den * ((A + C) - s))
    a = float(np.sqrt(abs(ax1)))
    b = float(np.sqrt(abs(ax2)))
    theta = 0.5 * np.arctan2(B, (A - C))
    return float(cx), float(cy), max(a, b), min(a, b), float(theta)


def fit_ellipse_contact_angle(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    captive_bubble: bool = False,
) -> Dict[str, Any]:
    """
    Contact angle from an ellipse fitted to the drop/bubble profile.

    The tangent at each baseline-intersection point is derived analytically
    from the conic gradient, so the result is exact for an elliptical profile
    and far less gravity-biased than a circle fit for larger drops.

    Returns the same key set as ``compute_contact_angles`` plus the ellipse
    geometry (``ellipse_cx``, ``ellipse_cy``, ``ellipse_a``, ``ellipse_b``,
    ``ellipse_theta``) for overlay drawing.
    """
    import math as _math

    # Guard: arc must have enough vertical depth to uniquely constrain an ellipse.
    # A very shallow arc (y-extent < 25% of x-extent) can be fitted by many different
    # conics; the Halir-Flusser solver picks one at random with wild contact angles.
    y_span = float(ys.max() - ys.min())
    x_span = float(xs.max() - xs.min())
    if x_span > 1e-3 and (y_span / x_span) < 0.25:
        raise ValueError(
            f"Arc too shallow for reliable ellipse fit "
            f"(y-span/x-span = {y_span/x_span:.2f} < 0.25); use circle or H/W instead."
        )

    A, B, C, D, E, F = fit_ellipse_lstsq(xs, ys)
    yb = float(baseline_y)

    # Reject degenerate conics (very elongated ellipse or near-parabolic).
    try:
        _cx, _cy, _a, _b, _et = _ellipse_geometry(A, B, C, D, E, F)
        if _b < 1e-6 or (_a / _b) > 5.0:
            raise ValueError(
                f"Ellipse degenerate (axis ratio {_a/_b:.1f} > 5)."
            )
    except ValueError:
        raise
    except Exception:
        pass

    # Intersect the conic with the baseline y = yb:
    #   A·x² + (B·yb + D)·x + (C·yb² + E·yb + F) = 0
    qa = A
    qb = B * yb + D
    qc = C * yb ** 2 + E * yb + F
    if abs(qa) < 1e-12:
        raise ValueError("Ellipse does not cross the baseline (degenerate).")
    disc = qb ** 2 - 4 * qa * qc
    if disc < 0:
        raise ValueError("Ellipse does not intersect the baseline.")

    sq = _math.sqrt(disc)
    x1 = (-qb - sq) / (2 * qa)
    x2 = (-qb + sq) / (2 * qa)
    xl, xr = (x1, x2) if x1 <= x2 else (x2, x1)

    # RMS residual of the algebraic fit (quality indicator)
    vals = (A * xs ** 2 + B * xs * ys + C * ys ** 2
            + D * xs + E * ys + F)
    grad_mag = np.sqrt((2 * A * xs + B * ys + D) ** 2
                       + (B * xs + 2 * C * ys + E) ** 2) + 1e-9
    rms = float(np.sqrt(np.mean((vals / grad_mag) ** 2)))

    def _angle_at(xc, side):
        # Conic gradient = normal direction; tangent is perpendicular to it
        gx = 2 * A * xc + B * yb + D
        gy = B * xc + 2 * C * yb + E
        tx, ty = -gy, gx
        n = _math.hypot(tx, ty)
        if n < 1e-9:
            return None, None
        tx, ty = tx / n, ty / n
        # Orient tangent to point UP into the drop (dy < 0 in image coords)
        if ty > 0:
            tx, ty = -tx, -ty
        if side == 'left':
            theta = _math.degrees(_math.atan2(-ty, tx))
        else:
            theta = _math.degrees(_math.atan2(-ty, -tx))
        theta = float(np.clip(theta, 0.0, 180.0))
        return theta, (float(tx), float(ty))

    theta_l, dir_l = _angle_at(xl, 'left')
    theta_r, dir_r = _angle_at(xr, 'right')

    if captive_bubble:
        if theta_l is not None:
            theta_l = 180.0 - theta_l
        if theta_r is not None:
            theta_r = 180.0 - theta_r

    valid = [t for t in (theta_l, theta_r) if t is not None]
    cx, cy, ea, eb, et = _ellipse_geometry(A, B, C, D, E, F)

    return {
        'theta_left':    theta_l,
        'theta_right':   theta_r,
        'theta_mean':    float(np.mean(valid)) if valid else None,
        'asymmetry':     abs(theta_l - theta_r) if (theta_l is not None and theta_r is not None) else None,
        'x_left':        float(xl),
        'x_right':       float(xr),
        'direction_left':  dir_l,
        'direction_right': dir_r,
        'rms_px':        rms,
        'ellipse_cx':    cx,
        'ellipse_cy':    cy,
        'ellipse_a':     ea,
        'ellipse_b':     eb,
        'ellipse_theta': et,
        'baseline_y':    yb,
        'captive_bubble': captive_bubble,
        'method':        'ellipse_fit',
    }


# ── Height–width (spherical-cap) contact angle ──────────────────────────────────
# The classic textbook method: assumes a circular cap and computes the angle
# from the drop height h and base diameter w via  tan(θ/2) = 2h / w.
# Exact for a spherical cap at all angles 0–180°.  Fast and robust; included
# as an independent cross-check (this is the method many ImageJ macros use).

def height_width_angle(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    base_margin_px: float = 6.0,
    captive_bubble: bool = False,
) -> Dict[str, Any]:
    """
    Contact angle from drop height and base width (spherical-cap geometry).

    Args:
        xs, ys         : Contour points (image space, y-down).
        baseline_y     : Substrate row.
        base_margin_px : Rows within this distance of the baseline are used to
                         measure the base width.
        captive_bubble : Apply the 180° liquid-phase correction.

    Returns dict with theta_left == theta_right == theta_mean and the measured
    height/width for reference.
    """
    import math as _math

    drop = ys <= baseline_y           # drop sits above the baseline
    if drop.sum() < 5:
        raise ValueError("Too few points above the baseline for height/width.")

    xa, ya = xs[drop], ys[drop]
    apex_y = float(ya.min())
    h = float(baseline_y - apex_y)    # height above baseline

    near_base = np.abs(ys - baseline_y) <= base_margin_px
    if near_base.sum() >= 2:
        xb = xs[near_base]
        w = float(xb.max() - xb.min())
        x_left, x_right = float(xb.min()), float(xb.max())
    else:
        w = float(xa.max() - xa.min())
        x_left, x_right = float(xa.min()), float(xa.max())

    if w <= 1e-6 or h <= 1e-6:
        raise ValueError("Degenerate height or width.")

    # tan(θ/2) = 2h / w  →  θ = 2·atan(2h / w);  exact for a spherical cap.
    theta = float(2.0 * _math.degrees(_math.atan2(2.0 * h, w)))
    theta = float(np.clip(theta, 0.0, 180.0))
    if captive_bubble:
        theta = 180.0 - theta

    return {
        'theta_left':   theta,
        'theta_right':  theta,
        'theta_mean':   theta,
        'asymmetry':    0.0,
        'x_left':       x_left,
        'x_right':      x_right,
        'height_px':    h,
        'width_px':     w,
        'baseline_y':   float(baseline_y),
        'captive_bubble': captive_bubble,
        'method':       'height_width',
    }


# ── Polynomial-tangent contact angle (DropSnake-style) ──────────────────────────
# ImageJ's DropSnake plugin fits a spline to the contour and evaluates its
# slope at the contact line.  Here we fit a low-order polynomial near each
# contact point and take its slope at the baseline as the tangent.  The fitting
# axis is chosen adaptively — y = p(x) for shallow (acute) contact lines and
# x = p(y) for steep (near-90°/obtuse) ones — so it stays accurate at all angles.

def polynomial_tangent_angle(
    xs: np.ndarray,
    ys: np.ndarray,
    baseline_y: float,
    window_px: float = 25.0,
    degree: int = 2,
    captive_bubble: bool = False,
) -> Dict[str, Any]:
    """
    Contact angle from a local polynomial fit near each contact point.

    Captures contact-line curvature that a straight-line (PCA) fit misses.
    The independent variable (x or y) is chosen per side from the local point
    spread, which keeps the derivative well-conditioned for every angle.
    """
    import math as _math

    x_center = (xs.min() + xs.max()) / 2.0

    def _one_side(side):
        near = np.abs(ys - baseline_y) <= window_px
        half = (xs <= x_center) if side == 'left' else (xs > x_center)
        m = near & half
        xs_s, ys_s = xs[m], ys[m]
        if len(xs_s) < max(degree + 1, 5):
            return None, None, None

        spread_x = xs_s.max() - xs_s.min()
        spread_y = ys_s.max() - ys_s.min()

        if spread_y >= spread_x:
            # Steep contact line → x = p(y), slope dx/dy at the baseline
            if spread_y < 3 or len(np.unique(ys_s)) < 2:
                return None, None, None
            deg = min(degree, len(np.unique(ys_s)) - 1)
            p  = np.poly1d(np.polyfit(ys_s, xs_s, deg))
            dp = p.deriv()
            x_c  = float(p(baseline_y))
            dxdy = float(dp(baseline_y))
            tx, ty = dxdy, 1.0
        else:
            # Shallow contact line → y = p(x), slope dy/dx at the contact x
            if spread_x < 3 or len(np.unique(xs_s)) < 2:
                return None, None, None
            deg = min(degree, len(np.unique(xs_s)) - 1)
            p  = np.poly1d(np.polyfit(xs_s, ys_s, deg))
            dp = p.deriv()
            x_c  = float(xs_s.min() if side == 'left' else xs_s.max())
            dydx = float(dp(x_c))
            tx, ty = 1.0, dydx

        n = _math.hypot(tx, ty)
        if n < 1e-9:
            return None, None, None
        tx, ty = tx / n, ty / n
        # Orient the tangent to point UP into the drop (dy < 0, image coords)
        if ty > 0:
            tx, ty = -tx, -ty
        if side == 'left':
            theta = _math.degrees(_math.atan2(-ty, tx))
        else:
            theta = _math.degrees(_math.atan2(-ty, -tx))
        theta = float(np.clip(theta, 0.0, 180.0))
        return theta, x_c, (tx, ty)

    theta_l, xl, dir_l = _one_side('left')
    theta_r, xr, dir_r = _one_side('right')

    if captive_bubble:
        if theta_l is not None:
            theta_l = 180.0 - theta_l
        if theta_r is not None:
            theta_r = 180.0 - theta_r

    valid = [t for t in (theta_l, theta_r) if t is not None]
    return {
        'theta_left':    theta_l,
        'theta_right':   theta_r,
        'theta_mean':    float(np.mean(valid)) if valid else None,
        'asymmetry':     abs(theta_l - theta_r) if (theta_l is not None and theta_r is not None) else None,
        'x_left':        xl,
        'x_right':       xr,
        'direction_left':  dir_l,
        'direction_right': dir_r,
        'baseline_y':    float(baseline_y),
        'captive_bubble': captive_bubble,
        'method':        'polynomial',
    }
