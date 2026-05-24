"""
visualization.py — Real-time overlay drawing for contact angle results.

Produces a BGR annotation image suitable for OpenCV display windows,
video output, or quick inspection. For publication-quality output use
publication.py instead.

The overlay shows:
  - Horizontal baseline (dashed white line)
  - Detected edge points (cyan)
  - Contact points (left = red, right = blue)
  - Tangent lines from each contact point
  - Angle arc between baseline and tangent
  - Angle readout text
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import cv2
import numpy as np


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
_COL_BASELINE  = (200, 200, 200)   # grey
_COL_EDGE      = (255, 200,   0)   # cyan-ish
_COL_LEFT      = (  0,  80, 220)   # red  (BGR)
_COL_RIGHT     = (220,  80,   0)   # blue (BGR)
_COL_TANGENT   = (  0, 220, 220)   # yellow
_COL_TEXT      = (255, 255, 255)   # white
_COL_TEXT_GOOD = (  0, 230,  90)   # green
_COL_TEXT_BAD  = (  0,  60, 220)   # red


def draw_overlay(
    image: np.ndarray,
    edges: Optional[np.ndarray],
    result: Optional[Dict[str, Any]],
    baseline_y: int,
    tangent_length_px: int = 55,
    arc_radius_px: int = 35,
    show_r2: bool = True,
) -> np.ndarray:
    """
    Draw a full analysis overlay on top of the image.

    Args:
        image             : BGR or grayscale source image.
        edges             : Binary edge image (may be None).
        result            : Contact angle result dict (may be None or empty).
        baseline_y        : Baseline row in image coordinates.
        tangent_length_px : Half-length of each tangent line (pixels).
        arc_radius_px     : Radius of the angle arc annotation (pixels).
        show_r2           : Whether to display R² values in the text overlay.

    Returns:
        BGR image with annotations.
    """
    # Work on a colour copy
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    h, w = overlay.shape[:2]

    # ── Edge overlay ──────────────────────────────────────────────────────────
    if edges is not None:
        edge_mask = edges > 0
        overlay[edge_mask] = (
            overlay[edge_mask].astype(np.float32) * 0.4
            + np.array(_COL_EDGE, dtype=np.float32) * 0.6
        ).astype(np.uint8)

    # ── Baseline ──────────────────────────────────────────────────────────────
    _draw_dashed_hline(overlay, baseline_y, _COL_BASELINE)

    if result is None:
        return overlay

    # ── Per-side annotations ──────────────────────────────────────────────────
    for side, col in (("left", _COL_LEFT), ("right", _COL_RIGHT)):
        x_c   = result.get(f"x_{side}")
        theta = result.get(f"theta_{side}")
        direc = result.get(f"direction_{side}")
        r2    = result.get(f"r2_{side}")

        if x_c is None or theta is None:
            continue

        xi = int(round(float(x_c)))
        yi = int(round(float(baseline_y)))

        # Contact point dot
        cv2.circle(overlay, (xi, yi), 5, col, -1, cv2.LINE_AA)

        # Tangent line
        if direc is not None:
            dx_t, dy_t = direc
            x0 = int(xi - dx_t * tangent_length_px)
            y0 = int(yi - dy_t * tangent_length_px)
            x1 = int(xi + dx_t * tangent_length_px)
            y1 = int(yi + dy_t * tangent_length_px)
            cv2.line(overlay, (x0, y0), (x1, y1), _COL_TANGENT, 1, cv2.LINE_AA)

        # Angle arc
        _draw_angle_arc(overlay, xi, yi, theta, side, arc_radius_px, col)

        # R² quality badge
        if show_r2 and r2 is not None:
            badge_col = _COL_TEXT_GOOD if r2 > 0.90 else _COL_TEXT_BAD
            bx = xi + 6 if side == "right" else xi - 62
            cv2.putText(overlay, f"R²={r2:.2f}", (bx, yi - arc_radius_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_col, 1, cv2.LINE_AA)

    # ── Angle readout ─────────────────────────────────────────────────────────
    _draw_angle_text(overlay, result, h)

    return overlay


def _draw_dashed_hline(
    img: np.ndarray,
    y: int,
    color: tuple,
    dash_len: int = 12,
    gap_len: int = 8,
) -> None:
    """Draw a dashed horizontal line across the full image width."""
    w = img.shape[1]
    x = 0
    while x < w:
        x_end = min(x + dash_len, w)
        cv2.line(img, (x, y), (x_end, y), color, 1, cv2.LINE_AA)
        x += dash_len + gap_len


def _draw_angle_arc(
    img: np.ndarray,
    xi: int,
    yi: int,
    theta: float,
    side: str,
    r: int,
    color: tuple,
) -> None:
    """
    Draw a circular arc showing the contact angle.

    The arc sweeps from the baseline direction to the tangent direction
    in the y-down image coordinate system:
        Left  side: baseline points +x  (right), sweep CCW by θ
        Right side: baseline points -x  (left),  sweep CW  by θ

    Parametrisation (equivalent for both sides):
        Left:  x = xi + r·cos(t),  y = yi − r·sin(t)  for t in [0, θ]
        Right: x = xi − r·cos(t),  y = yi − r·sin(t)  for t in [0, θ]
    """
    th_rad = math.radians(theta)
    t_vals = np.linspace(0.0, th_rad, 40)

    if side == "left":
        xs = (xi + r * np.cos(t_vals)).astype(int)
    else:
        xs = (xi - r * np.cos(t_vals)).astype(int)

    ys = (yi - r * np.sin(t_vals)).astype(int)

    h, w = img.shape[:2]
    pts = np.column_stack([xs, ys])
    # Keep only points inside the image and strictly above the baseline (yi)
    # to prevent the arc from sweeping below the substrate for obtuse angles.
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] <= yi)
    pts = pts[valid]

    for i in range(len(pts) - 1):
        cv2.line(img,
                 (int(pts[i, 0]),   int(pts[i, 1])),
                 (int(pts[i+1, 0]), int(pts[i+1, 1])),
                 color, 2, cv2.LINE_AA)


def _draw_angle_text(
    img: np.ndarray,
    result: Dict[str, Any],
    img_height: int,
) -> None:
    """Draw the angle readout in the top-left corner."""
    lines = []

    th_l = result.get("theta_left")
    th_r = result.get("theta_right")
    th_m = result.get("theta_mean")
    asym = result.get("asymmetry")

    if th_l is not None:
        lines.append((f"L: {th_l:.1f}", _COL_LEFT))
    if th_r is not None:
        lines.append((f"R: {th_r:.1f}", _COL_RIGHT))
    if th_m is not None:
        lines.append((f"mean: {th_m:.1f}", _COL_TEXT))
    if asym is not None:
        lines.append((f"|L-R|: {asym:.1f}", _COL_TEXT))

    y = 22
    for text, col in lines:
        # Drop shadow
        cv2.putText(img, text + "°", (9, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text + "°", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)
        y += 22
