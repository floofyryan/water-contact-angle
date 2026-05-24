"""
calibration.py — Pixel-to-physical-unit conversion and drop volume calculation.

ScaleCalibration stores the pixels-per-mm ratio obtained by imaging a
reference object of known size (e.g. a calibration grid, a needle of
known outer diameter, or a stage micrometer).

Spherical-cap geometry
----------------------
A sessile drop resting on a flat surface forms a spherical cap.
Given the contact angle θ and the base radius R (in mm), the drop
can be characterised as:

    h  = R · (1 − cos θ) / sin θ    (cap height)
    R_s = R / sin θ                 (sphere radius)
    V  = π·h·(3R² + h²) / 6        (volume, exact formula)

This is equivalent to:
    V  = (π·R_s³/3) · (2 + cos θ)(1 − cos θ)²

Usage
-----
    from contact_angle.calibration import ScaleCalibration

    cal = ScaleCalibration(pixels_per_mm=47.3)
    radius_mm = cal.px_to_mm(235)    # 5 mm
    vol_ul    = cal.spherical_cap_volume_ul(theta_deg=65.0, base_radius_mm=2.3)
"""

from __future__ import annotations

import math
from typing import Dict, Optional


class ScaleCalibration:
    """
    Pixel ↔ millimetre conversion for a fixed optical magnification.

    Parameters
    ----------
    pixels_per_mm : Number of image pixels per millimetre in the field of view.
                    Obtain from imaging a known-size reference object:
                        pixels_per_mm = measured_size_px / known_size_mm
    """

    def __init__(self, pixels_per_mm: float):
        if pixels_per_mm <= 0:
            raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")
        self.pixels_per_mm = float(pixels_per_mm)

    # ── Unit conversion ───────────────────────────────────────────────────────

    def px_to_mm(self, px: float) -> float:
        """Convert pixel length to millimetres."""
        return px / self.pixels_per_mm

    def mm_to_px(self, mm: float) -> float:
        """Convert millimetres to pixel length."""
        return mm * self.pixels_per_mm

    def px2_to_mm2(self, px2: float) -> float:
        """Convert pixel area to mm²."""
        return px2 / (self.pixels_per_mm ** 2)

    # ── Volume ────────────────────────────────────────────────────────────────

    def spherical_cap_volume_ul(
        self,
        theta_deg: float,
        base_radius_mm: float,
    ) -> Optional[float]:
        """
        Compute sessile-drop volume (µL) from contact angle and base radius.

        Uses the exact spherical-cap formula:
            V = π·h·(3R²+h²)/6
        where h = R·(1−cosθ)/sinθ is the cap height and R is the base radius.

        Returns None if the inputs are unphysical (θ ≤ 0° or θ ≥ 180° or R ≤ 0).
        """
        if not (0 < theta_deg < 180) or base_radius_mm <= 0:
            return None
        th  = math.radians(theta_deg)
        sin_th = math.sin(th)
        cos_th = math.cos(th)
        if abs(sin_th) < 1e-9:
            return None
        R  = base_radius_mm
        h  = R * (1.0 - cos_th) / sin_th
        V_mm3 = math.pi * h * (3 * R ** 2 + h ** 2) / 6.0
        return V_mm3 * 1e3   # mm³ → µL  (1 mm³ = 1 µL)

    def __repr__(self) -> str:
        return f"ScaleCalibration(pixels_per_mm={self.pixels_per_mm:.4f})"


# ── Convenience helpers used by EvaporationAnalyzer ──────────────────────────

def _spherical_cap_volume(vol_input: Dict) -> Optional[float]:
    """
    Compute spherical-cap drop volume (µL) from a result dict.

    Args:
        vol_input : dict with keys 'theta_mean' (degrees) and
                    'base_radius_mm' (mm). Either may be None.

    Returns:
        Volume in µL, or None if inputs are unavailable or unphysical.
    """
    theta = vol_input.get("theta_mean")
    r_mm  = vol_input.get("base_radius_mm")
    if theta is None or r_mm is None:
        return None
    if not (0 < theta < 180) or r_mm <= 0:
        return None
    try:
        # Inline calculation — avoids needing a ScaleCalibration instance
        th     = math.radians(theta)
        sin_th = math.sin(th)
        cos_th = math.cos(th)
        if abs(sin_th) < 1e-9:
            return None
        h      = r_mm * (1.0 - cos_th) / sin_th
        V_mm3  = math.pi * h * (3 * r_mm ** 2 + h ** 2) / 6.0
        return V_mm3 * 1e3
    except Exception:
        return None
