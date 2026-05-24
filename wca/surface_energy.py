"""
Surface energy estimation from contact angle measurements.

Methods implemented
-------------------
1. Young's equation (single liquid):
       γ_SV − γ_SL = γ_LV · cos θ   (spreading coefficient / adhesion work)

2. Owens-Wendt (two or more liquids, recommended):
       γ_L(1 + cosθ) = 2√(γ_S^d · γ_L^d) + 2√(γ_S^p · γ_L^p)
   Solves for dispersive (d) and polar (p) components of surface energy.
   Requires measurements with at least two liquids with known γ^d and γ^p.

3. Wu harmonic-mean (two or more liquids):
       γ_L(1 + cosθ) = 4·γ_S^d·γ_L^d/(γ_S^d + γ_L^d)
                      + 4·γ_S^p·γ_L^p/(γ_S^p + γ_L^p)
   Better for polymer–liquid systems with similar polarities.

Built-in test liquid database
------------------------------
Common liquids used in contact angle surface energy determination.
All surface tension values in mN/m at 20–25°C.

Usage
-----
    from contact_angle.surface_energy import (
        SurfaceEnergyAnalyzer, LIQUIDS
    )

    sea = SurfaceEnergyAnalyzer()
    sea.add_measurement(LIQUIDS['water'],         theta_deg=75.3)
    sea.add_measurement(LIQUIDS['diiodomethane'], theta_deg=42.1)

    result = sea.owens_wendt()
    print(result)

    result_wu = sea.wu()
    print(result_wu)

    # Single-liquid adhesion work
    from contact_angle.surface_energy import work_of_adhesion
    W = work_of_adhesion(75.3, gamma_lv=72.8)
    print(f"W_adhesion = {W:.2f} mN/m")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─── Test liquid database ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TestLiquid:
    """Surface tension components of a test liquid (mN/m at ~20°C)."""
    name:        str
    gamma_total: float   # total surface tension
    gamma_d:     float   # dispersive (Lifshitz-van der Waals) component
    gamma_p:     float   # polar component

    def __post_init__(self):
        if abs(self.gamma_total - (self.gamma_d + self.gamma_p)) > 0.5:
            raise ValueError(
                f"{self.name}: gamma_d + gamma_p must ≈ gamma_total"
            )


# Reference values from Owens & Wendt (1969) and Ström et al.
LIQUIDS: Dict[str, TestLiquid] = {
    "water": TestLiquid(
        "Water",             gamma_total=72.8, gamma_d=21.8, gamma_p=51.0
    ),
    "diiodomethane": TestLiquid(
        "Diiodomethane",     gamma_total=50.8, gamma_d=50.8, gamma_p=0.0
    ),
    "ethylene_glycol": TestLiquid(
        "Ethylene glycol",   gamma_total=48.0, gamma_d=29.0, gamma_p=19.0
    ),
    "formamide": TestLiquid(
        "Formamide",         gamma_total=58.0, gamma_d=39.5, gamma_p=18.5
    ),
    "glycerol": TestLiquid(
        "Glycerol",          gamma_total=63.4, gamma_d=37.0, gamma_p=26.4
    ),
    "hexadecane": TestLiquid(
        "Hexadecane",        gamma_total=27.5, gamma_d=27.5, gamma_p=0.0
    ),
}


# ─── Single-liquid functions ──────────────────────────────────────────────────

def work_of_adhesion(theta_deg: float, gamma_lv: float) -> float:
    """
    Thermodynamic work of adhesion (Dupré–Young equation).

    W_a = γ_LV · (1 + cos θ)

    Args:
        theta_deg : Contact angle in degrees.
        gamma_lv  : Liquid surface tension (mN/m).

    Returns:
        Work of adhesion (mN/m).
    """
    return gamma_lv * (1 + math.cos(math.radians(theta_deg)))


def spreading_coefficient(theta_deg: float, gamma_lv: float) -> float:
    """
    Spreading coefficient S = γ_SV − γ_SL − γ_LV = γ_LV(cosθ − 1).

    Negative for partial wetting; zero or positive for complete spreading.
    """
    return gamma_lv * (math.cos(math.radians(theta_deg)) - 1.0)


def young_dupre(
    theta_deg: float,
    gamma_lv: float,
) -> Dict[str, float]:
    """
    Single-liquid analysis via the Young-Dupré equation.

    Returns
    -------
    gamma_sv_minus_sl : γ_SV − γ_SL (cannot separate without second liquid)
    work_of_adhesion  : W_a = γ_LV(1 + cosθ)
    spreading_coeff   : S = γ_LV(cosθ − 1)
    cos_theta         : cos(θ)
    """
    cos_th = math.cos(math.radians(theta_deg))
    return {
        "theta_deg":          theta_deg,
        "gamma_lv":           gamma_lv,
        "cos_theta":          round(cos_th, 6),
        "gamma_sv_minus_sl":  round(gamma_lv * cos_th, 4),
        "work_of_adhesion":   round(gamma_lv * (1 + cos_th), 4),
        "spreading_coeff":    round(gamma_lv * (cos_th - 1), 4),
    }


# ─── Multi-liquid analyzer ────────────────────────────────────────────────────

@dataclass
class _Measurement:
    liquid: TestLiquid
    theta_deg: float


class SurfaceEnergyAnalyzer:
    """
    Compute solid surface energy from contact angle measurements with
    multiple test liquids.

    Workflow
    --------
        sea = SurfaceEnergyAnalyzer()
        sea.add_measurement(LIQUIDS['water'],         theta_deg=75.3)
        sea.add_measurement(LIQUIDS['diiodomethane'], theta_deg=42.1)

        r = sea.owens_wendt()
        print(f"γ_S = {r['gamma_s_total']:.2f} mN/m  "
              f"(d={r['gamma_s_d']:.2f}, p={r['gamma_s_p']:.2f})")
    """

    def __init__(self):
        self._measurements: List[_Measurement] = []

    def add_measurement(
        self,
        liquid: TestLiquid,
        theta_deg: float,
    ) -> "SurfaceEnergyAnalyzer":
        """Add a contact angle measurement. Returns self for chaining."""
        if not (0 <= theta_deg <= 180):
            raise ValueError(f"theta_deg must be in [0°, 180°], got {theta_deg}")
        self._measurements.append(_Measurement(liquid, theta_deg))
        return self

    def clear(self) -> None:
        """Remove all measurements."""
        self._measurements.clear()

    def _require(self, n: int = 2) -> None:
        if len(self._measurements) < n:
            raise RuntimeError(
                f"Need at least {n} measurement(s); have {len(self._measurements)}."
            )

    # ── Owens-Wendt ───────────────────────────────────────────────────────────

    def owens_wendt(self) -> Dict[str, float]:
        """
        Owens-Wendt geometric-mean model.

        Fits γ_S^d and γ_S^p from the linear system:
            γ_L(1+cosθ)/2 = √(γ_S^d) · √(γ_L^d) + √(γ_S^p) · √(γ_L^p)

        With 2 liquids this is exact; with more it uses least squares.

        Returns
        -------
        gamma_s_d, gamma_s_p, gamma_s_total : Surface energy components (mN/m)
        r_squared                             : Fit quality (1 = perfect)
        per_liquid                            : Per-liquid predicted vs measured
        """
        self._require(2)

        # Build linear system:  A · x = b   where x = [√γ_S^d, √γ_S^p]
        A = np.array([
            [math.sqrt(m.liquid.gamma_d), math.sqrt(m.liquid.gamma_p)]
            for m in self._measurements
        ])
        b = np.array([
            m.liquid.gamma_total * (1 + math.cos(math.radians(m.theta_deg))) / 2
            for m in self._measurements
        ])

        x, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        sqrt_gd, sqrt_gp = x

        if sqrt_gd < 0 or sqrt_gp < 0:
            # Negative solution: clamp and warn
            sqrt_gd = max(sqrt_gd, 0.0)
            sqrt_gp = max(sqrt_gp, 0.0)

        gd = float(sqrt_gd ** 2)
        gp = float(sqrt_gp ** 2)
        gt = gd + gp

        # R² from predicted vs measured left-hand side.
        # Use the clamped solution so reported R² matches reported gd/gp.
        x_clamped = np.array([sqrt_gd, sqrt_gp])
        b_pred = A @ x_clamped
        ss_res = float(np.sum((b - b_pred) ** 2))
        ss_tot = float(np.sum((b - b.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        per_liquid = []
        for m, b_m, b_p in zip(self._measurements, b, b_pred):  # noqa: B007 (b_m unused intentionally)
            cos_pred = 2 * b_p / m.liquid.gamma_total - 1
            theta_pred = math.degrees(math.acos(max(-1, min(1, cos_pred))))
            per_liquid.append({
                "liquid":       m.liquid.name,
                "theta_meas":   round(m.theta_deg, 2),
                "theta_pred":   round(theta_pred, 2),
                "residual_deg": round(abs(m.theta_deg - theta_pred), 2),
            })

        return {
            "method":         "Owens-Wendt",
            "gamma_s_d":      round(gd, 4),
            "gamma_s_p":      round(gp, 4),
            "gamma_s_total":  round(gt, 4),
            "r_squared":      round(r2, 6),
            "n_liquids":      len(self._measurements),
            "per_liquid":     per_liquid,
        }

    # ── Wu harmonic mean ──────────────────────────────────────────────────────

    def wu(
        self,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> Dict[str, float]:
        """
        Wu harmonic-mean model (iterative solution).

            γ_L(1+cosθ) = 4γ_S^d γ_L^d/(γ_S^d+γ_L^d) + 4γ_S^p γ_L^p/(γ_S^p+γ_L^p)

        Better than Owens-Wendt for polymer surfaces where
        γ_S^d and γ_L^d are similar in magnitude.

        Uses iterative refinement starting from the Owens-Wendt solution.
        """
        self._require(2)

        def _lhs(m):
            return m.liquid.gamma_total * (1 + math.cos(math.radians(m.theta_deg)))

        def _rhs(gd, gp, m):
            ld, lp = m.liquid.gamma_d, m.liquid.gamma_p
            d_term = 4 * gd * ld / (gd + ld) if (gd + ld) > 1e-9 else 0.0
            p_term = 4 * gp * lp / (gp + lp) if (gp + lp) > 1e-9 else 0.0
            return d_term + p_term

        # Initial guess from Owens-Wendt
        try:
            ow = self.owens_wendt()
            # Use `is not None` so a physically meaningful zero (completely
            # non-dispersive or non-polar surface) is not replaced by a default.
            gd = ow["gamma_s_d"] if ow["gamma_s_d"] is not None else 20.0
            gp = ow["gamma_s_p"] if ow["gamma_s_p"] is not None else 5.0
        except Exception:
            gd, gp = 20.0, 5.0

        gd = max(gd, 0.01)
        gp = max(gp, 0.01)

        # Iterative fixed-point (Schulz & Nardin approach)
        for _ in range(max_iter):
            # Build over-determined linear system in Wu form
            # Linearise: define y_d = 4 γ_L^d / (γ_S^d + γ_L^d) and similar for p
            # Then: LHS = y_d · γ_S^d + y_p · γ_S^p
            A = np.array([
                [
                    4 * m.liquid.gamma_d / (gd + m.liquid.gamma_d),
                    4 * m.liquid.gamma_p / (gp + m.liquid.gamma_p),
                ]
                for m in self._measurements
            ])
            b = np.array([_lhs(m) for m in self._measurements])
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            gd_new = max(float(x[0]), 0.01)
            gp_new = max(float(x[1]), 0.01)

            if abs(gd_new - gd) < tol and abs(gp_new - gp) < tol:
                break
            gd, gp = gd_new, gp_new

        gt = gd + gp

        # R²
        b_pred = np.array([_rhs(gd, gp, m) for m in self._measurements])
        b_arr  = np.array([_lhs(m) for m in self._measurements])
        ss_res = float(np.sum((b_arr - b_pred) ** 2))
        ss_tot = float(np.sum((b_arr - b_arr.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        per_liquid = []
        for m, lhs_pred in zip(self._measurements, b_pred):
            cos_pred = lhs_pred / m.liquid.gamma_total - 1
            theta_pred = math.degrees(math.acos(max(-1, min(1, cos_pred))))
            per_liquid.append({
                "liquid":       m.liquid.name,
                "theta_meas":   round(m.theta_deg, 2),
                "theta_pred":   round(theta_pred, 2),
                "residual_deg": round(abs(m.theta_deg - theta_pred), 2),
            })

        return {
            "method":        "Wu",
            "gamma_s_d":     round(gd, 4),
            "gamma_s_p":     round(gp, 4),
            "gamma_s_total": round(gt, 4),
            "r_squared":     round(r2, 6),
            "n_liquids":     len(self._measurements),
            "per_liquid":    per_liquid,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a formatted comparison of Owens-Wendt and Wu results."""
        if len(self._measurements) < 2:
            print("Need ≥ 2 measurements for surface energy decomposition.")
            return

        for fn in (self.owens_wendt, self.wu):
            r = fn()
            print(f"\n{'─'*54}")
            print(f"  Method : {r['method']}")
            print(f"  γ_S    = {r['gamma_s_total']:.2f} mN/m  "
                  f"(d={r['gamma_s_d']:.2f}, p={r['gamma_s_p']:.2f})")
            print(f"  R²     = {r['r_squared']:.4f}  "
                  f"(n={r['n_liquids']} liquids)")
            print(f"  {'Liquid':<20}  {'θ_meas':>8}  {'θ_pred':>8}  {'Δθ':>6}")
            for pl in r["per_liquid"]:
                print(f"  {pl['liquid']:<20}  {pl['theta_meas']:>7.2f}°  "
                      f"{pl['theta_pred']:>7.2f}°  {pl['residual_deg']:>5.2f}°")
        print(f"{'─'*54}\n")
