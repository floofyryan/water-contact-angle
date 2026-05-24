"""
advancing_receding.py — Advancing and receding contact angle analysis.

When a drop grows (liquid pumped in) the contact line advances — the
measured angle is the ADVANCING angle θ_adv.  When liquid is withdrawn
the contact line recedes — the measured angle is the RECEDING angle θ_rec.
The difference is the contact angle HYSTERESIS:

    H = θ_adv − θ_rec

This module analyses video of either:

  A. Pump-controlled grow/shrink experiment — supply/withdrawal video.
  B. Any video where contact-line motion is detected from the radius time
     series (displacement-based detection).

Regime detection
----------------
Each frame is classified as:
  'advancing'  — contact radius increasing
  'receding'   — contact radius decreasing
  'stationary' — contact radius approximately constant
  None         — fit failed

The advancing angle is the mean θ over advancing frames; receding angle
is the mean θ over receding frames.

Usage
-----
    from contact_angle.advancing_receding import AdvancingRecedingAnalyzer

    ar = AdvancingRecedingAnalyzer(
        mode='sessile',
        baseline_y=420,
        canny_low=25,
        canny_high=90,
    )
    ar.analyze_video('pump_cycle.mp4')
    print(ar.summary())
    ar.plot()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .analyzer import ContactAngleAnalyzer


class AdvancingRecedingAnalyzer:
    """
    Frame-by-frame advancing/receding contact angle analysis.

    Parameters
    ----------
    mode            : 'sessile' | 'captive_bubble'
    baseline_y      : Fixed baseline row (strongly recommended).
    frame_step      : Analyse every Nth frame.
    radius_tol_px   : Contact radius change (px/frame) below which the
                      contact line is considered stationary.
    smooth_window   : Frames for moving-average smoothing before regime
                      classification.
    **analyzer_kwargs : Forwarded to ContactAngleAnalyzer.
    """

    _REGIME_COLORS = {
        "advancing":   "#42a5f5",   # blue
        "receding":    "#ef5350",   # red
        "stationary":  "#66bb6a",   # green
        None:          "#555555",
    }

    def __init__(
        self,
        mode: str = "sessile",
        baseline_y: Optional[int] = None,
        frame_step: int = 1,
        radius_tol_px: float = 1.5,
        smooth_window: int = 9,
        **analyzer_kwargs,
    ):
        self.mode           = mode
        self.baseline_y     = baseline_y
        self.frame_step     = max(1, frame_step)
        self.radius_tol_px  = float(radius_tol_px)
        self.smooth_window  = max(3, smooth_window)
        self._kw            = analyzer_kwargs
        self._analyzer      = ContactAngleAnalyzer(mode=mode, **analyzer_kwargs)
        self._raw:  List[Dict[str, Any]] = []
        self._fps:  float = 30.0

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Analyse a video and classify each frame as advancing/receding/stationary.

        Returns a list of per-frame dicts with keys:
            frame, time_s, theta_left, theta_right, theta_mean,
            x_left, x_right, base_radius_px, regime
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        self._fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._raw  = []
        frame_idx  = 0
        n_proc     = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_step != 0:
                    frame_idx += 1
                    continue

                row = self._process_frame(frame, frame_idx)
                self._raw.append(row)
                n_proc += 1

                if show_progress and n_proc % 30 == 0:
                    pct = frame_idx / total * 100 if total > 0 else 0
                    th  = row.get("theta_mean")
                    msg = f"  [{frame_idx}/{total}] {pct:.0f}%"
                    if th is not None:
                        msg += f"  θ={th:.1f}°"
                    print(f"\r{msg}", end="", flush=True)

                if max_frames and n_proc >= max_frames:
                    break
                frame_idx += 1
        finally:
            cap.release()
            if show_progress:
                print(f"\n  Done: {n_proc} frames.")

        self._classify_regimes()
        return self._raw

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Dict[str, Any]:
        empty = {
            "frame": frame_idx,
            "time_s": round(frame_idx / self._fps, 4),
            "theta_left": None, "theta_right": None, "theta_mean": None,
            "x_left": None, "x_right": None,
            "base_radius_px": None,
            "regime": None,
        }
        try:
            result = self._analyzer.analyze(frame, baseline_y=self.baseline_y)
        except Exception:
            return empty

        xl = result.get("x_left")
        xr = result.get("x_right")
        if xl is None or xr is None:
            return empty

        radius_px = abs(xr - xl) / 2.0
        return {
            "frame":          frame_idx,
            "time_s":         round(frame_idx / self._fps, 4),
            "theta_left":     result.get("theta_left"),
            "theta_right":    result.get("theta_right"),
            "theta_mean":     result.get("theta_mean"),
            "x_left":         xl,
            "x_right":        xr,
            "base_radius_px": radius_px,
            "regime":         None,
        }

    # ── Regime classification ─────────────────────────────────────────────────

    def _classify_regimes(self) -> None:
        """Classify each frame as advancing, receding, or stationary."""
        radii = [r.get("base_radius_px") for r in self._raw]

        # Smooth radius series to reduce noise
        smoothed = _moving_avg(radii, self.smooth_window)

        for i, row in enumerate(self._raw):
            if smoothed[i] is None:
                row["regime"] = None
                continue

            # Estimate local slope using central difference over ±half_window
            hw = max(1, self.smooth_window // 2)
            lo = max(0, i - hw)
            hi = min(len(smoothed) - 1, i + hw)

            r_lo = smoothed[lo]
            r_hi = smoothed[hi]
            if r_lo is None or r_hi is None or hi == lo:
                row["regime"] = "stationary"
                continue

            slope = (r_hi - r_lo) / (hi - lo)   # px per frame

            if slope > self.radius_tol_px:
                row["regime"] = "advancing"
            elif slope < -self.radius_tol_px:
                row["regime"] = "receding"
            else:
                row["regime"] = "stationary"

    # ── Results ───────────────────────────────────────────────────────────────

    def advancing_angle(self) -> Optional[float]:
        """Mean contact angle over frames classified as 'advancing'."""
        vals = [r["theta_mean"] for r in self._raw
                if r.get("regime") == "advancing" and r.get("theta_mean") is not None]
        return float(np.mean(vals)) if vals else None

    def receding_angle(self) -> Optional[float]:
        """Mean contact angle over frames classified as 'receding'."""
        vals = [r["theta_mean"] for r in self._raw
                if r.get("regime") == "receding" and r.get("theta_mean") is not None]
        return float(np.mean(vals)) if vals else None

    def hysteresis(self) -> Optional[float]:
        """Contact angle hysteresis H = θ_adv − θ_rec (degrees)."""
        adv = self.advancing_angle()
        rec = self.receding_angle()
        if adv is None or rec is None:
            return None
        return adv - rec

    def summary(self) -> str:
        """Return a formatted text summary of the analysis."""
        adv = self.advancing_angle()
        rec = self.receding_angle()
        hys = self.hysteresis()

        lines = ["Advancing/receding contact angle summary:"]
        if adv is not None:
            n = sum(1 for r in self._raw if r.get("regime") == "advancing")
            lines.append(f"  θ_advancing = {adv:.1f}°  (n={n} frames)")
        else:
            lines.append("  θ_advancing = — (no advancing frames detected)")

        if rec is not None:
            n = sum(1 for r in self._raw if r.get("regime") == "receding")
            lines.append(f"  θ_receding  = {rec:.1f}°  (n={n} frames)")
        else:
            lines.append("  θ_receding  = — (no receding frames detected)")

        if hys is not None:
            lines.append(f"  Hysteresis  = {hys:.1f}°")
        else:
            lines.append("  Hysteresis  = — (need both phases)")

        return "\n".join(lines)

    def export_csv(self, path: Union[str, Path]) -> None:
        """Write per-frame results to CSV."""
        if not self._raw:
            print("No results to export.")
            return
        fields = ["frame", "time_s", "theta_left", "theta_right", "theta_mean",
                  "x_left", "x_right", "base_radius_px", "regime"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._raw)
        print(f"Exported {len(self._raw)} rows → {path}")

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot contact angle and base radius time series with regime background.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        times  = [r["time_s"]        for r in self._raw]
        theta  = [r.get("theta_mean") for r in self._raw]
        radius = [r.get("base_radius_px") for r in self._raw]
        regimes = [r.get("regime")   for r in self._raw]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.patch.set_facecolor("#1a1a1a")

        for ax in axes:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaa")
            for sp in ax.spines.values():
                sp.set_color("#444")

        # Regime background
        for ax in axes:
            prev_reg = None
            t_start  = times[0] if times else 0
            for t, reg in zip(times, regimes):
                if reg != prev_reg:
                    if prev_reg is not None:
                        col = self._REGIME_COLORS.get(prev_reg, "#555")
                        ax.axvspan(t_start, t, color=col, alpha=0.15)
                    t_start  = t
                    prev_reg = reg
            if prev_reg is not None and times:
                col = self._REGIME_COLORS.get(prev_reg, "#555")
                ax.axvspan(t_start, times[-1], color=col, alpha=0.15)

        # Panel 1 — contact angle
        ax0 = axes[0]
        pairs = [(t, v) for t, v in zip(times, theta) if v is not None]
        if pairs:
            t_, v_ = zip(*pairs)
            ax0.plot(t_, v_, color="white", lw=1.8)

        # Mark advancing/receding means
        adv = self.advancing_angle()
        rec = self.receding_angle()
        if adv is not None:
            ax0.axhline(adv, color=self._REGIME_COLORS["advancing"],
                        lw=1.5, ls="--", label=f"θ_adv = {adv:.1f}°")
        if rec is not None:
            ax0.axhline(rec, color=self._REGIME_COLORS["receding"],
                        lw=1.5, ls="--", label=f"θ_rec = {rec:.1f}°")
        ax0.set_ylabel("Contact angle (°)", color="#ccc", fontsize=10)
        # Store the angle legend before the regime legend overwrites it.
        leg_angles = ax0.legend(fontsize=9, facecolor="#333", labelcolor="white",
                                loc="upper right")

        # Panel 2 — base radius
        ax1 = axes[1]
        rpairs = [(t, v) for t, v in zip(times, radius) if v is not None]
        if rpairs:
            t_, v_ = zip(*rpairs)
            ax1.plot(t_, v_, color="#66bb6a", lw=1.5)
        ax1.set_ylabel("Base radius (px)", color="#ccc", fontsize=10)
        ax1.set_xlabel("Time (s)", color="#ccc", fontsize=11)

        for ax in axes:
            ax.grid(True, alpha=0.15, color="#555")
            ax.tick_params(colors="#aaa", labelsize=9)

        # Regime legend on ax1; restore the angle legend on ax0.
        patches = [mpatches.Patch(color=col, alpha=0.6, label=reg)
                   for reg, col in self._REGIME_COLORS.items() if reg]
        axes[1].legend(handles=patches, fontsize=8, facecolor="#333",
                       labelcolor="white", title="Regime", title_fontsize=8)
        if leg_angles is not None:
            axes[0].add_artist(leg_angles)

        fig.suptitle("Advancing / Receding Contact Angle", color="white",
                     fontsize=13, y=0.98)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a1a")
            print(f"Saved: {save_path}")
        plt.show()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _moving_avg(values: list, window: int) -> list:
    """Centred moving average, skipping None values."""
    hw  = window // 2
    out = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        lo = max(0, i - hw)
        hi = min(len(values), i + hw + 1)
        nb = [x for x in values[lo:hi] if x is not None]
        out.append(float(np.mean(nb)) if nb else None)
    return out
