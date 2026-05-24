"""
evaporation.py — Time-resolved contact angle analysis for evaporating drops.

Evaporation proceeds through three regimes that can be identified from
the θ(t) and R(t) time series:

  CCR  (Constant Contact Radius)  — contact line is pinned to the surface.
        R is flat, θ decreases. Common on rough or chemically heterogeneous
        surfaces at the start of evaporation.

  CCA  (Constant Contact Angle)   — contact line recedes freely.
        θ is flat, R decreases. Common on smooth homogeneous surfaces or
        later in evaporation.

  Stick-slip                       — alternating pinned/receding phases.
        Both θ and R show correlated step-wise changes. Common on surfaces
        with moderate pinning (e.g. partially functionalised PDMS).

Usage
-----
    from contact_angle.evaporation import EvaporationAnalyzer
    from contact_angle.calibration import ScaleCalibration

    cal = ScaleCalibration(pixels_per_mm=47.3)

    ea = EvaporationAnalyzer(
        mode='sessile',
        calibration=cal,
        baseline_y=420,     # strongly recommended: fix baseline
        canny_low=25,
        canny_high=90,
    )
    ea.analyze_video('evap.mp4', frame_step=5)
    ea.plot_time_series(save_path='evap_plot.png')
    ea.export_csv('evap_results.csv')
    print(ea.regime_summary())
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .analyzer  import ContactAngleAnalyzer
from .calibration import ScaleCalibration


# ── Regime detection helpers ────────────────────────────────────────────────

def _moving_std(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Rolling standard deviation, skipping None."""
    hw = window // 2
    out = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        lo = max(0, i - hw); hi = min(len(values), i + hw + 1)
        neighbourhood = [x for x in values[lo:hi] if x is not None]
        out.append(float(np.std(neighbourhood)) if len(neighbourhood) > 1 else 0.0)
    return out


def _moving_avg(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Centred moving average, skipping None."""
    hw = window // 2
    out = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        lo = max(0, i - hw); hi = min(len(values), i + hw + 1)
        nb = [x for x in values[lo:hi] if x is not None]
        out.append(float(np.mean(nb)) if nb else None)
    return out


def detect_regime(
    theta_series: List[Optional[float]],
    radius_series: List[Optional[float]],
    window: int = 15,
    theta_tol_deg: float = 2.0,
    radius_tol_frac: float = 0.02,
    trend_window: int = 25,
    theta_trend_threshold: float = 0.05,
    radius_trend_frac: float = 0.003,
) -> List[Optional[str]]:
    """
    Classify each frame as 'CCR', 'CCA', 'stick_slip', 'stable', or None.

    Uses both local variance (std) AND local linear trend (slope) so that
    slow monotonic changes (e.g. CCR where θ decreases at < 0.3°/frame)
    are correctly detected even when the within-window std is low.

    Parameters
    ----------
    theta_series           : Contact angle time series (degrees).
    radius_series          : Contact base radius time series (pixels or mm).
    window                 : Rolling window for std calculation.
    theta_tol_deg          : Std threshold — θ considered "stable" if std < this.
    radius_tol_frac        : Fractional std threshold for radius stability.
    trend_window           : Frames over which local slope is estimated.
    theta_trend_threshold  : |slope| threshold (°/frame) for θ "changing".
    radius_trend_frac      : Fractional |slope|/mean threshold for R "changing".

    CCR  : radius stable (low std AND low trend), theta changing.
    CCA  : theta stable, radius changing.
    stick_slip : both θ and R changing (correlated step-wise jumps).
    stable     : both stable — drop not yet actively evaporating.
    """
    theta_std  = _moving_std(theta_series,  window)
    radius_std = _moving_std(radius_series, window)
    theta_avg  = _moving_avg(theta_series,  window)
    radius_avg = _moving_avg(radius_series, window)

    # Local linear slope over trend_window
    hw = trend_window // 2
    n  = len(theta_series)
    theta_trend  = [None] * n
    radius_trend = [None] * n
    for i in range(n):
        lo = max(0, i - hw); hi = min(n, i + hw + 1)
        t_vals = [(j, theta_series[j])  for j in range(lo, hi) if theta_series[j]  is not None]
        r_vals = [(j, radius_series[j]) for j in range(lo, hi) if radius_series[j] is not None]
        if len(t_vals) >= 3:
            xs_t, ys_t = zip(*t_vals)
            theta_trend[i]  = float(np.polyfit(xs_t, ys_t, 1)[0])
        if len(r_vals) >= 3:
            xs_r, ys_r = zip(*r_vals)
            radius_trend[i] = float(np.polyfit(xs_r, ys_r, 1)[0])

    regimes = []
    for i in range(n):
        ts = theta_std[i];  rs = radius_std[i]
        ta = theta_avg[i];  ra = radius_avg[i]
        tt = theta_trend[i]; rt = radius_trend[i]

        if ts is None or rs is None or ra is None or ta is None or ra == 0:
            regimes.append(None)
            continue

        # "Changing" = high local std OR consistent slope
        theta_changing  = (ts > theta_tol_deg or
                           (tt is not None and abs(tt) > theta_trend_threshold))
        radius_changing = ((rs / ra > radius_tol_frac) or
                           (rt is not None and abs(rt) / ra > radius_trend_frac))

        if   radius_changing and not theta_changing:  regimes.append('CCA')
        elif theta_changing  and not radius_changing: regimes.append('CCR')
        elif theta_changing  and radius_changing:     regimes.append('stick_slip')
        else:                                         regimes.append('stable')

    return regimes


# ── Main class ──────────────────────────────────────────────────────────────

class EvaporationAnalyzer:
    """
    Frame-by-frame contact angle analysis for evaporating drops.

    Tracks contact angle, base radius, drop height, and volume over time.
    Detects evaporation regime (CCR / CCA / stick-slip) automatically.
    Computes evaporation rate dV/dt.

    Parameters
    ----------
    mode         : 'sessile' | 'captive_bubble'
    calibration  : ScaleCalibration for physical units. If None, all
                   measurements remain in pixels / pixel³.
    baseline_y   : Fix baseline at this row for all frames. Strongly
                   recommended — prevents baseline jitter from creating
                   spurious angle oscillations.
    frame_step   : Analyse every Nth frame.
    smooth_window: Frames for moving-average smoothing of raw time series.
    adaptive_window: If True, scale the PCA fitting window_px proportionally
                   to the measured base radius each frame. Prevents the fixed
                   window from pulling in mid-drop curvature as the drop shrinks.
    min_radius_px: Stop tracking when base radius drops below this many pixels.
                   Prevents garbage fits on nearly-vanished drops.
    **analyzer_kwargs: Forwarded to ContactAngleAnalyzer (canny_low, canny_high, etc.)
    """

    _REGIME_COLORS = {
        'CCR':        '#42a5f5',   # blue
        'CCA':        '#ef5350',   # red
        'stick_slip': '#ab47bc',   # purple
        'stable':     '#66bb6a',   # green
        None:         '#555555',
    }

    def __init__(
        self,
        mode: str = 'sessile',
        calibration: Optional[ScaleCalibration] = None,
        baseline_y: Optional[int] = None,
        frame_step: int = 1,
        smooth_window: int = 7,
        adaptive_window: bool = True,
        min_radius_px: float = 15.0,
        **analyzer_kwargs,
    ):
        self.mode           = mode
        self.calibration    = calibration
        self.baseline_y     = baseline_y
        self.frame_step     = max(1, frame_step)
        self.smooth_window  = smooth_window
        self.adaptive       = adaptive_window
        self.min_radius_px  = min_radius_px
        self._kw            = analyzer_kwargs

        self._analyzer  = ContactAngleAnalyzer(mode=mode, **analyzer_kwargs)
        self._raw:      List[Dict[str, Any]] = []
        self._fps:      float = 30.0

    # ── Analysis ─────────────────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        output_video: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process the video frame by frame.

        Returns list of per-frame result dicts with keys:
            frame, time_s, theta_left, theta_right, theta_mean,
            x_left, x_right, base_radius_px, drop_height_px,
            base_radius_mm, drop_height_mm, volume_ul,
            window_px_used, quality_ok, regime (filled in post-hoc).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h_vid     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_vid     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc,
                                     self._fps / self.frame_step,
                                     (w_vid, h_vid))

        self._raw = []
        frame_idx  = 0
        n_proc     = 0
        base_window_px = float(self._kw.get('window_px', 25.0))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_step != 0:
                    frame_idx += 1
                    continue

                row = self._process_frame(
                    frame, frame_idx, base_window_px
                )
                self._raw.append(row)

                if writer is not None:
                    try:
                        from .visualization import draw_overlay
                        if self._analyzer._result:
                            ov = self._analyzer.get_overlay()
                            writer.write(ov)
                        else:
                            writer.write(frame)
                    except Exception:
                        writer.write(frame)

                n_proc += 1
                if show_progress and n_proc % 20 == 0:
                    pct = frame_idx / total * 100 if total > 0 else 0
                    th  = row.get('theta_mean')
                    r   = row.get('base_radius_px')
                    if th is not None and r is not None:
                        print(f"\r  [{frame_idx}/{total}] {pct:.0f}%  "
                              f"θ={th:.1f}°  R={r:.1f}px",
                              end='', flush=True)
                    else:
                        print(f"\r  [{frame_idx}/{total}] {pct:.0f}%",
                              end='', flush=True)

                if max_frames and n_proc >= max_frames:
                    break
                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_progress:
                print(f"\n  Done: {n_proc} frames analysed.")

        self._annotate_regimes()
        return self._raw

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        base_window_px: float,
    ) -> Dict[str, Any]:
        """Run the full pipeline on a single frame."""
        empty = {
            'frame': frame_idx,
            'time_s': round(frame_idx / self._fps, 4),
            'theta_left': None, 'theta_right': None, 'theta_mean': None,
            'x_left': None, 'x_right': None,
            'base_radius_px': None, 'drop_height_px': None,
            'base_radius_mm': None, 'drop_height_mm': None,
            'volume_ul': None, 'window_px_used': base_window_px,
            'quality_ok': False, 'regime': None,
        }

        try:
            result = self._analyzer.analyze(frame, baseline_y=self.baseline_y)
        except Exception:
            return empty

        xl = result.get('x_left')
        xr = result.get('x_right')
        if xl is None or xr is None:
            return empty

        # Base radius and adaptive window
        base_radius_px = abs(xr - xl) / 2.0
        if base_radius_px < self.min_radius_px:
            return empty   # drop too small to measure reliably

        window_px_used = base_window_px
        if self.adaptive and base_radius_px > 0:
            # Scale window proportionally to radius; clamp to [8, 60]
            window_px_used = float(np.clip(base_radius_px * 0.22, 8, 60))
            # Re-analyse with the adaptive window if it differs significantly
            if abs(window_px_used - base_window_px) > 5:
                try:
                    from .fitting import compute_contact_angles
                    from .detection import extract_contour_points
                    xs_e, ys_e = extract_contour_points(
                        self._analyzer._edges, min_points=10
                    )
                    result_adapt = compute_contact_angles(
                        xs_e, ys_e,
                        baseline_y=float(self._analyzer._baseline_y),
                        window_px=window_px_used,
                    )
                    # Use adapted result if R² is better or similar
                    r2_vals_orig  = [v for v in (result.get('r2_left'),       result.get('r2_right'))       if v is not None]
                    r2_vals_adapt = [v for v in (result_adapt.get('r2_left'), result_adapt.get('r2_right')) if v is not None]
                    r2_orig  = min(r2_vals_orig)  if r2_vals_orig  else 0.0
                    r2_adapt = min(r2_vals_adapt) if r2_vals_adapt else 0.0
                    if r2_adapt >= r2_orig - 0.02:
                        result = result_adapt
                        result['baseline_y'] = self._analyzer._baseline_y
                except Exception:
                    pass

        # Drop height from edges
        edges = self._analyzer._edges
        bl_y  = self._analyzer._baseline_y
        drop_height_px = None
        if edges is not None and bl_y is not None:
            ys_edge, _ = np.where(edges > 0)
            if len(ys_edge):
                drop_height_px = float(abs(bl_y - ys_edge.min()))

        # Quality check: both R² should be > 0.85
        r2l = result.get('r2_left')  or 0
        r2r = result.get('r2_right') or 0
        quality_ok = r2l > 0.85 and r2r > 0.85

        row = {
            'frame':          frame_idx,
            'time_s':         round(frame_idx / self._fps, 4),
            'theta_left':     result.get('theta_left'),
            'theta_right':    result.get('theta_right'),
            'theta_mean':     result.get('theta_mean'),
            'x_left':         xl,
            'x_right':        xr,
            'r2_left':        r2l,
            'r2_right':       r2r,
            'base_radius_px': base_radius_px,
            'drop_height_px': drop_height_px,
            'base_radius_mm': None,
            'drop_height_mm': None,
            'volume_ul':      None,
            'window_px_used': window_px_used,
            'quality_ok':     quality_ok,
            'regime':         None,
        }

        # Physical units
        if self.calibration is not None:
            row['base_radius_mm'] = self.calibration.px_to_mm(base_radius_px)
            if drop_height_px is not None:
                row['drop_height_mm'] = self.calibration.px_to_mm(drop_height_px)
            # Spherical-cap volume
            from .calibration import _spherical_cap_volume
            vol_input = {
                'theta_mean':    result.get('theta_mean'),
                'base_radius_mm': row['base_radius_mm'],
            }
            row['volume_ul'] = _spherical_cap_volume(vol_input)

        return row

    def _annotate_regimes(self) -> None:
        """Fill in the 'regime' key for each frame after the full pass."""
        theta_s  = [r.get('theta_mean')     for r in self._raw]
        radius_s = [r.get('base_radius_px') for r in self._raw]

        smooth_window = max(2, self.smooth_window)   # minimum of 2 for meaningful std
        theta_sm  = _moving_avg(theta_s,  smooth_window)
        radius_sm = _moving_avg(radius_s, smooth_window)

        regimes = detect_regime(theta_sm, radius_sm,
                                window=smooth_window * 2)
        for row, reg in zip(self._raw, regimes):
            row['regime'] = reg

    # ── Derived quantities ────────────────────────────────────────────────────

    def evaporation_rate(
        self,
        smooth: bool = True,
    ) -> List[Optional[float]]:
        """
        Compute dV/dt (µL/s or px³/s) at each frame via central difference.

        Returns a list aligned with self._raw. None where volume is unavailable.
        """
        key = 'volume_ul'   # None when calibration is absent; handled below
        vols  = [r.get(key) for r in self._raw]
        times = [r['time_s'] for r in self._raw]

        if smooth:
            vols = _moving_avg(vols, self.smooth_window)

        rates = [None] * len(vols)
        for i in range(1, len(vols) - 1):
            v0 = vols[i - 1]; v1 = vols[i + 1]
            t0 = times[i - 1]; t1 = times[i + 1]
            if v0 is not None and v1 is not None and (t1 - t0) > 0:
                rates[i] = (v1 - v0) / (t1 - t0)

        return rates

    def regime_summary(self) -> str:
        """Return a text summary of regime fractions and durations."""
        regimes   = [r.get('regime') for r in self._raw]
        total     = len([x for x in regimes if x is not None])
        if total == 0:
            return "No regime data."

        counts = {}
        for reg in regimes:
            if reg is not None:
                counts[reg] = counts.get(reg, 0) + 1

        lines = ["Evaporation regime summary:"]
        total_time = self._raw[-1]['time_s'] - self._raw[0]['time_s'] if self._raw else 0
        for reg, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct  = cnt / total * 100
            secs = pct / 100 * total_time
            lines.append(f"  {reg:<12} {pct:5.1f}%  (~{secs:.0f}s)")

        # Initial and final measurements
        valid = [r for r in self._raw if r.get('theta_mean') is not None]
        if valid:
            first, last = valid[0], valid[-1]
            lines.append(f"\n  θ_initial  = {first['theta_mean']:.1f}°  "
                          f"(t={first['time_s']:.1f}s)")
            lines.append(f"  θ_final    = {last['theta_mean']:.1f}°  "
                          f"(t={last['time_s']:.1f}s)")
            if first.get('base_radius_mm') and last.get('base_radius_mm'):
                lines.append(f"  R_initial  = {first['base_radius_mm']:.3f} mm")
                lines.append(f"  R_final    = {last['base_radius_mm']:.3f} mm")
            if first.get('volume_ul') and last.get('volume_ul'):
                dv = last['volume_ul'] - first['volume_ul']   # negative for evaporation
                dt = last['time_s'] - first['time_s']
                lines.append(f"  ΔV         = {dv:.3f} µL  over {dt:.0f}s")
                # Negate so evaporation rate is reported as a positive value
                lines.append(f"  Mean rate  = {-dv/dt*1000:.3f} nL/s")

        return "\n".join(lines)

    # ── Output ────────────────────────────────────────────────────────────────

    def export_csv(self, path: Union[str, Path]) -> None:
        """Write per-frame results to CSV."""
        if not self._raw:
            print("No results to export.")
            return

        fields = [
            'frame', 'time_s',
            'theta_left', 'theta_right', 'theta_mean',
            'r2_left', 'r2_right',
            'base_radius_px', 'drop_height_px',
            'base_radius_mm', 'drop_height_mm', 'volume_ul',
            'window_px_used', 'quality_ok', 'regime',
        ]
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self._raw)
        print(f"Exported {len(self._raw)} rows → {path}")

    def plot_time_series(
        self,
        save_path: Optional[str] = None,
        show_regime: bool = True,
        show_quality: bool = True,
    ):
        """
        Multi-panel time series plot showing all tracked quantities.

        Panels (top to bottom):
          1. Contact angle θ (left, right, mean)
          2. Base radius R
          3. Volume V  (if calibration available)
          4. Evaporation rate dV/dt  (if calibration available)

        Regime is shown as a colour-coded background band.
        Low-quality frames (R² < 0.85) are shown as faded points.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        has_cal   = self.calibration is not None
        has_vol   = any(r.get('volume_ul') for r in self._raw)
        n_panels  = 2 + (1 if has_vol else 0) + (1 if has_vol else 0)

        fig = plt.figure(figsize=(13, 3.5 * n_panels))
        gs  = GridSpec(n_panels, 1, figure=fig, hspace=0.08)
        fig.patch.set_facecolor('#1a1a1a')

        axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
        for ax in axes:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#aaa')
            for sp in ax.spines.values():
                sp.set_color('#444')

        times = [r['time_s'] for r in self._raw]

        # ── Regime background ─────────────────────────────────────────────────
        if show_regime:
            regimes = [r.get('regime') for r in self._raw]
            for ax in axes:
                prev_reg = None; t_start = times[0]
                for t, reg in zip(times, regimes):
                    if reg != prev_reg:
                        if prev_reg is not None:
                            col = self._REGIME_COLORS.get(prev_reg, '#555')
                            ax.axvspan(t_start, t, color=col, alpha=0.10)
                        t_start  = t
                        prev_reg = reg
                if prev_reg is not None:
                    col = self._REGIME_COLORS.get(prev_reg, '#555')
                    ax.axvspan(t_start, times[-1], color=col, alpha=0.10)

        # Quality mask
        quality = [r.get('quality_ok', True) for r in self._raw]

        def _plot(ax, key, label, color, unit='', ls='-', lw=1.5):
            vals = [r.get(key) for r in self._raw]
            if show_quality:
                # High-quality frames → solid line; low-quality → faded scatter
                good  = [(t, v) for t, v, q in zip(times, vals, quality)
                         if v is not None and q]
                faded = [(t, v) for t, v, q in zip(times, vals, quality)
                         if v is not None and not q]
            else:
                good  = [(t, v) for t, v in zip(times, vals) if v is not None]
                faded = []
            if good:
                t_, v_ = zip(*good)
                ax.plot(t_, v_, color=color, lw=lw, ls=ls, label=label)
            if faded:
                tf, vf = zip(*faded)
                ax.scatter(tf, vf, color=color, s=8, alpha=0.3, zorder=2)
            ax.set_ylabel(f"{label} ({unit})" if unit else label,
                          color='#ccc', fontsize=10)

        # Panel 1 — Contact angle
        ax = axes[0]
        _plot(ax, 'theta_left',  'θ_L',    '#42a5f5', 'deg', '--', 1.2)
        _plot(ax, 'theta_right', 'θ_R',    '#ef5350', 'deg', '--', 1.2)
        _plot(ax, 'theta_mean',  'θ_mean', 'white',   'deg', '-',  2.0)
        ax.set_ylabel('Contact angle (°)', color='#ccc', fontsize=10)
        leg_angles = ax.legend(fontsize=9, facecolor='#333', labelcolor='white',
                               loc='upper right')
        ax.set_title('Evaporation Time Series', color='white',
                     fontsize=13, pad=8)

        # Panel 2 — Base radius
        ax = axes[1]
        if has_cal:
            _plot(ax, 'base_radius_mm', 'Base radius R', '#66bb6a', 'mm')
        else:
            _plot(ax, 'base_radius_px', 'Base radius R', '#66bb6a', 'px')

        # Panel 3 — Volume (if available)
        if has_vol and n_panels >= 3:
            ax = axes[2]
            _plot(ax, 'volume_ul', 'Volume V', '#ffa726', 'µL')

        # Panel 4 — Evaporation rate
        if has_vol and n_panels >= 4:
            ax    = axes[3]
            rates = self.evaporation_rate(smooth=True)
            pairs = [(t, -r) for t, r in zip(times, rates)
                     if r is not None]
            if pairs:
                t_, r_ = zip(*pairs)
                ax.plot(t_, r_, color='#ce93d8', lw=1.5)
                ax.axhline(0, color='#555', lw=0.8, ls='--')
            ax.set_ylabel('Evap. rate\n(µL/s)', color='#ccc', fontsize=10)

        # x-axis only on bottom panel
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        axes[-1].set_xlabel('Time (s)', color='#ccc', fontsize=11)

        for ax in axes:
            ax.grid(True, alpha=0.15, color='#555')
            ax.tick_params(colors='#aaa', labelsize=9)

        # Regime legend — placed upper-left; re-add angle legend so both coexist
        if show_regime:
            patches = [
                mpatches.Patch(color=col, alpha=0.5, label=reg)
                for reg, col in self._REGIME_COLORS.items()
                if reg and reg != 'stable'
            ]
            axes[0].legend(
                handles=patches,
                loc='upper left',
                fontsize=8,
                facecolor='#333',
                labelcolor='white',
                title='Regime',
                title_fontsize=8,
            )
            axes[0].add_artist(leg_angles)   # restore the angle legend

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a1a')
            print(f"Saved: {save_path}")
        plt.show()
        return fig
