"""
Publication-quality figure export.

Produces clean, annotated figures suitable for journal submission:
  - Drop image with arc-style contact angle annotations
  - Scale bar with proper physical units
  - Minimal, non-distracting overlay
  - Configurable font sizes, line widths, and colours
  - 300 dpi TIFF or PNG export

Usage
-----
    from contact_angle.publication import PublicationFigure
    from contact_angle.calibration import ScaleCalibration

    cal = ScaleCalibration(pixels_per_mm=47.3)

    fig = PublicationFigure(
        image=analyzer._oriented,
        edge_image=analyzer._edges,
        result=analyzer._result,
        baseline_y=analyzer._baseline_y,
        calibration=cal,
    )
    fig.save('figure_1a.tiff', dpi=300)
    fig.show()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.lines import Line2D


# ─── Default style ────────────────────────────────────────────────────────────

_STYLE = {
    "baseline_color":     "#ffffff",
    "baseline_lw":        1.0,
    "edge_color":         "#00e5ff",    # cyan
    "edge_alpha":         0.5,
    "contact_dot_radius": 4,            # pixels in image space
    "tangent_color":      "#ffeb3b",    # yellow
    "tangent_lw":         1.2,
    "tangent_length_mm":  0.4,          # how far tangent lines extend (mm)
    "arc_color_left":     "#ef5350",    # red
    "arc_color_right":    "#42a5f5",    # blue
    "arc_lw":             1.8,
    "arc_radius_mm":      0.25,         # arc radius in mm
    "label_fontsize":     9,
    "label_color":        "#ffffff",
    "scalebar_color":     "#ffffff",
    "scalebar_lw":        2.0,
    "bg_color":           "#111111",
}


class PublicationFigure:
    """
    Generate a publication-quality contact angle figure.

    The figure shows the drop image with:
      - Detected edges highlighted (optional)
      - Horizontal baseline
      - Contact points marked with dots
      - Tangent lines from each contact point
      - Angle arcs between baseline and tangent
      - Angle labels (degrees)
      - Scale bar

    Parameters
    ----------
    image      : BGR or grayscale image (oriented, not flipped back).
    edge_image : Binary edge image from detect_edges().
    result     : Result dict from compute_contact_angles().
    baseline_y : Baseline row in the (oriented) image.
    calibration: ScaleCalibration instance (required for scale bar and
                 physical arc radii).
    style      : Dict of style overrides (see _STYLE for keys).
    show_edges : Whether to overlay detected edges.
    """

    def __init__(
        self,
        image: np.ndarray,
        edge_image: np.ndarray,
        result: Dict[str, Any],
        baseline_y: int,
        calibration: Optional[Any] = None,   # ScaleCalibration | None
        style: Optional[Dict] = None,
        show_edges: bool = True,
    ):
        self.image      = image
        self.edges      = edge_image
        self.result     = result
        self.baseline_y = baseline_y
        self.cal        = calibration
        self.style      = dict(_STYLE)
        if style:
            self.style.update(style)
        self.show_edges = show_edges

        self._fig: Optional[plt.Figure] = None

    # ─── Build ────────────────────────────────────────────────────────────────

    def _build(self, figsize: Tuple[float, float] = (5, 4)) -> plt.Figure:
        """Render the figure and return it."""
        if self.result is None:
            raise ValueError(
                "PublicationFigure: result is None — run analysis before building the figure."
            )
        s = self.style
        h, w = self.image.shape[:2]

        # Convert to RGB display image
        if len(self.image.shape) == 2:
            display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        else:
            display = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).copy()

        # Overlay edges in semi-transparent cyan
        if self.show_edges and self.edges is not None:
            edge_mask = self.edges > 0
            edge_layer = np.zeros_like(display)
            edge_layer[edge_mask] = [0, 229, 255]
            alpha = s["edge_alpha"]
            display = np.where(
                edge_mask[:, :, None],
                (display * (1 - alpha) + edge_layer * alpha).astype(np.uint8),
                display,
            )

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(s["bg_color"])
        ax.set_facecolor(s["bg_color"])
        ax.imshow(display)
        ax.axis("off")

        # Baseline
        ax.axhline(
            self.baseline_y,
            color=s["baseline_color"],
            linewidth=s["baseline_lw"],
            linestyle="--",
            alpha=0.7,
        )

        # Per-side annotations
        for side, arc_color in [("left",  s["arc_color_left"]),
                                 ("right", s["arc_color_right"])]:
            x_c   = self.result.get(f"x_{side}")
            theta = self.result.get(f"theta_{side}")
            d     = self.result.get(f"direction_{side}")

            if x_c is None or theta is None:
                continue

            xi, yi = float(x_c), float(self.baseline_y)

            # Contact point dot
            ax.plot(xi, yi, "o",
                    color=arc_color,
                    markersize=s["contact_dot_radius"],
                    zorder=5)

            # Tangent line
            if d is not None:
                dx_t, dy_t = d   # unit vector pointing into drop (dy_t < 0)
                t_px = (
                    self.cal.mm_to_px(s["tangent_length_mm"])
                    if self.cal else 50
                )
                ax.plot(
                    [xi - dx_t * t_px, xi + dx_t * t_px],
                    [yi - dy_t * t_px, yi + dy_t * t_px],
                    color=s["tangent_color"],
                    linewidth=s["tangent_lw"],
                    zorder=4,
                )

            # Angle arc
            arc_r_px = (
                self.cal.mm_to_px(s["arc_radius_mm"])
                if self.cal else 30
            )
            self._draw_arc(ax, xi, yi, theta, side, arc_r_px, arc_color, s)

            # Angle label positioned along the arc bisector
            self._draw_label(ax, xi, yi, theta, side, arc_r_px, s)

        # Scale bar
        if self.cal is not None:
            self._draw_scalebar(ax, w, h, s)

        plt.tight_layout(pad=0.1)
        self._fig = fig
        return fig

    def _draw_arc(
        self,
        ax,
        xi: float, yi: float,
        theta: float,
        side: str,
        r: float,
        color: str,
        s: dict,
    ) -> None:
        """Draw a circular arc showing the contact angle."""
        # In matplotlib display coords (y-down), angles are measured from +x
        # axis clockwise.
        #
        # For LEFT contact point, the baseline points RIGHT (+x) and the
        # tangent goes up-right for acute angles. The arc spans from 0° (right)
        # up to θ (measured upward = negative y direction in mpl = negative
        # display angle). In mpl (y-down), "up" is negative degree so the arc
        # is at angle 360°-θ to 0° going clockwise.
        # Easier: use mpl Arc with angle_from and angle_to in display coords.
        #
        # mpl Arc angles are COUNTER-clockwise from +x in display coords (y-down).
        # In display coords, "up" is -y direction = 90° CCW from +x in standard,
        # but in mpl's imshow y-flipped axis it's actually 270°. This is tricky.
        # Workaround: draw a parametric arc manually.

        th_rad = math.radians(theta)

        if side == "left":
            # Arc from baseline (+x direction, angle=0 in image) to tangent direction
            # Baseline direction: (1, 0) in image coords (pointing right)
            # Tangent direction (into drop, upward): angle θ from baseline CCW in y-up
            # In image coords (y-down), CCW in y-up = CW in y-down
            t_vals = np.linspace(0, th_rad, 60)
            # In y-up frame: angle sweeps CCW from 0 to θ
            # x_arc = xi + r * cos(t), y_arc (y-up) = r * sin(t)
            # In image (y-down): y_image = yi - r * sin(t)
            x_arc = xi + r * np.cos(t_vals)
            y_arc = yi - r * np.sin(t_vals)
        else:
            # RIGHT contact: arc from baseline (-x direction, angle=180°) to tangent
            # Mirror of left side
            t_vals = np.linspace(0, th_rad, 60)
            x_arc = xi - r * np.cos(t_vals)
            y_arc = yi - r * np.sin(t_vals)

        ax.plot(x_arc, y_arc, color=color, linewidth=s["arc_lw"], zorder=5)

    def _draw_label(
        self,
        ax,
        xi: float, yi: float,
        theta: float,
        side: str,
        r: float,
        s: dict,
    ) -> None:
        """Place angle label at the arc midpoint, offset outward."""
        th_rad = math.radians(theta)
        half   = th_rad / 2
        label_r = r * 1.55   # slightly beyond the arc

        if side == "left":
            lx = xi + label_r * math.cos(half)
            ly = yi - label_r * math.sin(half)
            ha = "left"
        else:
            lx = xi - label_r * math.cos(half)
            ly = yi - label_r * math.sin(half)
            ha = "right"

        ax.text(
            lx, ly,
            f"{theta:.1f}°",
            color=s["label_color"],
            fontsize=s["label_fontsize"],
            ha=ha, va="center",
            fontweight="bold",
            bbox=dict(facecolor="none", edgecolor="none", pad=1),
            zorder=6,
        )

    def _draw_scalebar(
        self,
        ax,
        w: int, h: int,
        s: dict,
        bar_mm: float = 1.0,
    ) -> None:
        """Draw a scale bar in the bottom-right corner."""
        bar_px = self.cal.mm_to_px(bar_mm)
        margin = 0.05
        x_end  = w * (1 - margin)
        x_start = x_end - bar_px
        y_bar   = h * (1 - margin)

        ax.plot(
            [x_start, x_end], [y_bar, y_bar],
            color=s["scalebar_color"],
            linewidth=s["scalebar_lw"],
            solid_capstyle="butt",
            zorder=6,
        )
        # End ticks
        tick_h = h * 0.012
        for x in (x_start, x_end):
            ax.plot(
                [x, x], [y_bar - tick_h, y_bar + tick_h],
                color=s["scalebar_color"],
                linewidth=s["scalebar_lw"],
                zorder=6,
            )
        # Label
        ax.text(
            (x_start + x_end) / 2, y_bar - tick_h * 2.5,
            f"{bar_mm:g} mm",
            color=s["scalebar_color"],
            fontsize=s["label_fontsize"],
            ha="center", va="bottom",
            zorder=6,
        )

    # ─── Public methods ───────────────────────────────────────────────────────

    def show(self, figsize: Tuple[float, float] = (5, 4)) -> None:
        """Display the figure interactively."""
        fig = self._build(figsize=figsize)
        plt.show()

    def save(
        self,
        path: Union[str, Path],
        dpi: int = 300,
        figsize: Tuple[float, float] = (5, 4),
        facecolor: str = "black",
    ) -> None:
        """
        Save a publication-quality figure.

        Args:
            path    : Output path (.tiff, .png, .svg, .pdf).
            dpi     : Resolution (300 for most journals; 600 for line art).
            figsize : Figure size in inches. Default (5, 4) ≈ single column.
            facecolor: Figure background colour.
        """
        fig = self._build(figsize=figsize)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=facecolor,
            pad_inches=0.02,
        )
        print(f"Saved {dpi} dpi figure → {path}")
        plt.close(fig)

    @classmethod
    def from_analyzer(
        cls,
        analyzer,
        calibration=None,
        **kwargs,
    ) -> "PublicationFigure":
        """
        Convenience constructor from a ContactAngleAnalyzer instance
        (after calling analyze()).

        Example::

            ca = ContactAngleAnalyzer(mode='sessile')
            ca.analyze('drop.png')
            fig = PublicationFigure.from_analyzer(ca, calibration=cal)
            fig.save('figure_1.tiff')
        """
        return cls(
            image=analyzer._oriented,
            edge_image=analyzer._edges,
            result=analyzer._result,
            baseline_y=analyzer._baseline_y,
            calibration=calibration,
            **kwargs,
        )
