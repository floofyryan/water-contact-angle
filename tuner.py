"""
ThresholdTuner — interactive parameter tuning for ContactAngleAnalyzer.

Opens a matplotlib figure with live-updating sliders for every tunable
parameter. Click on the image to set the baseline manually. When you're
satisfied, call tuner.params to get a dict ready to pass directly to
ContactAngleAnalyzer.

Usage
-----
    from contact_angle.tuner import ThresholdTuner

    tuner = ThresholdTuner('drop.png', mode='sessile')
    tuner.show()                          # blocks until window is closed
    params = tuner.params                 # dict of final values
    ca = ContactAngleAnalyzer(**params)   # use them
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from . import detection as det
from . import fitting as fit
from . import preprocessing as prep
from . import visualization as viz


class ThresholdTuner:
    """
    Interactive parameter tuner.

    Parameters
    ----------
    image       : File path or BGR numpy array.
    mode        : 'sessile' | 'captive_bubble'.
    baseline_y  : Initial baseline row in original image coords.
                  None → auto-detect on first render.

    After calling show(), read tuner.params for the final settings.
    """

    # Slider definitions: (label, key, min, max, init, step)
    # IMPORTANT: blur_ksize must always produce odd values for cv2.GaussianBlur.
    # The valinit=5 with valstep=2 starting from valinit gives 1,3,5,7,... (all odd).
    # If valinit or valmin is changed, ensure they are odd and differ by an even step.
    _SLIDER_DEFS = [
        ("CLAHE clip",      "clahe_clip",               0.5,  6.0, 2.0, 0.1),
        ("Blur ksize",      "blur_ksize",                1,   15,   5,   2),
        ("Canny low",       "canny_low",                 5,  150,  30,   1),
        ("Canny high",      "canny_high",               20,  300, 100,   1),
        ("Baseline margin", "baseline_margin",           0,   30,   5,   1),
        ("Window px",       "window_px",                 5,  100,  25,   1),
    ]

    def __init__(
        self,
        image: Union[str, Path, np.ndarray],
        mode: str = "sessile",
        baseline_y: Optional[int] = None,
    ):
        if mode not in ("sessile", "captive_bubble"):
            raise ValueError("mode must be 'sessile' or 'captive_bubble'")

        self.mode = mode

        # Load & orient
        if isinstance(image, (str, Path)):
            raw = cv2.imread(str(image))
            if raw is None:
                raise FileNotFoundError(f"Cannot load: {image}")
        else:
            raw = np.asarray(image).copy()

        self._raw = raw
        if mode == "captive_bubble":
            self._oriented = prep.flip_for_captive_bubble(raw)
            self._flipped = True
        else:
            self._oriented = raw.copy()
            self._flipped = False

        self._h, self._w = self._oriented.shape[:2]

        # Current parameter values
        self._p: dict = {key: init for _, key, _, _, init, _ in self._SLIDER_DEFS}
        self._p["baseline_search_fraction"] = 0.20

        # Manual baseline (in oriented-image row coords)
        self._baseline_y: Optional[int] = None
        if baseline_y is not None:
            self._baseline_y = (
                self._h - 1 - baseline_y if self._flipped else baseline_y
            )

        # Results (updated on each redraw)
        self._result: Optional[dict] = None

        # Figure handles (set in show())
        self._fig = None
        self._sliders: dict = {}
        self._ax_pre = None
        self._ax_ov  = None
        self._im_pre = None
        self._im_ov  = None
        self._txt    = None
        # Button references must be kept alive to avoid garbage-collection
        self._btn_reset = None
        self._btn_print = None
        self._btn_clrbl = None

    # ─── Public ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_odd(v: int) -> int:
        """Return v if odd, else v+1. Used for OpenCV blur kernel sizes."""
        return v if v % 2 == 1 else v + 1

    @property
    def params(self) -> dict:
        """
        Return the current parameter dict, ready for ContactAngleAnalyzer.

        Example::

            ca = ContactAngleAnalyzer(mode=tuner.mode, **tuner.params)
        """
        out = dict(self._p)
        # blur_ksize must be an odd integer for cv2.GaussianBlur
        out["blur_ksize"]      = self._ensure_odd(int(out["blur_ksize"]))
        out["canny_low"]       = int(out["canny_low"])
        out["canny_high"]      = int(out["canny_high"])
        out["baseline_margin"] = int(out["baseline_margin"])
        out["window_px"]       = float(out["window_px"])
        out.pop("baseline_search_fraction", None)   # internal only
        return out

    def show(self) -> None:
        """
        Open the interactive tuning window. Blocks until the window is closed.

        Click anywhere on the LEFT image panel to set the baseline manually.
        Drag sliders to update detection parameters in real time.
        """
        n_sliders = len(self._SLIDER_DEFS)

        # Layout: 2 image panels + result text + N sliders + buttons
        slider_h = 0.032
        slider_gap = 0.008
        total_slider_h = (slider_h + slider_gap) * n_sliders
        btn_h = 0.045
        bottom_margin = 0.03
        slider_bottom = bottom_margin + btn_h + 0.02

        img_bottom = slider_bottom + total_slider_h + 0.03
        img_height = 1.0 - img_bottom - 0.08

        fig = plt.figure(figsize=(15, 9))
        self._fig = fig
        fig.patch.set_facecolor("#1e1e1e")

        # Image axes
        self._ax_pre = fig.add_axes([0.03, img_bottom, 0.44, img_height])
        self._ax_ov  = fig.add_axes([0.53, img_bottom, 0.44, img_height])
        for ax in (self._ax_pre, self._ax_ov):
            ax.set_facecolor("#2a2a2a")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        self._ax_pre.set_title(
            "Preprocessed + Edges  (click to set baseline)",
            color="white", fontsize=10, pad=4
        )
        self._ax_ov.set_title(
            "Analysis Overlay",
            color="white", fontsize=10, pad=4
        )

        # Click handler for baseline
        fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Sliders
        slider_left, slider_width = 0.12, 0.78
        for i, (label, key, lo, hi, init, step) in enumerate(self._SLIDER_DEFS):
            row = n_sliders - 1 - i
            y0 = slider_bottom + row * (slider_h + slider_gap)
            ax_s = fig.add_axes(
                [slider_left, y0, slider_width, slider_h],
                facecolor="#2d2d2d"
            )
            valinit = self._p[key]
            sldr = Slider(
                ax_s, label, lo, hi,
                valinit=valinit,
                valstep=step,
                color="#4a90d9",
                initcolor="#4a90d9",
            )
            sldr.label.set_color("white")
            sldr.valtext.set_color("#aaaaaa")
            sldr.on_changed(lambda val, k=key: self._on_slider(k, val))
            self._sliders[key] = sldr

        # Buttons
        ax_reset  = fig.add_axes([0.12, bottom_margin, 0.14, btn_h])
        ax_print  = fig.add_axes([0.30, bottom_margin, 0.20, btn_h])
        ax_clrbl  = fig.add_axes([0.54, bottom_margin, 0.18, btn_h])

        # Store button objects as instance attributes — matplotlib Button widgets
        # are garbage-collected if only referenced by local variables, which
        # causes them to stop responding to clicks in some backends.
        self._btn_reset = Button(ax_reset, "Reset defaults",  color="#333333", hovercolor="#555555")
        self._btn_print = Button(ax_print, "Print settings",  color="#333333", hovercolor="#555555")
        self._btn_clrbl = Button(ax_clrbl, "Auto baseline",   color="#333333", hovercolor="#555555")
        for btn, cb in [
            (self._btn_reset, self._on_reset),
            (self._btn_print, self._on_print),
            (self._btn_clrbl, self._on_clear_baseline),
        ]:
            btn.label.set_color("white")
            btn.on_clicked(cb)

        # Result text
        self._txt = fig.text(
            0.5, img_bottom - 0.025, "",
            ha="center", va="top",
            color="#dddddd", fontsize=10,
            fontfamily="monospace",
        )

        # First render
        self._redraw()
        plt.show()

    # ─── Event handlers ───────────────────────────────────────────────────────

    def _on_slider(self, key: str, val: float) -> None:
        self._p[key] = val
        self._redraw()

    def _on_click(self, event) -> None:
        """Set baseline_y by clicking on either image panel."""
        if event.inaxes not in (self._ax_pre, self._ax_ov):
            return
        if event.ydata is None:
            return
        self._baseline_y = int(round(event.ydata))
        self._redraw()

    def _on_reset(self, _event) -> None:
        for _, key, _, _, init, _ in self._SLIDER_DEFS:
            self._p[key] = init
            self._sliders[key].set_val(init)
        self._redraw()

    def _on_print(self, _event) -> None:
        p = self.params
        print("\n── ThresholdTuner: final parameters ──────────────────")
        for k, v in p.items():
            print(f"  {k:<25s} = {v}")
        if self._baseline_y is not None:
            bl_orig = (
                self._h - 1 - self._baseline_y
                if self._flipped else self._baseline_y
            )
            print(f"  {'baseline_y':<25s} = {bl_orig}  (pass to analyze())")
        print("  Copy into ContactAngleAnalyzer(**above_dict)")
        print("────────────────────────────────────────────────────\n")

    def _on_clear_baseline(self, _event) -> None:
        self._baseline_y = None
        self._redraw()

    # ─── Core redraw ──────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """Re-run the pipeline with current params and refresh displays."""
        p = self._p

        # Preprocessing
        ksize = self._ensure_odd(int(p["blur_ksize"]))
        processed = prep.preprocess(
            self._oriented,
            clahe_clip=float(p["clahe_clip"]),
            blur_ksize=ksize,
        )

        # Baseline
        if self._baseline_y is None:
            baseline_y = det.detect_baseline_auto(
                processed,
                search_fraction=float(p["baseline_search_fraction"]),
            )
        else:
            baseline_y = self._baseline_y

        # Edges
        edges = det.detect_edges(
            processed,
            baseline_y,
            canny_low=int(p["canny_low"]),
            canny_high=int(p["canny_high"]),
            baseline_margin=int(p["baseline_margin"]),
        )

        # Build preprocessed display: edges in yellow, baseline in white
        pre_disp = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        pre_disp[edges > 0] = [0, 255, 255]        # cyan edges
        cv2.line(pre_disp, (0, baseline_y), (self._w - 1, baseline_y),
                 (255, 255, 255), 1)

        # Contact angle computation
        result_str = ""
        overlay = viz.draw_overlay(
            self._oriented, edges,
            {"theta_left": None, "theta_right": None, "theta_mean": None,
             "asymmetry": None, "x_left": None, "x_right": None,
             "r2_left": None, "r2_right": None,
             "direction_left": None, "direction_right": None},
            baseline_y,
        )

        try:
            xs, ys = det.extract_contour_points(edges, min_points=20)
            result = fit.compute_contact_angles(
                xs, ys, float(baseline_y),
                window_px=float(p["window_px"]),
            )
            self._result = result
            overlay = viz.draw_overlay(
                self._oriented, edges, result, baseline_y
            )

            parts = []
            for side in ("left", "right"):
                th = result.get(f"theta_{side}")
                r2 = result.get(f"r2_{side}")
                n  = result.get(f"n_{side}")
                if th is not None:
                    parts.append(
                        f"θ_{side[0].upper()}={th:.1f}°  R²={r2:.3f}  n={n}"
                    )
            if result.get("theta_mean") is not None:
                parts.append(f"│  mean={result['theta_mean']:.1f}°")
            if result.get("asymmetry") is not None:
                parts.append(f"|ΔΘ|={result['asymmetry']:.1f}°")

            if self._baseline_y is not None:
                # Convert oriented-image row back to original-image row for display
                bl_orig = (self._h - 1 - baseline_y) if self._flipped else baseline_y
                bl_label = f"baseline={bl_orig} (manual)"
            else:
                bl_label = f"baseline={baseline_y} (auto)"
            result_str = "    ".join(parts) + f"    {bl_label}"

        except ValueError as exc:
            result_str = f"⚠  {exc}"
            self._result = None

        # Update image displays
        def _bgr2rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._im_pre is None:
            self._im_pre = self._ax_pre.imshow(
                _bgr2rgb(pre_disp), aspect="auto"
            )
            self._im_ov  = self._ax_ov.imshow(
                _bgr2rgb(overlay), aspect="auto"
            )
        else:
            self._im_pre.set_data(_bgr2rgb(pre_disp))
            self._im_ov.set_data(_bgr2rgb(overlay))

        if self._txt is not None:
            self._txt.set_text(result_str)

        self._fig.canvas.draw_idle()
