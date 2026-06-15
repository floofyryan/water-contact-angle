"""
analyzer.py — Central ContactAngleAnalyzer: the main entry point for analysis.

This class orchestrates the full pipeline:

    Image input
        ↓
    preprocessing.py   — CLAHE + Gaussian blur → grayscale
        ↓
    detection.py       — baseline detection, Canny edge detection
        ↓
    fitting.py         — PCA tangent estimation (or circle fit)
        ↓
    visualization.py   — annotated overlay image

All other modules (evaporation, live, tuner, publication) drive their
analysis through this class.

Usage
-----
    from contact_angle.analyzer import ContactAngleAnalyzer
    import cv2

    ca = ContactAngleAnalyzer(mode='sessile', canny_high=90, window_px=30)
    result = ca.analyze('drop.png')
    print(f"Left: {result['theta_left']:.1f}°  Right: {result['theta_right']:.1f}°")

    ca.show()          # OpenCV display window
    ca.save_overlay('annotated.png')
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from . import detection   as det
from . import fitting     as fit
from . import preprocessing as prep
from . import visualization as viz


class ContactAngleAnalyzer:
    """
    End-to-end contact angle analysis from a single image or array.

    Parameters
    ----------
    mode               : 'sessile' | 'captive_bubble'
    canny_low          : Lower Canny threshold.
    canny_high         : Upper Canny threshold.
    clahe_clip         : CLAHE clip limit for contrast enhancement.
    blur_ksize         : Gaussian blur kernel size (odd int).
    window_px          : Half-height (px) of PCA fitting window at each
                         contact point.
    baseline_margin    : Rows to exclude directly above the baseline.
    baseline_search_fraction : Fraction of image height searched for
                         the substrate when baseline_y is not fixed.
    use_circle_fit     : If True, use algebraic circle fit in addition to
                         PCA and include 'theta_circle' in results.
    """

    def __init__(
        self,
        mode: str = "sessile",
        canny_low: int = 30,
        canny_high: int = 100,
        clahe_clip: float = 2.0,
        blur_ksize: int = 5,
        window_px: float = 25.0,
        baseline_margin: int = 5,
        baseline_search_fraction: float = 0.20,
        use_circle_fit: bool = False,
        tilt_correction: bool = False,
        method: str = "pca",
    ):
        if mode not in ("sessile", "captive_bubble"):
            raise ValueError("mode must be 'sessile' or 'captive_bubble'")
        valid_methods = ("pca", "circle", "ellipse", "height_width", "polynomial")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self.mode                     = mode
        self.canny_low                = int(canny_low)
        self.canny_high               = int(canny_high)
        self.clahe_clip               = float(clahe_clip)
        self.blur_ksize               = int(blur_ksize)
        self.window_px                = float(window_px)
        self.baseline_margin          = int(baseline_margin)
        self.baseline_search_fraction = float(baseline_search_fraction)
        self.use_circle_fit           = use_circle_fit
        self.tilt_correction          = tilt_correction
        self.method                   = method

        # State set by analyze()
        self._raw:       Optional[np.ndarray]       = None  # original BGR input
        self._oriented:  Optional[np.ndarray]       = None  # BGR, substrate at bottom
        self._gray:      Optional[np.ndarray]       = None  # preprocessed grayscale
        self._edges:     Optional[np.ndarray]       = None  # binary edge image
        self._baseline_y: Optional[int]             = None
        self._result:    Optional[Dict[str, Any]]   = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        baseline_y: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on a single image.

        Args:
            image      : File path (str/Path) or BGR/grayscale numpy array.
            baseline_y : Substrate row in the ORIGINAL (pre-flip) image.
                         If None, auto-detected from the image.

        Returns:
            Result dict with keys:
                theta_left, theta_right, theta_mean, asymmetry
                x_left, x_right
                direction_left, direction_right
                r2_left, r2_right
                n_left, n_right
                baseline_y
                (+ circle fit keys if use_circle_fit=True)
        """
        # ── Load ──────────────────────────────────────────────────────────────
        self._raw = _load_image(image)

        # ── Orient ────────────────────────────────────────────────────────────
        if self.mode == "captive_bubble":
            self._oriented = prep.flip_for_captive_bubble(self._raw)
            if baseline_y is not None:
                h = self._oriented.shape[0]
                baseline_y_oriented = h - 1 - baseline_y
            else:
                baseline_y_oriented = None
        else:
            self._oriented     = self._raw.copy()
            baseline_y_oriented = baseline_y

        # ── Preprocess ────────────────────────────────────────────────────────
        self._gray = prep.preprocess(
            self._oriented,
            clahe_clip=self.clahe_clip,
            blur_ksize=self.blur_ksize,
        )

        # ── Baseline ──────────────────────────────────────────────────────────
        if baseline_y_oriented is None:
            self._baseline_y = det.detect_baseline_auto(
                self._gray,
                search_fraction=self.baseline_search_fraction,
            )
        else:
            self._baseline_y = int(baseline_y_oriented)

        # ── Tilt correction ───────────────────────────────────────────────────
        # If requested and the user hasn't pinned a manual baseline, detect the
        # substrate tilt polynomial and physically rotate the image so the
        # baseline becomes horizontal.  This is done BEFORE edge detection so
        # every downstream step works in the corrected coordinate system.
        if self.tilt_correction and baseline_y_oriented is None:
            try:
                from . import baseline as _bl_mod
                _poly = _bl_mod.detect_baseline_poly(
                    self._gray,
                    search_fraction=self.baseline_search_fraction,
                )
                if _poly.max_deviation_px() > 2.0:
                    self._oriented, self._baseline_y = _bl_mod.straighten_image(
                        self._oriented, _poly
                    )
                    self._gray = prep.preprocess(
                        self._oriented,
                        clahe_clip=self.clahe_clip,
                        blur_ksize=self.blur_ksize,
                    )
            except Exception:
                pass   # silently fall back to flat baseline

        # ── Edges ─────────────────────────────────────────────────────────────
        self._edges = det.detect_edges(
            self._gray,
            self._baseline_y,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            baseline_margin=self.baseline_margin,
            suppress_reflections=True,
        )

        # ── Contour points ────────────────────────────────────────────────────
        try:
            xs, ys = det.extract_contour_points(
                self._edges, min_points=10, baseline_y=self._baseline_y
            )
        except ValueError as exc:
            self._result = _empty_result(self._baseline_y, str(exc))
            return self._result

        cb = (self.mode == "captive_bubble")
        bl = float(self._baseline_y)

        # ── PCA contact angle (reference method, always computed) ──────────────
        result = fit.compute_contact_angles(
            xs, ys, baseline_y=bl, window_px=self.window_px,
        )
        # Captive bubble: PCA measures through the GAS phase in the flipped
        # image; convention reports through the LIQUID phase = 180° − θ.
        if cb:
            for key in ("theta_left", "theta_right", "theta_mean"):
                if result.get(key) is not None:
                    result[key] = 180.0 - result[key]
            if result.get("theta_left") is not None and result.get("theta_right") is not None:
                result["asymmetry"] = abs(result["theta_left"] - result["theta_right"])

        # ── Run every method for side-by-side comparison ──────────────────────
        methods = self._run_all_methods(xs, ys, bl, cb)

        # Comparison angles (means) for the readout / CSV
        result["theta_pca"] = result.get("theta_mean")
        for name in ("circle", "ellipse", "height_width", "polynomial"):
            m = methods.get(name)
            result[f"theta_{name}"] = m.get("theta_mean") if m else None

        # ── Geometry for overlay drawing (only the selected method's shape) ────
        draw_circle  = (self.method == "circle") or self.use_circle_fit
        if draw_circle and methods.get("circle"):
            c = methods["circle"]
            result["circle_cx"]      = c.get("cx")
            result["circle_cy"]      = c.get("cy")
            result["circle_radius"]  = c.get("radius")
            result["circle_x_left"]  = c.get("x_left")
            result["circle_x_right"] = c.get("x_right")
            result["circle_rms_px"]  = c.get("rms_px")
        if self.method == "ellipse" and methods.get("ellipse"):
            e = methods["ellipse"]
            for k in ("ellipse_cx", "ellipse_cy", "ellipse_a",
                      "ellipse_b", "ellipse_theta"):
                result[k] = e.get(k)
            result["ellipse_rms_px"] = e.get("rms_px")

        # ── Promote the selected method to the primary result keys ────────────
        # The overlay arc, tangent and main θ readout use these keys.
        if self.method != "pca" and methods.get(self.method):
            chosen = methods[self.method]
            for k in ("theta_left", "theta_right", "theta_mean", "asymmetry",
                      "x_left", "x_right", "direction_left", "direction_right"):
                if chosen.get(k) is not None:
                    result[k] = chosen[k]

        result["method"] = self.method
        self._result = result
        return result

    def _run_all_methods(self, xs, ys, baseline_y, cb) -> Dict[str, Dict[str, Any]]:
        """
        Run each available fitting method, returning {name: result_dict} for
        those that succeed.  Each is wrapped so one method failing never
        aborts the analysis.
        """
        out: Dict[str, Dict[str, Any]] = {}

        def _try(name, fn):
            try:
                out[name] = fn()
            except Exception:
                pass

        _try("circle", lambda: self._circle_with_cb(xs, ys, baseline_y, cb))
        _try("ellipse", lambda: fit.fit_ellipse_contact_angle(
            xs, ys, baseline_y, captive_bubble=cb))
        _try("height_width", lambda: fit.height_width_angle(
            xs, ys, baseline_y, captive_bubble=cb))
        _try("polynomial", lambda: fit.polynomial_tangent_angle(
            xs, ys, baseline_y, window_px=max(self.window_px, 30.0),
            captive_bubble=cb))
        return out

    @staticmethod
    def _circle_with_cb(xs, ys, baseline_y, cb) -> Dict[str, Any]:
        """Circle fit with the captive-bubble 180° correction applied."""
        circ = fit.fit_circle_contact_angle(xs, ys, baseline_y, captive_bubble=cb)
        if cb:
            for k in ("theta_left", "theta_right", "theta_mean"):
                if circ.get(k) is not None:
                    circ[k] = 180.0 - circ[k]
        return circ

    def get_overlay(self, **kwargs) -> np.ndarray:
        """
        Return an annotated BGR image from the last analyze() call.

        Raises RuntimeError if analyze() has not been called yet.
        """
        if self._oriented is None:
            raise RuntimeError("Call analyze() before get_overlay().")

        if self.mode == "captive_bubble":
            h = self._raw.shape[0]
            bl_orig = h - 1 - (self._baseline_y or 0)

            # Remap fitted-shape centres from flipped coords to original coords.
            # x is unaffected by a vertical flip; only y maps as y' = h-1-y.
            # For the rotated ellipse the tilt angle also negates.
            cb_result = copy.copy(self._result)
            if cb_result.get("circle_cy") is not None:
                cb_result["circle_cy"] = h - 1 - cb_result["circle_cy"]
            if cb_result.get("ellipse_cy") is not None:
                cb_result["ellipse_cy"] = h - 1 - cb_result["ellipse_cy"]
            if cb_result.get("ellipse_theta") is not None:
                cb_result["ellipse_theta"] = -cb_result["ellipse_theta"]

            return viz.draw_overlay(
                self._raw,
                np.flipud(self._edges),
                cb_result,
                bl_orig,
                captive_bubble=True,
                **kwargs,
            )

        return viz.draw_overlay(
            self._oriented,
            self._edges,
            self._result,
            self._baseline_y or 0,
            **kwargs,
        )

    def show(self, window_name: str = "Contact Angle") -> None:
        """Display the overlay in an OpenCV window (press any key to close)."""
        overlay = self.get_overlay()
        cv2.imshow(window_name, overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_overlay(self, path: Union[str, Path]) -> None:
        """Save the annotated overlay image to disk."""
        cv2.imwrite(str(path), self.get_overlay())
        print(f"Saved overlay → {path}")

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def theta_left(self) -> Optional[float]:
        return self._result.get("theta_left") if self._result else None

    @property
    def theta_right(self) -> Optional[float]:
        return self._result.get("theta_right") if self._result else None

    @property
    def theta_mean(self) -> Optional[float]:
        return self._result.get("theta_mean") if self._result else None

    def __repr__(self) -> str:
        th = f"θ={self.theta_mean:.1f}°" if self.theta_mean is not None else "no result"
        return f"ContactAngleAnalyzer(mode={self.mode!r}, {th})"


# ── Batch analysis ────────────────────────────────────────────────────────────

def analyze_batch(
    image_paths: List[Union[str, Path]],
    baseline_y: Optional[int] = None,
    **analyzer_kwargs,
) -> List[Dict[str, Any]]:
    """
    Analyse a list of images and return per-image result dicts.

    Args:
        image_paths    : List of file paths.
        baseline_y     : Fixed baseline row (applied to all images). If None,
                         auto-detected per image.
        **analyzer_kwargs : Forwarded to ContactAngleAnalyzer.

    Returns:
        List of result dicts in the same order as image_paths.
        Each dict has an extra 'path' key with the source file path.
    """
    ca = ContactAngleAnalyzer(**analyzer_kwargs)
    results = []
    for p in image_paths:
        try:
            r = ca.analyze(p, baseline_y=baseline_y)
        except Exception as exc:
            r = _empty_result(baseline_y, str(exc))
        r["path"] = str(p)
        results.append(r)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from disk or validate a numpy array."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image}")
        return img
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(image)
    raise TypeError(f"Expected str, Path, or ndarray; got {type(image)}")


def _empty_result(baseline_y: Optional[int], error: str = "") -> Dict[str, Any]:
    """Return a fully-keyed result dict with all values None."""
    return {
        "theta_left": None, "theta_right": None, "theta_mean": None,
        "asymmetry": None,
        "x_left": None, "x_right": None,
        "direction_left": None, "direction_right": None,
        "r2_left": None, "r2_right": None,
        "n_left": None, "n_right": None,
        "baseline_y": float(baseline_y) if baseline_y is not None else None,
        "error": error,
    }
