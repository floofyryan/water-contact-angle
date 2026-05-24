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
    ):
        if mode not in ("sessile", "captive_bubble"):
            raise ValueError("mode must be 'sessile' or 'captive_bubble'")
        self.mode                     = mode
        self.canny_low                = int(canny_low)
        self.canny_high               = int(canny_high)
        self.clahe_clip               = float(clahe_clip)
        self.blur_ksize               = int(blur_ksize)
        self.window_px                = float(window_px)
        self.baseline_margin          = int(baseline_margin)
        self.baseline_search_fraction = float(baseline_search_fraction)
        self.use_circle_fit           = use_circle_fit

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

        # ── Edges ─────────────────────────────────────────────────────────────
        self._edges = det.detect_edges(
            self._gray,
            self._baseline_y,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            baseline_margin=self.baseline_margin,
        )

        # ── Contour points ────────────────────────────────────────────────────
        try:
            xs, ys = det.extract_contour_points(self._edges, min_points=10)
        except ValueError as exc:
            self._result = _empty_result(self._baseline_y, str(exc))
            return self._result

        # ── PCA contact angle ─────────────────────────────────────────────────
        result = fit.compute_contact_angles(
            xs, ys,
            baseline_y=float(self._baseline_y),
            window_px=self.window_px,
        )

        # ── Optional circle fit ───────────────────────────────────────────────
        if self.use_circle_fit:
            try:
                cb = (self.mode == "captive_bubble")
                circ = fit.fit_circle_contact_angle(xs, ys,
                                                    float(self._baseline_y),
                                                    captive_bubble=cb)
                result["theta_circle"]    = circ.get("theta_mean")
                result["circle_rms_px"]   = circ.get("rms_px")
                result["circle_r2"]       = circ.get("r2_left")
            except Exception:
                result["theta_circle"]    = None
                result["circle_rms_px"]   = None
                result["circle_r2"]       = None

        self._result = result
        return result

    def get_overlay(self, **kwargs) -> np.ndarray:
        """
        Return an annotated BGR image from the last analyze() call.

        Raises RuntimeError if analyze() has not been called yet.
        """
        if self._oriented is None:
            raise RuntimeError("Call analyze() before get_overlay().")
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
