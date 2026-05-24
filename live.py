"""
LiveAnalyzer — real-time contact angle measurement from a camera feed.

Opens an OpenCV window showing the camera feed with live angle overlay.
Useful for monitoring evaporation, advancing/receding measurements, or
just dialling in parameters before a batch run.

Controls
--------
  b         : Set baseline — click on the video frame after pressing b
  s         : Save current frame + result to disk
  l         : Toggle CSV logging on/off
  +/-       : Raise/lower Canny high threshold by 5
  [/]       : Raise/lower window_px by 5
  r         : Reset to default parameters
  q / Esc   : Quit

Usage
-----
    from contact_angle.live import LiveAnalyzer

    la = LiveAnalyzer(
        mode='sessile',
        camera=0,               # device index or video file path
        baseline_y=420,         # optional fixed baseline
        log_csv='live_log.csv', # optional: log angles to CSV
        canny_low=30,
        canny_high=100,
        window_px=25,
    )
    la.run()
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from .analyzer import ContactAngleAnalyzer
from . import detection as det
from . import preprocessing as prep
from . import fitting as fit
from . import visualization as viz


class LiveAnalyzer:
    """
    Real-time contact angle analysis from a camera or video file.

    Parameters
    ----------
    mode        : 'sessile' | 'captive_bubble'
    camera      : Camera device index (int) or path to video file (str/Path).
    baseline_y  : Fixed baseline row. If None, auto-detected every frame
                  (slower and noisier — use manual for live work).
    log_csv     : Path to write a running CSV log. None = no logging.
    save_dir    : Directory for saved frames. Defaults to './live_saves/'.
    fps_limit   : Cap analysis rate (frames/sec). None = as fast as possible.
    **analyzer_kwargs : Forwarded to ContactAngleAnalyzer.
    """

    # Keys for in-window help overlay
    _HELP = [
        "b : set baseline (then click)",
        "s : save frame",
        "l : toggle CSV log",
        "+ / - : Canny high ±5",
        "[ / ] : window_px ±5",
        "r : reset params",
        "q/Esc : quit",
    ]

    def __init__(
        self,
        mode: str = "sessile",
        camera: Union[int, str, Path] = 0,
        baseline_y: Optional[int] = None,
        log_csv: Optional[Union[str, Path]] = None,
        save_dir: Union[str, Path] = "live_saves",
        fps_limit: Optional[float] = None,
        **analyzer_kwargs,
    ):
        self.mode        = mode
        self.camera      = camera
        self.baseline_y  = baseline_y
        self.log_csv     = Path(log_csv) if log_csv else None
        self.save_dir    = Path(save_dir)
        self._fps_limit  = fps_limit
        self._kw         = analyzer_kwargs   # mutable for runtime tuning

        self._analyzer   = ContactAngleAnalyzer(mode=mode, **analyzer_kwargs)
        self._result:    Optional[Dict[str, Any]] = None
        self._logging    = False
        self._csv_file   = None
        self._csv_writer = None
        self._waiting_for_baseline_click = False
        self._frame_count = 0
        self._save_count  = 0
        self._last_analyzed_frame: Optional[np.ndarray] = None
        self._last_overlay:        Optional[np.ndarray] = None

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open camera/video and start the live analysis loop."""
        # A string that is a digit (e.g. "0") should open a device, not a file.
        if isinstance(self.camera, int):
            cam_arg = int(self.camera)
        elif isinstance(self.camera, str) and self.camera.isdigit():
            cam_arg = int(self.camera)
        else:
            cam_arg = str(self.camera)
        cap = cv2.VideoCapture(cam_arg)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {self.camera}")

        win = "Contact Angle — Live  (press h for help)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, 600)
        cv2.setMouseCallback(win, self._on_mouse)

        print(f"\nLive analyzer started  [mode={self.mode}  camera={self.camera}]")
        print("Press 'h' in the window for controls.\n")

        min_interval = 1.0 / self._fps_limit if self._fps_limit else 0.0
        t_last = 0.0
        show_help = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Video ended — restart or stop
                    if isinstance(self.camera, (str, Path)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                t_now = time.perf_counter()
                if t_now - t_last < min_interval:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("h"):
                        show_help = not show_help
                    elif key in (ord("q"), 27):
                        break
                    else:
                        self._handle_key(key)
                    continue
                t_last = t_now

                # Analyse
                overlay = self._analyse_frame(frame)

                # Overlays: help, status bar
                self._draw_status(overlay)
                if show_help:
                    self._draw_help(overlay)
                if self._waiting_for_baseline_click:
                    self._draw_waiting(overlay)

                cv2.imshow(win, overlay)
                self._frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("h"):
                    show_help = not show_help
                elif key in (ord("q"), 27):
                    break
                else:
                    self._handle_key(key)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._close_csv()
            print(f"\nLive session ended: {self._frame_count} frames processed, "
                  f"{self._save_count} saved.")

    # ─── Frame processing ─────────────────────────────────────────────────────

    def _analyse_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run pipeline on one frame; return annotated BGR overlay."""
        self._last_analyzed_frame = frame.copy()   # copy — cap.read() may reuse buffer
        try:
            result = self._analyzer.analyze(
                frame,
                baseline_y=self.baseline_y,
            )
            self._result = result
            overlay = self._analyzer.get_overlay()
            self._last_overlay = overlay

            # Log to CSV
            if self._logging and self._csv_writer:
                row = {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame":     self._frame_count,
                    "theta_left":  result.get("theta_left"),
                    "theta_right": result.get("theta_right"),
                    "theta_mean":  result.get("theta_mean"),
                    "asymmetry":   result.get("asymmetry"),
                    "baseline_y":  result.get("baseline_y"),
                }
                self._csv_writer.writerow(row)
                self._csv_file.flush()

        except Exception as exc:
            overlay = frame.copy()
            cv2.putText(
                overlay, f"Error: {exc}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 1, cv2.LINE_AA,
            )

        return overlay

    # ─── Keyboard handling ────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key == ord("b"):
            self._waiting_for_baseline_click = True
            print("Click on the frame to set baseline.")

        elif key == ord("s"):
            self._save_frame()

        elif key == ord("l"):
            self._toggle_logging()

        elif key == ord("+") or key == ord("="):
            self._kw["canny_high"] = min(300, self._kw.get("canny_high", 100) + 5)
            self._rebuild_analyzer()
            print(f"canny_high → {self._kw['canny_high']}")

        elif key == ord("-"):
            self._kw["canny_high"] = max(20, self._kw.get("canny_high", 100) - 5)
            self._rebuild_analyzer()
            print(f"canny_high → {self._kw['canny_high']}")

        elif key == ord("]"):
            self._kw["window_px"] = min(120, self._kw.get("window_px", 25) + 5)
            self._rebuild_analyzer()
            print(f"window_px → {self._kw['window_px']}")

        elif key == ord("["):
            self._kw["window_px"] = max(5, self._kw.get("window_px", 25) - 5)
            self._rebuild_analyzer()
            print(f"window_px → {self._kw['window_px']}")

        elif key == ord("r"):
            self._kw = {}
            self.baseline_y = None
            self._rebuild_analyzer()
            print("Parameters reset to defaults.")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._waiting_for_baseline_click:
            self.baseline_y = y
            self._waiting_for_baseline_click = False
            print(f"Baseline set to y={y}")

    # ─── UI helpers ───────────────────────────────────────────────────────────

    def _draw_status(self, overlay: np.ndarray) -> None:
        """Draw angle readout and logging indicator at the bottom."""
        h, w = overlay.shape[:2]
        r = self._result

        if r is not None:
            th_l = r.get("theta_left")
            th_r = r.get("theta_right")
            th_m = r.get("theta_mean")

            parts = []
            if th_l is not None:
                parts.append(f"L:{th_l:.1f}")
            if th_r is not None:
                parts.append(f"R:{th_r:.1f}")
            if th_m is not None:
                parts.append(f"mean:{th_m:.1f}")
            text = "  ".join(parts) + "°" if parts else "No result"
        else:
            text = "No result"

        log_indicator = " [REC]" if self._logging else ""
        bl_str = f"  bl={self.baseline_y}" if self.baseline_y else "  bl=auto"

        cv2.putText(
            overlay,
            text + log_indicator + bl_str,
            (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 100) if self._logging else (200, 200, 200),
            1, cv2.LINE_AA,
        )

    def _draw_help(self, overlay: np.ndarray) -> None:
        """Draw semi-transparent help box."""
        h, w = overlay.shape[:2]
        panel_h = len(self._HELP) * 18 + 16
        panel_w = 230
        x0, y0 = w - panel_w - 10, 10
        sub = overlay[y0:y0 + panel_h, x0:x0 + panel_w]
        dark = (sub * 0.35).astype(np.uint8)
        overlay[y0:y0 + panel_h, x0:x0 + panel_w] = dark

        for i, line in enumerate(self._HELP):
            cv2.putText(
                overlay, line,
                (x0 + 6, y0 + 16 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

    def _draw_waiting(self, overlay: np.ndarray) -> None:
        h, w = overlay.shape[:2]
        cv2.putText(
            overlay, "Click to set baseline",
            (w // 2 - 90, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA,
        )

    # ─── CSV logging ──────────────────────────────────────────────────────────

    def _toggle_logging(self) -> None:
        if self._logging:
            self._close_csv()
            self._logging = False
            print("CSV logging stopped.")
        else:
            path = self.log_csv or Path(
                f"live_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            try:
                self._csv_file = open(path, "w", newline="")
                fields = ["timestamp", "frame", "theta_left", "theta_right",
                          "theta_mean", "asymmetry", "baseline_y"]
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=fields, extrasaction="ignore"
                )
                self._csv_writer.writeheader()
                self._logging = True
                print(f"CSV logging started → {path}")
            except Exception as exc:
                if self._csv_file:
                    self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
                print(f"Failed to start CSV logging: {exc}")

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ─── Saving ───────────────────────────────────────────────────────────────

    def _save_frame(self) -> None:
        """Save the last analyzed raw frame and its overlay as a matched pair."""
        if self._last_analyzed_frame is None:
            print("No analyzed frame available to save yet.")
            return
        self.save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        img_path = self.save_dir / f"frame_{ts}.png"
        cv2.imwrite(str(img_path), self._last_analyzed_frame)

        if self._last_overlay is not None:
            try:
                ov_path = self.save_dir / f"overlay_{ts}.png"
                cv2.imwrite(str(ov_path), self._last_overlay)
            except Exception:
                pass

        th = self._result.get("theta_mean") if self._result else None
        th_str = f" θ={th:.1f}°" if th is not None else ""
        print(f"Saved frame → {img_path}{th_str}")
        self._save_count += 1

    # ─── Rebuild analyzer after param changes ─────────────────────────────────

    def _rebuild_analyzer(self) -> None:
        self._analyzer = ContactAngleAnalyzer(mode=self.mode, **self._kw)
