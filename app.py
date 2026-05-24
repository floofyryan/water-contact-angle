"""
app.py — Main GUI application for the Water Contact Angle Analyzer.

Provides a tkinter-based launcher window that ties together all analysis
modes. Designed to be packaged into a standalone .exe via PyInstaller.

Modes available from the launcher:
  1. Single image   — analyze one image file
  2. Batch images   — analyze a folder of images, export CSV
  3. Evaporation    — time-lapse video analysis (EvaporationAnalyzer)
  4. Adv/Rec        — advancing/receding video analysis
  5. Live camera    — real-time camera feed (LiveAnalyzer)
  6. Tuner          — interactive parameter tuning GUI (ThresholdTuner)
  7. Surface energy — multi-liquid surface energy calculation

Run directly:
    python app.py
    python -m contact_angle          # if installed as a package

Build to .exe (from this directory):
    pip install pyinstaller
    pyinstaller --onefile --windowed app.py --name "ContactAngleAnalyzer"
"""

from __future__ import annotations

import csv
import ctypes
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Optional

import cv2
import numpy as np


# ── Windows DPI awareness — must run before any tkinter window is created ──────
# Without this, on 125 %/150 % scaled displays tkinter Toplevel windows appear
# as tiny grey boxes because Tk renders at 96 DPI regardless of screen scale.
def _set_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # fallback (Win Vista+)
        except Exception:
            pass

_set_dpi_aware()


# ── Frozen-exe: ensure MEIPASS is on sys.path ────────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    if sys._MEIPASS not in sys.path:
        sys.path.insert(0, sys._MEIPASS)


# ── Portable sibling-module importer ─────────────────────────────────────────
def _pkg_import(module: str, *names):
    """
    Import names from a sibling analysis module. Three contexts:
      1. Package (python -m water-contact-angle)  → relative import via __package__
      2. Frozen exe (PyInstaller)                 → import as 'wca.module'
         runtime_hook_wca.py pre-creates the 'wca' virtual package so that
         relative imports inside the sibling modules keep working.
      3. Plain script (python app.py)             → absolute import
    """
    import importlib
    pkg = __package__
    if pkg:
        mod = importlib.import_module(f".{module}", package=pkg)
    elif getattr(sys, "frozen", False):
        mod = importlib.import_module(f"wca.{module}")
    else:
        mod = importlib.import_module(module)
    if len(names) == 1:
        return getattr(mod, names[0])
    return tuple(getattr(mod, n) for n in names)


# ── Lazy imports of heavy modules (keeps startup fast) ────────────────────────

def _get_analyzer():
    return _pkg_import("analyzer", "ContactAngleAnalyzer")

def _get_evaporation():
    return _pkg_import("evaporation", "EvaporationAnalyzer")

def _get_ar():
    return _pkg_import("advancing_receding", "AdvancingRecedingAnalyzer")

def _get_live():
    return _pkg_import("live", "LiveAnalyzer")

def _get_tuner():
    return _pkg_import("tuner", "ThresholdTuner")

def _get_sea():
    return _pkg_import("surface_energy", "SurfaceEnergyAnalyzer", "LIQUIDS")


# ── Drag-and-drop via Windows API (ctypes only, no third-party library) ───────
def _setup_drop(widget, callback):
    """
    Register a tkinter widget as a Windows drag-and-drop target.
    Calls callback([path_str, ...]) on the main thread when files are dropped.
    Uses only ctypes — works in any frozen exe without extra packages.
    """
    try:
        import ctypes
        import ctypes.wintypes as wt

        LONG_PTR = ctypes.c_ssize_t   # 64-bit signed pointer on 64-bit Windows

        shell32 = ctypes.windll.shell32
        user32  = ctypes.windll.user32

        # Must set restype/argtypes before any call — ctypes defaults to c_int
        # (32-bit) which silently truncates 64-bit pointers and causes crashes.
        user32.SetWindowLongPtrW.restype  = LONG_PTR
        user32.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, LONG_PTR]
        user32.CallWindowProcW.restype    = LONG_PTR
        user32.CallWindowProcW.argtypes   = [LONG_PTR, wt.HWND, wt.UINT,
                                              wt.WPARAM, wt.LPARAM]
        user32.DefWindowProcW.restype     = LONG_PTR

        widget.update_idletasks()          # ensure HWND exists
        hwnd = widget.winfo_id()
        if not hwnd:
            return
        shell32.DragAcceptFiles(hwnd, True)

        WM_DROPFILES = 0x0233
        WNDPROCTYPE  = ctypes.WINFUNCTYPE(
            LONG_PTR, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM
        )

        _old = [LONG_PTR(0)]

        def _wndproc(h, msg, wp, lp):
            try:
                if msg == WM_DROPFILES:
                    n = shell32.DragQueryFileW(wp, 0xFFFFFFFF, None, 0)
                    files = []
                    for i in range(n):
                        sz  = shell32.DragQueryFileW(wp, i, None, 0) + 1
                        buf = ctypes.create_unicode_buffer(sz)
                        shell32.DragQueryFileW(wp, i, buf, sz)
                        files.append(buf.value)
                    shell32.DragFinish(wp)
                    widget.after(0, lambda f=files: callback(f))
                    return 0
                if _old[0]:
                    return user32.CallWindowProcW(_old[0], h, msg, wp, lp)
                return user32.DefWindowProcW(h, msg, wp, lp)
            except Exception:
                return user32.DefWindowProcW(h, msg, wp, lp)

        proc     = WNDPROCTYPE(_wndproc)
        _old[0]  = user32.SetWindowLongPtrW(hwnd, -4, proc)
        widget._dnd_proc = proc   # prevent garbage collection
        widget._dnd_old  = _old
    except Exception:
        pass


# ── Shared parameter panel ────────────────────────────────────────────────────

class ParamPanel(ttk.LabelFrame):
    """
    Reusable panel of analysis parameters shown in each analysis dialog.
    Mirrors the parameters of ContactAngleAnalyzer.

    Uses ttk.LabelFrame (not tk.LabelFrame) because only ttk supports the
    'padding' option; tk.LabelFrame silently raises TclError for it.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, text="Analysis parameters", **kw)
        self._vars: dict = {}
        self._build()

    def _build(self):
        defs = [
            ("Mode",           "mode",       "sessile",  "combo", ["sessile", "captive_bubble"]),
            ("Canny low",      "canny_low",  30,         "int",   None),
            ("Canny high",     "canny_high", 100,        "int",   None),
            ("CLAHE clip",     "clahe_clip", 2.0,        "float", None),
            ("Blur ksize",     "blur_ksize", 5,          "int",   None),
            ("Window px",      "window_px",  25,         "float", None),
            ("Baseline margin","baseline_margin", 5,     "int",   None),
            ("Baseline y",     "baseline_y", "",         "int_opt", None),
        ]
        for row, (label, key, default, typ, choices) in enumerate(defs):
            tk.Label(self, text=label + ":").grid(row=row, column=0, sticky="e",
                                                   padx=4, pady=2)
            if typ == "combo":
                var = tk.StringVar(value=default)
                w   = ttk.Combobox(self, textvariable=var, values=choices,
                                   width=18, state="readonly")
            else:
                var = tk.StringVar(value=str(default))
                w   = tk.Entry(self, textvariable=var, width=20)
            w.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            self._vars[key] = (var, typ)

    def get(self) -> dict:
        """Return current parameters as a dict suitable for ContactAngleAnalyzer."""
        out = {}
        for key, (var, typ) in self._vars.items():
            raw = var.get().strip()
            if typ == "int":
                v = int(raw) if raw else 0
                # blur_ksize must be odd for GaussianBlur; correct silently
                if key == "blur_ksize" and v > 0 and v % 2 == 0:
                    v += 1
                out[key] = v
            elif typ == "float":
                out[key] = float(raw) if raw else 0.0
            elif typ == "int_opt":
                out[key] = int(raw) if raw else None
            else:
                out[key] = raw
        return out


# ── Base Toplevel with visible error reporting ────────────────────────────────

class _SafeWindow(tk.Toplevel):
    """
    Toplevel base class that catches exceptions from _build() and displays
    them as visible red text rather than producing a silent grey screen.
    """
    def _safe_build(self) -> None:
        try:
            self._build()
        except Exception:
            import traceback
            msg = traceback.format_exc()
            tk.Label(
                self,
                text="Window failed to build — see error below:\n\n" + msg,
                fg="red", justify="left", wraplength=560,
                font=("Courier", 8),
            ).pack(padx=10, pady=10)


# ── Single image window ───────────────────────────────────────────────────────

class SingleImageWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Single Image Analysis")
        self.resizable(True, True)
        self._path: Optional[Path] = None
        self._safe_build()
        self.minsize(400, 300)
        self.lift()
        self.focus_force()

    def _build(self):
        self._params = ParamPanel(self, padding=6)
        self._params.pack(fill="x", padx=10, pady=5)

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Button(frm, text="Browse image…", command=self._browse).pack(side="left")
        self._path_lbl = tk.Label(frm, text="No file selected  (or drag & drop)", fg="grey")
        self._path_lbl.pack(side="left", padx=8)

        # Drag-and-drop registered after build completes
        self.after(200, lambda: _setup_drop(self, self._on_drop))

        btn_frm = tk.Frame(self)
        btn_frm.pack(pady=8)
        tk.Button(btn_frm, text="Analyse", width=14,
                  command=self._run).pack(side="left", padx=4)
        tk.Button(btn_frm, text="Save overlay…", width=14,
                  command=self._save).pack(side="left", padx=4)

        self._result_lbl = tk.Label(self, text="", font=("Courier", 11),
                                    justify="left")
        self._result_lbl.pack(padx=10, pady=4)
        self._ca = None

    def _on_drop(self, files) -> None:
        """Handle files dropped onto the window."""
        if not files:
            return
        self._path = Path(files[0])
        self._path_lbl.config(text=str(self._path), fg="black")

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")]
        )
        if p:
            self._path = Path(p)
            self._path_lbl.config(text=p, fg="black")

    def _run(self):
        if self._path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        p = self._params.get()
        try:
            CA = _get_analyzer()
            self._ca = CA(
                mode=p["mode"],
                canny_low=p["canny_low"],
                canny_high=p["canny_high"],
                clahe_clip=p["clahe_clip"],
                blur_ksize=p["blur_ksize"],
                window_px=p["window_px"],
                baseline_margin=p["baseline_margin"],
            )
            r = self._ca.analyze(self._path, baseline_y=p["baseline_y"])

            lines = []
            if r.get("theta_left")  is not None: lines.append(f"Left:  {r['theta_left']:.2f}°")
            if r.get("theta_right") is not None: lines.append(f"Right: {r['theta_right']:.2f}°")
            if r.get("theta_mean")  is not None: lines.append(f"Mean:  {r['theta_mean']:.2f}°")
            if r.get("asymmetry")   is not None: lines.append(f"|L-R|: {r['asymmetry']:.2f}°")
            if r.get("r2_left")     is not None: lines.append(f"R²_L:  {r['r2_left']:.4f}")
            if r.get("r2_right")    is not None: lines.append(f"R²_R:  {r['r2_right']:.4f}")
            lines.append(f"Baseline y: {r.get('baseline_y')}")
            self._result_lbl.config(text="\n".join(lines))
            # Bug 6 fix: cv2.waitKey(0) would block tkinter's main thread.
            # Run the OpenCV display window in a daemon thread instead.
            ca_ref = self._ca
            threading.Thread(
                target=lambda: ca_ref.show("Analysis result  (press any key to close)"),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _save(self):
        if self._ca is None:
            messagebox.showwarning("No result", "Run analysis first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("All", "*.*")],
        )
        if p:
            self._ca.save_overlay(p)
            messagebox.showinfo("Saved", f"Overlay saved to:\n{p}")


# ── Batch analysis window ─────────────────────────────────────────────────────

class BatchWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Batch Image Analysis")
        self._folder: Optional[Path] = None
        self._safe_build()
        self.minsize(400, 300)
        self.lift()
        self.focus_force()

    def _build(self):
        self._params = ParamPanel(self, padding=6)
        self._params.pack(fill="x", padx=10, pady=5)

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Button(frm, text="Select folder…", command=self._browse_folder).pack(side="left")
        self._folder_lbl = tk.Label(frm, text="No folder selected", fg="grey")
        self._folder_lbl.pack(side="left", padx=8)

        tk.Button(self, text="Run batch & export CSV", width=26,
                  command=self._run_batch).pack(pady=8)

        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.pack(fill="x", padx=10)
        self._status = tk.Label(self, text="")
        self._status.pack()

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select image folder")
        if d:
            self._folder = Path(d)
            self._folder_lbl.config(text=d, fg="black")

    def _run_batch(self):
        if self._folder is None:
            messagebox.showwarning("No folder", "Select a folder first.")
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        paths = sorted(f for f in self._folder.iterdir() if f.suffix.lower() in exts)
        if not paths:
            messagebox.showinfo("Empty", "No image files found in that folder.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="batch_results.csv",
        )
        if not out_path:
            return

        p = self._params.get()
        self._progress["maximum"] = len(paths)
        self._progress["value"]   = 0

        def _worker():
            ContactAngleAnalyzer = _pkg_import("analyzer", "ContactAngleAnalyzer")
            ca = ContactAngleAnalyzer(
                mode=p["mode"],
                canny_low=p["canny_low"],
                canny_high=p["canny_high"],
                clahe_clip=p["clahe_clip"],
                blur_ksize=p["blur_ksize"],
                window_px=p["window_px"],
                baseline_margin=p["baseline_margin"],
            )
            rows = []
            for i, img_path in enumerate(paths):
                try:
                    r = ca.analyze(img_path, baseline_y=p["baseline_y"])
                except Exception as exc:
                    r = {"error": str(exc)}
                r["filename"] = img_path.name
                rows.append(r)
                # Bug 1 fix: marshal tkinter updates back to the main thread.
                _i, _name = i, img_path.name
                self.after(0, lambda v=_i+1, n=_name: (
                    self._progress.__setitem__("value", v),
                    self._status.config(text=f"{v}/{len(paths)}: {n}"),
                ))

            fields = ["filename", "theta_left", "theta_right", "theta_mean",
                      "asymmetry", "r2_left", "r2_right", "baseline_y", "error"]
            with open(out_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            msg = f"Done — {len(rows)} images → {out_path}"
            self.after(0, lambda: self._status.config(text=msg))
            self.after(0, lambda: messagebox.showinfo(
                "Batch complete",
                f"Processed {len(rows)} images.\nResults saved to:\n{out_path}"
            ))

        threading.Thread(target=_worker, daemon=True).start()


# ── Evaporation window ────────────────────────────────────────────────────────

class EvaporationWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Evaporation Video Analysis")
        self._video: Optional[Path] = None
        self._safe_build()
        self.minsize(400, 300)
        self.lift()
        self.focus_force()

    def _build(self):
        self._params = ParamPanel(self, padding=6)
        self._params.pack(fill="x", padx=10, pady=5)

        # Extra evaporation-specific fields
        xfrm = tk.LabelFrame(self, text="Evaporation options", padding=6)
        xfrm.pack(fill="x", padx=10, pady=5)
        rows = [
            ("Frame step",      "frame_step",      "1"),
            ("Smooth window",   "smooth_window",   "7"),
            ("px/mm (0=none)",  "pixels_per_mm",   "0"),
        ]
        self._extra_vars = {}
        for r, (lbl, key, dflt) in enumerate(rows):
            tk.Label(xfrm, text=lbl + ":").grid(row=r, column=0, sticky="e", padx=4)
            var = tk.StringVar(value=dflt)
            tk.Entry(xfrm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=4)
            self._extra_vars[key] = var

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Button(frm, text="Select video…", command=self._browse).pack(side="left")
        self._vid_lbl = tk.Label(frm, text="No file selected", fg="grey")
        self._vid_lbl.pack(side="left", padx=8)

        tk.Button(self, text="Analyse video", width=20, command=self._run).pack(pady=8)
        self._status = tk.Label(self, text="")
        self._status.pack()

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")],
        )
        if p:
            self._video = Path(p)
            self._vid_lbl.config(text=p, fg="black")

    def _run(self):
        if self._video is None:
            messagebox.showwarning("No video", "Select a video file first.")
            return
        p    = self._params.get()
        ex   = {k: v.get().strip() for k, v in self._extra_vars.items()}
        ppmm = float(ex["pixels_per_mm"]) if ex["pixels_per_mm"] else 0

        def _worker():
            # Bug 2 fix: all tkinter widget mutations use self.after() from this thread.
            try:
                EvaporationAnalyzer = _pkg_import("evaporation", "EvaporationAnalyzer")
                cal = None
                if ppmm > 0:
                    ScaleCalibration = _pkg_import("calibration", "ScaleCalibration")
                    cal = ScaleCalibration(ppmm)
                ea = EvaporationAnalyzer(
                    mode=p["mode"],
                    calibration=cal,
                    baseline_y=p["baseline_y"],
                    frame_step=int(ex["frame_step"]),
                    smooth_window=int(ex["smooth_window"]),
                    canny_low=p["canny_low"],
                    canny_high=p["canny_high"],
                    clahe_clip=p["clahe_clip"],
                    blur_ksize=p["blur_ksize"],
                    window_px=p["window_px"],
                    baseline_margin=p["baseline_margin"],
                )
                self.after(0, lambda: self._status.config(text="Analysing…"))
                ea.analyze_video(self._video)
                self.after(0, lambda: self._status.config(
                    text="Analysis complete — saving outputs…"))

                stem = self._video.stem
                out_dir = self._video.parent
                ea.export_csv(out_dir / f"{stem}_results.csv")
                print(ea.regime_summary())
                ea.plot_time_series(save_path=str(out_dir / f"{stem}_plot.png"))
                _msg = f"Done. Files saved to {out_dir}"
                self.after(0, lambda m=_msg: self._status.config(text=m))
            except Exception as exc:
                _e = str(exc)
                self.after(0, lambda e=_e: self._status.config(text=f"Error: {e}"))
                self.after(0, lambda e=_e: messagebox.showerror("Error", e))

        threading.Thread(target=_worker, daemon=True).start()


# ── Advancing/receding window ─────────────────────────────────────────────────

class AdvRecWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Advancing/Receding Contact Angle")
        self._video: Optional[Path] = None
        self._safe_build()
        self.minsize(400, 300)
        self.lift()
        self.focus_force()

    def _build(self):
        self._params = ParamPanel(self, padding=6)
        self._params.pack(fill="x", padx=10, pady=5)

        xfrm = tk.LabelFrame(self, text="Options", padding=6)
        xfrm.pack(fill="x", padx=10, pady=5)
        rows = [
            ("Frame step",       "frame_step",      "1"),
            ("Smooth window",    "smooth_window",   "9"),
            ("Radius tol (px/f)","radius_tol_px",   "1.5"),
        ]
        self._extra_vars = {}
        for r, (lbl, key, dflt) in enumerate(rows):
            tk.Label(xfrm, text=lbl + ":").grid(row=r, column=0, sticky="e", padx=4)
            var = tk.StringVar(value=dflt)
            tk.Entry(xfrm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=4)
            self._extra_vars[key] = var

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Button(frm, text="Select video…", command=self._browse).pack(side="left")
        self._vid_lbl = tk.Label(frm, text="No file selected", fg="grey")
        self._vid_lbl.pack(side="left", padx=8)

        tk.Button(self, text="Analyse", width=16, command=self._run).pack(pady=8)
        self._result_lbl = tk.Label(self, text="", font=("Courier", 11), justify="left")
        self._result_lbl.pack(padx=10, pady=4)

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if p:
            self._video = Path(p)
            self._vid_lbl.config(text=p, fg="black")

    def _run(self):
        if self._video is None:
            messagebox.showwarning("No video", "Select a video file first.")
            return
        p  = self._params.get()
        ex = {k: v.get().strip() for k, v in self._extra_vars.items()}

        def _worker():
            # Bug 3 fix: all tkinter widget mutations use self.after() from this thread.
            try:
                AdvancingRecedingAnalyzer = _pkg_import("advancing_receding", "AdvancingRecedingAnalyzer")
                ar = AdvancingRecedingAnalyzer(
                    mode=p["mode"],
                    baseline_y=p["baseline_y"],
                    frame_step=int(ex["frame_step"]),
                    smooth_window=int(ex["smooth_window"]),
                    radius_tol_px=float(ex["radius_tol_px"]),
                    canny_low=p["canny_low"],
                    canny_high=p["canny_high"],
                    clahe_clip=p["clahe_clip"],
                    blur_ksize=p["blur_ksize"],
                    window_px=p["window_px"],
                    baseline_margin=p["baseline_margin"],
                )
                ar.analyze_video(self._video)
                _summary = ar.summary()
                self.after(0, lambda s=_summary: self._result_lbl.config(text=s))
                out_dir = self._video.parent
                stem    = self._video.stem
                ar.export_csv(out_dir / f"{stem}_advrec.csv")
                ar.plot(save_path=str(out_dir / f"{stem}_advrec_plot.png"))
            except Exception as exc:
                _e = str(exc)
                self.after(0, lambda e=_e: messagebox.showerror("Error", e))

        threading.Thread(target=_worker, daemon=True).start()


# ── Live camera launcher ──────────────────────────────────────────────────────

class LiveWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Live Camera Analysis")
        self._safe_build()
        self.minsize(400, 280)
        self.lift()
        self.focus_force()

    def _build(self):
        self._params = ParamPanel(self, padding=6)
        self._params.pack(fill="x", padx=10, pady=5)

        frm = ttk.LabelFrame(self, text="Camera", padding=6)
        frm.pack(fill="x", padx=10, pady=5)
        tk.Label(frm, text="Camera / video path:").grid(row=0, column=0, sticky="e")
        self._cam_var = tk.StringVar(value="0")
        tk.Entry(frm, textvariable=self._cam_var, width=30).grid(row=0, column=1)
        tk.Button(frm, text="Browse…", command=self._browse_vid).grid(row=0, column=2, padx=4)

        tk.Button(self, text="Start live analysis", width=20,
                  command=self._start).pack(pady=10)
        tk.Label(self, text="(Controls shown in the OpenCV window — press h for help)",
                 fg="grey").pack()

    def _browse_vid(self):
        p = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if p:
            self._cam_var.set(p)

    def _start(self):
        p   = self._params.get()
        cam = self._cam_var.get().strip()

        def _worker():
            # Bug 4 fix: messagebox must be called on the main thread.
            # la.run() itself (OpenCV HighGUI) is fine on a background thread on Linux/Win.
            try:
                LiveAnalyzer = _pkg_import("live", "LiveAnalyzer")
                la = LiveAnalyzer(
                    mode=p["mode"],
                    camera=cam,
                    baseline_y=p["baseline_y"],
                    canny_low=p["canny_low"],
                    canny_high=p["canny_high"],
                    clahe_clip=p["clahe_clip"],
                    blur_ksize=p["blur_ksize"],
                    window_px=p["window_px"],
                    baseline_margin=p["baseline_margin"],
                )
                la.run()
            except Exception as exc:
                _e = str(exc)
                self.after(0, lambda e=_e: messagebox.showerror("Error", e))

        threading.Thread(target=_worker, daemon=True).start()


# ── Tuner launcher ────────────────────────────────────────────────────────────

class TunerWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Parameter Tuner")
        self._path: Optional[Path] = None
        self._safe_build()
        self.minsize(350, 200)
        self.lift()
        self.focus_force()

    def _build(self):
        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=10)
        tk.Label(frm, text="Mode:").pack(side="left")
        self._mode_var = tk.StringVar(value="sessile")
        ttk.Combobox(frm, textvariable=self._mode_var,
                     values=["sessile", "captive_bubble"],
                     state="readonly", width=16).pack(side="left", padx=6)

        frm2 = tk.Frame(self)
        frm2.pack(fill="x", padx=10)
        tk.Button(frm2, text="Select image…", command=self._browse).pack(side="left")
        self._lbl = tk.Label(frm2, text="No file selected", fg="grey")
        self._lbl.pack(side="left", padx=8)

        tk.Button(self, text="Open Tuner", width=16, command=self._open).pack(pady=10)

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")]
        )
        if p:
            self._path = Path(p)
            self._lbl.config(text=p, fg="black")

    def _open(self):
        if self._path is None:
            messagebox.showwarning("No image", "Select an image first.")
            return

        def _worker():
            # Bug 5 fix: messagebox must be called on the main thread.
            # t.show() (matplotlib GUI) is acceptable on a background thread on Linux/Win.
            try:
                ThresholdTuner = _pkg_import("tuner", "ThresholdTuner")
                t = ThresholdTuner(self._path, mode=self._mode_var.get())
                t.show()
            except Exception as exc:
                _e = str(exc)
                self.after(0, lambda e=_e: messagebox.showerror("Error", e))

        threading.Thread(target=_worker, daemon=True).start()


# ── Surface energy window ─────────────────────────────────────────────────────

class SurfaceEnergyWindow(_SafeWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Surface Energy Calculator")
        self._rows: list = []
        self._safe_build()
        self.minsize(400, 350)
        self.lift()
        self.focus_force()

    def _build(self):
        try:
            SEA, LIQUIDS = _get_sea()
        except Exception as exc:
            tk.Label(self, text=f"Could not load surface energy module:\n{exc}",
                     fg="red", justify="left", wraplength=380).pack(padx=12, pady=12)
            return
        self._LIQUIDS = LIQUIDS
        self._SEA_cls = SEA

        tk.Label(self, text="Add contact angle measurements:", font=("", 10, "bold")).pack(pady=6)

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10)
        tk.Label(frm, text="Liquid:").grid(row=0, column=0, sticky="e")
        self._liquid_var = tk.StringVar(value="water")
        ttk.Combobox(frm, textvariable=self._liquid_var,
                     values=list(LIQUIDS.keys()), state="readonly",
                     width=18).grid(row=0, column=1, sticky="w", padx=4)

        tk.Label(frm, text="θ (°):").grid(row=0, column=2, sticky="e")
        self._theta_var = tk.StringVar()
        tk.Entry(frm, textvariable=self._theta_var, width=8).grid(row=0, column=3)
        tk.Button(frm, text="Add", command=self._add_row).grid(row=0, column=4, padx=4)

        # Measurement list
        self._list_frame = tk.Frame(self)
        self._list_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(self._list_frame, text="Liquid            θ (°)\n" + "─"*32,
                 font=("Courier", 9), justify="left").pack(anchor="w")
        self._list_lbl = tk.Label(self._list_frame, text="",
                                   font=("Courier", 10), justify="left")
        self._list_lbl.pack(anchor="w")

        btn_frm = tk.Frame(self)
        btn_frm.pack(pady=6)
        tk.Button(btn_frm, text="Calculate", width=12, command=self._calc).pack(side="left", padx=4)
        tk.Button(btn_frm, text="Clear all", width=10, command=self._clear).pack(side="left", padx=4)

        self._result_lbl = tk.Label(self, text="", font=("Courier", 10), justify="left")
        self._result_lbl.pack(padx=10, pady=6)

    def _add_row(self):
        liq = self._liquid_var.get()
        raw = self._theta_var.get().strip()
        try:
            theta = float(raw)
        except ValueError:
            messagebox.showerror("Bad input", "Enter a valid angle in degrees.")
            return
        self._rows.append((liq, theta))
        self._refresh_list()

    def _refresh_list(self):
        lines = [f"{liq:<18} {th:.2f}" for liq, th in self._rows]
        self._list_lbl.config(text="\n".join(lines) if lines else "(none)")

    def _clear(self):
        self._rows.clear()
        self._refresh_list()
        self._result_lbl.config(text="")

    def _calc(self):
        if len(self._rows) < 2:
            messagebox.showwarning("Too few", "Add at least 2 measurements.")
            return
        sea = self._SEA_cls()
        for liq, theta in self._rows:
            sea.add_measurement(self._LIQUIDS[liq], theta)

        lines = []
        for fn_name, fn in [("Owens-Wendt", sea.owens_wendt), ("Wu", sea.wu)]:
            try:
                r = fn()
                lines.append(f"── {fn_name} ──")
                lines.append(f"  γ_S  = {r['gamma_s_total']:.2f} mN/m")
                lines.append(f"  γ_S^d = {r['gamma_s_d']:.2f}   γ_S^p = {r['gamma_s_p']:.2f}")
                lines.append(f"  R²   = {r['r_squared']:.4f}")
            except Exception as exc:
                lines.append(f"  {fn_name}: {exc}")

        self._result_lbl.config(text="\n".join(lines))


# ── Main launcher ─────────────────────────────────────────────────────────────

class LauncherApp(tk.Tk):
    """Main launcher window."""

    def __init__(self):
        super().__init__()
        self.title("Water Contact Angle Analyzer")
        self.resizable(False, False)
        self._build()

    def _build(self):
        tk.Label(self,
                 text="Water Contact Angle Analyzer",
                 font=("", 15, "bold")).pack(pady=(18, 4))
        tk.Label(self,
                 text="Select an analysis mode:",
                 font=("", 10)).pack(pady=(0, 12))

        modes = [
            ("Single image",          SingleImageWindow, "#2196f3"),
            ("Batch images → CSV",    BatchWindow,       "#4caf50"),
            ("Evaporation video",     EvaporationWindow, "#ff9800"),
            ("Advancing / Receding",  AdvRecWindow,      "#9c27b0"),
            ("Live camera",           LiveWindow,        "#f44336"),
            ("Parameter tuner",       TunerWindow,       "#607d8b"),
            ("Surface energy",        SurfaceEnergyWindow, "#795548"),
        ]

        for label, cls, color in modes:
            btn = tk.Button(
                self, text=label,
                width=26, height=2,
                font=("", 10),
                bg=color, fg="white",
                activebackground=color, activeforeground="white",
                relief="flat", cursor="hand2",
                command=lambda c=cls: c(self),
            )
            btn.pack(pady=3, padx=30)

        tk.Label(self, text="", height=1).pack()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        # Running inside a PyInstaller bundle — imports are already resolved.
        main()
    else:
        # Running as a plain script — add parent to sys.path so relative imports work.
        _here = Path(__file__).resolve().parent
        _parent = _here.parent
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))

        _pkg = _here.name
        import importlib
        mod = importlib.import_module(f"{_pkg}.app")
        mod.main()
