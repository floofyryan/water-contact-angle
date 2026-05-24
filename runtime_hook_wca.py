"""
PyInstaller runtime hook — runs before app.py starts.

Creates a virtual 'wca' package whose __path__ points at sys._MEIPASS so that
all the sibling analysis modules (analyzer.py, detection.py, …) can be imported
as 'wca.analyzer', 'wca.detection', etc.  This preserves their relative-import
chains (from .detection import …) which would otherwise break because app.py
runs as __main__ with no package context.
"""
import sys
import types

if hasattr(sys, "_MEIPASS"):
    _wca = types.ModuleType("wca")
    _wca.__path__ = [sys._MEIPASS]
    _wca.__package__ = "wca"
    _wca.__spec__ = None
    sys.modules["wca"] = _wca
