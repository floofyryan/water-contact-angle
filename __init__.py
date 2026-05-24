"""
Water Contact Angle Analyzer
============================
A complete Python package for measuring contact angles from optical images.

Supports:
  • Sessile drop (single image, batch, time-lapse evaporation)
  • Captive bubble
  • Advancing / receding (hysteresis)
  • Live camera feed
  • Tilt-corrected baselines
  • Surface energy (Owens-Wendt, Wu)
  • Publication-quality figure export

Quick start
-----------
    from contact_angle import ContactAngleAnalyzer

    ca = ContactAngleAnalyzer(mode='sessile')
    result = ca.analyze('drop.png')
    print(result['theta_mean'])
    ca.show()
"""

from .analyzer    import ContactAngleAnalyzer, analyze_batch
from .calibration import ScaleCalibration

# Heavy optional submodules — wrapped so a missing optional dependency
# (e.g. matplotlib not installed) doesn't prevent `import contact_angle`.
try:
    from .evaporation        import EvaporationAnalyzer
except Exception:
    pass

try:
    from .advancing_receding import AdvancingRecedingAnalyzer
except Exception:
    pass

try:
    from .live               import LiveAnalyzer
except Exception:
    pass

try:
    from .surface_energy     import SurfaceEnergyAnalyzer, LIQUIDS
except Exception:
    pass

try:
    from .baseline           import PolyBaseline, detect_baseline_poly
except Exception:
    pass

try:
    from .publication        import PublicationFigure
except Exception:
    pass

try:
    from .tuner              import ThresholdTuner
except Exception:
    pass

__version__ = "1.0.0"
__all__ = [
    "ContactAngleAnalyzer",
    "analyze_batch",
    "ScaleCalibration",
    "EvaporationAnalyzer",
    "AdvancingRecedingAnalyzer",
    "LiveAnalyzer",
    "SurfaceEnergyAnalyzer",
    "LIQUIDS",
    "PolyBaseline",
    "detect_baseline_poly",
    "PublicationFigure",
    "ThresholdTuner",
]
