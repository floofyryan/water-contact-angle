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

from .analyzer            import ContactAngleAnalyzer, analyze_batch
from .calibration         import ScaleCalibration
from .evaporation         import EvaporationAnalyzer
from .advancing_receding  import AdvancingRecedingAnalyzer
from .live                import LiveAnalyzer
from .surface_energy      import SurfaceEnergyAnalyzer, LIQUIDS
from .baseline            import PolyBaseline, detect_baseline_poly
from .publication         import PublicationFigure
from .tuner               import ThresholdTuner

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
