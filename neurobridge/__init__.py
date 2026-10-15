"""NeuroBridge - brain-translation training system for single-channel frontal EEG.

Target hardware: NingNao / SereniBrain TH21A BLE headband
  single dry frontal channel, 250 Hz, 24-bit signed ADC counts.

Pipeline:  acquire -> mark -> present -> preprocess -> features -> decode -> agent
"""

__version__ = "0.1.0"

from .config import ROOT, DATA_DIR, SESSION_DIR, get_settings  # noqa: F401

__all__ = ["__version__", "ROOT", "DATA_DIR", "SESSION_DIR", "get_settings"]
