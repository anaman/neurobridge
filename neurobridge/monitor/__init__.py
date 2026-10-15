"""Standalone monitor: record, display and replay NeuroBridge sessions.

The pieces are deliberately separate so each can be tested without a screen:

* :mod:`~neurobridge.monitor.sources` - where samples come from (headband, a
  running bridge, or a recording).
* :mod:`~neurobridge.monitor.recorder` - CSV + native session writing.
* :mod:`~neurobridge.monitor.app` - the tkinter display and its main loop.

``app.py`` is the only module that imports tkinter, so importing this package on a
headless machine stays safe.
"""
from __future__ import annotations

from .recorder import MonitorRecorder, RecorderReceipt
from .sources import (BridgeSource, Chunk, DeviceSource, MonitorSource, Reading,
                      ReplaySource, Snapshot, open_source_for_mode)

__all__ = [
    "MonitorRecorder", "RecorderReceipt",
    "BridgeSource", "Chunk", "DeviceSource", "MonitorSource", "Reading",
    "ReplaySource", "Snapshot", "open_source_for_mode",
]
