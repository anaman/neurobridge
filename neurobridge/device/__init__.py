"""EEG sources: the live TH21A headband, a synthetic generator, quality checks."""

from .base import (  # noqa: F401
    EEGSource,
    LSLEEGSink,
    LSLMarkerSink,
    MarkerSink,
    MultiMarkerSink,
    Packet,
    SessionMarkerSink,
    TimingReport,
    clock,
)
from .quality import QualityReport, alpha_reactivity, assess, contact_from_selfcheck  # noqa: F401

__all__ = [
    "EEGSource", "Packet", "clock", "MarkerSink", "MultiMarkerSink",
    "LSLMarkerSink", "LSLEEGSink", "SessionMarkerSink", "TimingReport",
    "assess", "QualityReport", "alpha_reactivity", "contact_from_selfcheck",
    "open_source",
]


def open_source(name: str = "auto", **kwargs):
    """Factory: 'th21a' | 'fake' | 'auto' -> an :class:`EEGSource`.

    ``auto`` probes for the headband once and falls back to the synthetic
    source, so no command in the system can fail purely because the hardware
    is not on the desk.
    """
    name = (name or "auto").lower()
    if name == "auto":
        from ..config import resolve_device_name
        name = resolve_device_name("auto")
    if name in ("fake", "synthetic", "sim"):
        from .fake import FakeSource
        return FakeSource(**kwargs)
    if name in ("th21a", "ningnao", "serenibrain", "headband", "real"):
        from .th21a import TH21A
        if not kwargs.get("address"):
            # Honour `address:` / NB_ADDRESS here too. The runner passed it
            # explicitly but open_source() did not, so the bridge and the monitor
            # silently used the hardcoded default while the runner used the
            # configured one - two answers to the same question.
            from ..config import get_settings
            kwargs["address"] = get_settings().address
        return TH21A(**kwargs)
    raise ValueError(f"unknown device {name!r} (want auto|th21a|fake)")
