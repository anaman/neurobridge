"""EEG sources and marker sinks.

The rest of NeuroBridge is synchronous on purpose: presentation (pygame,
sounddevice) is synchronous, and threading only the BLE transport keeps the
timing model simple and testable. An ``EEGSource`` therefore exposes a
non-blocking ``poll()`` that returns whatever packets have arrived.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class Packet:
    """One BLE notification worth of samples.

    ``t_mono`` is the arrival timestamp of the *last* sample in the packet
    (that is the only sample we can actually observe arriving). Per-sample
    times are estimated by walking backwards at the nominal rate.
    """

    values: np.ndarray                  # raw ADC counts
    t_mono: float                       # monotonic arrival time (perf_counter)
    indices: np.ndarray | None = None   # device sample counter (8-bit, wraps)
    t_samples: np.ndarray | None = None # estimated per-sample monotonic times

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        n = self.values.size
        if self.t_samples is None:
            fs = 250.0
            offs = (np.arange(n) - (n - 1)) / fs
            self.t_samples = self.t_mono + offs

    @property
    def n(self) -> int:
        return int(self.values.size)


class EEGSource(abc.ABC):
    """A stream of EEG packets at a nominal sampling rate."""

    name: str = "abstract"
    fs: float = 250.0

    #: some sources can report electrode contact quality (device self-check)
    contact: str = "unknown"

    def open(self) -> None:
        """Connect / start streaming. Must be idempotent-safe to call once."""

    def close(self) -> None:
        """Stop streaming and release the transport."""

    @abc.abstractmethod
    def poll(self, timeout_s: float = 0.0) -> list[Packet]:
        """Return packets received since the last call (possibly empty)."""

    def notify_marker(self, code: int, name: str, t_mono: float,
                      meta: dict[str, Any] | None = None) -> None:
        """Called by the runner for every marker, *before* presentation.

        Real hardware ignores this. The synthetic source uses it to inject
        stimulus-locked responses, which is what makes `nb simulate` a genuine
        end-to-end test of the whole pipeline.
        """

    @property
    def info(self) -> dict[str, Any]:
        return {"source": self.name, "fs": self.fs}

    def stats(self) -> dict[str, Any]:
        return {}

    # -- helpers ----------------------------------------------------
    def __enter__(self) -> "EEGSource":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class MarkerSink:
    """Receives marker events. Compose several with :class:`MultiMarkerSink`."""

    def send(self, code: int, name: str, t_mono: float,
             meta: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MultiMarkerSink(MarkerSink):
    def __init__(self, sinks: Iterable[MarkerSink] = ()) -> None:
        self.sinks = [s for s in sinks if s is not None]

    def add(self, sink: MarkerSink | None) -> None:
        if sink is not None:
            self.sinks.append(sink)

    def send(self, code: int, name: str, t_mono: float,
             meta: dict[str, Any] | None = None) -> None:
        for s in self.sinks:
            try:
                s.send(code, name, t_mono, meta)
            except Exception:
                pass          # a broken sink must never break a session

    def close(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass


class LSLMarkerSink(MarkerSink):
    """Publish markers on the lab streaming layer as a string stream."""

    def __init__(self, name: str = "NeuroBridgeMarkers", source_id: str = "nb_markers") -> None:
        from pylsl import StreamInfo, StreamOutlet
        info = StreamInfo(name, "Markers", 1, 0, "string", source_id)
        desc = info.desc()
        desc.append_child_value("manufacturer", "NeuroBridge")
        self._outlet = StreamOutlet(info)

    def send(self, code: int, name: str, t_mono: float,
             meta: dict[str, Any] | None = None) -> None:
        self._outlet.push_sample([f"{name}:{int(code)}"])


class LSLEEGSink:
    """Publish acquired samples as an LSL EEG stream (one channel).

    BLE packets arrive in bursts, so samples are pushed at packet arrival.
    Consumers that need regular sampling should resample; NeuroBridge's own
    pipeline uses the stored monotonic timestamps instead.
    """

    def __init__(self, fs: float = 250.0, name: str = "TH21A",
                 source_id: str = "nb_th21a", channel: str = "Fpz") -> None:
        from pylsl import StreamInfo, StreamOutlet
        info = StreamInfo(name, "EEG", 1, fs, "float32", source_id)
        chans = info.desc().append_child("channels")
        ch = chans.append_child("channel")
        ch.append_child_value("label", channel)
        ch.append_child_value("unit", "counts")
        ch.append_child_value("type", "EEG")
        self._outlet = StreamOutlet(info, max_buffered=360)

    def push(self, values: np.ndarray, t: float | None = None) -> None:
        for v in np.asarray(values, dtype=float).ravel():
            try:
                self._outlet.push_sample([float(v)], t)
            except Exception:
                return


class SessionMarkerSink(MarkerSink):
    """Bridge a :class:`~neurobridge.store.SessionWriter` into the marker chain."""

    def __init__(self, writer, test_id: str | None = None, block: int | None = None) -> None:
        self.writer = writer
        self.test_id = test_id
        self.block = block
        self.times: list[tuple[str, int, float]] = []

    def send(self, code: int, name: str, t_mono: float,
             meta: dict[str, Any] | None = None) -> None:
        self.writer.event(int(code), name=name, t_mono=t_mono,
                          test_id=self.test_id, block=self.block, meta=meta or {})
        self.times.append((name, int(code), float(t_mono)))


def clock() -> float:
    """The one clock used for everything: ``time.perf_counter``.

    It is monotonic, high resolution, and shared by stimulus timestamps and
    packet arrival timestamps in the same process -- which is the only reason
    marker alignment is meaningful at all here.
    """
    return time.perf_counter()


@dataclass
class TimingReport:
    """Diagnostics on marker/stimulus timing, printed by the runner."""

    n_markers: int = 0
    mean_interval_ms: float = 0.0
    jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0
    late_markers: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_markers": self.n_markers,
            "mean_interval_ms": round(self.mean_interval_ms, 3),
            "jitter_ms": round(self.jitter_ms, 3),
            "max_jitter_ms": round(self.max_jitter_ms, 3),
            "late_markers": self.late_markers,
            "notes": self.notes,
        }
