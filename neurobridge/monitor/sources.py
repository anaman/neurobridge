"""Where the monitor's data comes from.

Three sources, one interface, because the monitor has to work in three
situations that are genuinely different rather than pretending to be one:

* :class:`DeviceSource` - the monitor owns the headband. Full-rate raw samples and
  a locally computed state estimate.
* :class:`BridgeSource` - ``nb serve`` already owns the headband (only one process
  can hold the BLE link, so this is the *only* option when the main app is up).
  State comes from the bridge; raw samples come from ``/samples/stream``.
* :class:`ReplaySource` - a previously recorded session, from the native format or
  from a monitor CSV. No hardware involved.

Each yields :class:`Reading` objects: raw sample chunks to draw and record, plus
an optional :class:`Snapshot` for the fields. The estimator is the *same*
:class:`~neurobridge.agent.state.StateEstimator` the bridge uses, so a value shown
here and a value served on ``/state`` cannot disagree.
"""
from __future__ import annotations

import abc
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..agent.link import BridgeClient, LinkError, Provenance, iter_sse
from ..agent.state import StateEstimate, StateEstimator
from ..config import get_settings

__all__ = ["Chunk", "Snapshot", "Reading", "MonitorSource", "DeviceSource",
           "BridgeSource", "ReplaySource", "open_source_for_mode", "BANDS_FOR_DISPLAY"]

#: display order for the band bars, low frequency first
BANDS_FOR_DISPLAY = ("delta", "theta", "alpha", "beta", "gamma")


@dataclass
class Chunk:
    """Raw samples with their monotonic times."""

    values: np.ndarray
    times: np.ndarray
    gap: bool = False

    @property
    def n(self) -> int:
        return int(self.values.size)


@dataclass
class Snapshot:
    """One state estimate, plus the display-only band shares."""

    t_mono: float
    t_wall: str
    fields: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    artifact_level: float = 0.0
    contact: str = "unknown"
    source: str = "unknown"
    seq: int | None = None
    informative: bool = False
    derivation: str = ""
    bands: dict[str, float] = field(default_factory=dict)   # percent, 0..100
    intents: list[dict] = field(default_factory=list)
    calibrated: bool | None = None

    @property
    def span(self) -> dict[str, Any]:
        return {"confidence": self.confidence, "artifact": self.artifact_level,
                "contact": self.contact, "source": self.source, "seq": self.seq}


@dataclass
class Reading:
    chunks: list[Chunk] = field(default_factory=list)
    snapshot: Snapshot | None = None

    @property
    def n_samples(self) -> int:
        return sum(c.n for c in self.chunks)


def _original_device(session_dir: Path, meta: dict[str, Any]) -> str:
    """Which device made the recording we are about to replay.

    Read from the monitor's own ``monitor.json`` first, then from the native
    ``meta.json`` (``nb run`` records it under ``device.source``). Returns ``""``
    when the recording does not say - and the banner then claims nothing, rather
    than defaulting to "real EEG", which is how a replay of a synthetic capture
    would end up being described as somebody's brain.
    """
    for name in ("monitor.json", "meta.json"):
        p = Path(session_dir) / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        prov = d.get("provenance") or {}
        if isinstance(prov, dict) and prov.get("device"):
            return str(prov["device"])
        dev = d.get("device")
        if isinstance(dev, dict) and dev.get("source"):
            return str(dev["source"])
        if isinstance(dev, str) and dev:
            return dev
    return ""


def _bands_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Relative band shares as percentages, for the display bars."""
    out: dict[str, float] = {}
    for b in BANDS_FOR_DISPLAY:
        v = metrics.get(f"rel_{b}")
        if isinstance(v, (int, float)):
            out[b] = float(v) * 100.0
    return out


def _snapshot_from_estimate(est: StateEstimate, seq: int | None = None,
                            calibrated: bool | None = None) -> Snapshot:
    info = dict(est.model_info or {})
    fields = info.get("fields") or {}
    derivations = {v.get("derivation") for v in fields.values() if isinstance(v, dict)}
    derivations.discard(None)
    informative = bool(derivations) and not derivations.issubset(
        {"uncalibrated_neutral", "no_data"})
    return Snapshot(
        t_mono=float(est.t_mono), t_wall=str(est.t_wall),
        fields={f: float(getattr(est, f, 0.5)) for f in
                ("attention", "arousal", "load", "fatigue", "relaxation")},
        confidence=float(est.confidence), artifact_level=float(est.artifact_level),
        contact=str(est.contact), source=str(est.source), seq=seq,
        informative=informative,
        derivation=",".join(sorted(derivations)),
        bands=_bands_from_metrics(info.get("metrics") or {}),
        intents=list(est.intents or []), calibrated=calibrated)


def _snapshot_from_payload(payload: dict[str, Any]) -> Snapshot:
    """Build a snapshot from a bridge /state payload (already JSON-friendly)."""
    info = dict(payload.get("model_info") or {})
    derivations = {v.get("derivation") for v in (info.get("fields") or {}).values()
                   if isinstance(v, dict)}
    derivations.discard(None)
    informative = bool(derivations) and not derivations.issubset(
        {"uncalibrated_neutral", "no_data"})
    cand = payload.get("candidate") or {}
    return Snapshot(
        t_mono=float(payload.get("t_mono") or 0.0),
        t_wall=str(payload.get("t_wall") or ""),
        fields={f: float(payload.get(f, 0.5) or 0.5) for f in
                ("attention", "arousal", "load", "fatigue", "relaxation")},
        confidence=float(payload.get("confidence") or 0.0),
        artifact_level=float(payload.get("artifact_level") or 0.0),
        contact=str(payload.get("contact") or "unknown"),
        source=str(payload.get("source") or payload.get("_device") or "unknown"),
        seq=(int(payload["seq"]) if isinstance(payload.get("seq"), int) else None),
        informative=informative, derivation=",".join(sorted(derivations)),
        bands=_bands_from_metrics(info.get("metrics") or {}),
        intents=list(payload.get("intents") or []),
        calibrated=bool(info.get("calibrated")) if "calibrated" in info else None)


# --------------------------------------------------------------------- base
class MonitorSource(abc.ABC):
    """A pollable stream of sample chunks and state snapshots."""

    name: str = "source"
    fs: float = 250.0

    def __init__(self, state_hz: float = 2.0) -> None:
        self.state_hz = max(0.1, float(state_hz))
        self._stop = threading.Event()
        self._next_state = 0.0

    # -- lifecycle
    def open(self) -> None:                                  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        self._stop.set()

    # -- data
    @abc.abstractmethod
    def poll(self, timeout_s: float = 0.02) -> Reading:
        """Return everything available now. Must never block longer than timeout."""

    # -- optional operations, honest about being unsupported
    def calibrate(self, force: bool = False) -> dict[str, Any]:
        return {"ok": False, "error": "unsupported",
                "detail": f"{self.name} cannot calibrate"}

    def provenance(self) -> Provenance:
        return Provenance(device=self.name, mode="live")

    @property
    def replay(self) -> Any:
        return None

    def _state_due(self) -> bool:
        now = time.perf_counter()
        if now >= self._next_state:
            self._next_state = now + 1.0 / self.state_hz
            return True
        return False


# ------------------------------------------------------------------- device
class DeviceSource(MonitorSource):
    """Own the headband: raw samples plus a local state estimate."""

    def __init__(self, device: str = "auto", state_hz: float = 2.0,
                 models_dir: str | Path | None = None) -> None:
        super().__init__(state_hz=state_hz)
        self.device = str(device)
        self.name = str(device)
        self._src = None
        self._est: StateEstimator | None = None
        self._pending: list[Chunk] = []
        self._seq = 0
        self._models_dir = models_dir

    def open(self) -> None:
        from ..device import open_source
        self._src = open_source(self.device)
        self._src.open()
        self.name = str(getattr(self._src, "name", self.device))
        self.fs = float(getattr(self._src, "fs", 250.0) or 250.0)
        self._est = StateEstimator(fs=self.fs, device=self.name,
                                   models_dir=self._models_dir)
        # publish the resolved source, not the alias: "auto" is not a device
        self._est.source_name = self.name

    def close(self) -> None:
        super().close()
        if self._src is not None:
            try:
                self._src.close()
            except Exception:
                pass
            self._src = None

    def provenance(self) -> Provenance:
        return Provenance(device=self.name, requested_device=self.device,
                          mode="live",
                          calibrated=bool(self._est.calibrated) if self._est else None)

    def poll(self, timeout_s: float = 0.02) -> Reading:
        out = Reading()
        if self._src is None or self._est is None:
            return out
        self._est.device_contact = str(getattr(self._src, "contact", "unknown")
                                       or "unknown")
        try:
            packets = self._src.poll(timeout_s=timeout_s)
        except Exception:
            packets = []
        for p in packets:
            v = np.asarray(getattr(p, "values", []), dtype=np.float64).reshape(-1)
            if v.size == 0:
                continue
            t = getattr(p, "t_samples", None)
            if t is None:
                t = np.full(v.shape, float(getattr(p, "t_mono", time.perf_counter())))
            t = np.asarray(t, dtype=np.float64).reshape(-1)
            if t.size != v.size:
                t = np.linspace(float(t[0]), float(t[-1]), v.size) if t.size else t
            # Record the times the estimator actually stored, not the raw packet
            # times: overlapping BLE packets get nudged forward internally, and a
            # CSV whose timeline differs from the analysed one is a data-integrity
            # trap (non-monotonic t_mono, phantom gaps).
            stored = self._est.push(v, t)
            self._pending.append(Chunk(values=v, times=stored))
        if self._pending:
            out.chunks = self._pending
            self._pending = []
        if self._state_due():
            self._seq += 1
            out.snapshot = _snapshot_from_estimate(
                self._est.estimate(), seq=self._seq,
                calibrated=bool(self._est.calibrated))
        return out

    def calibrate(self, force: bool = False) -> dict[str, Any]:
        if self._est is None:
            return {"ok": False, "error": "not_open"}
        res = dict(self._est.calibrate(force=force))
        res["calibrated"] = bool(self._est.calibrated)
        return res


# ------------------------------------------------------------------- bridge
class BridgeSource(MonitorSource):
    """Attach to a running ``nb serve`` and take both streamed endpoints."""

    def __init__(self, url: str | None = None, state_hz: float = 2.0,
                 max_hz: float = 250.0) -> None:
        super().__init__(state_hz=state_hz)
        self.client = BridgeClient(url)
        self.name = f"bridge:{self.client.base_url}"
        self.max_hz = float(max_hz)
        self._q_samples: queue.Queue = queue.Queue(maxsize=4000)
        self._q_state: queue.Queue = queue.Queue(maxsize=400)
        self._threads: list[threading.Thread] = []
        self._prov = Provenance(device="unknown", mode="live")
        self._minfo: dict[str, Any] = {}

    def open(self) -> None:
        self._minfo = self.client.health()
        self._prov = Provenance.from_health(self._minfo)
        self.fs = float(self._minfo.get("fs_effective")
                        or self._minfo.get("fs_nominal") or 250.0)
        self.name = self._prov.device
        self._spawn("/samples/stream", self._q_samples,
                    {"chunk_ms": 100, "max_hz": self.max_hz})
        self._spawn("/state/stream", self._q_state, None)

    def _spawn(self, path: str, sink: queue.Queue,
               params: dict[str, Any] | None) -> None:
        def run() -> None:
            try:
                for obj in self.client.stream_json(path, params=params,
                                                   device=self._prov.device,
                                                   stop=self._stop):
                    try:
                        sink.put_nowait(obj)
                    except queue.Full:
                        try:                       # drop the oldest, keep moving
                            sink.get_nowait()
                            sink.put_nowait(obj)
                        except Exception:
                            pass
            except LinkError:
                pass
            except Exception:
                pass

        th = threading.Thread(target=run, name=f"monitor-{path.strip('/')}",
                              daemon=True)
        th.start()
        self._threads.append(th)

    def close(self) -> None:
        super().close()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()

    def provenance(self) -> Provenance:
        return self._prov

    def poll(self, timeout_s: float = 0.02) -> Reading:
        out = Reading()
        deadline = time.perf_counter() + max(0.0, timeout_s)
        got = False
        while True:
            try:
                obj = self._q_samples.get_nowait()
            except queue.Empty:
                break
            vals = obj.get("values") or []
            if not vals:
                continue
            v = np.asarray(vals, dtype=np.float64)
            t0 = float(obj.get("t0") or 0.0)
            fs = float(obj.get("fs") or self.fs or 250.0)
            stride = int(obj.get("stride") or 1)
            t = t0 + np.arange(v.size) * (stride / fs)
            out.chunks.append(Chunk(values=v, times=t, gap=bool(obj.get("gap"))))
            got = True
            if time.perf_counter() >= deadline:
                break
        latest = None
        while True:
            try:
                latest = self._q_state.get_nowait()
            except queue.Empty:
                break
        if latest is not None and self._state_due():
            out.snapshot = _snapshot_from_payload(latest)
        elif latest is not None and out.snapshot is None and not got:
            pass
        return out

    def calibrate(self, force: bool = False) -> dict[str, Any]:
        return self.client.calibrate(force=force)


# ------------------------------------------------------------------- replay
class ReplaySource(MonitorSource):
    """Replay a recorded session (native format or a monitor CSV).

    The replayed samples are pushed through a fresh :class:`StateEstimator`, so
    the fields on screen during a replay are computed by exactly the code that
    computes them live. That also means a replay starts uncalibrated, like a live
    session does - the status bar says so rather than showing 0.5 as if it meant
    something.
    """

    def __init__(self, path: str | Path, state_hz: float = 2.0,
                 speed: float = 1.0, models_dir: str | Path | None = None,
                 loop: bool = False) -> None:
        super().__init__(state_hz=state_hz)
        self.path = Path(path)
        self.speed = float(speed)
        self.loop = bool(loop)
        self.name = f"replay:{self.path.name}"
        self._models_dir = models_dir
        self._est: StateEstimator | None = None
        self._seq = 0
        self._i = 0
        self._play = True
        self._wall0 = 0.0
        self._t_start = 0.0
        self.title = self.path.name

    # -- loading
    def _load(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.path.is_dir():
            from ..store import Session
            s = Session.load(self.path)
            meta = dict(s.meta)
            meta.setdefault("_original_device", _original_device(self.path, meta))
            return (np.asarray(s.eeg, dtype=np.float64),
                    np.asarray(s.t, dtype=np.float64), meta)
        # a monitor CSV: full-rate rows, so read only what we need
        import csv as _csv
        vals: list[float] = []
        ts: list[float] = []
        with self.path.open(newline="") as fh:
            rd = _csv.DictReader(fh)
            for row in rd:
                try:
                    vals.append(float(row["value"]))
                    ts.append(float(row["t_mono"]))
                except (KeyError, TypeError, ValueError):
                    continue
        meta = {"_original_device": _original_device(self.path.parent, {})}
        return np.asarray(vals, dtype=np.float64), np.asarray(ts, dtype=np.float64), meta

    def open(self) -> None:
        v, t, meta = self._load()
        if v.size == 0:
            raise ValueError(f"{self.path} contains no samples")
        self._v, self._t, self.meta = v, t, meta
        self.title = str(meta.get("session_id") or self.path.name)
        self._original_device = str(meta.get("_original_device") or "")
        self.fs = float(meta.get("fs") or (v.size / max(1e-9, t[-1] - t[0])) or 250.0)
        self._est = StateEstimator(fs=self.fs, device="replay",
                                   models_dir=self._models_dir)
        self._est.source_name = f"replay:{self.path.name}"
        self._i = 0
        self._t_start = float(t[0])
        self._wall0 = time.perf_counter()
        self.duration_s = float(t[-1] - t[0])
        self._play = True

    def close(self) -> None:
        super().close()

    def provenance(self) -> Provenance:
        return Provenance(device=f"replay:{self.path.name}", mode="replay",
                          session=self.title,
                          original_device=str(getattr(self, "_original_device", "")),
                          calibrated=bool(self._est.calibrated) if self._est else None)

    # -- transport controls
    def play(self) -> None:
        self._play = True
        self._wall0 = time.perf_counter() - (self._cursor_rel() / max(self.speed, 1e-6))

    def pause(self) -> None:
        self._play = False

    @property
    def playing(self) -> bool:
        return self._play

    def set_speed(self, speed: float) -> None:
        rel = self._cursor_rel()
        self.speed = max(0.05, float(speed))
        self._wall0 = time.perf_counter() - (rel / self.speed)

    def _cursor_rel(self) -> float:
        if self._i <= 0 or not hasattr(self, "_t"):
            return 0.0
        return float(self._t[min(self._i, self._t.size - 1)] - self._t_start)

    @property
    def position(self) -> tuple[float, float]:
        """(elapsed_s, duration_s) for the scrubber."""
        return self._cursor_rel(), float(getattr(self, "duration_s", 0.0))

    def seek(self, rel_s: float) -> None:
        rel_s = max(0.0, min(float(rel_s), float(getattr(self, "duration_s", 0.0))))
        self._i = int(np.searchsorted(self._t, self._t_start + rel_s))
        self._wall0 = time.perf_counter() - (rel_s / max(self.speed, 1e-6))
        if self._est is not None:
            self._est.reset()
            self._est.source_name = f"replay:{self.path.name}"
        self._seq = 0

    def restart(self) -> None:
        self.seek(0.0)
        self._play = True

    def calibrate(self, force: bool = False) -> dict[str, Any]:
        if self._est is None:
            return {"ok": False, "error": "not_open"}
        res = dict(self._est.calibrate(force=force))
        res["calibrated"] = bool(self._est.calibrated)
        return res

    # -- data
    def poll(self, timeout_s: float = 0.02) -> Reading:
        out = Reading()
        if self._est is None or not hasattr(self, "_v"):
            return out
        if not self._play:
            time.sleep(min(timeout_s, 0.02))
            if self._state_due():
                self._seq += 1
                out.snapshot = _snapshot_from_estimate(
                    self._est.estimate(), seq=self._seq,
                    calibrated=bool(self._est.calibrated))
            return out

        rel_now = (time.perf_counter() - self._wall0) * self.speed
        target = self._t_start + rel_now
        j = int(np.searchsorted(self._t, target, side="right"))
        if j > self._i:
            v = self._v[self._i:j]
            t = self._t[self._i:j]
            stored = self._est.push(v, t)
            out.chunks.append(Chunk(values=v, times=stored))
            self._i = j
        else:
            time.sleep(min(timeout_s, 0.01))
        if self._i >= self._v.size:
            if self.loop:
                self.restart()
            else:
                self._play = False
        if self._state_due():
            self._seq += 1
            out.snapshot = _snapshot_from_estimate(
                self._est.estimate(), seq=self._seq,
                calibrated=bool(self._est.calibrated))
        return out


# ------------------------------------------------------------------- factory
def open_source_for_mode(device: str = "auto", url: str | None = None,
                         replay: str | Path | None = None,
                         state_hz: float | None = None,
                         models_dir: str | Path | None = None,
                         chunk_hz: float = 250.0) -> MonitorSource:
    """Pick a source from the command line: replay > attach > own the device."""
    hz = float(state_hz if state_hz is not None else get_settings().state_hz)
    if replay:
        return ReplaySource(replay, state_hz=hz, models_dir=models_dir)
    if url:
        return BridgeSource(url, state_hz=hz, max_hz=chunk_hz)
    return DeviceSource(device, state_hz=hz, models_dir=models_dir)
