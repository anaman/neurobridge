"""Session storage: EEG samples, marker events and metadata on disk.

One session directory looks like::

    data/sessions/s-20261013-142530-ab12/
        meta.json      device, fs, montage, settings snapshot, quality verdict
        eeg.f32.npy    float32 ADC counts, sample-major
        time.f64.npy   float64 monotonic seconds, one per sample
        events.jsonl   one JSON object per marker, append-only + flushed
        notes.md       free text (operator notes, per-test notes)
        summary.json   written on close: counts, duration, quality

Everything is written through ``SessionWriter`` during acquisition, and read
back through ``Session``. Both use plain numpy/JSON so a session stays
readable even if NeuroBridge is uninstalled.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import SESSION_DIR, ensure_dirs, new_session_id

# ---------------------------------------------------------------- markers
# Marker codes are integers so they survive an LSL round-trip unambiguously.
# Every test spec reserves codes from these ranges (see catalog/schema.py).
MARKER_CODES: dict[str, int] = {
    "session_start": 1,
    "session_end": 2,
    "block_start": 10,
    "block_end": 11,
    "test_start": 12,
    "test_end": 13,
    "break_start": 14,
    "break_end": 15,
    "instruction": 16,
    "practice_start": 17,
    "practice_end": 18,
    "trial_onset": 100,
    "stim_onset": 101,
    "stim_offset": 102,
    "fixation_onset": 103,
    "cue_onset": 104,
    "response_prompt": 105,
    "response": 106,
    "feedback": 107,
    "target": 110,
    "standard": 111,
    "deviant": 112,
    "distractor": 113,
    "novel": 114,
    "class_a": 120,
    "class_b": 121,
    "class_c": 122,
    "class_d": 123,
    "question": 130,
    "answer_yes": 131,
    "answer_no": 132,
    "probe": 133,
    "blink": 140,
    "clench": 141,
    "swallow": 142,
    "artifact_motion": 143,
    "reject": 199,
    "latency_probe": 200,
    "sham": 201,
}

#: code -> canonical name, used when reading sessions back
MARKER_NAMES: dict[int, str] = {v: k for k, v in MARKER_CODES.items()}


def code_of(name: str) -> int:
    """Marker code for a *symbolic* name from the global table, or an int passthrough.

    This table is only the framing vocabulary (session/block/trial events and
    the generic class names). Tests declare their own codes in the spec's
    ``markers`` block and callers must use ``TestSpec.code(name)`` instead of
    this function, because two specs in one tier may reuse a code for different
    meanings.
    """
    if isinstance(name, int):
        return int(name)
    if name not in MARKER_CODES:
        raise KeyError(
            f"unknown marker {name!r}. Use store.code_of() only for framing markers "
            f"({sorted(MARKER_CODES)}), or TestSpec.code(name) for a test's own markers.")
    return MARKER_CODES[name]


def name_of(code: int) -> str:
    return MARKER_NAMES.get(int(code), f"code_{int(code)}")


#: How trustworthy the stimulus timeline is for this recording. Adopted from
#: the imported external protocol (docs/PROTOCOL_MAPPING.md): a bridge-side
#: software timestamp is never called an exact visual or acoustic onset.
TIMING_QUALITY = ("verified", "coarse", "unverified")

#: The three-label policy. Every trial should be describable three ways, and
#: they must never be conflated:
#:
#: ``task_label``      what the trial presented or asked for (a stimulus identity,
#:                     a requested intention). This is NOT a fact about the person.
#: ``reported_label``  what the participant actually reported or did, including
#:                     "unknown", "skipped" and "failed". Missing is not neutral.
#: ``predicted_label`` what a model estimated from eligible EEG, with uncertainty
#:                     and signal-quality information attached.
#:
#: A picture of a forest is a forest stimulus; it is not a ground-truth label for
#: calmness. A "YES" cue is a requested intention; it is not proof of an
#: internally expressed YES. A correct answer shows task performance, not the
#: neural mechanism used.
LABEL_POLICY = ("task_label", "reported_label", "predicted_label")


# ---------------------------------------------------------------- events
@dataclass
class Event:
    """A stimulus/response event, stamped on the shared monotonic clock."""

    t_mono: float
    code: int
    name: str = ""
    test_id: str | None = None
    trial: int | None = None
    block: int | None = None
    phase: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # the three-label policy -- see LABEL_POLICY above
    task_label: str | None = None
    reported_label: str | None = None
    predicted_label: str | None = None
    #: 'verified' | 'coarse' | 'unverified' -- see TIMING_QUALITY
    timing_quality: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = name_of(self.code)
        self.t_mono = float(self.t_mono)
        self.code = int(self.code)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["t_wall"] = getattr(self, "t_wall", None)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        d.pop("t_wall", None)
        return cls(**d)


# ---------------------------------------------------------------- writer
class SessionWriter:
    """Streaming writer used while a session runs.

    Events are flushed immediately (a crash must not lose the timeline).
    Samples are buffered and flushed in chunks to keep the acquisition
    callback cheap.
    """

    def __init__(self, session_id: str | None = None, meta: dict[str, Any] | None = None,
                 root: Path | None = None, flush_every: int = 25_000) -> None:
        ensure_dirs()
        self.session_id = session_id or new_session_id()
        self.path = Path(root or SESSION_DIR) / self.session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every

        self.meta: dict[str, Any] = dict(meta or {})
        self.meta.setdefault("session_id", self.session_id)
        self.meta.setdefault("created_wall", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.meta.setdefault("created_mono", time.perf_counter())

        self._events_path = self.path / "events.jsonl"
        self._events_fh = self._events_path.open("a", buffering=1)
        self._notes_path = self.path / "notes.md"
        self._buf_v: list[np.ndarray] = []
        self._buf_t: list[np.ndarray] = []
        self._n_buffered = 0
        self._n_samples = 0
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._closed = False
        self._t0: float | None = None
        self._write_meta()

    # -- metadata ---------------------------------------------------
    def _write_meta(self) -> None:
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def set_meta(self, **kw: Any) -> None:
        self.meta.update(kw)
        self._write_meta()

    # -- events -----------------------------------------------------
    def event(self, code: int | str, name: str = "", t_mono: float | None = None,
              **kw: Any) -> Event:
        """Record a marker. Returns the Event (with its timestamp).

        Accepts the label-policy keywords (``task_label``, ``reported_label``,
        ``predicted_label``) and ``timing_quality`` directly; anything else goes
        into ``meta``. Use them: a stimulus identity is not a ground truth about
        the person, and a requested intention is not an expressed one.
        """
        ev = Event(t_mono=time.perf_counter() if t_mono is None else float(t_mono),
                   code=code if isinstance(code, int) else code_of(code),
                   name=name, **kw)
        self._events_fh.write(json.dumps(ev.to_json(), default=str) + "\n")
        return ev

    def note(self, text: str) -> None:
        with self._notes_path.open("a") as fh:
            fh.write(text.rstrip() + "\n")

    # -- samples ----------------------------------------------------
    def add_samples(self, values: Sequence[float] | np.ndarray,
                    t_mono: Sequence[float] | np.ndarray | float) -> None:
        """Append one or more samples with their monotonic timestamps."""
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        if np.isscalar(t_mono):
            t = np.full(v.shape, float(t_mono), dtype=np.float64)
        else:
            t = np.asarray(t_mono, dtype=np.float64).reshape(-1)
            if t.shape != v.shape:
                raise ValueError(f"values {v.shape} and times {t.shape} must match")
        if v.size == 0:
            return
        if self._t0 is None:
            self._t0 = float(t[0])
        self._chunks.append((v, t))
        self._n_buffered += v.size
        self._n_samples += v.size
        if self._n_buffered >= self.flush_every:
            self._flush_samples()

    def _flush_samples(self) -> None:
        """Persist buffered samples. np.save is atomic-ish per file, so we
        append by rewriting from the accumulated chunks (sessions are small:
        ~1 MB per 17 min at 250 Hz float32)."""
        # chunks stay in memory for the append; on very long sessions the
        # writer switches to a raw binary append instead of holding arrays.
        try:
            self._write_arrays()
        except Exception:
            self._write_arrays_append()
        self._n_buffered = 0

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._chunks:
            return np.zeros(0, np.float32), np.zeros(0, np.float64)
        v = np.concatenate([c[0] for c in self._chunks]).astype(np.float32, copy=False)
        t = np.concatenate([c[1] for c in self._chunks]).astype(np.float64, copy=False)
        return v, t

    def _write_arrays(self) -> None:
        v, t = self._stack()
        np.save(self.path / "eeg.f32.npy", v, allow_pickle=False)
        np.save(self.path / "time.f64.npy", t, allow_pickle=False)

    def _write_arrays_append(self) -> None:
        """Fallback for memory pressure: keep the last flush in a .bin append log."""
        with (self.path / "eeg.bin").open("ab") as fh:
            v, t = self._stack()
            fh.write(v.tobytes())
        with (self.path / "time.bin").open("ab") as fh:
            fh.write(t.tobytes())
        self.meta["storage"] = "bin"
        self._write_meta()

    # -- close ------------------------------------------------------
    def close(self, summary: dict[str, Any] | None = None) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if not (self.path / "eeg.bin").exists():
            self._write_arrays()
        v, t = self._stack()
        dur = float(t[-1] - t[0]) if t.size else 0.0
        fs_eff = float(v.size / dur) if dur > 0 else 0.0
        summary = dict(summary or {})
        summary.setdefault("n_samples", int(v.size))
        summary.setdefault("duration_s", round(dur, 3))
        summary.setdefault("fs_effective", round(fs_eff, 3))
        summary.setdefault("n_events", _count_lines(self._events_path))
        if v.size:
            summary.setdefault("raw_min", float(np.min(v)))
            summary.setdefault("raw_max", float(np.max(v)))
            summary.setdefault("raw_ptp", float(np.ptp(v)))
        self.meta["closed_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.meta["summary"] = summary
        self._write_meta()
        (self.path / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._events_fh.close()
        return self.path


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


# ---------------------------------------------------------------- reader
class Session:
    """Read-only view of a recorded session."""

    def __init__(self, path: Path, meta: dict[str, Any], eeg: np.ndarray,
                 t: np.ndarray, events: list[Event]) -> None:
        self.path = Path(path)
        self.meta = meta
        self.eeg = eeg
        self.t = t
        self.events = events

    # -- loading ----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Session":
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"no such session directory: {path}")
        meta_p = path / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

        if (path / "eeg.bin").exists():
            eeg = np.fromfile(path / "eeg.bin", dtype=np.float32)
            t = np.fromfile(path / "time.bin", dtype=np.float64)
        else:
            eeg_p = path / "eeg.f32.npy"
            t_p = path / "time.f64.npy"
            eeg = np.load(eeg_p) if eeg_p.exists() else np.zeros(0, np.float32)
            t = np.load(t_p) if t_p.exists() else np.zeros(0, np.float64)

        events: list[Event] = []
        ev_p = path / "events.jsonl"
        if ev_p.exists():
            for line in ev_p.read_text().splitlines():
                line = line.strip()
                if line:
                    events.append(Event.from_json(json.loads(line)))
        events.sort(key=lambda e: e.t_mono)
        return cls(path, meta, eeg, t, events)

    @classmethod
    def latest(cls) -> "Session":
        from .config import list_sessions
        sessions = list_sessions()
        if not sessions:
            raise FileNotFoundError("no sessions recorded yet; run `nb run T002` first")
        return cls.load(sessions[-1])

    # -- convenience ------------------------------------------------
    @property
    def fs(self) -> float:
        return float(self.meta.get("fs") or self.meta.get("settings", {}).get("fs") or 250.0)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def notes(self) -> str:
        p = self.path / "notes.md"
        return p.read_text() if p.exists() else ""

    def events_for(self, test_id: str | None = None,
                   codes: Iterable[int] | None = None,
                   names: Iterable[str] | None = None,
                   phase: str | None = None) -> list[Event]:
        want_codes = set(int(c) for c in codes) if codes is not None else None
        want_names = set(names) if names is not None else None
        out = []
        for e in self.events:
            if test_id is not None and e.test_id != test_id:
                continue
            if phase is not None and e.phase != phase:
                continue
            if want_codes is not None and e.code not in want_codes:
                continue
            if want_names is not None and e.name not in want_names:
                continue
            out.append(e)
        return out

    def test_ids(self) -> list[str]:
        seen: list[str] = []
        for e in self.events:
            if e.test_id and e.test_id not in seen:
                seen.append(e.test_id)
        return seen

    def events_dataframe(self):
        import pandas as pd
        return pd.DataFrame([e.to_json() for e in self.events])

    def sample_times(self) -> np.ndarray:
        """Monotonic timestamps, made relative to the first sample."""
        return self.t - self.t[0] if self.t.size else self.t

    def summary(self) -> dict[str, Any]:
        return dict(self.meta.get("summary", {}) or {})

    # -- optional synchronised voice recording -----------------------
    @property
    def audio_path(self) -> Path | None:
        """``audio.wav`` if this session was recorded with the microphone."""
        p = self.path / "audio.wav"
        return p if p.exists() else None

    @property
    def audio_meta(self) -> dict[str, Any]:
        """Contents of ``audio.json`` (device, fs, latency, t_start_mono)."""
        p = self.path / "audio.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def load_audio(self):
        """-> (samples float32 in [-1,1], fs) or ``(None, 0)`` when absent.

        Imported lazily so the store keeps working on a machine without
        sounddevice or any audio support at all.
        """
        p = self.audio_path
        if p is None:
            return None, 0
        from .voice import read_wav
        return read_wav(p)

    def has_audio(self) -> bool:
        return self.audio_path is not None

    def audio_summary(self) -> dict[str, Any]:
        """What the microphone recorded, if anything (never raises)."""
        if not self.has_audio():
            return {"available": False}
        m = self.audio_meta
        return {"available": True, "file": str(self.audio_path),
                "device": m.get("device"), "fs": m.get("fs"),
                "duration_s": m.get("duration_s"), "rms_dbfs": m.get("rms_dbfs"),
                "input_latency_ms": m.get("input_latency_ms"),
                "timing_quality": "coarse"}

    def __repr__(self) -> str:
        return (f"<Session {self.path.name}: {self.eeg.size} samples, "
                f"{self.duration_s:.1f}s, {len(self.events)} events>")
