"""Session recording for the standalone monitor.

Two consumers, two formats, written at the same time:

* **CSV**, because a human asked for the data in a spreadsheet-readable form and
  that is the format most other tools will actually open. Full rate, one row per
  sample - no decimation. The point of recording is that you can go back and ask
  a question you had not thought of, so throwing away 20-45 Hz content to save
  disk would defeat the exercise.
* **The native session format** (:class:`~neurobridge.store.SessionWriter`), so a
  monitor recording is a first-class session that ``nb eval``, ``nb train`` and
  ``nb report`` can use alongside anything recorded by ``nb run``.

Both are written from the same calls, so they cannot drift apart. A CSV file
that has lost its neighbouring ``meta.json`` is still self-describing: the
source is a column on every state row and the monitor's own provenance block is
written to ``monitor.json``.

WHY the provenance is recorded so insistently: the same mistake this project
refuses to make at the API level - presenting generated numbers as somebody's
brain state - is just as easy to make with a CSV that does not say where it came
from a week later.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import SESSION_DIR, ensure_dirs
from ..store import SessionWriter, new_session_id
from ..agent.link import Provenance

__all__ = ["MonitorRecorder", "RecorderReceipt", "EEG_CSV_COLUMNS", "STATE_CSV_COLUMNS"]

EEG_CSV_COLUMNS = ["sample", "t_mono", "t_rel_s", "value"]
STATE_CSV_COLUMNS = ["t_mono", "t_wall", "seq", "source", "contact", "confidence",
                     "artifact_level", "attention", "arousal", "load", "fatigue",
                     "relaxation", "informative", "derivation"]
EVENT_CSV_COLUMNS = ["t_mono", "t_wall", "name", "meta"]


@dataclass
class RecorderReceipt:
    """What was written, and the caveats that belong with it."""

    path: Path
    session_id: str
    n_samples: int = 0
    n_state_rows: int = 0
    n_events: int = 0
    duration_s: float = 0.0
    fs_effective: float = 0.0
    csv_eeg: Path | None = None
    csv_state: Path | None = None
    csv_events: Path | None = None
    native: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "session_id": self.session_id,
            "n_samples": self.n_samples, "n_state_rows": self.n_state_rows,
            "n_events": self.n_events, "duration_s": round(self.duration_s, 3),
            "fs_effective": round(self.fs_effective, 3),
            "csv_eeg": str(self.csv_eeg) if self.csv_eeg else None,
            "csv_state": str(self.csv_state) if self.csv_state else None,
            "csv_events": str(self.csv_events) if self.csv_events else None,
            "native": self.native, "provenance": self.provenance,
        }


class MonitorRecorder:
    """Stream a session to CSV and to the native format at the same time."""

    def __init__(self, out_dir: Path | str | None = None,
                 session_id: str | None = None,
                 meta: dict[str, Any] | None = None,
                 provenance: Provenance | None = None,
                 csv_enabled: bool = True, native_enabled: bool = True,
                 flush_every: int = 500, state_flush_every: int = 8) -> None:
        ensure_dirs()
        self.session_id = session_id or new_session_id("mon")
        self.path = Path(out_dir or SESSION_DIR) / self.session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.csv_enabled = bool(csv_enabled)
        self.native_enabled = bool(native_enabled)
        #: 500 samples = 2 s of EEG, and 8 state rows = 4 s. Both are small writes
        #: that keep the CSVs tail-able while a session is running, and bound what
        #: a hard kill can lose. A clean close flushes everything regardless.
        self.flush_every = max(1, int(flush_every))
        self.state_flush_every = max(1, int(state_flush_every))
        self.provenance = provenance or Provenance()

        self.n_samples = 0
        self.n_state_rows = 0
        self.n_events = 0
        self._t0: float | None = None
        self._t_last: float | None = None
        self._closed = False

        #: rows buffered between flushes; the acquisition path must stay cheap
        self._v_buf: list[str] = []
        self._s_buf: list[list[Any]] = []

        meta = dict(meta or {})
        meta.setdefault("session_id", self.session_id)
        meta.setdefault("kind", "monitor")
        meta.setdefault("fs", 250.0)
        # Same shape the runner writes ({"source": ...}), so anything that reads a
        # session's device - `nb sessions list`, the replay provenance lookup -
        # finds it here too instead of treating a monitor recording as unknown.
        meta.setdefault("device", {
            "source": self.provenance.device,
            "fs": float(meta.get("fs") or 250.0),
            "montage": self.provenance.label,
        })
        meta["provenance"] = self.provenance.as_dict()
        meta["monitor"] = {"csv": self.csv_enabled, "native": self.native_enabled}
        self.meta = meta

        self._eeg_fh = None
        self._eeg_writer = None
        self._state_fh = None
        self._state_writer = None
        self._event_fh = None
        self._event_writer = None
        if self.csv_enabled:
            self._eeg_fh = (self.path / "eeg.csv").open("w", newline="", buffering=1)
            self._eeg_writer = csv.writer(self._eeg_fh)
            self._eeg_writer.writerow(EEG_CSV_COLUMNS)
            self._state_fh = (self.path / "state.csv").open("w", newline="", buffering=1)
            self._state_writer = csv.writer(self._state_fh)
            self._state_writer.writerow(STATE_CSV_COLUMNS)
            self._event_fh = (self.path / "events.csv").open("w", newline="", buffering=1)
            self._event_writer = csv.writer(self._event_fh)
            self._event_writer.writerow(EVENT_CSV_COLUMNS)

        self._writer: SessionWriter | None = None
        if self.native_enabled:
            self._writer = SessionWriter(session_id=self.session_id, meta=self.meta,
                                         root=self.path.parent)
            self._writer.note(f"# monitor session {self.session_id}")
            self._writer.note(f"- source: {self.provenance.device} "
                              f"(requested {self.provenance.requested_device or '-'})")
            self._writer.note(f"- mode: {self.provenance.mode}")
            self._writer.note(f"- CSV: eeg.csv, state.csv, events.csv")

        self._write_provenance()

    # ------------------------------------------------------------------ meta
    def _write_provenance(self) -> None:
        (self.path / "monitor.json").write_text(json.dumps({
            "session_id": self.session_id,
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "provenance": self.provenance.as_dict(),
            "label": self.provenance.label,
            "banner": self.provenance.banner(),
            "csv_columns": {"eeg": EEG_CSV_COLUMNS, "state": STATE_CSV_COLUMNS,
                            "events": EVENT_CSV_COLUMNS},
            "note": ("CSV timestamps are monotonic seconds (time.perf_counter); "
                     "t_rel_s is relative to the first sample of this session. "
                     "Values are raw uncalibrated ADC counts, not microvolts."),
        }, indent=2, default=str))

    # --------------------------------------------------------------- samples
    def add_samples(self, values: Iterable[float] | np.ndarray,
                    times: Iterable[float] | np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return
        if t.size != v.size:
            raise ValueError(f"values {v.shape} and times {t.shape} must match")
        if self._t0 is None:
            self._t0 = float(t[0])
        self._t_last = float(t[-1])
        if self._eeg_writer is not None:
            base = self.n_samples
            for i in range(v.size):
                self._v_buf.append(
                    f"{base + i},{t[i]:.6f},{t[i] - self._t0:.6f},{v[i]:.6f}")
            if len(self._v_buf) >= self.flush_every:
                self._flush_samples()
        if self._writer is not None:
            self._writer.add_samples(v, t)
        self.n_samples += int(v.size)

    def _flush_samples(self) -> None:
        if not self._v_buf or self._eeg_fh is None:
            return
        self._eeg_fh.write("\n".join(self._v_buf) + "\n")
        self._v_buf.clear()

    # ----------------------------------------------------------------- state
    def add_state(self, t_mono: float, t_wall: str, fields: dict[str, float],
                  *, confidence: float, artifact_level: float, contact: str,
                  source: str, seq: int | None, informative: bool = True,
                  derivation: str = "", intents: list[dict] | None = None) -> None:
        """One row per state update (the bridge publishes at ~2 Hz)."""
        if self._state_writer is None:
            self.n_state_rows += 1
            return
        self._s_buf.append([
            f"{float(t_mono):.6f}", t_wall, "" if seq is None else int(seq), source,
            contact, f"{float(confidence):.4f}", f"{float(artifact_level):.4f}",
            *[f"{float(fields.get(f, 0.5)):.4f}" for f in
              ("attention", "arousal", "load", "fatigue", "relaxation")],
            int(bool(informative)), derivation,
        ])
        self.n_state_rows += 1
        if len(self._s_buf) >= self.state_flush_every:
            self._flush_state()

    def _flush_state(self) -> None:
        if not self._s_buf or self._state_fh is None:
            return
        self._state_writer.writerows(self._s_buf)
        self._s_buf.clear()

    # ---------------------------------------------------------------- events
    def event(self, name: str, meta: dict[str, Any] | None = None,
              t_mono: float | None = None) -> float:
        t = float(time.perf_counter() if t_mono is None else t_mono)
        wall = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta = dict(meta or {})
        if self._event_writer is not None:
            self._event_writer.writerow([f"{t:.6f}", wall, str(name),
                                         json.dumps(meta, default=str)])
        if self._writer is not None:
            # code 0 with the real name in `name`: SessionWriter.event() only
            # accepts a *known framing marker* as a string code, so passing
            # "monitor_mark" would raise and silently divert every marker to the
            # fallback log instead of the session timeline.
            self._writer.event(0, name=str(name), t_mono=t, meta=meta)
        self.n_events += 1
        return t

    # ----------------------------------------------------------------- close
    def close(self, summary: dict[str, Any] | None = None) -> RecorderReceipt:
        if self._closed:
            return self.receipt()
        self._closed = True
        self._flush_samples()
        self._flush_state()
        dur = float((self._t_last or 0.0) - (self._t0 or 0.0))
        eff = float(self.n_samples / dur) if dur > 0 else 0.0
        for fh in (self._eeg_fh, self._state_fh, self._event_fh):
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass
        if self._writer is not None:
            try:
                self._writer.close({**(summary or {}), "kind": "monitor"})
            except Exception:
                pass
        rec = self.receipt()
        (self.path / "monitor_summary.json").write_text(
            json.dumps(rec.as_dict(), indent=2, default=str))
        return rec

    def receipt(self) -> RecorderReceipt:
        dur = float((self._t_last or 0.0) - (self._t0 or 0.0))
        return RecorderReceipt(
            path=self.path, session_id=self.session_id, n_samples=self.n_samples,
            n_state_rows=self.n_state_rows, n_events=self.n_events,
            duration_s=dur,
            fs_effective=float(self.n_samples / dur) if dur > 0 else 0.0,
            csv_eeg=(self.path / "eeg.csv") if self.csv_enabled else None,
            csv_state=(self.path / "state.csv") if self.csv_enabled else None,
            csv_events=(self.path / "events.csv") if self.csv_enabled else None,
            native=self.native_enabled, provenance=self.provenance.as_dict())

    def __enter__(self) -> "MonitorRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
