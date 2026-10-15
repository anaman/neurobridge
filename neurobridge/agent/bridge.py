"""FastAPI bridge: exposes one honest frontal-EEG state stream to an LLM agent.

Design decisions worth stating out loud
---------------------------------------
* **Two modes, chosen at construction.** ``live`` opens an ``EEGSource`` in a
  background thread and feeds the estimator; ``replay`` walks a recorded session
  directory (or a synthetic session when no session exists yet). Request handlers
  therefore *never* touch BLE: they read the estimator's cached window, which is
  what keeps a 2 Hz poll cheap while acquisition runs at 250 Hz.
* **Nothing here is a thought decoder.** The routes publish arousal, gross
  attention, a cognitive-load *proxy*, fatigue, relaxation, signal quality, a
  confidence and two artefact commands (blink, jaw clench). ``/schema`` says so
  in machine-readable form, including a ready-made LLM tool definition that
  repeats the limitation.
* **Never an HTML error page.** Every failure path returns JSON, and the global
  handler puts the traceback in ``detail`` so the agent (or a human) can see what
  actually broke instead of a 500 page.
* **No blocking calls in handlers.** All device I/O lives in the feeder thread;
  handlers only read locked snapshots or the on-disk session/report files.

Route summary (see docs/INTERFACES.md section 4 for the frozen contract):
``GET /health``, ``GET /schema``, ``GET /state``, ``GET /state/stream`` (SSE),
``GET /events``, ``POST /mark``, ``WS /ws``, ``GET /report/{test_id}``.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:                                             # optional: no web stack required
    from fastapi import WebSocket
    from fastapi.routing import APIWebSocketRoute
except Exception:                                # pragma: no cover - fastapi absent
    WebSocket = Any                             # type: ignore[assignment,misc]
    APIWebSocketRoute = None                    # type: ignore[assignment,misc]

from .. import __version__
from ..config import (MODEL_DIR, REPORT_DIR, SESSION_DIR, get_settings,
                      list_sessions)
from ..device.base import Packet, clock
from ..store import Session, SessionWriter
from .link import Provenance
from .state import IntentDetector, StateEstimate, StateEstimator

__all__ = ["create_app", "serve", "state_to_agent_json", "BridgeState"]

#: state-test ids whose trained model is relevant, by catalogue tier (see
#: docs/CATALOG_SPEC.md: tiers 1, 2, 6, 7 carry agent_use=state|command|both).
STATE_TESTS = ("T101", "T102", "T104", "T105", "T108", "T201", "T202", "T203",
               "T204", "T205", "T206", "T207", "T208", "T501", "T502", "T505",
               "T506", "T507", "T601", "T603", "T605", "T606", "T607", "T608",
               "T701", "T702", "T706", "T707", "T708")

#: fields as the agent sees them. ``derivation`` is filled in at runtime from
#: the estimator's model_info, because on day one every field is heuristic and
#: the agent must be able to tell the difference.
FIELD_DOCS: dict[str, dict[str, Any]] = {
    "attention": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("Gross engagement of attention. Monotone in "
                    "beta/(theta+alpha) relative to your own session baseline. "
                    "0.5 == at baseline; it does NOT say what you are attending to."),
        "why": "Beta rises and slow activity falls with task engagement at Fpz.",
        "action_hint": "Only act on sustained movement, not single samples.",
    },
    "arousal": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("Alertness/activation. Monotone in beta/alpha and in mean beta "
                    "power vs baseline. Low values are consistent with drowsiness."),
        "why": "The classic arousal axis is the fastest thing a frontal dry channel measures.",
        "action_hint": "Treat low arousal as 'offer a break', never as a command.",
    },
    "load": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("Cognitive-effort PROXY only, monotone in frontal theta/beta vs "
                    "baseline. It is not a reading of what you are thinking about."),
        "why": "Frontal-midline theta rises with working-memory demand at Fpz.",
        "action_hint": "Use to pace difficulty; when high, simplify or slow down.",
    },
    "fatigue": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("Fatigue/drowsiness. Monotone in (theta+delta)/beta drift, alpha "
                    "slowing (peak frequency dropping) and time-on-task."),
        "why": "Both slow-wave increase and alpha slowing are reported drowsiness signs.",
        "action_hint": "Suggest rest at high values; never fail the user for it.",
    },
    "relaxation": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": "Calm/relaxed. Monotone in relative alpha (8-13 Hz share) vs baseline.",
        "why": "Alpha is the one rhythm this montage sees well with eyes closed.",
        "action_hint": "Useful for pacing and breaks, not for control.",
    },
    "valence": {
        "type": "null", "range": None, "nullable": True,
        "meaning": ("ALWAYS null. Frontal-asymmetry valence requires two symmetric "
                    "channels; this device has one, so it is undefined and never faked."),
        "why": "Kept in the schema only so consumers can rely on a stable shape.",
        "action_hint": "Never use. If you need affect, ask the user in words.",
    },
    "confidence": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("How much this window can be trusted, combining signal quality "
                    "(contact, saturation, drift), artefact level (motion, mains) and "
                    "distance from your baseline distribution. Below 0.2 the capture is "
                    "saturated or off-head and every field above is meaningless."),
        "why": "A dry electrode drifts and rails; an agent must be able to tell.",
        "action_hint": "Below 0.4 do not act on state - ask a question instead.",
    },
    "artifact_level": {
        "type": "number", "range": [0, 1], "nullable": False,
        "meaning": ("Share of the window spoiled by artefacts (saturation, motion jumps, "
                    "mains hum). Diagnostic, not a brain signal."),
        "why": "Tells the agent whether a sudden field swing is brain or cable.",
        "action_hint": "High values invalidate a single-sample reading.",
    },
    "contact": {
        "type": "string", "range": None, "nullable": False,
        "meaning": ("Electrode self-check / synthetic contact description. 'no electrodes "
                    "reporting OK' means the numbers are electrode noise, not EEG."),
        "why": "The TH21A reports seating quality; surface it instead of hiding it.",
        "action_hint": "If not good, ask the human to reseat the band.",
    },
    "intents": {
        "type": "array", "range": None, "nullable": False,
        "meaning": ("Recent detected artefact commands: {'intent': 'blink'|'clench', "
                    "'t': monotonic seconds, 'confidence': 0..1}. blink == confirm/select, "
                    "clench == cancel/escape. These are the only commands this hardware "
                    "supports reliably."),
        "why": "Blink and jaw clench are enormous at Fpz and need no training.",
        "action_hint": "Act only when intent confidence >= 0.6.",
    },
    "model_info": {
        "type": "object", "range": None, "nullable": False,
        "meaning": ("Provenance, per field: 'trained_model' vs 'heuristic_baseline' vs "
                    "'uncalibrated_neutral', plus calibration path, model ids, quality "
                    "report, errors and honesty notes. Read this before trusting a field."),
        "why": "On day one nothing is trained; a bridge that hid that would lie.",
        "action_hint": "If calibrated == false, treat all fields as relative at best.",
    },
    "t_mono": {
        "type": "number", "range": None, "nullable": False,
        "meaning": "Monotonic seconds (time.perf_counter) of this estimate; the shared "
                   "clock for markers, samples and agent actions.",
        "why": "Wall time can jump; monotonic time cannot.",
        "action_hint": "Use for `since` on /events and for POST /mark alignment.",
    },
    "t_wall": {
        "type": "string", "range": None, "nullable": False,
        "meaning": "Human-readable local time of this estimate (for logs only).",
        "why": "Humans read logs; monotonic is for arithmetic.",
        "action_hint": "Do not compute with this.",
    },
    "seq": {
        "type": "integer", "range": None, "nullable": False,
        "meaning": "Monotonically increasing sequence number of pushed payloads. A gap "
                   "means you dropped a message (or the bridge restarted).",
        "why": "Agents that miss a drop silently act on stale state.",
        "action_hint": "On a gap, re-read /state once before acting.",
    },
    "source": {
        "type": "string", "range": None, "nullable": False,
        "meaning": "Which device produced this: 'fake' (synthetic) or 'th21a' (real headband).",
        "why": "A synthetic session must never be mistaken for a real one.",
        "action_hint": "Never present synthetic state to a human as their brain state.",
    },
}

#: copy-pasteable tool definitions for an LLM tool-calling loop.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_brain_state",
            "description": (
                "Read the user's current frontal-EEG state from the NeuroBridge "
                "headband (one dry frontal electrode). Returns arousal, gross "
                "attention, a cognitive-load PROXY, fatigue, relaxation, signal "
                "quality and a confidence. It cannot read words, sentences, images "
                "or intentions. Values are 0..1 relative to the user's own session "
                "baseline; 0.5 means 'at baseline'. If confidence < 0.4, do not act "
                "on the reading. Do not call more often than twice per second."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "explain": {
                        "type": "boolean",
                        "description": "Include model_info provenance and raw metrics.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_brain_state",
            "description": (
                "Wait until the user's EEG state crosses a condition, then return it. "
                "Use instead of polling get_brain_state in a loop, to avoid acting on "
                "one noisy sample. Examples: wait until load > 0.7 (user is "
                "overloaded), or arousal < 0.3 (user is getting drowsy)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["attention", "arousal", "load", "fatigue",
                                 "relaxation", "confidence"],
                    },
                    "above": {"type": "number", "minimum": 0, "maximum": 1},
                    "below": {"type": "number", "minimum": 0, "maximum": 1},
                    "min_confidence": {"type": "number", "minimum": 0, "maximum": 1,
                                       "default": 0.4},
                    "timeout_s": {"type": "number", "default": 60},
                },
                "required": ["field", "timeout_s"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_brain_event",
            "description": (
                "Write an agent-side marker into the same EEG timeline the headband "
                "samples use. Call it immediately before and after any action you take "
                "that could affect the user, so later analysis can align your actions "
                "with the brain data. Example names: agent_prompt_shown, "
                "agent_paused, agent_break_offer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short snake_case event name"},
                    "meta": {"type": "object", "description": "arbitrary JSON details",
                             "additionalProperties": True},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate_signal",
            "description": (
                "Read the current CANDIDATE signal from the headband (one dry "
                "frontal electrode). This is a cautious candidate, not a fact: "
                "candidate_label is 'yes', 'no' or 'uncertain', and it is "
                "'uncertain' whenever signal quality, confidence or a validated "
                "decision rule is missing. class_probabilities is null unless a "
                "model produced calibrated probabilities. A candidate 'yes' is "
                "NOT consent and must never trigger a consequential or external "
                "action; at most it may support a reversible comfort action "
                "such as pausing stimulus presentation, with the manual stop "
                "still available. If the answer matters, ask the user and call "
                "confirm_candidate with their answer."
            ),
            "parameters": {"type": "object", "properties": {},
                           "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_candidate",
            "description": (
                "Record the participant's own correction of a candidate signal. "
                "Call this whenever you asked the user about a candidate and they "
                "answered. The correction is stored separately from the "
                "prediction, which was already logged when it was made. Never "
                "invent the response, and never call with 'yes' on the user's "
                "behalf: self-report is authoritative about their experience."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_seq": {"type": "integer",
                                      "description": "seq of the candidate being corrected"},
                    "participant_response": {"type": "string",
                                             "enum": ["yes", "no", "skipped", "unsure"]},
                    "note": {"type": "string", "description": "optional context"},
                },
                "required": ["candidate_seq", "participant_response"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_brain_events",
            "description": (
                "List recent markers and detected blink/clench commands in monotonic "
                "order. Use to see whether the user just blinked (confirm/select) or "
                "clenched (cancel/escape) and to correlate their reactions with your "
                "own actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {"type": "number",
                              "description": "monotonic seconds lower bound (from t_mono)"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# --------------------------------------------------------------------- payload
def state_to_agent_json(estimate: StateEstimate, *, seq: int | None = None) -> dict[str, Any]:
    """The exact JSON an agent consumes for one state.

    Adds ``seq`` (monotonic push counter) and guarantees ``t_mono`` is present
    so a consumer can detect dropped messages and align with /events and /mark.
    Everything else is ``StateEstimate.as_dict()``; no field is invented here.
    """
    d = estimate.as_dict()
    d["t_mono"] = float(d.get("t_mono") or 0.0)
    if seq is not None:
        d["seq"] = int(seq)
    return d


# ----------------------------------------------------------------- bridge state
class BridgeState:
    """Mutable bridge runtime: feeder lifecycle, events, sequence numbers.

    Kept as one object (rather than module globals) so the app can be built more
    than once in a process - tests and ``nb simulate`` both do that - and so the
    feeder can be shut down deterministically.
    """

    def __init__(self, estimator: StateEstimator | None = None,
                 device: str = "auto", mode: str = "live",
                 replay_session: str | Path | None = None,
                 session: str | Path | None = None) -> None:
        self.settings = get_settings()
        self.device = str(device)
        self.mode = "replay" if (mode == "replay" or replay_session) else "live"
        self.replay_session = str(replay_session) if replay_session else None
        self.estimator = estimator or StateEstimator(
            fs=float(self.settings.fs), device=self.device,
            replay_session=self.replay_session)
        self.detector = IntentDetector(fs=float(self.estimator.fs))
        self.seq = 0
        self.started_mono = clock()
        self.started_wall = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.source = None
        #: the device that is *actually* producing samples. Differs from
        #: ``self.device`` whenever the operator passed an alias ("auto"), so it
        #: is kept separate and propagated to the estimator through the setter
        #: below. Unresolved until the feeder opens a source.
        self._source_name = str(device)
        self.session_path: Path | None = Path(session) if session else None
        self.writer: SessionWriter | None = None
        self.events: deque[dict] = deque(maxlen=5000)
        self.n_samples_total = 0
        self.n_intents = {"blink": 0, "clench": 0}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_state: StateEstimate | None = None

        # --- external protocol §10: candidate signal + corrections ------------
        #: local, non-identifying participant id. One person at a desk; a study
        #: with real people must set this per participant from the session meta.
        self.participant_id = "local_participant_01"
        #: null until a model has actually been trained and saved. Never guessed.
        self.model_version: str | None = None
        #: every candidate ever broadcast, so POST /confirm can reference one by
        #: seq even if the agent missed the SSE frame. Bounded: this is a demo
        #: bridge, not the system of record - the durable copy is the session
        #: event log.
        self.candidates: deque[dict] = deque(maxlen=2000)
        self.candidate_seq = 0
        self.confirmations: deque[dict] = deque(maxlen=2000)
        self.idle_required_action_s = 3.0
        self.metrics_window_s = 600.0
        self.intent_history: deque[dict] = deque(maxlen=5000)
        self.signal_quality_counts: dict[str, int] = {
            "verified": 0, "coarse": 0, "unverified": 0}
        self.n_state_payloads = 0
        self.seq_gap_count = 0
        self.estimated_dropped_seq = 0
        self._last_state_dump: dict[str, float] = {}
        self._first_state_mono: float | None = None
        #: explicitly declared no-command block (see mark_idle_start/end)
        self.declared_idle_seconds: float | None = None
        self._idle_start_mono: float | None = None

    # ------------------------------------------------------------- lifecycle
    @property
    def source_name(self) -> str:
        """The source actually producing samples; /health reports this as ``device``."""
        return self._source_name

    @source_name.setter
    def source_name(self, value: str) -> None:
        """Record the resolved source and tell the estimator to publish it.

        The estimator never opens a device, so it cannot discover this itself,
        and ``source`` is the field an LLM agent reads to decide whether it is
        looking at a person. Propagating here (rather than at each call site)
        means the live path, the synthetic-fallback path and the replay path all
        stay truthful, including the replay label ``replay:<session>``.
        """
        self._source_name = str(value)
        est = getattr(self, "estimator", None)
        if est is not None:
            est.source_name = self._source_name

    def start(self) -> None:
        """Start the feeder thread. Never raises: a dead device is degraded."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        target = self._run_replay if self.mode == "replay" else self._run_live
        self._thread = threading.Thread(target=self._guarded(target),
                                        name="bridge-feeder", daemon=True)
        self._thread.start()

    def _guarded(self, target):
        def wrapper() -> None:
            try:
                target()
            except Exception as exc:                 # pragma: no cover - thread
                self.errors.append(f"feeder died: {exc}")
                self.warnings.append(traceback.format_exc(limit=6))
        return wrapper

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
            self.source = None

    # ------------------------------------------------------------- feeders
    def _open_source(self) -> Any:
        from ..device import open_source          # lazy: no hardware at import
        name = self.device
        if name in ("fake", "synthetic", "sim"):
            return open_source("fake", realtime=True)
        return open_source(name)

    def _run_live(self) -> None:
        """Poll the device forever, pushing samples into the estimator."""
        try:
            self.source = self._open_source()
            self.source.open()
            self.source_name = getattr(self.source, "name", self.device)
            fs = float(getattr(self.source, "fs", self.estimator.fs) or self.estimator.fs)
            if abs(fs - self.estimator.fs) > 1.0:
                self.warnings.append(f"device fs {fs} != estimator fs {self.estimator.fs}")
        except Exception as exc:
            self.errors.append(f"could not open {self.device}: {exc}; "
                               "switching to synthetic")
            from ..device import open_source
            self.source = open_source("fake", realtime=True)
            self.source.open()
            self.source_name = "fake"
        while not self._stop.is_set():
            # The electrode self-check arrives with the handshake, so it is read
            # on every pass rather than once after open(): until it lands the
            # source reports "unknown". Without this the state payload would say
            # "unknown" forever and the one field that tells the agent to ask the
            # wearer to reseat the band would never work.
            self.estimator.device_contact = str(
                getattr(self.source, "contact", "unknown") or "unknown")
            try:
                packets = self.source.poll(timeout_s=0.01)
            except Exception as exc:
                self.errors.append(f"poll failed: {exc}")
                time.sleep(0.05)
                continue
            if not packets:
                time.sleep(0.002)
                continue
            self._ingest_packets(packets)

    def _run_replay(self) -> None:
        """Replay a recorded session, or synthesise one when none exists.

        Replay is what lets the agent-facing contract be exercised and demoed
        without the headband on the desk, and it is the honest way to develop
        prompts: the *timing* of the stream is identical to live mode.
        """
        path = self._resolve_replay_path()
        speed = 1.0
        if path is None:
            self.warnings.append("no session to replay; using the synthetic source")
            self._run_live()
            return
        try:
            sess = Session.load(path)
        except Exception as exc:
            self.errors.append(f"cannot load {path}: {exc}; using synthetic source")
            self._run_live()
            return
        self.session_path = Path(path)
        self.source_name = f"replay:{Path(path).name}"
        v = np.asarray(sess.eeg, dtype=np.float64)
        t = np.asarray(sess.t, dtype=np.float64)
        if v.size == 0:
            self.warnings.append("session has no samples; using the synthetic source")
            self._run_live()
            return
        # re-base recorded timestamps onto this process's monotonic clock so
        # markers logged by the agent share one clock with the replayed samples
        t = t - t[0]
        fs = float(sess.fs or self.estimator.fs)
        chunk = max(1, int(round(fs * 0.1)))         # 100 ms slices
        i = 0
        wall0 = clock()
        while not self._stop.is_set():
            j = min(v.size, i + chunk)
            if i >= v.size:                          # loop the recording
                i = 0
                wall0 = clock()
                continue
            rel = t[i:j]
            target = wall0 + float(rel[-1]) / speed
            sl = target - clock()
            if sl > 0:
                time.sleep(min(sl, 0.2))
            self.estimator.push(v[i:j], rel + wall0)
            self._detect(v[i:j], rel + wall0)
            self.n_samples_total += j - i
            i = j

    def _resolve_replay_path(self) -> Path | None:
        if not self.replay_session:
            return None
        p = Path(self.replay_session)
        if p.is_dir() and (p / "meta.json").exists():
            return p
        cand = Path(SESSION_DIR) / self.replay_session
        if cand.is_dir() and (cand / "meta.json").exists():
            return cand
        sessions = list_sessions()
        return sessions[-1] if sessions else None

    def _ingest_packets(self, packets: Iterable[Packet]) -> None:
        pk = list(packets)
        self.estimator.push_packets(pk)
        for p in pk:
            vals = np.asarray(getattr(p, "values", []), dtype=np.float64).reshape(-1)
            ts = getattr(p, "t_samples", None)
            if ts is None:
                ts = float(getattr(p, "t_mono", clock()))
            self._detect(vals, np.asarray(ts, dtype=np.float64).reshape(-1))
            self.n_samples_total += int(vals.size)

    def _detect(self, values: np.ndarray, t: np.ndarray) -> None:
        if values.size == 0:
            return
        try:
            intents = self.detector.push(values, t)
        except Exception as exc:                     # detection must never kill the feed
            self.errors.append(f"intent detector: {exc}")
            return
        for it in intents:
            self.add_event({"type": "intent", "name": it.get("intent", "?"),
                            "t_mono": float(it.get("t", 0.0)), "meta": it,
                            "source": "detector"})
            # the metrics endpoint must use the same detections the stream does
            self.intent_history.append({"intent": it.get("intent", "?"),
                                        "t": float(it.get("t", 0.0)),
                                        "confidence": float(it.get("confidence", 0.0))})
            self.n_intents[it.get("intent", "")] = self.n_intents.get(it.get("intent", ""), 0) + 1
            self.estimator.note_intents([it])

    # ------------------------------------------------------------- events
    def add_event(self, ev: dict[str, Any]) -> dict[str, Any]:
        """Append to the in-memory event ring, forcing a monotonic timestamp."""
        ev = dict(ev)
        ev.setdefault("t_mono", clock())
        ev["t_mono"] = float(ev["t_mono"])
        ev.setdefault("t_wall", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self._lock:
            self.events.append(ev)
        return ev

    def mark(self, name: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record an agent-side marker into the session log (or the fallback).

        WHY a fallback file: a session is only active while the runner is
        recording; an agent that talks to the bridge outside a session still
        wants its actions anchored in time. Those markers go to
        ``data/sessions/_agent_events.jsonl`` and are reported alongside the real
        ones, with ``"session": null`` so nobody confuses the two.
        """
        t = clock()
        meta = dict(meta or {})
        ev = {"type": "mark", "name": str(name), "t_mono": t, "meta": meta,
              "source": "agent", "session": None}
        writer = self.writer
        if writer is not None:
            try:
                writer.event(0, name=str(name), t_mono=t,
                             meta={"agent": True, **meta})
                ev["session"] = writer.session_id
            except Exception as exc:
                self.errors.append(f"session writer rejected mark: {exc}")
                self._append_fallback(ev)
        elif self.session_path is not None and (self.session_path / "events.jsonl").exists():
            try:
                self._append_jsonl(self.session_path / "events.jsonl", {
                    "code": 0, "name": f"agent:{name}", "t_mono": t,
                    "test_id": None, "trial": None, "block": None, "phase": None,
                    "meta": {"agent": True, **meta}, "t_wall": ev["t_wall"]})
                ev["session"] = self.session_path.name
            except Exception as exc:
                self.errors.append(f"session log rejected mark: {exc}")
                self._append_fallback(ev)
        else:
            self._append_fallback(ev)
        with self._lock:
            self.events.append(ev)
        return ev

    def _append_fallback(self, ev: dict[str, Any]) -> None:
        """Append an event to the non-session agent/participant log.

        The label-policy fields are written explicitly (null when unknown) so a
        reader of ``_agent_events.jsonl`` sees the same three-label shape as a
        session's ``events.jsonl`` and never has to guess what was omitted.
        """
        try:
            p = Path(SESSION_DIR) / "_agent_events.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = dict(ev)
            for key in ("task_label", "reported_label", "predicted_label",
                        "timing_quality"):
                rec.setdefault(key, None)
            rec.setdefault("test_id", None)
            rec.setdefault("trial", None)
            rec.setdefault("block", None)
            rec.setdefault("phase", None)
            rec.setdefault("t_wall", time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._append_jsonl(p, rec)
            ev["file"] = str(p)
        except Exception as exc:                     # pragma: no cover - disk
            self.errors.append(f"fallback event log failed: {exc}")

    @staticmethod
    def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
        with Path(path).open("a", buffering=1) as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    def recent_events(self, since: float | None = None, limit: int = 100) -> list[dict]:
        """Recorded markers + detected intents + agent marks, merged and sorted."""
        out: list[dict] = []
        with self._lock:
            out += [dict(e) for e in self.events]
        out += self._disk_events()
        if since is not None:
            out = [e for e in out if float(e.get("t_mono", 0.0)) >= float(since)]
        out.sort(key=lambda e: float(e.get("t_mono", 0.0)))
        return out[-int(max(1, limit)):]

    def _disk_events(self) -> list[dict]:
        """Markers written by the runner/replay session, read from events.jsonl.

        Read fresh on each call (the file is small and append-only) so a running
        acquisition is visible to the agent without sharing memory with it.
        """
        path = None
        if self.session_path is not None:
            path = Path(self.session_path) / "events.jsonl"
        else:
            sessions = list_sessions()
            if sessions:
                path = sessions[-1] / "events.jsonl"
        out: list[dict] = []
        if path and path.exists():
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    d.setdefault("type", "marker")
                    out.append(d)
            except Exception as exc:
                self.errors.append(f"cannot read {path}: {exc}")
        # Always merge the fallback agent/participant log, even while a session
        # is active: agent marks and participant corrections made outside a
        # recording are still part of the timeline the agent must see.
        fb = Path(SESSION_DIR) / "_agent_events.jsonl"
        if fb.exists():
            try:
                for line in fb.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
            except Exception as exc:
                self.errors.append(f"cannot read {fb}: {exc}")
        # a mark may be present in both the session log and the fallback file
        # (session started after the mark); keep the first occurrence
        seen: set[tuple[float, str]] = set()
        uniq: list[dict] = []
        for e in out:
            key = (round(float(e.get("t_mono", 0.0)), 3), str(e.get("name") or ""))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        return uniq

    # ------------------------------------------------------------- snapshots
    def state_payload(self) -> dict[str, Any]:
        """Next sequenced state payload (also caches the raw estimate).

        A ``candidate`` block is attached to every payload because the external
        protocol's §10 contract is about the *state* the agent consumes, not an
        extra endpoint it might forget to call. The candidate is subject to the
        same seq counter, so a confirmation can reference exactly the frame the
        agent saw.
        """
        est = self.estimator.estimate()
        with self._lock:
            self.seq += 1
            seq = self.seq
            self._last_state = est
            self.n_state_payloads += 1
            if self._first_state_mono is None:
                self._first_state_mono = clock()
            cand = self._build_candidate(est, seq)
            self.candidate_seq = seq
            self.candidates.append(cand)
            self._track_quality(cand["signal_quality"])
        # Log the prediction BEFORE it can be corrected. This happens here, not
        # in the correction handler, because the protocol requires the prediction
        # to be durable first; a correction must never be the only record.
        self._log_candidate_event(cand)
        payload = state_to_agent_json(est, seq=seq)
        payload["candidate"] = cand
        return payload

    # ------------------------------------------- external protocol §10: candidate
    def current_candidate(self) -> dict[str, Any]:
        """The candidate for the latest state, forcing one if none exists yet."""
        with self._lock:
            if self.candidates:
                return dict(self.candidates[-1])
        return self.state_payload()["candidate"]

    def _signal_quality(self, est: StateEstimate) -> str:
        """Map the estimator's own quality verdict onto the §10 vocabulary.

        ``verified`` is reserved for a capture that passed the device self-check
        *and* shows no saturation or motion artefact **and** came from the real
        headband. The check is written as "is this the real device?" rather than
        "is this the fake one?" on purpose: it fails closed. A synthetic capture,
        an unresolved alias (``auto``), a replay and an unknown source all land on
        ``coarse``, because none of them is evidence about a person. A railed or
        off-head capture is ``unverified`` because nothing from it can be trusted.
        """
        info = dict(est.model_info or {})
        q = info.get("quality") or {}
        verdict = str(q.get("verdict") or "")
        if est.source != "th21a":
            return "coarse"
        if verdict in ("POOR",) or est.confidence < 0.2:
            return "unverified"
        if verdict in ("GOOD", "FAIR") and est.contact.startswith("good"):
            return "verified"
        return "coarse"

    def _window_acquisition_times(self, est: StateEstimate) -> tuple[float | None, float | None]:
        """(start, end) of the estimate's window on the session clock, or (None, None).

        The values are seconds relative to the first sample the estimator buffered
        (the *acquisition* clock), not wall time, and not a stimulus onset. When
        the buffer is too short to define a window they stay null rather than
        being approximated -- "missing is not neutral".
        """
        try:
            t0 = float(self.estimator._ring.t_earliest)  # noqa: SLF001 - same package
            t1 = float(self.estimator._ring.t_latest)
        except Exception:
            return None, None
        if not (t1 > t0):
            return None, None
        return round(t0 - self.started_mono, 3), round(t1 - self.started_mono, 3)

    # ------------------------------------------------------- raw sample access
    def samples_since(self, t_mono: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """Raw samples newer than ``t_mono``, plus whether a gap was skipped.

        The bridge owns the BLE link, so a second process cannot open the headband
        for itself. This cursor is how an attached recorder or display still gets
        the raw counts rather than a lossy summary.

        ``gap`` is True when ``t_mono`` predates the oldest buffered sample: the
        caller fell behind (or the bridge restarted) and some samples can no
        longer be served. Saying so explicitly is the difference between a log
        with a documented hole and a log that silently splices two moments
        together.
        """
        ring = self.estimator._ring                      # noqa: SLF001 - same package
        with self.estimator._lock:                       # noqa: SLF001
            if ring.size == 0:
                z = np.zeros(0, dtype=np.float64)
                return z, z.copy(), False
            earliest = float(ring.t_earliest)
            gap = float(t_mono) < earliest - 1e-9
            lo = earliest if gap else float(t_mono)
            v, t = ring.window_between(lo, float(ring.t_latest))
        keep = t > float(t_mono)                         # window_between is inclusive
        return v[keep], t[keep], gap

    def sample_cursor(self) -> float:
        """Latest sample time, i.e. 'start from here' for a fresh subscriber."""
        return float(self.estimator._ring.t_latest)      # noqa: SLF001

    def _decision_rule(self, field: str) -> tuple[str, float, str] | None:
        """The validated decision rule for a field, if one exists.

        Returns ``(label_high, threshold, version)``. **Today this always returns
        ``None``**: a rule only exists once a model has been trained, evaluated
        and a threshold chosen on independent data. Until then every candidate is
        ``uncertain``, which is the honest answer, not a placeholder.
        """
        return None

    def _build_candidate(self, est: StateEstimate, seq: int) -> dict[str, Any]:
        """The §10 candidate signal for one state estimate (caller holds the lock).

        The rules are deliberately strict and are the whole point of the shape:

        * ``candidate_label`` is ``"uncertain"`` whenever the signal-quality gate,
          the confidence gate, or a *validated* decision rule is missing. It is
          never the argmax of an unvalidated model.
        * ``class_probabilities`` is ``null`` unless a model produced them, and
          ``probabilities_calibrated`` is ``false`` until a real calibration step
          ran. A softmax over an uncalibrated LDA is not a calibrated
          probability, so it must not be presented as one.
        * ``model_version`` is ``null`` until a model is trained.
        * ``participant_confirmation`` is filled only by POST /confirm, never by
          the model.
        """
        info = dict(est.model_info or {})
        t0, t1 = self._window_acquisition_times(est)
        quality = self._signal_quality(est)
        trained = list(info.get("trained_models") or [])
        model_version = self.model_version
        if model_version is None and trained:
            # a model exists but has not been promoted to a versioned release
            model_version = None
        rule = self._decision_rule("intent")
        conf = float(est.confidence)

        # probabilities exist only when a trained model drove a field; we do not
        # invent a probability for a heuristic percentile.
        class_probabilities: dict[str, float] | None = None
        derivations = {f: v.get("derivation") for f, v in (info.get("fields") or {}).items()}
        if any(d == "trained_model" for d in derivations.values()):
            # the estimator exposes the trained fields, but mapping several
            # 0..1 fields onto a class-probability vector would be a fabrication
            class_probabilities = None

        candidate_label = "uncertain"
        reason = []
        if quality == "unverified":
            reason.append("signal_quality=unverified")
        if conf < 0.4:
            reason.append(f"confidence={conf:.2f} < 0.4")
        if rule is None:
            reason.append("no validated decision rule")
        if not trained:
            reason.append("no trained model")

        confirmed = None
        for c in reversed(self.confirmations):
            if int(c.get("candidate_seq", -1)) == int(seq):
                confirmed = c.get("participant_response")
                break

        return {
            "schema_version": "1.0",
            "participant_id": self.participant_id,
            "model_version": model_version,
            "scope": "intentional_yes_no_rest",
            "window_start_acquisition_s": t0,
            "window_end_acquisition_s": t1,
            "signal_quality": quality,
            "candidate_label": candidate_label,
            "class_probabilities": class_probabilities,
            "probabilities_calibrated": False,
            "decision_rule_version": None,
            "recording_context_supported": False,
            "participant_confirmation": confirmed,
            "uncertain_reasons": reason,
            "evidence": {
                "state_seq": int(seq),
                "confidence": round(conf, 3),
                "derivation_by_field": derivations,
                "source": est.source,
                "timing_quality": "unverified",
                "note": ("A candidate label is a possible intended signal, not a "
                         "fact, not consent, and not a command."),
            },
        }

    def _track_quality(self, quality: str) -> None:
        self.signal_quality_counts[quality] = self.signal_quality_counts.get(quality, 0) + 1

    def confirm(self, candidate_seq: int, participant_response: str,
                note: str = "") -> dict[str, Any]:
        """Record the participant's correction of a candidate, without rewriting it.

        Protocol §10 requires the prediction to be logged **before** the
        correction and the correction to be stored **separately**. This bridge
        therefore:

        1. finds the already-broadcast candidate (durably written to the session
           event log by :meth:`state_payload` -> :meth:`_log_candidate_event`),
        2. writes one immutable ``reported_label`` event carrying the
           participant's response, and
        3. leaves the original ``predicted_label`` event untouched.

        The three labels land in their own fields (``task_label`` didn't apply
        here, ``predicted_label`` was the candidate, ``reported_label`` is the
        correction) rather than inside ``meta``, exactly as the store's label
        policy requires.
        """
        allowed = ("yes", "no", "skipped", "unsure")
        resp = str(participant_response or "").strip().lower()
        if resp not in allowed:
            return {"ok": False, "error": "bad_response",
                    "detail": f"participant_response must be one of {allowed}",
                    "got": participant_response}
        with self._lock:
            cand = None
            for c in self.candidates:
                if int(c.get("evidence", {}).get("state_seq", -1)) == int(candidate_seq):
                    cand = dict(c)
                    break
            if cand is None and self.candidates and int(candidate_seq) == int(self.candidate_seq):
                cand = dict(self.candidates[-1])
        if cand is None:
            return {"ok": False, "error": "unknown_candidate_seq",
                    "detail": (f"candidate_seq {candidate_seq} is not in this bridge's "
                               "recent candidate history (bounded to 2000); the "
                               "prediction could not have been made, so no "
                               "correction is stored against it"),
                    "oldest_available_seq": (int(self.candidates[0]["evidence"]["state_seq"])
                                             if self.candidates else None)}
        ev = self._write_correction_event(cand, resp, note)
        correction = {
            "type": "participant_correction",
            "candidate_seq": int(candidate_seq),
            "participant_response": resp,
            "candidate_label": cand.get("candidate_label"),
            "note": str(note or ""),
            "t_mono": ev.get("t_mono"),
            "t_wall": ev.get("t_wall"),
            "file": ev.get("file"),
            "session": ev.get("session"),
            "policy": ("The prediction was logged before this correction and is not "
                       "modified. Corrections are reviewed offline; a production "
                       "model is not retrained per correction."),
        }
        with self._lock:
            self.confirmations.append(correction)
        return {"ok": True, "correction": correction}

    def _write_correction_event(self, cand: dict[str, Any], response: str,
                                note: str) -> dict[str, Any]:
        """Append the participant's report as its own event, never as an edit."""
        t = clock()
        meta = {
            "agent": False,
            "participant": True,
            "candidate_seq": int(cand.get("evidence", {}).get("state_seq", -1)),
            "candidate_schema_version": cand.get("schema_version"),
            "candidate_label_at_time": cand.get("candidate_label"),
            "signal_quality": cand.get("signal_quality"),
            "model_version": cand.get("model_version"),
            "probabilities_calibrated": cand.get("probabilities_calibrated"),
            "participant_confirmation": response,
            "note": str(note or ""),
        }
        # use the store's label policy fields where a writer is available
        writer = self.writer
        path = None
        if writer is not None:
            try:
                writer.event("participant_confirm", name="participant_confirm",
                             t_mono=t, reported_label=response, meta=meta)
                path = str(writer.path / "events.jsonl")
            except Exception as exc:
                self.errors.append(f"session writer rejected correction: {exc}")
        if path is None:
            target = (self.session_path / "events.jsonl") if self.session_path else None
            if target is not None and target.exists():
                try:
                    self._append_jsonl(target, {
                        "code": 0, "name": "participant_confirm", "t_mono": t,
                        "t_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "test_id": None, "trial": None, "block": None, "phase": None,
                        "task_label": None, "predicted_label": cand.get("candidate_label"),
                        "reported_label": response, "timing_quality": None,
                        "meta": meta})
                    path = str(target)
                except Exception as exc:
                    self.errors.append(f"session log rejected correction: {exc}")
        rec = {"type": "participant_correction", "name": "participant_confirm",
               "t_mono": t, "t_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "task_label": None, "predicted_label": cand.get("candidate_label"),
               "reported_label": response, "meta": meta, "session": None}
        self._append_fallback(rec)                   # always visible via /events
        if path is None:
            path = str(Path(SESSION_DIR) / "_agent_events.jsonl")
        with self._lock:                             # live view without a disk read
            self.events.append(dict(rec))
        return {"t_mono": t, "t_wall": rec["t_wall"],
                "file": path, "session": (self.session_path.name if self.session_path
                                          else None)}

    def _log_candidate_event(self, cand: dict[str, Any]) -> None:
        """Durably log the prediction *before* any correction can arrive.

        Called while the candidate is created, i.e. before the agent has even
        seen the SSE frame. This ordering is the point: if the participant
        corrects a candidate, a prediction event must already exist on disk to
        correct.
        """
        rec = {
            "type": "candidate",
            "name": "candidate_signal",
            "t_mono": clock(),
            "t_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_label": None,
            "predicted_label": cand.get("candidate_label"),
            "reported_label": None,
            "meta": {
                "candidate_seq": int(cand.get("evidence", {}).get("state_seq", -1)),
                "schema_version": cand.get("schema_version"),
                "signal_quality": cand.get("signal_quality"),
                "model_version": cand.get("model_version"),
                "probabilities_calibrated": cand.get("probabilities_calibrated"),
                "decision_rule_version": cand.get("decision_rule_version"),
                "recording_context_supported": cand.get("recording_context_supported"),
                "uncertain_reasons": cand.get("uncertain_reasons", []),
                "scope": cand.get("scope"),
                "timing_quality": "unverified",
            },
        }
        try:
            if self.writer is not None:
                self.writer.event("candidate_signal", name="candidate_signal",
                                  t_mono=rec["t_mono"],
                                  predicted_label=cand.get("candidate_label"),
                                  meta=rec["meta"])
            elif self.session_path is not None:
                self._append_jsonl(self.session_path / "events.jsonl", rec)
            # the fallback log is written in every case so an agent that only
            # reads /events can always see the prediction that was made
            self._append_fallback(rec)
        except Exception as exc:                     # pragma: no cover - disk
            self.errors.append(f"could not log candidate: {exc}")

    # ------------------------------------------------------------- calibration
    def calibrate(self, force: bool = False) -> dict[str, Any]:
        """Capture the personal baseline on demand and report the provenance.

        WHY this is manual: every 0..1 field is relative to this baseline, so the
        moment it is captured defines the reference for the whole session. An
        automatic capture at 60 s would freeze whatever the wearer happened to be
        doing - and, as the first live run showed, a drifting electrode produces a
        baseline that makes every later field meaningless. The operator seats the
        band, waits, and asks.

        The baseline is persisted per session id, so a bridge restart mid-session
        adopts it rather than silently reverting the fields to un-baselined 0.5.
        """
        with self._lock:
            source = self._source_name
        prov = Provenance(device=source, requested_device=self.device, mode=self.mode)
        try:
            res = dict(self.estimator.calibrate(force=bool(force)))
        except Exception as exc:
            res = {"ok": False, "error": type(exc).__name__, "reason": str(exc)}
        res["calibrated"] = bool(self.estimator.calibrated)
        res["provenance"] = prov.as_dict()
        res["is_live_human"] = bool(prov.is_live_human)
        if not prov.is_live_human:
            res["warning"] = ("the source is not a live headband, so this baseline "
                              "does not describe a person")
        return res

    # ------------------------------------------------------------- metrics
    #: tests whose intent classes are a cued/requested action; an intent detected
    #: within ``idle_required_action_s`` of one of these markers is NOT idle.
    REQUESTED_ACTION_MARKERS = (
        "stim_onset", "cue_onset", "response_prompt", "question", "probe",
        "answer_yes", "answer_no", "trial_onset",
    )

    def metrics(self, window_s: float | None = None,
                idle_window_s: float | None = None,
                include_events: bool = True) -> dict[str, Any]:
        """False-activation and stream-health metrics for the external protocol.

        The idle false-activation rate is the honest cost of a continuous
        decoder, so it is computed from the *same* ``IntentDetector`` instance
        the live stream uses -- same smoothing, same refractory -- rather than
        from a re-implementation with different constants. An intent is counted
        as idle when no requested-action marker (``stim_onset``, ``cue_onset``,
        ``response_prompt``, ...) occurred within ``idle_required_action_s``
        before it.
        """
        now = clock()
        window_s = float(self.metrics_window_s if window_s is None else window_s)
        wn = max(1.0, now - window_s)
        intents = [i for i in list(self.intent_history) if float(i.get("t", 0.0)) >= wn]
        n = len(intents)
        stream_min = max(1e-9, (now - self.started_mono) / 60.0)

        # /health calls this on every poll, so the (disk-backed) event scan is
        # skippable; the resulting rates then cover only in-memory markers.
        action_times = self._requested_action_times() if include_events else []
        idle_cut = float(self.idle_required_action_s)
        idle = [i for i in intents
                if not any(0.0 <= float(i["t"]) - a <= idle_cut
                           for a in action_times if a <= float(i["t"]))]
        # idle exposure: seconds of this session that are NOT inside a
        # requested-action window, so the denominator matches the numerator
        marker_windows = [(a, a + idle_cut) for a in action_times]
        active_s = _union_length(marker_windows, self.started_mono, now)
        idle_s = max(0.0, (now - self.started_mono) - active_s)
        per_min = (len(intents) / stream_min) if stream_min > 0 else 0.0
        # A rate needs exposure. Below the minimum idle time the honest answer is
        # "not measurable yet", not a number computed from 1.2 s of data that can
        # read as millions per minute. The external protocol wants this measured
        # over a real no-command recording (minutes, not seconds).
        idle_min_s = 60.0
        idle_rate = (round(len(idle) / (idle_s / 60.0), 3) if idle_s >= idle_min_s
                     else None)

        with self._lock:
            by_kind: dict[str, int] = {}
            for i in intents:
                k = str(i.get("intent", "?"))
                by_kind[k] = by_kind.get(k, 0) + 1

        return {
            "ok": True,
            "window_s": round(window_s, 1),
            "uptime_s": round(now - self.started_mono, 1),
            "stream_minutes": round(stream_min, 3),
            "intent_events": n,
            "intent_events_by_kind": by_kind,
            "intent_events_per_min": round(per_min, 3),
            "idle_required_action_s": idle_cut,
            "idle_seconds": round(idle_s, 1),
            "idle_intent_events": len(idle),
            # null (not 0) until there is enough idle exposure to divide by
            "idle_false_activations_per_min": idle_rate,
            "idle_rate_measurable": bool(idle_rate is not None),
            "idle_rate_min_exposure_s": idle_min_s,
            "idle_rate_note": ("null until at least %.0f s of no-command (idle) "
                               "recording exist; a rate over seconds is noise"
                               % idle_min_s),
            "detector": {
                "class": "IntentDetector",
                "blink_sigma": float(self.detector.blink_sigma),
                "clench_band": list(self.detector.clench_band),
                "refractory_s": float(self.detector.refractory_s),
                "min_baseline_s": float(self.detector.min_baseline_s),
                "edge_pad_s": round(float(self.detector._edge_pad) / float(self.detector.fs), 3),
                "note": ("Computed from the live detector instance, so the "
                         "smoothing/refractory here are the deployment rules."),
                "counts": self.detector.counts,
            },
            "state_updates": int(self.n_state_payloads),
            "seq": int(self.seq),
            "dropped_seq": int(self.estimated_dropped_seq),
            "seq_gap_events": int(self.seq_gap_count),
            "signal_quality_counts": dict(self.signal_quality_counts),
            "candidates": int(self.candidate_seq),
            "confirmations": len(self.confirmations),
            "n_samples": int(self.n_samples_total),
            "declared_idle": (self.declared_idle_seconds is not None),
            "declared_idle_seconds": (round(self.declared_idle_seconds, 1)
                                      if self.declared_idle_seconds is not None else None),
            "honesty": ("A false-activation rate is only meaningful over a real "
                        "no-command recording; a synthetic source understates it."),
        }

    def _requested_action_times(self) -> list[float]:
        """Timestamps of markers that requested a participant action."""
        out: list[float] = []
        names = set(self.REQUESTED_ACTION_MARKERS)
        for ev in self.recent_events(limit=5000):
            nm = str(ev.get("name") or "").split(":")[-1]
            if nm in names or ev.get("task_label"):
                try:
                    out.append(float(ev.get("t_mono", 0.0)))
                except Exception:
                    continue
        return sorted(out)

    def mark_idle_start(self) -> float:
        """Mark the start of an explicitly declared no-command (idle) block."""
        self._idle_start_mono = clock()
        self.declared_idle_seconds = 0.0
        return self._idle_start_mono

    def mark_idle_end(self) -> float:
        """Close an explicitly declared idle block; returns its duration."""
        if self._idle_start_mono is None:
            return 0.0
        self.declared_idle_seconds = max(0.0, clock() - self._idle_start_mono)
        return float(self.declared_idle_seconds)

    def status(self) -> dict[str, Any]:
        est = self._last_state or self.estimator.estimate()
        sessions = list_sessions()
        on_disk = sorted(p.stem for p in Path(MODEL_DIR).glob("*.joblib")) \
            if Path(MODEL_DIR).exists() else []
        info = dict(est.model_info or {})
        # `models` counts files on disk; `trained_models` counts the ones that
        # actually loaded and are therefore driving a field right now. A corrupt
        # file must not be reported as a trained model.
        loaded = sorted(set(info.get("trained_models") or []))
        return {
            "ok": True,
            "mode": self.mode,
            "device": self.source_name,
            "requested_device": self.device,
            "fs_nominal": float(self.estimator.fs),
            "fs_effective": round(float(self.estimator.effective_fs), 3),
            "sessions": len(sessions),
            "models": len(on_disk),
            "models_on_disk": on_disk,
            "trained_models": loaded,
            "model_load_errors": list(info.get("errors") or []),
            "field_derivations": {f: v.get("derivation")
                                  for f, v in (info.get("fields") or {}).items()},
            "estimator_ready": bool(self.estimator.ready),
            "calibrated": bool(self.estimator.calibrated),
            "calibration_confidence_ok": bool(est.confidence >= 0.4),
            "confidence": round(float(est.confidence), 3),
            "n_samples": int(self.n_samples_total),
            "n_intents": dict(self.n_intents),
            "seq": int(self.seq),
            "uptime_s": round(clock() - self.started_mono, 1),
            "started_wall": self.started_wall,
            "session": (self.session_path.name if self.session_path else None),
            "participant_id": self.participant_id,
            "model_version": self.model_version,
            "candidate_seq": int(self.candidate_seq),
            "confirmations": len(self.confirmations),
            "signal_quality_counts": dict(self.signal_quality_counts),
            "state_updates": int(self.n_state_payloads),
            "dropped_seq": int(self.estimated_dropped_seq),
            "idle_false_activations_per_min": self.metrics(include_events=False)[
                "idle_false_activations_per_min"],
            "version": __version__,
            "warnings": list(self.warnings)[-5:],
            "errors": list(self.errors)[-5:],
            "device_info": (self.source.info if self.source is not None
                            and hasattr(self.source, "info") else {}),
            "honest_limits": (
                "One dry frontal channel, uncalibrated counts: arousal, gross "
                "attention, load proxy, fatigue, relaxation, blink/clench commands. "
                "No words, sentences, images, intentions or valence."
            ),
        }

    def schema(self) -> dict[str, Any]:
        """The agent-facing contract: fields, provenance, cadence and tools."""
        est = self._last_state or self.estimator.estimate()
        info = dict(est.model_info or {})
        derivations = {f: v.get("derivation") for f, v in (info.get("fields") or {}).items()}
        fields: dict[str, Any] = {}
        for name, doc in FIELD_DOCS.items():
            d = dict(doc)
            if name in derivations:
                d["derivation"] = derivations[name]
                d["derivation_note"] = _derivation_note(derivations[name])
            fields[name] = d
        return {
            "version": __version__,
            "name": "NeuroBridge agent bridge",
            "device": {
                "model": "NingNao/SereniBrain TH21A",
                "montage": "one dry frontal electrode (Fpz-ish, above the eyebrows)",
                "fs_hz": float(self.estimator.fs),
                "units": "uncalibrated 24-bit signed ADC counts",
                "known_failure_modes": [
                    "rails at the 24-bit limit when the band is off the head",
                    "slow drift over the session (removed below 1 Hz)",
                    "motion artefacts from head movement and cable tug",
                    "mains hum at 50/60 Hz (notched, but grounding matters)",
                ],
            },
            "cadence": {
                "state_hz": float(self.settings.state_hz),
                "windows_s": float(self.estimator.window_s),
                "update_step_s": float(self.estimator.step_s),
                "client_poll_hz": float(self.settings.state_hz),
                "staleness_timeout_s": 3.0,
                "note": ("Fields average a ~4 s window, so single 0.5 s samples are "
                         "correlated. Prefer /state/stream and act on trends over "
                         ">=2 s, never on one sample."),
            },
            "fields": fields,
            "intents": {
                "blink": {
                    "meaning": "confirm / select / yes",
                    "method": "0.1-5 Hz monophasic deflection > median + 6*1.4826*MAD",
                    "reliability": "high (blink is ~100x cortical amplitude at Fpz)",
                    "action_hint": "Act only at confidence >= 0.6; confirm after acting.",
                },
                "clench": {
                    "meaning": "cancel / escape / no",
                    "method": "20-45 Hz EMG envelope > median + 5*1.4826*MAD, 0.2-1.5 s",
                    "reliability": "high, but jaw tension also happens while talking/eating",
                    "action_hint": "Never bind clench to a destructive action.",
                },
                "unsupported": [
                    "word, sentence or image reading",
                    "intention or imagery decoding",
                    "valence / emotion from frontal asymmetry (needs 2+ channels)",
                    "multi-class mental typing",
                ],
            },
            "provenance": {
                "calibrated": bool(info.get("calibrated")),
                "calibration_path": info.get("calibration_path"),
                "trained_models": info.get("trained_models", []),
                "derivation_by_field": derivations,
                "heuristic_definition": ("Robust z-score of each metric against the "
                                         "first ~60 s baseline (median + MAD), squashed "
                                         "to 0..1. Monotone, not calibrated."),
                "disclosure": info.get("honesty", ""),
                "valence_note": info.get("valence_note", ""),
                "errors": info.get("errors", []),
                "warnings": info.get("warnings", []),
            },
            "ground_rules": [
                "confidence < 0.4 -> do not act on state; ask the human a question instead.",
                "confidence < 0.2 -> the capture is saturated or off-head: tell the human "
                "to reseat the band and ignore all fields until it recovers.",
                "blink == confirm, clench == cancel; both need confidence >= 0.6.",
                "Never claim to read thoughts, words or intentions from this signal.",
                "Never present synthetic ('source': 'fake') state as the human's own.",
                "POST /mark around every user-visible action so the timeline can be "
                "analysed later.",
            ],
            "routes": {
                "GET /health": "mode, device, fs, sessions, models, readiness",
                "GET /state": "one current state payload",
                "GET /state/stream": "SSE, one event per update at state_hz",
                "GET /samples/stream": ("SSE raw EEG counts for an attached recorder or "
                                        "display: ?max_hz=&chunk_ms=&from_now=. Frames carry "
                                        "t0, fs and values. max_hz decimates for drawing "
                                        "and does NOT anti-alias - record at full rate."),
                "POST /calibrate": ("capture the personal baseline now; 200 once it "
                                    "exists, 409 {'reason':'not_enough_data'} while "
                                    "collecting. Nothing else establishes one."),
                "WS /ws": "same payloads; inbound {'type':'mark'|'ping'}",
                "GET /events": "?since=<t_mono>&limit=N markers + intents, sorted",
                "POST /mark": "{'name': str, 'meta': {}} agent marker into the EEG timeline",
                "GET /candidate": "external-protocol §10 candidate signal (uncertain until validated)",
                "POST /confirm": "{'candidate_seq':N,'participant_response':'yes|no|skipped|unsure'} participant correction",
                "GET /metrics": "false activations/min, idle false activations/min, stream health",
                "GET /report/{test_id}": "latest evaluation JSON for a test, if any",
                "GET /schema": "this document",
            },
            "candidate_contract": {
                "schema_version": "1.0",
                "scope": "intentional_yes_no_rest",
                "routes": {"GET /candidate": "current candidate",
                           "POST /confirm": "log the participant's correction separately",
                           "field": "every /state and /state/stream payload carries `candidate`"},
                "rules": [
                    "candidate_label is 'uncertain' whenever signal quality, confidence "
                    "or a validated decision rule is missing; the most likely class is "
                    "never substituted.",
                    "class_probabilities is null unless a model produced them; "
                    "probabilities_calibrated stays false until a real calibration step ran.",
                    "model_version is null until a model is trained.",
                    "A candidate 'yes' is NOT consent and never authorises a consequential "
                    "action; only a reversible comfort action (e.g. pause) may follow.",
                    "The prediction is logged before any correction; the correction is "
                    "stored separately and never overwrites it.",
                ],
                "labels": ["yes", "no", "uncertain"],
                "unknown_is_not_neutral": True,
            },
            "metrics_endpoint": "GET /metrics (intent_events_per_min, "
                                "idle_false_activations_per_min, state_updates, "
                                "dropped_seq, uptime_s, signal_quality_counts)",
            "tools": TOOL_DEFS,
            "curl_examples": _curl_examples(self.settings.bridge_host,
                                            self.settings.bridge_port),
        }


def _union_length(spans: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Total seconds covered by ``spans`` inside [lo, hi], counted once.

    Used to compute idle exposure: overlapping requested-action windows (a cue
    followed by a prompt 2 s later) must not be double-counted, or the idle
    denominator would be too small and the false-activation rate too high.
    """
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in spans if b > lo and a < hi)
    total = 0.0
    cur_start = cur_end = None
    for a, b in clipped:
        if b <= a:
            continue
        if cur_end is None:
            cur_start, cur_end = a, b
        elif a <= cur_end:
            cur_end = max(cur_end, b)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = a, b
    if cur_end is not None:
        total += cur_end - cur_start
    return float(total)


def _derivation_note(derivation: str | None) -> str:
    if derivation == "trained_model":
        return ("A decoder trained on this subject's recorded sessions drives this "
                "field (probability of the 'high' class). Still single-channel EEG; "
                "check /health for the model id and the test's cross-validated score.")
    if derivation == "heuristic_baseline":
        return ("HEURISTIC: a robust z-score of a band-power ratio against the first "
                "~60 s baseline, squashed to 0..1. Monotone and relative; not a "
                "measurement and not validated against any ground truth yet.")
    if derivation == "uncalibrated_neutral":
        return ("No baseline captured yet, so this field is pinned at 0.5 and carries "
                "no information. Wait for calibration (about 60 s) before using it.")
    return "Not yet computed."


def _curl_examples(host: str, port: int) -> dict[str, str]:
    base = f"http://{host}:{port}"
    return {
        "health": f"curl -s {base}/health | python -m json.tool",
        "state": f"curl -s {base}/state | python -m json.tool",
        "stream": f"curl -N {base}/state/stream",
        "mark": ("curl -s -X POST {0}/mark -H 'content-type: application/json' "
                 "-d '{{\"name\":\"agent_paused\",\"meta\":{{\"reason\":\"user asked\"}}}}'"
                 ).format(base),
        "events": f"curl -s '{base}/events?limit=20' | python -m json.tool",
        "schema": f"curl -s {base}/schema | python -m json.tool",
        "report": f"curl -s {base}/report/T201 | python -m json.tool",
    }


# ----------------------------------------------------------------------- app
def create_app(estimator: StateEstimator | None = None, device: str = "auto",
               mode: str = "live", replay_session: str | Path | None = None,
               session: str | Path | None = None):
    """Build the FastAPI app. ``mode`` is ``live`` or ``replay``.

    Importing fastapi/uvicorn only happens here, not in ``state.py``, so the
    estimator stays usable (and testable) with no web stack at all.
    """
    from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse

    bs = BridgeState(estimator=estimator, device=device, mode=mode,
                     replay_session=replay_session, session=session)
    app = FastAPI(
        title="NeuroBridge agent bridge",
        version=__version__,
        description=("Honest single-channel frontal-EEG state for an LLM agent. "
                     "Not a thought decoder; see /schema."),
    )
    app.state.bridge = bs

    # ---------------------------------------------------------- lifecycle
    @app.on_event("startup")
    async def _startup() -> None:
        bs.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        bs.stop()

    # ---------------------------------------------------------- error shapes
    @app.exception_handler(Exception)
    async def _any_error(request: Request, exc: Exception) -> JSONResponse:
        """Never an HTML error page: always JSON, always with the traceback."""
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": type(exc).__name__, "detail": str(exc),
                     "traceback": traceback.format_exc(),
                     "path": str(request.url.path),
                     "hint": "This is a bridge bug, not an EEG result. Do not act "
                             "on any state you read while errors are occurring."},
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=int(exc.status_code),
                            content={"ok": False, "error": "http_error",
                                     "detail": exc.detail, "path": str(request.url.path)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422,
                            content={"ok": False, "error": "validation_error",
                                     "detail": exc.errors(),
                                     "path": str(request.url.path)})

    # ---------------------------------------------------------- routes
    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(bs.status())

    @app.get("/schema")
    async def schema() -> JSONResponse:
        return JSONResponse(bs.schema())

    @app.get("/version")
    async def version() -> JSONResponse:
        return JSONResponse({"version": __version__, "schema": "/schema"})

    @app.get("/state")
    async def state(explain: bool = Query(False)) -> JSONResponse:
        payload = bs.state_payload()
        if not explain:
            payload.pop("model_info", None)
        return JSONResponse(payload)

    @app.get("/state/explain")
    async def state_explain() -> JSONResponse:
        return JSONResponse(bs.state_payload())

    @app.get("/state/stream")
    async def state_stream() -> StreamingResponse:
        """SSE at ``settings.state_hz`` with heartbeat comments."""
        hz = max(0.1, float(bs.settings.state_hz))
        period = 1.0 / hz

        async def gen():
            last = time.monotonic()
            while True:
                payload = bs.state_payload()
                yield f"event: state\ndata: {json.dumps(payload, default=str)}\n\n"
                now = time.monotonic()
                if now - last > 10.0:                 # idle heartbeat
                    last = now
                    yield ": heartbeat\n\n"
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache",
                                          "x-accel-buffering": "no",
                                          "connection": "keep-alive"})

    @app.get("/samples/stream")
    async def samples_stream(
        max_hz: float = Query(250.0, ge=1.0, le=1000.0),
        chunk_ms: float = Query(100.0, ge=20.0, le=2000.0),
        from_now: bool = Query(True),
    ) -> StreamingResponse:
        """Raw EEG counts as SSE, for an attached recorder or live display.

        ``event: samples`` frames carry ``t0`` (monotonic time of the first
        sample), ``fs`` and the ``values`` array. Samples within a frame are
        uniformly spaced: the device reports one arrival time per 10-sample
        packet, so that is the honest resolution and it is the same assumption
        :class:`StateEstimator` already makes.

        ``max_hz`` decimates by picking every n-th sample for display clients. It
        does **not** anti-alias, so a decimated stream is for drawing only - a
        recorder must take the full rate, which is the default.
        """
        period = max(0.02, float(chunk_ms) / 1000.0)
        step = max(1, int(round(float(bs.estimator.fs) / max(1.0, float(max_hz)))))

        async def gen():
            cursor = bs.sample_cursor() if from_now else 0.0
            while True:
                v, t, gap = bs.samples_since(cursor)
                if v.size:
                    cursor = float(t[-1])
                    vv, tt = (v[::step], t[::step]) if step > 1 else (v, t)
                    yield ("event: samples\ndata: " + json.dumps({
                        "t0": float(tt[0]), "t_last": float(tt[-1]),
                        "fs": float(bs.estimator.fs), "stride": int(step),
                        "n": int(vv.size), "gap": bool(gap),
                        "values": [float(x) for x in vv],
                    }) + "\n\n")
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache",
                                          "x-accel-buffering": "no",
                                          "connection": "keep-alive"})

    @app.get("/events")
    async def events(since: float | None = Query(None), limit: int = Query(100, ge=1, le=5000)
                     ) -> JSONResponse:
        ev = bs.recent_events(since=since, limit=limit)
        return JSONResponse({"ok": True, "count": len(ev), "events": ev,
                             "t_mono_now": clock()})

    @app.post("/mark")
    async def mark(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
        name = str(payload.get("name") or "").strip()
        if not name:
            return JSONResponse(status_code=400,
                                content={"ok": False, "error": "missing_name",
                                         "detail": "POST /mark needs {'name': str, "
                                                   "'meta': {}}",
                                         "example": {"name": "agent_paused",
                                                     "meta": {"reason": "overload"}}})
        meta = payload.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"value": meta}
        ev = bs.mark(name, meta)
        return JSONResponse({"ok": True, "event": ev, "t_mono": ev["t_mono"]})

    @app.get("/report/{test_id}")
    async def report(test_id: str) -> JSONResponse:
        """Latest saved evaluation for a test; a 404-shaped JSON when absent."""
        found = _find_report(test_id)
        if found is None:
            return JSONResponse(status_code=404, content={
                "ok": False, "error": "no_report", "test_id": test_id,
                "detail": (f"no evaluation JSON for {test_id!r} in {REPORT_DIR}; "
                           "run `nb evaluate` for that test first"),
                "searched": str(REPORT_DIR)})
        path, data = found
        return JSONResponse({"ok": True, "test_id": test_id, "path": str(path),
                             "report": data})

    @app.get("/candidate")
    async def candidate() -> JSONResponse:
        """External protocol §10 candidate signal for the current window.

        The shape is fixed and every unknown stays ``null``. Until a model is
        trained and a decision rule validated, ``candidate_label`` is
        ``"uncertain"`` -- that is the answer, not a placeholder to be filled by
        the most likely class.
        """
        return JSONResponse(bs.current_candidate())

    @app.post("/confirm")
    async def confirm(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
        """Record the participant's correction of a candidate, separately.

        The candidate was already durably logged when it was created, so this
        endpoint only *adds* the participant's own report. It never edits the
        prediction, and it never retrains anything: corrections are reviewed
        offline first.
        """
        seq = payload.get("candidate_seq")
        if seq is None:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "missing_candidate_seq",
                "detail": "POST /confirm needs {'candidate_seq': N, "
                          "'participant_response': 'yes'|'no'|'skipped'|'unsure', "
                          "'note': ''}",
                "current_candidate_seq": int(bs.candidate_seq)})
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "bad_candidate_seq",
                "detail": "candidate_seq must be an integer"})
        out = bs.confirm(seq, str(payload.get("participant_response") or ""),
                         str(payload.get("note") or ""))
        return JSONResponse(out, status_code=200 if out.get("ok") else 400)

    @app.post("/calibrate")
    async def calibrate(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
        """Capture the personal baseline now (the estimator's ~60 s window).

        Returns 200 with ``calibrated: true`` once a baseline exists, and 409 with
        a ``reason`` (normally ``not_enough_data``) while the bridge is still
        collecting. The baseline defines every later 0..1 field, so it is captured
        on request rather than automatically: see :meth:`BridgeState.calibrate`.
        """
        force = bool((payload or {}).get("force", False))
        res = bs.calibrate(force=force)
        return JSONResponse(res, status_code=200 if res.get("calibrated") else 409)

    @app.get("/metrics")
    async def metrics(window_s: float | None = Query(None, ge=1.0, le=86400.0)
                      ) -> JSONResponse:
        """False-activation and stream-health metrics.

        ``idle_false_activations_per_min`` is the honest cost of a continuous
        decoder: detections during periods with no requested action, measured
        with the same detector instance (and therefore the same smoothing and
        refractory settings) that the live stream uses.
        """
        return JSONResponse(bs.metrics(window_s=window_s))

    @app.post("/idle/start")
    async def idle_start() -> JSONResponse:
        """Declare the start of an explicit no-command recording block."""
        return JSONResponse({"ok": True, "t_mono": bs.mark_idle_start(),
                             "note": "explicit idle block started"})

    @app.post("/idle/end")
    async def idle_end() -> JSONResponse:
        """Close an explicit no-command block and report its duration."""
        return JSONResponse({"ok": True, "idle_seconds": bs.mark_idle_end()})

    # Register ``WS /ws`` explicitly, at the front of the route table.
    #
    # WHY not `@app.websocket("/ws")` on a nested function: this module uses
    # `from __future__ import annotations`, so a nested function's annotation is
    # the *string* "WebSocket". FastAPI resolves such strings against the module
    # globals, where a closure-local `WebSocket` import does not exist; it then
    # treats the socket as a required *query* parameter, fails validation before
    # the handshake and uvicorn answers HTTP 403. A module-level endpoint whose
    # annotation resolves to the real class avoids the whole trap.
    if APIWebSocketRoute is not None:
        app.router.routes.insert(0, APIWebSocketRoute("/ws", _ws_endpoint, name="ws"))

    return app


async def _ws_endpoint(websocket: WebSocket) -> None:
    """``WS /ws``: state pushes plus inbound ``{'type':'mark'|'ping'}``.

    The bridge is reached through ``websocket.app.state.bridge`` so this can be
    a module-level function (see the note in :func:`create_app` for why that is
    load-bearing rather than stylistic).
    """
    bs: BridgeState = websocket.app.state.bridge
    await websocket.accept()
    hz = max(0.1, float(bs.settings.state_hz))
    period = 1.0 / hz
    try:
        await websocket.send_json({"type": "hello", "schema": "/schema",
                                   "state_hz": hz, "version": __version__})
        while True:
            await websocket.send_json({"type": "state",
                                       "data": bs.state_payload()})
            try:
                raw = await asyncio.wait_for(websocket.receive_text(),
                                             timeout=period)
            except asyncio.TimeoutError:
                continue
            await _handle_ws_message(websocket, bs, raw)
    except Exception:
        # client went away, or the socket died: nothing to report, just stop
        try:
            await websocket.close()
        except Exception:
            pass


async def _handle_ws_message(websocket, bs: BridgeState, raw: str) -> None:
    """Inbound WS control messages. Marks are recorded, pings are answered."""
    try:
        msg = json.loads(raw or "{}")
    except Exception:
        await websocket.send_json({"type": "error", "detail": "inbound message must be JSON"})
        return
    if not isinstance(msg, dict):
        await websocket.send_json({"type": "error", "detail": "inbound message must be an object"})
        return
    kind = str(msg.get("type") or "")
    if kind == "ping":
        await websocket.send_json({"type": "pong", "t_mono": clock()})
    elif kind == "mark":
        ev = bs.mark(str(msg.get("name") or "agent_mark"), msg.get("meta") or {})
        await websocket.send_json({"type": "mark_ack", "event": ev})
    elif kind == "confirm":
        out = bs.confirm(int(msg.get("candidate_seq", -1)),
                         str(msg.get("participant_response") or ""),
                         str(msg.get("note") or ""))
        await websocket.send_json({"type": "confirm_ack", **out})
    else:
        await websocket.send_json({"type": "error",
                                   "detail": f"unknown message type {kind!r} "
                                             "(want 'mark', 'ping' or 'confirm')"})


def _find_report(test_id: str) -> tuple[Path, dict[str, Any]] | None:
    """Newest saved report JSON for a test id, if any.

    Report writers are free to name their JSON ``<test>.json``,
    ``<test>_eval.json`` or nest it in ``<test>/...json``; all three shapes are
    accepted, newest first, and a corrupt file is skipped rather than raised.

    The search is deliberately defensive: ``test_id`` comes from the URL, and a
    glob pattern built from user input can raise ``ValueError`` on illegal
    wildcards (or be abused for traversal). Failures therefore degrade to "no
    report" instead of a 500 - a missing report is not an error condition.
    """
    root = Path(REPORT_DIR)
    if not root.exists():
        return None
    tid = Path(test_id.strip()).name                  # no traversal, no slashes
    if not tid or tid in (".", ".."):
        return None
    cands: list[Path] = []
    patterns = (f"{tid}.json", f"{tid}_*.json", f"{tid}*.eval.json",
                f"*{tid}*.json", f"{tid}/**/*.json")
    for pat in patterns:
        try:
            cands += [p for p in root.glob(pat) if p.is_file()]
        except (ValueError, OSError):
            continue                                  # illegal wildcard: skip it
    try:
        nested = [p for p in (root / tid).rglob("*.json") if p.is_file()]
        cands += nested
    except (ValueError, OSError):
        pass
    seen: set[Path] = set()
    for p in sorted(cands, key=lambda x: _safe_mtime(x), reverse=True):
        if p in seen:
            continue
        seen.add(p)
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        return p, data
    return None


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------- serve
def serve(host: str | None = None, port: int | None = None, device: str = "auto",
          replay_session: str | Path | None = None) -> None:
    """Run the bridge headless and print the exact URL a human can open.

    ``device='auto'`` probes for the headband (~8 s) and falls back to the
    synthetic source, so the bridge always comes up; ``replay_session`` switches
    to replaying a recorded session instead of acquiring live.

    The URLs are printed with ``flush=True`` because a human verifies this by
    reading the terminal: when stdout is a pipe (systemd, a supervisor, a
    container log) Python block-buffers it, and an unflushed banner would either
    never appear or arrive only after the server died.
    """
    import uvicorn

    s = get_settings()
    host = str(host or s.bridge_host)
    port = int(port or s.bridge_port)
    mode = "replay" if replay_session else "live"
    app = create_app(device=device, mode=mode, replay_session=replay_session)
    url = f"http://{host}:{port}"
    print(f"[neurobridge] mode      : {mode}"
          + (f" (session {replay_session})" if replay_session else f" (device {device})"),
          flush=True)
    print(f"[neurobridge] schema    : {url}/schema", flush=True)
    print(f"[neurobridge] health    : {url}/health", flush=True)
    print(f"[neurobridge] state     : {url}/state", flush=True)
    print(f"[neurobridge] SSE stream: {url}/state/stream", flush=True)
    print(f"[neurobridge] websocket : ws://{host}:{port}/ws", flush=True)
    print(f"[neurobridge] metrics   : {url}/metrics", flush=True)
    print(f"[neurobridge] candidate : {url}/candidate", flush=True)
    print(f"[neurobridge] state rate: {s.state_hz} Hz "
          f"(~{s.state_hz * 60:.0f} payloads/min)", flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:                        # pragma: no cover - operator
        print("[neurobridge] stopped", flush=True)
