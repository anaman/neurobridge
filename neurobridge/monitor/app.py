"""The monitor window: live display, recording, and replay.

Layout, and why
---------------
* **Waveform on the left, numbers on the right.** The trace is the thing you
  cannot get from a table, and it deserves the space; the fields are a dashboard
  and read fine in a column.
* **A provenance banner across the top, always.** The monitor can be showing a
  live person, a bridge, or a replay, and those look identical on screen. The
  banner is the one element that must never be ambiguous, so it is drawn from
  :class:`~neurobridge.agent.link.Provenance` and turns red for anything that is
  not a live headband.
* **One worker thread, one queue.** Acquisition and CSV writing must not block
  the GUI, and tkinter is not thread-safe, so the worker only ever touches the
  source and the recorder and hands :class:`~neurobridge.monitor.sources.Reading`
  objects to the UI through a bounded queue. The UI redraws on a timer.

Everything the window displays is derived from the same ``StateEstimator`` the
HTTP bridge uses, so a number here and a number on ``/state`` cannot disagree.
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np

from ..agent.link import Provenance
from ..config import SESSION_DIR, ensure_dirs
from .recorder import MonitorRecorder
from .sources import (BANDS_FOR_DISPLAY, BridgeSource, DeviceSource, MonitorSource,
                      Reading, ReplaySource, Snapshot, open_source_for_mode)

# dark palette, matching the stimulus presenter so the two windows agree
BG = "#121214"
PANEL = "#1a1a1e"
FG = "#e8e8ea"
DIM = "#8a8a92"
ACCENT = "#4ea1d3"
WARN = "#d9534f"
OK = "#5cb85c"
FIELD_COLOURS = {
    "attention": "#4ea1d3", "arousal": "#e0a458", "load": "#c05b8a",
    "fatigue": "#8a7fd3", "relaxation": "#5cb85c",
}
STATE_FIELDS = ("attention", "arousal", "load", "fatigue", "relaxation")

#: Selectable view lengths, in minutes. A six-second window shows you a blink; a
#: sixty-minute one shows you a session. Both are useful, and which one you want
#: depends on the question, so it is a control rather than a constant.
VIEW_CHOICES = (1, 5, 10, 15, 30, 60)

#: How much history the trace keeps in memory, independent of the current view.
#: Always the longest selectable view, so widening the window uncovers the past
#: instead of showing an empty chart. 60 min at 250 Hz is ~14 MB as float64.
RETAIN_MINUTES = max(VIEW_CHOICES)
#: Redraw budget by view length: a 250 Hz trace an hour long is ~900k points, so
#: the long views refresh more slowly than the short ones. The eye cannot read
#: 25 fps of one-hour-ago anyway.
REDRAW_INTERVAL_S = {1: 0.04, 5: 0.06, 10: 0.10, 15: 0.15, 30: 0.25, 60: 0.40}


@dataclass
class MonitorConfig:
    """Everything the window needs, resolved by the CLI before any UI exists."""

    device: str = "auto"
    url: str | None = None
    replay: str | Path | None = None
    record: bool = True
    out_dir: Path | None = None
    #: Visible trace length in seconds. The toolbar sets this from VIEW_CHOICES;
    #: --window overrides it for a close look at a few seconds.
    window_s: float = 60.0
    state_hz: float = 2.0
    max_hz: float = 250.0
    session_id: str | None = None


class MonitorApp:
    """The tkinter application. Construct, then call :meth:`run`."""

    def __init__(self, root: tk.Tk, config: MonitorConfig) -> None:
        self.root = root
        self.config = config
        ensure_dirs()
        self.source = open_source_for_mode(
            device=config.device, url=config.url, replay=config.replay,
            state_hz=config.state_hz, chunk_hz=config.max_hz)
        self.source.open()

        self.prov: Provenance = self.source.provenance()
        self.recorder: MonitorRecorder | None = None
        self.recording = False
        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._error: str | None = None

        # trace buffer: up to RETAIN_MINUTES of raw samples. A deque, because the
        # trim drops from the front and list.pop(0) is O(n) - at 250 Hz into an
        # hour-long buffer that is thousands of O(n) pops per second.
        self._t_buf: deque[np.ndarray] = deque()
        self._v_buf: deque[np.ndarray] = deque()
        self._n_total = 0
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._rate_window: list[tuple[float, int]] = []
        self._snap: Snapshot | None = None
        self._snap_wall = 0.0
        self._intents = 0
        self._yscale: tuple[float, float] | None = None
        self._started = time.perf_counter()
        self._last_draw = 0.0

        self._build_ui()
        self._refresh_banner()
        self._start_worker()
        self.root.after(40, self._tick)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.root.title("NeuroBridge Monitor")
        self.root.configure(bg=BG)
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=4)
        style.configure("Dark.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)

        # -- provenance banner
        self.banner = tk.Label(self.root, text="", bg=BG, fg=FG, anchor="w",
                               font=("TkDefaultFont", 10, "bold"), padx=8, pady=5)
        self.banner.pack(fill="x")

        # -- toolbar
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(fill="x")
        self.btn_replay = self._button(bar, "Open replay…", self.on_open_replay)
        self.btn_play = self._button(bar, "Pause", self.on_toggle_play)
        self.btn_restart = self._button(bar, "Restart", self.on_restart)
        tk.Label(bar, text="speed", bg=PANEL, fg=DIM).pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1x")
        sp = ttk.Combobox(bar, textvariable=self.speed_var, width=5, state="readonly",
                          values=["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        sp.pack(side="left")
        sp.bind("<<ComboboxSelected>>", self.on_speed)
        tk.Label(bar, text="view", bg=PANEL, fg=DIM).pack(side="left", padx=(12, 2))
        self.view_var = tk.StringVar(value=self._view_label(self.config.window_s))
        self.view_box = ttk.Combobox(
            bar, textvariable=self.view_var, width=7, state="readonly",
            values=[self._view_label(m * 60.0) for m in VIEW_CHOICES])
        self.view_box.pack(side="left")
        self.view_box.bind("<<ComboboxSelected>>", self.on_view_change)
        self.btn_record = self._button(bar, "● Record", self.on_toggle_record,
                                       fg=WARN)
        self.btn_cal = self._button(bar, "Calibrate", self.on_calibrate)
        self.btn_mark = self._button(bar, "Mark", self.on_mark)
        self.btn_quit = self._button(bar, "Quit", self.on_quit)

        # -- main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=6, pady=6)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.trace_id = self.canvas.create_line(0, 0, 0, 0, fill=ACCENT, width=1)
        self.canvas.bind("<Configure>", lambda e: self._redraw(force=True))

        right = tk.Frame(main, bg=BG, width=330)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self.band_canvas = self._panel(right, "Band power (% of 1-45 Hz)", 132)
        self.field_canvas = self._panel(right, "State (0.5 = your baseline)", 158)
        self.signal_canvas = self._panel(right, "Signal", 150)
        self.intent_label = tk.Label(right, text="", bg=BG, fg=DIM, anchor="w",
                                     justify="left", wraplength=320)
        self.intent_label.pack(fill="x", pady=(6, 0))

        # -- replay scrubber
        scrub = tk.Frame(self.root, bg=BG)
        scrub.pack(fill="x", padx=8)
        self.scrub = ttk.Scale(scrub, from_=0.0, to=1.0, orient="horizontal",
                               command=self.on_scrub)
        self.scrub.pack(side="left", fill="x", expand=True)
        self.scrub.bind("<ButtonRelease-1>", lambda e: setattr(self, "_scrubbing", False))
        self.scrub.bind("<ButtonPress-1>", lambda e: setattr(self, "_scrubbing", True))
        self._scrubbing = False
        self.pos_label = tk.Label(scrub, text="--:-- / --:--", bg=BG, fg=DIM,
                                  width=16)
        self.pos_label.pack(side="left", padx=6)

        # -- status bar
        self.status = tk.Label(self.root, text="starting…", bg=BG, fg=DIM, anchor="w",
                               padx=8, pady=4, font=("TkFixedFont", 9))
        self.status.pack(fill="x")

        self.root.bind("<space>", lambda e: self.on_toggle_play())
        self.root.bind("<Escape>", lambda e: self.on_quit())
        self.root.bind("q", lambda e: self.on_quit())
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self._is_replay = isinstance(self.source, ReplaySource)
        for w in (self.btn_play, self.btn_restart, self.scrub):
            if not self._is_replay:
                w.configure(state="disabled") if hasattr(w, "configure") else None

    def _button(self, parent: tk.Widget, text: str, cmd: Any, fg: str = FG) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd, bg=PANEL, fg=fg,
                      activebackground=BG, activeforeground=FG, relief="flat",
                      padx=8, pady=3, highlightthickness=0, cursor="hand2")
        b.pack(side="left", padx=2, pady=2)
        return b

    def _panel(self, parent: tk.Widget, title: str, height: int) -> tk.Canvas:
        tk.Label(parent, text=title, bg=BG, fg=DIM, anchor="w").pack(fill="x")
        c = tk.Canvas(parent, bg=PANEL, height=height, highlightthickness=0)
        c.pack(fill="x", pady=(2, 8))
        return c

    # ------------------------------------------------------------- worker
    def _start_worker(self) -> None:
        def run() -> None:
            while not self._stop.is_set():
                try:
                    reading = self.source.poll(0.02)
                except Exception as exc:                  # a dead source must not hang the UI
                    self._error = f"{type(exc).__name__}: {exc}"
                    time.sleep(0.2)
                    continue
                if self.recording and self.recorder is not None:
                    try:
                        for c in reading.chunks:
                            self.recorder.add_samples(c.values, c.times)
                        if reading.snapshot is not None:
                            s = reading.snapshot
                            self.recorder.add_state(
                                s.t_mono, s.t_wall, s.fields, confidence=s.confidence,
                                artifact_level=s.artifact_level, contact=s.contact,
                                source=s.source, seq=s.seq, informative=s.informative,
                                derivation=s.derivation, intents=s.intents)
                    except Exception as exc:
                        self._error = f"recorder: {exc}"
                try:
                    self._q.put_nowait(reading)
                except queue.Full:
                    pass

        self._worker = threading.Thread(target=run, name="monitor-worker", daemon=True)
        self._worker.start()

    # -------------------------------------------------------------- drawing
    def _tick(self) -> None:
        drained = 0
        while True:
            try:
                reading = self._q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            self._absorb(reading)
            if drained > 500:
                break
        if drained:
            self._redraw()
        self._update_status()
        self._update_scrubber()
        delay = 40 if self.source.fs else 60
        self.root.after(delay, self._tick)

    def _absorb(self, reading: Reading) -> None:
        for c in reading.chunks:
            if c.n == 0:
                continue
            self._t_buf.append(c.times)
            self._v_buf.append(c.values)
            self._n_total += c.n
            if self._t_first is None:
                self._t_first = float(c.times[0])
            self._t_last = float(c.times[-1])
            self._rate_window.append((time.perf_counter(), c.n))
        # Retain the longest selectable view (not just the current one) so that
        # widening the window reveals history instead of an empty chart.
        cutoff = (self._t_last or 0.0) - (RETAIN_MINUTES * 60.0 + 5.0)
        while self._t_buf and float(self._t_buf[0][-1]) < cutoff:
            self._t_buf.popleft()
            self._v_buf.popleft()
        self._rate_window = [r for r in self._rate_window
                             if time.perf_counter() - r[0] < 3.0]
        if reading.snapshot is not None:
            self._snap = reading.snapshot
            self._snap_wall = time.perf_counter()
            self._intents += len(reading.snapshot.intents or [])

    def _visible(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The samples inside the current view, oldest first.

        Walks the deque backwards and stops as soon as the window is covered, so
        a 1-minute view concatenates ~15k samples rather than the full hour-long
        buffer it is sitting in.
        """
        if not self._t_buf:
            return None
        t_hi = float(self._t_buf[-1][-1])
        t_lo = t_hi - self.config.window_s
        chunks: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for i in range(len(self._t_buf) - 1, -1, -1):
            tc = self._t_buf[i]
            if float(tc[-1]) < t_lo and chunks:
                break
            chunks.append(tc)
            vals.append(self._v_buf[i])
        if not chunks:
            return None
        chunks.reverse()
        vals.reverse()
        t = np.concatenate(chunks)
        v = np.concatenate(vals)
        m = t >= t_lo
        return t[m], v[m]

    def _redraw(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_draw) < self._redraw_interval():
            return
        self._last_draw = now
        self._draw_trace()
        self._draw_bands()
        self._draw_fields()
        self._draw_signal()

    @staticmethod
    def _view_label(window_s: float) -> str:
        """'1 min' / '5 min' / '6 s' - the label the combobox shows."""
        if window_s >= 60.0:
            return f"{window_s / 60.0:g} min"
        return f"{window_s:g} s"

    def _redraw_interval(self) -> float:
        """Redraw budget for the current view length (see REDRAW_INTERVAL_S)."""
        minutes = max(1.0, self.config.window_s / 60.0)
        nearest = min(REDRAW_INTERVAL_S, key=lambda m: abs(m - minutes))
        return REDRAW_INTERVAL_S[nearest]

    def _draw_trace(self) -> None:
        W = max(50, self.canvas.winfo_width())
        H = max(50, self.canvas.winfo_height())
        vis = self._visible()
        if vis is None or vis[0].size < 2:
            return
        t, v = vis
        t_hi = float(t[-1])
        t_lo = t_hi - self.config.window_s
        span = max(1e-6, t_hi - t_lo)

        # robust vertical scale, smoothed so the trace does not breathe. The
        # percentile is taken from a subsample: over an hour-long window the full
        # sort would cost more than the drawing it feeds, and a display scale
        # does not need every point.
        vs = v if v.size <= 40000 else v[:: max(1, v.size // 20000)]
        lo = float(np.percentile(vs, 2))
        hi = float(np.percentile(vs, 98))
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.12 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if self._yscale is None:
            self._yscale = (lo, hi)
        else:
            a = 0.25
            self._yscale = (self._yscale[0] * (1 - a) + lo * a,
                            self._yscale[1] * (1 - a) + hi * a)
        ylo, yhi = self._yscale
        yspan = max(1e-9, yhi - ylo)

        def ymap(vals: np.ndarray) -> np.ndarray:
            return H - (vals - ylo) / yspan * H

        if t.size >= W:
            edges = np.linspace(t_lo, t_hi, W + 1)
            idx = np.clip(np.searchsorted(t, edges), 0, t.size - 1)
            starts = np.maximum.accumulate(idx[:-1])
            vmin = np.minimum.reduceat(v, starts)
            vmax = np.maximum.reduceat(v, starts)
            xs = (np.arange(W) + 0.5) / W * W
            pts = np.empty(W * 4, dtype=np.float64)
            pts[0::4] = xs
            pts[1::4] = ymap(vmin)
            pts[2::4] = xs
            pts[3::4] = ymap(vmax)
        else:
            xs = (t - t_lo) / span * W
            ys = ymap(v)
            pts = np.empty(xs.size * 4, dtype=np.float64)
            pts[0::4] = xs
            pts[1::4] = ys
            pts[2::4] = xs + 0.6
            pts[3::4] = ys
        self.canvas.coords(self.trace_id, *pts.tolist())
        self.canvas.itemconfigure(self.trace_id, fill=ACCENT)

    def _bar_rows(self, canvas: tk.Canvas, rows: list[tuple[str, float, str, str]],
                  vmax: float = 1.0) -> None:
        """Draw label + meter + value for each row, on a plain Canvas."""
        canvas.delete("row")
        W = max(120, canvas.winfo_width())
        H = max(40, canvas.winfo_height())
        n = max(1, len(rows))
        rh = H / n
        lab_w, val_w = 78, 56
        bar_x0, bar_x1 = lab_w + 2, W - val_w
        for i, (label, frac, text, colour) in enumerate(rows):
            cy = i * rh + rh / 2
            canvas.create_text(4, cy, text=label, anchor="w", fill=DIM,
                               font=("TkFixedFont", 8), tags="row")
            canvas.create_rectangle(bar_x0, cy - rh * 0.26, bar_x1, cy + rh * 0.26,
                                    fill=BG, outline="#333338", tags="row")
            f = max(0.0, min(1.0, float(frac) / vmax if vmax else 0.0))
            if f > 0:
                canvas.create_rectangle(bar_x0, cy - rh * 0.26,
                                        bar_x0 + (bar_x1 - bar_x0) * f,
                                        cy + rh * 0.26, fill=colour, outline="",
                                        tags="row")
            canvas.create_text(W - 4, cy, text=text, anchor="e", fill=FG,
                               font=("TkFixedFont", 8), tags="row")

    def _draw_bands(self) -> None:
        snap = self._snap
        if snap is None or not snap.bands:
            self.band_canvas.delete("row")
            self.band_canvas.create_text(4, 10, text="no band data yet", anchor="w",
                                         fill=DIM, tags="row")
            return
        rows = [(b, float(snap.bands.get(b, 0.0)), f"{snap.bands.get(b, 0.0):.1f}%",
                 ACCENT) for b in BANDS_FOR_DISPLAY]
        self._bar_rows(self.band_canvas, rows, vmax=100.0)

    def _draw_fields(self) -> None:
        snap = self._snap
        if snap is None:
            self.field_canvas.delete("row")
            self.field_canvas.create_text(4, 10, text="no state yet", anchor="w",
                                          fill=DIM, tags="row")
            return
        rows = []
        for f in STATE_FIELDS:
            val = float(snap.fields.get(f, 0.5))
            if snap.informative:
                rows.append((f, val, f"{val:.2f}", FIELD_COLOURS[f]))
            else:
                rows.append((f, 0.0, "no info", "#44444a"))
        self._bar_rows(self.field_canvas, rows, vmax=1.0)

    def _draw_signal(self) -> None:
        c = self.signal_canvas
        c.delete("row")
        snap = self._snap
        W = max(150, c.winfo_width())
        if snap is None:
            c.create_text(4, 10, text="waiting for state…", anchor="w", fill=DIM,
                          tags="row")
            return
        age = time.perf_counter() - self._snap_wall
        stale = age > 3.0
        lines = [
            ("confidence", f"{snap.confidence:.2f}" +
             ("  (act ≥0.4)" if snap.confidence < 0.4 else "  ok")),
            ("artifact", f"{snap.artifact_level:.2f}"),
            ("contact", snap.contact[:22]),
            ("source", snap.source),
            ("seq", "—" if snap.seq is None else str(snap.seq)),
        ]
        rows = [(k, v, WARN if (stale or (k == "confidence" and snap.confidence < 0.4))
                 else FG) for k, v in lines]
        rh = max(14, c.winfo_height() / max(1, len(rows)))
        for i, (k, v, col) in enumerate(rows):
            y = i * rh + rh / 2
            c.create_text(4, y, text=k, anchor="w", fill=DIM,
                          font=("TkFixedFont", 8), tags="row")
            c.create_text(86, y, text=str(v), anchor="w", fill=col,
                          font=("TkFixedFont", 8), tags="row")
        if stale:
            c.create_text(4, len(rows) * rh + 8,
                          text=f"STALE: no payload for {age:.1f}s — do not trust "
                               "the last value", anchor="w", fill=WARN,
                          font=("TkDefaultFont", 8, "bold"), tags="row")

    def _refresh_banner(self) -> None:
        """The one label that must never be ambiguous."""
        prov = self.prov
        try:
            prov = self.source.provenance()
            self.prov = prov
        except Exception:
            pass
        colour = OK if prov.is_live_human else WARN
        self.banner.configure(text=prov.banner(), fg=colour)

    def _update_status(self) -> None:
        snap = self._snap
        elapsed = time.perf_counter() - self._started
        rate = sum(n for _, n in self._rate_window) / 3.0 if self._rate_window else 0.0
        dur = 0.0
        if self._t_first is not None and self._t_last is not None:
            dur = self._t_last - self._t_first
        eff = self._n_total / dur if dur > 0 else 0.0
        parts = [
            f"source={self.prov.device}",
            f"fs={self.source.fs:.0f}Hz",
            f"rate={rate:.0f}Hz",
            f"view={self._view_label(self.config.window_s)}",
            f"samples={self._n_total}",
            f"fs_eff={eff:.1f}Hz",
            f"elapsed={elapsed:5.0f}s",
            ("REC " + str(self.recorder.path.name)) if self.recording and self.recorder
            else "not recording",
        ]
        if snap is not None:
            parts.append(f"conf={snap.confidence:.2f}")
        if self._error:
            parts.append(f"ERROR {self._error}")
        self.status.configure(text="  ".join(parts),
                              fg=WARN if self._error else DIM)
        self.intent_label.configure(
            text=f"intents this session: {self._intents}"
                 + ("" if snap is None else
                    f"   |   calibrated: {snap.calibrated}"))
        self._refresh_banner()

    def _update_scrubber(self) -> None:
        if not isinstance(self.source, ReplaySource) or self._scrubbing:
            return
        pos, dur = self.source.position
        if dur > 0:
            self.scrub.set(pos / dur)
        self.pos_label.configure(text=f"{_mmss(pos)} / {_mmss(dur)}")
        if self.source.playing and self.btn_play.cget("text") != "Pause":
            self.btn_play.configure(text="Pause")
        if not self.source.playing and self.btn_play.cget("text") != "Play":
            self.btn_play.configure(text="Play")

    # ------------------------------------------------------------ commands
    def on_open_replay(self) -> None:
        initial = str(SESSION_DIR if Path(SESSION_DIR).exists() else Path.home())
        chosen = filedialog.askopenfilename(
            title="Open a session folder's eeg.csv, or cancel to pick a session folder",
            initialdir=initial, filetypes=[("Monitor CSV", "eeg.csv"),
                                           ("CSV", "*.csv"), ("All files", "*.*")])
        if chosen:
            self._switch_replay(Path(chosen))
            return
        d = filedialog.askdirectory(title="…or choose a recorded session folder",
                                    initialdir=initial)
        if d:
            self._switch_replay(Path(d))

    def _switch_replay(self, path: Path) -> None:
        try:
            self._stop.set()
            if self._worker is not None:
                self._worker.join(timeout=1.0)
            self.source.close()
            if self.recording:
                self.on_toggle_record()          # close the file cleanly first
            self._stop = threading.Event()
            self.source = ReplaySource(path, state_hz=self.config.state_hz)
            self.source.open()
            self._is_replay = True
            self._t_buf.clear()
            self._v_buf.clear()
            self._n_total = 0
            self._t_first = self._t_last = None
            self._yscale = None
            self._snap = None
            self._intents = 0
            for w in (self.btn_play, self.btn_restart, self.scrub):
                try:
                    w.configure(state="normal")
                except Exception:
                    pass
            self._refresh_banner()
            self._start_worker()
        except Exception as exc:
            messagebox.showerror("Cannot open", f"{path}\n\n{type(exc).__name__}: {exc}")

    def on_toggle_play(self) -> None:
        if isinstance(self.source, ReplaySource):
            self.source.play() if not self.source.playing else self.source.pause()
            self.btn_play.configure(text="Pause" if self.source.playing else "Play")

    def on_restart(self) -> None:
        if isinstance(self.source, ReplaySource):
            self.source.restart()
            self._t_buf.clear()
            self._v_buf.clear()
            self._yscale = None

    def on_view_change(self, _evt: Any = None) -> None:
        """Set the visible trace length from the toolbar (1…60 minutes)."""
        try:
            minutes = float(str(self.view_var.get()).split()[0])
        except (ValueError, IndexError):
            return
        self.config.window_s = minutes * 60.0
        # Refit the vertical scale at once: keeping the old one would show the
        # new span through the old window's amplitude for a second or two.
        self._yscale = None
        self._redraw(force=True)

    def on_speed(self, _evt: Any = None) -> None:
        if isinstance(self.source, ReplaySource):
            try:
                self.source.set_speed(float(self.speed_var.get().rstrip("x")))
            except ValueError:
                pass

    def on_scrub(self, value: str) -> None:
        if isinstance(self.source, ReplaySource) and self._scrubbing:
            dur = self.source.position[1]
            self.source.seek(float(value) * dur)

    def on_toggle_record(self) -> None:
        if self.recording:
            assert self.recorder is not None
            rec = self.recorder.close()
            self.recording = False
            self.btn_record.configure(text="● Record", fg=WARN)
            messagebox.showinfo(
                "Recording saved",
                f"{rec.n_samples} samples, {rec.n_state_rows} state rows, "
                f"{rec.n_events} events\n\n{rec.path}\n\n"
                f"CSV: eeg.csv, state.csv, events.csv")
        else:
            prov = self.prov
            self.recorder = MonitorRecorder(
                out_dir=self.config.out_dir, session_id=self.config.session_id,
                provenance=prov,
                meta={"fs": self.source.fs,
                      "monitor_source": type(self.source).__name__,
                      "url": getattr(getattr(self.source, "client", None),
                                     "base_url", None),
                      "replay_of": str(self.source.path)
                      if isinstance(self.source, ReplaySource) else None})
            self.recorder.event("record_start", {"source": prov.device})
            self.recording = True
            self.btn_record.configure(text="■ Stop", fg=OK)

    def on_calibrate(self) -> None:
        res = self.source.calibrate()
        if res.get("calibrated"):
            self.recorder and self.recording and self.recorder.event("calibrate", res)
            messagebox.showinfo("Calibrated",
                                "Baseline captured. The five fields are now relative "
                                "to your own baseline (0.5 = at baseline).")
        else:
            messagebox.showwarning(
                "No baseline yet",
                f"{res.get('reason') or res.get('error') or res}\n\n"
                "Leave it running and retry once there is ~65 s of clean signal.")

    def on_mark(self) -> None:
        name = f"manual_{int(time.time())}"
        t = self.recorder.event("manual_mark", {"by": "operator"}) if self.recorder \
            else time.perf_counter()
        try:
            if isinstance(self.source, BridgeSource):
                self.source.client.mark(name, {"by": "monitor"})
                self.recorder.event("manual_mark", {"by": "operator"})
        except Exception:
            pass
        self.status.configure(text=f"marked at t_mono={t:.2f}")

    def on_quit(self) -> None:
        try:
            self._stop.set()
            if self._worker is not None:
                self._worker.join(timeout=1.0)
            if self.recording and self.recorder is not None:
                self.recorder.close()
            self.source.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def run_monitor(config: MonitorConfig) -> int:
    """Open the window and block until it is closed."""
    try:
        root = tk.Tk()
    except Exception as exc:                     # pragma: no cover - no display
        print(f"cannot open a window ({type(exc).__name__}: {exc}).\n"
              "Use --headless to record without a display.")
        return 2
    app = MonitorApp(root, config)
    if config.record:
        app.on_toggle_record()
    app.run()
    return 0


def run_headless(config: MonitorConfig, seconds: float | None = None,
                 on_tick: Any = None) -> Any:
    """Record without a window. Returns the recorder receipt (or None).

    The same source and recorder objects the window uses, so a headless capture
    is byte-for-byte the same kind of recording - this is what makes the monitor
    usable over SSH and testable in CI, rather than a second code path that
    drifts from the GUI one.
    """
    source = open_source_for_mode(device=config.device, url=config.url,
                                  replay=config.replay, state_hz=config.state_hz,
                                  chunk_hz=config.max_hz)
    source.open()
    prov = source.provenance()
    print(prov.banner())
    recorder = None
    if config.record:
        recorder = MonitorRecorder(
            out_dir=config.out_dir, session_id=config.session_id, provenance=prov,
            meta={"fs": source.fs, "monitor_source": type(source).__name__})
        recorder.event("record_start", {"source": prov.device})
    n = 0
    snaps = 0
    t0 = time.perf_counter()
    next_report = t0 + 2.0
    try:
        while True:
            reading = source.poll(0.02)
            for c in reading.chunks:
                n += c.n
                if recorder is not None:
                    recorder.add_samples(c.values, c.times)
            if reading.snapshot is not None:
                snaps += 1
                s = reading.snapshot
                if recorder is not None:
                    recorder.add_state(s.t_mono, s.t_wall, s.fields,
                                       confidence=s.confidence,
                                       artifact_level=s.artifact_level,
                                       contact=s.contact, source=s.source, seq=s.seq,
                                       informative=s.informative,
                                       derivation=s.derivation, intents=s.intents)
                if on_tick is not None:
                    on_tick(reading)
            now = time.perf_counter()
            if now >= next_report:
                next_report = now + 2.0
                conf = reading.snapshot.confidence if reading.snapshot else None
                conf_s = f"  conf={conf:.2f}" if conf is not None else ""
                print(f"  {now - t0:6.1f}s  samples={n:7d}  "
                      f"snapshots={snaps:4d}{conf_s}", flush=True)
            if seconds is not None and (now - t0) >= float(seconds):
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as exc:
        print(f"\nmonitor error: {type(exc).__name__}: {exc}")
    finally:
        source.close()
    if recorder is None:
        return None
    rec = recorder.close()
    print(f"\nrecorded {rec.n_samples} samples, {rec.n_state_rows} state rows, "
          f"{rec.n_events} events -> {rec.path}")
    return rec
