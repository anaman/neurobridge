"""NeuroBridge control panel (Tkinter).

A desktop front-end for the same commands the CLI runs, so the system menu can
launch something you can click. It is deliberately a *control panel* rather than
a re-implementation:

* every button shells out to ``python -m neurobridge <args>``, so what you see
  here is exactly what ``./bin/nb`` does — there is no second code path to drift;
* output streams into the console pane as it is produced, so a session's
  prompts, quality verdict and timing report are visible while it runs;
* long commands run in a worker thread, so the window never freezes and Stop
  always works.

The GUI is for setup, launching, monitoring and analysis. The experiments
themselves open their own pygame window (fullscreen if you ask for it), because
stimulus timing must not go through a toolkit event loop.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from .config import (DOCS_DIR, LOG_DIR, MODEL_DIR, REPORT_DIR, ROOT, SESSION_DIR,
                     get_settings)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

APP_TITLE = "NeuroBridge — EEG training control panel"


# ---------------------------------------------------------------- helpers
def _nb_argv(*args: str) -> list[str]:
    """Invoke the CLI through the same interpreter, so the venv is used."""
    return [sys.executable, "-m", "neurobridge", *args]


def status_lines() -> list[tuple[str, str]]:
    """Cheap status facts (no BLE scan: that is a button, not a startup cost)."""
    from .catalog import load_specs, validate_catalog
    from .voice import available as mic_available, default_input_name

    specs = load_specs()
    problems = validate_catalog()
    sessions = [p for p in SESSION_DIR.iterdir() if (p / "meta.json").exists()] \
        if SESSION_DIR.exists() else []
    models = list(MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else []
    reports = list(REPORT_DIR.glob("*.json")) if REPORT_DIR.exists() else []
    try:
        from .device.th21a import DEFAULT_ADDRESS
        addr = DEFAULT_ADDRESS
    except Exception:
        addr = "?"
    return [
        ("catalog", f"{len(specs)}/64 specs" + ("" if not problems else f"  ({len(problems)} with problems)")),
        ("headband", f"TH21A at {addr}  (press Scan to confirm it is advertising)"),
        ("microphone", f"{default_input_name() or 'none'}" if mic_available() else "unavailable"),
        ("sessions", f"{len(sessions)} recorded"),
        ("models", f"{len(models)} trained"),
        ("reports", f"{len(reports)} evaluations"),
        ("project", str(ROOT)),
    ]


# ---------------------------------------------------------------- dialog
class RunDialog(tk.Toplevel):
    """Ask which tests to run, and with which options."""

    def __init__(self, master, default_ids: str = "T002 T003"):
        super().__init__(master)
        self.title("Run a session")
        self.transient(master)
        self.resizable(False, False)
        self.result: list[str] | None = None

        frm = ttk.Frame(self, padding=12)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="Test ids (space separated)").grid(row=0, column=0, sticky="w")
        self.ids = tk.StringVar(value=default_ids)
        ttk.Entry(frm, textvariable=self.ids, width=34).grid(row=1, column=0, columnspan=2, sticky="we")

        self.mic = tk.BooleanVar(value=False)
        self.fake = tk.BooleanVar(value=False)
        self.dry = tk.BooleanVar(value=False)
        self.fullscreen = tk.BooleanVar(value=True)
        self.noeval = tk.BooleanVar(value=False)
        self.advance = tk.StringVar(value="key")
        ttk.Checkbutton(frm, text="Record the microphone (--mic, stored locally)",
                        variable=self.mic).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Use the synthetic device (--device fake)",
                        variable=self.fake).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Rehearsal only (--dry-run: markers, no samples)",
                        variable=self.dry).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Fullscreen stimuli (covers the desktop; recommended)",
                        variable=self.fullscreen).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Skip analysis afterwards", variable=self.noeval)\
            .grid(row=6, column=0, columnspan=2, sticky="w")

        box = ttk.LabelFrame(frm, text="Advancing between segments", padding=6)
        box.grid(row=7, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Radiobutton(box, text="Wait for SPACE  (nothing moves on by itself)",
                        variable=self.advance, value="key").pack(anchor="w")
        ttk.Radiobutton(box, text="Auto-advance on a timer  (rehearsal)",
                        variable=self.advance, value="timer").pack(anchor="w")
        ttk.Label(box, foreground="#666", wraplength=320, justify="left",
                  text=("A rehearsal (--dry-run) always uses the timer, and so does any "
                        "run with no display and no keyboard.")).pack(anchor="w")

        ttk.Label(frm, foreground="#666", wraplength=330, justify="left",
                  text=("Put the headband on first: nose-bridge switch, band above the "
                        "eyebrows, no hair under the electrodes. Press Enter when the "
                        "session starts.")).grid(row=8, column=0, columnspan=2,
                                                 sticky="w", pady=(10, 0))
        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Start", command=self._ok).grid(row=0, column=1)
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()

    def _ok(self) -> None:
        ids = self.ids.get().split()
        if not ids:
            messagebox.showwarning("NeuroBridge", "Enter at least one test id, e.g. T002 T003")
            return
        argv = ["run", *ids, "-y"]
        if self.mic.get():
            argv.append("--mic")
        if self.fake.get():
            argv += ["--device", "fake"]
        if self.dry.get():
            argv.append("--dry-run")
        argv.append("--fullscreen" if self.fullscreen.get() else "--windowed")
        argv += ["--advance", self.advance.get()]
        if self.noeval.get():
            argv.append("--no-eval")
        self.result = argv
        self.destroy()


# ---------------------------------------------------------------- main window
class App(ttk.Frame):
    """The control panel."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=0)
        self.master = master
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str | None] = queue.Queue()
        self._on_done: Any = None
        self._log_fh: Any = None
        self._log_path: Path | None = None
        self._is_session = False          # minimise the panel so the stimuli get focus
        self._last_output = 0.0
        self._stall_warned = False
        self._build_header()
        self._build_buttons()
        self._build_console()
        self.refresh_status()
        self.after(80, self._drain)

    # -- layout -----------------------------------------------------
    def _build_header(self) -> None:
        head = ttk.Frame(self, padding=(12, 10))
        head.grid(row=0, column=0, columnspan=2, sticky="we")
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="NeuroBridge", font=("TkDefaultFont", 16, "bold"))\
            .grid(row=0, column=0, sticky="w")
        ttk.Label(head, text="EEG training & brain-to-agent bridge",
                  foreground="#555").grid(row=1, column=0, sticky="w")
        ttk.Button(head, text="Refresh status", command=self.refresh_status)\
            .grid(row=0, column=1, rowspan=2, sticky="e")
        self.status = ttk.Label(self, justify="left", font=("TkFixedFont", 9),
                                padding=(12, 2, 12, 6))
        self.status.grid(row=1, column=0, columnspan=2, sticky="we")

    def _build_buttons(self) -> None:
        side = ttk.Frame(self, padding=(12, 4))
        side.grid(row=2, column=0, sticky="nsw")

        self.buttons: list[ttk.Button] = []

        def group(title: str) -> None:
            ttk.Label(side, text=title, font=("TkDefaultFont", 9, "bold"))\
                .pack(anchor="w", pady=(10, 2))

        def button(text: str, cmd, tip: str = "") -> None:
            b = ttk.Button(side, text=text, width=30, command=cmd)
            b.pack(anchor="w", pady=1)
            if tip:
                _Tooltip(b, tip)
            self.buttons.append(b)

        group("Setup")
        button("Check everything (headband + mic)", lambda: self.run(["doctor", "--scan", "--mic"]),
               "Environment, packages, data tree, catalog, display, audio, microphone, "
               "then a 10 s BLE scan for the headband.")
        button("Quick check (no scan)", lambda: self.run(["doctor"]))
        button("Test the session screen (5 s)", lambda: self.run(["screen-test", "--seconds", "5"]),
               "Shows the presentation surface for five seconds and reports whether it "
               "covers the whole display. Run this before a session if you have ever "
               "seen the desktop behind the stimuli.")
        button("Plan my sessions", lambda: self.run(["plan", "--remaining"]),
               "Packs the tests you have not recorded into ~45 minute sittings.")
        button("Progress", lambda: self.run(["progress"]))

        group("Record")
        button("Run a session…", self.run_session,
               "Pick test ids; the session writes EEG, markers and (optionally) audio.")
        button("First session (T002 T003)", lambda: self.run(["run", "T002", "T003", "-y"]),
               "~11 minutes: resting baselines plus the alpha-reactivity check. "
               "Do this before anything else.")
        button("Run with microphone (T502)", lambda: self.run(["run", "T502", "--mic", "-y"]),
               "Reading aloud vs silent, with voice captured for fluency metrics.")

        group("Monitor")
        button("Open live monitor", lambda: self.launch_monitor([]),
               "Starts the standalone monitor window: live trace, band power, state "
               "fields, and CSV recording. It runs as its own process, so it keeps "
               "running if this panel closes - and it can also be started on its own "
               "with bin/nb-monitor.")
        button("Monitor, attached to the bridge",
               lambda: self.launch_monitor(["--url", "http://127.0.0.1:8731"]),
               "Use this when `nb serve` already holds the headband: the monitor "
               "attaches to the bridge for both state and raw samples instead of "
               "opening the device a second time.")
        button("Replay a recording…", self.launch_monitor_replay,
               "Open a previous session folder (or a monitor eeg.csv) and play it "
               "back with scrubbing and speed control.")

        group("Verify without hardware")
        button("Simulate a test…", lambda: self.prompt_run(["simulate"], "Test id", "T301"),
               "Generates a synthetic session and runs the full analysis. Proves the "
               "pipeline works before you trust it.")
        button("Null-model check (T301)", lambda: self.run(["simulate", "T301", "--effect-size", "0"]),
               "Nothing is injected, so it MUST land at chance. If it decodes, the "
               "pipeline is leaking.")

        group("Analyse")
        button("Evaluate a test…", self.prompt_eval, "Runs the pre-registered gates.")
        button("Evaluate locked (held-out)", self.prompt_locked,
               "Trains on development sessions, scores the frozen locked ones.")
        button("Freeze a split…", self.prompt_split,
               "Marks the newest sessions LOCKED so they can never be fitted on.")
        button("Sessions", lambda: self.run(["sessions", "list"]))

        group("Voice (microphone)")
        button("Record a voice sample…", self.prompt_voice_record,
               "Read something aloud; no headband needed. Gives onset latency, "
               "duration, rate and pauses.")
        button("Analyse last session's voice", lambda: self.run(["voice", "--session", "latest"]))

        group("Agent bridge")
        button("Start the bridge", lambda: self.run(["serve"]),
               "Serves /state, /candidate, /schema and /metrics on 127.0.0.1:8731. "
               "Leave it running; press Stop to shut it down.")
        button("Show the agent contract", lambda: self.open_doc("agent-contract"))

        group("Docs")
        button("Read me", lambda: self.open_doc("README"))
        button("The 64 tests", lambda: self.open_doc("CURRICULUM"))
        button("Verification record", lambda: self.open_doc("VERIFICATION"))
        button("Hardware & electrode prep", lambda: self.open_doc("HARDWARE"))

        group("Logs (for reporting problems)")
        button("Collect diagnostics…", self.collect_diagnostics,
               "Writes one file with versions, settings (no credentials), catalog "
               "state, session summaries, doctor output and recent logs — then puts "
               "its path on the clipboard so you can paste it.")
        button("Save console log…", self.save_log,
               "Saves everything shown in the console to a file you can share.")
        button("Copy console log", self.copy_log,
               "Copies the whole console to the clipboard.")
        button("Open logs folder", self.open_logs_dir)
        button("Copy last session's run.log", self.copy_session_log,
               "The runner writes run.log into each session directory; this copies "
               "the newest one so you can share exactly what the session printed.")
        button("Clear console", self.clear_console)

        ttk.Separator(side).pack(fill="x", pady=10)
        self.stop_btn = ttk.Button(side, text="Stop", width=30, command=self.stop, state="disabled")
        self.stop_btn.pack(anchor="w")
        ttk.Button(side, text="Quit", width=30, command=self.master.destroy).pack(anchor="w", pady=1)

    def _build_console(self) -> None:
        wrap = ttk.Frame(self, padding=(0, 4, 12, 12))
        wrap.grid(row=2, column=1, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(1, weight=1)

        bar = ttk.Frame(wrap)
        bar.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 4))
        ttk.Label(bar, text="Status / output", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.log_label = ttk.Label(bar, text="", foreground="#666")
        self.log_label.pack(side="left", padx=(10, 0))
        ttk.Button(bar, text="Save…", width=8, command=self.save_log).pack(side="right", padx=2)
        ttk.Button(bar, text="Copy", width=8, command=self.copy_log).pack(side="right", padx=2)
        ttk.Button(bar, text="Diagnostics", width=12,
                   command=self.collect_diagnostics).pack(side="right", padx=2)

        self.console = tk.Text(wrap, wrap="none", font=("TkFixedFont", 9),
                               background="#14161a", foreground="#dfe4ea",
                               insertbackground="#dfe4ea", relief="flat")
        self.console.grid(row=1, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=self.console.yview)
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(wrap, orient="horizontal", command=self.console.xview)
        xscroll.grid(row=2, column=0, sticky="we")
        self.console.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.progress = ttk.Progressbar(wrap, mode="indeterminate")
        self.progress.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.console.tag_configure("err", foreground="#ff8a80")
        self.console.tag_configure("ok", foreground="#9ae6b4")
        self.console.tag_configure("cmd", foreground="#7cc4ff")

    # -- status -----------------------------------------------------
    def refresh_status(self) -> None:
        try:
            rows = status_lines()
        except Exception as exc:                     # never break the window
            rows = [("status", f"unavailable: {type(exc).__name__}: {exc}")]
        width = max(len(k) for k, _ in rows)
        self.status.configure(text="\n".join(f"{k.ljust(width)}   {v}" for k, v in rows))

    # -- command running --------------------------------------------
    def launch_monitor(self, extra: list[str]) -> None:
        """Spawn the standalone monitor as its own detached process.

        Deliberately *not* run through self.run(): the monitor is a window that
        should outlive this panel, and routing it through the console runner would
        tie its lifetime to the panel and block every other button while it was
        open. ``start_new_session`` detaches it from this process group so closing
        the panel does not take the recording down with it.
        """
        argv = _nb_argv("monitor", *extra)
        self._write(f"\n$ nb monitor {' '.join(extra)}\n", "cmd")
        try:
            subprocess.Popen(argv, cwd=str(ROOT), env=dict(os.environ),
                             start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._write("[monitor started in its own window; it keeps running if this "
                        "panel closes]\n", "cmd")
        except Exception as exc:
            messagebox.showerror("Monitor", f"could not start the monitor:\n{exc}")

    def launch_monitor_replay(self) -> None:
        from .config import SESSION_DIR
        d = filedialog.askdirectory(title="Choose a recorded session folder",
                                    initialdir=str(SESSION_DIR))
        if d:
            self.launch_monitor(["--replay", d, "--no-record"])

    def run(self, argv: list[str], on_done=None) -> None:
        if self.proc is not None:
            messagebox.showinfo("NeuroBridge", "A command is already running. Press Stop first.")
            return
        self._on_done = on_done
        self._last_output = time.time()
        self._stall_warned = False
        self._write(f"\n$ nb {' '.join(argv)}\n", "cmd")
        # A session opens its own fullscreen pygame window and waits for SPACE
        # there. If this control panel keeps keyboard focus, that SPACE never
        # arrives and the session looks hung, so step out of the way.
        self._is_session = bool(argv) and argv[0] == "run"
        if self._is_session:
            self._write("[the stimulus window takes over; press SPACE there to advance,\n"
                        " q or ESC to abort. This panel is minimised — click its taskbar\n"
                        " entry to get it back.]\n", "cmd")
            try:
                self.master.iconify()
            except Exception:
                pass
        self._open_auto_log(argv)
        env = dict(os.environ)
        env.setdefault("TERM", "dumb")
        env["NO_COLOR"] = "1"                        # keep the console readable
        try:
            self.proc = subprocess.Popen(
                _nb_argv(*argv), cwd=str(ROOT), env=env, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as exc:
            self._write(f"could not start: {exc}\n", "err")
            self.proc = None
            self._close_auto_log()
            return
        self._set_running(True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _open_auto_log(self, argv: list[str]) -> None:
        """Append everything to a dated log, so a problem can be reported later
        even if the user never pressed Save."""
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._log_path = LOG_DIR / f"commands-{time.strftime('%Y%m%d')}.log"
            self._log_fh = self._log_path.open("a", encoding="utf-8")
            s = get_settings()
            self._log_fh.write(
                "\n" + "=" * 78 + "\n"
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] nb {' '.join(argv)}\n"
                f"env: fullscreen={s.fullscreen} advance={getattr(s, 'advance', '?')} "
                f"device={s.device} backend={s.visual_backend} "
                f"python={sys.version.split()[0]} display={os.environ.get('DISPLAY', '-')}\n")
            self._log_fh.flush()
            self.log_label.configure(text=f"logging to {self._log_path.name}")
        except Exception:
            self._log_fh = None

    def _close_auto_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
        self._log_fh = None

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                clean = ANSI_RE.sub("", line)
                self.q.put(clean)
                if self._log_fh is not None:
                    try:
                        self._log_fh.write(f"{time.strftime('%H:%M:%S')} {clean}")
                        self._log_fh.flush()
                    except Exception:
                        pass
        except Exception:
            pass
        code = self.proc.wait() if self.proc else None
        self.q.put(None)
        self.q.put(f"[exit {code}]\n")

    def _drain(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    self.proc = None
                    self._set_running(False)
                    self._close_auto_log()
                    if self._is_session:
                        self._is_session = False
                        try:
                            self.master.deiconify()
                            self.master.lift()
                        except Exception:
                            pass
                    self.refresh_status()
                    cb, self._on_done = self._on_done, None
                    if cb is not None:
                        try:
                            cb()
                        except Exception as exc:
                            self._write(f"[callback failed: {exc}]\n", "err")
                    continue
                self._last_output = time.time()
                self._stall_warned = False
                self._write(item, "err" if "error" in item.lower() or "FAIL" in item else None)
        except queue.Empty:
            pass
        # A session waiting for SPACE produces no output at all. Say so once,
        # rather than leaving the user staring at a still screen.
        if (self.proc is not None and self._is_session and not self._stall_warned
                and time.time() - self._last_output > 90.0):
            self._stall_warned = True
            self._write("\n[hint] no output for 90 s. A session in `--advance key` mode "
                        "waits for SPACE in its own window — click that window and press "
                        "SPACE (or q to abort). If it never gets focus, restart with "
                        "`--advance timer`.\n", "err")
        self.after(80, self._drain)

    def _write(self, text: str, tag: str | None = None) -> None:
        self.console.insert("end", text, tag or ())
        self.console.see("end")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for b in self.buttons:
            b.configure(state=state)
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start(60)
        else:
            self.progress.stop()

    def stop(self) -> None:
        if self.proc is None:
            return
        self._write("\n[stopping…]\n", "err")
        try:
            self.proc.terminate()
            threading.Timer(4.0, lambda: self.proc and self.proc.kill()).start()
        except Exception:
            pass

    # -- logs (so a problem can be reported) ------------------------
    def console_text(self) -> str:
        return self.console.get("1.0", "end-1c")

    def save_log(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        default = str(LOG_DIR / f"console-{time.strftime('%Y%m%d-%H%M%S')}.log")
        path = filedialog.asksaveasfilename(
            parent=self, title="Save the console log", initialdir=str(LOG_DIR),
            initialfile=Path(default).name, defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*")])
        if not path:
            return
        body = self._log_with_header()
        Path(path).write_text(body, encoding="utf-8")
        self._write(f"[saved console log to {path}]\n", "ok")
        messagebox.showinfo("NeuroBridge", f"Saved to:\n{path}")

    def copy_log(self) -> None:
        body = self._log_with_header()
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(body)
            self.master.update_idletasks()
            self._write("[console log copied to the clipboard]\n", "ok")
        except Exception as exc:
            messagebox.showerror("NeuroBridge", f"could not copy: {exc}")

    def _log_with_header(self) -> str:
        """Console contents plus the header that makes a paste actionable."""
        import platform
        from . import __version__
        s = get_settings()
        head = [
            "=" * 78,
            f"NeuroBridge console log   ({__version__})",
            f"copied      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"host        : {platform.node()}  {platform.platform()}",
            f"python      : {sys.version.split()[0]}",
            f"display     : {os.environ.get('DISPLAY', '(unset)')}  "
            f"desktop={os.environ.get('XDG_CURRENT_DESKTOP', '?')}",
            f"settings    : fullscreen={s.fullscreen} advance={getattr(s, 'advance', '?')} "
            f"device={s.device} visual={s.visual_backend} audio={s.audio_backend}",
            "-" * 78,
            "For a fuller bundle use 'Collect diagnostics' (writes one file with",
            "versions, settings without credentials, catalog state, session",
            "summaries, doctor output and recent logs).",
            "=" * 78,
            "",
        ]
        return "\n".join(head) + self.console_text()

    def clear_console(self) -> None:
        self.console.delete("1.0", "end")

    def open_logs_dir(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(LOG_DIR)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("NeuroBridge", f"could not open {LOG_DIR}:\n{exc}")

    def copy_session_log(self) -> None:
        """Copy the newest session's run.log — what the session itself printed."""
        files = sorted(SESSION_DIR.glob("*/run.log"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \
            if SESSION_DIR.exists() else []
        if not files:
            messagebox.showinfo("NeuroBridge",
                                "No session has a run.log yet.\n\n"
                                "run.log is written by the runner during a session "
                                "(sessions recorded before this feature have only "
                                "meta.json, summary.json and events.jsonl).")
            return
        newest = files[0]
        try:
            body = newest.read_text(encoding="utf-8", errors="replace")
            self.master.clipboard_clear()
            self.master.clipboard_append(body)
            self.master.update_idletasks()
        except Exception as exc:
            messagebox.showerror("NeuroBridge", f"could not read {newest}:\n{exc}")
            return
        self._write(f"[copied {newest} to the clipboard ({len(body)} bytes)]\n", "ok")
        if messagebox.askyesno("NeuroBridge",
                               f"Copied the newest session log to the clipboard:\n{newest}\n\n"
                               f"Open its folder?"):
            try:
                subprocess.Popen(["xdg-open", str(newest.parent)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def collect_diagnostics(self) -> None:
        note = simpledialog.askstring(
            "Collect diagnostics",
            "What went wrong? (optional, helps when we look at the file)",
            parent=self)
        if note is None:                                # cancelled
            return
        argv = ["diagnose", "--mic"]
        if note.strip():
            argv += ["--note", note.strip()]
        self.run(argv, on_done=self._after_diagnostics)

    def _after_diagnostics(self) -> None:
        """Put the newest bundle's path on the clipboard and offer to reveal it."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(LOG_DIR.glob("diagnostic-*.txt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            messagebox.showwarning("NeuroBridge", "no diagnostic file was produced")
            return
        newest = files[-1]
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(str(newest))
            self.master.update_idletasks()
            self._write(f"[diagnostic path copied to the clipboard: {newest}]\n", "ok")
        except Exception:
            pass
        if messagebox.askyesno(
                "Diagnostics ready",
                f"Wrote:\n{newest}\n\nIts path is on the clipboard. Open the folder now?"):
            self.open_logs_dir()

    # -- prompts ----------------------------------------------------
    def prompt_run(self, base: list[str], label: str, default: str) -> None:
        """Ask for one argument and run ``base + [that argument]``."""
        val = simpledialog.askstring("NeuroBridge", f"{label}:", initialvalue=default, parent=self)
        if val and val.strip():
            self.run([*base, val.strip()])

    def run_session(self) -> None:
        dlg = RunDialog(self, default_ids="T002 T003")
        self.wait_window(dlg)
        if dlg.result:
            self.run(dlg.result)

    def prompt_eval(self) -> None:
        tid = simpledialog.askstring("NeuroBridge", "Which test id?",
                                     initialvalue="T003", parent=self)
        if not tid:
            return
        sess = simpledialog.askstring("NeuroBridge",
                                      "Session id (blank = latest):", initialvalue="", parent=self)
        argv = ["eval", tid.strip()]
        if sess and sess.strip():
            argv += ["--session", sess.strip()]
        self.run(argv)

    def prompt_locked(self) -> None:
        tid = simpledialog.askstring("NeuroBridge", "Which test id (held-out evaluation)?",
                                     initialvalue="T201", parent=self)
        if tid:
            self.run(["eval", tid.strip(), "--locked"])

    def prompt_split(self) -> None:
        tid = simpledialog.askstring("NeuroBridge", "Freeze a split for which test id?",
                                     initialvalue="T003", parent=self)
        if tid:
            self.run(["split", "create", "--spec", tid.strip()])

    def prompt_voice_record(self) -> None:
        secs = simpledialog.askstring("NeuroBridge", "Record for how many seconds?",
                                      initialvalue="30", parent=self)
        if not secs:
            return
        self.run(["voice", "--record", secs.strip()])

    def open_doc(self, which: str) -> None:
        paths = {
            "README": ROOT / "README.md",
            "CURRICULUM": DOCS_DIR / "CURRICULUM.md",
            "VERIFICATION": DOCS_DIR / "VERIFICATION.md",
            "HARDWARE": DOCS_DIR / "HARDWARE.md",
            "agent-contract": ROOT / "neurobridge" / "agent" / "protocol.md",
        }
        p = paths.get(which)
        if p is None or not p.exists():
            messagebox.showwarning("NeuroBridge", f"document not found: {which}")
            return
        try:
            subprocess.Popen(["xdg-open", str(p)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("NeuroBridge", f"could not open {p}:\n{exc}")


class _Tooltip:
    """Minimal hover tooltip (ttk has none)."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500) -> None:
        self.widget, self.text, self.delay = widget, text, delay
        self.tip: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    def _enter(self, _e=None) -> None:
        self.after_id = self.widget.after(self.delay, self._show)

    def _leave(self, _e=None) -> None:
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None

    def _show(self) -> None:
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        y = self.widget.winfo_rooty()
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=340,
                 font=("TkDefaultFont", 8)).pack()


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m neurobridge.gui`` / ``nb gui``."""
    try:
        # className sets WM_CLASS, which is what the panel matches against
        # StartupWMClass in the .desktop entry -- without it the window gets a
        # generic taskbar icon instead of ours.
        root = tk.Tk(className="NeuroBridge")
    except tk.TclError as exc:
        print(f"no display available for the GUI ({exc}).\n"
              f"Use the command line instead: ./bin/nb --help", file=sys.stderr)
        return 2
    root.title(APP_TITLE)
    root.geometry("1180x720")
    root.minsize(980, 560)
    style = ttk.Style()
    for theme in ("clam", "alt", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break
    App(root)
    # the window icon: applied now for the WM's initial mapping, and again once
    # the window is realised, because some WMs/WMs-versions only honour
    # _NET_WM_ICON after the toplevel has been mapped.
    def _apply_icon() -> None:
        try:
            icon = ROOT / "assets" / "neurobridge-256.png"
            if icon.exists():
                img = tk.PhotoImage(file=str(icon))
                root.iconphoto(True, img)
                root._nb_icon = img                 # keep a reference alive
        except Exception:
            pass
    _apply_icon()
    root.after(300, _apply_icon)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
