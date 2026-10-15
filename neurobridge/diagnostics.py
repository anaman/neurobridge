"""Collect a single shareable diagnostic bundle.

The point is to make "here is what went wrong" cheap: one text file that
contains everything needed to reproduce or explain a problem, and nothing that
should not be shared.

Two rules:

* **Credentials never leave the machine.** API tokens are reported as
  ``present`` / ``absent``, never by value, and the settings dump is filtered
  rather than copied.
* **Read-only.** Collection must not change state: it reads logs and session
  metadata, and only *writes* the one report file. It never touches session
  data (the user asked for updates without data deletion, and a diagnostic that
  mutates the thing it is diagnosing is worse than useless).
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

from .config import (DATA_DIR, LOG_DIR, MODEL_DIR, REPORT_DIR, ROOT, SESSION_DIR,
                     get_settings)

#: Settings whose *values* are secrets: report presence only.
SECRET_KEYS = ("freesound_token", "openverse_client_secret", "openverse_client_id")

#: Lines of each log tail to include.
LOG_TAIL = 120


def _versions() -> dict[str, str]:
    mods = ["numpy", "scipy", "sklearn", "mne", "torch", "pygame", "sounddevice",
            "bleak", "fastapi", "uvicorn", "pylsl", "braindecode", "matplotlib",
            "soundfile", "pandas", "httpx"]
    out: dict[str, str] = {}
    for m in mods:
        try:
            mod = __import__(m)
            out[m] = str(getattr(mod, "__version__", "?"))
        except Exception as exc:                     # missing is information too
            out[m] = f"MISSING ({type(exc).__name__})"
    return out


def _safe_settings() -> dict[str, Any]:
    s = get_settings()
    out: dict[str, Any] = {}
    for k, v in s.as_dict().items():
        if k in SECRET_KEYS:
            out[k] = "present (value withheld)" if str(v or "").strip() else "absent"
        else:
            out[k] = v
    return out


def _session_rows(limit: int = 40) -> list[dict[str, Any]]:
    if not SESSION_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    from . import store
    dirs = sorted((p for p in SESSION_DIR.iterdir() if (p / "meta.json").exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for d in dirs:
        try:
            s = store.Session.load(d)
            summary = s.summary()
            rows.append({
                "session": d.name,
                "tests": s.test_ids(),
                "duration_s": round(s.duration_s, 1),
                "samples": int(s.eeg.size),
                "events": len(s.events),
                "quality": summary.get("quality_verdict"),
                "quality_score": summary.get("quality_score"),
                "fs_effective": summary.get("fs_effective"),
                "timing_quality": s.meta.get("timing_quality"),
                "audio": s.has_audio(),
                "fullscreen": s.meta.get("run", {}).get("fullscreen")
                if isinstance(s.meta.get("run"), dict) else s.meta.get("fullscreen"),
                "aborted": "ABORTED" in s.notes.upper(),
            })
        except Exception as exc:
            rows.append({"session": d.name, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _log_tails(limit_files: int = 6) -> dict[str, list[str]]:
    if not LOG_DIR.exists():
        return {}
    out: dict[str, list[str]] = {}
    files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:limit_files]:
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception:
            continue
        out[p.name] = lines[-LOG_TAIL:]
    return out


def _recent_problems(scan_lines: int = 4000) -> list[str]:
    """Pull traceback blocks out of the logs, because that is what gets shared."""
    if not LOG_DIR.exists():
        return []
    found: list[str] = []
    for p in sorted(LOG_DIR.glob("*.log"), key=lambda q: q.stat().st_mtime, reverse=True)[:4]:
        try:
            lines = p.read_text(errors="replace").splitlines()[-scan_lines:]
        except Exception:
            continue
        i = 0
        while i < len(lines):
            if "Traceback (most recent call last)" in lines[i]:
                block = lines[i:i + 25]
                found.append(f"--- {p.name} ---\n" + "\n".join(block))
                i += 25
            else:
                i += 1
    return found[-6:]


def _doctor_text(scan: bool, mic: bool) -> str:
    """Run the doctor check and capture its output (it is written for humans)."""
    import argparse
    import io
    from contextlib import redirect_stdout
    from .cli import cmd_doctor
    buf = io.StringIO()
    ns = argparse.Namespace(scan=scan, mic=mic, verbose=False)
    try:
        with redirect_stdout(buf):
            cmd_doctor(ns)
    except Exception as exc:
        buf.write(f"\ndoctor failed: {type(exc).__name__}: {exc}\n")
    return buf.getvalue()


def collect(scan: bool = False, mic: bool = False, note: str = "",
            out_dir: Path | None = None) -> Path:
    """Write a diagnostic bundle and return its path."""
    out_dir = Path(out_dir or LOG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"diagnostic-{stamp}.txt"

    from . import __version__
    from .catalog import load_specs, validate_catalog

    L: list[str] = []
    add = L.append

    add("=" * 78)
    add("NeuroBridge diagnostic bundle")
    add("=" * 78)
    add(f"collected     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"version       : {__version__}")
    add(f"note from user: {note or '(none)'}")
    add(f"host          : {socket.gethostname()}  ({platform.machine()})")
    add(f"os            : {platform.platform()}")
    add(f"python        : {sys.version.split()[0]}  at {sys.executable}")
    add(f"project       : {ROOT}")
    add(f"load average  : {os.getloadavg() if hasattr(os, 'getloadavg') else 'n/a'}")
    add(f"cpus          : {os.cpu_count()}")
    add("")

    add("-" * 78)
    add("session environment (display / desktop)")
    add("-" * 78)
    for k in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP",
              "DESKTOP_SESSION", "TERM", "SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        add(f"  {k:20s} = {os.environ.get(k, '(unset)')}")
    add("")

    add("-" * 78)
    add("package versions")
    add("-" * 78)
    for k, v in _versions().items():
        add(f"  {k:14s} {v}")
    add("")

    add("-" * 78)
    add("settings (secrets withheld)")
    add("-" * 78)
    for k, v in sorted(_safe_settings().items()):
        add(f"  {k:26s} {v}")
    add("")

    add("-" * 78)
    add("catalog")
    add("-" * 78)
    specs = load_specs()
    problems = validate_catalog()
    add(f"  specs loaded   : {len(specs)}")
    add(f"  invalid specs  : {len(problems)}")
    for tid, errs in list(problems.items())[:10]:
        add(f"    {tid}: {errs[0]}")
    add("")

    add("-" * 78)
    add(f"doctor output (scan={scan}, mic_probe={mic})")
    add("-" * 78)
    add(_doctor_text(scan, mic))
    add("")

    add("-" * 78)
    add("sessions (newest first)")
    add("-" * 78)
    for r in _session_rows():
        add("  " + json.dumps(r, default=str))
    add("")
    add(f"  models : {len(list(MODEL_DIR.glob('*'))) if MODEL_DIR.exists() else 0}")
    add(f"  reports: {len(list(REPORT_DIR.glob('*'))) if REPORT_DIR.exists() else 0}")
    add(f"  data   : {DATA_DIR}")
    add("")

    tails = _log_tails()
    if tails:
        add("-" * 78)
        add("recent logs (tail)")
        add("-" * 78)
        for name, lines in tails.items():
            add(f"  === {name} (last {len(lines)} lines) ===")
            L.extend("  " + ln for ln in lines)
            add("")

    probs = _recent_problems()
    if probs:
        add("-" * 78)
        add("recent tracebacks found in the logs")
        add("-" * 78)
        for block in probs:
            L.extend(block.splitlines())
            add("")

    add("-" * 78)
    add("what to do with this file")
    add("-" * 78)
    add("  Attach or paste this whole file when reporting a problem. It contains")
    add("  versions, settings (no credentials), catalog state, session summaries,")
    add("  the doctor output and the tails of recent logs.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path
