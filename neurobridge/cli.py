#!/usr/bin/env python3
"""NeuroBridge command line.

    nb doctor                     environment / hardware / credential check
    nb catalog list|show|validate|stats
    nb fetch T601 --limit 4       resolve (and download/generate) stimuli
    nb run T002 T003              live session (real headband, or --device fake)
    nb run T301 --dry-run         timing-only rehearsal, no stimulus, no samples
    nb simulate T301              offline synthetic session + full analysis (no hardware)
    nb train T201                 train a decoder from recorded sessions
    nb eval T301                  evaluate a test against its acceptance gates
    nb report --curriculum        progress across all 64 tests
    nb serve                      brain-to-agent HTTP/SSE/WebSocket bridge
    nb link status|now|watch|record|report|mark
                                  agent-side link to a running bridge
    nb sessions list|show|export
    nb state                      one-shot live state estimate

Everything degrades instead of failing: no headband -> the synthetic source, no
network -> locally generated stimuli, no display -> terminal presentation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import (CACHE_DIR, DOCS_DIR, LOG_DIR, MODEL_DIR, REPORT_DIR, ROOT,
                     SESSION_DIR, ensure_dirs, get_settings)

# rich is a hard dependency but never let its absence break a session
try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
except Exception:                       # pragma: no cover
    _console = None


# ---------------------------------------------------------------- output helpers
def say(msg: str = "") -> None:
    if _console:
        _console.print(msg)
    else:
        print(msg)


def head(title: str) -> None:
    say(f"\n[bold]{title}[/bold]" if _console else f"\n=== {title} ===")


def table(title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    if _console:
        t = Table(title=title, show_lines=False, header_style="bold")
        for c in columns:
            t.add_column(str(c))
        for r in rows:
            t.add_row(*[str(x) for x in r])
        say(t)
    else:
        say(title)
        say("  " + " | ".join(str(c) for c in columns))
        for r in rows:
            say("  " + " | ".join(str(x) for x in r))


def ok(msg: str, flag: bool = True) -> str:
    mark = "[green]OK[/green]" if flag else "[red]FAIL[/red]"
    return f"{mark} {msg}" if _console else f"{'OK  ' if flag else 'FAIL'} {msg}"


def emit_json(obj: Any) -> None:
    """Write JSON straight to stdout, bypassing rich.

    rich soft-wraps to the terminal width and will insert a newline *inside* a
    JSON string literal, which produces a document no parser accepts. Anything
    another program is meant to read must not go through the console.
    """
    sys.stdout.write(json.dumps(obj, indent=2, default=str) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- session resolution
def resolve_session(arg: str | None):
    """Accept a session id, a path, 'latest', or None (= latest)."""
    from . import store
    if arg in (None, "", "latest"):
        return store.Session.latest()
    p = Path(arg)
    if p.is_dir():
        return store.Session.load(p)
    cand = SESSION_DIR / arg
    if cand.is_dir():
        return store.Session.load(cand)
    matches = sorted(d for d in SESSION_DIR.glob(f"*{arg}*") if d.is_dir())
    if len(matches) == 1:
        return store.Session.load(matches[0])
    if not matches:
        raise SystemExit(f"no session matching {arg!r}")
    raise SystemExit(f"{arg!r} is ambiguous: {[m.name for m in matches]}")


# ---------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    head(f"NeuroBridge {__version__} - environment check")
    checks: list[str] = []
    settings = get_settings()
    ensure_dirs()

    # 1. python + deps
    import numpy
    say(ok(f"python {sys.version.split()[0]}  numpy {numpy.__version__}  at {sys.executable}"))
    optional = {
        "scipy": "filters", "sklearn": "decoders", "mne": "spectral analysis",
        "matplotlib": "reports", "soundfile": "audio stimuli", "pygame": "visual presentation",
        "sounddevice": "precise audio timing", "bleak": "headband BLE", "fastapi": "agent bridge",
        "torch": "EEGNet decoder", "pylsl": "lab streaming layer", "yaml": "catalog",
    }
    missing = []
    for mod, why in optional.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(f"{mod} ({why})")
    say(ok("all optional packages present", not missing) if not missing
        else ok(f"missing optional packages: {', '.join(missing)}", False))

    # 2. data tree
    say(ok(f"data tree at {ROOT / 'data'}"))
    for name, d in (("sessions", SESSION_DIR), ("models", MODEL_DIR), ("reports", REPORT_DIR)):
        n = len(list(d.glob("*"))) if d.exists() else 0
        say(f"      {name}: {n}")

    # 3. catalog
    from .catalog import validate_catalog, load_specs
    specs = load_specs()
    problems = validate_catalog()
    say(ok(f"catalog: {len(specs)}/64 test specs, {len(problems)} file(s) with problems",
           len(specs) == 64 and not problems))
    for tid, errs in list(problems.items())[:6]:
        say(f"      [yellow]{tid}[/yellow]: {errs[0]}" if _console else f"      {tid}: {errs[0]}")

    # 4. display / audio
    import os
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    try:
        import pygame
        pygame.display.init()
        drv = pygame.display.get_driver()
        pygame.display.quit()
        pygame.font.init()
        say(ok(f"display: {drv} (DISPLAY={os.environ.get('DISPLAY')})", has_display))
    except Exception as exc:
        say(ok(f"display unavailable ({type(exc).__name__}) - terminal presentation will be used", False))
    try:
        import sounddevice as sd
        outs = [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
        lat = sd.query_devices(kind="output")["default_low_output_latency"]
        say(ok(f"audio out: {len(outs)} output device(s), default latency {lat*1000:.0f} ms"))
        settings.output_latency_offset_s = settings.output_latency_offset_s or 0.0
    except Exception as exc:
        say(ok(f"sounddevice unavailable ({type(exc).__name__}) - falling back to aplay", False))

    # 4b. microphone (voice channel for reading aloud, spoken responses, fluency)
    try:
        from .voice import list_input_devices, probe
        ins = list_input_devices()
        if ins:
            dflt = next((d for d in ins if d["default"]), ins[0])
            say(ok(f"microphone: {dflt['name']} ({len(ins)} input device(s), "
                   f"input latency ~{dflt['input_latency_ms']:.0f} ms)"))
            for d in ins[:3]:
                say(f"      [{d['index']}] {d['name']}  {d['channels']} ch"
                    + ("  (default)" if d["default"] else ""))
        else:
            say(ok("no microphone input device found - voice capture unavailable", False))
        if getattr(args, "mic", False):
            say("      probing the microphone for 2 s (this listens to the room) ...")
            p = probe(2.0)
            if p.get("ok"):
                say(ok(f"mic probe: {p['verdict']} - rms {p['rms_dbfs']} dBFS, "
                       f"peak {p['peak']}, clipped {p['clipped_pct']}%"))
                if not p.get("kept"):
                    say("      (the 2 s sample was discarded; pass keep via nb voice --record "
                        "if you want one saved)")
            else:
                say(ok(f"mic probe failed: {p.get('error')}", False))
    except Exception as exc:
        say(ok(f"microphone check failed: {type(exc).__name__}: {exc}", False))

    # 5. headband
    if args.scan:
        import asyncio
        from .device.th21a import find_headband
        say("    scanning for the headband (10 s) ...")
        dev = asyncio.run(find_headband(timeout=10.0))
        if dev is None:
            say(ok("headband not advertising - long-press power 3 s, then retry", False))
        else:
            say(ok(f"headband found: {dev.name} [{dev.address}]"))
    else:
        say("    (BLE scan skipped - pass --scan to probe the headband)")

    # 6. credentials + network
    from .config import env_token
    fs_tok = env_token("NB_FREESOUND_TOKEN")
    say(ok("Freesound token present (real audio available)", True) if fs_tok
        else ok("no NB_FREESOUND_TOKEN - audio falls back to synthesized stimuli", False))
    try:
        import httpx
        with httpx.Client(timeout=4.0) as c:
            r = c.get("https://api.openverse.org/v1/images/", params={"q": "test", "page_size": 1})
        say(ok(f"Openverse reachable (HTTP {r.status_code})", r.status_code == 200))
    except Exception as exc:
        say(ok(f"Openverse unreachable ({type(exc).__name__}) - builtin images will be used", False))

    # 7. ffmpeg / attribution
    import shutil
    say(ok("ffmpeg present (mp3 -> wav conversion)", bool(shutil.which("ffmpeg")))
        if shutil.which("ffmpeg") else
        ok("ffmpeg missing - Freesound mp3 previews will be used as-is", False))

    head("Next steps")
    say("  1. nb catalog list                 see all 64 tests")
    say("  2. put on the headband: nose-bridge switch, band above the eyebrows, no hair under the electrodes")
    say("  3. nb run T002 T003                first hardware session (~8 min)")
    say("  4. nb simulate T301                prove the analysis pipeline works before you trust it")
    return 0


# ---------------------------------------------------------------- catalog
def cmd_catalog(args: argparse.Namespace) -> int:
    from .catalog import TIERS, load_catalog, load_one, load_specs, validate_catalog
    specs = load_specs()
    if args.action == "validate":
        problems = validate_catalog()
        if not problems:
            say(ok(f"all {len(specs)} specs valid"))
            return 0
        for tid, errs in problems.items():
            say(f"[red]{tid}[/red]" if _console else tid)
            for e in errs:
                say(f"    - {e}")
        return 1
    if args.action == "list":
        for tier, name in TIERS.items():
            rows = [s for s in specs if s.tier == tier]
            if not rows:
                continue
            head(f"Tier {tier} - {name}  ({len(rows)} tests)")
            table("", ["id", "feasibility", "min", "modality", "classes", "agent", "title"],
                  [[s.id, s.feasibility, f"{s.duration_min:.0f}", ",".join(s.modality),
                    ",".join(s.classes), s.agent_use, s.title] for s in rows])
        total = sum(s.duration_min for s in specs)
        say(f"\n{len(specs)} tests, {total:.0f} min of recording in total "
            f"(~{total/45:.0f} sessions at 45 min each)")
        return 0
    if args.action == "stats":
        from collections import Counter
        from .simulate import estimate_duration_min
        c = Counter(s.feasibility for s in specs)
        table("Feasibility", ["band", "count"], [[k, c.get(k, 0)] for k in
              ("established", "promising", "exploratory", "control")])
        c2 = Counter(m for s in specs for m in s.modality)
        table("Modality", ["modality", "tests"], sorted(c2.items()))
        c3 = Counter(s.agent_use for s in specs)
        table("Agent use", ["use", "tests"], sorted(c3.items()))
        c4 = Counter(s.tier for s in specs)
        table("Tier", ["tier", "tests"], sorted(c4.items()))
        est = {s.id: estimate_duration_min(s) for s in specs}
        declared = sum(v[0] for v in est.values())
        mechanical = sum(v[1] for v in est.values())
        planning = sum(v[2] for v in est.values())
        table("Duration (minutes)", ["stat", "declared", "design-derived", "planning"], [
            ["total", f"{declared:.0f}", f"{mechanical:.0f}", f"{planning:.0f}"],
            ["mean per test", f"{declared/len(specs):.1f}", f"{mechanical/len(specs):.1f}",
             f"{planning/len(specs):.1f}"],
            ["longest", f"{max(v[0] for v in est.values()):.0f}",
             f"{max(v[1] for v in est.values()):.0f}",
             f"{max(v[2] for v in est.values()):.0f}"],
            ["sittings @45 min", f"{declared/45:.1f}", f"{mechanical/45:.1f}", f"{planning/45:.1f}"],
        ])
        say("planning books the larger of declared and design-derived; "
            "see `nb catalog audit` for the per-test disagreements")
        return 0
    # show
    if args.action == "docs":
        return _write_curriculum_doc(specs)
    if args.action == "audit":
        return _audit_catalog(specs)
    for tid in args.ids:
        s = load_one(tid)
        head(f"{s.id} - {s.title}")
        say(f"[bold]tier[/bold] {s.tier} ({s.tier_name})   "
            f"[bold]feasibility[/bold] {s.feasibility}   "
            f"[bold]agent_use[/bold] {s.agent_use}" if _console else
            f"tier {s.tier} ({s.tier_name}) | {s.feasibility} | agent_use={s.agent_use}")
        say(f"goal: {s.goal.strip()}")
        table("Design", ["trials", "blocks", "minutes", "classes"],
              [[s.trials, s.design.get("blocks"), f"{s.duration_min:.1f}", ", ".join(s.classes)]])
        table("Processing", ["epoch", "bandpass", "features", "decoder"],
              [[str(s.epoch), str(s.processing.get("bandpass")),
                ", ".join(map(str, s.features)), json.dumps(s.decoder)]])
        table("Gates", list(s.acceptance.keys()), [list(s.acceptance.values())])
        say("[bold]risks[/bold]" if _console else "risks")
        for r in s.risks:
            say(f"    - {r}")
        if s.source_path:
            say(f"\nfile: {s.source_path}")
    return 0


# ---------------------------------------------------------------- fetch
def cmd_fetch(args: argparse.Namespace) -> int:
    from .catalog import load_specs
    from .stimuli import prepare
    specs = load_specs()
    ids = args.ids or [s.id for s in specs]
    rows = []
    for tid in ids:
        s = next((x for x in specs if x.id == tid or x.slug == tid), None)
        if s is None:
            say(f"[yellow]no such test {tid}[/yellow]" if _console else f"no such test {tid}")
            continue
        if not s.needs_stimuli:
            rows.append([s.id, "none", 0, "device-only test"])
            continue
        ss = prepare(s, allow_download=not args.offline, force=args.force,
                     verbose=args.verbose, limit=args.limit or 0 or None)
        problems = ss.check()
        rows.append([s.id, s.stimuli.get("source", "?"), len(ss),
                     "; ".join(ss.warnings[:2]) or ("ok" if not problems else "; ".join(problems[:2]))])
    table("Stimuli", ["test", "source", "items", "notes"], rows)
    say(f"\ncache: {CACHE_DIR}")
    say("licences/attribution: data/stimuli/ATTRIBUTION.md")
    return 0


# ---------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    from .catalog import load_one
    from .runner import RunConfig, run_sequence, run_test
    settings = get_settings()
    ids = args.ids
    if args.fullscreen:
        settings.fullscreen = True
    if getattr(args, "windowed", False):
        settings.fullscreen = False
    if args.no_display:
        settings.visual_backend = "terminal"
    for tid in ids:
        s = load_one(tid)
        say(f"[bold]{s.id}[/bold] {s.title}  ({s.feasibility}, ~{s.duration_min:.0f} min, "
            f"{len(s.classes)} classes)" if _console else
            f"{s.id} {s.title} ({s.feasibility}, ~{s.duration_min:.0f} min)")
    planned = sum(load_one(t).duration_min for t in ids)
    say(f"planned recording: {planned:.0f} min (budget ~45 min)")
    if not args.yes:
        try:
            input("press Enter to start, Ctrl-C to abort > ")
        except KeyboardInterrupt:
            say("aborted")
            return 130
    # --mic is passed through only when this runner build supports it, so the
    # CLI keeps working against an older RunConfig
    from dataclasses import fields as _dc_fields
    supported = {f.name for f in _dc_fields(RunConfig)}
    extra: dict[str, Any] = {}
    # fullscreen is the default now (settings.fullscreen); --windowed opts out
    fs_choice: bool | None = None
    if getattr(args, "fullscreen", False):
        fs_choice = True
    if getattr(args, "windowed", False):
        fs_choice = False
    if getattr(args, "advance", None) and "advance" in supported:
        extra["advance"] = args.advance
        if args.advance == "timer":
            say("advance: timer - segments will move on by themselves "
                "(use the default `--advance key` to wait for SPACE)")
        else:
            say("advance: key - each instruction, cue and break waits for SPACE")
    if getattr(args, "mic", False):
        if "mic" in supported:
            extra["mic"] = True
            if getattr(args, "mic_device", None) and "mic_device" in supported:
                extra["mic_device"] = args.mic_device
            say("microphone: ON - audio.wav is written into the session directory and "
                "stays on this machine; voice metrics never enter the EEG features")
        else:
            say(ok("--mic requested but this runner build has no microphone support yet; "
                   "record separately with `nb voice --record <seconds>`", False))
    common: dict[str, Any] = dict(device=args.device, session_id=args.session,
                                  dry_run=args.dry_run, duration_scale=args.scale,
                                  verbose=not args.quiet, limit=args.limit or None,
                                  allow_download=not args.offline, **extra)
    if "fullscreen" in supported:
        common["fullscreen"] = fs_choice
    common = {k: v for k, v in common.items() if k in supported}
    if len(ids) == 1:
        cfg = RunConfig(spec=load_one(ids[0]), **common)
        path = run_test(cfg)
    else:
        path = run_sequence(ids, **common)
    say(f"\nsession written to {path}")
    if not args.no_eval and not args.dry_run:
        return _evaluate_all(path, ids, permutations=0, quiet=False)
    return 0


# ---------------------------------------------------------------- simulate
def cmd_simulate(args: argparse.Namespace) -> int:
    from .catalog import load_one
    from .simulate import simulate_test
    s = load_one(args.id)
    say(f"simulating {s.id} {s.title} "
        f"(effect_size={args.effect_size}, seed={args.seed}, scale={args.scale})")
    path = simulate_test(s, effect_size=args.effect_size, seed=args.seed,
                         scale=args.scale, seed_note=args.note)
    say(f"synthetic session: {path}")
    return _evaluate_all(path, [s.id], permutations=args.permutations, quiet=False)


# ---------------------------------------------------------------- train / eval
def _session_paths(args: argparse.Namespace, test_id: str) -> list[Path]:
    from . import store, splits
    name = getattr(args, "split", None) or "default"
    force = bool(getattr(args, "force", False))

    if getattr(args, "session", None):
        p = resolve_session(args.session).path
        try:
            splits.check_not_locked([p.name], name, force=force)
        except splits.LockedSessionError as exc:
            raise SystemExit(str(exc))
        return [p]

    # a frozen split, if one exists, decides what may be fitted on
    try:
        dev = splits.dev_sessions(name)
    except FileNotFoundError:
        dev = []
    if dev:
        usable = []
        for p in dev:
            try:
                if test_id in store.Session.load(p).test_ids():
                    usable.append(p)
            except Exception:
                continue
        if usable:
            say(f"using {len(usable)} development session(s) from split {name!r} "
                f"({len(splits.load_split(name).locked)} locked, never fitted on)")
            return usable

    if getattr(args, "all_sessions", False):
        out = []
        for d in sorted(SESSION_DIR.iterdir()):
            if not (d / "meta.json").exists():
                continue
            try:
                s = store.Session.load(d)
            except Exception:
                continue
            if test_id in s.test_ids():
                out.append(d)
        return out
    s = store.Session.latest()
    if test_id in s.test_ids():
        return [s.path]
    # look for the most recent session containing the test
    for d in sorted(SESSION_DIR.iterdir(), reverse=True):
        if not (d / "meta.json").exists():
            continue
        try:
            ss = store.Session.load(d)
        except Exception:
            continue
        if test_id in ss.test_ids():
            return [d]
    return []


def _evaluate_all(session_path: Path | str, ids: Sequence[str],
                  permutations: int = 0, quiet: bool = False) -> int:
    from .catalog import load_one
    from .pipeline import evaluate as E
    from . import store
    failures = 0
    for tid in ids:
        spec = load_one(tid)
        session = store.Session.load(session_path)
        if tid not in session.test_ids():
            say(ok(f"{tid}: no data in this session", False))
            continue
        try:
            res = E.evaluate_session(session, spec)
        except Exception as exc:
            say(ok(f"{tid}: evaluation failed: {type(exc).__name__}: {exc}", False))
            failures += 1
            continue
        if permutations:
            # permutation testing is driven by the spec's `metrics` list inside
            # evaluate_session; the flag only exists to explain that here.
            say("      (permutation testing runs when the spec lists permutation_p)")
        gates = E.apply_gates(spec, res)
        cv = res.get("cv", {})
        acc = cv.get("balanced_accuracy")
        auc = cv.get("auc")
        passed = gates.get("passed")
        line = (f"{tid} {spec.title}: "
                f"bal-acc {acc if acc is None else round(acc,3)} "
                f"(chance {cv.get('chance')}), auc {None if auc is None else round(auc,3)}, "
                f"n={cv.get('n')} -> {_verdict_str(passed)}")
        say(ok(line, passed is True))
        for c in gates.get("checks", []):
            say(f"      {c['metric']}: {_gate_value_str(c)} vs gate {c['gate']} "
                f"{_gate_result_str(c)}")
        for note in (cv.get("notes") or [])[:3]:
            say(f"      note: {note}")
        if res.get("report_path"):
            say(f"      report: {res['report_path']}")
        # persist a machine-readable result so `nb progress` and the agent
        # bridge can both see the latest gate outcome without recomputing
        try:
            (REPORT_DIR / f"{tid}.json").write_text(json.dumps({
                "spec_id": tid,
                "title": spec.title,
                "feasibility": spec.feasibility,
                "session": Path(session_path).name,
                "evaluated_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cv": {k: v for k, v in (res.get("cv") or {}).items() if k != "confusion"},
                "gates": gates,
                "passed": gates.get("passed"),
                "verdict": gates.get("verdict"),
            }, indent=2, default=str))
        except Exception:
            pass
        if passed is False:
            failures += 1
    return 1 if failures else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .catalog import load_one
    from .pipeline import train_and_evaluate
    from . import store, splits
    spec = load_one(args.id)
    paths = _session_paths(args, spec.id)
    if not paths:
        say(ok(f"{spec.id}: no recorded session contains this test yet", False))
        say(f"      run `nb run {spec.id}` (or `nb simulate {spec.id}`) first")
        return 1
    if len(paths) == 1:
        say("      note: one session only - cross-validation here is within a "
            "single seating, which is the weakest form of evidence. Record a "
            "second session (a new day) before believing the number.")
    say(f"training {spec.id} from {len(paths)} session(s): "
        + ", ".join(p.name for p in paths))
    try:
        split = splits.load_split(args.split or "default")
        say(f"      split {split.name!r}: {split.n_dev} dev, {split.n_locked} locked "
            f"-> {split.rule}")
    except FileNotFoundError:
        pass
    res = train_and_evaluate(paths, spec, save=not args.no_save)
    cv = res.get("cv", {})
    table("Training", ["metric", "value"], [
        ["sessions", len(paths)],
        ["epochs", cv.get("n")],
        ["balanced accuracy", _fmt(cv.get("balanced_accuracy"))],
        ["auc", _fmt(cv.get("auc"))],
        ["f1", _fmt(cv.get("f1"))],
        ["chance", _fmt(cv.get("chance"))],
        ["model", res.get("model_path", "-")],
    ])
    for note in cv.get("notes") or []:
        say(f"    note: {note}")
    return 0


def _fmt(x: Any) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) else str(x))


def _verdict_str(passed: Any) -> str:
    """Three-valued verdict: a gate that could not be computed is not a failure."""
    return "PASS" if passed is True else ("FAIL" if passed is False else "INSUFFICIENT DATA")


def _gate_value_str(c: dict) -> str:
    if not c.get("applicable", True):
        return "n/a"
    v = c.get("value")
    if v is None or not c.get("evaluated", False):
        return "not evaluable"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


def _gate_result_str(c: dict) -> str:
    if not c.get("applicable", True):
        return "n/a"
    if not c.get("evaluated", False):
        return "not evaluated"
    return "PASS" if c.get("ok") else "FAIL"


def cmd_eval(args: argparse.Namespace) -> int:
    from .catalog import load_one
    from .pipeline import evaluate as E, write_test_report
    from . import store, splits
    spec = load_one(args.id)

    if getattr(args, "locked", False):
        # Held-out evaluation: fit on the development sessions, then score the
        # locked ones. This is the only number worth quoting in a write-up.
        name = args.split or "default"
        try:
            split = splits.load_split(name)
        except FileNotFoundError as exc:
            say(ok(str(exc), False))
            say("      create one with `nb split create --spec " + spec.id + "`")
            return 1
        dev = [SESSION_DIR / s for s in split.dev]
        locked = [SESSION_DIR / s for s in split.locked]
        head(f"{spec.id} locked evaluation  (split {split.name!r})")
        say(f"fit on {len(dev)} development session(s): {', '.join(split.dev) or '—'}")
        say(f"score on {len(locked)} LOCKED session(s): {', '.join(split.locked) or '—'}")
        if not dev or not locked:
            say(ok("the split needs at least one session on each side", False))
            return 1
        try:
            res_dev = E.train_and_evaluate(dev, spec, save=True)
        except Exception as exc:
            say(ok(f"training on development sessions failed: {type(exc).__name__}: {exc}", False))
            return 1
        dev_cv = res_dev.get("cv", {})
        table("Development (cross-validated)", ["metric", "value"], [
            ["balanced accuracy", _fmt(dev_cv.get("balanced_accuracy"))],
            ["chance", _fmt(dev_cv.get("chance"))],
            ["epochs", dev_cv.get("n")],
        ])
        model = None
        try:
            from .pipeline import load_model
            model = load_model(spec.id)
        except Exception:
            pass
        rows, any_pass = [], False
        for p in locked:
            try:
                s = store.Session.load(p)
            except Exception:
                continue
            if spec.id not in s.test_ids():
                rows.append([p.name, "-", "-", "test not in this session"])
                continue
            res = E.evaluate_session(s, spec, model=model)
            cv = res.get("cv", {})
            gates = E.apply_gates(spec, res)
            any_pass = any_pass or bool(gates.get("passed"))
            rows.append([p.name, _fmt(cv.get("balanced_accuracy")),
                         _fmt(cv.get("auc")),
                         "PASS" if gates.get("passed") else "FAIL"])
        table("Locked (held-out)", ["session", "bal-acc", "auc", "gates"], rows)
        say("\nthe locked number is the claim; the cross-validated one is the estimate")
        return 0 if any_pass else 1

    session = resolve_session(args.session)
    if spec.id not in session.test_ids():
        say(ok(f"{spec.id} is not in {session.path.name}", False))
        say(f"      tests present: {', '.join(session.test_ids()) or 'none'}")
        return 1
    res = E.evaluate_session(session, spec)
    gates = E.apply_gates(spec, res)
    cv = res.get("cv", {})
    head(f"{spec.id} {spec.title}  [{spec.feasibility}]")
    table("Cross-validation", ["metric", "value", "chance"], [
        ["balanced accuracy", _fmt(cv.get("balanced_accuracy")), _fmt(cv.get("chance"))],
        ["auc", _fmt(cv.get("auc")), "0.500"],
        ["macro f1", _fmt(cv.get("f1")), "-"],
        ["epochs", cv.get("n"), "-"],
        ["information transfer", _fmt(res.get("itr_bits_per_min")) + " bits/min", "-"],
    ])
    table("Acceptance gates", ["metric", "value", "gate", "result"],
          [[c["metric"], _gate_value_str(c), c["gate"], _gate_result_str(c)]
           for c in gates.get("checks", [])])
    for note in cv.get("notes") or []:
        say(f"    note: {note}")
    for w in res.get("warnings", [])[:5]:
        say(f"    warn: {w}")
    try:
        p = write_test_report(spec, res, session_path=session.path)
        say(f"\nreport: {p}")
    except Exception as exc:
        say(f"[yellow]report generation failed: {exc}[/yellow]" if _console
            else f"report generation failed: {exc}")
    verdict = gates.get("passed")
    if verdict is True:
        say("\nverdict: PASS - this test decoded above its gate")
        code = 0
    elif verdict is False:
        say("\nverdict: FAIL - see the gate table and the risks listed for this test")
        code = 1
    else:
        say("\nverdict: INSUFFICIENT DATA - no gate could be evaluated for this capture; "
            "this is a design limitation, not a negative result (see the gate reasons)")
        code = 2
    return code


def cmd_report(args: argparse.Namespace) -> int:
    from .catalog import load_specs
    from .pipeline import report as R
    from . import store
    if args.curriculum:
        results: dict[str, Any] = {}
        for d in sorted(SESSION_DIR.iterdir()) if SESSION_DIR.exists() else []:
            if not (d / "meta.json").exists():
                continue
            try:
                s = store.Session.load(d)
            except Exception:
                continue
            for tid in s.test_ids():
                results[tid] = {"session": d.name, "session_path": str(d)}
        p = R.write_curriculum_report(results)
        say(f"curriculum report: {p}")
        return 0
    session = resolve_session(args.session)
    ids = args.ids or session.test_ids()
    out = R.write_session_report(session, {i: {} for i in ids})
    say(f"session report: {out}")
    return 0


# ---------------------------------------------------------------- sessions
def cmd_sessions(args: argparse.Namespace) -> int:
    from . import store
    if args.action == "list":
        rows = []
        for d in sorted(SESSION_DIR.iterdir()) if SESSION_DIR.exists() else []:
            if not (d / "meta.json").exists():
                continue
            try:
                s = store.Session.load(d)
            except Exception:
                continue
            rows.append([d.name, f"{s.duration_s/60:.1f} min", s.eeg.size,
                         len(s.events), ", ".join(s.test_ids()) or "-",
                         (s.meta.get("summary", {}) or {}).get("quality_verdict", "-")])
        table("Sessions", ["id", "duration", "samples", "events", "tests", "quality"], rows)
        say(f"\n{SESSION_DIR}")
        return 0
    if args.action == "show":
        s = resolve_session(args.id)
        head(f"Session {s.path.name}")
        say(json.dumps(s.meta, indent=2, default=str)[:2000])
        table("Tests", ["test", "events"], [
            [t, len(s.events_for(test_id=t))] for t in s.test_ids()])
        if s.notes:
            say("\nnotes:\n" + s.notes[:1500])
        return 0
    if args.action == "export":
        import csv
        s = resolve_session(args.id)
        out = s.path / "eeg.csv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample", "t_mono", "t_rel", "raw"])
            for i in range(s.eeg.size):
                w.writerow([i, f"{s.t[i]:.6f}", f"{s.t[i]-s.t[0]:.6f}", f"{s.eeg[i]:.0f}"])
        say(f"wrote {out} ({s.eeg.size} rows)")
        return 0
    return 1


# ---------------------------------------------------------------- serve / state
def cmd_serve(args: argparse.Namespace) -> int:
    from .agent import serve
    settings = get_settings()
    host = args.host or settings.bridge_host
    port = args.port or settings.bridge_port
    say(f"starting brain-to-agent bridge on http://{host}:{port}")
    say(f"  agent contract:  http://{host}:{port}/schema")
    say(f"  live state:      http://{host}:{port}/state")
    say(f"  SSE stream:      http://{host}:{port}/state/stream")
    serve(host=host, port=port, device=args.device, replay_session=args.replay)
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    from .agent import StateEstimator
    from .device import open_source
    est = StateEstimator()
    src = open_source(args.device)
    # publish the resolved source, not the alias: "auto" is not a device name
    est.source_name = getattr(src, "name", args.device)
    say(f"opening {src.name} for {args.seconds:.0f} s ...")
    src.open()
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.seconds:
            for pkt in src.poll(0.05):
                est.push(pkt.values, pkt.t_samples)
            time.sleep(0.05)
    finally:
        src.close()
    st = est.estimate()
    table("State estimate", ["field", "value"], [
        [k, _fmt(v)] for k, v in st.as_dict().items()
        if k in ("attention", "arousal", "load", "fatigue", "relaxation",
                 "confidence", "artifact_level", "contact", "source")])
    if not est.ready:
        say("\nnot enough data yet - the estimator needs ~30 s of clean signal")
    say("\nnote: with no trained model these are baseline-relative heuristics, not measurements.")
    return 0


# ---------------------------------------------------------------- monitor
def cmd_monitor(args: argparse.Namespace) -> int:
    """Standalone monitor: record, display and replay sessions."""
    from .monitor.app import MonitorConfig, run_headless, run_monitor

    out_dir = Path(args.out) if args.out else None
    config = MonitorConfig(
        device=args.device, url=(args.url or None),
        replay=(args.replay or None), record=not args.no_record,
        out_dir=out_dir, window_s=float(args.window if args.window else
                                        args.view_minutes * 60),
        state_hz=float(args.state_hz), max_hz=float(args.max_hz),
        session_id=args.session_id)

    mode = ("replay " + str(config.replay)) if config.replay else (
        ("attach " + str(config.url)) if config.url else ("device " + config.device))
    say(f"monitor: {mode}"
        + ("  [headless]" if args.headless else "")
        + ("" if config.record else "  [display only, not recording]"))

    if args.headless:
        rec = run_headless(config, seconds=(args.seconds if args.seconds > 0 else None))
        if rec is not None:
            table("Recorded", ["field", "value"], [
                ["session", rec.session_id],
                ["folder", rec.path],
                ["eeg.csv", rec.csv_eeg],
                ["state.csv", rec.csv_state],
                ["events.csv", rec.csv_events],
                ["samples", rec.n_samples],
                ["state rows", rec.n_state_rows],
                ["events", rec.n_events],
                ["fs_effective", round(rec.fs_effective, 2)],
                ["native session", rec.native],
            ])
        return 0

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        say("no display detected (DISPLAY and WAYLAND_DISPLAY are unset).\n"
            "Run with --headless to record without a window.")
        return 2
    return run_monitor(config)


# ---------------------------------------------------------------- agent link
def _link_style(prov) -> str:
    """Colour for a provenance banner: red unless it is a live human."""
    return "bold green" if prov.is_live_human else "bold red"


def _link_banner(prov) -> None:
    text = prov.banner()
    say(f"[{_link_style(prov)}]{text}[/{_link_style(prov)}]" if _console else text)


def _link_field_cell(st: dict) -> str:
    """Render one field without ever implying a number where there is none.

    A field derived from ``uncalibrated_neutral`` is pinned at 0.5 regardless of
    the wearer, so printing "0.50" would be a fabricated measurement. It is shown
    as ``-`` instead, and the caller prints why.
    """
    if not st.get("n"):
        return "-"
    if not st.get("informative"):
        return "-"
    return f"{st['mean']:.2f}"


def _link_report_rows(summary: dict) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, st in (summary.get("fields") or {}).items():
        tr = st.get("trend") or {}
        # a trend across a field pinned at a constant is not a trend
        usable = bool(st.get("n")) and bool(st.get("informative"))
        rows.append([
            name, _link_field_cell(st),
            f"{st['min']:.2f}" if usable else "-",
            f"{st['max']:.2f}" if usable else "-",
            tr.get("direction", "-") if usable else "-",
            f"{tr['delta']:+.2f}" if usable and tr.get("delta") is not None else "-",
            ",".join(st.get("derivation") or []),
        ])
    return rows


def _pick_log(explicit: str | None) -> Path:
    from .agent import link as nblink
    if explicit:
        return Path(explicit)
    today = nblink.default_log_path()
    if today.exists():
        return today
    found = sorted(Path(LOG_DIR).glob("agent-link-*.jsonl"))
    if not found:
        raise SystemExit(f"no agent-link logs in {LOG_DIR}; run `nb link record` first")
    return found[-1]


def cmd_link(args: argparse.Namespace) -> int:
    """Agent-side link: read, watch, record and summarise bridge state."""
    from .agent import link as nblink

    action = getattr(args, "link_action", None) or "status"
    client = nblink.BridgeClient(getattr(args, "url", None))
    as_json = bool(getattr(args, "json", False))

    # ---------------------------------------------------------- status
    if action == "status":
        health = client.health()
        prov = nblink.Provenance.from_health(health)
        if as_json:
            live = nblink.probe_liveness(client)
            emit_json({"health": health, "provenance": prov.as_dict(),
                       "liveness": {k: v for k, v in live.items()
                                    if k not in ("first", "second")}})
            return 0
        say(f"bridge      {client.base_url}")
        _link_banner(prov)
        table("Source (truth from /health, not /state)", ["field", "value"], [
            ["device (actual)", prov.device],
            ["requested_device", prov.requested_device or "-"],
            ["mode", prov.mode],
            ["verdict", prov.label],
            ["live human?", "yes" if prov.is_live_human else "no"],
            ["version", prov.version or "-"],
            ["uptime_s", health.get("uptime_s")],
        ])
        table("Estimator", ["field", "value"], [
            ["estimator_ready", health.get("estimator_ready")],
            ["calibrated", health.get("calibrated")],
            ["confidence", health.get("confidence")],
            ["n_samples", health.get("n_samples")],
            ["field_derivations", json.dumps(health.get("field_derivations") or {})],
            ["trained models", health.get("models") or 0],
            ["sessions on disk", health.get("sessions")],
        ])
        live = nblink.probe_liveness(client)
        say(f"\nstream advancing: {'yes' if live['advanced'] else 'NO - state is frozen'}"
            f"  (t_mono +{live['dt_mono']}s, seq +{live['seq_delta']})")
        if not prov.is_live_human:
            say("\nThis bridge is NOT monitoring a person right now, so nothing it "
                "reports can describe your brain. To get real data: put the TH21A "
                "band on, power it, and restart with `./bin/nb-serve --device th21a`.")
        return 0

    # ---------------------------------------------------------- now
    if action == "now":
        prov = client.provenance()
        state = client.state(explain=True)
        if as_json:
            emit_json({"provenance": prov.as_dict(), "state": state})
            return 0
        _link_banner(prov)
        mi = state.get("model_info") or {}
        deriv = {f: (v or {}).get("derivation")
                 for f, v in (mi.get("fields") or {}).items()}
        rows = []
        for name in nblink.STATE_FIELDS:
            d = deriv.get(name, "unknown")
            informative = d not in nblink.UNINFORMATIVE_DERIVATIONS
            rows.append([name, f"{state.get(name):.2f}" if informative else "-",
                         d, "yes" if informative else "NO"])
        table(f"State @ {state.get('t_wall')} (seq {state.get('seq')})",
              ["field", "value", "derivation", "informative"], rows)
        table("Signal", ["field", "value"], [
            ["confidence", f"{state.get('confidence')} (act only >= 0.4)"],
            ["artifact_level", state.get("artifact_level")],
            ["contact", state.get("contact")],
            ["source (as reported by /state)", state.get("source")],
            ["valence", "always null on one channel - never estimated"],
            ["intents", json.dumps(state.get("intents") or [])],
        ])
        cand = state.get("candidate") or {}
        say(f"\ncandidate: {cand.get('candidate_label')} "
            f"(quality {cand.get('signal_quality')}, "
            f"calibrated={cand.get('probabilities_calibrated')})")
        for why in (cand.get("uncertain_reasons") or []):
            say(f"  - {why}")
        return 0

    # ---------------------------------------------------------- watch
    if action == "watch":
        seconds = nblink.parse_duration(getattr(args, "seconds", None)) or 0.0
        stop = threading.Event()
        t_end = time.perf_counter() + seconds if seconds > 0 else None
        say(f"watching {client.base_url}/state/stream "
            f"(display only - use `nb link record` to keep a log)")
        prev: float | None = None
        n = 0
        stale_mark = "[red]STALE[/red]" if _console else "STALE"
        try:
            for payload in client.stream(stop=stop):
                now = float(payload.get("_recv_mono") or 0.0)
                if prev is not None and (now - prev) > 3.0:
                    say(f"  {stale_mark} {now - prev:.1f}s with no payload - "
                        "disregard the last value and check /health")
                prev = now
                n += 1
                mi = payload.get("model_info") or {}
                deriv = {f: (v or {}).get("derivation")
                         for f, v in (mi.get("fields") or {}).items()}
                cells = []
                for name in nblink.STATE_FIELDS:
                    informative = deriv.get(name) not in nblink.UNINFORMATIVE_DERIVATIONS
                    cells.append(f"{name[:4]}={payload.get(name):.2f}" if informative
                                 else f"{name[:4]}=--")
                say(f"  {payload.get('_recv_wall')} seq={payload.get('seq')} "
                    f"conf={payload.get('confidence'):.2f} " + " ".join(cells))
                if t_end is not None and time.perf_counter() >= t_end:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
        say(f"\n{n} payload(s) displayed")
        return 0

    # ---------------------------------------------------------- record
    if action == "record":
        seconds = nblink.parse_duration(getattr(args, "seconds", None))
        out = Path(args.out) if getattr(args, "out", None) else nblink.default_log_path()
        prov = client.provenance()
        _link_banner(prov)
        say(f"recording to {out}"
            + (f" for {seconds:.0f}s" if seconds else " until interrupted"))
        say("this keeps running while you talk to the agent; Ctrl-C to stop")
        last = [0.0]

        def _progress(payload: dict) -> None:
            now = float(payload.get("_recv_mono") or 0.0)
            if last[0] and (now - last[0]) > 3.0:
                say(f"  STALE {now - last[0]:.1f}s gap")
            last[0] = now

        receipt = nblink.record(client, out_path=out, seconds=seconds,
                                on_payload=_progress)
        if not receipt["n_payloads"]:
            say("[red]nothing recorded[/red]" if _console else "nothing recorded")
            return 1
        table("Recorded", ["field", "value"], [
            ["file", receipt["path"]],
            ["payloads", receipt["n_payloads"]],
            ["from", receipt["first_wall"]],
            ["to", receipt["last_wall"]],
            ["source (truth)", receipt["provenance"].get("device")],
        ])
        return 0

    # ---------------------------------------------------------- report
    if action == "report":
        path = _pick_log(getattr(args, "log", None))
        last_s = nblink.parse_duration(getattr(args, "last", None))
        header, rows = nblink.load_log(path, last_s=last_s)
        settings = get_settings()
        summary = nblink.summarize(rows, header, expected_hz=settings.state_hz)
        if as_json:
            emit_json(summary)
            return 0
        say(f"log         {path}")
        if last_s:
            say(f"window      last {last_s:.0f}s of the recording")
        prov = nblink.Provenance(
            device=str(summary["provenance"].get("device") or "unknown"),
            requested_device=str(summary["provenance"].get("requested_device") or ""),
            mode=str(summary["provenance"].get("mode") or "unknown"))
        _link_banner(prov)
        if not summary.get("n_payloads"):
            say("no payloads in this window")
            return 1
        table("Window", ["metric", "value"], [
            ["payloads", summary["n_payloads"]],
            ["from -> to", f"{summary['window']['first_wall']} -> "
                           f"{summary['window']['last_wall']}"],
            ["duration_s", summary["window"]["duration_s"]],
            ["measured rate_hz", summary["rate_hz"]],
            ["expected rate_hz", summary.get("expected_hz")],
            ["max gap_s", summary["staleness"]["max_gap_s"]],
            ["gaps > 3s", summary["staleness"]["gap_events_gt_3s"]],
            ["seq", f"{summary['seq']['first']} -> {summary['seq']['last']}"],
            ["dropped payloads", summary["seq"]["dropped"]],
        ])
        table("Fields (mean over the window)", ["field", "mean", "min", "max",
                                                "trend", "delta", "derivation"],
              _link_report_rows(summary))
        if not summary.get("informative"):
            say("\nevery field is uninformative right now: with no personal baseline "
                "they are pinned at 0.5.\nWhat is meaningful today is signal quality "
                "- confidence, artifact_level and contact below.")
        table("Signal quality", ["metric", "value"], [
            ["confidence mean/min/max",
             f"{summary['confidence'].get('mean')} / {summary['confidence'].get('min')}"
             f" / {summary['confidence'].get('max')}"],
            ["artifact mean", summary["artifact_level"].get("mean")],
            ["contact values", ", ".join(summary["contact"]) or "-"],
            ["intents", json.dumps(summary["intents"]["by_kind"]) or "{}"],
        ])
        head("Honest caveats for this window")
        for line in summary["limitations"]:
            say(f"  - {line}")
        return 0

    # ---------------------------------------------------------- calibrate
    if action == "calibrate":
        prov = client.provenance()
        _link_banner(prov)
        force = bool(getattr(args, "force", False))
        if not prov.is_live_human and not force:
            say("\nrefusing to capture a personal baseline from a source that is "
                "not a live headband: the fields would be relative to nothing "
                "about you. Pass --force only for development.")
            return 1
        say("capturing the personal baseline (robust median/MAD over the first "
            "~60 s of this session) ...")
        res = client.calibrate(force=force)
        if as_json:
            emit_json(res)
            return 0 if res.get("calibrated") else 1
        table("Calibration", ["field", "value"], [
            ["status", res.get("status") or res.get("error") or "-"],
            ["calibrated", res.get("calibrated")],
            ["samples used", res.get("n_samples")],
            ["windows used", res.get("n_windows")],
            ["reason", res.get("reason") or "-"],
            ["baseline file", res.get("path") or "-"],
            ["source", (res.get("provenance") or {}).get("device")],
        ])
        if res.get("warning"):
            say(f"\nwarning: {res['warning']}")
        if res.get("calibrated"):
            if res.get("is_live_human"):
                say("\nbaseline established. The five fields are now relative to "
                    "this person's own baseline, so 0.5 means 'at baseline'. They "
                    "remain baseline-relative heuristics, not measurements of "
                    "thought.")
            else:
                say("\nbaseline stored, but from a NON-LIVE source: it describes "
                    "nobody and exists only for development.")
        else:
            reason = res.get("reason") or res.get("error") or "unknown"
            req = res.get("need_s")
            say(f"\nno baseline yet: {reason}"
                + (f" (needs ~{req:.0f}s more)" if isinstance(req, (int, float))
                   else ""))
            say("leave the bridge running, sit still, and retry once it has ~65 s "
                "of clean signal.")
        return 0

    # ---------------------------------------------------------- mark
    if action == "mark":
        meta: dict[str, Any] = {}
        for pair in (getattr(args, "meta", None) or []):
            if "=" in pair:
                k, v = pair.split("=", 1)
                meta[k.strip()] = v.strip()
        res = client.mark(args.name, meta)
        say(f"marked {args.name!r} at t_mono={res.get('t_mono')} "
            f"(session={res.get('event', {}).get('session')})")
        return 0

    say(f"unknown link action {action!r}")
    return 2


# ---------------------------------------------------------------- catalog audit
#: Feature strings the pipeline implements (docs/CATALOG_SPEC.md).
FEATURE_PREFIXES = ("bandpower:", "relbandpower:", "ratio:", "erp:", "erp_peak:",
                    "hjorth:", "entropy:", "stats:", "slope:", "peakfreq:",
                    "aperiodic:", "cov:")
#: Gates that are meaningful for a test with only one class.
NON_ACCURACY_GATES = ("alpha_ratio_closed_open", "response_snr", "latency_jitter_ms",
                      "artifact_fraction", "drift_ratio", "band_delta_pct",
                      "saturation_pct", "mains_pct", "false_positive_rate")
GATE_REFERENCE = {"established": (0.68, 0.85), "promising": (0.58, 0.70),
                  "exploratory": (0.50, 0.62), "control": (0.50, 0.60)}


def _audit_catalog(specs) -> int:
    """Checks the schema validator cannot make: does each spec hold together?"""
    from collections import defaultdict
    from .simulate import estimate_duration_min

    problems: list[str] = []
    notes: list[str] = []
    durations = []
    codes = defaultdict(list)

    for s in specs:
        # 1. the design must actually build
        try:
            declared, mech, planning = estimate_duration_min(s)
        except Exception as exc:
            problems.append(f"{s.id}: timeline does not build: {type(exc).__name__}: {exc}")
            continue
        durations.append((s.id, declared, mech, planning))
        if declared and mech and (max(declared, mech) / max(1e-6, min(declared, mech)) > 1.35):
            notes.append(f"{s.id}: declared {declared:.0f} min vs design-derived {mech:.0f} min "
                         f"({100*(mech-declared)/declared:+.0f}%)")
        if planning > 20:
            notes.append(f"{s.id}: planned at {planning:.0f} min - too long for a 45 min sitting "
                         f"with setup; split it or raise the budget")

        # 2. feature vocabulary
        for f in s.features:
            name = str(f).split(":")[0] + ":"
            if name not in FEATURE_PREFIXES:
                problems.append(f"{s.id}: unknown feature {f!r}")

        # 3. gates sane for the feasibility band
        acc_gates = {k: v for k, v in s.acceptance.items() if k in ("balanced_accuracy", "auc")}
        if len(s.classes) < 2:
            if acc_gates:
                problems.append(f"{s.id}: single class {s.classes} but an accuracy gate {acc_gates}")
            if not (set(s.acceptance) & set(NON_ACCURACY_GATES)):
                notes.append(f"{s.id}: single-class test with no non-accuracy gate "
                             f"-> only {list(s.acceptance)}")
        else:
            lo, hi = GATE_REFERENCE.get(s.feasibility, (0.0, 1.0))
            for metric, gate in acc_gates.items():
                if metric == "balanced_accuracy" and not (lo <= gate <= hi):
                    notes.append(f"{s.id}: balanced_accuracy gate {gate} outside the "
                                 f"{s.feasibility} band [{lo}, {hi}]")
                if metric == "auc" and gate > 0.90:
                    problems.append(f"{s.id}: auc gate {gate} is not attainable on this hardware")

        # 4. how many classes given one frontal channel
        if len(s.classes) > 3:
            notes.append(f"{s.id}: {len(s.classes)} classes on one channel - expect heavy averaging")

        # 5. marker code sharing within a tier
        for name, code in s.markers.items():
            codes[(s.tier, code)].append(s.id)

    shared = {k: sorted(set(v)) for k, v in codes.items() if len(set(v)) > 1}
    tiers_shared = defaultdict(set)
    for (tier, _), ids in shared.items():
        tiers_shared[tier].update(ids)

    total_declared = sum(d[1] for d in durations)
    total_planning = sum(d[3] for d in durations)
    table("Duration", ["stat", "minutes"], [
        ["declared total", f"{total_declared:.0f}"],
        ["design-derived total", f"{sum(d[2] for d in durations):.0f}"],
        ["planning total (conservative)", f"{total_planning:.0f}"],
        ["sittings at 45 min", f"{total_planning/45:.1f}"],
        ["longest single test", f"{max((d[3] for d in durations), default=0):.0f}"],
    ])
    say(f"\nmarker codes are shared by {len(tiers_shared)} tier(s) across tests "
        f"({dict((k, len(v)) for k, v in sorted(tiers_shared.items()))}) - "
        "harmless because events carry test_id, but never resolve a marker by code alone")

    if notes:
        head(f"Review ({len(notes)})")
        for n in notes:
            say(f"  - {n}")
    if problems:
        head(f"Problems ({len(problems)})")
        for p in problems:
            say(f"  - {p}")
        return 1
    say("\nno structural problems found")
    return 0


# ---------------------------------------------------------------- curriculum doc
def _write_curriculum_doc(specs) -> int:
    """Render all 64 specs into docs/CURRICULUM.md (readable, one place)."""
    from .catalog import TIERS
    from .simulate import estimate_duration_min
    # planning minutes = the larger of declared and design-derived (see nb catalog audit)
    minutes = {s.id: estimate_duration_min(s)[2] for s in specs}
    lines: list[str] = [
        "# Curriculum — the 64 tests",
        "",
        "*Generated by `nb catalog docs` from `neurobridge/catalog/tests/*.yaml`; "
        "edit the YAML, not this file.*",
        "",
        "Each test is one stimulation protocol plus the analysis recipe chosen "
        "**before** recording: epoch window, features, decoder, cross-validation "
        "and acceptance gates. `feasibility` is an honest guess at what a single "
        "dry frontal electrode can support.",
        "",
        "| band | meaning |",
        "|---|---|",
        "| `established` | replicated in the literature and plausible on this montage |",
        "| `promising` | real but modest effect; needs good contact and averaging |",
        "| `exploratory` | interesting, weak or unknown here; lenient gate, dataset value |",
        "| `control` | null protocol; **must** stay at chance |",
        "",
        "## Overview",
        "",
        "| id | tier | feasibility | min | trials | modality | agent | classes | title |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in specs:
        lines.append(
            f"| {s.id} | {s.tier} | {s.feasibility} | {minutes[s.id]:.0f} | {s.trials} | "
            f"{', '.join(s.modality)} | {s.agent_use} | {', '.join(s.classes)} | {s.title} |")
    total = sum(minutes.values())
    lines += [
        "",
        f"**Total recording time: {total:.0f} minutes** — about {total/45:.0f} sittings "
        "at 45 minutes each, plus setup, breaks and repeats.",
        "",
        "## Detail",
        "",
    ]
    for tier, name in TIERS.items():
        rows = [s for s in specs if s.tier == tier]
        if not rows:
            continue
        lines += [f"### Tier {tier} — {name}", ""]
        for s in rows:
            lines += [
                f"#### {s.id} — {s.title}",
                "",
                f"*feasibility:* `{s.feasibility}` · *agent use:* `{s.agent_use}` · "
                f"*modality:* {', '.join(s.modality)} · *planned:* {minutes[s.id]:.0f} min "
                f"(declared {s.duration_min:.0f} min; {s.trials} trials in "
                f"{s.design.get('blocks', '?')} blocks)",
                "",
                s.goal.strip().replace("\n", " "),
                "",
                f"- **classes:** {', '.join('`%s`' % c for c in s.classes)}",
                f"- **stimuli:** `{s.stimuli.get('source', 'none')}`"
                + (f" (fallback `{s.stimuli.get('fallback')}`)" if s.stimuli.get("fallback") else ""),
                f"- **epoch:** {s.processing.get('epoch', 'fixed windows')} s"
                + (f", baseline {s.processing.get('baseline')}" if s.processing.get("baseline") else "")
                + (f", epoching `{s.processing.get('epoching')}`" if s.processing.get("epoching") else ""),
                f"- **features:** {', '.join('`%s`' % f for f in s.features)}",
                f"- **decoder:** `{s.decoder.get('model')}` with `{s.decoder.get('cv')}` CV",
                "- **gates:** " + ", ".join(f"{k} ≥ {v}" for k, v in s.acceptance.items()),
                "- **honest failure modes:**",
            ]
            lines += [f"    - {r.strip().replace(chr(10), ' ')}" for r in s.risks]
            if s.notes:
                lines += ["", f"*notes:* {s.notes.strip().replace(chr(10), ' ')}"]
            lines.append("")
    out = DOCS_DIR / "CURRICULUM.md"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    say(f"wrote {out}  ({len(specs)} tests, {total:.0f} min total)")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the Tkinter control panel (also what the system menu entry runs)."""
    try:
        from .gui import main as gui_main
    except Exception as exc:
        say(ok(f"the GUI needs Tkinter ({type(exc).__name__}: {exc}). "
               f"Install python3-tk, or use ./bin/nb on the command line.", False))
        return 2
    return int(gui_main() or 0)


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Write one shareable file with everything needed to debug a problem."""
    from .diagnostics import collect
    say("collecting diagnostics (read-only; no data is modified or deleted) ...")
    path = collect(scan=args.scan, mic=args.mic, note=args.note or "")
    size = path.stat().st_size
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    say(ok(f"wrote {shown}  ({size} bytes)"))
    say("attach or paste that whole file when reporting a problem — it has versions,")
    say("settings (credentials withheld), catalog state, session summaries, the doctor")
    say("output and the tails of recent logs.")
    if args.print:
        say("")
        say(path.read_text())
    return 0


def cmd_screen_test(args: argparse.Namespace) -> int:
    """Show the session surface for a few seconds and report whether it covers
    the whole display.

    Exists because "the stimuli window was not full screen" is otherwise only
    discoverable by wearing the headband and starting a session. This measures
    the presentation surface against the real desktop size and exits.
    """
    import time as _t
    try:
        from .runner.present_visual import VisualPresenter
    except Exception as exc:
        say(ok(f"presentation layer unavailable: {type(exc).__name__}: {exc}", False))
        return 2
    settings = get_settings()
    try:
        import pygame
        pygame.display.init()
        desk = [tuple(s) for s in pygame.display.get_desktop_sizes()]
    except Exception as exc:
        say(ok(f"no display available ({type(exc).__name__}): the session would use "
               f"terminal presentation", False))
        return 2

    fullscreen = not args.windowed
    say(f"desktop size(s) : {desk}")
    say(f"requesting      : {'fullscreen' if fullscreen else 'windowed'} "
        f"(settings.screen_size={tuple(settings.screen_size)})")
    v = VisualPresenter(size=tuple(settings.screen_size), fullscreen=fullscreen)
    if not getattr(v, "active", False):
        say(ok("presenter fell back to the terminal backend (no window opened)", False))
        return 2

    surface = getattr(v, "_screen", None)
    surf_size = tuple(surface.get_size()) if surface is not None else None
    driver = ""
    try:
        import pygame as _pg
        driver = _pg.display.get_driver()
    except Exception:
        pass
    covers = bool(surf_size and desk and surf_size in desk)
    say(f"presentation    : {surf_size}  driver={driver}")
    say(ok(f"covers a whole desktop ({desk[0]})", covers) if fullscreen
        else f"windowed (as requested): {surf_size}")

    text = ("NEUROBRIDGE — screen test\n\n"
            "This panel should cover the WHOLE display:\n"
            "no desktop, no panels, no wallpaper, no cursor.\n\n"
            f"surface {surf_size}   desktop {desk[0] if desk else '?'}\n\n"
            f"holding {args.seconds:.0f} s — press SPACE to end early")
    v.show_text(text, font_size=30)
    t0 = _t.time()
    while _t.time() - t0 < args.seconds:
        keys = [e for e in v.poll_events() if e.get("type") == "key"]
        if any(str(e.get("key", "")).lower() in ("space", "return", "enter", "escape")
               for e in keys):
            break
        v.pump(0.05)
    v.close()
    say("")
    if fullscreen and not covers:
        say(ok("NOT full screen: the surface is smaller than the desktop. If you are "
               "on a HiDPI or multi-monitor setup, set `screen_size` in "
               "neurobridge.yaml or run `nb run --windowed` deliberately.", False))
        return 1
    say(ok("screen test passed"))
    return 0


# ---------------------------------------------------------------- voice
def cmd_voice(args: argparse.Namespace) -> int:
    """Analyse a session's microphone recording, or record a standalone sample."""
    from . import store, voice
    if args.record:
        out = SESSION_DIR / f"mic-{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            rec = voice.MicRecorder(voice.MicConfig(device=args.device, fs=args.fs,
                                                    consent=True), out_dir=out)
            rec.start()
        except voice.MicUnavailable as exc:
            say(ok(f"microphone unavailable: {exc}", False))
            return 1
        say(f"recording {args.record:.0f} s from {rec.info.get('device')} -> {out}")
        say("      speak now (read something aloud if you are measuring reading)")
        try:
            t0 = time.time()
            while time.time() - t0 < args.record:
                time.sleep(0.2)
        except KeyboardInterrupt:
            say("      stopped early")
        p = rec.stop()
        if p is None:
            say(ok("nothing was captured", False))
            return 1
        info = rec.info
        say(ok(f"wrote {p}  ({info.get('duration_s')} s, rms {info.get('rms_dbfs')} dBFS)"))
        say("      analyse it with: nb voice --session " + out.name)
        return 0

    session = resolve_session(args.session)
    if not session.has_audio():
        say(ok(f"{session.path.name} has no audio.wav", False))
        say("      record one with `nb run <test> --mic`, or a standalone sample with "
            "`nb voice --record 60`")
        return 1
    res = voice.analyze_session(session)
    if not res.get("available"):
        say(ok(res.get("reason", "audio not analysable"), False))
        return 1
    s = res["summary"]
    head(f"Voice — {session.path.name}")
    say(f"device {res.get('input_latency_ms')} ms input latency, "
        f"{s.get('n_trials')} trial window(s), {s.get('n_spoken')} with speech")
    table("Speech", ["metric", "value"], [
        ["silent trials", _fmt(s.get("silent_fraction"))],
        ["onset latency (median)", f"{s.get('onset_latency_ms_median')} ms"],
        ["onset latency (IQR)", f"{s.get('onset_latency_ms_iqr')} ms"],
        ["speech duration (median)", f"{s.get('speech_duration_ms_median')} ms"],
        ["speech rate (median)", f"{s.get('syllables_per_s_median')} syllables/s"],
        ["pauses / restarts", s.get("n_restarts_or_pauses")],
    ])
    rows = [[t.get("trial"), t.get("onset_latency_ms"), t.get("speech_duration_ms"),
             t.get("syllables_per_s"), t.get("n_pauses"), t.get("rms_dbfs"),
             "yes" if t.get("spoke") else "silent"]
            for t in res["trials"][: args.limit or 25]]
    table("Per trial", ["trial", "onset ms", "dur ms", "syl/s", "pauses", "dBFS", "spoke"], rows)
    out = session.path / "voice_metrics.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    say(f"\nfull metrics: {out}")
    say("\ncaveats")
    for c in res["caveats"]:
        say(f"  - {c}")
    return 0


# ---------------------------------------------------------------- splits
def cmd_split(args: argparse.Namespace) -> int:
    """Create and inspect frozen development / locked-evaluation splits."""
    from . import splits
    if args.action == "list":
        names = splits.list_splits()
        if not names:
            say("no splits yet - create one with `nb split create --spec T201`")
            return 0
        rows = []
        for n in names:
            try:
                s = splits.load_split(n)
            except Exception:
                continue
            rows.append([n, s.spec_id or "all", s.n_dev, s.n_locked,
                         s.created_wall, s.rule[:52]])
        table("Splits", ["name", "spec", "dev", "locked", "created", "rule"], rows)
        say(f"\n{splits.SPLIT_DIR}")
        return 0
    if args.action == "show":
        s = splits.load_split(args.name)
        say(s.to_markdown())
        return 0
    # create
    try:
        s = splits.create_split(spec_id=args.spec, name=args.name,
                                dev_frac=args.dev_frac, seed=args.seed,
                                lock_newest=not args.random, note=args.note or "",
                                force=args.force)
    except (ValueError, FileExistsError) as exc:
        say(ok(str(exc), False))
        return 1
    say(s.to_markdown())
    say(f"manifest: {splits.manifest_path(s.name)}")
    say(f"next: nb train {s.spec_id or '<test>'} --split {s.name}   "
        f"(development only)")
    say(f"      nb eval  {s.spec_id or '<test>'} --locked --split {s.name}")
    return 0


# ---------------------------------------------------------------- planning
#: Tests that must be done once, early, because everything else depends on
#: trusting the channel (contact, latency, artefacts) or on knowing the
#: pipeline's false-positive floor.
PREREQUISITES = ["T001", "T002", "T003", "T007", "T004", "T005", "T006"]

#: Tests worth repeating periodically to confirm nothing has drifted.
RECHECK = ["T003", "T008"]

_FEAS_ORDER = {"established": 0, "promising": 1, "exploratory": 2, "control": 3}


def _curriculum_order(specs) -> list:
    """Prerequisites first, then easiest/most reliable tiers first."""
    by_id = {s.id: s for s in specs}
    out = [by_id[i] for i in PREREQUISITES if i in by_id]
    rest = [s for s in specs if s.id not in PREREQUISITES]
    rest.sort(key=lambda s: (s.tier, _FEAS_ORDER.get(s.feasibility, 9), s.duration_min, s.id))
    return out + rest


def cmd_plan(args: argparse.Namespace) -> int:
    """Pack the 64 tests into ~45 minute sittings."""
    from .catalog import load_specs
    from . import store
    specs = load_specs()
    done = set()
    for d in sorted(SESSION_DIR.iterdir()) if SESSION_DIR.exists() else []:
        if not (d / "meta.json").exists():
            continue
        try:
            done.update(store.Session.load(d).test_ids())
        except Exception:
            pass
    ordered = _curriculum_order(specs)
    if args.remaining:
        ordered = [s for s in ordered if s.id not in done]
    from .simulate import estimate_duration_min
    # conservative: never under-book a sitting
    minutes = {s.id: estimate_duration_min(s)[2] for s in ordered}
    budget = float(args.minutes)
    sessions: list[list] = []
    cur: list = []
    used = 0.0
    for s in ordered:
        # a single long test gets its own session rather than being split
        if cur and used + minutes[s.id] > budget:
            sessions.append(cur)
            cur, used = [], 0.0
        cur.append(s)
        used += minutes[s.id]
    if cur:
        sessions.append(cur)
    for i, sess in enumerate(sessions, 1):
        total = sum(minutes[s.id] for s in sess)
        head(f"Session {i}  ({total:.0f} min of recording, ~{total + 5:.0f} min with setup)")
        table("", ["id", "min", "feasibility", "modality", "title"],
              [[s.id, f"{minutes[s.id]:.0f}", s.feasibility, ",".join(s.modality), s.title]
               for s in sess])
        say("  nb run " + " ".join(s.id for s in sess) + " -y")
    remaining = [s for s in specs if s.id not in done]
    say(f"\n{len(done)}/{len(specs)} tests already recorded, "
        f"{len(remaining)} remaining, {len(sessions)} sitting(s) at {budget:.0f} min")
    if not args.remaining and done:
        say("tip: --remaining plans only what you have not recorded yet")
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    """Per-test status: recorded, trained, gate outcome."""
    from .catalog import load_specs
    from . import store
    specs = load_specs()
    recorded: dict[str, int] = {}
    for d in sorted(SESSION_DIR.iterdir()) if SESSION_DIR.exists() else []:
        if not (d / "meta.json").exists():
            continue
        try:
            for t in store.Session.load(d).test_ids():
                recorded[t] = recorded.get(t, 0) + 1
        except Exception:
            pass
    trained = {p.stem for p in MODEL_DIR.glob("*")} if MODEL_DIR.exists() else set()
    rows = []
    counts = {"pass": 0, "fail": 0, "untested": 0, "unrecorded": 0, "insufficient": 0}
    for s in specs:
        rep = REPORT_DIR / f"{s.id}.json"
        verdict, metric = "-", "-"
        if rep.exists():
            try:
                d = json.loads(rep.read_text())
                verdict = d.get("verdict") or _verdict_str(d.get("passed"))
                acc = (d.get("cv") or {}).get("balanced_accuracy")
                metric = "-" if acc is None else f"{acc:.3f}"
                key = ("pass" if d.get("passed") is True else
                       "fail" if d.get("passed") is False else "insufficient")
                counts[key] = counts.get(key, 0) + 1
            except Exception:
                verdict = "?"
        elif s.id in recorded:
            verdict = "recorded"
            counts["untested"] += 1
        else:
            verdict = "not run"
            counts["unrecorded"] += 1
        rows.append([s.id, s.tier, s.feasibility, recorded.get(s.id, 0),
                     "yes" if s.id in trained else "-", metric, verdict, s.title[:38]])
    table("Progress", ["id", "tier", "feasibility", "sessions", "model", "bal-acc", "status", "title"],
          rows)
    say(f"\n{counts['pass']} passed  {counts['fail']} failed  "
        f"{counts.get('insufficient', 0)} insufficient data  "
        f"{counts['untested']} recorded-not-evaluated  {counts['unrecorded']} not run")
    say("a FAIL is a normal, expected outcome for `exploratory` tests; "
        "repeated failure of an `established` test means the electrode contact or the rig changed")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    head(f"NeuroBridge {__version__}")
    say(f"root:      {ROOT}")
    say(f"sessions:  {SESSION_DIR}")
    say(f"models:    {MODEL_DIR}")
    say(f"reports:   {REPORT_DIR}")
    say(f"docs:      {DOCS_DIR}")
    settings = get_settings()
    table("Settings", ["key", "value"], sorted(settings.as_dict().items()))
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"NeuroBridge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check environment, hardware, credentials")
    d.add_argument("--scan", action="store_true", help="also scan for the headband (10 s)")
    d.add_argument("--mic", action="store_true",
                   help="also probe the microphone for 2 s (listens to the room)")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("catalog", help="browse and validate the 64 tests")
    c.add_argument("action", choices=["list", "show", "validate", "stats", "docs", "audit"])
    c.add_argument("ids", nargs="*")
    c.set_defaults(func=cmd_catalog)

    f = sub.add_parser("fetch", help="resolve/download/generate stimuli for tests")
    f.add_argument("ids", nargs="*")
    f.add_argument("--limit", type=int, default=0, help="cap items per class")
    f.add_argument("--force", action="store_true", help="ignore the cache")
    f.add_argument("--offline", action="store_true", help="never touch the network")
    f.add_argument("--verbose", action="store_true")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run a live session (real headband or synthetic)")
    r.add_argument("ids", nargs="+")
    r.add_argument("--device", default=None, help="auto|th21a|fake")
    r.add_argument("--session", default=None, help="session id to use")
    r.add_argument("--dry-run", action="store_true", help="timing rehearsal: markers, no samples")
    r.add_argument("--scale", type=float, default=1.0, help="time scaling for smoke tests")
    r.add_argument("--limit", type=int, default=0, help="cap trials")
    r.add_argument("--offline", action="store_true")
    r.add_argument("--fullscreen", action="store_true",
                   help="force fullscreen stimuli (this is already the default)")
    r.add_argument("--windowed", action="store_true",
                   help="run the stimuli in a window instead of fullscreen")
    r.add_argument("--advance", choices=["key", "timer"], default=None,
                   help="'key' (default) waits for SPACE between instructions, cues and "
                        "breaks; 'timer' auto-advances (rehearsal and automation)")
    r.add_argument("--no-display", action="store_true", help="terminal presentation")
    r.add_argument("--mic", action="store_true",
                   help="record the microphone alongside the EEG (stored locally)")
    r.add_argument("--mic-device", default=None, help="input device index or name")
    r.add_argument("--no-eval", action="store_true", help="skip post-run analysis")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("-y", "--yes", action="store_true", help="start without confirmation")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="offline synthetic session + full analysis")
    s.add_argument("id")
    s.add_argument("--effect-size", type=float, default=1.0, help="0.0 == pure null model")
    s.add_argument("--seed", type=int, default=20261013)
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--permutations", type=int, default=0)
    s.add_argument("--note", default=None)
    s.set_defaults(func=cmd_simulate)

    t = sub.add_parser("train", help="train a decoder from recorded sessions")
    t.add_argument("id")
    t.add_argument("--session", default=None)
    t.add_argument("--all-sessions", action="store_true")
    t.add_argument("--split", default=None, help="frozen split to fit on (development side)")
    t.add_argument("--no-save", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="evaluate a test against its acceptance gates")
    e.add_argument("id")
    e.add_argument("--session", default=None)
    e.add_argument("--locked", action="store_true",
                   help="held-out evaluation: fit on dev, score the locked sessions")
    e.add_argument("--split", default=None)
    e.set_defaults(func=cmd_eval)

    sp = sub.add_parser("split", help="frozen development / locked-evaluation splits")
    sp.add_argument("action", choices=["create", "show", "list"])
    sp.add_argument("--spec", default=None, help="test id this split is for")
    sp.add_argument("--name", default="default")
    sp.add_argument("--dev-frac", type=float, default=0.7)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--random", action="store_true",
                    help="random split instead of locking the newest sessions")
    sp.add_argument("--note", default=None)
    sp.add_argument("--force", action="store_true", help="revise an existing frozen split")
    sp.set_defaults(func=cmd_split)

    rp = sub.add_parser("report", help="write analysis reports")
    rp.add_argument("ids", nargs="*")
    rp.add_argument("--session", default=None)
    rp.add_argument("--curriculum", action="store_true", help="64-test progress table")
    rp.set_defaults(func=cmd_report)

    ss = sub.add_parser("sessions", help="inspect recorded sessions")
    ss.add_argument("action", choices=["list", "show", "export"])
    ss.add_argument("id", nargs="?", default="latest")
    ss.set_defaults(func=cmd_sessions)

    sv = sub.add_parser("serve", help="brain-to-agent HTTP/SSE/WebSocket bridge")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--device", default="auto")
    sv.add_argument("--replay", default=None, help="replay a recorded session instead of live")
    sv.set_defaults(func=cmd_serve)

    st = sub.add_parser("state", help="one-shot live state estimate")
    st.add_argument("--device", default="auto")
    st.add_argument("--seconds", type=float, default=45.0)
    st.set_defaults(func=cmd_state)

    lk = sub.add_parser("link",
                        help="agent-side link: status/now/watch/record/report/"
                             "calibrate/mark")
    lk.add_argument("--url", default=None,
                    help="bridge base URL (default from settings: 127.0.0.1:8731)")
    lk.add_argument("--json", action="store_true", help="machine-readable output")
    lk.set_defaults(func=cmd_link, link_action="status")
    lsub = lk.add_subparsers(dest="link_action")

    la = lsub.add_parser("status", help="is the bridge up, and is it a real person?")
    la.set_defaults(func=cmd_link)

    ln = lsub.add_parser("now", help="one honest snapshot with provenance")
    ln.set_defaults(func=cmd_link)

    lw = lsub.add_parser("watch", help="follow the live stream (display only)")
    lw.add_argument("--seconds", default="60s", help="e.g. 30s, 5m (default 60s)")
    lw.set_defaults(func=cmd_link)

    lr = lsub.add_parser("record", help="log the stream to JSONL while you talk")
    lr.add_argument("--out", default=None, help="log path (default data/logs/agent-link-<date>.jsonl)")
    lr.add_argument("--seconds", default=None, help="e.g. 90s, 10m (default: until Ctrl-C)")
    lr.set_defaults(func=cmd_link)

    lp = lsub.add_parser("report", help="summarise a recorded window with its caveats")
    lp.add_argument("--log", default=None, help="log path (default: newest agent-link log)")
    lp.add_argument("--last", default=None, help="e.g. 10m, 90s (default: whole log)")
    lp.set_defaults(func=cmd_link)

    lm = lsub.add_parser("mark", help="write an agent marker into the EEG timeline")
    lm.add_argument("name")
    lm.add_argument("--meta", action="append", metavar="K=V")
    lm.set_defaults(func=cmd_link)

    lc = lsub.add_parser("calibrate",
                         help="capture the personal baseline the fields are relative to")
    lc.add_argument("--force", action="store_true",
                    help="allow it on a non-live source (development only)")
    lc.set_defaults(func=cmd_link)

    mo = sub.add_parser("monitor",
                        help="standalone monitor: record to CSV, display, replay")
    mo.add_argument("--device", default="auto",
                    help="own the headband: auto|th21a|fake")
    mo.add_argument("--url", default=None,
                    help="attach to a running `nb serve` instead of owning the device")
    mo.add_argument("--replay", default=None,
                    help="replay a session folder or a monitor eeg.csv")
    mo.add_argument("--no-record", action="store_true", help="display without recording")
    mo.add_argument("--out", default=None, help="folder for recordings (default data/sessions)")
    mo.add_argument("--session-id", default=None)
    mo.add_argument("--window", type=float, default=None,
                    help="exact trace length in seconds (overrides --view-minutes)")
    mo.add_argument("--view-minutes", type=int, default=1, choices=[1, 5, 10, 15, 30, 60],
                    help="initial chart view length (also switchable in the window)")
    mo.add_argument("--state-hz", type=float, default=2.0)
    mo.add_argument("--max-hz", type=float, default=250.0,
                    help="cap the sample rate taken from an attached bridge")
    mo.add_argument("--headless", action="store_true",
                    help="record without a window (SSH, scripts, tests)")
    mo.add_argument("--seconds", type=float, default=0.0,
                    help="with --headless: stop after N seconds (0 = until Ctrl-C)")
    mo.set_defaults(func=cmd_monitor)

    pl = sub.add_parser("plan", help="pack the 64 tests into ~45 minute sittings")
    pl.add_argument("--minutes", type=float, default=45.0, help="recording budget per sitting")
    pl.add_argument("--remaining", action="store_true", help="only tests not yet recorded")
    pl.set_defaults(func=cmd_plan)

    pr = sub.add_parser("progress", help="per-test status: recorded, trained, gate outcome")
    pr.set_defaults(func=cmd_progress)

    vo = sub.add_parser("voice", help="analyse a session's microphone recording")
    vo.add_argument("--session", default=None)
    vo.add_argument("--record", type=float, default=0.0, metavar="SECONDS",
                    help="record a standalone sample instead of analysing one")
    vo.add_argument("--device", default=None, help="input device index or name")
    vo.add_argument("--fs", type=int, default=16000)
    vo.add_argument("--limit", type=int, default=25, help="trials to print")
    vo.set_defaults(func=cmd_voice)

    i = sub.add_parser("info", help="paths, settings, version")
    i.set_defaults(func=cmd_info)

    g = sub.add_parser("gui", help="open the control panel (system-menu entry point)")
    g.set_defaults(func=cmd_gui)

    dg = sub.add_parser("diagnose", help="write a shareable diagnostic bundle")
    dg.add_argument("--scan", action="store_true", help="also scan for the headband (10 s)")
    dg.add_argument("--mic", action="store_true", help="also probe the microphone (2 s)")
    dg.add_argument("--note", default=None, help="what went wrong, in your words")
    dg.add_argument("--print", action="store_true", help="also print the bundle")
    dg.set_defaults(func=cmd_diagnose)

    st = sub.add_parser("screen-test",
                        help="show the session surface for a few seconds and report "
                             "whether it covers the whole display")
    st.add_argument("--seconds", type=float, default=5.0)
    st.add_argument("--windowed", action="store_true", help="test the windowed path")
    st.set_defaults(func=cmd_screen_test)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_dirs()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        say("\ninterrupted")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        say(f"[red]error:[/red] {type(exc).__name__}: {exc}" if _console
            else f"error: {type(exc).__name__}: {exc}")
        if getattr(args, "verbose", False) or "--traceback" in (argv or sys.argv):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
