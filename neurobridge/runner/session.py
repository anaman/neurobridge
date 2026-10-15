"""Session runner: presentation, acquisition and marker alignment.

Timing model (read this before touching the loop)
-------------------------------------------------
The TH21A sends 10 samples per BLE notification at ~25 notifications/s, so
packets arrive in bursts and there is no per-sample hardware timestamp we can
trust.  The only clock shared by the stimulus code and the packet callback is
``time.perf_counter()`` in *this* process (``device.base.clock``).  Everything
below therefore follows four rules:

1. Never ``time.sleep`` for more than a few milliseconds.  Every wait goes
   through :meth:`Runner.wait` / ``_sleep_until``, which poll the source every
   <= 5 ms, stream packets into the :class:`~neurobridge.store.SessionWriter`,
   and drain the pygame event queue.  A 30 s break must not stall acquisition
   and a 5 ms slice must not lose a keystroke.
2. Timestamp the stimulus with the value the presenter returns *after* the
   frame flip / audio start, not with the time we started drawing or decoding.
   For audio the residual output latency is reported separately and can be
   folded in later via ``settings.output_latency_offset_s`` -- we never bake an
   unmeasured constant into the marker.
3. Emit the **class marker immediately before ``stim_onset`` at the same
   timestamp**, and duplicate the label into ``meta``.  Downstream epoching can
   then label an epoch from the marker code (``spec.label_map()``), the event
   name, or the metadata, which keeps this runner compatible with any of those
   conventions.  The event ``name`` is the spec's marker name verbatim
   (``eyes_open``, ``deviant``, ...): ``device/fake.py`` keys its injected band
   changes and ``agent/state.py`` keys its state labels off those exact names,
   so renaming one silently breaks both.
4. Marker codes come from the spec, never the global table.  Two specs in the
   same tier deliberately reuse a code (1210 means something different in T201
   and T203); ``spec.code(name, default)`` resolves the spec's own block first
   and only falls back to ``store.MARKER_CODES`` for framing names
   (``session_start``, ``test_start``, ``stim_offset``, ...).

Abort handling
--------------
``q``, ESC, a window close, SIGTERM or Ctrl-C all set an abort flag.  The
*current* trial is always finished (so the file never ends mid-stimulus), the
reason is written to ``notes.md``, the BLE client and the window are closed,
and a valid session directory is still returned: a truncated session with a
timing report is data, a half-written one is garbage.
"""
from __future__ import annotations

import os
import platform
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .. import __version__, store
from ..catalog.schema import TestSpec, load_one
from ..config import Settings, get_settings, new_session_id
from ..device import open_source
from ..device.base import (EEGSource, LSLMarkerSink, MarkerSink, MultiMarkerSink,
                           SessionMarkerSink, TimingReport, clock)
from ..device.quality import assess, contact_from_selfcheck
from ..stimuli.base import Stimulus, StimulusSet, prepare
from .present_audio import AudioPresenter
from .present_visual import VisualPresenter
from .timeline import (MISS_CLASSES, Phase, Trial, build_trials, make_phases,
                       miss_class, plan_summary, summarise)

#: Default length of the inter-test break in a multi-test sequence.
SEQUENCE_BREAK_S = 20.0
#: Session budget from docs/CATALOG_SPEC.md; exceeded -> loud warning.
SESSION_BUDGET_MIN = 45.0
#: How often (at most) the acquisition loop slices its sleep.
POLL_SLICE_S = 0.004
#: Default cue length when a spec has per-block instructions but no cue length.
DEFAULT_CUE_S = 2.0


class _NotifySink(MarkerSink):
    """Marker sink that forwards every marker to the EEG source.

    ``FakeSource.notify_marker`` schedules the stimulus-locked response that
    makes ``nb simulate`` a real end-to-end test, so this adapter is part of the
    marker chain rather than a special case in the trial loop -- exactly one
    injection per marker, even if another sink fails.
    """

    def __init__(self, source: EEGSource | None) -> None:
        self.source = source

    def send(self, code: int, name: str, t_mono: float,
             meta: dict[str, Any] | None = None) -> None:
        if self.source is None:
            return
        self.source.notify_marker(int(code), str(name), float(t_mono), meta or {})


@dataclass
class RunConfig:
    """Everything :func:`run_test` needs.  Defaults run a fake, no-display test."""

    spec: TestSpec
    device: str = "auto"
    session_id: str | None = None
    no_device: bool = False
    dry_run: bool = False
    duration_scale: float = 1.0
    fullscreen: bool | None = None
    verbose: bool = True
    limit: int | None = None
    allow_download: bool = True
    seed: int | None = None
    #: Opt-in microphone capture (voice.py) alongside the EEG, on the same clock.
    mic: bool = False
    mic_device: str | int | None = None
    #: Operator advancement through instruction/cue/break segments:
    #: ``"key"`` waits for SPACE (interactive sessions), ``"timer"`` auto-advances
    #: after the planned duration (rehearsal, headless, automated runs).
    advance: str = "key"
    #: Set by :func:`run_sequence`: several tests, one BLE connection, one session.
    sequence: list[TestSpec] | None = None


class Runner:
    """Drives one test (or a sequence of tests) against one EEG source."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.settings: Settings = get_settings()
        self.spec: TestSpec = config.spec
        self._specs: list[TestSpec] = list(config.sequence or [config.spec])
        seed = config.seed if config.seed is not None else self.settings.seed
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._sim_rng = np.random.default_rng(self.seed + 7)
        self.scale = max(1e-4, float(config.duration_scale or 1.0))
        self.session_id = config.session_id or new_session_id()
        self._offset = float(self.settings.output_latency_offset_s or 0.0)

        self.source: EEGSource | None = None
        self.writer: store.SessionWriter | None = None
        self.visual: VisualPresenter | None = None
        self.audio: AudioPresenter | None = None
        self.sink: MultiMarkerSink | None = None
        self.session_sink: SessionMarkerSink | None = None

        self._plans: dict[str, tuple[StimulusSet | None, list[Trial], list[Phase]]] = {}
        self._session_path: Path | None = None
        self._setup_done = False
        self._finalized = False
        self._torn_down = False
        self._summary: dict[str, Any] = {}

        self._samples: list[np.ndarray] = []
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._n_packets = 0
        self._events: list[dict[str, Any]] = []
        self._waits: list[tuple[float, float]] = []
        self._n_markers = 0
        self._n_trials_done = 0
        self._n_trials_planned = 0
        self._responses: list[dict[str, Any]] = []
        self._n_simulated_responses = 0
        self._warnings: list[str] = []
        self._abort = False
        self._abort_reason = ""
        self._traceback = ""
        self._advance = False
        self._cur_trial: Trial | None = None
        self._trial_onset: float | None = None
        self._trial_response: float | None = None
        self._prev_sigterm: Any = None
        self._fallback_stimuli_used = False
        # -- microphone (voice.py).  None means "not recording"; the error is
        # kept so the summary can say *why* instead of silently omitting it.
        self._mic: Any = None
        self._audio_path: Path | None = None
        self._audio_info: dict[str, Any] = {}
        self._audio_t_start: float | None = None
        self._audio_error: str | None = None
        # -- operator advancement + run.log + screensaver.
        self._advance_mode = "timer"          # effective: "key" | "timer"
        self._advance_reason = ""
        self._cap_hit = False
        self._screensaver_inhibit = "none"
        self._display: dict[str, Any] = {}
        self._log_fh: Any = None
        self._log_path: Path | None = None
        self._log_buffer: list[str] = []
        self._log_t0 = clock()
        self._stdin_fd: int | None = None
        self._stdin_mode: Any = None

    # ------------------------------------------------------------ properties
    @property
    def session_path(self) -> Path | None:
        return self._session_path

    @property
    def fs(self) -> float:
        return float(getattr(self.source, "fs", None) or self.settings.fs or 250.0)

    @property
    def aborted(self) -> bool:
        return bool(self._abort)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def is_fake(self) -> bool:
        return str(getattr(self.source, "name", "")) in ("fake", "synthetic", "sim")

    # ------------------------------------------------------------ setup
    def setup(self) -> None:
        """Open device, prepare stimuli, build the plan, open presenters/writer."""
        if self._setup_done:
            return
        # Signals first: opening a display/device can take a second, and a
        # SIGTERM during setup must still be recorded rather than killing the
        # process before anything is written.
        self._install_signal_handler()
        self.source = self._open_source()
        # Build every test's plan up front: a spec that cannot be prepared must
        # abort *before* the session directory exists.
        for spec in self._specs:
            self._plans[spec.id] = self._plan(spec)
        self._n_trials_planned = sum(len(p[1]) for p in self._plans.values())

        if not self.config.dry_run:
            self.visual = VisualPresenter(
                size=tuple(self.settings.screen_size),
                fullscreen=bool(self.settings.fullscreen if self.config.fullscreen is None
                                else self.config.fullscreen),
                backend=self.settings.visual_backend,
                echo=self._echo_text,
            )
            self.audio = AudioPresenter(backend=self.settings.audio_backend)
            self._warnings.extend(f"visual: {w}" for w in self.visual.warnings)
            self._warnings.extend(f"audio: {w}" for w in self.audio.warnings)

        # Resolve how operator-facing segments advance, and inhibit blanking,
        # *before* the writer so both land in the session meta and run.log header.
        self._resolve_advance()
        if self.visual is not None:
            try:
                self._screensaver_inhibit = self.visual.inhibit_screensaver()
            except Exception as exc:
                self._screensaver_inhibit = "none"
                self._warn(f"screensaver inhibit failed: {exc.__class__.__name__}: {exc}")
        # Snapshot the display facts while the window is still open: teardown
        # closes the presenter (and restores the cursor) before the summary is
        # written, so reading the live object then would report cursor_hidden=False.
        self._display = self._display_facts()

        self.writer = store.SessionWriter(session_id=self.session_id, meta=self._meta())
        self.session_sink = SessionMarkerSink(self.writer, test_id=self.spec.id)
        self.sink = MultiMarkerSink([self.session_sink, _NotifySink(self.source)])
        if self.settings.lsl_enabled:
            try:
                self.sink.add(LSLMarkerSink())
            except Exception as exc:
                self._warn(f"LSL disabled: {exc.__class__.__name__}: {exc}")

        self._open_log()
        self._start_mic()
        for w in self._warnings:
            self._say(f"[yellow]warning:[/yellow] {w}")
        self._setup_done = True

    def _open_source(self) -> EEGSource:
        """Resolve and open the device, falling back to the fake source.

        A real BLE connect can fail for a dozen reasons (headband off, adapter
        busy, another client connected).  Rather than aborting a session that is
        already late, we log it and continue on the synthetic source -- the
        session meta records which source actually produced the samples, so no
        analysis can be fooled.
        """
        name = "fake" if self.config.no_device else (self.config.device or self.settings.device or "auto")
        if str(name).lower() == "auto":
            from ..config import resolve_device_name
            name = resolve_device_name("auto")
        lname = str(name).lower()
        if lname in ("fake", "synthetic", "sim"):
            kwargs: dict[str, Any] = {"fs": self.settings.fs, "seed": self.seed}
        else:
            kwargs = {"fs": self.settings.fs, "address": self.settings.address,
                      "connect_timeout_s": self.settings.connect_timeout_s}
        src: EEGSource | None = None
        try:
            src = open_source(lname, **kwargs)
            src.open()
            return src
        except Exception as exc:
            self._warn(f"device {lname!r} failed to open ({exc.__class__.__name__}: {exc}); "
                       f"falling back to the synthetic source")
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
        from ..device.fake import FakeSource
        fake = FakeSource(fs=self.settings.fs, seed=self.seed)
        fake.open()
        return fake

    def _plan(self, spec: TestSpec) -> tuple[StimulusSet | None, list[Trial], list[Phase]]:
        """Prepare stimuli + build trials/phases for one spec."""
        stimset = self._prepare_stimuli(spec)
        trials = build_trials(spec, stimset, self.rng)
        limit = self.config.limit
        if limit is not None and 0 < int(limit) < len(trials):
            # Sample evenly across the whole plan instead of taking the first N:
            # condition/run designs are laid out as long runs of one class, so a
            # prefix would smoke-test a single class and hide half the pipeline.
            idx = np.unique(np.rint(np.linspace(0, len(trials) - 1, int(limit))).astype(int))
            trials = [trials[int(i)] for i in idx]
            for j, t in enumerate(trials):
                t.index = j
            blocks = max(1, int((spec.design or {}).get("blocks") or 1))
            present = {t.block for t in trials}
            if len(present) < min(blocks, len(trials)):
                n = len(trials)
                for i, t in enumerate(trials):   # re-spread blocks if sampling lost some
                    t.block = min(blocks - 1, (i * blocks) // max(1, n))
        phases = make_phases(spec, trials,
                             include_practice=bool((spec.design or {}).get("practice")))
        info = plan_summary(spec, trials, phases)
        self._say(f"[bold]{spec.id}[/bold] {spec.title} - shape={info.get('shape')} "
                  f"{len(trials)} trials, {len([p for p in phases if p.kind == 'block'])} blocks"
                  + (f", limit={limit}" if limit else "")
                  + (f", scale={self.scale:g}" if self.scale != 1.0 else ""))
        self._say(f"   timeline keys: {', '.join(info.get('keys') or []) or '(defaults)'}")
        return stimset, trials, phases

    def _prepare_stimuli(self, spec: TestSpec) -> StimulusSet:
        """Resolve the spec's stimuli, refusing only when nothing at all is left.

        Soft problems (a fallback was used, a file vanished) are printed and
        recorded: the stimuli module already degrades to synth/builtin, and a
        session that still presents *something* is worth more than no session.
        Hard problem (``source != none`` but zero items) refuses before the BLE
        connection and the session directory exist.
        """
        problems: list[str] = []
        stimset: StimulusSet | None = None
        try:
            stimset = prepare(spec, allow_download=self.config.allow_download,
                              verbose=self.config.verbose, limit=self.config.limit)
        except Exception as exc:
            msg = (f"stimuli.prepare unavailable for {spec.id}: "
                   f"{exc.__class__.__name__}: {exc}")
            self._warn(msg)
            problems.append(msg)
        generator = (spec.stimuli or {}).get("generator")
        if stimset is None or (spec.needs_stimuli and not stimset.items and not generator):
            stimset = self._fallback_stimuli(spec)
            self._fallback_stimuli_used = True
        try:
            problems.extend(stimset.check())
        except Exception:
            pass
        for p in problems:
            self._warn(f"{spec.id}: {p}")

        # Drop stimuli whose file disappeared between prepare() and now; a trial
        # pointing at a missing file would present a black frame and silently
        # corrupt the class balance.
        valid = [s for s in stimset.items if s.exists()]
        if len(valid) != len(stimset.items):
            dropped = len(stimset.items) - len(valid)
            self._warn(f"{spec.id}: dropped {dropped} stimulus file(s) missing from disk")
            stimset.items = valid

        if spec.needs_stimuli and not stimset.items and not generator:
            raise RuntimeError(
                f"refusing to start {spec.id} ({spec.slug}): the spec needs stimuli "
                f"but none could be resolved. Problems: {problems or ['unknown']}")
        return stimset

    # ------------------------------------------------------------ fallback stimuli
    def _fallback_stimuli(self, spec: TestSpec) -> StimulusSet:
        """Minimal offline stimulus resolver used only when ``prepare()`` fails.

        The full builder lives in ``neurobridge/stimuli/builder.py``; this keeps
        the runner usable (and every protocol smoke-testable) if that module is
        missing or raises, without editing anything outside ``runner/``.  It
        synthesizes tones via ``stimuli.synth`` and draws built-in fallback
        images with PIL, exactly as the builder's own fallback would.
        """
        stim = spec.stimuli or {}
        kind = str(stim.get("kind") or "none")
        out = StimulusSet(spec_id=spec.id)
        if not spec.needs_stimuli or stim.get("generator"):
            out.notes.append("no stimulus assets required (source none / generated frames)")
            return out
        out.warnings.append("runner fallback stimuli used (stimuli.prepare unavailable)")
        queries = stim.get("queries") or {}
        duration = float(stim.get("duration_s") or 0.0)
        for cls in spec.classes:
            names = queries.get(cls) or queries.get(str(cls)) or []
            if isinstance(names, str):
                names = [names]
            if kind in ("audio", "tone"):
                chosen = None
                for nm in names:
                    chosen = _synth_name(str(nm))
                    if chosen:
                        break
                chosen = chosen or "tone_1000"
                try:
                    from ..stimuli import synth
                    path = synth.generate(chosen)
                except Exception as exc:
                    self._warn(f"synth fallback failed for {cls}: {exc.__class__.__name__}: {exc}")
                    path = None
                out.items.append(Stimulus(kind=kind if kind in ("audio", "tone") else "audio",
                                          label=cls, path=path,
                                          duration_s=duration or 0.5, source="synth",
                                          meta={"generator": chosen}))
            elif kind == "text":
                for txt in _fallback_text(cls, names if names else [cls]):
                    out.items.append(Stimulus(kind="text", label=cls, text=txt,
                                              duration_s=duration or 1.0,
                                              source="textbank"))
            elif kind == "image":
                path = self._fallback_image(cls)
                out.items.append(Stimulus(kind="image", label=cls, path=path,
                                          duration_s=duration or 0.5,
                                          source="builtin"))
        return out

    def _fallback_image(self, cls: str) -> Path | None:
        """Draw a deterministic class-coloured PNG so image tests run offline."""
        try:
            from PIL import Image, ImageDraw
            from ..stimuli.base import kind_dir
            d = kind_dir("image") / "runner_fallback"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{_slug(cls)}.png"
            if path.exists():
                return path
            digest = sum(ord(c) * (i + 7) for i, c in enumerate(cls)) % 360
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(digest / 360.0, 0.55, 0.75)
            img = Image.new("RGB", (512, 512), (int(r * 255), int(g * 255), int(b * 255)))
            dr = ImageDraw.Draw(img)
            dr.rectangle([24, 24, 487, 487], outline=(245, 245, 245), width=6)
            dr.text((40, 250), cls[:24], fill=(20, 20, 20))
            img.save(path)
            return path
        except Exception as exc:
            self._warn(f"image fallback failed for {cls}: {exc.__class__.__name__}: {exc}")
            return None

    def _meta(self) -> dict[str, Any]:
        dev: dict[str, Any] = {}
        if self.source is not None:
            try:
                dev = dict(self.source.info)
            except Exception:
                dev = {}
        specs_meta: list[dict[str, Any]] = []
        for spec in self._specs:
            ss, trials, phases = self._plans.get(spec.id, (None, [], []))
            specs_meta.append({
                "id": spec.id, "slug": spec.slug, "title": spec.title,
                "tier": spec.tier, "tier_name": spec.tier_name,
                "feasibility": spec.feasibility, "modality": spec.modality,
                "classes": spec.classes, "agent_use": spec.agent_use,
                "duration_min": spec.duration_min,
                "timeline": plan_summary(spec, trials, phases),
                "plan": summarise(phases),
                "n_trials": len(trials),
                "stimuli": ss.summary() if ss is not None else {},
            })
        return {
            "session_id": self.session_id,
            "kind": "sequence" if len(self._specs) > 1 else "single",
            "version": __version__,
            "version_source": "neurobridge.__version__",   # git-less: no VCS at runtime
            "python": platform.python_version(),
            "created_mono": clock(),
            "spec_id": self.spec.id,
            "spec_slug": self.spec.slug,
            "tier": self.spec.tier,
            "tier_name": self.spec.tier_name,
            "feasibility": self.spec.feasibility,
            "test_title": self.spec.title,
            "modality": list(self.spec.modality),
            "classes": list(self.spec.classes),
            "tests": [s.id for s in self._specs],
            "specs": specs_meta,
            "device": dev,
            "fs": self.fs,
            "channel_name": self.settings.channel_name,
            "montage": dev.get("montage", f"single dry frontal electrode ({self.settings.channel_name})"),
            "settings": self.settings.as_dict(),
            "run": {
                "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "started_mono": clock(),
                "device_requested": "fake" if self.config.no_device else self.config.device,
                "dry_run": bool(self.config.dry_run),
                "duration_scale": self.scale,
                "limit": self.config.limit,
                "seed": self.seed,
                "allow_download": bool(self.config.allow_download),
                "fullscreen": self.config.fullscreen,
                "output_latency_offset_s": self._offset,
                "smoke_test": bool(self.config.dry_run or self.scale != 1.0 or self.config.limit),
                "simulated_inputs": bool(self.is_fake),
                "stimulus_fallback": bool(self._fallback_stimuli_used),
                "mic": bool(self.config.mic),
                "mic_device": self.config.mic_device,
            },
            "presentation": {
                **(self._display or self._display_facts()),
                "visual_backend": (self.visual.backend if self.visual is not None else "none (dry run)"),
                "visual_active": (self.visual.active if self.visual is not None else False),
                "audio_backend": (self.audio.backend if self.audio is not None else "none (dry run)"),
                "audio_latency_ms": (self.audio.latency_ms if self.audio is not None else 0.0),
                "advance": str(self.config.advance),
                "advance_effective": self._advance_mode,
                "advance_reason": self._advance_reason or None,
                "advance_cap_hit": bool(self._cap_hit),
            },
            "warnings": list(self._warnings),
        }

    # ------------------------------------------------------------ microphone
    AUDIO_START_CODE = 9500
    AUDIO_STOP_CODE = 9501

    def _start_mic(self) -> None:
        """Start opt-in microphone capture on the shared monotonic clock.

        Deliberately after the session directory and the EEG stream exist:
        ``audio.wav``/``audio.json`` live next to ``eeg.f32.npy``, and the
        recorder's ``t_start_mono`` is only comparable to ``stim_onset`` because
        both come from ``time.perf_counter()`` in this process.

        A microphone problem must never abort an EEG session, so *every* failure
        path warns once and continues silently without audio.
        """
        if not self.config.mic:
            return
        if self.writer is not None:
            self.writer.set_meta(audio_consent=True, mic_requested=True,
                                 audio_consent_wall=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 audio_consent_source="RunConfig.mic")
        if self.config.dry_run:
            self._audio_error = "dry run: microphone not started"
            self._say("[yellow]mic:[/yellow] dry run - no audio recorded")
            return
        try:
            from .. import voice
        except Exception as exc:
            self._audio_error = f"voice module unavailable ({exc.__class__.__name__}: {exc})"
            self._warn(f"mic not recorded: {self._audio_error}")
            return
        try:
            cfg = voice.MicConfig(device=self.config.mic_device,
                                  fs=int(getattr(voice, "DEFAULT_FS", 16000)),
                                  channels=1, consent=True)
            rec = voice.MicRecorder(cfg, out_dir=self.writer.path if self.writer else None)
            t_start = float(rec.start())
        except Exception as exc:                 # MicUnavailable and anything else
            self._audio_error = f"{exc.__class__.__name__}: {exc}"
            self._mic = None
            self._say(f"[yellow]mic:[/yellow] not recorded - {self._audio_error}")
            return
        self._mic = rec
        self._audio_t_start = t_start
        self._audio_info = dict(getattr(rec, "info", {}) or {})
        meta = {"t_start_mono": t_start, "fs": self._audio_info.get("fs"),
                "device": self._audio_info.get("device"),
                "input_latency_ms": self._audio_info.get("input_latency_ms"),
                "consent": True}
        # Explicit t_mono: the marker must carry the clock time of the first
        # captured sample block, not the (later) time we got around to logging it.
        self._mark_at(self.AUDIO_START_CODE, "audio_start", t_start, meta,
                      apply_offset=False)
        self._say(f"[bold]mic[/bold] recording via {self._audio_info.get('device')!r} "
                  f"@ {self._audio_info.get('fs')} Hz "
                  f"(input latency {self._audio_info.get('input_latency_ms')} ms)")

    def _stop_mic(self) -> None:
        """Stop capture, flush the WAV/JSON, and mark ``audio_stop``.

        Runs on every teardown path (normal, abort, exception) so an aborted
        session still keeps whatever was captured; the marker is written while
        the session writer is still open.
        """
        if self._mic is None:
            return
        rec = self._mic
        self._mic = None
        try:
            self._audio_path = rec.stop()
        except Exception as exc:
            self._audio_error = f"stop failed: {exc.__class__.__name__}: {exc}"
            self._warn(f"mic: {self._audio_error}")
        try:
            self._audio_info = dict(getattr(rec, "info", {}) or {})
        except Exception:
            pass
        if self._audio_t_start is None:
            self._audio_t_start = getattr(rec, "t_start", None)
        meta = {
            "t_start_mono": self._audio_t_start,
            "duration_s": self._audio_info.get("duration_s"),
            "rms_dbfs": self._audio_info.get("rms_dbfs"),
            "input_latency_ms": self._audio_info.get("input_latency_ms"),
            "path": str(self._audio_path) if self._audio_path else None,
            "error": self._audio_error,
        }
        try:
            self._mark_at(self.AUDIO_STOP_CODE, "audio_stop", clock(), meta,
                          apply_offset=False)
        except Exception as exc:
            self._warn(f"audio_stop marker failed: {exc.__class__.__name__}: {exc}")
        if self._audio_path is not None:
            self._say(f"[bold]mic[/bold] wrote {Path(self._audio_path).name} "
                      f"({self._audio_info.get('duration_s')} s, "
                      f"rms {self._audio_info.get('rms_dbfs')} dBFS)")
        else:
            self._say(f"[yellow]mic:[/yellow] no audio written ({self._audio_error or 'empty stream'})")

    def _mic_summary_line(self) -> str:
        info = self._audio_info or {}

        def _fmt(value: Any, suffix: str = "") -> str:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return "?"
            return f"{out:g}{suffix}" if np.isfinite(out) else "n/a"

        if self._audio_path is not None:
            dur = info.get("duration_s")
            fs = info.get("fs")
            rms = info.get("rms_dbfs")
            lat = info.get("input_latency_ms")
            khz = (float(fs) / 1000.0) if fs else None
            return (f"{_fmt(dur)} s @ {_fmt(khz, ' kHz')}, "
                    f"rms {_fmt(rms, ' dBFS')}, input latency {_fmt(lat, ' ms')}")
        if self.config.mic:
            return f"not recorded ({self._audio_error or 'unavailable'})"
        return "not recorded (--mic not set)"

    # ------------------------------------------------------------ run log
    def _echo_text(self, text: str) -> None:
        """Presenter echo hook: terminal stimulus prints also land in run.log."""
        print(text)
        self._log_line(text)

    def _log_line(self, text: str) -> None:
        """Append one line, prefixed with elapsed seconds, to run.log."""
        line = f"[{clock() - self._log_t0:9.3f}s] {text}"
        if self._log_fh is not None:
            try:
                self._log_fh.write(line + "\n")
            except Exception:
                pass
        else:
            self._log_buffer.append(line)

    def _log_text(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self._log_line(line)

    def _open_log(self) -> None:
        """Create/append ``<session_dir>/run.log`` and flush buffered lines.

        Appending (never truncating) means a resumed or repeated write into the
        same session directory keeps the earlier transcript, which is what the
        operator will attach to a bug report alongside the GUI console log.
        """
        if self.writer is None:
            return
        try:
            path = Path(self.writer.path) / "run.log"
            self._log_path = path
            self._log_fh = path.open("a", buffering=1)
            for line in self._header_lines():
                self._log_fh.write(line + "\n")
            for line in self._log_buffer:
                self._log_fh.write(line + "\n")
            self._log_buffer.clear()
        except Exception as exc:
            self._log_fh = None
            self._warn(f"run.log unavailable: {exc.__class__.__name__}: {exc}")

    def _display_facts(self) -> dict[str, Any]:
        """Display/window facts captured while the presenter is still open."""
        v = self.visual
        return {
            "fullscreen": bool(v.fullscreen) if v is not None else False,
            "desktop_size": (list(v.desktop_size) if v is not None and v.desktop_size else None),
            "window_size": (list(v.screen_size) if v is not None else None),
            "requested_window_size": (list(v.requested_size) if v is not None else None),
            "cursor_hidden": bool(v.cursor_hidden) if v is not None else False,
            "visual_sdl_driver": (v.sdl_driver if v is not None else ""),
            "visual_sdl_env": (v.sdl_env if v is not None else {}),
            "visual_open_path": (v.open_path if v is not None else "none"),
            "screensaver_inhibit": self._screensaver_inhibit,
        }

    def _header_lines(self) -> list[str]:
        spec = self.spec
        vis = self.visual
        aud = self.audio
        full = bool(getattr(vis, "fullscreen", False)) if vis is not None else False
        desk = list(getattr(vis, "desktop_size", None) or []) if vis else []
        win = list(getattr(vis, "screen_size", None) or []) if vis else []
        cursor = bool(getattr(vis, "cursor_hidden", False)) if vis else False
        requested = "fake" if self.config.no_device else self.config.device
        advance = f"requested={self.config.advance} effective={self._advance_mode}"
        if self._advance_reason:
            advance += f" ({self._advance_reason})"
        return [
            "=" * 78,
            f"NeuroBridge run.log   appended {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"session_id : {self.session_id}",
            f"spec       : {spec.id} {spec.slug}  (tier {spec.tier}, {spec.feasibility})",
            f"device     : requested={requested} actual={getattr(self.source, 'name', '?')}"
            f"  fs={self.fs:g} Hz  channel={self.settings.channel_name}",
            f"started    : {time.strftime('%Y-%m-%dT%H:%M:%S')}  (t0 mono {self._log_t0:.3f})",
            f"advance    : {advance}",
            f"fullscreen : {full}  desktop={desk or '?'}  window={win or '?'}"
            f"  cursor_hidden={cursor}",
            f"mic        : requested={bool(self.config.mic)}  device={self.config.mic_device}",
            f"backends   : visual={getattr(vis, 'backend', 'none')}"
            f"  audio={getattr(aud, 'backend', 'none')}"
            f"  audio_latency_ms={getattr(aud, 'latency_ms', 0.0)}",
            f"screensaver_inhibit: {self._screensaver_inhibit}",
            "=" * 78,
        ]

    # ------------------------------------------------------------ advancement
    def _resolve_advance(self) -> None:
        """Decide whether operator-facing segments wait for a key or a timer.

        ``"key"`` is the interactive default, but a session must never hang:
        with no interactive input at all (display backend inactive *and* stdin
        not a tty) and for rehearsals (``dry_run``) we silently fall back to
        ``"timer"``, say so in one line, and record the effective mode and the
        reason in session meta.
        """
        want = str(self.config.advance or "key").strip().lower()
        reason = ""
        if want not in ("key", "timer"):
            reason = f"unknown advance mode {want!r}"
            want = "timer"
        if want == "key" and self.config.dry_run:
            want, reason = "timer", "dry_run forces timer"
        if want == "key":
            interactive = bool(self.visual is not None and self.visual.active)
            try:
                tty = bool(sys.stdin is not None and sys.stdin.isatty())
            except Exception:
                tty = False
            if not interactive and not tty:
                want = "timer"
                reason = ("no interactive input (visual backend inactive and "
                          "stdin is not a tty)")
        self._advance_mode = want
        self._advance_reason = reason
        if reason:
            # Operational notice: emitted even in quiet mode, because silently
            # switching to auto-advance would change what the session means.
            self._console(f"advance: timer ({reason})")
            self._log_line(f"advance: timer ({reason})")
        if want == "key" and self.visual is not None and not self.visual.active:
            self._setup_stdin_keys()

    def _setup_stdin_keys(self) -> None:
        """cbreak stdin so SPACE/ENTER advances without needing line buffering."""
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._stdin_mode = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._stdin_fd = fd
        except Exception:
            self._stdin_fd = None
            self._stdin_mode = None

    def _restore_stdin(self) -> None:
        if self._stdin_fd is not None and self._stdin_mode is not None:
            try:
                import termios
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_mode)
            except Exception:
                pass
        self._stdin_fd = None
        self._stdin_mode = None

    def _poll_stdin(self) -> None:
        """Non-blocking SPACE/ENTER/q read for the no-display + tty case."""
        if self._stdin_fd is None:
            return
        try:
            import select
            ready, _, _ = select.select([self._stdin_fd], [], [], 0)
            if not ready:
                return
            data = os.read(self._stdin_fd, 1)
        except Exception:
            return
        if not data:
            return
        ch = data.decode("utf-8", "ignore")
        if ch in (" ", "\n", "\r"):
            self._advance = True
        elif ch in ("q", "\x1b"):
            self._abort = True
            self._abort_reason = self._abort_reason or "key 'q'/ESC on stdin"

    def _show_prompt(self, text: str, font_size: int = 32) -> None:
        if self.config.dry_run:
            self._log_line(text)
            return
        if self.visual is not None and self.visual.active:
            self.visual.show_text(text, font_size=font_size)
        else:
            self._console(text)

    def _operator_wait(self, seconds: float, prompt: str,
                       body: str | None = None) -> None:
        """Wait for the operator on an instruction/cue/break/end segment.

        ``key`` mode blocks on SPACE/RETURN with no planned ceiling (only a
        1800 s safety cap, logged if reached); ``timer`` mode keeps the planned
        duration with a visible countdown.  Abort always breaks the wait.
        """
        self._advance = False
        display = (body or prompt).strip()
        if self._advance_mode == "key":
            text = (f"{display}\n\nPress SPACE to continue" if display
                    else "Press SPACE to continue")
            self._show_prompt(text)
            self._log_line(f"advance: waiting for SPACE ({prompt})")
            cap = 1800.0
            end = clock() + cap
            while not self._advance and not self._abort:
                self._tick()
                if clock() >= end:
                    self._cap_hit = True
                    self._warn(f"advance key-wait hit the {cap:.0f}s safety cap; continuing")
                    break
                time.sleep(min(POLL_SLICE_S, max(0.0, end - clock())))
            self._advance = False
            return
        self._timer_segment_wait(seconds, display)

    def _timer_segment_wait(self, seconds: float, display: str) -> None:
        sec = max(0.0, float(seconds)) * self.scale
        if (self.visual is not None and self.visual.active and not self.config.dry_run
                and sec >= 1.0):
            start = clock()
            end = start + sec
            last: int | None = None
            while True:
                self._tick()
                if self._advance or self._abort:
                    break
                rem = end - clock()
                if rem <= 0:
                    break
                left = int(np.ceil(rem / self.scale)) if self.scale > 0 else 0
                if left != last:
                    txt = f"{display}\n\n{left}s" if display else f"{left}s"
                    self.visual.show_text(txt, font_size=32)
                    last = left
                time.sleep(min(POLL_SLICE_S, rem))
            self._waits.append((sec, clock() - start))
        else:
            if self.config.verbose:
                self._console(display)
            self.wait(seconds, until=lambda: self._advance)
        self._advance = False

    def _install_signal_handler(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            self._abort = True
            self._abort_reason = self._abort_reason or f"signal {signum}"
        try:
            self._prev_sigterm = signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError, RuntimeError):
            self._prev_sigterm = None      # not the main thread: Ctrl-C still works

    # ------------------------------------------------------------ markers
    def mark(self, code: int | str, name: str, meta: dict[str, Any] | None = None,
             *, t_mono: float | None = None) -> float:
        """Record one marker on the shared clock and fan it out to every sink.

        ``t_mono`` may be supplied explicitly for **back-dated** markers: T101
        and T108 only know whether a probe was a hit or a lapse after the
        response window, yet the marker must carry the *probe* timestamp (see
        the module docstring of ``timeline``).  Passing it here avoids
        re-stamping.
        """
        return self._mark_at(code, name, clock() if t_mono is None else t_mono, meta)

    def _mcode(self, key: str, default: int | None = None) -> int:
        """Marker code resolved from the spec first, the global table second."""
        if default is None:
            default = store.MARKER_CODES.get(key)
        return int(self.spec.code(key, default))

    def _mark_at(self, code: int | str, name: str, t_mono: float,
                 meta: dict[str, Any] | None = None, *, notify: bool = True,
                 apply_offset: bool = True) -> float:
        """Marker with an explicit timestamp (used with presenter onset times).

        ``settings.output_latency_offset_s`` is applied here and nowhere else:
        it is the one place where a measured output+acquisition latency is
        folded into the timeline, so an analysis can always un-apply it from
        the session meta.  ``notify=False`` suppresses the synthetic-source
        injection for back-dated markers that were already scheduled at their
        true onset.  ``apply_offset=False`` is for the microphone boundary
        markers: ``audio.json`` stores the recorder's raw ``t_start_mono`` on
        the same clock, so shifting only the markers would misalign speech.
        """
        t = float(t_mono) + (self._offset if apply_offset else 0.0)
        icode = int(code) if not isinstance(code, str) else self._mcode(code)
        meta = dict(meta or {})
        self._n_markers += 1
        if self.sink is not None:
            for sink in self.sink.sinks:
                if isinstance(sink, _NotifySink) and not notify:
                    continue
                try:
                    sink.send(icode, str(name), t, meta)
                except Exception:
                    pass
        self._events.append({"t": t, "code": icode, "name": str(name), "meta": meta})
        return t

    def _pre_notify(self, name: str, t_mono: float, meta: dict[str, Any] | None = None) -> None:
        """Schedule a synthetic response at ``t_mono`` without writing an event.

        Used by outcome-gated trials: the probe is delivered (and the fake
        source's response scheduled) at the true onset, but the class marker is
        only written after the response window, back-dated to that onset.
        """
        if self.source is None:
            return
        try:
            self.source.notify_marker(0, str(name), float(t_mono), dict(meta or {}))
        except Exception:
            pass

    # ------------------------------------------------------------ acquisition
    def _poll_device(self) -> None:
        """Drain arrived packets into the writer; never raise into the UI loop."""
        if self.source is None:
            return
        try:
            packets = self.source.poll(0.0)
        except Exception as exc:
            self._warn(f"device poll failed: {exc.__class__.__name__}: {exc}")
            return
        for p in packets:
            self._n_packets += 1
            if self.config.dry_run or self.writer is None:
                continue
            try:
                self.writer.add_samples(p.values, p.t_samples)
            except Exception as exc:
                self._warn(f"add_samples failed: {exc.__class__.__name__}: {exc}")
                continue
            arr = np.asarray(p.values, dtype=np.float64).reshape(-1)
            self._samples.append(arr)
            ts = np.asarray(p.t_samples, dtype=np.float64).reshape(-1)
            if ts.size:
                if self._t_first is None:
                    self._t_first = float(ts[0])
                self._t_last = float(ts[-1])

    def _tick(self, poll_cb: Callable[[], Any] | None = None) -> None:
        """One <= 5 ms slice: acquire samples, drain display events, run cb."""
        def _cb() -> None:
            self._poll_device()
            if poll_cb is not None:
                try:
                    poll_cb()
                except Exception:
                    pass

        self._poll_stdin()               # only armed for no-display + tty key mode
        if self.visual is not None:
            # pump(0.0) drains the SDL queue without sleeping and hands back any
            # key/quit events with their timestamps.
            for ev in self.visual.pump(0.0, poll_cb=_cb):
                self._handle_event(ev)
        else:
            _cb()

    def _sleep_until(self, end_real: float,
                     periodic: Sequence[tuple[str, float]] | None = None) -> None:
        """Poll-and-sleep until ``end_real`` (a monotonic deadline)."""
        t0 = clock()
        nxt: dict[str, float] = {name: period for name, period in (periodic or [])}
        while True:
            self._tick()
            now = clock()
            if periodic and self.scale > 0:
                elapsed = (now - t0) / self.scale
                for name, period in periodic:
                    while elapsed >= nxt.get(name, period) and period > 0:
                        self._mark(self._mcode(name), name,
                                   {"period_s": period, "test": self.spec.id})
                        nxt[name] = nxt.get(name, period) + period
            if self._abort:
                return
            rem = end_real - now
            if rem <= 0:
                return
            time.sleep(min(POLL_SLICE_S, rem))

    def wait(self, seconds: float, poll_cb: Callable[[], Any] | None = None, *,
             scale: bool = True, record: bool | None = None,
             until: Callable[[], bool] | None = None) -> bool:
        """Poll-friendly sleep used for breaks, instructions and ITIs.

        ``scale`` multiplies the duration by ``config.duration_scale``.  Waits
        (ITI, rest, breaks, instructions, sustained holds) are scaled for smoke
        tests; a brief *discrete* stimulus presentation is not -- a 100 ms ERP
        tone must actually sound for 100 ms or the protocol is no longer the
        protocol.  All timings pass through here or ``_sleep_until`` so
        acquisition can never stall.
        """
        sec = max(0.0, float(seconds))
        real = sec * (self.scale if scale else 1.0)
        if record is None:
            record = bool(scale)
        start = clock()
        end = start + real
        while True:
            self._tick(poll_cb)
            if until is not None and until():
                break
            if self._abort:
                break
            rem = end - clock()
            if rem <= 0:
                break
            time.sleep(min(POLL_SLICE_S, rem))
        actual = clock() - start
        if record:
            # Record (requested, actual) in REAL seconds: a compressed wait is
            # still a real wait, and dividing by duration_scale would turn a
            # 2 ms scheduling overhead into a 100 ms phantom jitter.
            self._waits.append((real, actual))
        return not self._abort

    def _wait_iti(self, onset: float, end_real: float, nominal: float) -> None:
        """Finishing ITI of a trial, with the wait error tracked for the report."""
        t0 = clock()
        requested = max(0.0, end_real - t0)
        self._sleep_until(end_real)
        if requested > 0:
            self._waits.append((requested, clock() - t0))

    # ------------------------------------------------------------ input
    def _handle_event(self, ev: dict[str, Any]) -> None:
        typ = ev.get("type")
        if typ == "quit":
            self._abort = True
            self._abort_reason = self._abort_reason or "display window closed"
            return
        if typ != "key":
            return
        key = str(ev.get("key") or "").lower()
        t = float(ev.get("t") or clock())
        if key in ("q", "escape"):
            self._abort = True
            self._abort_reason = self._abort_reason or f"key {key!r}"
            return
        if key in ("space", "return", "enter"):
            if self._cur_trial is not None and self._trial_onset is not None:
                rt = t - self._trial_onset
                if -0.05 <= rt < 60.0:
                    self._trial_response = float(rt)
                    self._emit_response(self._cur_trial, t, float(rt), key)
            else:
                self._advance = True

    def _emit_response(self, trial: Trial, t_mono: float, rt: float, key: str,
                       *, simulated: bool = False) -> None:
        meta = {"rt": round(float(rt), 5), "key": key, "trial": trial.index,
                "label": trial.label, "class": trial.label, "block": trial.block}
        if simulated:
            meta["simulated"] = True
        self._responses.append({"t": float(t_mono), "rt": float(rt), "trial": trial.index,
                                "label": trial.label, "block": trial.block,
                                "simulated": bool(simulated)})
        self._mark_at(self._mcode("response", store.MARKER_CODES.get("response")),
                      "response", t_mono, meta)

    # ------------------------------------------------------------ presentation
    def _text_font_size(self, stim: Stimulus) -> int:
        n = len(stim.text or "")
        if n > 600:
            return 20
        if n > 220:
            return 26
        return 32

    def _show(self, trial: Trial, body_real: float) -> float | None:
        """Start presenting one stimulus; return the presenter's onset timestamp."""
        if self.config.dry_run:
            return None
        gen = (self.spec.stimuli or {}).get("generator")
        if gen == "flash_frames":
            if self.visual is not None and self.visual.active:
                return self.visual.flash(duration_s=max(0.0, body_real))
            return None
        if gen == "flicker_frames":
            return None                      # driven by flicker() during the hold
        stim = trial.stim
        if stim is None:
            if self.visual is not None and self.visual.active:
                return self.visual.show_fixation()
            return None
        if stim.kind in ("audio", "tone"):
            if self.audio is None:
                return None
            if stim.path:
                t = self.audio.play(stim.path)
                if self.audio.latency_ms:
                    self._audio_latency_ms = self.audio.latency_ms
                return t
            if stim.kind == "tone":
                freq = float(stim.meta.get("freq") or 1000.0)
                dur = float(stim.duration_s or body_real or 0.3)
                return self.audio.play_tone(freq, dur)
            return None
        if self.visual is None:
            return None
        if stim.kind == "image" and stim.path:
            return self.visual.show_image(stim.path)
        if stim.kind == "text":
            return self.visual.show_text(stim.text, font_size=self._text_font_size(stim))
        return self.visual.show_fixation()

    def _stop_stimulus(self, trial: Trial) -> None:
        if self.audio is None or trial.stim is None:
            return
        if trial.stim.kind in ("audio", "tone"):
            try:
                self.audio.stop()
            except Exception:
                pass

    def _clear_screen(self) -> None:
        if self.config.dry_run or self.visual is None or not self.visual.active:
            return
        try:
            self.visual.show_blank()
        except Exception:
            pass

    def _class_meta(self, trial: Trial, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "label": trial.label, "class": trial.label or None,
            "block": trial.block, "trial": trial.index,
            "rare": bool((trial.meta or {}).get("rare")),
            "practice": bool((trial.meta or {}).get("practice")),
            "synth": trial.label,            # FakeSource keys its response on this
            "kind": (trial.stim.kind if trial.stim is not None
                     else str((self.spec.stimuli or {}).get("kind") or "none")),
            "mode": (trial.meta or {}).get("mode"),
        }
        for key in ("level", "level_db", "condition", "phase", "of", "passage", "cue"):
            if key in (trial.meta or {}):
                meta[key] = (trial.meta or {})[key]
        if extra:
            meta.update(extra)
        return meta

    def _emit_onset(self, trial: Trial, t_mono: float, *, label: str | None = None,
                    onset_code: int | None = None, notify: bool = True) -> None:
        """Class marker immediately before ``stim_onset`` at the same instant."""
        lab = trial.label if label is None else label
        code = trial.onset_code if onset_code is None else onset_code
        meta = self._class_meta(trial)
        if label is not None:
            meta["label"] = lab
            meta["class"] = lab
            meta["synth"] = lab
        stim = trial.stim
        meta["stim"] = stim.key if stim is not None else None
        meta["source"] = stim.source if stim is not None else "none"
        if getattr(self, "_audio_latency_ms", 0.0):
            meta["audio_latency_ms"] = self._audio_latency_ms
        if lab and code:
            self._mark_at(code, lab, t_mono, meta, notify=notify)
        self._mark_at(self._mcode("stim_onset"), "stim_onset", t_mono, meta, notify=notify)

    def _periodic(self) -> list[tuple[str, float]]:
        """Periodic markers to emit while a sustained window is held (T707/T708)."""
        d = self.spec.design or {}
        out: list[tuple[str, float]] = []
        for key, name in (("probe_every_s", "probe"), ("feedback_update_s", "feedback")):
            try:
                period = float(d.get(key))
            except (TypeError, ValueError):
                continue
            if period > 0 and name in (self.spec.markers or {}):
                out.append((name, period))
        return out

    def _trial_timing(self, trial: Trial) -> tuple[float, float, str]:
        body = float((trial.meta or {}).get("body_s") or 0.0)
        itv = float((trial.meta or {}).get("itv_s") or body)
        mode = str((trial.meta or {}).get("mode") or "stim")
        return max(0.0, body), max(body, itv), mode

    # ------------------------------------------------------------ trial loop
    def _run_trial(self, trial: Trial) -> None:
        body, itv, mode = self._trial_timing(trial)
        self._cur_trial = trial
        self._trial_response = None
        self._audio_latency_ms = 0.0
        try:
            if mode == "gated":
                self._run_gated(trial, body, itv)
            elif mode == "seating":
                self._run_seating(trial, body, itv)
            elif mode == "hold":
                self._run_hold(trial, body, itv)
            else:
                self._run_stim(trial, body, itv)
        finally:
            self._cur_trial = None
            self._trial_onset = None
            self._n_trials_done += 1

    def _run_stim(self, trial: Trial, body: float, itv: float) -> None:
        """Discrete stimulus: presented at real speed, ITI compressed for smoke tests."""
        t_show = self._show(trial, body) if body > 0 else None
        onset = t_show if t_show is not None else clock()
        self._trial_onset = onset
        self._emit_onset(trial, onset)
        presented = 0.0
        if self._has_presentable(trial) and body > 0:
            self._sleep_until(onset + body)      # presentation is never scaled
            presented = body
        self._stop_stimulus(trial)
        if trial.stim is not None:
            self._mark(self._mcode("stim_offset", store.MARKER_CODES.get("stim_offset")),
                       "stim_offset", {"trial": trial.index, "block": trial.block,
                                       "label": trial.label, "class": trial.label})
        self._clear_screen()
        remaining = max(0.0, itv - body)
        self._wait_iti(onset, onset + presented + remaining * self.scale, remaining)
        self._log_trial(trial, body, itv, presented)

    def _run_hold(self, trial: Trial, body: float, itv: float) -> None:
        """Sustained class window: the whole body is compressed by duration_scale."""
        body_real = body * self.scale
        gen = (self.spec.stimuli or {}).get("generator")
        periodic = self._periodic()
        if gen == "flicker_frames" and self.visual is not None and self.visual.active:
            # A real-time flicker cannot be compressed to a zero-length window,
            # so it runs at wall-clock speed; flicker() returns the first flip.
            hz = _flicker_hz(trial.label) or 10.0
            t_show = self.visual.flicker(hz, body_real, poll_cb=self._tick)
        else:
            t_show = self._show(trial, body_real)
            onset = t_show if t_show is not None else clock()
            self._trial_onset = onset
            self._emit_onset(trial, onset)
            self._sleep_until(onset + body_real, periodic)
            self._stop_stimulus(trial)
            self._mark(self._mcode("stim_offset", store.MARKER_CODES.get("stim_offset")),
                       "stim_offset", {"trial": trial.index, "block": trial.block,
                                       "label": trial.label, "class": trial.label})
            self._clear_screen()
            remaining = max(0.0, itv - body)
            self._wait_iti(onset, onset + body_real + remaining * self.scale, remaining)
            self._log_trial(trial, body, itv, body_real)
            return
        # Flicker path: the body is already spent; only bookkeeping is left.
        onset = t_show if t_show is not None else clock()
        self._trial_onset = onset
        self._emit_onset(trial, onset)
        self._stop_stimulus(trial)
        self._mark(self._mcode("stim_offset", store.MARKER_CODES.get("stim_offset")),
                   "stim_offset", {"trial": trial.index, "block": trial.block,
                                   "label": trial.label, "class": trial.label})
        self._clear_screen()
        remaining = max(0.0, itv - body)
        self._wait_iti(onset, onset + body_real + remaining * self.scale, remaining)
        self._log_trial(trial, body, itv, body_real)

    def _run_gated(self, trial: Trial, body: float, itv: float) -> None:
        """Outcome-gated probe (T101 SART, T108 microsleep).

        The probe is delivered and the synthetic response scheduled at the true
        onset, but the class is only known after the response window: a hit is
        the go class, a miss is the ``lapse``/``missed`` class.  The class marker
        and ``stim_onset`` are therefore written *after* the window and
        **back-dated** to the probe timestamp, class marker first, so the epoch
        label is unambiguous (``epochs.py`` resolves ``meta.label`` too).
        """
        classes = list(self.spec.classes or [])
        miss = miss_class(classes) or (classes[-1] if classes else "")
        go = next((c for c in classes if c != miss), "")
        nominal = trial.label or go
        window_real = body * self.scale
        t_show = self._show(trial, min(body, 1.0) * self.scale if body else 0.0)
        onset = t_show if t_show is not None else clock()
        self._trial_onset = onset
        self._pre_notify(nominal, onset, self._class_meta(trial))
        # Simulated subject: the synthetic source stands in for a person, so a
        # fake session also exercises the response path (marked meta.simulated).
        if self.is_fake and nominal and _norm(nominal) not in MISS_CLASSES:
            if float(self._sim_rng.random()) < 0.9:
                rt = 0.3 + 0.3 * float(self._sim_rng.random())
                end = onset + min(rt, body) * self.scale
                self._sleep_until(end)
                self._trial_response = rt
                self._n_simulated_responses += 1
                self._emit_response(trial, end, rt, "space", simulated=True)
        self._sleep_until(onset + window_real)
        responded = self._trial_response is not None
        if responded:
            outcome = go
        elif self.is_fake:
            # Keep the scheduled class so a simulated session still contains
            # both classes; real sessions use the actual outcome (a miss).
            outcome = nominal
        else:
            outcome = miss
        self._stop_stimulus(trial)
        self._emit_onset(trial, onset, label=outcome,
                         onset_code=_code_for(self.spec, outcome), notify=False)
        self._mark(self._mcode("stim_offset", store.MARKER_CODES.get("stim_offset")),
                   "stim_offset", {"trial": trial.index, "block": trial.block,
                                   "label": outcome, "class": outcome,
                                   "responded": responded})
        self._clear_screen()
        remaining = max(0.0, itv - body)
        self._wait_iti(onset, onset + window_real + remaining * self.scale, remaining)
        self._log_trial(trial, body, itv, window_real)

    def _run_seating(self, trial: Trial, body: float, itv: float) -> None:
        """T001-style seating attempt: cue, settle, contact verdict, hold."""
        d = self.spec.design or {}
        cue = _num(d.get("cue_duration_s")) or 0.0
        settle = _num(d.get("settle_s")) or 0.0
        text = _format_instruction(d.get("instruction"), block=trial.block + 1,
                                   trial=trial.index + 1)
        if self.visual is not None and self.visual.active and text:
            self.visual.show_text(text, font_size=26)
        elif text:
            print(text)
        if "cue_onset" in (self.spec.markers or {}):
            self._mark(self._mcode("cue_onset"), "cue_onset", {"block": trial.block})
        if cue > 0:
            self.wait(cue)
        if "response_prompt" in (self.spec.markers or {}):
            self._mark(self._mcode("response_prompt"), "response_prompt",
                       {"block": trial.block})
        if settle > 0:
            self.wait(settle)
        contact = str(getattr(self.source, "contact", "") or "")
        n_ok = contact_from_selfcheck(contact)[1] if contact else 0
        label = trial.label
        classes = list(self.spec.classes or [])
        if len(classes) >= 2:
            label = classes[0] if n_ok >= 3 else classes[1]
        onset = clock()
        self._trial_onset = onset
        if "attempt_start" in (self.spec.markers or {}):
            self._mark(self._mcode("attempt_start"), "attempt_start",
                       {"block": trial.block, "contact": contact, "n_ok": n_ok})
        self._emit_onset(trial, onset, label=label, onset_code=_code_for(self.spec, label))
        self._sleep_until(onset + body * self.scale, self._periodic())
        self._mark(self._mcode("stim_offset", store.MARKER_CODES.get("stim_offset")),
                   "stim_offset", {"trial": trial.index, "label": label, "class": label})
        self._clear_screen()
        if "attempt_end" in (self.spec.markers or {}):
            self._mark(self._mcode("attempt_end"), "attempt_end",
                       {"block": trial.block, "contact": contact, "n_ok": n_ok})
        remaining = max(0.0, itv - body)
        self._wait_iti(onset, onset + body * self.scale + remaining * self.scale, remaining)
        self._log_trial(trial, body, itv, body * self.scale)

    def _has_presentable(self, trial: Trial) -> bool:
        if (self.spec.stimuli or {}).get("generator"):
            return True
        return trial.stim is not None

    def _log_trial(self, trial: Trial, body: float, itv: float, presented: float) -> None:
        self._say(f"   trial {trial.index + 1}: {trial.label or '(unlabelled)':>16s} "
                  f"mode={(trial.meta or {}).get('mode')} body={body:.2f}s itv={itv:.2f}s"
                  f" presented={presented:.2f}s"
                  + ("  [rare]" if (trial.meta or {}).get("rare") else ""))

    # ------------------------------------------------------------ phases
    def _instruction_text(self) -> str:
        spec = self.spec
        d = spec.design or {}
        custom = _format_instruction(d.get("instruction"), block=1, trial=1)
        lines = [
            spec.title,
            "",
            f"{spec.id}  -  {spec.feasibility}  -  about {spec.duration_min:.0f} min",
            "",
        ]
        if custom:
            lines.append(custom)
        else:
            lines += [spec.goal.strip(), "", "Classes: " + ", ".join(spec.classes)]
        lines += ["", "SPACE = response      Q / ESC = abort"]
        if not (self.visual is not None and self.visual.active):
            lines.insert(-1, "Auto-advancing (no display).")
        return "\n".join(x for x in lines if x is not None)

    def _run_instruction(self, phase: Phase) -> None:
        if self._abort:
            return
        dur = float(phase.duration_s if phase.duration_s is not None
                    else (self.spec.design or {}).get("instruction_s") or 6.0)
        text = self._instruction_text()
        self._mark(self._mcode("instruction", store.MARKER_CODES.get("instruction")),
                   "instruction", {"test": self.spec.id, "duration_s": dur})
        self._operator_wait(dur, "Instructions", body=text)
        self._clear_screen()

    def _block_cue(self, phase: Phase) -> None:
        """Show the spec's per-block instruction text (T001, tier 6/7)."""
        d = self.spec.design or {}
        text = _format_instruction(d.get("instruction"), block=phase.trials[0].block + 1
                                   if phase.trials else 1, trial=1)
        if not text:
            return
        dur = _num(d.get("cue_duration_s")) or DEFAULT_CUE_S
        if "cue_onset" in (self.spec.markers or {}):
            self._mark(self._mcode("cue_onset"), "cue_onset",
                       {"block": phase.trials[0].block if phase.trials else None,
                        "phase": phase.label})
        self._operator_wait(dur, phase.label, body=text)

    def _mark(self, code: int | str, name: str,
              meta: dict[str, Any] | None = None) -> float:
        return self._mark_at(code, name, clock(), meta)

    def _run_rest(self, phase: Phase) -> None:
        dur = float(phase.duration_s or 0.0)
        self._mark(self._mcode("break_start", store.MARKER_CODES.get("break_start")),
                   "break_start", {"phase": phase.label, "duration_s": dur})
        self._operator_wait(dur, phase.label, body="Break - look away, blink, relax.")
        self._clear_screen()
        self._mark(self._mcode("break_end", store.MARKER_CODES.get("break_end")),
                   "break_end", {"phase": phase.label})

    def _run_one(self, spec: TestSpec) -> None:
        """Run every phase of one test.  The current trial is always finished."""
        self.spec = spec
        _, trials, phases = self._plans[spec.id]
        if self.session_sink is not None:
            self.session_sink.test_id = spec.id
            self.session_sink.block = None
        self._say(f"[bold cyan]== {spec.id} {spec.title} ==[/bold cyan]")
        self._mark(self._mcode("test_start", store.MARKER_CODES.get("test_start")),
                   "test_start", {"spec": spec.id, "slug": spec.slug,
                                  "trials": len(trials), "phase": "test"})
        for phase in phases:
            if self._abort:
                break
            if phase.kind == "instruction":
                self._run_instruction(phase)
            elif phase.kind in ("block", "practice"):
                self._run_block(phase)
            elif phase.kind == "rest":
                self._run_rest(phase)
            if self._abort:
                break
        self._mark(self._mcode("test_end", store.MARKER_CODES.get("test_end")),
                   "test_end", {"spec": spec.id, "trials_done": self._n_trials_done,
                                "phase": "test"})

    def _run_block(self, phase: Phase) -> None:
        practice = phase.kind == "practice"
        block = -1 if practice else int(phase.trials[0].block if phase.trials else 0)
        if self.session_sink is not None:
            self.session_sink.block = None if practice else block
        start_key = "practice_start" if practice else "block_start"
        end_key = "practice_end" if practice else "block_end"
        self._mark(self._mcode(start_key, store.MARKER_CODES.get(start_key)), start_key,
                   {"block": block, "phase": phase.label, "n_trials": len(phase.trials),
                    "practice": practice})
        self._say(f"  -- {phase.label} ({len(phase.trials)} trials) --")
        if not practice and phase.trials:
            self._block_cue(phase)
        for trial in list(phase.trials):
            if self._abort:
                break                      # abort is checked *between* trials only
            self._run_trial(trial)
        self._mark(self._mcode(end_key, store.MARKER_CODES.get(end_key)), end_key,
                   {"block": block, "phase": phase.label, "practice": practice})

    def _inter_test_break(self, current: TestSpec, upcoming: TestSpec) -> None:
        self._mark(self._mcode("break_start", store.MARKER_CODES.get("break_start")),
                   "break_start", {"tests": [current.id, upcoming.id],
                                   "phase": "between_tests"})
        self._operator_wait(SEQUENCE_BREAK_S, "Next test",
                            body=f"Next: {upcoming.id} {upcoming.title}")
        self._clear_screen()
        self._mark(self._mcode("break_end", store.MARKER_CODES.get("break_end")),
                   "break_end", {"tests": [current.id, upcoming.id],
                                 "phase": "between_tests"})

    def _run_all(self) -> None:
        # Discard anything the source queued before the session clock started.
        if self.source is not None:
            try:
                self.source.poll(0.0)
            except Exception:
                pass
        self._mark(self._mcode("session_start", store.MARKER_CODES.get("session_start")),
                   "session_start", {"tests": [s.id for s in self._specs],
                                     "session_id": self.session_id})
        if not self._abort:
            first = self._specs[0]
            self._operator_wait(3.0, "Ready to begin",
                                body=f"Ready to begin: {first.id} {first.title}")
        for i, spec in enumerate(self._specs):
            if self._abort:
                break
            self._run_one(spec)
            if self._abort:
                break
            if i + 1 < len(self._specs):
                self._inter_test_break(spec, self._specs[i + 1])
        if not self._abort:
            self._operator_wait(3.0, "Session complete",
                                body="Session complete - thank you.")
        self._mark(self._mcode("session_end", store.MARKER_CODES.get("session_end")),
                   "session_end", {"aborted": bool(self._abort),
                                   "reason": self._abort_reason or None})

    # ------------------------------------------------------------ lifecycle
    def run(self) -> Path:
        """Run the session and always return a usable session path."""
        try:
            if not self._setup_done:
                self.setup()
            self._run_all()
        except KeyboardInterrupt:
            self._abort = True
            self._abort_reason = self._abort_reason or "KeyboardInterrupt"
        except Exception as exc:                    # never lose a session to a bug
            self._abort = True
            self._abort_reason = self._abort_reason or f"{exc.__class__.__name__}: {exc}"
            self._traceback = traceback.format_exc()
            self._warn(f"session error: {exc.__class__.__name__}: {exc}")
        finally:
            self.teardown()
        return self._session_path or Path(".")

    def teardown(self) -> None:
        """Close presenters/source/writer.  Idempotent; never leaves a half-open
        BLE client or a stuck pygame window."""
        if self._torn_down:
            return
        self._torn_down = True
        if self._prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except (ValueError, OSError, RuntimeError):
                pass
            self._prev_sigterm = None
        # Release the screensaver/blanking inhibitor explicitly (idempotent):
        # this must happen on the abort path too, which always reaches teardown.
        if self.visual is not None:
            try:
                self.visual.release_screensaver()
            except Exception:
                pass
        self._restore_stdin()
        # Window first: the operator must never be trapped behind a fullscreen
        # frame because a later step threw.
        for closer in (getattr(self.visual, "close", None), getattr(self.audio, "close", None)):
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        # Microphone before the writer closes: the WAV is flushed and the
        # audio_stop marker is appended while events.jsonl is still open.  This
        # runs on the abort path too, so partial speech is never lost.
        self._stop_mic()
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
        if self.sink is not None:
            try:
                self.sink.close()
            except Exception:
                pass
        self._finalize()
        if self.writer is not None:
            try:
                self._write_notes()
            except Exception:
                pass
            try:
                self._session_path = self.writer.close(summary=self._summary)
            except Exception as exc:
                self._warn(f"close failed: {exc.__class__.__name__}: {exc}")
                self._session_path = getattr(self.writer, "path", None)
        self._print_summary()
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        values = np.concatenate(self._samples) if self._samples else np.zeros(0, dtype=np.float64)
        if values.size:
            try:
                contact = getattr(self.source, "contact", "unknown")
                quality: dict[str, Any] = assess(values, fs=self.fs, contact=contact).as_dict()
            except Exception as exc:
                quality = {"verdict": "UNKNOWN", "score": 0,
                           "error": f"{type(exc).__name__}: {exc}"}
        else:
            quality = {"verdict": "N/A", "score": 0, "skipped": "no samples recorded",
                       "messages": ["quality assessment skipped: dry run or no device"]}
        timing = self._timing_report()
        stats: dict[str, Any] = {}
        if self.source is not None:
            try:
                stats = dict(self.source.stats())
            except Exception:
                stats = {}
        soa_ms: float | None = None
        onsets = [e["t"] for e in self._events if e["name"] == "stim_onset"]
        if len(onsets) >= 2:
            soa_ms = float(np.median(np.diff(np.asarray(onsets, dtype=float))) * 1000.0)
        span = (float(self._t_last - self._t_first)
                if self._t_first is not None and self._t_last is not None else 0.0)
        fs_eff = float(values.size / span) if span > 0 else 0.0
        self._summary = {
            "n_samples": int(values.size),
            "duration_s": round(span, 3),
            "fs_effective": round(fs_eff, 3),
            "n_events": len(self._events),
            "n_markers": self._n_markers,
            "n_packets": self._n_packets,
            "n_responses": len(self._responses),
            "n_simulated_responses": self._n_simulated_responses,
            "n_trials_done": self._n_trials_done,
            "n_trials_planned": self._n_trials_planned,
            "median_soa_ms": (round(soa_ms, 3) if soa_ms is not None else None),
            "quality": quality,
            "timing": timing.as_dict(),
            "device": stats,
            "aborted": bool(self._abort),
            "abort_reason": self._abort_reason or None,
            "dry_run": bool(self.config.dry_run),
            "duration_scale": self.scale,
            "limit": self.config.limit,
            "smoke_test": bool(self.config.dry_run or self.scale != 1.0 or self.config.limit),
            "warnings": list(self._warnings),
            "audio": bool(self._audio_path is not None),
            "audio_t_start_mono": self._audio_t_start,
            "audio_duration_s": (self._audio_info or {}).get("duration_s"),
            "audio_rms_dbfs": (self._audio_info or {}).get("rms_dbfs"),
            "audio_input_latency_ms": (self._audio_info or {}).get("input_latency_ms"),
            "audio_error": self._audio_error,
            "mic_requested": bool(self.config.mic),
            "advance": str(self.config.advance),
            "advance_effective": self._advance_mode,
            "advance_reason": self._advance_reason or None,
            "advance_cap_hit": bool(self._cap_hit),
            "fullscreen": bool((self._display or {}).get("fullscreen")),
            "desktop_size": (self._display or {}).get("desktop_size"),
            "window_size": (self._display or {}).get("window_size"),
            "visual_sdl_driver": (self._display or {}).get("visual_sdl_driver", ""),
            "cursor_hidden": bool((self._display or {}).get("cursor_hidden")),
            "screensaver_inhibit": self._screensaver_inhibit,
            "run_log": (str(self._log_path) if self._log_path is not None else None),
        }

    def _timing_report(self) -> TimingReport:
        tr = TimingReport()
        notes: list[str] = []
        times = np.asarray([e["t"] for e in self._events], dtype=float)
        if times.size >= 2:
            tr.mean_interval_ms = float(np.mean(np.diff(times)) * 1000.0)
        if self._waits:
            req = np.asarray([w[0] for w in self._waits], dtype=float) * 1000.0
            act = np.asarray([w[1] for w in self._waits], dtype=float) * 1000.0
            err = act - req
            tr.jitter_ms = float(np.std(err, ddof=1)) if err.size > 1 else 0.0
            tr.max_jitter_ms = float(np.max(np.abs(err)))
            warn = float(self.settings.marker_jitter_warn_ms or 25.0)
            tr.late_markers = int(np.sum(np.abs(err) > warn))
            notes.append(f"jitter from {err.size} polled waits; mean error {np.mean(err):+.2f} ms")
            if tr.late_markers:
                notes.append(f"{tr.late_markers} wait(s) exceeded the "
                             f"{warn:.0f} ms marker jitter warning threshold")
        else:
            notes.append("no timed waits recorded")
        tr.n_markers = len(self._events)
        tr.notes = notes
        return tr

    def _write_notes(self) -> None:
        if self.writer is None:
            return
        parts: list[str] = []
        if self._warnings:
            parts.append("## Warnings\n" + "\n".join(f"- {w}" for w in self._warnings))
        cfg = self.config
        if cfg.dry_run:
            parts.append("## DRY RUN\n\nNo stimuli were presented and no samples were recorded; "
                         "markers and metadata only.")
        if cfg.duration_scale != 1.0 or cfg.limit:
            parts.append(f"## SMOKE TEST\n\nduration_scale={self.scale:g}, limit={cfg.limit}. "
                         "Waits (ITI, rest, instructions) and sustained windows were "
                         "compressed; discrete stimulus presentation was not. Do not "
                         "analyse this as a full run.")
        if self.is_fake:
            parts.append("## SIMULATED SESSION\n\nThe synthetic EEG source stood in for the "
                         "headband. Outcome-gated tests (T101/T108) simulate responses; "
                         "those `response` markers carry `meta.simulated = true`.")
        if self._abort:
            parts.append(
                "## ABORTED\n\n"
                f"- reason: {self._abort_reason or 'unknown'}\n"
                f"- at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- trials completed: {self._n_trials_done}/{self._n_trials_planned}\n"
                f"- samples recorded: {int(sum(a.size for a in self._samples))}\n\n"
                "The trial in progress was finished before closing, so the timeline is "
                "not truncated mid-stimulus. The data before the abort is valid.")
        if self._traceback:
            parts.append("## Traceback\n\n```\n" + self._traceback + "```")
        if parts:
            self.writer.note("\n\n".join(parts))

    # ------------------------------------------------------------ reporting
    def _console(self, msg: str) -> None:
        """Print to the terminal only (no run.log); markup is stripped for plain."""
        try:
            from rich.console import Console
            Console().print(msg)
        except Exception:
            print(_strip_markup(msg))

    def _say(self, msg: str) -> None:
        """Console + run.log.  Everything the runner prints goes through here."""
        self._log_line(_strip_markup(msg))
        if self.config.verbose:
            self._console(msg)

    def _warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.append(msg)
        self._log_line(f"WARNING: {msg}")
        if self.config.verbose:
            self._console(f"[yellow]warning:[/yellow] {msg}")

    def _print_summary(self) -> None:
        s = self._summary or {}
        q = s.get("quality") or {}
        t = s.get("timing") or {}
        verdict = q.get("verdict") or q.get("skipped") or "n/a"
        path = str(self._session_path or "?")
        desk = s.get("desktop_size") or []
        win = s.get("window_size") or []
        rows = [
            ("session", path),
            ("tests", ", ".join(spec.id for spec in self._specs)),
            ("duration", f"{float(s.get('duration_s') or 0):.1f} s"),
            ("samples", f"{int(s.get('n_samples') or 0)} @ {float(s.get('fs_effective') or 0):.1f} Hz effective"),
            ("events", f"{int(s.get('n_events') or 0)} ({int(s.get('n_markers') or 0)} markers, "
                       f"{int(s.get('n_responses') or 0)} responses)"),
            ("trials", f"{s.get('n_trials_done', 0)}/{s.get('n_trials_planned', 0)}"),
            ("quality", f"{verdict} ({q.get('score', '-')}/{q.get('max_score', '-')})"),
            ("mic", self._mic_summary_line()),
            ("advance", (f"{s.get('advance_effective')}"
                         + (f" ({s.get('advance_reason')})" if s.get("advance_reason") else "")
                         + (", cap hit" if s.get("advance_cap_hit") else ""))),
            ("display", (f"fullscreen={s.get('fullscreen')} driver={s.get('visual_sdl_driver')}"
                         f" desktop={desk or '?'} window={win or '?'}"
                         f" cursor_hidden={s.get('cursor_hidden')}")),
            ("screensaver_inhibit", str(s.get("screensaver_inhibit"))),
            ("run.log", str(s.get("run_log"))),
            ("timing", f"mean interval {float(t.get('mean_interval_ms') or 0):.1f} ms, "
                       f"jitter sd {float(t.get('jitter_ms') or 0):.2f} ms, "
                       f"max {float(t.get('max_jitter_ms') or 0):.2f} ms"),
        ]
        # Log the summary even in quiet mode: run.log is the bug-report artifact.
        for k, v in rows:
            self._log_line(f"summary {k}: {v}")
        for spec in self._specs:
            info = (self._plans.get(spec.id) and
                    plan_summary(spec, self._plans[spec.id][1], self._plans[spec.id][2])) or {}
            if info:
                self._log_line(f"summary {spec.id} timeline: shape={info.get('shape')} "
                               f"n={info.get('n_trials')} body={info.get('body_s')}s "
                               f"itv={info.get('itv_s')}s labels={info.get('label_counts')}")
                self._log_line(f"summary {spec.id} keys: {', '.join(info.get('keys') or [])}")
        if int(t.get("late_markers") or 0):
            self._log_line(f"summary JITTER: {t['late_markers']} wait(s) exceeded "
                           f"{self.settings.marker_jitter_warn_ms:.0f} ms")
        if self._abort:
            self._log_line(f"summary ABORTED: {self._abort_reason or 'unknown'}")
        if not self.config.verbose:
            print(f"session -> {path}")
            return
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title=f"NeuroBridge session {self.session_id}", show_header=False,
                          title_style="bold")
            for k, v in rows:
                table.add_row(f"[dim]{k}[/dim]", str(v))
            console.print(table)
            for spec in self._specs:
                info = (self._plans.get(spec.id) and
                        plan_summary(spec, self._plans[spec.id][1], self._plans[spec.id][2])) or {}
                if info:
                    console.print(f"  [dim]{spec.id} timeline[/dim] shape={info.get('shape')} "
                                  f"n={info.get('n_trials')} body={info.get('body_s')}s "
                                  f"itv={info.get('itv_s')}s labels={info.get('label_counts')}")
                    console.print(f"  [dim]{spec.id} keys[/dim] {', '.join(info.get('keys') or [])}")
            if int(t.get("late_markers") or 0):
                console.print(f"[yellow]JITTER WARNING:[/yellow] {t['late_markers']} wait(s) "
                              f"exceeded {self.settings.marker_jitter_warn_ms:.0f} ms "
                              f"(threshold settings.marker_jitter_warn_ms)")
            if s.get("smoke_test"):
                console.print("[yellow]SMOKE TEST:[/yellow] duration_scale="
                              f"{self.scale:g}, limit={self.config.limit} - not a full session")
            if self._abort:
                console.print(f"[red]ABORTED:[/red] {self._abort_reason or 'unknown'}")
            if not q.get("usable", q.get("verdict") in ("N/A", "UNKNOWN")):
                console.print(f"[red]QUALITY:[/red] {verdict} - the pipeline will refuse to "
                              "train on this capture")
        except Exception:
            print(f"session -> {path}")
            for k, v in rows:
                print(f"  {k:12s} {v}")
            if int(t.get("late_markers") or 0):
                print(f"  JITTER WARNING: {t['late_markers']} late waits")
            if self._abort:
                print(f"  ABORTED: {self._abort_reason}")


# ---------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _norm(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))[:40]


def _code_for(spec: TestSpec, label: str) -> int:
    if not label:
        return 0
    try:
        return int(spec.code(_norm(label)))
    except Exception:
        return int(store.MARKER_CODES.get(_norm(label), 0))


def _flicker_hz(label: str) -> float | None:
    digits = "".join(c for c in str(label) if c.isdigit())
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def _format_instruction(text: Any, **kw: Any) -> str:
    """Format a spec's instruction template, tolerating unknown placeholders."""
    if not text:
        return ""
    body = str(text).strip()
    try:
        return body.format(**kw)
    except (KeyError, IndexError, ValueError):
        return body


def _strip_markup(text: str) -> str:
    import re
    return re.sub(r"\[/?[a-zA-Z0-9 _#=.,-]+\]", "", text)


#: Minimal keyword -> synth generator mapping for the runner's offline fallback.
_SYNTH_HINTS: tuple[tuple[str, str], ...] = (
    ("click", "click_train"), ("syllable", "syllable_ba"), ("pseudo_speech", "pseudo_speech"),
    ("reversed", "pseudo_speech_reversed"), ("melod", "melody_familiar"),
    ("music", "melody_familiar"), ("chord", "chord_consonant"), ("disson", "chord_dissonant"),
    ("beat", "beat_120"), ("tempo", "beat_120"), ("pink", "noise_pink"),
    ("brown", "noise_brown"), ("white", "noise_white"), ("ambien", "ambience_calm"),
    ("noise", "noise_pink"), ("speech", "pseudo_speech"), ("tone", "tone_1000"),
)


def _synth_name(text: str) -> str | None:
    from ..stimuli import synth
    raw = str(text).strip().lower()
    if raw in synth.NAMES:
        return raw
    key = raw.replace(" ", "_").replace("-", "_")
    if key in synth.NAMES:
        return key
    for needle, name in _SYNTH_HINTS:
        if needle in raw and name in synth.NAMES:
            return name
    digits = "".join(c for c in raw if c.isdigit())
    if "hz" in raw or "tone" in raw:
        if digits and f"tone_{digits}" in synth.NAMES:
            return f"tone_{digits}"
        return "tone_1000"
    return None


def _fallback_text(label: str, names: Sequence[str]) -> list[str]:
    """Small self-authored text bank so text tests run with no downloads."""
    key = _norm(label)
    cat = " ".join(str(n) for n in names).lower()
    blob = f"{key} {cat}"
    if "digit" in blob or "number" in blob:
        return [str(d) for d in range(10)]
    if "nback" in blob or "letter" in blob:
        return list("BCDFGHJKLM")
    if "pseudoword" in blob or "nonword" in blob:
        return ["florp", "tranik", "meple", "grastin", "cholent"]
    if "low_freq" in blob or "rare" in blob:
        return ["abacus", "quixotic", "peregrine", "ephemeral", "sesquipedalian"]
    if "high_freq" in blob or "common" in blob:
        return ["house", "water", "light", "table", "window"]
    if "garden" in blob:
        return ["The horse raced past the barn fell.",
                "The old man the boats.",
                "While Anna dressed the baby played."]
    if "incongruent" in blob:
        return ["The cat barked at the postman.",
                "She buttered the guitar before breakfast.",
                "The ocean sang a lullaby to the desert."]
    if "congruent" in blob:
        return ["The dog barked at the postman.",
                "She buttered the toast before breakfast.",
                "The ocean roared under the grey sky."]
    if "stroop" in blob:
        return ["RED", "BLUE", "GREEN", "YELLOW"]
    if "passage" in blob or "prose" in blob or "reading" in blob:
        return ["The kitchen window was open, and the smell of rain moved "
                "through the house while the kettle began to sing. Nobody "
                "hurried; the afternoon belonged to no one in particular."]
    if "shuffled" in blob:
        return ["window kitchen the open was rain and of smell the"]
    if "positive" in blob:
        return ["The letter said the garden had survived the winter and would bloom again."]
    if "negative" in blob:
        return ["The letter said the garden had frozen through and nothing would return."]
    if "neutral" in blob:
        return ["The letter described the garden and the tools stored in the shed."]
    return [f"{label.replace('_', ' ').capitalize()}: sample item for offline testing."]


# ---------------------------------------------------------------- entry points
def run_test(config: RunConfig) -> Path:
    """Run one test end to end and return the session directory."""
    runner = Runner(config)
    try:
        runner.setup()          # may refuse to start (no stimuli) before anything opens
        return runner.run()
    finally:
        runner.teardown()


def run_sequence(spec_ids: Sequence[str | TestSpec], **kw: Any) -> Path:
    """Run several tests in order in ONE session over ONE BLE connection.

    A single connection matters for two reasons: reconnecting takes ~10 s and
    often needs the headband reseated, and staying connected keeps electrode
    drift continuous so it can be modelled across tests.  Each test still gets
    its own ``test_start``/``test_end`` markers and its own stimulus plan; the
    planned total is checked against the ~45 minute budget and warned about.
    ``kw`` is forwarded to :class:`RunConfig` (do not pass ``spec``).
    """
    specs = [load_one(s) if isinstance(s, str) else s for s in spec_ids]
    if not specs:
        raise ValueError("run_sequence needs at least one spec")
    config = RunConfig(spec=specs[0], sequence=specs, **kw)
    total_min = sum(float(s.duration_min or 0.0) for s in specs)
    runner = Runner(config)
    if total_min > SESSION_BUDGET_MIN:
        runner._warn(f"planned sequence duration {total_min:.0f} min exceeds the "
                     f"~{SESSION_BUDGET_MIN:.0f} min session budget")
    if config.verbose:
        runner._say(f"[bold]sequence[/bold] {', '.join(s.id for s in specs)} "
                    f"- planned {total_min:.0f} min")
    try:
        runner.setup()
        return runner.run()
    finally:
        runner.teardown()
