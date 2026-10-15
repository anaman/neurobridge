"""Microphone capture synchronized with EEG and markers.

Why this exists
---------------
Almost every interesting thing about reading is *behavioural*, and a dry frontal
EEG channel on its own cannot see it:

* when you actually started speaking, relative to the cue (a real reaction time,
  not a self-report);
* how long you spoke, how fast, and how many times you paused or restarted
  (fluency and effort, which is exactly the "how do I process reading" question);
* whether a "reading" trial was silent because you were reading or because you
  skipped it.

Adding the microphone turns `reported_label` from a guess into an observation,
and it exposes a confound honestly: reading aloud moves the jaw, and jaw muscle
activity is *large* at Fpz. With audio you can check whether an EEG difference
between aloud and silent reading is cortical or muscular, instead of assuming.

Two rules this module enforces
------------------------------
1. **Opt-in, local, logged.** Nothing starts recording unless a caller passes an
   explicit consent flag; the consent and the device name are written into the
   session metadata; audio is written to the session directory and nowhere else.
2. **Voice measures are behaviour, not brain features.** `trial_voice_metrics`
   output must never enter the EEG feature matrix. A classifier trained to
   predict "spoke" from audio-correlated features has learned the microphone,
   not the cortex. Keep voice metrics in the behavioural/report layer, or use
   them as the *target* of a separate cross-modality question ("does frontal
   theta predict speech onset latency?").
"""
from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Speech analysis defaults. 16 kHz mono is plenty for speech timing and keeps a
#: 10 minute recording at ~19 MB (16-bit) instead of ~100 MB.
DEFAULT_FS = 16_000


class MicUnavailable(RuntimeError):
    """Raised when no input device exists or sounddevice cannot open one."""


# ---------------------------------------------------------------- discovery
def list_input_devices() -> list[dict[str, Any]]:
    """Input-capable devices as plain dicts (empty list if sounddevice is absent)."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    try:
        default_in = None
        try:
            default_in = sd.default.device[0]
        except Exception:
            pass
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) <= 0:
                continue
            out.append({
                "index": i,
                "name": d.get("name", "?"),
                "channels": int(d["max_input_channels"]),
                "default_samplerate": float(d.get("default_samplerate") or 0),
                "default": (i == default_in),
                "input_latency_ms": round(1000.0 * float(d.get("default_low_input_latency") or 0), 2),
            })
    except Exception:
        return []
    return out


def default_input_name() -> str:
    for d in list_input_devices():
        if d["default"]:
            return d["name"]
    return ""


def available() -> bool:
    try:
        import sounddevice as sd
        return bool(sd.query_devices(kind="input"))
    except Exception:
        return False


def probe(seconds: float = 1.5, fs: int = DEFAULT_FS,
          consent: bool = True, keep: bool = False) -> dict[str, Any]:
    """Record a moment of audio and report the measured level.

    Used by ``nb doctor`` to answer the only question that matters before a
    session: is the microphone actually picking anything up? A silent room and a
    muted microphone look identical in the metadata and completely different
    here.

    ``keep=False`` (the default) deletes the WAV after measuring it: a
    diagnostic should not silently leave a recording of the room on disk. Pass
    ``keep=True`` when you want the sample for inspection.
    """
    rec = MicRecorder(MicConfig(fs=fs, consent=bool(consent)))
    out_dir = None
    try:
        from .config import CACHE_DIR
        out_dir = CACHE_DIR / "mic-probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec.out_dir = out_dir
        rec.start()
        time.sleep(max(0.2, seconds))
        path = rec.stop()
        if path is None:
            return {"ok": False, "error": "no audio was captured (silent stream?)",
                    "device": rec.info.get("device")}
        x, fs_out = read_wav(path)
    except MicUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    dbfs = 20 * np.log10(max(rms, 1e-9))
    # digital full scale is 0 dBFS; anything above it is a device artefact, not
    # a level, so clamp the reported figure and say so
    capped = bool(dbfs > 6.0)
    result = {
        "ok": True, "device": rec.info.get("device"), "fs": fs_out,
        "seconds": x.size / fs_out, "rms_dbfs": round(min(float(dbfs), 6.0), 1),
        "peak": round(peak, 3), "clipped_pct": round(100 * float(np.mean(np.abs(x) >= 0.995)), 3),
        "nonfinite_pct": rec.info.get("nonfinite_pct", 0.0),
        "level_capped": capped,
        "verdict": ("signal present" if -50 < dbfs <= 6 else
                    ("level above digital full scale - input glitch, check the device"
                     if capped else "very quiet - check the microphone is not muted")),
        "wav": str(path) if keep else None,
        "kept": bool(keep),
    }
    if not keep:
        rec.drop_wav(path)
    return result


# ---------------------------------------------------------------- config
@dataclass
class MicConfig:
    """Recording parameters. ``consent`` must be True or ``start()`` refuses."""

    device: int | str | None = None
    fs: int = DEFAULT_FS
    channels: int = 1
    blocksize: int = 512
    gain: float = 1.0
    consent: bool = False
    save_wav: bool = True
    filename: str = "audio.wav"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- recorder
class MicRecorder:
    """Blocking-free microphone recorder with monotonic block timestamps.

    The audio callback is the *only* place a timestamp is taken, using the same
    ``time.perf_counter()`` clock as the EEG packet arrivals and the stimulus
    markers. That shared clock is what makes onset latencies meaningful; a
    wall-clock or per-block counter would not be comparable.
    """

    def __init__(self, config: MicConfig | None = None, out_dir: Path | None = None) -> None:
        self.cfg = config or MicConfig()
        self.out_dir = Path(out_dir) if out_dir else None
        self._blocks: list[np.ndarray] = []
        self._block_times: list[float] = []
        self._stream = None
        self._t_start: float | None = None
        self._t_stop: float | None = None
        self._status_events: list[str] = []
        self._n_frames = 0
        self.info: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def t_start(self) -> float | None:
        """Monotonic time of the first captured sample block."""
        return self._t_start

    def start(self) -> float:
        if not self.cfg.consent:
            raise MicUnavailable(
                "microphone recording requires explicit consent "
                "(MicConfig(consent=True), CLI: --mic); nothing was recorded")
        try:
            import sounddevice as sd
        except Exception as exc:                     # pragma: no cover
            raise MicUnavailable(f"sounddevice unavailable: {exc!r}") from exc
        dev = self.cfg.device
        try:
            info = sd.query_devices(dev, kind="input")
        except Exception as exc:
            raise MicUnavailable(f"no usable input device ({dev!r}): {exc}") from exc

        def callback(indata, frames, time_info, status) -> None:
            now = time.perf_counter()
            if status:
                self._status_events.append(str(status))
            block = np.asarray(indata[:, 0], dtype=np.float32)
            if self._t_start is None:
                self._t_start = now
            self._blocks.append(block)
            self._block_times.append(now)
            self._n_frames += block.size

        try:
            self._stream = sd.InputStream(
                device=dev, channels=int(self.cfg.channels), samplerate=int(self.cfg.fs),
                blocksize=int(self.cfg.blocksize), dtype="float32", callback=callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise MicUnavailable(f"could not open the input stream: {exc}") from exc

        self.info = {
            "device": info.get("name", str(dev)),
            "device_index": info.get("index"),
            "fs": int(self.cfg.fs),
            "channels": int(self.cfg.channels),
            "blocksize": int(self.cfg.blocksize),
            "input_latency_ms": round(1000.0 * float(
                info.get("default_low_input_latency") or 0.0), 2),
            "gain": float(self.cfg.gain),
            "consent": True,
            "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return float(self._t_start or time.perf_counter())

    def stop(self) -> Path | None:
        """Stop, optionally write ``audio.wav``, return its path (or None)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._t_stop = time.perf_counter()
        x = self.signal()
        # RMS in float64: a float32 accumulator overflows to +inf on an input
        # glitch, which then reports an impossible level (seen in practice as
        # "rms_dbfs": Inf for a genuinely -3 dBFS capture)
        if x.size:
            x64 = x.astype(np.float64)          # float64: a float32 accumulator
            rms = float(np.sqrt(np.mean(x64 * x64)))   # overflowed on device glitches
            if not np.isfinite(rms):
                rms_dbfs = None
            else:
                # digital full scale is 0 dBFS; above that is an artefact, not a level
                rms_dbfs = round(min(20.0 * np.log10(max(rms, 1e-9)), 6.0), 1)
        else:
            rms_dbfs = None
        self.info.update({
            "n_frames": int(x.size),
            "duration_s": round(float(x.size) / self.cfg.fs, 3) if x.size else 0.0,
            "n_blocks": len(self._blocks),
            "status_events": self._status_events[:20],
            "t_start_mono": self._t_start,
            "t_stop_mono": self._t_stop,
            "rms_dbfs": rms_dbfs,
            "clipped_pct": (round(100.0 * float(np.mean(np.abs(x.astype(np.float64)) >= 0.995)), 3)
                            if x.size else None),
        })
        if not self.cfg.save_wav or x.size == 0 or self.out_dir is None:
            return None
        path = self.out_dir / self.cfg.filename
        write_wav(path, x, self.cfg.fs)
        (self.out_dir / "audio.json").write_text(json.dumps(self.info, indent=2, default=str))
        return path

    # -- data -------------------------------------------------------
    def signal(self) -> np.ndarray:
        """The captured audio, with non-finite samples replaced by silence.

        A device glitch can deliver NaN/Inf (seen in practice as an impossible
        ``rms_dbfs`` of +125 dBFS while the written WAV was merely clipped at
        about -3 dBFS). Those values must not reach a metric: they are counted,
        reported in ``info['nonfinite_samples']``, and zeroed in the returned
        copy so downstream analysis cannot be poisoned by them.
        """
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate(self._blocks)
        bad = ~np.isfinite(x)
        n_bad = int(bad.sum())
        if n_bad:
            self.info["nonfinite_samples"] = n_bad
            self.info["nonfinite_pct"] = round(100.0 * n_bad / max(1, x.size), 3)
            x = np.where(bad, 0.0, x)
        return x

    def frame_times(self) -> np.ndarray:
        """One timestamp per block, i.e. per ``blocksize`` samples."""
        return np.asarray(self._block_times, dtype=np.float64)

    def sample_times(self) -> np.ndarray:
        """Per-sample monotonic times, interpolated between block timestamps.

        Audio blocks arrive at a steady rate, so interpolating between block
        timestamps is accurate to well under a millisecond -- far better than the
        BLE path, where packets arrive in bursts. Each block timestamp is taken
        when its callback fires, i.e. at the arrival of that block's last sample.
        """
        x = self.signal()
        if x.size == 0 or not self._block_times:
            return np.zeros(0, dtype=np.float64)
        if len(self._block_times) == 1:
            base = float(self._t_start or self._block_times[0])
            return base + np.arange(x.size) / float(self.cfg.fs)
        edges = np.asarray(self._block_times, dtype=np.float64)
        return _interp_sample_times(edges, int(self.cfg.blocksize), x.size)

    def drop_wav(self, path: Path | None = None) -> None:
        p = path or (self.out_dir / self.cfg.filename if self.out_dir else None)
        if p and p.exists():
            p.unlink()


def _interp_sample_times(block_times: np.ndarray, blocksize: int, n: int) -> np.ndarray:
    idx = np.arange(n)
    anchors = np.minimum(np.arange(len(block_times)) * blocksize + blocksize - 1, n - 1)
    return np.interp(idx, anchors, block_times)


# ---------------------------------------------------------------- wav i/o
def write_wav(path: Path, x: np.ndarray, fs: int = DEFAULT_FS) -> Path:
    """Write 16-bit mono PCM. Plain stdlib so the file needs no library to read."""
    x = np.clip(np.asarray(x, dtype=np.float32).ravel(), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono PCM wav back to float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    return x, int(fs)


# ---------------------------------------------------------------- speech analysis
def frame_energy(x: np.ndarray, fs: int, frame_ms: float = 20.0,
                 hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-wise RMS and zero-crossing rate. Returns (rms, zcr, frame_times_s)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    n = max(1, int(fs * frame_ms / 1000.0))
    hop = max(1, int(fs * hop_ms / 1000.0))
    if x.size < n:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    n_frames = 1 + (x.size - n) // hop
    rms = np.empty(n_frames, dtype=np.float32)
    zcr = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n]
        rms[i] = float(np.sqrt(np.mean(seg ** 2)))
        zcr[i] = float(np.mean(np.abs(np.diff(np.signbit(seg).astype(np.int8)))))
    times = (np.arange(n_frames) * hop + n / 2) / fs
    return rms, zcr, times


def speech_mask(x: np.ndarray, fs: int, frame_ms: float = 20.0, hop_ms: float = 10.0,
                min_speech_ms: float = 120.0, hangover_ms: float = 150.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy+ZCR voice-activity detection, returning (mask, rms, frame_times).

    A deliberately simple detector: an adaptive noise floor (10th percentile of
    frame RMS) with a 6 dB-ish margin, plus a zero-crossing sanity check for
    fricatives. It is good enough to answer "did speech start, and when", which
    is the measurement this module exists for. It is *not* a phonetic
    aligner -- for word-level timing, run a real ASR/aligner over the wav and
    join on the timestamps.
    """
    rms, zcr, times = frame_energy(x, fs, frame_ms, hop_ms)
    if rms.size == 0:
        return np.zeros(0, dtype=bool), rms, times
    floor = float(np.percentile(rms, 10))
    peak = float(np.percentile(rms, 99))
    if peak <= floor * 1.5:
        return np.zeros_like(rms, dtype=bool), rms, times
    thr = max(floor * 3.0, floor + 0.08 * (peak - floor))
    mask = rms > thr
    # fill short gaps (stop closures) and drop short blips
    hang = max(1, int(hangover_ms / hop_ms))
    minrun = max(1, int(min_speech_ms / hop_ms))
    mask = _bridge(mask, hang)
    mask = _drop_short(mask, minrun)
    return mask, rms, times


def _bridge(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _drop_short(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def syllable_rate(mask: np.ndarray, rms: np.ndarray, hop_ms: float = 10.0,
                  peak_hz_range: tuple[float, float] = (2.0, 8.0)) -> float | None:
    """Speech rate proxy: peaks per second in the voiced amplitude envelope.

    Syllable nuclei produce 2-8 amplitude peaks per second in normal speech, so
    peak-picking on the smoothed envelope is a standard cheap proxy. It is a
    proxy: report it as `syllables/s (envelope peaks)`, never as a word rate.
    """
    if mask.size == 0 or not mask.any():
        return None
    env = rms.copy()
    voiced = mask
    if voiced.sum() < 5:
        return None
    # smooth
    k = np.ones(5) / 5.0
    env = np.convolve(env, k, mode="same")
    peaks = 0
    for i in range(1, env.size - 1):
        if not voiced[i]:
            continue
        if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] > 1.15 * np.median(env[voiced]):
            peaks += 1
    dur_s = float(voiced.sum()) * hop_ms / 1000.0
    if dur_s <= 0.2:
        return None
    rate = peaks / dur_s
    lo, hi = peak_hz_range
    return float(min(max(rate, lo), hi))


@dataclass
class VoiceMetrics:
    """Behavioural measurements for one trial. Never EEG features."""

    trial: int | None = None
    onset_code: int | None = None
    cue_t_mono: float | None = None
    speech_onset_t_mono: float | None = None
    onset_latency_ms: float | None = None
    speech_duration_ms: float | None = None
    speech_offset_t_mono: float | None = None
    voiced_fraction: float | None = None
    syllables_per_s: float | None = None
    n_pauses: int | None = None
    pause_total_ms: float | None = None
    rms_dbfs: float | None = None
    peak: float | None = None
    clipped_pct: float | None = None
    spoke: bool | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_window(x: np.ndarray, fs: int, t_audio0: float, cue_t: float,
                   pre_s: float = 0.5, post_s: float = 6.0,
                   trial: int | None = None,
                   onset_code: int | None = None) -> VoiceMetrics:
    """Measure one trial's speech from a recording aligned to marker times.

    ``t_audio0`` is the monotonic time of the first recorded sample; the window
    is ``[cue_t - pre_s, cue_t + post_s]`` so a small timing error cannot clip
    the onset away.
    """
    m = VoiceMetrics(trial=trial, onset_code=onset_code, cue_t_mono=float(cue_t))
    i0 = int(max(0.0, (cue_t - pre_s - t_audio0)) * fs)
    i1 = int(max(0.0, (cue_t + post_s - t_audio0)) * fs)
    i0 = min(i0, x.size)
    i1 = min(i1, x.size)
    seg = x[i0:i1]
    if seg.size < fs // 10:
        m.note = "no audio window for this trial (recording shorter than the trial)"
        return m

    m.rms_dbfs = round(float(20 * np.log10(max(np.sqrt(np.mean(seg ** 2)), 1e-9))), 1)
    m.peak = round(float(np.max(np.abs(seg))), 4)
    m.clipped_pct = round(100 * float(np.mean(np.abs(seg) >= 0.995)), 3)

    mask, rms, times = speech_mask(seg, fs)
    if mask.size == 0 or not mask.any():
        m.spoke = False
        m.voiced_fraction = 0.0
        m.note = "no speech detected in the window"
        return m

    m.spoke = True
    m.voiced_fraction = round(float(mask.mean()), 3)
    # first voiced frame = speech onset
    first = int(np.argmax(mask))
    last = int(mask.size - 1 - np.argmax(mask[::-1]))
    onset_audio = t_audio0 + i0 / fs + float(times[first])
    offset_audio = t_audio0 + i0 / fs + float(times[last])
    m.speech_onset_t_mono = round(onset_audio, 6)
    m.speech_offset_t_mono = round(offset_audio, 6)
    m.onset_latency_ms = round(1000.0 * (onset_audio - cue_t), 1)
    m.speech_duration_ms = round(1000.0 * (offset_audio - onset_audio), 1)
    m.syllables_per_s = (round(syllable_rate(mask, rms, hop_ms=10.0), 2)
                         if syllable_rate(mask, rms, hop_ms=10.0) else None)
    # pauses: unvoiced gaps inside the speech span
    span = mask[first:last + 1]
    n_pauses, total = 0, 0
    i = 0
    while i < span.size:
        if not span[i]:
            j = i
            while j < span.size and not span[j]:
                j += 1
            if (j - i) * 10.0 >= 200.0:              # >=200 ms counts as a pause
                n_pauses += 1
                total += (j - i) * 10.0
            i = j
        else:
            i += 1
    m.n_pauses = n_pauses
    m.pause_total_ms = round(total, 1) if n_pauses else 0.0
    return m


def analyze_session(session, event_name: str = "stim_onset",
                    post_s: float = 6.0) -> dict[str, Any]:
    """Per-trial voice metrics for a whole session (needs ``audio.wav``).

    Returns a JSON-serialisable dict with a per-trial list and summary
    statistics, plus the caveats that belong next to any behavioural number.
    """
    audio = getattr(session, "audio_path", None)
    if audio is None or not Path(audio).exists():
        return {"available": False,
                "reason": "this session has no audio.wav (record with --mic)"}
    x, fs = read_wav(Path(audio))
    meta = {}
    am = Path(audio).with_name("audio.json")
    if am.exists():
        try:
            meta = json.loads(am.read_text())
        except Exception:
            meta = {}
    t0 = meta.get("t_start_mono")
    if t0 is None:
        return {"available": False,
                "reason": "audio.json has no t_start_mono, so the recording cannot "
                          "be aligned to the EEG clock"}
    rows = []
    onsets = [e for e in session.events if e.name == event_name]
    if not onsets:
        # A microphone-only recording (nb voice --record) has no markers. That is
        # the common case for "how do I process reading?" - someone reading aloud
        # with no headband - so analyse the whole file as a single window rather
        # than returning nothing.
        whole = analyze_window(x, fs, float(t0), float(t0), pre_s=0.0,
                               post_s=float(x.size) / fs + 0.1)
        whole.note = (whole.note + "; no markers in this session: the whole "
                      "recording was analysed as one window").strip("; ")
        rows.append(whole.as_dict())
    else:
        for e in onsets:
            rows.append(analyze_window(x, fs, float(t0), e.t_mono, post_s=post_s,
                                       trial=e.trial, onset_code=e.code).as_dict())
    spoke = [r for r in rows if r.get("spoke")]
    lat = [r["onset_latency_ms"] for r in spoke if r.get("onset_latency_ms") is not None]
    dur = [r["speech_duration_ms"] for r in spoke if r.get("speech_duration_ms") is not None]
    rate = [r["syllables_per_s"] for r in spoke if r.get("syllables_per_s")]
    summary = {
        "n_trials": len(rows),
        "n_spoken": len(spoke),
        "silent_fraction": round(1 - len(spoke) / len(rows), 3) if rows else None,
        "onset_latency_ms_median": round(float(np.median(lat)), 1) if lat else None,
        "onset_latency_ms_iqr": (round(float(np.percentile(lat, 75) - np.percentile(lat, 25)), 1)
                                 if len(lat) >= 4 else None),
        "speech_duration_ms_median": round(float(np.median(dur)), 1) if dur else None,
        "syllables_per_s_median": round(float(np.median(rate)), 2) if rate else None,
        "n_restarts_or_pauses": int(sum(r.get("n_pauses") or 0 for r in spoke)),
    }
    return {
        "available": True, "fs": fs, "audio": str(audio),
        "timing_quality": "coarse" if meta.get("input_latency_ms") is None else "coarse",
        "input_latency_ms": meta.get("input_latency_ms"),
        "trials": rows, "summary": summary,
        "caveats": [
            "voice metrics are behaviour, not EEG features - never feed them to the "
            "brain decoder",
            "onset latency is cue-to-voiced-audio; it includes microphone and "
            "buffer latency and is NOT a cortical latency",
            "syllables/s is an envelope-peak proxy, not a word rate",
            "reading aloud moves the jaw, and jaw EMG is large at Fpz: use these "
            "metrics to check whether an EEG difference is muscular",
        ],
    }


# ---------------------------------------------------------------- round trip
def measure_roundtrip(play_tone, recorder: MicRecorder, freq: float = 1000.0,
                      dur_s: float = 0.15, search_s: float = 2.0) -> dict[str, Any]:
    """Measure output+input latency by playing a tone and finding it on the mic.

    ``play_tone(freq, dur_s)`` must return the monotonic time it started. The
    difference between that timestamp and the tone's arrival in the recording is
    the total acoustic round-trip latency of this machine — the number that makes
    a cue-to-speech onset latency interpretable, and the same trick extends the
    T004/T005 latency calibration to the voice path.
    """
    if not recorder.recording:
        recorder.start()
    t_play = float(play_tone(freq, dur_s))
    time.sleep(min(search_s, dur_s + 0.4))
    x = recorder.signal()
    times = recorder.sample_times()
    if x.size == 0:
        return {"ok": False, "reason": "no audio recorded"}
    fs = recorder.cfg.fs
    # band-limited energy around the tone, then find the first strong frame
    n = int(0.01 * fs)
    seg = np.abs(x)
    if seg.size < n:
        return {"ok": False, "reason": "recording too short"}
    env = np.convolve(seg, np.ones(n) / n, mode="same")
    base = float(np.percentile(env, 50))
    thr = max(base * 4.0, float(np.max(env)) * 0.35)
    hits = np.where(env > thr)[0]
    if hits.size == 0:
        return {"ok": False, "reason": "the played tone was not found in the recording; "
                                       "check the speakers, the microphone, and mute switches"}
    i = int(hits[0])
    t_arrive = float(times[min(i, times.size - 1)])
    return {"ok": True, "play_t_mono": t_play, "arrival_t_mono": round(t_arrive, 6),
            "roundtrip_ms": round(1000.0 * (t_arrive - t_play), 1),
            "note": "output + acoustic + input latency; subtract half of it from a "
                    "cue-to-speech latency if you need a first-order correction"}
