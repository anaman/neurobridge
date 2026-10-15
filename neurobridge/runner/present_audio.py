"""Audio presentation for stimulus-locked tests.

Timing model
------------
The EEG timeline is only meaningful because *every* timestamp in this process
comes from ``time.perf_counter()`` (see :func:`neurobridge.device.base.clock`).
For audio there is an unavoidable gap between "I asked the OS to play" and
"sound left the speaker": the output buffer, the ALSA/PulseAudio mixer and the
amplifier.  We cannot observe the true acoustic onset from software, so:

* :meth:`AudioPresenter.play` returns the monotonic timestamp taken *as close
  to the start command as possible* -- immediately after ``sd.play()`` for the
  PortAudio backend, immediately after ``Popen`` for ``aplay``.
* the *nominal* device latency is exposed separately as
  :attr:`AudioPresenter.latency_ms`, and the runner records it in the marker
  metadata and can fold it into ``settings.output_latency_offset_s``.  Keeping
  the raw measurement and the correction apart means a session can always be
  re-aligned after the fact instead of baking in a guess.

Backend ladder
--------------
``sounddevice`` -> ``aplay`` -> ``null``, resolved once in ``__init__`` and
never raising.  A machine with no sound card, no PortAudio and no ``aplay``
still runs a full protocol: the ``null`` backend is a metronome that returns
correct onset timestamps, so the marker/EEG path is exercised end to end.

Pre-loading
-----------
Decoding a WAV/FLAC/MP3 (and, for ``aplay``, rendering a temp WAV) is done the
first time a file is requested and cached in a tiny LRU.  A 500 ms tone in a
trial loop must not pay file-read cost inside the inter-stimulus interval,
where every millisecond of CPU time is jitter.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..device.base import clock

#: Per-modality default presentation length when a spec does not give one.
DEFAULT_TONE_FREQ_HZ = 1000.0

_BACKEND_LADDER: dict[str, tuple[str, ...]] = {
    "auto": ("sounddevice", "aplay", "null"),
    "sounddevice": ("sounddevice", "aplay", "null"),
    "portaudio": ("sounddevice", "aplay", "null"),
    "aplay": ("aplay", "null"),
    "alsa": ("aplay", "null"),
    "null": ("null",),
    "none": ("null",),
    "off": ("null",),
}


class AudioPresenter:
    """Play WAV/FLAC/MP3 files, float arrays and tones, with safe fallbacks."""

    def __init__(self, backend: str = "auto", fs_out: int = 44100, gain: float = 1.0) -> None:
        self.fs_out = int(fs_out)
        self.gain = float(gain)
        self._backend = "null"
        self._sd: Any = None
        self._sf: Any = None
        self._latency_ms = 0.0
        self._warnings: list[str] = []
        self._procs: list[subprocess.Popen] = []
        # LRU caches: clip key -> (samples|None, fs, playable_wav_path|None)
        self._clips: "OrderedDict[str, tuple[np.ndarray | None, int, str | None]]" = OrderedDict()
        self._clip_max = 16
        self._tone_cache: "OrderedDict[tuple[float, float, int], np.ndarray]" = OrderedDict()
        self._tone_max = 32
        self._tmpdir: Path | None = None
        self._nfailed = 0
        self._resolve(backend)

    # ------------------------------------------------------------ resolution
    def _resolve(self, requested: str) -> None:
        order = _BACKEND_LADDER.get((requested or "auto").strip().lower(), _BACKEND_LADDER["auto"])
        for name in order:
            if name == "sounddevice" and self._try_sounddevice():
                self._backend = "sounddevice"
                return
            if name == "aplay" and self._try_aplay():
                self._backend = "aplay"
                return
            if name == "null":
                self._backend = "null"
                return
        self._backend = "null"

    def _try_sounddevice(self) -> bool:
        try:
            import sounddevice as sd
        except Exception as exc:                       # pragma: no cover - env dependent
            self._warn(f"sounddevice unavailable ({exc.__class__.__name__}: {exc})")
            return False
        try:
            dev = sd.query_devices(kind="output")
        except Exception as exc:                       # no PortAudio device / no server
            self._warn(f"sounddevice has no output device ({exc.__class__.__name__})")
            return False
        low = dev.get("default_low_output_latency") if isinstance(dev, dict) else None
        high = dev.get("default_high_output_latency") if isinstance(dev, dict) else None
        # PortAudio reports the *hardware* latency of the default output; that is
        # the number an offline realignment wants, so we surface it verbatim.
        self._latency_ms = 1000.0 * float(low if low else (high or 0.0))
        self._sd = sd
        return True

    def _try_aplay(self) -> bool:
        exe = shutil.which("aplay")
        if not exe:
            self._warn("aplay not on PATH")
            return False
        # "aplay exists" is not "there is a sound card": on a headless box aplay
        # is installed but `aplay -l` reports no cards.  Probe once so we degrade
        # to the null backend instead of spawning a process per stimulus.
        try:
            proc = subprocess.run([exe, "-l"], capture_output=True, text=True, timeout=3.0)
        except Exception:
            proc = None
        blob = ((proc.stdout or "") + (proc.stderr or "")).lower() if proc is not None else ""
        if proc is not None and ("no soundcards" in blob or "cannot open" in blob):
            self._warn("aplay reports no sound card")
            return False
        if proc is not None and proc.returncode != 0 and not blob.strip():
            self._warn("aplay -l failed")
            return False
        self._aplay_exe = exe
        self._latency_ms = 0.0          # ALSA gives us no honest number here
        return True

    def _warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.append(msg)

    # ------------------------------------------------------------ properties
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def latency_ms(self) -> float:
        """Nominal output latency for the chosen backend (0.0 when unknown)."""
        return float(self._latency_ms)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def is_playing(self) -> bool:
        self._prune_procs()
        if self._procs:
            return True
        if self._backend == "sounddevice" and self._sd is not None:
            try:
                return bool(self._sd.get_stream().active)
            except Exception:
                return False
        return False

    # ------------------------------------------------------------ playback
    def play(self, path: str | Path, *, blocking: bool = False) -> float:
        """Start a file. Returns the monotonic onset estimate. Never raises."""
        try:
            data, fs, wav = self._clip(path)
        except Exception as exc:
            self._warn(f"cannot load audio {path}: {exc.__class__.__name__}: {exc}")
            return clock()
        if self._backend == "sounddevice" and data is not None:
            t = self._sd_play(data, fs)
        elif self._backend == "aplay" and wav:
            t = self._aplay_play(wav)
        else:
            t = clock()                      # null backend: a precise, silent metronome
        if blocking:
            self._wait_done()
        return t

    def play_array(self, x: np.ndarray, fs: int, *, blocking: bool = False) -> float:
        """Play an in-memory buffer. Returns the monotonic onset estimate."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return clock()
        if self.gain != 1.0:
            arr = np.clip(arr * self.gain, -1.0, 1.0).astype(np.float32)
        if self._backend == "sounddevice":
            t = self._sd_play(arr, float(fs))
        elif self._backend == "aplay":
            key = self._array_key(arr, fs)
            wav = self._ensure_wav(arr, fs, key)
            t = self._aplay_play(wav) if wav else clock()
        else:
            t = clock()
        if blocking:
            self._wait_done()
        return t

    def play_tone(self, freq: float, dur_s: float, *, blocking: bool = False) -> float:
        """Synthesize and play a ramped sine tone (cached by freq/duration)."""
        fs = int(self.fs_out)
        key = (round(float(freq), 3), round(float(dur_s), 4), fs)
        x = self._tone_cache.get(key)
        if x is None:
            x = self._make_tone(freq, dur_s, fs)
            self._tone_cache[key] = x
            while len(self._tone_cache) > self._tone_max:
                self._tone_cache.popitem(last=False)
        else:
            self._tone_cache.move_to_end(key)
        return self.play_array(x, fs, blocking=blocking)

    def stop(self) -> None:
        """Stop anything currently playing. Safe to call at any time."""
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                pass
        for proc in list(self._procs):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
        self._procs = []

    def close(self) -> None:
        self.stop()
        self._clips.clear()
        self._tone_cache.clear()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    # ------------------------------------------------------------ internals
    def _sd_play(self, data: np.ndarray, fs: float) -> float:
        try:
            # A generous output buffer avoids ALSA underruns when the host is
            # busy polling BLE at 4 ms slices; it does not change the reported
            # onset (that is latched immediately after the call either way).
            self._sd.play(data, samplerate=float(fs), blocking=False, latency="high")
        except TypeError:
            try:
                self._sd.play(data, samplerate=float(fs), blocking=False)
            except Exception as exc:
                self._warn(f"sounddevice playback failed ({exc.__class__.__name__}: {exc})")
        except Exception as exc:
            self._warn(f"sounddevice playback failed ({exc.__class__.__name__}: {exc})")
        # Timestamp immediately after the start command: this is the closest
        # observable proxy for the true acoustic onset.
        return clock()

    def _aplay_play(self, wav: str | Path) -> float:
        try:
            proc = subprocess.Popen([getattr(self, "_aplay_exe", "aplay"), "-q", str(wav)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._procs.append(proc)
        except Exception as exc:
            self._warn(f"aplay playback failed ({exc.__class__.__name__}: {exc})")
        return clock()

    def _wait_done(self) -> None:
        """Block until playback finishes using the same poll-friendly sleep as
        the runner (2 ms slices) rather than ``sd.wait()``/``proc.wait()`` so a
        long clip cannot hide an abort for seconds."""
        deadline = clock() + 600.0
        while clock() < deadline:
            if self._backend == "sounddevice":
                try:
                    if not self._sd.get_stream().active:
                        return
                except Exception:
                    return
            else:
                self._prune_procs()
                if not self._procs:
                    return
            time.sleep(0.002)

    def _prune_procs(self) -> None:
        self._procs = [p for p in self._procs if p.poll() is None]

    def _tone_lru(self, key: str) -> tuple[np.ndarray | None, int, str | None] | None:
        hit = self._clips.get(key)
        if hit is not None:
            self._clips.move_to_end(key)
        return hit

    def _clip(self, path: str | Path) -> tuple[np.ndarray | None, int, str | None]:
        p = Path(path)
        key = f"{p.resolve()}|{p.stat().st_mtime_ns if p.exists() else 0}"
        hit = self._tone_lru(key)
        if hit is not None:
            return hit
        data, fs, wav = self._read_audio(p)
        self._clips[key] = (data, fs, wav)
        while len(self._clips) > self._clip_max:
            self._clips.popitem(last=False)
        return data, fs, wav

    def _read_audio(self, p: Path) -> tuple[np.ndarray | None, int, str | None]:
        """Decode a file once; return (samples|None, fs, aplay-playable path)."""
        data: np.ndarray | None = None
        fs = int(self.fs_out)
        wav: str | None = None
        try:
            import soundfile as sf
            self._sf = sf
        except Exception as exc:
            self._warn(f"soundfile unavailable ({exc.__class__.__name__})")
            sf = None
        if sf is not None:
            try:
                data, fs = sf.read(str(p), dtype="float32", always_2d=False)
                data = np.asarray(data, dtype=np.float32)
                fs = int(fs)
            except Exception as exc:
                self._warn(f"soundfile cannot read {p.name} ({exc.__class__.__name__})")
                data = None
        if data is None:
            # MP3 and exotic codecs: transcode once to a temp WAV when ffmpeg is
            # available.  Without ffmpeg we hand the original path to aplay and
            # let it fail loudly in the warning list rather than silently.
            data, fs = self._ffmpeg_decode(p)
        if data is not None:
            if self.gain != 1.0:
                data = np.clip(data * self.gain, -1.0, 1.0).astype(np.float32)
            wav = self._ensure_wav(data, fs, self._file_key(p))
        elif p.suffix.lower() == ".wav":
            wav = str(p)                     # aplay can play it directly
        elif self._backend == "aplay":
            wav = str(p)
        return data, fs, wav

    def _ffmpeg_decode(self, p: Path) -> tuple[np.ndarray | None, int]:
        exe = shutil.which("ffmpeg")
        if not exe:
            return None, int(self.fs_out)
        try:
            tmp = self._tmp_path(self._file_key(p) + ".wav")
            proc = subprocess.run(
                [exe, "-y", "-loglevel", "error", "-i", str(p),
                 "-ac", "1", "-ar", str(self.fs_out), str(tmp)],
                capture_output=True, timeout=60.0)
            if proc.returncode != 0:
                self._warn(f"ffmpeg failed on {p.name}: {proc.stderr.decode('utf8', 'ignore')[:120]}")
                return None, int(self.fs_out)
            import soundfile as sf
            data, fs = sf.read(str(tmp), dtype="float32", always_2d=False)
            return np.asarray(data, dtype=np.float32), int(fs)
        except Exception as exc:
            self._warn(f"ffmpeg decode failed for {p.name} ({exc.__class__.__name__})")
            return None, int(self.fs_out)

    def _ensure_wav(self, x: np.ndarray, fs: int, key: str) -> str | None:
        """Render a float array to a 16-bit WAV for aplay, cached on disk."""
        try:
            path = self._tmp_path(key + ".wav")
            if path.exists():
                return str(path)
            import soundfile as sf
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 1:
                sf.write(str(path), arr, int(fs), subtype="PCM_16")
            else:
                sf.write(str(path), arr, int(fs), subtype="PCM_16")
            return str(path)
        except Exception as exc:
            self._warn(f"cannot render temp WAV ({exc.__class__.__name__})")
            return None

    def _tmp_path(self, name: str) -> Path:
        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="nb-audio-"))
        return self._tmpdir / name

    @staticmethod
    def _file_key(p: Path) -> str:
        try:
            raw = f"{p.resolve()}|{p.stat().st_mtime_ns}"
        except OSError:
            raw = str(p)
        return hashlib.sha1(raw.encode("utf8")).hexdigest()[:16]

    @staticmethod
    def _array_key(x: np.ndarray, fs: int) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        h.update(str(int(fs)).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _make_tone(freq: float, dur_s: float, fs: int, amp: float = 0.3,
                   ramp_ms: float = 8.0) -> np.ndarray:
        n = int(max(1, round(max(0.0, float(dur_s)) * fs)))
        t = np.arange(n, dtype=np.float64) / fs
        x = amp * np.sin(2.0 * np.pi * float(freq) * t)
        r = int(min(n // 2, ramp_ms * 1e-3 * fs))
        if r > 1:                            # click-free edges: a step onset
            w = np.linspace(0.0, 1.0, r)     # would smear the ERP with a click
            x[:r] *= w
            x[-r:] *= w[::-1]
        return x.astype(np.float32)
