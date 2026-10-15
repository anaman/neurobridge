"""Deterministic offline audio synthesis.

Why a real synthesizer rather than a stub
-----------------------------------------
The rig must be able to run the whole 64-test curriculum with no network, no
Freesound token and no sound files on disk.  Several tests are only meaningful
if the *acoustics* differ in the right dimension, so a placeholder (a sine
beep) would produce stimuli that cannot elicit the effect under study:

* ``syllable_ba`` vs ``syllable_da`` -- the oddball in T305 is carried entirely
  by the second-formant transition (rising /ba/ vs falling /da/), so this module
  implements a small cascade formant synthesizer with a consonant burst, not a
  single resonator.
* ``pseudo_speech`` -- stands in for real speech when Freesound has no token
  (T605/T606).  It must be speech-like in *prosody* (declining pitch contour,
  ~4 Hz syllable envelope) while being unintelligible; otherwise it would not
  even approximate the listening condition, and its exact time reversal gives
  the reversed-speech control for free.
* ``beat_60`` / ``beat_120`` -- sustained-state tempo tracking needs >= 30 s of
  unambiguous beats with an audible accent, so these are long looped patterns
  and not a handful of clicks.

Everything is normalised with :func:`normalize_rms` to -23 dBFS by default, so
a loudness ladder (T607) and an oddball (standard vs deviant) are not confounded
with a level difference.  All randomness is seeded, so a re-generated file is
byte-identical and a session can be reproduced from its spec alone.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..config import AUDIO_DIR, ensure_dirs

#: Where generated audio lives.  ``generate()`` writes here so the runner can
#: treat a synthesized stimulus exactly like a downloaded one.
SYNTH_DIR = AUDIO_DIR / "synth"

#: Default RMS target: -23 dBFS is the EBU R128 / ATSC A/85 anchor for speech
#: and is a comfortable, non-fatiguing level for long listening blocks.
DEFAULT_DBFS = -23.0

#: One process-wide RNG.  Generation is content-hashed and idempotent, so
#: determinism only needs to hold for a given run/seed.
_RNG = np.random.default_rng(20261013)

# A few public-domain / traditional tunes (all pre-1900, no copyright).
_ODE_TO_JOY = (659.25, 659.25, 698.46, 783.99, 783.99, 698.46, 659.25, 587.33,
               523.25, 523.25, 587.33, 659.25, 659.25, 587.33, 587.33)
_TWINKLE = (523.25, 523.25, 783.99, 783.99, 880.00, 880.00, 783.99,
            698.46, 698.46, 659.25, 659.25, 587.33, 587.33, 523.25)

#: Relative formant targets (Hz) for the vowels used by the syllable generator.
#: Values are textbook Peterson & Barney style midpoints, not measured data --
#: they only need to be *different enough* to be discriminable.
VOWEL_FORMANTS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # vowel: (F1, F2, F3), (bandwidths F1, F2, F3)
    "a": ((730.0, 1090.0, 2440.0), (90.0, 110.0, 160.0)),
    "e": ((530.0, 1840.0, 2480.0), (80.0, 100.0, 150.0)),
    "i": ((270.0, 2290.0, 3010.0), (70.0, 100.0, 170.0)),
    "o": ((570.0, 840.0, 2410.0), (85.0, 100.0, 160.0)),
    "u": ((300.0, 870.0, 2240.0), (70.0, 100.0, 170.0)),
}

#: Consonant onset targets.  ``burst`` = broadband noise burst duration (s),
#: ``aspiration`` = voiceless noise after the burst, ``transition`` = time for
#: the formants to travel from the onset locus to the vowel target.
#: The /b/ vs /d/ contrast lives in F2 (locus ~800 Hz vs ~1700 Hz).
CONSONANTS: dict[str, dict[str, float]] = {
    "b": {"f1": 200.0, "f2": 800.0, "f3": 2000.0, "burst": 0.012,
          "aspiration": 0.005, "transition": 0.040, "voiced": 1.0},
    "p": {"f1": 250.0, "f2": 850.0, "f3": 2000.0, "burst": 0.016,
          "aspiration": 0.045, "transition": 0.045, "voiced": 0.0},
    "d": {"f1": 300.0, "f2": 1700.0, "f3": 2600.0, "burst": 0.012,
          "aspiration": 0.006, "transition": 0.040, "voiced": 1.0},
    "t": {"f1": 320.0, "f2": 1750.0, "f3": 2600.0, "burst": 0.016,
          "aspiration": 0.045, "transition": 0.045, "voiced": 0.0},
    "g": {"f1": 250.0, "f2": 2000.0, "f3": 2400.0, "burst": 0.014,
          "aspiration": 0.008, "transition": 0.045, "voiced": 1.0},
    "k": {"f1": 280.0, "f2": 1900.0, "f3": 2400.0, "burst": 0.018,
          "aspiration": 0.048, "transition": 0.048, "voiced": 0.0},
    "m": {"f1": 250.0, "f2": 1100.0, "f3": 2200.0, "burst": 0.006,
          "aspiration": 0.0, "transition": 0.050, "voiced": 1.0},
    "n": {"f1": 250.0, "f2": 1600.0, "f3": 2500.0, "burst": 0.006,
          "aspiration": 0.0, "transition": 0.050, "voiced": 1.0},
    "s": {"f1": 400.0, "f2": 4000.0, "f3": 5500.0, "burst": 0.090,
          "aspiration": 0.030, "transition": 0.035, "voiced": 0.0},
    "z": {"f1": 350.0, "f2": 3500.0, "f3": 5000.0, "burst": 0.080,
          "aspiration": 0.010, "transition": 0.035, "voiced": 0.6},
}

#: Onsets / nuclei / codas used to build pronounceable pseudowords.  Kept
#: orthographically legal in English so a lexical decision (T406) is a real
#: word/non-word judgement and not a "is this English spelling?" task.
ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "cl", "dr", "fl", "gl", "gr", "pl", "pr", "sk", "sl", "sn",
          "sp", "st", "str", "tr", "bl", "cr", "fr", "sh", "th", "ch")
NUCLEI = ("a", "e", "i", "o", "u", "ai", "ea", "ee", "oa", "oo", "ou", "ie")
CODAS = ("", "", "b", "ck", "d", "f", "g", "k", "l", "ll", "m", "mp", "n",
         "nd", "ng", "nt", "p", "r", "rd", "rk", "rn", "s", "sh", "sk", "st", "t", "th", "x")


# ------------------------------------------------------------------ primitives
def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def normalize_rms(x: np.ndarray, target_dbfs: float = DEFAULT_DBFS) -> np.ndarray:
    """Scale ``x`` (float32) to ``target_dbfs`` RMS, then peak-limit to 0.99.

    Peak limiting after scaling is deliberate: a click or a formant transient
    can clip even at a modest RMS, and a clipped stimulus changes the spectral
    content the analysis is looking for.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    target = 10.0 ** (float(target_dbfs) / 20.0)
    rms = _rms(x)
    if rms > 1e-12:
        x = x * (target / rms)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


def _ramp(n: int, fs: int, ramp_ms: float) -> np.ndarray:
    """Raised-cosine attack/release window of ``n`` samples."""
    env = np.ones(max(n, 1), dtype=np.float64)
    k = int(max(1, round(float(ramp_ms) * 1e-3 * fs)))
    k = min(k, max(1, n // 2))
    if n > 2 and k > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k))
        env[:k] = w
        env[-k:] = w[::-1]
    return env


def _place(chunks: list[tuple[float, np.ndarray]], total_s: float,
           fs: int) -> np.ndarray:
    """Add ``(start_s, samples)`` pairs into one buffer (later events sum)."""
    out = np.zeros(max(1, int(round(total_s * fs))), dtype=np.float64)
    for start, sig in chunks:
        i0 = int(round(start * fs))
        if i0 < 0:
            sig = sig[-i0:]
            i0 = 0
        if i0 >= out.size:
            continue
        seg = np.asarray(sig, dtype=np.float64).ravel()
        i1 = min(out.size, i0 + seg.size)
        out[i0:i1] += seg[:i1 - i0]
    return out


# ------------------------------------------------------------------ tones etc.
def tone(freq: float, dur: float, fs: int = 44100, amp: float = 0.3,
         ramp_ms: float = 8.0) -> np.ndarray:
    """Pure tone with a raised-cosine gate (no click at onset/offset)."""
    n = max(1, int(round(float(dur) * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    x = np.sin(2.0 * np.pi * float(freq) * t) * float(amp)
    return (x * _ramp(n, fs, ramp_ms)).astype(np.float32)


def click(fs: int = 44100, dur: float = 0.004) -> np.ndarray:
    """Very short broadband click (a delta is too harsh for speakers/ears)."""
    n = max(1, int(round(float(dur) * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    env = np.exp(-t / max(1e-4, 0.0012))
    x = _RNG.standard_normal(n) * env
    # A tiny DC-free high-pass so the click does not thump the speaker.
    x = np.diff(np.concatenate([[0.0], x]))
    return normalize_rms(x, -20.0)


def click_train(n: int = 60, isi: float = 1.2, fs: int = 44100
                ) -> tuple[np.ndarray, list[float]]:
    """``n`` clicks spaced ``isi`` seconds apart.

    Returns ``(signal, onset_times_s)``.  Used by T004 (auditory latency
    calibration): the *times* are what the analyst regresses against, so they
    are returned explicitly rather than re-derived by peak-picking.
    """
    n = max(1, int(n))
    isi = max(1e-3, float(isi))
    total = (n - 1) * isi + 0.05
    c = click(fs=fs)
    times = [i * isi for i in range(n)]
    x = _place([(t0, c) for t0 in times], total, fs)
    return normalize_rms(x, -20.0), times


def pink_noise(dur: float, fs: int = 44100, rng: np.random.Generator | None = None) -> np.ndarray:
    """1/f noise via spectral shaping (exact slope, unlike a filter cascade).

    ``rng`` is injectable so callers that must be reproducible from a seed
    (``pseudo_speech``) do not depend on the process-wide generator.
    """
    rng = rng or _RNG
    n = max(2, int(round(float(dur) * fs)))
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0
    x = np.fft.irfft(spec * scale, n=n)
    return normalize_rms(x, -20.0)


def _highpass20(x: np.ndarray, fs: int) -> np.ndarray:
    """One-pole DC blocker at ~20 Hz; kills the inaudible wander of brown noise."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    a = float(np.exp(-2.0 * np.pi * 20.0 / fs))
    y = np.empty_like(x)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(x.size):
        prev_y = a * (prev_y + x[i] - prev_x)
        prev_x = x[i]
        y[i] = prev_y
    return y


def noise(dur: float, kind: str = "pink", fs: int = 44100, amp: float = 0.25,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """White, pink or brown noise with a 10 ms gate.

    Brown noise is the integral of white (random walk), high-passed at ~20 Hz
    so it does not contain a large inaudible DC wander that would eat headroom.
    """
    rng = rng or _RNG
    n = max(1, int(round(float(dur) * fs)))
    kind = (kind or "pink").strip().lower()
    if kind == "pink":
        x = pink_noise(dur, fs, rng=rng).astype(np.float64)
    elif kind in ("brown", "red"):
        x = _highpass20(np.cumsum(rng.standard_normal(n)), fs)
    elif kind == "white":
        x = rng.standard_normal(n)
    else:
        raise ValueError(f"unknown noise kind {kind!r} (white|pink|brown)")
    x = x * (float(amp) / max(1e-9, float(np.max(np.abs(x)))))
    return (x * _ramp(n, fs, 10.0)).astype(np.float32)


def _detune(freq: float, cents: float) -> float:
    return float(freq) * (2.0 ** (float(cents) / 1200.0))


def chord(freqs: Sequence[float], dur: float, fs: int = 44100,
          *, dissonant: bool = False) -> np.ndarray:
    """Additive chord; ``dissonant`` adds 2nds and mild detuning roughness.

    Consonant = integer-ratio intervals (octave, fifth).  Dissonant = a
    minor-second cluster plus a few cents of detune per partial, which is what
    actually produces the beating/roughness the auditory system hears as
    unpleasant (T603).
    """
    n = max(1, int(round(float(dur) * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    x = np.zeros(n)
    for i, f in enumerate(freqs):
        f = float(f)
        if f <= 0:
            continue
        cents = (7.0 * (i + 1)) if dissonant else 0.0
        partial = np.sin(2.0 * np.pi * _detune(f, cents) * t)
        partial += 0.25 * np.sin(2.0 * np.pi * _detune(2.0 * f, cents) * t)
        x += partial / max(1, len(freqs))
    if dissonant and len(freqs) >= 2:
        f0 = float(freqs[0])
        x += 0.35 * np.sin(2.0 * np.pi * _detune(f0 * (2.0 ** (1.0 / 12.0)), 11.0) * t)
    return normalize_rms((x * _ramp(n, fs, 15.0)).astype(np.float32), DEFAULT_DBFS)


def melody(notes: Sequence[float], note_dur: float, fs: int = 44100,
           dur: float | None = None) -> np.ndarray:
    """Sequence of gated tones (one note per ``note_dur``).

    ``dur=None`` -> play the sequence once at its natural length.  When ``dur`` is
    given the notes are tiled (or truncated) to that length: a sustained-state
    block needs continuous musical material, and re-tiling the *same* tune keeps
    familiarity intact for the familiar-melody condition (T604) while the
    unfamiliar sequence is a different pitch class set entirely.
    """
    note_dur = max(1e-3, float(note_dur))
    seq = [float(f) for f in notes if float(f) > 0]
    if not seq:
        seq = [440.0]
    natural = note_dur * len(seq)
    total = float(dur) if dur else natural
    n_notes = max(1, int(math.ceil(total / note_dur)))
    chunks = []
    for i in range(n_notes):
        start = i * note_dur
        if start >= total:
            break
        chunks.append((start, tone(seq[i % len(seq)], min(note_dur * 0.92, total - start),
                                   fs=fs, amp=1.0)))
    x = _place(chunks, total, fs)
    return normalize_rms(x, DEFAULT_DBFS)


def _valve(x: np.ndarray) -> np.ndarray:
    """Soft-clip a pulse train; a hard rectangle would alias badly."""
    return np.tanh(2.5 * np.asarray(x, dtype=np.float64))


def _formant_resonator(x: np.ndarray, f: np.ndarray, bw: np.ndarray,
                       fs: int) -> np.ndarray:
    """Time-varying 2-pole resonator, one sample at a time.

    The recursion is direct-form-I in the *output* domain::

        y[n] = K[n] * x[n] + b1[n] * y[n-1] + b2[n] * y[n-2]

    with the pole pair placed by ``b1 = 2 r cos(theta)`` and ``b2 = -r^2``
    (radius ``r = exp(-pi * bw / fs)``, both strictly inside the unit circle),
    and ``K`` the *reciprocal* peak magnitude so the resonator has unity gain at
    its formant.  Two details are easy to get wrong and both bit this file:

    * the state must be the output samples.  Carrying an un-normalised
      intermediate and dividing only at emission multiplies the response by
      K^-2 (~570,000x for a 90 Hz bandwidth) and overflows float64.
    * the ``z^-2`` coefficient is ``-r^2``, not ``+r^2``.  Using ``+r^2`` puts
      the poles at radius ``1/r > 1`` and the filter diverges geometrically.

    A Python loop is unavoidable: the coefficients change every sample during a
    formant transition, which ``scipy.signal.lfilter`` cannot express.  These
    signals are at most a few seconds long, so correctness wins over speed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    y = np.zeros(n)
    if n == 0:
        return y
    f = np.broadcast_to(np.asarray(f, dtype=np.float64), (n,))
    bw = np.broadcast_to(np.asarray(bw, dtype=np.float64), (n,))
    r = np.clip(np.exp(-np.pi * np.maximum(bw, 1.0) / fs), 0.0, 0.999999)
    theta = 2.0 * np.pi * np.clip(f, 1.0, 0.45 * fs) / fs
    b1 = 2.0 * r * np.cos(theta)
    b2 = -r * r
    # Exact |H| at the resonant frequency for a pole pair of radius r:
    #   peak = 1 / ((1-r) * sqrt(1 - 2 r cos(2 theta) + r^2))
    denom = (1.0 - r) * np.sqrt(np.maximum(1e-12, 1.0 - 2.0 * r * np.cos(2.0 * theta) + r * r))
    gain = np.maximum(1e-12, denom)                 # 1 / peak  -> unity at formant
    y1 = 0.0
    y2 = 0.0
    for i in range(n):
        v = gain[i] * x[i] + b1[i] * y1 + b2[i] * y2
        y2 = y1
        y1 = v
        y[i] = v
    return y


def _glottal_source(f0: np.ndarray, fs: int, jitter: float = 0.0) -> np.ndarray:
    """Band-limited-ish pulse train at a time-varying ``f0`` (phase accumulator)."""
    n = f0.size
    phase = np.cumsum(f0) / float(fs)
    frac = phase - np.floor(phase)
    if jitter > 0:
        frac = np.mod(frac + jitter * np.sin(2.0 * np.pi * phase * 0.5), 1.0)
    # asymmetric glottal pulse: fast open, slower close
    pulse = np.where(frac < 0.45,
                     np.sin(np.pi * frac / 0.45) ** 2,
                     -0.25 * np.exp(-(frac - 0.45) * 9.0))
    return pulse - float(np.mean(pulse))


def _pre_emphasis(x: np.ndarray, coeff: float = 0.95) -> np.ndarray:
    """First-difference emphasis of the glottal source.

    A glottal pulse train has roughly a 1/f^2 spectral tilt: at 44.1 kHz the
    second formant of a 110 Hz voice sits 25-30 dB below the fundamental.  Left
    alone, F1 dominates everything and the /ba/-/da/ F2 transition -- the whole
    point of the speech-syllable oddball -- is ~0.1% of the signal energy and
    acoustically invisible.  Pre-emphasis is the standard fix (it is what the
    vocal tract's lip radiation does in a real voice).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def _noise_burst(n: int, fs: int, decay_s: float,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or _RNG
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n, dtype=np.float64) / float(fs)
    return rng.standard_normal(n) * np.exp(-t / max(1e-4, decay_s))


def syllable(vowel: str = "a", consonant: str = "b", dur: float = 0.25,
             f0: float = 110.0, fs: int = 44100) -> np.ndarray:
    """CV syllable from a two/three-formant cascade with a consonant burst.

    Structure: burst (+ aspiration for voiceless stops) -> 40 ms formant
    transition from the consonant locus to the vowel target -> steady vowel.
    The /ba/-/da/ difference is *only* the F2 locus (800 Hz vs 1700 Hz), which
    is the acoustic cue the speech-oddball test (T305) depends on.
    """
    vowel = (vowel or "a").strip().lower()[:1]
    consonant = (consonant or "b").strip().lower()[:1]
    if vowel not in VOWEL_FORMANTS:
        raise ValueError(f"unknown vowel {vowel!r}; have {sorted(VOWEL_FORMANTS)}")
    if consonant not in CONSONANTS:
        raise ValueError(f"unknown consonant {consonant!r}; have {sorted(CONSONANTS)}")
    c = CONSONANTS[consonant]
    targets, bws = VOWEL_FORMANTS[vowel]
    n = max(8, int(round(float(dur) * fs)))
    dur = n / float(fs)
    t = np.arange(n, dtype=np.float64) / float(fs)

    # --- voiced source: slight declination + vibrato => natural, not buzzy
    contour = 1.0 - 0.18 * (t / max(1e-6, dur))
    vibrato = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.5 * t)
    f0_track = float(f0) * contour * vibrato
    src = _glottal_source(f0_track, fs, jitter=0.01)

    # --- excitation: burst + aspiration (voiceless) + voicing
    onset = min(0.09, 0.35 * dur)
    n_on = int(round(onset * fs))
    exc = np.zeros(n)
    burst_n = min(n, int(round(c["burst"] * fs)))
    if burst_n > 0:
        exc[:burst_n] += 1.6 * _noise_burst(burst_n, fs, c["burst"] * 0.45)
    asp_n = int(round(c["aspiration"] * fs))
    if asp_n > 0 and burst_n < n:
        i0 = burst_n
        exc[i0:i0 + asp_n] += 0.7 * _noise_burst(min(asp_n, n - i0), fs,
                                                 max(1e-4, c["aspiration"] * 0.8))
    if c["voiced"] > 0:
        i0 = max(0, int(round(0.35 * n_on)))
        exc[i0:] += c["voiced"] * src[:n - i0]
    exc = _pre_emphasis(exc)

    # --- formant tracks (Hz and bandwidth per sample)
    # The transition is a *fixed* duration (the consonant's locus), not a
    # fraction of the syllable: stretching it over most of the syllable would
    # mean the vowel target is never reached, and /ba/ vs /da/ would become
    # acoustically identical -- exactly the cue T305 depends on.
    n_tr = int(round(min(float(c["transition"]), 0.6 * dur) * fs))
    n_tr = max(1, min(n_tr, n))
    f1 = np.empty(n)
    f2 = np.empty(n)
    f3 = np.empty(n)
    w = np.linspace(0.0, 1.0, n_tr) if n_tr > 1 else np.ones(1)
    smooth = 0.5 - 0.5 * np.cos(np.pi * w)          # raised-cosine glide
    f1[:n_tr] = c["f1"] + (targets[0] - c["f1"]) * smooth
    f2[:n_tr] = c["f2"] + (targets[1] - c["f2"]) * smooth
    f3[:n_tr] = c["f3"] + (targets[2] - c["f3"]) * smooth
    f1[n_tr:], f2[n_tr:], f3[n_tr:] = targets[0], targets[1], targets[2]

    y = _formant_resonator(exc, f1, np.full(n, bws[0]), fs)
    y = _formant_resonator(y, f2, np.full(n, bws[1]), fs)
    y = _formant_resonator(y, f3, np.full(n, bws[2]), fs)

    # --- amplitude envelope: quick onset, gentle release
    env = _ramp(n, fs, 12.0)
    rel = int(round(min(0.06, 0.3 * dur) * fs))
    if rel > 1:
        env[-rel:] *= np.linspace(1.0, 0.0, rel)
    y = y * env
    return normalize_rms(y, DEFAULT_DBFS)


def amplitude_modulate(carrier_hz: float, mod_hz: float, depth: float,
                       dur: float, fs: int = 44100) -> np.ndarray:
    """AM tone: carrier at ``carrier_hz``, envelope at ``mod_hz``.

    ``depth`` in 0..1.  Used for steady-state entrainment (T103) where the
    envelope frequency, not the carrier, is the stimulus of interest.
    """
    n = max(1, int(round(float(dur) * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    depth = float(np.clip(depth, 0.0, 1.0))
    env = 1.0 - depth + depth * (0.5 - 0.5 * np.cos(2.0 * np.pi * float(mod_hz) * t))
    carrier = np.sin(2.0 * np.pi * float(carrier_hz) * t)
    return normalize_rms((carrier * env * _ramp(n, fs, 10.0)).astype(np.float32), DEFAULT_DBFS)


def beat(fs: int = 44100, bpm: float = 60.0, dur: float = 32.0,
         accent_every: int = 4) -> np.ndarray:
    """Metronome/percussive loop at ``bpm`` with an accented downbeat.

    ``accent_every`` = beats per bar; the accented beat is both louder and
    pitched lower (a kick-ish thump) plus a high click, so the bar structure is
    unambiguous for tempo tracking (T602) and for a sustained-state block.
    """
    fs = int(fs)
    bpm = max(1.0, float(bpm))
    dur = max(1.0, float(dur))
    period = 60.0 / bpm
    n_beats = int(np.floor(dur / period)) + 1
    chunks: list[tuple[float, np.ndarray]] = []
    for i in range(n_beats):
        beat_s = i * period
        if beat_s >= dur:
            break
        if accent_every and (i % int(accent_every) == 0):
            thump = tone(60.0, 0.18, fs=fs, amp=0.7, ramp_ms=4.0)
            thump = thump * np.exp(-np.arange(thump.size) / (0.05 * fs))
            chunks.append((beat_s, thump))
            chunks.append((beat_s, click(fs=fs) * 1.4))
        else:
            c = click(fs=fs)
            chunks.append((beat_s, c))
    x = _place(chunks, dur, fs)
    return normalize_rms(x, DEFAULT_DBFS)


def silence(dur: float = 1.0, fs: int = 44100) -> np.ndarray:
    """Digital silence with a tiny dither floor (some DACs mute hard at 0)."""
    n = max(1, int(round(float(dur) * fs)))
    return (1e-6 * _RNG.standard_normal(n)).astype(np.float32)


def _ambience(dur: float, fs: int, *, tense: bool) -> np.ndarray:
    """Calm vs tense soundscape from filtered noise + sparse events.

    Calm: low-frequency-dominant pink noise with slow swells and occasional
    distant birds (short high chirps).  Tense: broadband noise with a slowly
    rising band, irregular low rumbles and an occasional dissonant sustained
    tone.  Both are describable rather than "random noise", which is what makes
    them usable as an arousal manipulation (T104).
    """
    n = max(1, int(round(float(dur) * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    rng = np.random.default_rng(31337 if tense else 20261013)
    base = pink_noise(dur, fs).astype(np.float64)
    if tense:
        # slow upward sweep of a band-limited noise layer + irregular rumbles
        sweep = 0.5 + 0.5 * np.sin(2.0 * np.pi * (0.05 + 0.02 * t) * t)
        x = base * (0.6 + 0.6 * sweep)
        for k in range(int(max(1, dur / 7.0))):
            t0 = float(rng.uniform(0, max(0.1, dur - 1.0)))
            rumble = tone(float(rng.uniform(38, 70)), 1.2, fs=fs, amp=0.5, ramp_ms=60.0)
            x = x + _place([(t0, rumble)], dur, fs) * 0.7
        for k in range(int(max(1, dur / 9.0))):
            t0 = float(rng.uniform(0, max(0.1, dur - 2.0)))
            f = float(rng.choice([220.0, 233.1, 311.1]))
            x = x + _place([(t0, tone(f, 1.6, fs=fs, amp=0.25, ramp_ms=150.0))], dur, fs)
    else:
        swell = 1.0 + 0.35 * np.sin(2.0 * np.pi * 0.07 * t)
        x = base * swell
        for k in range(int(max(1, dur / 6.0))):
            t0 = float(rng.uniform(0, max(0.1, dur - 0.4)))
            f0 = float(rng.uniform(1800, 3200))
            chirp = amplitude_modulate(f0, 14.0, 0.9, 0.18, fs=fs) * 0.25
            x = x + _place([(t0, chirp.astype(np.float64))], dur, fs)
    return normalize_rms(x, DEFAULT_DBFS)


# ------------------------------------------------------------------ speech-like
def _phrase_formants(rng: np.random.Generator, syllable_s: float, fs: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one speech-like phrase: excitation + formant tracks + envelope."""
    n = max(16, int(round(syllable_s * fs)))
    out_exc = np.zeros(n)
    out_a1 = np.zeros(n)
    out_a2 = np.zeros(n)
    out_a3 = np.zeros(n)
    out_env = np.zeros(n)
    pos = 0
    # a phrase ends with a falling pitch and a lengthened final syllable
    vow_keys = list(VOWEL_FORMANTS)
    while pos < n - 8:
        is_final = pos > 0.72 * n
        syl_len = int(rng.uniform(0.14, 0.26) * fs * (1.5 if is_final else 1.0))
        syl_len = min(syl_len, n - pos)
        if syl_len < 8:
            break
        vowel = str(rng.choice(vow_keys))
        cons = str(rng.choice(["b", "d", "g", "m", "n", "z", "l", "v", ""]))
        c = CONSONANTS.get(cons or "m", CONSONANTS["m"])
        targets, _bw = VOWEL_FORMANTS[vowel]
        m = syl_len
        tt = np.arange(m, dtype=np.float64) / fs
        # excitation: optional consonant noise + voiced source
        exc = np.zeros(m)
        burst_n = min(m, int(round(c["burst"] * fs))) if cons else 0
        if burst_n > 0:
            exc[:burst_n] += 1.2 * _noise_burst(burst_n, fs,
                                                max(1e-4, c["burst"] * 0.5), rng=rng)
        voiced = c["voiced"] if cons else 1.0
        if voiced <= 0:
            voiced = 1.0
        f0 = float(rng.uniform(100.0, 175.0)) * (1.0 - (0.22 if is_final else 0.0) * tt / max(1e-6, tt[-1]))
        exc += voiced * _glottal_source(f0, fs, jitter=0.02)
        exc = _pre_emphasis(exc)
        # fixed-length consonant transition, as in syllable()
        n_tr = max(1, min(m, int(round(min(float(c["transition"]), 0.5 * m / fs) * fs))))
        w = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n_tr))
        a1 = np.full(m, targets[0])
        a2 = np.full(m, targets[1])
        a3 = np.full(m, targets[2])
        if cons:
            a1[:n_tr] = c["f1"] + (targets[0] - c["f1"]) * w
            a2[:n_tr] = c["f2"] + (targets[1] - c["f2"]) * w
            a3[:n_tr] = c["f3"] + (targets[2] - c["f3"]) * w
        env = np.sin(np.linspace(0.0, np.pi, m)) ** 1.2
        out_exc[pos:pos + m] = exc
        out_a1[pos:pos + m] = a1
        out_a2[pos:pos + m] = a2
        out_a3[pos:pos + m] = a3
        out_env[pos:pos + m] = env
        pos += m + int(rng.uniform(0.0, 0.05) * fs)     # short inter-syllable gap
    return out_exc, out_a1, out_a2, out_a3, out_env


def pseudo_speech(dur: float = 6.0, seed: int = 20261013, fs: int = 44100,
                  f0: float = 120.0) -> np.ndarray:
    """Unintelligible but speech-like babble, deterministic from ``seed``.

    Prosody carries the "this is speech" percept: phrases of 3-7 syllables at
    ~4 Hz with a declining pitch contour over each phrase, followed by a short
    pause.  Formants move as in real CV syllables, but no phoneme inventory or
    lexicon is used, so the result cannot be transcribed -- exactly the
    requirement for standing in as a speech stimulus when no Freesound token
    exists, and for which the exact time reversal is the reversed-speech
    control (T605).
    """
    fs = int(fs)
    dur = max(0.2, float(dur))
    rng = np.random.default_rng(int(seed))
    chunks: list[tuple[float, np.ndarray]] = []
    t_cursor = 0.0
    while t_cursor < dur - 0.2:
        syl_rate = float(rng.uniform(3.4, 5.0))          # syllables per second
        n_syl = int(rng.integers(3, 8))
        phrase_s = n_syl / syl_rate
        phrase_s = min(phrase_s, dur - t_cursor)
        exc, a1, a2, a3, env = _phrase_formants(rng, phrase_s, fs)
        if exc.size:
            y = _formant_resonator(exc, a1, np.full(exc.size, 90.0), fs)
            y = _formant_resonator(y, a2, np.full(exc.size, 110.0), fs)
            y = _formant_resonator(y, a3, np.full(exc.size, 170.0), fs)
            y = y * env
            # global pitch is already in the source; scale to requested f0 band
            y = y * (float(f0) / 120.0)
            chunks.append((t_cursor, normalize_rms(y, DEFAULT_DBFS)))
        t_cursor += phrase_s + float(rng.uniform(0.25, 0.6))   # breath pause
    x = _place(chunks, dur, fs)
    if not np.any(x):
        x = pink_noise(dur, fs, rng=rng).astype(np.float64)
    return normalize_rms(x, DEFAULT_DBFS)


# ------------------------------------------------------------------ IO
def save_wav(path: Path, x: np.ndarray, fs: int = 44100) -> Path:
    """Write ``x`` as 16-bit PCM WAV (soundfile), creating parents."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(x, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:                                   # never write clipped ints
        data = data / peak
    sf.write(str(path), data.astype(np.float32), int(fs), subtype="PCM_16")
    return path


# ------------------------------------------------------------------ named generators
def _parse_tone_tail(tail: str, dur: float) -> float | None:
    """``1000_50ms`` -> 0.05; ``1000`` -> ``dur``.  ``None`` means "no match"."""
    if not tail:
        return float(dur)
    parts = [p for p in tail.split("_") if p]
    if len(parts) > 2:
        return None
    if len(parts) == 1:
        return float(dur)
    value, unit = parts[1], parts[0]
    unit = unit.lower()
    try:
        amount = float(value)
    except ValueError:
        return None
    if unit.endswith("ms"):
        return amount / 1000.0
    if unit.endswith("s"):
        return amount
    return None


def _parse_name(name: str) -> tuple[Callable[..., np.ndarray], dict[str, Any]] | None:
    """Resolve a generator *name* (not the overrides) to a callable + defaults.

    Pattern matching rather than a fixed lookup because the catalog keeps
    inventing frequencies and durations for deviant conditions (T301 uses 1200
    and 1500 Hz, T303 a 50 ms vs 100 ms duration deviant, T306 a 700 Hz roving
    standard).  Inventing a new Python name for each would mean editing this
    module every time a spec changes; the patterns cover the whole space.
    """
    key = (name or "").strip().lower()
    if key in NAMES:
        # Fixed entry point for readability, but *defaults only*: overrides win.
        return None, {}
    m = re.fullmatch(r"tone_(\d+(?:\.\d+)?)(?:_([0-9a-z]+))?", key)
    if m:
        freq = float(m.group(1))
        dur = _parse_tone_tail(m.group(2) or "", 0.5)
        if dur is None:
            return None
        return (lambda **k: tone(float(k.pop("freq", freq)), **k)), {"dur": dur}
    m = re.fullmatch(r"noise_([a-z]+)", key)
    if m and m.group(1) in ("white", "pink", "brown", "red"):
        kind = "brown" if m.group(1) == "red" else m.group(1)
        return (lambda **k: noise(k.pop("kind", kind), **k)), {"dur": 3.0}
    m = re.fullmatch(r"syllable_([a-z]{1,2})", key)
    if m:
        pair = m.group(1)
        vowel = pair[0]
        cons = pair[1] if len(pair) > 1 else ""
        return (lambda **k: syllable(k.pop("vowel", vowel), k.pop("consonant", cons), **k)), {"dur": 0.3}
    m = re.fullmatch(r"am_(\d+(?:\.\d+)?)hz", key)
    if m:
        mod = float(m.group(1))
        return (lambda **k: amplitude_modulate(float(k.pop("carrier_hz", 1000.0)), mod,
                                               float(k.pop("depth", 1.0)), **k)), {"dur": 30.0}
    m = re.fullmatch(r"beat_(\d+(?:\.\d+)?)", key)
    if m:
        bpm = float(m.group(1))
        return (lambda **k: beat(bpm=bpm, **k)), {"dur": 32.0}
    m = re.fullmatch(r"(?:click_train_)?(\d+)hz", key)
    if m:
        freq = float(m.group(1))
        return (lambda **k: tone(float(k.pop("freq", freq)), **k)), {"dur": 0.5}
    if key.startswith("tone_"):
        raise KeyError(_pattern_help(name))
    if key.startswith("noise_"):
        raise KeyError(_pattern_help(name))
    if key.startswith("syllable_"):
        raise KeyError(_pattern_help(name))
    m = re.fullmatch(r"chord_(consonant|dissonant)", key)
    if m:
        diss = m.group(1) == "dissonant"
        return (lambda **k: chord((261.63, 277.18, 369.99) if diss else (261.63, 329.63, 392.00),
                                  k.pop("dur", 2.0), dissonant=diss, **k)), {}
    if key == "pseudo_speech_reversed":
        return (lambda **k: pseudo_speech(k.pop("dur", 8.0), k.pop("seed", 20261013), **k
                                          )[::-1].copy()), {}
    if key == "pseudo_speech":
        return (lambda **k: pseudo_speech(**k)), {"dur": 8.0}
    return None


def _pattern_help(name: str) -> str:
    """The error message for an unrecognised generator name.

    Deliberately lists the patterns: the catalog is written by other people, and
    "unknown generator" without the grammar is a support ticket.
    """
    return (
        f"unknown synth generator {name!r}. Understood patterns: "
        "'tone_<hz>' and 'tone_<hz>_<dur_ms>ms' (e.g. tone_1200, tone_1000_50ms); "
        "'noise_<white|pink|brown>'; 'syllable_<cv>' (e.g. syllable_ba, syllable_da); "
        "'am_<hz>hz'; 'beat_<bpm>'; 'pseudo_speech[_reversed]'; 'chord_consonant|dissonant'; "
        "'melody_familiar|unfamiliar'; 'ambience_calm|tense'; "
        f"'click'; 'click_train'; 'silence'; plus the fixed set {sorted(NAMES)}"
    )


def _resolve_overrides(name: str, kw: dict[str, Any]) -> tuple[Callable[..., np.ndarray], dict[str, Any]]:
    """Return ``(fn, kwargs)`` for ``name`` with spec-supplied overrides applied.

    Recognised overrides (all optional): ``freq``/``freq_hz``, ``dur``/``dur_s``,
    ``amp``, ``fs``, ``seed`` and ``tone_duration_ms`` (the duration-deviant
    spelling used by the MMN specs).  Unknown overrides are passed through to
    the generator, which raises if it cannot honour them.
    """
    key = (name or "").strip().lower()
    overrides = dict(kw)
    if key in NAMES:
        fn = NAMES[key]
        defaults: dict[str, Any] = {}
    else:
        parsed = _parse_name(key)
        if parsed is None:
            raise KeyError(_pattern_help(name))
        fn, defaults = parsed
    params: dict[str, Any] = dict(defaults)
    # duration spellings
    for src in ("tone_duration_ms", "dur_ms", "duration_ms"):
        if src in overrides:
            params["dur"] = float(overrides.pop(src)) / 1000.0
    for src in ("dur", "dur_s", "duration_s"):
        if src in overrides:
            params["dur"] = float(overrides.pop(src))
    for src in ("freq", "freq_hz", "frequency_hz"):
        if src in overrides:
            params["freq"] = float(overrides.pop(src))
    if "amp" in overrides:
        params["amp"] = float(overrides.pop("amp"))
    params.update(overrides)                     # pass the rest straight through
    return fn, params


def _names() -> dict[str, Callable[..., np.ndarray]]:
    """The documented *common* generators (the catalog is not limited to these).

    These are the names the INTERFACES contract lists; they are treated as
    defaults, not a whitelist: :func:`generate` also resolves the dynamic
    ``tone_*`` / ``noise_*`` / ``syllable_*`` / ``am_*`` / ``beat_*`` patterns
    that the 64 specs actually use.

    Three families exist because the specs ask for three things: fixed
    references (tones, clicks, noise), musical material (chords, melodies) and
    speech surrogates (syllables, pseudo-speech).  ``beat_*`` and ``ambience_*``
    are long-form (>= 30 s) so a sustained-state block does not loop audibly.
    """
    return {
        # fixed references
        "tone_500": lambda **k: tone(k.pop("freq", 500.0), k.pop("dur", 0.5), **k),
        "tone_1000": lambda **k: tone(k.pop("freq", 1000.0), k.pop("dur", 0.5), **k),
        "tone_1200": lambda **k: tone(k.pop("freq", 1200.0), k.pop("dur", 0.5), **k),
        "tone_1500": lambda **k: tone(k.pop("freq", 1500.0), k.pop("dur", 0.5), **k),
        "tone_2000": lambda **k: tone(k.pop("freq", 2000.0), k.pop("dur", 0.5), **k),
        "click": lambda **k: click(**k),
        "click_train": lambda **k: click_train(k.pop("n", 60), k.pop("isi", 1.2),
                                               k.pop("fs", 44100))[0],
        # noise
        "noise_pink": lambda **k: noise(k.pop("dur", 3.0), k.pop("kind", "pink"), **k),
        "noise_white": lambda **k: noise(k.pop("dur", 3.0), k.pop("kind", "white"), **k),
        "noise_brown": lambda **k: noise(k.pop("dur", 3.0), k.pop("kind", "brown"), **k),
        # speech surrogates
        "syllable_ba": lambda **k: syllable(k.pop("vowel", "a"), k.pop("consonant", "b"), **k),
        "syllable_da": lambda **k: syllable(k.pop("vowel", "a"), k.pop("consonant", "d"), **k),
        "syllable_pa": lambda **k: syllable(k.pop("vowel", "a"), k.pop("consonant", "p"), **k),
        "syllable_ga": lambda **k: syllable(k.pop("vowel", "a"), k.pop("consonant", "g"), **k),
        "pseudo_speech": lambda **k: pseudo_speech(k.pop("dur", 8.0), k.pop("seed", 20261013), **k),
        "pseudo_speech_reversed": lambda **k: pseudo_speech(
            k.pop("dur", 8.0), k.pop("seed", 20261013), **k)[::-1].copy(),
        # music
        "chord_consonant": lambda **k: chord((261.63, 329.63, 392.00),
                                             k.pop("dur", 2.0), **k),
        "chord_dissonant": lambda **k: chord((261.63, 277.18, 369.99),
                                             k.pop("dur", 2.0), dissonant=True, **k),
        "melody_familiar": lambda **k: melody(_ODE_TO_JOY, k.pop("note_dur", 0.45), **k),
        "melody_unfamiliar": lambda **k: melody(_unfamiliar_notes(), k.pop("note_dur", 0.45), **k),
        "am_10hz": lambda **k: amplitude_modulate(k.pop("carrier_hz", 1000.0), 10.0,
                                                  k.pop("depth", 1.0), k.pop("dur", 30.0), **k),
        "beat_60": lambda **k: beat(bpm=60.0, accent_every=k.pop("accent_every", 4), **k),
        "beat_120": lambda **k: beat(bpm=120.0, accent_every=k.pop("accent_every", 4), **k),
        "ambience_calm": lambda **k: _ambience(k.pop("dur", 30.0), k.pop("fs", 44100),
                                               tense=False, **k),
        "ambience_tense": lambda **k: _ambience(k.pop("dur", 30.0), k.pop("fs", 44100),
                                                tense=True, **k),
        "silence": lambda **k: silence(**k),
    }


def _unfamiliar_notes(n: int = 14, seed: int = 4242) -> tuple[float, ...]:
    """A deterministic, deliberately unsingable-to-most-people pitch sequence.

    Chosen from an atonal set class (no tonal centre, no folk-tune shape) so it
    cannot be a familiar melody by accident -- that is the contrast in T604.
    """
    rng = np.random.default_rng(int(seed))
    scale = (261.63, 277.18, 311.13, 369.99, 392.00, 415.30, 466.16, 523.25)
    return tuple(float(rng.choice(scale)) for _ in range(int(n)))


NAMES: dict[str, Callable[..., np.ndarray]] = _names()


def is_generator(name: str) -> bool:
    """True when ``name`` resolves to a generator (fixed name or dynamic pattern)."""
    key = str(name or "").strip().lower()
    if key in NAMES:
        return True
    return _parse_name(key) is not None


#: Generators whose length is *not* a free parameter: the material has an
#: intrinsic duration (a melody, a metronome bar) or is defined by a fixed
#: acoustic event (a syllable, a click).  The builder must not stretch these to
#: ``stimuli.duration_s`` -- doing so would destroy the contrast under test.
INTRINSIC_TOKENS = (
    "pseudoword", "chord_", "melody_", "pseudo_speech", "beat_",
    "ambience_", "click", "silence", "syllable_",
)


def generator_token(name: str) -> str:
    """The *family* of a generator name, used to decide override policy.

    ``tone_1000_50ms`` -> ``tone``, ``noise_brown`` -> ``noise``, and the
    intrinsically-timed families are returned as ``chord``, ``melody``, ``beat``,
    ``ambience``, ``pseudo_speech``, ``syllable``, ``click`` or ``pseudoword``.
    """
    key = str(name or "").strip().lower()
    if key.startswith("tone") or re.fullmatch(r"\d+hz", key):
        return "tone"
    if key.startswith("noise"):
        return "noise"
    if key.startswith("syllable"):
        return "syllable"
    if key.startswith("chord"):
        return "chord"
    if key.startswith("melody"):
        return "melody"
    if key.startswith("pseudo_speech"):
        return "pseudo_speech"
    if key.startswith("beat"):
        return "beat"
    if key.startswith("ambience"):
        return "ambience"
    if key.startswith("click"):
        return "click"
    if key.startswith("am_"):
        return "am"
    if key == "silence":
        return "silence"
    return key


def is_intrinsic(name: str) -> bool:
    """True when ``name`` has a musically/acoustically fixed length."""
    token = generator_token(name)
    return any(token.startswith(t) or t.startswith(token) for t in INTRINSIC_TOKENS) \
        if token else False


def _safe_token(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]", "", str(value))[:32]


def generate(name: str, force: bool = False, **kw: Any) -> Path:
    """Render a generator (fixed name or ``tone_<hz>``-style pattern) to WAV.

    Idempotent by design: the file name encodes the generator and the
    spec-supplied overrides, and an existing file is re-used unless
    ``force=True``.  A curriculum rerun is therefore cheap, and a session's
    stimuli stay reproducible from its spec.

    Overrides (``freq``, ``dur``, ``tone_duration_ms``, ``amp``, ``fs``,
    ``seed``) let a spec parameterise a generator without inventing a new name,
    which is how the duration-deviant MMN tests express a 50 ms vs 100 ms
    standard.
    """
    key = str(name or "").strip().lower()
    if not is_generator(key):
        _resolve_overrides(key, {})                 # raises the explanatory KeyError
    overrides = {k: v for k, v in kw.items()}
    # `target_dbfs` is a *post-gain*, not a synthesis parameter: it exists so a
    # spec can build a loudness ladder (T607's levels_dbfs) out of one generator
    # without inventing a new name or re-normalising inside every primitive.
    target = overrides.pop("target_dbfs", None)
    fn, params = _resolve_overrides(key, dict(overrides))
    fs = int(params.pop("fs", 44100))
    ensure_dirs()
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    tag = ""
    if target is not None:
        overrides["target_dbfs"] = target
    if overrides:
        blob = ",".join(f"{k}={_safe_token(v)}" for k, v in sorted(overrides.items()))
        tag = "-" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
    elif fs != 44100:
        tag = "-" + hashlib.sha1(f"fs={fs}".encode("utf-8")).hexdigest()[:8]
    safe = _safe_token(key) or "stimulus"
    path = SYNTH_DIR / f"{safe}{tag}.wav"
    if path.exists() and not force:
        return path
    try:
        x = fn(fs=fs, **params)
    except (TypeError, ValueError) as exc:
        # A malformed *name* (tone_bogus, syllable_xq) is a naming error, not a
        # synthesis error: report it with the grammar so the spec author can fix
        # it, and do not leave a half-written file behind.
        raise KeyError(f"{_pattern_help(name)} [{type(exc).__name__}: {exc}]") from exc
    if target is not None:
        x = normalize_rms(x, float(target))
    return save_wav(path, x, fs=fs)


def generate_all(force: bool = False) -> dict[str, Path]:
    """Render every named generator (used by the smoke test and `nb stim synth`)."""
    started = time.time()
    out: dict[str, Path] = {}
    for name in NAMES:
        out[name] = generate(name, force=force)
    _ = time.time() - started
    return out


__all__ = [
    "NAMES", "SYNTH_DIR", "VOWEL_FORMANTS", "CONSONANTS", "ONSETS", "NUCLEI", "CODAS",
    "tone", "click", "click_train", "noise", "pink_noise", "chord", "melody",
    "syllable", "amplitude_modulate", "beat", "silence", "pseudo_speech",
    "normalize_rms", "save_wav", "generate", "generate_all", "is_generator",
]
