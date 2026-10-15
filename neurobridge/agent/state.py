"""Honest state estimation and artefact-command detection for ONE dry frontal channel.

Why this file is deliberately conservative
-------------------------------------------
The acquisition hardware is a NingNao/SereniBrain TH21A: **one dry electrode**
on the forehead (Fpz-ish, nose-bridge switch), 250 Hz, 24-bit *signed ADC
counts*, uncalibrated, drifting, and railed when the band is off the head.
A single frontal channel with no reference montage cannot be source-localised,
cannot be spatially filtered, and carries no lateralised information. The
honest ceiling of what can be derived is:

* **arousal / alertness**          - beta/alpha and beta power vs a personal baseline
* **gross attention**              - beta vs (theta+alpha) vs baseline
* **cognitive-load *proxy***       - frontal theta/beta vs baseline (a proxy for
  effort, not a measurement of "how hard you are thinking")
* **fatigue / drowsiness**         - (theta+delta)/beta drift, alpha slowing, time-on-task
* **relaxation**                   - relative alpha vs baseline
* **blink / jaw-clench artefacts** - the only two *commands* this montage
  supports with useful reliability, because they are enormous at Fpz:
  a blink is 1-2 orders of magnitude larger than cortical activity.

What is **not** possible and is therefore never claimed anywhere in this file:
reading words, sentences, images or intentions; multi-class "mental typing";
valence or any affective valence from frontal asymmetry (which requires at
least two symmetric channels - with one channel the asymmetry is undefined).
``StateEstimate.valence`` exists only so the wire schema is stable, and it is
always ``None``; see ``model_info['valence_note']``.

Every derived number here is *relative to the subject's own baseline*, because
the device is uncalibrated and the absolute counts differ between sessions,
electrode seating and even the same seating after 20 minutes of drift. Fields
are reported in 0..1 as monotone (not linear, not calibrated) mappings of a
robust z-score against that baseline. On day one, before any model has been
trained, those fields come from the *heuristic baseline mapping* documented
below; once a decoder exists for a state test, that model's probability output
replaces the heuristic for the matching field. ``model_info`` always says which
case applies, per field, so a consuming agent can never mistake a heuristic for
a trained decoder.

Threading model
---------------
``push()`` is called from the acquisition thread; ``estimate()`` is called from
HTTP handler threads. Both go through one ``threading.Lock``, and every lock
critical section is tiny (append / copy a window). ``estimate()`` never waits
on the device: it reads whatever the ring buffer holds, and reuses the cached
result when no new sample has arrived, which is what keeps it cheap enough to
call at 2 Hz from the bridge.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import MODEL_DIR, new_session_id
from ..device.base import Packet
from ..device.quality import SAT, QualityReport, assess

__all__ = ["StateEstimate", "StateEstimator", "IntentDetector", "BandPower"]

#: Relative band-power definitions (Hz). Identical to device/quality.BANDS so a
#: window scored here and a capture scored by the pipeline agree.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

#: Analysis band. 1 Hz lower edge: sub-1 Hz is electrode drift and has no
#: cortical meaning on a dry electrode; 45 Hz upper edge: above that is EMG
#: (which is exactly what we exploit for clench detection, elsewhere).
BAND_LO, BAND_HI = 1.0, 45.0

#: Default state-test -> field mapping. These are the catalog tests whose
#: ``agent_use`` is ``state``/``both`` (see docs/CATALOG_SPEC.md) and whose
#: class labels line up with a field here. A trained model for one of these
#: tests replaces the heuristic for exactly that field.
FIELD_TESTS: dict[str, tuple[str, ...]] = {
    "attention": ("T101",),                      # sustained attention (SART)
    "arousal": ("T104", "T102"),                 # audio arousal, drowsiness
    "load": ("T201", "T202", "T203", "T204", "T707"),
    "fatigue": ("T105", "T102", "T108"),
    "relaxation": ("T708",),
}

#: Substrings that identify the "high" end of a binary class list, per metric.
#: Ordered: the first class whose lowercased label contains one of these wins.
_HIGH_WORDS: dict[str, tuple[str, ...]] = {
    "attention": ("attend", "focused", "target", "alert", "high"),
    "arousal": ("excit", "alert", "high", "arous"),
    "load": ("load_high", "high", "hard", "incongruent", "dual", "arithmetic"),
    "fatigue": ("drows", "late", "tired", "fatigue", "missed", "low"),
    "relaxation": ("relax", "calm", "low"),
}
_LOW_WORDS: dict[str, tuple[str, ...]] = {
    "attention": ("lapse", "mind", "distract", "low"),
    "arousal": ("calm", "drows", "low", "relax"),
    "load": ("load_low", "low", "easy", "congruent", "single", "counting"),
    "fatigue": ("alert", "early", "rested", "responded"),
    "relaxation": ("effort", "tense", "high"),
}

#: ML feature window used when a trained model is loaded.
_N_FEATURE_STATS = ("std", "ptp", "mean", "skew", "kurt", "zcr")


# --------------------------------------------------------------------- helpers
def _robust_z(x: float, med: float, sd: float) -> float:
    """Robust z-score with a floor on the scale.

    A MAD of exactly zero happens on quiet synthetic data or a flatlined
    electrode; dividing by it would produce infinities that then flow into the
    reported fields, so the scale is floored at 1e-6 of the magnitude.
    """
    scale = max(float(sd), abs(float(med)) * 1e-3, 1e-6)
    return (float(x) - float(med)) / scale


def _sigmoid(z: float) -> float:
    """Logistic squash of a z-score into 0..1 (0.5 == at the baseline)."""
    if z >= 30:
        return 1.0
    if z <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-float(z)))


#: Squash gain used only for the low-sample fallback path (see _squash_mean).
#: 0.6 is deliberately *not* 1.0: one MAD-sized deviation should move a field by
#: ~0.15, not ~0.5, or ordinary single-window noise would pin fields at 0/1 and
#: the agent would receive a saturated signal that carries no gradient.
_Z_GAIN = 0.6

#: Field value at the edge of the baseline range: the percentile is mapped into
#: 0.15..0.85, leaving 0.15 of headroom on each side for "outside your baseline".
#: A hard 0/1 at the edge would tell the agent the user is certainly at a maximum
#: when the honest statement is only "outside the range 60 s of calibration saw".
_PCT_PULL = 0.7
#: Share of the 0.15 top/bottom band recovered from the robust z-score when the
#: value is beyond the baseline range. Percentile alone cannot order two
#: conditions that are both above every baseline window, but the z-score can.
_TAIL_RECOVER = 0.5


def _pct_stretch(p: float, z: float = 0.0) -> float:
    """Map a baseline percentile (and, past the edge, a z-score) to 0..1.

    Monotone with 0.5 at the baseline median. The percentile sets the interior,
    which is scale-free: a metric whose baseline spread happened to be tiny
    cannot pin the field at 0/1 the way a raw z-score does. Beyond the baseline
    range there is no percentile left to rank with, so the robust z-score adds a
    bounded tail term (``tanh``, i.e. < 1 for any finite z). That keeps the field
    inside 0.15..0.85 for the interior and 0.01..0.99 in the tails, so a value far
    outside the baseline still reads differently from one slightly outside it.
    """
    p = min(1.0, max(0.0, float(p)))
    base = 0.5 + _PCT_PULL * (p - 0.5)
    if p >= 1.0 or p <= 0.0:
        tail = math.tanh(abs(float(z)) / 4.0)
        sign = 1.0 if p >= 1.0 else -1.0
        return base + sign * 0.15 * _TAIL_RECOVER * tail
    return base


def _squash_mean(zs: Sequence[float], gain: float = _Z_GAIN) -> float:
    """Average several z-scores in probability space (low-sample fallback).

    Used only when a baseline has too few windows for a percentile distribution.
    Averaging in probability space rather than in z-space keeps a ratio kernel
    and an absolute-power kernel on comparable footing.
    """
    if not zs:
        return 0.5
    return float(np.mean([_sigmoid(gain * float(z)) for z in zs]))


class _Ring:
    """Growable ring buffer of (value, t_mono) with monotonic timestamps.

    ``push`` is O(1) amortised. Windows are copied out under the caller's lock,
    so ``estimate()`` can never block the acquisition thread for longer than a
    memcpy. Values are stored float64 because the raw counts are 24-bit and a
    float32 accumulator would lose the low bits of a slow drift.
    """

    def __init__(self, capacity: int = 250 * 60 * 30) -> None:
        self._v = np.zeros(int(capacity), dtype=np.float64)
        self._t = np.zeros(int(capacity), dtype=np.float64)
        self._cap = int(capacity)
        self._n = 0          # valid samples
        self._w = 0          # write cursor
        self._total = 0      # samples ever pushed (never decreases)

    @property
    def size(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def total(self) -> int:
        return self._total

    @property
    def t_latest(self) -> float:
        if self._n == 0:
            return 0.0
        return float(self._t[(self._w - 1) % self._cap])

    @property
    def t_earliest(self) -> float:
        if self._n == 0:
            return 0.0
        return float(self._t[(self._w - self._n) % self._cap])

    @property
    def t_oldest(self) -> float:
        """Alias of :attr:`t_earliest` (kept for readable call sites)."""
        return self.t_earliest

    def clear(self) -> None:
        self._n = 0
        self._w = 0
        self._total = 0

    def push(self, values: np.ndarray, times: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        if v.size != t.size:
            raise ValueError(f"values {v.shape} and times {t.shape} must have equal length")
        n = v.size
        if n == 0:
            return t
        # keep timestamps monotonic even if a caller hands us a packet whose
        # estimated per-sample times overlap the previous packet (BLE jitter)
        if self._n and t[0] < self.t_latest:
            step = self._median_step()
            t = t - t[0] + (self.t_latest + step)
        if n >= self._cap:                       # a single huge push: keep the tail
            v = v[-self._cap:]
            t = t[-self._cap:]
            n = v.size
        end = self._w + n
        if end <= self._cap:
            self._v[self._w:end] = v
            self._t[self._w:end] = t
        else:
            first = self._cap - self._w
            self._v[self._w:] = v[:first]
            self._t[self._w:] = t[:first]
            self._v[: end - self._cap] = v[first:]
            self._t[: end - self._cap] = t[first:]
        self._w = end % self._cap
        self._n = min(self._cap, self._n + n)
        self._total += n
        return t

    def _median_step(self) -> float:
        """Median sample interval, used only to nudge overlapping packets."""
        if self._n < 2:
            return 1.0 / 250.0
        k = min(200, self._n)
        t = self.window(self._n, copy=True)[1][-k:]
        d = np.diff(t)
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0 / 250.0

    def window(self, n: int, *, copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Last ``n`` samples in chronological order."""
        n = int(min(max(n, 0), self._n))
        if n == 0:
            z = np.zeros(0, dtype=np.float64)
            return z, z.copy()
        start = (self._w - n) % self._cap
        if start + n <= self._cap:
            v, t = self._v[start:start + n], self._t[start:start + n]
        else:
            first = self._cap - start
            v = np.concatenate((self._v[start:], self._v[: n - first]))
            t = np.concatenate((self._t[start:], self._t[: n - first]))
        if copy:
            return v.copy(), t.copy()
        return v, t

    def window_between(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """All buffered samples with ``t0 <= t <= t1`` (chronological)."""
        v, t = self.window(self._n, copy=False)
        if t.size == 0:
            return v.copy(), t.copy()
        m = (t >= float(t0)) & (t <= float(t1))
        return v[m].copy(), t[m].copy()


# ------------------------------------------------------------------ band power
@dataclass
class BandPower:
    """Absolute + relative band power of one window, computed in one place.

    Kept as a tiny value object so the heuristic path and the feature-extraction
    path can never disagree about what "relative alpha" means.
    """

    abs_power: dict[str, float]
    rel_power: dict[str, float]
    total_power: float
    freqs: np.ndarray
    psd: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "abs_power": {k: round(v, 6) for k, v in self.abs_power.items()},
            "rel_power": {k: round(v, 6) for k, v in self.rel_power.items()},
            "total_power_1_45": round(self.total_power, 6),
        }


def compute_band_power(values: np.ndarray, fs: float) -> BandPower:
    """Relative/absolute band power via a Hann-windowed periodogram.

    A Welch-style periodogram would be marginally more stable, but a state
    update every 500 ms on a 4 s window is already the average of ~4 s of data
    and the extra FFTs are not worth the latency. Sub-1 Hz energy is removed
    before the FFT because dry-electrode drift there is 10-100x the cortical
    signal and would otherwise dominate every relative power.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v - float(np.mean(v)) if v.size else v
    n = v.size
    if n < 16 or fs <= 0:
        z = {b: 0.0 for b in BANDS}
        return BandPower(z, dict(z), 0.0, np.zeros(0), np.zeros(0))
    # linear detrend removes the residual drift ramp
    x = v - np.linspace(v[0], v[-1], n)
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    fr = np.fft.rfftfreq(n, d=1.0 / float(fs))
    # scale so the power is in counts^2 (window-energy normalisation)
    psd = (np.abs(spec) ** 2) / (float(fs) * float(np.sum(w ** 2)) + 1e-12)
    inband = (fr >= BAND_LO) & (fr <= min(BAND_HI, fs / 2.0 - 1e-9))
    total = float(psd[inband].sum())
    abs_p: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        m = (fr >= lo) & (fr < min(hi, fs / 2.0 - 1e-9))
        abs_p[name] = float(psd[m].sum())
    rel = {k: (v_ / total if total > 0 else 0.0) for k, v_ in abs_p.items()}
    return BandPower(abs_p, rel, total, fr, psd)


def _alpha_peak_freq(bp: BandPower, lo: float = 7.0, hi: float = 13.5) -> float:
    """Dominant frequency inside the alpha band, 0.0 when no alpha is present."""
    if bp.freqs.size == 0:
        return 0.0
    m = (bp.freqs >= lo) & (bp.freqs <= hi)
    if not np.any(m):
        return 0.0
    return float(bp.freqs[m][int(np.argmax(bp.psd[m]))])


def _artifact_metrics(v: np.ndarray, fs: float) -> dict[str, float]:
    """Saturation, motion jumps and mains share - the three kill criteria.

    Deliberately the same quantities as ``device.quality.assess`` so the bridge
    and the offline pipeline agree on why a window was rejected.
    """
    out = {"saturation_pct": 0.0, "motion_jump_pct": 0.0, "mains_pct": 0.0,
           "flatline_pct": 0.0, "ptp": 0.0}
    if v.size < 4:
        return out
    out["saturation_pct"] = 100.0 * float(np.mean(np.abs(v) >= SAT - 2))
    out["ptp"] = float(np.ptp(v))
    d1 = np.abs(np.diff(v))
    med = float(np.median(d1))
    if med <= 0:
        out["flatline_pct"] = 100.0
        return out
    out["motion_jump_pct"] = 100.0 * float(np.mean(d1 > 20.0 * med))
    x = v - float(np.mean(v))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    tot = float(spec[(fr >= BAND_LO) & (fr <= min(BAND_HI, fs / 2.0))].sum()) or 1.0
    mains = 0.0
    for f0 in (50.0, 60.0):
        if f0 < fs / 2.0:
            m = (fr >= f0 - 1.0) & (fr <= f0 + 1.0)
            mains = max(mains, 100.0 * float(spec[m].sum()) / tot)
    out["mains_pct"] = mains
    return out


# --------------------------------------------------------------- state object
@dataclass
class StateEstimate:
    """One honest snapshot of what a single frontal channel currently shows.

    All six 0..1 fields are *relative* signals against the subject's own
    session baseline. They are monotone but not linear, not calibrated, and not
    comparable between subjects or sessions. ``valence`` is always ``None``:
    frontal-asymmetry valence needs two symmetric channels, so with one
    electrode it is mathematically unavailable and is never faked.
    """

    t_mono: float
    t_wall: str
    attention: float = 0.5
    arousal: float = 0.5
    load: float = 0.5
    fatigue: float = 0.5
    relaxation: float = 0.5
    valence: float | None = None
    artifact_level: float = 0.0
    contact: str = "unknown"
    confidence: float = 0.0
    source: str = "fake"
    intents: list[dict] = field(default_factory=list)
    model_info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-safe view (numpy scalars are not serialisable by stdlib json)."""
        def _f(x: Any) -> float:
            return float(x) if x is not None else 0.0
        return {
            "t_mono": float(self.t_mono),
            "t_wall": str(self.t_wall),
            "attention": _f(self.attention),
            "arousal": _f(self.arousal),
            "load": _f(self.load),
            "fatigue": _f(self.fatigue),
            "relaxation": _f(self.relaxation),
            "valence": None if self.valence is None else float(self.valence),
            "artifact_level": _f(self.artifact_level),
            "contact": str(self.contact),
            "confidence": _f(self.confidence),
            "source": str(self.source),
            "intents": [dict(i) for i in (self.intents or [])],
            "model_info": dict(self.model_info or {}),
        }

    def __repr__(self) -> str:                       # pragma: no cover - debug aid
        return (f"<StateEstimate att={self.attention:.2f} aro={self.arousal:.2f} "
                f"load={self.load:.2f} fat={self.fatigue:.2f} rel={self.relaxation:.2f} "
                f"conf={self.confidence:.2f} contact={self.contact}>")


# ------------------------------------------------------------------- estimator
class StateEstimator:
    """Rolling-window state estimator for one dry frontal channel.

    Field definitions (all 0..1, all computed from *relative* band power against
    the personal baseline captured at session start -- the device is
    uncalibrated, so absolute counts mean nothing):

    ``attention``
        Monotone in ``beta / (theta + alpha)`` relative to baseline. This tracks
        gross engagement/alertness, not "what you are attending to".
    ``arousal``
        Monotone in ``beta / alpha`` and in mean beta power relative to baseline.
    ``load``
        Monotone in frontal ``theta / beta`` relative to baseline. An effort
        **proxy** only: it indexes how much frontal theta-band activity has
        risen relative to the subject's own resting pattern.
    ``fatigue``
        Monotone in ``(theta + delta) / beta`` drift, in alpha *slowing* (peak
        frequency dropping), and in time-on-task since the session start.
    ``relaxation``
        Monotone in relative alpha (8-13 Hz share of 1-45 Hz) vs baseline.
    ``valence``
        **Always ``None``.** Frontal-asymmetry valence is impossible with one
        channel; the field exists only so the JSON schema is stable.
    ``confidence``
        0..1 combination of signal quality (device self-check + saturation +
        drift), artefact level (motion jumps, mains) and the distance of the
        current window from the baseline distribution (Ledoit-Wolf shrinkage
        Mahalanobis). It is forced below 0.2 whenever contact is bad or the
        capture is saturated, because in that state the numbers above are
        meaningless and an agent must not act on them.

    Heuristic vs trained
    --------------------
    On day one no model exists, so the fields above are produced by the
    documented heuristic baseline mapping. When ``data/models/<TEST_ID>.joblib``
    exists for a state test, ``pipeline.decoders.load_model`` is used and the
    model's probability output replaces the heuristic for the matching field.
    ``model_info`` reports, per field, ``{"derivation": "trained_model"}`` or
    ``{"derivation": "heuristic_baseline"}`` and never overstates what is known.

    The estimator is import-safe without hardware and without the pipeline
    package: every optional dependency is imported lazily and defensively.
    """

    def __init__(self, fs: float = 250.0, window_s: float = 4.0, step_s: float = 0.5,
                 models_dir: Path | str | None = None, device: str = "auto",
                 replay_session: str | Path | None = None) -> None:
        self.fs = float(fs)
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        #: first 60 s of the session become the personal baseline (median+MAD);
        #: 60 s is long enough for >=10 independent 4 s windows and short enough
        #: that a subject will actually sit still for it.
        self._baseline_window_s = 60.0
        #: start-up transient deliberately excluded from the baseline (filter
        #: warm-up, band still being seated); see :meth:`calibrate`.
        self._baseline_skip_s = 3.0
        self.models_dir = Path(models_dir) if models_dir is not None else Path(MODEL_DIR)
        self.device = str(device)
        #: The device *actually* producing samples, set by whoever opens the
        #: source (the bridge feeder, or `nb state`). ``self.device`` is only the
        #: *requested* alias ("auto"), and publishing an alias as the source is
        #: how a synthetic capture gets described to an agent as a person's brain.
        #: None until the owner resolves it; see :attr:`source_label`.
        self.source_name: str | None = None
        #: The device's own electrode self-check ("good (4/4 electrodes OK)" or
        #: "no electrodes reporting OK"), supplied by whoever opened the source.
        #: It is the only evidence about *seating* rather than about the signal,
        #: and protocol.md promises it in the ``contact`` field, so it has to
        #: reach :func:`assess` - an off-head capture then caps its own confidence
        #: instead of being averaged in as if it were cortex.
        self.device_contact: str = "unknown"
        self.replay_session = replay_session
        self.session_id = new_session_id("agent")
        self.started_mono = time.perf_counter()

        #: 30 minutes of context: enough for session-scale fatigue drift without
        #: letting a long-running bridge grow unbounded.
        self._ring = _Ring(capacity=max(int(self.fs * 60 * 30), 250 * 60))
        self._lock = threading.Lock()
        self._cache: StateEstimate | None = None
        self._cache_total = -1
        self._baseline: dict[str, Any] | None = None
        self._baseline_path: Path | None = None
        self._baseline_from_disk = False
        self._calibrated_at: float | None = None
        self._warnings: list[str] = []
        self._intents: list[dict] = []                # fed by the bridge/IntentDetector
        self._last_metrics: dict[str, float] = {}
        #: EWMA state for the field-driving metrics (see _estimate_from)
        self._smooth_metrics: dict[str, float] = {}
        self._ewma_alpha = 0.35
        self._n_windows = 0
        self._model_errors: list[str] = []
        self._models: dict[str, Any] = {}             # field -> TrainedModel
        self._model_meta: dict[str, dict[str, Any]] = {}
        self._quality: QualityReport | None = None
        self._load_models()
        self._load_baseline_for_session()

    # ------------------------------------------------------------- ingestion
    def push(self, values: Sequence[float] | np.ndarray,
             t_mono: Sequence[float] | np.ndarray | float | None = None) -> np.ndarray:
        """Append raw ADC counts with their monotonic timestamps.

        ``t_mono`` may be a scalar (applied to all samples), an array of the
        same length, or ``None``, in which case a nominal 1/fs grid ending at
        now is generated -- the same convention as ``device.base.Packet``.

        Returns the timestamps **as stored**, which is not always what was passed
        in: a packet whose estimated times overlap the previous one gets nudged
        forward (see ``_Ring.push``). A recorder must use the return value, or its
        CSV timeline drifts from the timeline the estimator actually analysed.
        """
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return np.zeros(0, dtype=np.float64)
        if t_mono is None:
            t = time.perf_counter() + (np.arange(v.size) - (v.size - 1)) / self.fs
        elif np.isscalar(t_mono):
            t = np.full(v.size, float(t_mono), dtype=np.float64)
        else:
            t = np.asarray(t_mono, dtype=np.float64).reshape(-1)
            if t.size != v.size:
                raise ValueError(f"values {v.shape} and times {t.shape} must match")
        with self._lock:
            return self._ring.push(v, t)

    def push_packets(self, packets: Iterable[Packet]) -> None:
        """Append whole :class:`device.base.Packet` objects.

        Uses ``packet.t_samples`` when present (per-sample estimated times),
        otherwise falls back to the packet's arrival time so a packet is never
        silently dropped just because a driver omitted the per-sample grid.
        """
        for p in packets or ():
            vals = np.asarray(getattr(p, "values", []), dtype=np.float64).reshape(-1)
            if vals.size == 0:
                continue
            ts = getattr(p, "t_samples", None)
            if ts is None:
                ts = float(getattr(p, "t_mono", time.perf_counter()))
            self.push(vals, ts)

    def note_intents(self, intents: Iterable[dict]) -> None:
        """Record recently detected artefact commands so they ride along in state."""
        with self._lock:
            for i in intents or ():
                self._intents.append(dict(i))
            if len(self._intents) > 200:
                self._intents = self._intents[-200:]

    def reset(self) -> None:
        """Forget the buffer and the in-memory baseline (keeps models loaded)."""
        with self._lock:
            self._ring.clear()
            self._cache = None
            self._cache_total = -1
            self._baseline = None
            self._baseline_from_disk = False
            self._calibrated_at = None
            self._intents = []
            self._n_windows = 0
            self._last_metrics = {}
            self._smooth_metrics = {}
            self._quality = None

    # ------------------------------------------------------------- properties
    @property
    def ready(self) -> bool:
        """True when a state can be reported at all.

        Readiness deliberately does *not* require calibration: the bridge must
        answer ``/state`` within seconds of start-up (with low confidence)
        rather than withholding data. ``confidence`` is what carries the
        "do not act on this yet" information.
        """
        return self._ring.size >= max(64, int(self.fs * min(self.window_s, 1.0)))

    @property
    def n_samples(self) -> int:
        return self._ring.size

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> dict[str, Any] | None:
        return self._baseline

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def quality(self) -> QualityReport | None:
        return self._quality

    @property
    def effective_fs(self) -> float:
        """Measured sampling rate of the buffered data (0.0 when unknown)."""
        v, t = self._ring.window(min(self._ring.size, int(self.fs * 10)), copy=False)
        if t.size < 8:
            return 0.0
        d = np.diff(t)
        d = d[d > 0]
        if d.size < 4:
            return 0.0
        med = float(np.median(d))
        return 1.0 / med if med > 0 else 0.0

    @property
    def field_derivations(self) -> dict[str, str]:
        """field -> 'trained_model' | 'heuristic_baseline' (for /schema)."""
        out = {}
        for f in ("attention", "arousal", "load", "fatigue", "relaxation"):
            out[f] = "trained_model" if f in self._models else "heuristic_baseline"
        out["valence"] = "unavailable"
        return out

    # ------------------------------------------------------------- calibration
    def calibrate(self, force: bool = False) -> dict:
        """Capture the first ~60 s as the personal baseline (median + MAD).

        Robust statistics (median/MAD, not mean/std) because a single blink or
        jaw clench inside the calibration minute would otherwise define the
        baseline for the whole session. The baseline is persisted to
        ``data/models/baseline_<session>.json`` so a bridge restart, or the
        offline pipeline, can reload exactly what the fields were relative to.

        The first ``baseline_skip_s`` seconds are discarded: whether the samples
        come from an IIR anti-alias filter switching on or from the operator
        still seating the band, that segment is a start-up transient and letting
        it define the baseline biases every later field.
        """
        with self._lock:
            if self._baseline is not None and not force:
                return self._baseline_summary("already_calibrated")
            t0 = self._ring.t_earliest + self._baseline_skip_s
            t1 = self._ring.t_latest
            if (t1 - t0) > self.baseline_window_s:
                t1 = t0 + self.baseline_window_s
            v, t = self._ring.window_between(t0, t1)
            if v.size < int(min(self.window_s, 1.0) * self.fs):
                return {"ok": False, "reason": "not_enough_data",
                        "n_samples": int(v.size), "need_s": self.window_s,
                        "skipped_s": self._baseline_skip_s}
            base = self._build_baseline(v, t)

        if base is None:
            return {"ok": False, "reason": "no_usable_calibration_windows",
                    "n_samples": int(v.size)}
        path = self._save_baseline(base)
        with self._lock:
            self._baseline = base
            self._baseline_from_disk = False
            self._calibrated_at = time.perf_counter()
            self._baseline_path = path
            self._cache = None                 # fields change once calibrated
            self._cache_total = -1
        return self._baseline_summary("fresh")

    @property
    def baseline_window_s(self) -> float:
        """Seconds of session start used as the personal baseline."""
        return self._baseline_window_s

    def _build_baseline(self, v: np.ndarray, t: np.ndarray) -> dict[str, Any] | None:
        """Robust baseline stats from up to 60 s of contiguous data.

        Two deliberate choices, both of which were bugs first:

        1. **Windows are end-to-end at ``window_s`` spacing (no overlap).** With
           a 0.5 s step and a 4 s window the samples are 88% shared, so the MAD
           measures almost no variance and every later z-score is inflated until
           the 0..1 fields pin at 0 or 1.
        2. **The published statistics describe the *smoothed* series, not the raw
           windows.** Runtime fields are driven by an EWMA of the metrics (see
           :meth:`_estimate_from`); comparing that low-variance series against a
           raw-window MAD is the same inflation bug one level up. The raw-window
           median/MAD are stored too (``raw_median``/``raw_mad``/``spread``) so a
           `_zscore` scale can be floored by the natural window-to-window spread.

        A short calibration cannot estimate a spread; it is marked
        ``low_sample`` and callers (and ``/health``) can see it is weak.
        """
        win = int(self.window_s * self.fs)
        rows: list[dict[str, float]] = []
        for i in range(0, max(1, v.size - win + 1), win):
            seg = v[i:i + win]
            if seg.size < win:
                break
            rows.append(self._window_metrics(seg))
        if not rows:
            return None
        keys = sorted(rows[0].keys())

        def _series(rs: list[dict[str, float]], k: str) -> np.ndarray:
            return np.asarray([r[k] for r in rs], dtype=float)

        def _med_mad(x: np.ndarray) -> tuple[float, float]:
            m = float(np.median(x))
            return m, float(np.median(np.abs(x - m)))

        # raw-window statistics: used only as a scale floor / anomaly reference
        raw_med = {k: _med_mad(_series(rows, k))[0] for k in keys}
        raw_mad = {k: _med_mad(_series(rows, k))[1] for k in keys}
        spread = {k: float(np.std(_series(rows, k))) for k in keys}

        # EWMA series exactly as the runtime smoother produces it
        smooth_rows: list[dict[str, float]] = []
        state: dict[str, float] = {}
        a = self._ewma_alpha
        for r in rows:
            if not state:
                state = dict(r)
            else:
                for k, val in r.items():
                    state[k] = (1.0 - a) * state.get(k, val) + a * val
            smooth_rows.append(dict(state))
        # the first few EWMA samples are still converging; keep them for the
        # median but compute the spread from the converged tail when possible
        tail = smooth_rows[2:] if len(smooth_rows) > 4 else smooth_rows
        med = {k: _med_mad(_series(tail, k))[0] for k in keys}
        mad = {k: _med_mad(_series(tail, k))[1] for k in keys}

        base: dict[str, Any] = {
            "session_id": self.session_id,
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fs": self.fs,
            "n_windows": len(rows),
            "low_sample": len(rows) < 5,
            "window_s": self.window_s,
            "ewma_alpha": a,
            "non_overlapping": True,
            "n_samples": int(v.size),
            "median": med,
            "mad": mad,
            "raw_median": raw_med,
            "raw_mad": raw_mad,
            "spread": spread,
            "metrics": rows[0],
            "matrix": [[r[k] for k in keys] for r in rows],
            "smooth_matrix": [[r[k] for k in keys] for r in smooth_rows],
            "keys": keys,
            "contact": (self._quality.contact if self._quality else "unknown"),
            "usable": bool(self._quality.usable) if self._quality else True,
            "note": ("Relative baseline only: the device is uncalibrated, so these "
                     "are the subject's own counts, not physical units."),
        }
        # shrinkage covariance for the confidence distance; optional
        try:
            from sklearn.covariance import LedoitWolf
            X = np.asarray(base["matrix"], dtype=float)
            if X.shape[0] >= 2 and np.all(np.isfinite(X)):
                base["cov"] = LedoitWolf().fit(X).covariance_.tolist()
        except Exception as exc:                      # pragma: no cover - optional
            self._warnings.append(f"baseline covariance unavailable: {exc}")
        return base

    def _save_baseline(self, base: dict[str, Any]) -> Path:
        path = Path(self.models_dir) / f"baseline_{self.session_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(base, indent=2, default=float))
        except Exception as exc:                      # pragma: no cover - disk
            self._warnings.append(f"could not persist baseline: {exc}")
        return path

    def load_baseline(self, path: str | Path) -> dict:
        """Reload a persisted baseline (median/MAD + optional covariance)."""
        p = Path(path)
        try:
            base = json.loads(p.read_text())
        except Exception as exc:
            self._warnings.append(f"could not load baseline {p}: {exc}")
            return {"ok": False, "reason": "unreadable", "path": str(p)}
        with self._lock:
            self._baseline = base
            self._baseline_from_disk = True
            self._baseline_path = p
            self._cache = None
            self._cache_total = -1
        return self._baseline_summary("loaded_from_disk")

    def _load_baseline_for_session(self) -> None:
        """Pick up ``data/models/baseline_<session>.json`` when it exists.

        A bridge that restarts mid-session should not silently reset the fields
        to un-baselined 0.5s, so an existing file for this restart's session id
        is adopted. Baselines from *other* sessions are deliberately NOT adopted:
        a baseline from another electrode seating is not valid for this one, and
        silently importing it would make every field a comparison against a
        different measurement session.
        """
        p = Path(self.models_dir) / f"baseline_{self.session_id}.json"
        if p.exists():
            self.load_baseline(p)

    def _baseline_summary(self, status: str) -> dict:
        b = self._baseline or {}
        return {
            "ok": bool(b),
            "status": status,
            "session_id": self.session_id,
            "n_windows": int(b.get("n_windows", 0)),
            "n_samples": int(b.get("n_samples", 0)),
            "median": b.get("median", {}),
            "mad": b.get("mad", {}),
            "path": str(self._baseline_path) if self._baseline_path else None,
            "loaded_from_disk": bool(self._baseline_from_disk),
            "note": b.get("note", ""),
        }

    # ------------------------------------------------------------- estimation
    def estimate(self) -> StateEstimate:
        """Compute (or reuse) the current state. Safe to call from any thread.

        Cheap by construction: when no new sample has arrived since the last
        call the cached estimate is returned immediately, so the bridge can be
        polled at 2 Hz while the acquisition thread runs at 250 Hz.
        """
        total = self._ring.total
        with self._lock:
            if self._cache is not None and total == self._cache_total:
                return self._cache
        now = time.perf_counter()
        with self._lock:
            win = int(max(self.window_s * self.fs, 64))
            v, t = self._ring.window(win)
            intents = list(self._intents[-5:])
        est = self._estimate_from(v, t, now, intents)
        with self._lock:
            self._cache = est
            self._cache_total = total
        return est

    @property
    def source_label(self) -> str:
        """The source to publish in a state estimate: the real one, or "unknown".

        Deliberately does **not** fall back to ``self.device``. That attribute is
        the alias the operator typed, and reporting it as the source is exactly
        the failure the field exists to prevent: ``"auto"`` is not a device, so a
        consumer cannot tell a synthetic fallback from a headband. Before a source
        is open the honest answer is "unknown", and the bridge treats anything
        that is not a real headband as coarse rather than verified.
        """
        return str(self.source_name or "unknown")

    def _estimate_from(self, v: np.ndarray, t: np.ndarray, now: float,
                       intents: list[dict]) -> StateEstimate:
        if v.size < 64:
            # Say "no_data" *per field*. Passing no derivation would let
            # _model_info default every field to "heuristic_baseline", i.e. claim
            # that baseline-relative heuristics exist when there is not even one
            # window of data - which a consumer reasonably reads as "0.5 means at
            # your baseline".
            return StateEstimate(
                t_mono=now, t_wall=time.strftime("%Y-%m-%dT%H:%M:%S"),
                contact="unknown", confidence=0.0, source=self.source_label,
                intents=intents,
                model_info=self._model_info(
                    "no_data", {},
                    derivation={f: "no_data" for f in
                                ("attention", "arousal", "load", "fatigue",
                                 "relaxation")}),
            )
        q = assess(v, self.fs, contact=self.device_contact) if v.size >= self.fs else None
        if q is not None:
            self._quality = q
        contact = getattr(q, "contact", "unknown") if q else "unknown"
        if self.source_label == "fake":
            contact = "synthetic (perfect contact)"

        metrics = self._window_metrics(v)
        self._last_metrics = metrics
        self._n_windows += 1
        # Light EWMA smoothing for the fields only. Band ratios of a 4 s window
        # still fluctuate by a MAD or two, and a raw per-window z-score would
        # make the 0..1 fields jump 0.2 between two updates 0.5 s apart; the
        # agent is told to act on trends, so the trend is what gets published.
        # Raw metrics stay available in model_info['metrics'] for debugging.
        if not self._smooth_metrics:
            self._smooth_metrics = dict(metrics)
        else:
            a = self._ewma_alpha
            for k, val in metrics.items():
                prev = self._smooth_metrics.get(k, val)
                self._smooth_metrics[k] = (1.0 - a) * prev + a * val
        smoothed = dict(self._smooth_metrics)
        base = self._baseline
        fields: dict[str, float] = {}
        derivation: dict[str, str] = {}
        if base:
            z = self._zscore(smoothed, base)
            fields = self._heuristic_fields(z, now)
            derivation = {k: "heuristic_baseline" for k in fields}
        else:
            z = {}
            # un-baselined: report the neutral midpoint, never a fake number
            fields = {k: 0.5 for k in ("attention", "arousal", "load",
                                       "fatigue", "relaxation")}
            derivation = {k: "uncalibrated_neutral" for k in fields}

        # trained models win over the heuristic, per field, when loadable
        for fname, model in self._models.items():
            p = self._model_probability(model, v, t)
            if p is None:
                continue
            fields[fname] = p
            derivation[fname] = "trained_model"

        confidence = self._confidence(metrics, z, q)
        # never report a state confidently from a saturated/railed capture
        sat = metrics.get("saturation_pct", 0.0)
        if q is not None and not q.usable:
            confidence = min(confidence, 0.19)
        if sat > 0.05 or contact.startswith(("no electrodes", "poor")):
            confidence = min(confidence, 0.19)

        return StateEstimate(
            t_mono=now,
            t_wall=time.strftime("%Y-%m-%dT%H:%M:%S"),
            attention=float(fields.get("attention", 0.5)),
            arousal=float(fields.get("arousal", 0.5)),
            load=float(fields.get("load", 0.5)),
            fatigue=float(fields.get("fatigue", 0.5)),
            relaxation=float(fields.get("relaxation", 0.5)),
            valence=None,
            artifact_level=self._artifact_level(metrics),
            contact=str(contact),
            confidence=float(confidence),
            source=self.source_label,
            intents=intents,
            model_info=self._model_info(
                "ok" if base else "uncalibrated",
                z, derivation=derivation, metrics=metrics, quality=q),
        )

    # ------------------------------------------------------------- metrics
    def _window_metrics(self, v: np.ndarray) -> dict[str, float]:
        """The complete feature set derived from one window of raw counts."""
        bp = compute_band_power(v, self.fs)
        rel = bp.rel_power
        ab = bp.abs_power
        eps = 1e-9
        theta_beta = float(np.log10((ab["theta"] + eps) / (ab["beta"] + eps)))
        beta_alpha = float(np.log10((ab["beta"] + eps) / (ab["alpha"] + eps)))
        # attention ratio, log-scaled so a z-score against the log is symmetric
        att_ratio = float(np.log10((ab["beta"] + eps) / (ab["theta"] + ab["alpha"] + eps)))
        slow_fast = float(np.log10((ab["theta"] + ab["delta"] + eps) / (ab["beta"] + eps)))
        art = _artifact_metrics(v, self.fs)
        m: dict[str, float] = {
            "att_ratio": att_ratio,
            "beta_alpha": beta_alpha,
            "theta_beta": theta_beta,
            "log_beta_abs": float(np.log10(ab["beta"] + eps)),
            "log_alpha_abs": float(np.log10(ab["alpha"] + eps)),
            "log_total": float(np.log10(bp.total_power + eps)),
            "rel_beta": float(rel["beta"]),
            "rel_alpha": float(rel["alpha"]),
            "rel_theta": float(rel["theta"]),
            "rel_delta": float(rel["delta"]),
            "rel_gamma": float(rel["gamma"]),
            "slow_fast": slow_fast,
            "alpha_peak_hz": _alpha_peak_freq(bp),
            "saturation_pct": float(art["saturation_pct"]),
            "motion_jump_pct": float(art["motion_jump_pct"]),
            "mains_pct": float(art["mains_pct"]),
            "raw_ptp": float(art["ptp"]),
            "std": float(np.std(v)),
        }
        return m

    def _zscore(self, metrics: dict[str, float], base: dict[str, Any]) -> dict[str, float]:
        """Robust z-score of each metric against the personal baseline.

        The scale is ``max(MAD_of_smoothed_series, 0.5 * spread)`` where
        ``spread`` is the standard deviation of the *raw* non-overlapping
        baseline windows. The floor exists because a quiet 60 s calibration can
        give a MAD near zero, and dividing by it turns a single ordinary window
        into a z of 10 and pins the published field at 0/1. Half the raw spread
        is the smallest scale at which "this window differs from baseline" still
        means something.
        """
        med = base.get("median") or {}
        mad = base.get("mad") or {}
        spread = base.get("spread") or {}
        z: dict[str, float] = {}
        for k, val in metrics.items():
            if k not in med:
                continue
            scale = max(float(mad.get(k, 0.0)), 0.5 * float(spread.get(k, 0.0)))
            z[k] = _robust_z(val, float(med[k]), scale)
        return z

    def _pct(self, kernel: str, value: float) -> float | None:
        """Percentile of ``value`` inside the personal baseline distribution.

        Returns ``None`` when the baseline has too few windows for an empirical
        CDF to mean anything (< 5), in which case the caller falls back to the
        z-score sigmoid. The comparison series is the same EWMA series that the
        live estimate produces, so the two are directly comparable.
        """
        base = self._baseline or {}
        keys = base.get("keys") or []
        mat = base.get("smooth_matrix")
        if not mat or kernel not in keys or len(mat) < 5:
            return None
        i = keys.index(kernel)
        col = [float(row[i]) for row in mat if len(row) > i]
        if not col:
            return None
        n = len(col)
        less = sum(1 for x in col if x < value)
        equal = sum(1 for x in col if x == value)
        return (less + 0.5 * equal) / n

    def _heuristic_fields(self, z: dict[str, float], now: float) -> dict[str, float]:
        """The documented day-one mapping: metrics -> personal-baseline percentile.

        Each field is a monotone (not linear) function of the stated ratio,
        measured as a percentile inside the subject's own baseline distribution
        and stretched into 0..1 so that 0.5 means "at your baseline". A percentile
        rather than a raw z-score because a quiet calibration can give a near-zero
        MAD, and dividing by it would pin every field at 0 or 1 the moment the
        subject does anything at all. When the baseline has too few windows for a
        percentile, the documented z-score sigmoid is used instead.

        Nothing here is trained and nothing here is a measurement; see
        ``model_info`` for the per-field disclosure.
        """

        def field(kernel: str, sign: float = 1.0) -> float:
            """0..1 field for one metric kernel ('-' inverts the direction)."""
            p = self._pct(kernel, self._smooth_metrics.get(kernel, float("nan")))
            zz = sign * z.get(kernel, 0.0)
            if p is not None and np.isfinite(p):
                p = p if sign > 0 else 1.0 - p
                return _pct_stretch(p, zz)
            # too few baseline windows for an empirical CDF: z-score fallback
            return _sigmoid(_Z_GAIN * zz)

        def combo(kernels: Sequence[tuple[str, float]]) -> float:
            """Mean of several kernels, each flipped if needed."""
            vals = [field(k, s) for k, s in kernels]
            return float(np.mean(vals)) if vals else 0.5

        attention = field("att_ratio")
        arousal = combo([("beta_alpha", 1.0), ("log_beta_abs", 1.0)])
        load = field("theta_beta")
        relax = field("rel_alpha")
        # Fatigue combines three independent, documented drowsiness signs:
        #   1. slow activity rising relative to fast ((theta+delta)/beta drift),
        #   2. alpha slowing (peak frequency dropping) - the classic marker,
        #   3. a *drowsiness proxy*: absolute beta power falling well below this
        #      subject's own baseline. Deliberately NOT wall-clock time-on-task:
        #      an agent must not be told the user is tired merely because the
        #      session has run long, and a trained T105 model can add the
        #      session-decrement term once it exists.
        fatigue = combo([("slow_fast", 1.0), ("alpha_peak_hz", -1.0),
                         ("log_beta_abs", -1.0)])
        return {"attention": attention, "arousal": arousal, "load": load,
                "fatigue": fatigue, "relaxation": relax}

    def _artifact_level(self, metrics: dict[str, float]) -> float:
        """0..1 artefact burden: quick 0..1 combine of the three kill criteria."""
        sat = min(1.0, metrics.get("saturation_pct", 0.0) / 5.0)
        mot = min(1.0, metrics.get("motion_jump_pct", 0.0) / 5.0)
        mains = min(1.0, metrics.get("mains_pct", 0.0) / 50.0)
        return float(min(1.0, max(sat, mot, 0.5 * mains)))

    def _baseline_distance(self, z: dict[str, float]) -> float:
        """Shrinkage-Mahalanobis-ish distance of this window from the baseline.

        Falls back to the max |z| when no covariance was stored (short
        calibration, or sklearn missing). Capped so one wild outlier cannot make
        the distance, and therefore the confidence, meaningless.
        """
        base = self._baseline or {}
        keys = base.get("keys") or []
        cov = base.get("cov")
        if not z:
            return 0.0
        if cov and keys:
            try:
                x = np.array([float(z.get(k, 0.0)) for k in keys], dtype=float)
                C = np.asarray(cov, dtype=float)
                d2 = float(x @ np.linalg.solve(C, x))
                return float(math.sqrt(max(0.0, d2 / max(1, len(keys)))))
            except Exception:
                pass
        return float(max(abs(v) for v in z.values()))

    def _confidence(self, metrics: dict[str, float], z: dict[str, float],
                    q: QualityReport | None) -> float:
        """0..1 confidence in the fields above (never in the EEG's *content*)."""
        if self._baseline is None:
            # uncalibrated: ceiling well under the action gate
            return 0.15
        # QualityReport.score is a 0..max_score scale (currently 4 points); read
        # the maximum off the report rather than hard-coding it, so a change to
        # the quality rubric cannot silently inflate every confidence.
        if q is not None:
            quality = q.score / float(getattr(q, "max_score", 3) or 3)
        else:
            quality = 0.5
        art = 1.0 - self._artifact_level(metrics)
        dist = self._baseline_distance(z)
        novelty = 1.0 / (1.0 + max(0.0, dist - 1.0) / 2.0)   # 1.0 near baseline
        conf = 0.15 + 0.85 * (quality * art * novelty)
        if metrics.get("saturation_pct", 0.0) > 0.05:
            conf = min(conf, 0.19)
        return float(max(0.0, min(1.0, conf)))

    # ------------------------------------------------------------- models
    def _load_models(self) -> None:
        """Best-effort load of trained state models from ``models_dir``.

        A corrupt or single-class model must never take the bridge down: every
        failure is recorded in ``model_info['errors']`` and the heuristic simply
        continues to serve that field.
        """
        try:
            from ..pipeline import decoders  # type: ignore
        except Exception as exc:
            self._warnings.append(f"pipeline.decoders unavailable ({exc}); heuristics only")
            return
        loader = getattr(decoders, "load_model", None)
        path_fn = getattr(decoders, "model_path", None)
        if loader is None:
            self._warnings.append("pipeline.decoders.load_model missing; heuristics only")
            return
        for fname, spec_ids in FIELD_TESTS.items():
            for sid in spec_ids:
                path = None
                try:
                    if callable(path_fn):
                        path = Path(path_fn(sid))
                    else:
                        path = Path(self.models_dir) / f"{sid}.joblib"
                except Exception:
                    path = Path(self.models_dir) / f"{sid}.joblib"
                if not path.exists():
                    continue
                try:
                    model = loader(str(path))
                    classes = list(getattr(model, "classes", []) or [])
                    if len(classes) < 2:
                        raise ValueError(f"model has {len(classes)} class(es); not usable")
                    if not hasattr(model, "predict_proba"):
                        raise ValueError("model has no predict_proba")
                    self._models[fname] = model
                    self._model_meta[fname] = {
                        "spec_id": getattr(model, "spec_id", sid),
                        "path": str(path),
                        "classes": classes,
                    }
                    break                        # first available spec wins the field
                except Exception as exc:
                    self._model_errors.append(f"{sid}: {exc}")
        return

    def _model_probability(self, model: Any, v: np.ndarray,
                           t: np.ndarray) -> float | None:
        """Probability of the 'high' class for this window, or None.

        Feature extraction prefers ``pipeline.features.extract`` so the vector
        matches how the model was trained; if the pipeline is unavailable we
        synthesise the vocabulary subset this project uses, and if even that
        fails we return None so the heuristic stays in charge.
        """
        names = list(getattr(model, "feature_names", []) or [])
        feats = self._feature_vector(v, t, names)
        if feats is None:
            return None
        try:
            X = np.asarray(feats, dtype=float).reshape(1, -1)
            proba = np.asarray(model.predict_proba(X), dtype=float).reshape(-1)
        except Exception as exc:
            self._model_errors.append(f"{getattr(model, 'spec_id', '?')}: predict failed: {exc}")
            return None
        classes = list(getattr(model, "classes", []) or [])
        if proba.size < 2 or len(classes) != proba.size:
            return None
        idx = _high_class_index(classes, self._field_of_model(model))
        return float(max(0.0, min(1.0, proba[idx])))

    def _field_of_model(self, model: Any) -> str:
        """Which field a loaded model drives (for label-direction keywords)."""
        for fname, m in self._models.items():
            if m is model:
                return fname
        sid = str(getattr(model, "spec_id", ""))
        for fname, sids in FIELD_TESTS.items():
            if sid in sids:
                return fname
        return ""

    def _feature_vector(self, v: np.ndarray, t: np.ndarray,
                        names: Sequence[str]) -> np.ndarray | None:
        if not names:
            return None
        try:
            from ..pipeline import features as F  # type: ignore
            feats = F.extract(v, self.fs, features=list(names))
            return F.feature_vector(feats, list(names))
        except Exception:
            pass
        try:
            return self._minimal_features(v, list(names))
        except Exception:
            return None

    def _minimal_features(self, v: np.ndarray, names: Sequence[str]) -> np.ndarray | None:
        """Minimal implementation of the documented feature vocabulary.

        WHY: this code must run even when ``pipeline/`` is mid-rewrite, and a
        trained model whose features cannot be reproduced is worse than no
        model. Only the feature strings this project actually uses for state
        tests are implemented; anything else returns None -> heuristic fallback.
        """
        bp = compute_band_power(v, self.fs)
        vv = np.asarray(v, dtype=np.float64)
        d1 = np.diff(vv) if vv.size > 1 else np.zeros(0)
        out: list[float] = []
        for name in names:
            kind, _, rest = str(name).partition(":")
            if kind == "bandpower":
                val = math.log10(bp.abs_power.get(rest, 0.0) + 1e-12)
            elif kind == "relbandpower":
                val = 100.0 * bp.rel_power.get(rest, 0.0)
            elif kind == "ratio":
                a, _, b = rest.partition("_")
                val = math.log10((bp.abs_power.get(a, 0.0) + 1e-12) /
                                 (bp.abs_power.get(b, 0.0) + 1e-12))
            elif kind == "hjorth":
                var = float(np.var(vv)) + 1e-12
                mob = float(np.sqrt(np.var(d1) / var)) if d1.size else 0.0
                if rest == "mobility":
                    val = mob
                else:
                    d2 = np.diff(d1) if d1.size > 1 else np.zeros(0)
                    mob2 = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12))) if d2.size else 0.0
                    val = mob2 / (mob + 1e-12)
            elif kind == "stats":
                if rest == "std":
                    val = float(np.std(vv))
                elif rest == "ptp":
                    val = float(np.ptp(vv))
                elif rest == "mean":
                    val = float(np.mean(vv))
                elif rest == "skew":
                    s = float(np.std(vv)) or 1.0
                    val = float(np.mean(((vv - vv.mean()) / s) ** 3))
                elif rest == "kurt":
                    s = float(np.std(vv)) or 1.0
                    val = float(np.mean(((vv - vv.mean()) / s) ** 4))
                elif rest == "zcr":
                    val = float(np.mean(np.diff(np.signbit(vv)))) if vv.size > 1 else 0.0
                else:
                    return None
            elif kind == "slope" and rest == "mean":
                val = float(np.mean(np.abs(d1))) if d1.size else 0.0
            elif kind == "peakfreq" and rest == "dominant":
                if bp.freqs.size == 0:
                    val = 0.0
                else:
                    m = (bp.freqs >= BAND_LO) & (bp.freqs <= min(BAND_HI, self.fs / 2 - 1e-9))
                    val = float(bp.freqs[m][int(np.argmax(bp.psd[m]))]) if np.any(m) else 0.0
            elif kind == "entropy" and rest == "spectral":
                p = bp.psd.copy()
                p = p[p > 0]
                p = p / (p.sum() + 1e-12)
                val = float(-np.sum(p * np.log2(p + 1e-12)))
            else:
                return None                          # unsupported -> heuristic
            out.append(float(val))
        return np.asarray(out, dtype=float)

    def _model_info(self, status: str, z: dict[str, float],
                    derivation: dict[str, str] | None = None,
                    metrics: dict[str, float] | None = None,
                    quality: QualityReport | None = None) -> dict:
        derivation = derivation or {}
        fields = {}
        for f in ("attention", "arousal", "load", "fatigue", "relaxation"):
            fields[f] = {
                "derivation": derivation.get(f, "heuristic_baseline"),
                **({"model": self._model_meta[f]} if f in self._model_meta else {}),
            }
        return {
            "status": status,
            "fs": self.fs,
            "window_s": self.window_s,
            "step_s": self.step_s,
            "n_samples": int(self._ring.size),
            "n_windows": int(self._n_windows),
            "calibrated": self._baseline is not None,
            "calibration_path": str(self._baseline_path) if self._baseline_path else None,
            "baseline_loaded_from_disk": bool(self._baseline_from_disk),
            "fields": fields,
            "trained_models": sorted(self._models.keys()),
            "models_available": {"available": {k: v for k, v in self._model_meta.items()}},
            "errors": list(self._model_errors),
            "warnings": list(self._warnings),
            "z": {k: round(float(v), 3) for k, v in (z or {}).items()},
            "metrics": {k: round(float(v), 6) for k, v in (metrics or {}).items()},
            "quality": quality.as_dict() if quality is not None else None,
            "contact": (quality.contact if quality is not None else "unknown"),
            "honesty": (
                "One uncalibrated dry frontal channel. Fields are relative to the "
                "subject's own baseline and are not measurements; 0.5 means 'at "
                "baseline', not 'average human'. No word, sentence, image or "
                "intention reading is possible with this montage."
            ),
            "valence_note": (
                "valence is always null: frontal-asymmetry valence requires at least "
                "two symmetric channels, and this device has one. It is never faked."
            ),
            "disclaimer": (
                "attention/arousal/load/fatigue/relaxation are monotone proxies. "
                "load is an effort proxy (theta/beta), not a reading of thought "
                "content."
            ),
        }


def _high_class_index(classes: Sequence[str], field_name: str = "") -> int:
    """Index of the class representing the 'high' end of a binary state.

    Class labels in the catalog are human-written (``load_high``/``load_low``,
    ``calm``/``exciting``, ...), so the mapping is keyword-based, scoped to the
    field the model drives, with a documented fallback: if nothing matches, the
    last class is treated as high and ``model_info`` still discloses the model
    id, so a wrong guess is visible rather than hidden.
    """
    labels = [str(c).lower() for c in classes]
    if not labels:
        return 0
    high_lists = ([_HIGH_WORDS[field_name]] if field_name in _HIGH_WORDS
                  else list(_HIGH_WORDS.values()))
    low_lists = ([_LOW_WORDS[field_name]] if field_name in _LOW_WORDS
                 else list(_LOW_WORDS.values()))
    for words in high_lists:
        for w in words:
            for i, lab in enumerate(labels):
                if w in lab:
                    return i
    for words in low_lists:
        for w in words:
            for i, lab in enumerate(labels):
                if w in lab:
                    return (1 - i) if len(labels) == 2 else (i + 1) % len(labels)
    return len(labels) - 1


# ------------------------------------------------------------- intent detector
class IntentDetector:
    """Online blink / jaw-clench **command** detection on one frontal channel.

    These two artefacts are the only commands this hardware supports reliably:
    at Fpz a blink is ~100x the cortical amplitude and a jaw clench produces a
    broadband 20-45 Hz EMG burst, so both are detectable from a single dry
    electrode without any model. Everything else an agent might want (words,
    selections beyond yes/no, imagined movement) is not available here.

    Detection rules (documented so they can be argued with):

    * **blink** - 0.1-5 Hz component exceeds ``median + blink_sigma * 1.4826*MAD``
      computed over a rolling 3 s baseline *ending before the current chunk*,
      then returns toward baseline. The exceedance must last 0.15-2.0 s: a blink
      is a large *slow monophasic* deflection, so a narrow spike is rejected.
    * **clench** - 20-45 Hz envelope (Hilbert magnitude) exceeds
      ``median + 5 * 1.4826*MAD`` of a rolling 5 s baseline, for 0.2-1.5 s.
    * both channels use a refractory period (default 0.6 s) so one artefact
      cannot fire twice on its rising and falling edges.

    ``confidence`` comes from how far above threshold the peak was relative to
    the baseline scale (``min(1, (peak - thr) / (4 * scale))``), so a marginal
    twitch is reported as a marginal command and the agent can ignore it.
    """

    def __init__(self, fs: float = 250.0, blink_sigma: float = 6.0,
                 clench_band: tuple[float, float] = (20, 45),
                 refractory_s: float = 0.6) -> None:
        self.fs = float(fs)
        self.blink_sigma = float(blink_sigma)
        self.clench_band = (float(clench_band[0]), float(clench_band[1]))
        self.refractory_s = float(refractory_s)
        self._lock = threading.Lock()

        self._raw = _Ring(capacity=max(int(self.fs * 20), 4096))
        self._lf_base = _Ring(capacity=max(int(self.fs * 3), 512))
        self._hf_base = _Ring(capacity=max(int(self.fs * 5), 1024))
        #: filtered samples that have not yet been judged: they are held until
        #: ``edge_pad`` samples of *later* data exist, because a zero-phase
        #: filter produces huge transients at the edges of every window and a
        #: naive implementation would fire on its own filter artefact.
        self._lf_pending: list[np.ndarray] = []
        self._hf_pending: list[np.ndarray] = []
        self._pending_t: list[np.ndarray] = []
        #: filter context required on each side of a judged sample (0.4 s):
        #: sosfiltfilt pads with zeros, so this is the region where the
        #: transient has decayed below the cortical signal floor.
        self._edge_pad = max(32, int(0.4 * self.fs))
        self._max_chunk = 0

        #: per-channel peak/width state machine state (see :meth:`_scan`)
        self._chan: dict[str, dict[str, float | bool]] = {
            "lf": {"armed": False, "start": 0.0, "peak": 0.0, "peak_t": 0.0,
                   "last_over": 0.0, "last_emit": -1e9},
            "hf": {"armed": False, "start": 0.0, "peak": 0.0, "peak_t": 0.0,
                   "last_over": 0.0, "last_emit": -1e9},
        }
        self._thr_cache: dict[str, tuple[float, float, float]] = {}
        self._warnings: list[str] = []
        self._n_blink = 0
        self._n_clench = 0
        #: median+MAD thresholds are only meaningful once this much baseline
        #: exists; before that, detection stays off rather than firing on noise.
        self.min_baseline_s = 2.0

    # ------------------------------------------------------------- ingestion
    def push(self, values: Sequence[float] | np.ndarray,
             t_mono: Sequence[float] | np.ndarray | float | None = None) -> list[dict]:
        """Feed a chunk of raw counts; returns any commands completed by it.

        Detection is causal and edge-safe: ``edge_pad`` samples at both ends of
        every filtered window are discarded, so the answer does not depend on
        where the chunk boundaries fell. The cost is one chunk of latency
        (~edge_pad / fs, i.e. 0.4 s at 250 Hz), which is irrelevant for a
        deliberate blink or jaw clench but fatal if ignored (the filter's edge
        transient is larger than a clench).
        """
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return []
        if t_mono is None:
            t = time.perf_counter() + (np.arange(v.size) - (v.size - 1)) / self.fs
        elif np.isscalar(t_mono):
            t = np.full(v.size, float(t_mono), dtype=np.float64)
        else:
            t = np.asarray(t_mono, dtype=np.float64).reshape(-1)
            if t.size != v.size:
                raise ValueError(f"values {v.shape} and times {t.shape} must match")

        out: list[dict] = []
        with self._lock:
            self._raw.push(v, t)
            big = int(v.size) > self._max_chunk
            if big or self._raw.capacity < 4 * int(v.size) + 4 * self._edge_pad:
                self._max_chunk = max(self._max_chunk, int(v.size))
                self._raw = _Ring(capacity=max(self._raw.capacity,
                                               4 * self._max_chunk + 4 * self._edge_pad))
            ctx = self._edge_pad
            n = min(self._raw.size, int(v.size) + 2 * ctx)
            raw, rt = self._raw.window(n)
            if raw.size <= 2 * ctx:                  # not enough context yet
                return []
            n_scan = raw.size - 2 * ctx              # samples with context on both sides
            scan_raw = raw[ctx:ctx + n_scan]
            scan_t = rt[ctx:ctx + n_scan]
            lf = self._lowfreq(raw)[ctx:ctx + n_scan]
            hf = self._clench_envelope(raw)[ctx:ctx + n_scan]
            self._lf_pending.append(lf)
            self._hf_pending.append(hf)
            self._pending_t.append(scan_t)
            pending = int(sum(a.size for a in self._lf_pending))
            if pending < self._edge_pad:             # keep one chunk in hand
                return []
            lf_new = np.concatenate(self._lf_pending)
            hf_new = np.concatenate(self._hf_pending)
            t_new = np.concatenate(self._pending_t)
            self._lf_pending, self._hf_pending, self._pending_t = [], [], []
            out += self._scan(prefix="lf", band="blink", sig=self.blink_sigma,
                              t=t_new, new_sig=lf_new,
                              base_ring=self._lf_base, refractory=self.refractory_s)
            out += self._scan(prefix="hf", band="clench", sig=5.0,
                              t=t_new, new_sig=hf_new,
                              base_ring=self._hf_base, refractory=self.refractory_s)
        return sorted(out, key=lambda d: d["t"])

    # ------------------------------------------------------------- detectors
    def _scan(self, *, prefix: str, band: str, sig: float, t: np.ndarray,
              new_sig: np.ndarray, base_ring: _Ring,
              refractory: float) -> list[dict]:
        """Peak/width state machine shared by both artefact channels.

        Why a state machine instead of a simple threshold crossing: the
        *duration* is what separates a blink (slow, up to 2 s) from a spike, and
        the *recovery* is what defines the command end. The baseline ring is
        advanced *after* the scan so the artefact itself cannot lift its own
        detection threshold.
        """
        out: list[dict] = []
        if new_sig.size == 0:
            return out
        med, scale, n_base = self._threshold(base_ring)
        st = self._chan[prefix]
        armed = bool(st["armed"])
        start = float(st["start"])
        peak = float(st["peak"])
        peak_t = float(st["peak_t"])
        last_over = float(st.get("last_over", start))
        last_emit = float(st["last_emit"])

        thr = med + sig * scale
        rec = med + 0.25 * (thr - med)
        ready = n_base >= int(self.min_baseline_s * self.fs)
        for i in range(new_sig.size):
            x = float(new_sig[i])
            ti = float(t[i]) if i < t.size else 0.0
            if not armed:
                if ready and x > thr and (ti - last_emit) > refractory:
                    armed, start, peak, peak_t, last_over = True, ti, x, ti, ti
                continue
            if x > peak:
                peak, peak_t = x, ti
            if x > thr:
                last_over = ti
            # The width is threshold-entry -> LAST exceedance, not entry -> peak:
            # a jaw clench's 20-45 Hz envelope is a beat pattern with many local
            # maxima, so peak_t lands on an arbitrary oscillation and would make a
            # 1 s clench look 80 ms wide.
            dur = last_over - start
            if x < rec or dur > 2.5:
                lo, hi = (0.15, 2.0) if band == "blink" else (0.2, 1.5)
                if lo <= dur <= hi and peak > thr:
                    base_scale = max(scale, 1e-9)
                    conf = min(1.0, (peak - thr) / (4.0 * base_scale))
                    out.append({"intent": band, "t": round(float(peak_t), 4),
                                "confidence": round(float(conf), 3),
                                "duration_s": round(float(dur), 3),
                                "peak": round(float(peak), 1),
                                "threshold": round(float(thr), 1),
                                "method": ("0.1-5 Hz slow deflection" if band == "blink"
                                           else f"{self.clench_band[0]:.0f}-"
                                                f"{self.clench_band[1]:.0f} Hz EMG envelope")})
                    last_emit = float(last_over)
                    if band == "blink":
                        self._n_blink += 1
                    else:
                        self._n_clench += 1
                armed, peak = False, 0.0
        st.update({"armed": armed, "start": start, "peak": peak,
                   "peak_t": peak_t, "last_over": last_over,
                   "last_emit": last_emit})
        # advance the rolling baseline only with post-scan knowledge
        base_ring.push(new_sig, t if t.size == new_sig.size
                       else np.full(new_sig.size, float(t[-1]) if t.size else 0.0))
        return out

    def _threshold(self, ring: _Ring) -> tuple[float, float, int]:
        """Cached (median, 1.4826*MAD, n) for a baseline ring."""
        key = "lf" if ring is self._lf_base else "hf"
        n = ring.size
        cached = self._thr_cache.get(key)
        if cached is not None and cached[2] == n:
            return cached[0], cached[1], n
        if n == 0:
            return 0.0, 1.0, 0
        v, _ = ring.window(n, copy=False)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        # 1.4826*MAD == sigma for Gaussian data; floored because an all-quiet
        # baseline has MAD ~ 0 and would otherwise make the detector infinitely
        # sensitive to one noisy sample.
        scale = max(1.4826 * mad, 0.1 * abs(med), 1.0)
        self._thr_cache[key] = (med, scale, n)
        return med, scale, n

    # ------------------------------------------------------------- filtering
    def _lowfreq(self, v: np.ndarray) -> np.ndarray:
        """0.1-5 Hz component: blinks are the only thing that big and that slow."""
        try:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(3, (0.1, 5.0), btype="bandpass", fs=self.fs, output="sos")
            return np.asarray(sosfiltfilt(sos, v), dtype=np.float64)
        except Exception as exc:                     # pragma: no cover - scipy path
            self._warn(exc)
            return self._ma_band(v)

    def _clench_envelope(self, v: np.ndarray) -> np.ndarray:
        """Hilbert magnitude of the 20-45 Hz band: jaw-clench EMG power."""
        try:
            from scipy.signal import butter, hilbert, sosfiltfilt
            lo, hi = self.clench_band
            hi = min(hi, self.fs / 2.0 - 1.0)
            sos = butter(3, (lo, hi), btype="bandpass", fs=self.fs, output="sos")
            band = sosfiltfilt(sos, v)
            return np.abs(hilbert(band)).astype(np.float64)
        except Exception as exc:                     # pragma: no cover - scipy path
            self._warn(exc)
            return np.abs(self._ma_band(v))

    def _ma_band(self, v: np.ndarray) -> np.ndarray:
        """Dependency-light fallback: sliding-mean highpass then rectify.

        Only used if scipy is unavailable. It is deliberately crude and the
        detector records the degradation in ``warnings`` so nobody mistakes the
        fallback for the calibrated path.
        """
        n = max(3, int(0.2 * self.fs))
        k = np.ones(n) / n
        slow = np.convolve(v, k, mode="same")
        return v - slow

    def _warn(self, exc: Exception) -> None:
        msg = f"filter fallback in use: {exc}"
        if msg not in self._warnings:
            self._warnings.append(msg)

    # ------------------------------------------------------------- status
    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def counts(self) -> dict[str, int]:
        return {"blink": self._n_blink, "clench": self._n_clench}

    def reset(self) -> None:
        with self._lock:
            for r in (self._raw, self._lf_base, self._hf_base):
                r.clear()
            self._lf_pending, self._hf_pending, self._pending_t = [], [], []
            for ch in self._chan.values():
                ch.update({"armed": False, "start": 0.0, "peak": 0.0,
                           "peak_t": 0.0, "last_over": 0.0, "last_emit": -1e9})
            self._thr_cache.clear()
