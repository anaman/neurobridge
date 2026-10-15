"""Preprocessing and artefact detection for the single dry frontal channel.

Why this module exists in this shape
------------------------------------
The TH21A gives us **24-bit signed ADC counts that are not calibrated**. The
counts are proportional to volts but the constant of proportionality is
unknown and changes with electrode impedance, so *no absolute amplitude is
ever meaningful*. Everything downstream is therefore relative (band power
ratios, within-session z-scores) and the only hard numbers we trust from the
raw stream are:

* the 24-bit rail (``|x| >= 2**23 - 1``) which means the amplifier saturated
  and the electrode is off the head, and
* sample-to-sample structure (jumps = motion, long constant runs = flatline).

Filtering choices, and why:

* **IIR notch at 50 and 60 Hz, Q ~ 30, zero-phase.** Mains hum is the single
  largest contaminant at Fpz. An IIR notch is far cheaper than FFT masking for
  a streaming system, and Q ~ 30 is narrow enough to leave 45 Hz gamma usable
  while still killing the 50/60 Hz lines. Both mains frequencies are notched
  because the headband may be used on either side of the Atlantic and a
  mismatch between notch and line frequency would leave a large residual.
* **Order-4 Butterworth bandpass through ``sosfiltfilt``.** Second-order
  sections are numerically stable at low cut-offs (0.3 Hz at 250 Hz is a
  normalised frequency of 0.0024, where a plain ``b, a`` form loses precision).
  ``sosfiltfilt`` runs the filter forwards and backwards, cancelling phase
  distortion: ERP latency is the whole point of several tests here, so a
  non-linear-phase filter would shift components by tens of milliseconds and
  bias every latency feature. Order 4 is a compromise between a sharp enough
  roll-off and ringing.
* **Detrend before filtering.** Dry electrodes drift by hundreds of thousands
  of counts over minutes. Removing the linear trend first stops the IIR
  sections from ringing on that ramp and keeps the filtfilt edge transient
  small.
* **Latency offsets shift onset *times*, never the signal.** The stimulus and
  its electrocortical response are separated by the output+acquisition
  latency. Shifting the signal would smear the ERP; shifting the epoch onset
  keeps the waveform intact and corrects the alignment, which is what latency
  features need.

Nothing here writes to disk, imports a device, or touches matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import signal as sps

from ..config import get_settings
from ..device.quality import SAT, assess, QualityReport

__all__ = [
    "filter_signal",
    "detect_artifacts",
    "artifact_mask",
    "PreprocessResult",
    "preprocess",
    "preprocess_session",
    "NOTCH_Q",
    "BANDPASS_ORDER",
]

#: Q of the mains notches. High enough to be narrow, low enough to be stable.
NOTCH_Q = 30.0

#: Butterworth order. Works with ``sosfiltfilt`` and does not ring badly.
BANDPASS_ORDER = 4

#: Time constant used for "the signal is stuck" detection (seconds).
_FLATLINE_S = 0.2

#: Window used for the peak-to-peak rejection sweep (seconds).
_PTP_WINDOW_S = 0.5


# ---------------------------------------------------------------- filtering
def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``sosfiltfilt`` that degrades gracefully on very short arrays.

    ``sosfiltfilt`` needs at least a few samples; a 40 ms calibration click
    train can be shorter than the default pad length. Instead of raising (which
    would take down a whole session's analysis) we shrink the padding, and for
    pathological inputs fall back to a single causal pass.
    """
    n = int(x.size)
    if n == 0:
        return x.copy()
    if n < 8:
        return sps.sosfilt(sos, x)
    try:
        default_pad = 3 * (2 * len(sos) + 1)
        padlen = int(min(default_pad, max(0, n - 1)))
        return sps.sosfiltfilt(sos, x, padlen=padlen)
    except ValueError:
        return sps.sosfilt(sos, x)


def filter_signal(values: np.ndarray, fs: float,
                  bandpass: Sequence[float] = (0.3, 45.0),
                  notch: Iterable[float] | None = (50.0, 60.0),
                  *, detrend: bool = True, q: float = NOTCH_Q,
                  order: int = BANDPASS_ORDER) -> np.ndarray:
    """Detrend, notch and band-pass a single channel; never mutates ``values``.

    Parameters
    ----------
    values:
        Raw (uncalibrated) ADC counts, 1-D. Larger shapes are flattened.
    fs:
        Sampling rate in Hz.
    bandpass:
        ``(low, high)`` in Hz. A non-positive low edge becomes a low-pass.
        Edges at or above Nyquist are clamped instead of raising: a spec can
        legitimately ask for 45 Hz at 100 Hz.
    notch:
        Mains frequencies to remove. Frequencies at/above Nyquist are skipped.
    detrend:
        Remove a linear trend before filtering. Keep this on for dry
        electrodes; the drift is real and huge.
    q, order:
        Filter design parameters, exposed so the chosen values can be recorded
        in ``PreprocessResult.meta`` and reproduced exactly.

    Returns
    -------
    np.ndarray
        Float64 filtered signal, same length as the input. The input array is
        never modified in place.
    """
    x = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    fs = float(fs)
    if x.size == 0 or fs <= 0:
        return x

    if detrend and x.size > 2:
        x = sps.detrend(x, type="linear")

    nyq = fs / 2.0

    # ---- mains notches (IIR, Q ~ 30, zero-phase) --------------------
    for f0 in (notch or ()):
        f0 = float(f0)
        if not np.isfinite(f0) or f0 <= 0 or f0 >= nyq * 0.98:
            continue                      # cannot notch at/above Nyquist
        b, a = sps.iirnotch(w0=f0, Q=float(q), fs=fs)
        x = _safe_sosfiltfilt(sps.tf2sos(b, a), x)

    # ---- Butterworth band-pass -------------------------------------
    lo, hi = (float(bandpass[0]), float(bandpass[1]))
    hi = min(hi, nyq * 0.98)
    lo = max(lo, 0.0)
    if hi > 0 and (lo >= hi or lo <= 0):
        # Degenerate band: fall back to a low-pass at the upper edge.
        sos = sps.butter(int(order), hi, btype="low", fs=fs, output="sos")
    else:
        sos = sps.butter(int(order), [lo, hi], btype="band", fs=fs, output="sos")
    x = _safe_sosfiltfilt(sos, x)
    return x


# ---------------------------------------------------------------- artefacts
def _robust_step_scale(d: np.ndarray) -> float:
    """Median absolute first difference, floored away from zero.

    A median is used instead of a standard deviation because the very jumps we
    are hunting would inflate a std and hide themselves.
    """
    if d.size == 0:
        return 1.0
    med = float(np.median(d))
    return med if med > 0 else 1.0


def artifact_mask(values: np.ndarray, fs: float,
                  ptp_limit: float | None = None) -> np.ndarray:
    """Boolean mask of *samples* that should not enter an epoch.

    Three independent mechanisms, because they mean different things:

    * **saturation** — the 24-bit rail was hit, so the value carries no
      information at all (electrode off the head);
    * **flatline** — a long run of identical samples: the amplifier or the BLE
      link froze, and the "EEG" is a constant;
    * **motion jumps** — sample-to-sample steps far above the robust local
      scale, typical of cable tug or jaw clench. Both samples adjacent to the
      jump are flagged, because the artifact has no single-sample location.

    ``ptp_limit`` additionally flags every sample of any 0.5 s window whose
    peak-to-peak exceeds the limit. That is the epoch-level rejection rule
    applied pointwise; keeping it here means epoch rejection and the quality
    report agree instead of silently disagreeing.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    bad = np.zeros(n, dtype=bool)
    if n == 0:
        return bad

    # 1. saturation at either rail
    bad |= np.abs(v) >= (SAT - 2.0)

    # 2. flatline: runs of exactly-equal consecutive samples
    if n > 2:
        d = np.abs(np.diff(v))
        equal = d <= 0.0
        idx = np.flatnonzero(equal)
        if idx.size:
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            min_run = max(2, int(round(_FLATLINE_S * float(fs))))
            for run in np.split(idx, splits):
                if run.size + 1 >= min_run:
                    bad[run[0]: run[-1] + 2] = True

    # 3. motion jumps
    if n > 3:
        d1 = np.abs(np.diff(v))
        scale = _robust_step_scale(d1)
        jump = d1 > 20.0 * scale
        bad[:-1] |= jump
        bad[1:] |= jump

    # 4. sliding peak-to-peak
    if ptp_limit is not None and np.isfinite(ptp_limit):
        limit = float(ptp_limit)
        w = max(2, int(round(_PTP_WINDOW_S * float(fs))))
        if n >= w:
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                win = sliding_window_view(v, w)
                ptp = win.max(axis=1) - win.min(axis=1)
                for i in np.flatnonzero(ptp > limit):
                    bad[i: i + w] = True
            except Exception:                     # pragma: no cover - numpy fallback
                if float(np.ptp(v)) > limit:
                    bad[:] = True
        elif float(np.ptp(v)) > limit:
            bad[:] = True
    return bad


def detect_artifacts(values: np.ndarray, fs: float,
                     ptp_limit: float | None = None) -> dict[str, Any]:
    """Summarise artefact load as percentages plus a single ``bad_frac``.

    The percentages are reported separately because they call for different
    operator actions: saturation means reseat the band, motion means sit still,
    flatline means a hardware/link problem. ``bad_frac`` is the fraction of
    samples any mechanism flagged, and is what the reports quote.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    out: dict[str, Any] = {
        "saturation_pct": 0.0,
        "motion_pct": 0.0,
        "flatline_pct": 0.0,
        "bad_frac": 0.0,
        "n_samples": n,
        "ptp_limit": None if ptp_limit is None else float(ptp_limit),
    }
    if n == 0:
        return out

    sat = np.abs(v) >= (SAT - 2.0)
    out["saturation_pct"] = 100.0 * float(np.mean(sat))

    if n > 3:
        d1 = np.abs(np.diff(v))
        scale = _robust_step_scale(d1)
        jump = d1 > 20.0 * scale
        out["motion_pct"] = 100.0 * float(np.mean(jump))

    if n > 2:
        flat = np.zeros(n, dtype=bool)
        d = np.abs(np.diff(v))
        idx = np.flatnonzero(d <= 0.0)
        if idx.size:
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            min_run = max(2, int(round(_FLATLINE_S * float(fs))))
            for run in np.split(idx, splits):
                if run.size + 1 >= min_run:
                    flat[run[0]: run[-1] + 2] = True
        out["flatline_pct"] = 100.0 * float(np.mean(flat))

    mask = artifact_mask(v, fs, ptp_limit=ptp_limit)
    out["bad_frac"] = float(np.mean(mask))
    out["n_bad"] = int(np.count_nonzero(mask))
    return out


# ---------------------------------------------------------------- session API
@dataclass
class PreprocessResult:
    """Filtered signal + everything needed to justify or reproduce it."""

    signal: np.ndarray
    fs: float
    quality: QualityReport
    artifacts: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(np.size(self.signal))


def _processing_of(spec: Any) -> dict[str, Any]:
    proc = getattr(spec, "processing", None)
    return dict(proc) if isinstance(proc, dict) else {}


def _filter_params(spec: Any, kw: dict[str, Any]) -> dict[str, Any]:
    """Resolve filter parameters, with explicit kwargs winning over the spec.

    Precedence: call kwargs > spec.processing > global settings. Recording the
    winner in ``meta`` is what makes a figure reproducible six months later.
    """
    settings = get_settings()
    proc = _processing_of(spec)

    bandpass = kw.get("bandpass", proc.get("bandpass", settings.bandpass_hz))
    if bandpass is None:
        bandpass = settings.bandpass_hz
    notch = kw.get("notch", proc.get("notch", proc.get("notch_hz", settings.notch_hz)))
    if notch is None:
        notch = ()
    if isinstance(notch, (int, float)):
        notch = (float(notch),)
    return {
        "bandpass": (float(bandpass[0]), float(bandpass[1])),
        "notch": tuple(float(f) for f in notch),
        "detrend": bool(kw.get("detrend", True)),
        "q": float(kw.get("q", NOTCH_Q)),
        "order": int(kw.get("order", BANDPASS_ORDER)),
    }


def latency_offset_s(spec: Any) -> float:
    """Effective stimulus-output latency correction, in seconds.

    Positive means "the neural response happens this much *after* the marker",
    so onsets are shifted later by this amount (never the signal). A spec may
    pin its own value (``processing.latency_offset_s`` or ``_ms``); otherwise
    the machine-wide value from T004/T005 calibration is used.
    """
    proc = _processing_of(spec)
    for key in ("latency_offset_s",):
        if proc.get(key) is not None:
            return float(proc[key])
    if proc.get("latency_offset_ms") is not None:
        return float(proc["latency_offset_ms"]) / 1000.0
    return float(get_settings().output_latency_offset_s)


def artifact_ptp_limit(spec: Any) -> float | None:
    """Peak-to-peak rejection threshold in ADC counts (spec first, else settings)."""
    proc = _processing_of(spec)
    v = proc.get("artifact_ptp", None)
    if v is None:
        v = get_settings().artifact_ptp_counts
    return None if v in (None, 0) else float(v)


def preprocess(values: np.ndarray, fs: float, spec: Any = None,
               **kw: Any) -> np.ndarray:
    """Convenience wrapper: filter one array using a spec's processing block.

    Returns a *new* array. Kept separate from :func:`preprocess_session` so the
    runner (which has arrays, not sessions) can use the identical filter recipe.
    """
    p = _filter_params(spec, kw)
    return filter_signal(values, fs, bandpass=p["bandpass"], notch=p["notch"],
                         detrend=p["detrend"], q=p["q"], order=p["order"])


def preprocess_session(session: Any, spec: Any = None) -> PreprocessResult:
    """Filter a recorded :class:`~neurobridge.store.Session` and judge it.

    The quality report is computed on the **raw** signal on purpose: saturation
    and drift are properties of the capture, and the band-pass would erase
    exactly the evidence we want to report. The artifacts dict is also computed
    on the raw signal for the same reason.

    The returned ``meta`` records the exact filter parameters, the session id,
    ``fs`` and the effective latency offset, so a report can state what was
    actually done rather than what the defaults are.
    """
    raw = np.array(getattr(session, "eeg", np.zeros(0)), dtype=np.float64,
                   copy=True).reshape(-1)
    try:
        fs = float(getattr(spec, "fs", None) or getattr(session, "fs", 250.0))
    except Exception:
        fs = 250.0

    session_id = str(getattr(session, "meta", {}).get("session_id")
                     or getattr(getattr(session, "path", None), "name", "unknown"))
    contact = str(getattr(session, "meta", {}).get("contact", "unknown"))
    p = _filter_params(spec, {})
    ptp = artifact_ptp_limit(spec)
    lat = latency_offset_s(spec)

    warnings: list[str] = []
    if raw.size == 0:
        warnings.append("session has no samples; nothing to filter")
        return PreprocessResult(
            signal=np.zeros(0, dtype=np.float64), fs=fs,
            quality=assess(raw, fs, contact=contact),
            artifacts=detect_artifacts(raw, fs, ptp_limit=ptp),
            meta=_meta(session_id, fs, raw, p, ptp, lat, warnings),
        )
    if raw.size < 8:
        warnings.append(f"only {raw.size} samples: filtering is nominal at best")
    if raw.size / max(fs, 1e-9) < 1.0:
        warnings.append("less than 1 s of data: metrics are indicative only")

    quality = assess(raw, fs, contact=contact)
    artifacts = detect_artifacts(raw, fs, ptp_limit=ptp)
    if artifacts.get("saturation_pct", 0.0) > 0.05:
        warnings.append("saturation present: the amplifier railed, reseat the band")
    if artifacts.get("bad_frac", 0.0) > 0.5:
        warnings.append("more than half the samples are flagged as artefact")

    filtered = filter_signal(raw, fs, bandpass=p["bandpass"], notch=p["notch"],
                             detrend=p["detrend"], q=p["q"], order=p["order"])
    meta = _meta(session_id, fs, raw, p, ptp, lat, warnings)
    meta["quality_source"] = "raw"
    meta["raw_mean"] = float(np.mean(raw))
    meta["raw_ptp"] = float(np.ptp(raw))
    return PreprocessResult(signal=filtered, fs=fs, quality=quality,
                            artifacts=artifacts, meta=meta)


def _meta(session_id: str, fs: float, raw: np.ndarray, p: dict[str, Any],
          ptp: float | None, lat: float, warnings: list[str]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "fs": float(fs),
        "n_samples": int(raw.size),
        "duration_s": float(raw.size / fs) if fs > 0 else 0.0,
        "bandpass_hz": [p["bandpass"][0], p["bandpass"][1]],
        "notch_hz": list(p["notch"]),
        "notch_q": float(p["q"]),
        "bandpass_order": int(p["order"]),
        "detrend": bool(p["detrend"]),
        "filter": "scipy.signal.sosfiltfilt (zero-phase), st=sos",
        "artifact_ptp": None if ptp is None else float(ptp),
        "latency_offset_s": float(lat),
        "warnings": list(warnings),
    }
