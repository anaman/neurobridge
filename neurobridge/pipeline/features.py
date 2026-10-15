"""Feature extraction: the vocabulary implemented by ``docs/CATALOG_SPEC.md``.

Why these features, and not "just feed the raw signal to a CNN"
--------------------------------------------------------------
One dry frontal electrode at 250 Hz, an uncalibrated 24-bit ADC and a handful
of trials per block. Under those constraints a small number of *interpretable,
low-variance* features beats a large learned representation, and every feature
here has a documented reason to exist:

* **Absolute amplitudes are meaningless** (the ADC gain is unknown), so the
  workhorses are ``relbandpower`` (a percentage of 1-45 Hz power — gain
  cancels) and ``ratio`` (a dB-scale ratio of two bands). ``bandpower`` is
  log10 absolute power: useful *within* a session, never compared across
  sessions, and included because log-power is roughly Gaussian, which is what
  the linear decoders assume.
* **ERP windows keyed in milliseconds** are the only way to express
  "the P300 deflection", and milliseconds make a spec portable across sampling
  rates.
* **``aperiodic:exponent``** separates a genuine band-power change from a
  shift of the 1/f background. Without it, "more theta" could just be a
  drowsier, flatter spectrum. FOOOF-lite: fit log-power against log-frequency
  with the alpha peak excluded, then report the residual alpha peak separately.
* **Artifact-sensitive features are deliberately absent** (no raw amplitude,
  no kurtosis-heavy summaries that explode with one blink). The one exception
  is ``erp_peak``, which is offered but documented as variance-inflated:
  single-trial peaks are noise maxima, so prefer ``erp:`` means.

Short signals (< 1 s) are common here — a 250-sample epoch is one second, and
a 40 ms click-train epoch is 10 samples. Every function therefore degrades to a
periodogram estimate or a NaN rather than raising; NaN is then handled by the
decoder's imputer.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal as sps
from scipy import stats as sstats

__all__ = [
    "BANDS",
    "FEATURE_NAMES",
    "DEFAULT_FEATURES",
    "band_powers",
    "extract",
    "feature_names",
    "feature_vector",
    "extract_epochs",
]

#: Canonical bands. ``beta`` overlaps ``low_beta``/``high_beta`` on purpose:
#: the catalog uses both coarse and split beta, and an overlapping definition
#: here is less surprising than silently redefining one of them.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "beta": (13.0, 30.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}

_STATS = ("std", "ptp", "mean", "skew", "kurt", "zcr")
_HJORTH = ("mobility", "complexity")
_LOG_FLOOR = 1e-12
_ALPHA_LO, _ALPHA_HI = 7.5, 12.5     # neighbourhood excluded from the 1/f fit
_FIT_LO, _FIT_HI = 2.0, 40.0

#: Features used when a caller passes ``features=None``: a general-purpose,
#: gain-invariant set. Specs should always pass an explicit list.
DEFAULT_FEATURES: list[str] = [
    "bandpower:theta", "bandpower:alpha", "bandpower:beta",
    "relbandpower:theta", "relbandpower:alpha", "relbandpower:beta",
    "ratio:theta_beta", "ratio:alpha_beta",
    "hjorth:mobility", "hjorth:complexity", "entropy:spectral",
    "slope:mean", "peakfreq:dominant", "aperiodic:exponent",
]

#: Canonical band order, used for stable ratio naming.
_BAND_ORDER = list(BANDS)

#: Every supported feature string. Parameterised families are listed both as
#: concrete examples (``bandpower:alpha``, ``ratio:theta_beta``) and as
#: templates (``erp:<name>:<lo_ms>:<hi_ms>``) so a caller can discover the
#: whole vocabulary from one place.
FEATURE_NAMES: list[str] = (
    [f"bandpower:{b}" for b in _BAND_ORDER]
    + [f"relbandpower:{b}" for b in _BAND_ORDER]
    + [f"ratio:{a}_{b}" for i, a in enumerate(_BAND_ORDER) for b in _BAND_ORDER[i + 1:]]
    + ["erp:<name>:<lo_ms>:<hi_ms>", "erp_peak:<name>:<lo_ms>:<hi_ms>"]
    + [f"hjorth:{h}" for h in _HJORTH]
    + ["entropy:spectral"]
    + [f"stats:{s}" for s in _STATS]
    + ["slope:mean", "peakfreq:dominant",
       "aperiodic:exponent", "aperiodic:alpha_peak", "cov:tangent"]
)


# ---------------------------------------------------------------- parsing
def _as_list(features: Any) -> list[str]:
    if features is None:
        return list(DEFAULT_FEATURES)
    if isinstance(features, str):
        return [features]
    if isinstance(features, Mapping):
        return [str(k) for k in features.keys()]
    return [str(f) for f in features]


def _split_ratio(spec: str) -> tuple[str, str]:
    """Split ``a_b`` into two band names, tolerating bands with underscores.

    ``ratio:low_beta_high_beta`` is genuinely ambiguous by string splitting, so
    we try every known band as the left operand, longest name first.
    """
    for a in sorted(BANDS, key=len, reverse=True):
        if spec == a or spec.startswith(a + "_"):
            rest = spec[len(a):].lstrip("_")
            if rest in BANDS:
                return a, rest
    for a in _BAND_ORDER:
        for b in _BAND_ORDER:
            if spec == f"{a}_{b}":
                return a, b
    raise ValueError(f"ratio: cannot parse {spec!r} into two bands from {_BAND_ORDER}")


def _parse_erp(spec: str, family: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) == 4:                       # erp:P3:250:450
        _, name, lo, hi = parts
    elif len(parts) == 3:                     # erp:250:450 (unnamed component)
        _, lo, hi = parts
        name = family
    else:
        raise ValueError(f"{spec!r} must look like {family}:<name>:<lo_ms>:<hi_ms>")
    try:
        return str(name), float(lo), float(hi)
    except ValueError as exc:
        raise ValueError(f"{spec!r}: window bounds must be numbers in ms") from exc


def _validate(spec: str) -> None:
    """Raise ``ValueError`` for anything outside the vocabulary."""
    head = spec.split(":", 1)[0]
    if spec in BANDS:
        return
    if head in ("bandpower", "relbandpower"):
        band = spec.split(":", 1)[1] if ":" in spec else ""
        if band not in BANDS:
            raise ValueError(f"unknown band {band!r} in {spec!r}; have {_BAND_ORDER}")
        return
    if head == "ratio":
        _split_ratio(spec.split(":", 1)[1])
        return
    if head in ("erp", "erp_peak"):
        _parse_erp(spec, head)
        return
    if head == "hjorth":
        if spec.split(":", 1)[1] not in _HJORTH:
            raise ValueError(f"unknown hjorth parameter in {spec!r}; have {list(_HJORTH)}")
        return
    if spec == "entropy:spectral" or spec in ("slope:mean", "peakfreq:dominant",
                                              "aperiodic:exponent", "aperiodic:alpha_peak",
                                              "cov:tangent"):
        return
    if head == "stats":
        if spec.split(":", 1)[1] not in _STATS:
            raise ValueError(f"unknown statistic in {spec!r}; have {list(_STATS)}")
        return
    raise ValueError(
        f"unknown feature {spec!r}; supported: {sorted(FEATURE_NAMES)}"
    )


def _expand(features: Any) -> list[str]:
    """Resolve ``bandpower:*`` style wildcards and validate every entry."""
    out: list[str] = []
    for f in _as_list(features):
        s = str(f).strip()
        if s in ("bandpower:*", "bandpower:all"):
            out += [f"bandpower:{b}" for b in _BAND_ORDER]
            continue
        if s in ("relbandpower:*", "relbandpower:all"):
            out += [f"relbandpower:{b}" for b in _BAND_ORDER]
            continue
        _validate(s)
        out.append(s)
    return out


def feature_names(features: Any = None) -> list[str]:
    """Ordered output names for a feature request.

    ``cov:tangent`` expands to one column per upper-triangular covariance entry;
    on this single-channel montage that is exactly ``cov:0_0``.
    """
    names: list[str] = []
    for s in _expand(features):
        if s == "cov:tangent":
            names.append("cov:0_0")
        else:
            names.append(s)
    return names


# ---------------------------------------------------------------- spectrum
def _welch(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD with the parameters fixed by the catalog spec.

    ``n_fft = min(len, 2*fs)`` gives ~2 Hz resolution and two full seconds of
    data per segment at 250 Hz -- long enough to resolve alpha from theta on a
    single channel. Hann window and 50% overlap keep the variance of the
    estimate down without leaking much. Signals shorter than 8 samples fall
    back to a scaled periodogram, because Welch needs a segment.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    fs = float(fs)
    if n < 8:
        w = np.hanning(n) if n > 2 else np.ones(n)
        xc = x - float(np.mean(x))
        X = np.fft.rfft(xc * w)
        denom = fs * float(np.sum(w ** 2)) or 1.0
        P = (np.abs(X) ** 2) / denom
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        if n > 1:
            if n % 2 == 0:
                P[1:-1] *= 2.0
            else:
                P[1:] *= 2.0
        return f, P
    nper = int(max(8, min(n, int(round(2.0 * fs)))))
    nover = nper // 2
    f, P = sps.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nover,
                     nfft=nper, detrend="constant", scaling="density")
    return f, P


def _band_power(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> float:
    """Integrated power in ``[lo, hi)`` Hz."""
    if f.size < 2 or P.size != f.size:
        return 0.0
    m = (f >= lo) & (f < hi)
    if not np.any(m):
        return 0.0
    df = float(f[1] - f[0])
    return float(np.sum(P[m]) * df)


def _absolute_band_powers(values: np.ndarray, fs: float) -> dict[str, float]:
    f, P = _welch(values, fs)
    return {b: _band_power(f, P, lo, hi) for b, (lo, hi) in BANDS.items()}


def band_powers(values: np.ndarray, fs: float) -> dict[str, float]:
    """log10 absolute band power per canonical band (Welch PSD, integrated).

    Log10 because band power is roughly log-normal: the value is additive in
    dB, and linear decoders behave far better on it. The absolute scale is
    meaningless across sessions on an uncalibrated device, so treat this as a
    within-session feature (see also ``relbandpower``).
    """
    abs_p = _absolute_band_powers(values, fs)
    return {b: float(np.log10(max(v, _LOG_FLOOR))) for b, v in abs_p.items()}


# ---------------------------------------------------------------- per-window
class _Ctx:
    """Caches the PSD and absolute band powers for one window.

    Ten band-power features on the same window must not recompute ten Welch
    transforms: a leave-one-block-out CV over a few hundred epochs would
    otherwise spend most of its time in FFTs.
    """

    __slots__ = ("x", "fs", "times_ms", "_spec", "_abs")

    def __init__(self, values: np.ndarray, fs: float,
                 times_s: np.ndarray | None = None) -> None:
        self.x = np.asarray(values, dtype=np.float64).reshape(-1)
        self.fs = float(fs)
        if times_s is None:
            self.times_ms = np.arange(self.x.size, dtype=np.float64) / max(self.fs, 1e-9) * 1000.0
        else:
            self.times_ms = np.asarray(times_s, dtype=np.float64).reshape(-1) * 1000.0
        self._spec: tuple[np.ndarray, np.ndarray] | None = None
        self._abs: dict[str, float] | None = None

    @property
    def spec(self) -> tuple[np.ndarray, np.ndarray]:
        if self._spec is None:
            self._spec = _welch(self.x, self.fs)
        return self._spec

    @property
    def absolute(self) -> dict[str, float]:
        if self._abs is None:
            f, P = self.spec
            self._abs = {b: _band_power(f, P, lo, hi) for b, (lo, hi) in BANDS.items()}
        return self._abs


def _total_power_1_45(ctx: _Ctx) -> float:
    f, P = ctx.spec
    return max(_band_power(f, P, 1.0, 45.0), 0.0)


def _hjorth(ctx: _Ctx) -> tuple[float, float]:
    x = ctx.x
    if x.size < 3:
        return (float("nan"), float("nan"))
    d1 = np.diff(x)
    var0 = float(np.var(x))
    var1 = float(np.var(d1))
    if var0 <= 0:
        return (float("nan"), float("nan"))
    mob = float(np.sqrt(var1 / var0))
    if var1 <= 0:
        return (mob, float("nan"))
    d2 = np.diff(d1)
    var2 = float(np.var(d2))
    mob_d = float(np.sqrt(var2 / var1))
    comp = mob_d / mob if mob > 0 else float("nan")
    return (mob, comp)


def _spectral_entropy(ctx: _Ctx) -> float:
    f, P = ctx.spec
    m = (f >= 1.0) & (f <= 45.0) & (P > 0)
    if not np.any(m):
        return float("nan")
    p = P[m] / float(np.sum(P[m]))
    h = -float(np.sum(p * np.log(p)))
    return h / float(np.log(p.size)) if p.size > 1 else 0.0


def _dominant_freq(ctx: _Ctx) -> float:
    f, P = ctx.spec
    m = (f >= 1.0) & (f <= 45.0)
    if not np.any(m):
        return float("nan")
    idx = np.flatnonzero(m)
    return float(f[idx[int(np.argmax(P[m]))]])


def _aperiodic(ctx: _Ctx) -> tuple[float, float]:
    """FOOOF-lite: 1/f exponent plus the residual alpha peak height.

    Returns ``(exponent, alpha_peak_log10)``. Exponent is ``-slope`` of
    log10(P) against log10(f), so a steeper 1/f gives a larger exponent.
    Alpha (7.5-12.5 Hz) is excluded from the fit; whatever is left over in that
    band is the alpha peak, reported in log10 power units.
    """
    f, P = ctx.spec
    m = (f >= _FIT_LO) & (f <= _FIT_HI) & (P > 0)
    if int(np.count_nonzero(m)) < 4:
        return (float("nan"), float("nan"))
    lf = np.log10(f[m])
    lp = np.log10(P[m])
    alpha = (f[m] >= _ALPHA_LO) & (f[m] <= _ALPHA_HI)
    fit = ~alpha
    if int(np.count_nonzero(fit)) < 4:
        return (float("nan"), float("nan"))
    slope, intercept = np.polyfit(lf[fit], lp[fit], 1)
    resid = lp - (slope * lf + intercept)
    peak = float(np.max(resid[alpha])) if np.any(alpha) else float("nan")
    return (float(-slope), peak)


def _erp_window(ctx: _Ctx, lo_ms: float, hi_ms: float) -> np.ndarray:
    m = (ctx.times_ms >= lo_ms) & (ctx.times_ms <= hi_ms)
    return ctx.x[m]


def _one(ctx: _Ctx, spec: str) -> float:
    x = ctx.x
    head, _, tail = spec.partition(":")
    if head == "bandpower":
        return float(np.log10(max(ctx.absolute[tail], _LOG_FLOOR)))
    if head == "relbandpower":
        tot = _total_power_1_45(ctx)
        return 100.0 * ctx.absolute[tail] / tot if tot > 0 else 0.0
    if head == "ratio":
        a, b = _split_ratio(tail)
        pa = max(ctx.absolute[a], _LOG_FLOOR)
        pb = max(ctx.absolute[b], _LOG_FLOOR)
        return float(np.log10(pa / pb))
    if head == "erp":
        _, lo, hi = _parse_erp(spec, "erp")
        w = _erp_window(ctx, lo, hi)
        return float(np.mean(w)) if w.size else float("nan")
    if head == "erp_peak":
        _, lo, hi = _parse_erp(spec, "erp_peak")
        w = _erp_window(ctx, lo, hi)
        if not w.size:
            return float("nan")
        i = int(np.argmax(np.abs(w)))
        return float(w[i])
    if head == "hjorth":
        mob, comp = _hjorth(ctx)
        return mob if tail == "mobility" else comp
    if spec == "entropy:spectral":
        return _spectral_entropy(ctx)
    if head == "stats":
        if x.size == 0:
            return float("nan")
        if tail == "std":
            return float(np.std(x))
        if tail == "ptp":
            return float(np.ptp(x))
        if tail == "mean":
            return float(np.mean(x))
        if tail == "skew":
            return float(sstats.skew(x)) if x.size > 2 else float("nan")
        if tail == "kurt":
            return float(sstats.kurtosis(x)) if x.size > 3 else float("nan")
        if tail == "zcr":
            if x.size < 2:
                return 0.0
            dur = x.size / max(ctx.fs, 1e-9)
            return float(np.count_nonzero(np.diff(np.signbit(x)))) / max(dur, 1e-9)
    if spec == "slope:mean":
        return float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    if spec == "peakfreq:dominant":
        return _dominant_freq(ctx)
    if spec == "aperiodic:exponent":
        return _aperiodic(ctx)[0]
    if spec == "aperiodic:alpha_peak":
        return _aperiodic(ctx)[1]
    if spec == "cov:tangent":
        # Single-channel covariance collapses to the variance; expressed as a
        # scalar so that ``extract`` keeps its dict[str, float] contract.
        return float(np.var(x)) if x.size > 1 else float("nan")
    raise ValueError(f"unknown feature {spec!r}; supported: {sorted(FEATURE_NAMES)}")


def extract(values: np.ndarray, fs: float, features: Any = None,
            *, times: np.ndarray | None = None) -> dict[str, float]:
    """Compute named features for one window.

    Parameters
    ----------
    values:
        Windowed signal (one epoch, one sustained-state window, or a whole
        short recording).
    fs:
        Sampling rate.
    features:
        Any subset of :data:`FEATURE_NAMES`. ``None`` uses
        :data:`DEFAULT_FEATURES`.
    times:
        Per-sample times in **seconds** relative to the event. Required for
        ``erp:*``/``erp_peak:*`` when the window does not start at t=0 (i.e.
        always, for a baseline-straddling epoch). Extra keyword: the frozen
        signature had no way to pass it.

    Returns
    -------
    dict
        feature string -> float. Unknown strings raise ``ValueError`` listing
        the supported vocabulary.
    """
    ctx = _Ctx(values, fs, times_s=times)
    return {s: _one(ctx, s) for s in _expand(features)}


def feature_vector(feats: Mapping[str, float], names: Sequence[str] | None = None) -> np.ndarray:
    """Order ``feats`` into a fixed-length float vector.

    Missing keys become NaN (never 0.0): zero is a plausible band-power value
    and would silently bias the decoder, whereas NaN is imputed explicitly.
    """
    keys = list(names) if names is not None else sorted(feats.keys())
    out = np.full(len(keys), np.nan, dtype=np.float64)
    for i, k in enumerate(keys):
        if k in feats:
            try:
                out[i] = float(feats[k])
            except (TypeError, ValueError):
                out[i] = np.nan
    return out


# ---------------------------------------------------------------- epochs
def _epoch_signal(row: np.ndarray) -> np.ndarray:
    """Flatten one epoch row: (samples,) stays, (1, samples) loses the axis."""
    r = np.asarray(row, dtype=np.float64)
    if r.ndim > 1:
        r = r.reshape(-1) if r.shape[0] == 1 else r[0]
    return r.reshape(-1)


def _covariance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Regularised covariance of one epoch, shape ``(ch, ch)``.

    The ridge ``eps * trace(C)/ch * I`` keeps the matrix strictly positive
    definite, which ``logm`` and the matrix square root require. Without it a
    flat channel (variance ~0) produces a singular matrix and the Riemannian
    transform becomes NaN.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ch = arr.shape[0]
    if arr.shape[1] < 2:
        C = np.eye(ch) * max(float(np.var(arr)), 0.0)
    else:
        C = np.atleast_2d(np.cov(arr, bias=False))
    C = np.asarray(C, dtype=np.float64)
    tr = float(np.trace(C)) / ch
    ridge = eps * (tr if tr > 0 else 1.0)
    return C + ridge * np.eye(ch)


def extract_epochs(epochs: Any, spec: Any) -> tuple[np.ndarray, list[str]]:
    """Build the ``(n_epochs, n_features)`` design matrix for a spec.

    Column ordering is derived once from ``spec.features`` and reused for every
    epoch, so the matrix is directly comparable across sessions and across
    callers (the CLI and the agent bridge must agree).

    Special case: when the feature list is exactly ``cov:tangent`` the returned
    array is ``(n_epochs, n_channels, n_channels)`` — covariance matrices, which
    is what :class:`~neurobridge.pipeline.decoders.RiemannianTangentSpace`
    consumes. Mixing ``cov:tangent`` with scalar features is rejected, because
    the result would be neither a covariance nor a feature vector.
    """
    feats_spec = _expand(getattr(spec, "features", None) or DEFAULT_FEATURES)
    n = len(epochs)
    Xsrc = np.asarray(epochs.X)

    if all(f == "cov:tangent" for f in feats_spec):
        ch = 1
        rows = []
        for i in range(n):
            C = _covariance(_epoch_signal(Xsrc[i]) if Xsrc.ndim == 2 else Xsrc[i])
            ch = C.shape[0]
            rows.append(C)
        if rows:
            X = np.stack(rows, axis=0)
        else:
            X = np.zeros((0, ch, ch), dtype=np.float64)
        names = [f"cov:{i}_{j}" for i in range(ch) for j in range(i, ch)]
        return X, names

    if any(f == "cov:tangent" for f in feats_spec):
        raise ValueError(
            "cov:tangent cannot be combined with scalar features: the Riemannian "
            "decoder needs covariance matrices, not a mixed vector"
        )

    names = feature_names(feats_spec)
    out = np.full((n, len(names)), np.nan, dtype=np.float64)
    times = np.asarray(getattr(epochs, "times", np.zeros(0)), dtype=np.float64)
    fs = float(getattr(epochs, "fs", 250.0))
    for i in range(n):
        x = _epoch_signal(Xsrc[i]) if Xsrc.ndim > 1 else np.asarray(Xsrc[i], dtype=np.float64)
        feats = extract(x, fs, feats_spec, times=times if times.size == x.size else None)
        out[i] = feature_vector(feats, names)
    return out, names
