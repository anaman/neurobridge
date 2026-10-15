"""Decoders: small, robust estimators for one noisy frontal channel.

Design rules that come straight from the hardware and the study design:

* **Every model is wrapped in a Pipeline** (NaN imputer -> optional
  standardisation -> estimator). A single bad epoch must not poison the fit,
  and features have wildly different units (log10 power ~ -2, Hjorth mobility
  ~ 0.1, milliseconds-scale ERP means in ADC counts ~ 1e3). Standardising
  per-feature is what makes LDA/logreg/SVM behave.
* **1-channel data must work.** No model may assume a channel dimension
  greater than one. Riemannian geometry on a 1x1 covariance degenerates to
  log-variance, which is a perfectly good (and gain-invariant) feature; EEGNet
  is adapted so its depthwise convolution accepts a single channel.
* **Degenerate input must not raise.** Fewer than 8 epochs, one class, or a
  constant feature matrix are all *normal* outcomes for a short session on a
  dry electrode; the pipeline returns a :class:`TrainedModel` with
  ``meta['skipped']`` set and a majority-class predictor, so the caller can
  report "not trained" rather than crash.
* **Riemannian mean = arithmetic mean of the covariance matrices.** The
  affine-invariant (geodesic) mean is the theoretically correct centre on the
  SPD manifold, but the arithmetic mean is what the original tangent-space
  papers use, it is dramatically cheaper, and it is stable for the small
  covariance matrices here (1x1 to a few channels). Documented so nobody
  assumes otherwise.
* **torch is optional.** ``EEGNet`` is exposed lazily through a module
  ``__getattr__``: importing this module (and therefore the whole pipeline)
  works on a machine with no torch installed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import linalg as sla
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import MODEL_DIR

__all__ = [
    "TrainedModel",
    "train",
    "build_pipeline",
    "save_model",
    "load_model",
    "model_path",
    "RiemannianTangentSpace",
    "EEGNetClassifier",
    "EEGNet",
]

_MODEL_NAMES = ("lda", "logreg", "svm", "riemann", "eegnet")
_MIN_EPOCHS = 8


# ---------------------------------------------------------------- helpers
def _n_rows(X: Any) -> int:
    A = np.asarray(X)
    return int(A.shape[0]) if A.ndim else 0


def _flatten(A: np.ndarray) -> np.ndarray:
    """(n, ch, time) or (n, ch, ch) -> (n, ch*time); 2-D passes through."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 3:
        return A.reshape(A.shape[0], -1)
    if A.ndim == 1:
        return A.reshape(-1, 1)
    return A


def _majority(y: np.ndarray) -> Any:
    vals, counts = np.unique(np.asarray(y).reshape(-1), return_counts=True)
    if vals.size == 0:
        return 0
    return vals[int(np.argmax(counts))]


def _all_constant(X: Any) -> bool:
    """True when no feature carries any variation (or all are NaN)."""
    A = _flatten(X)
    if A.shape[0] == 0 or A.shape[1] == 0:
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stds = np.nanstd(A, axis=0)
    finite = np.isfinite(stds)
    if not np.any(finite):
        return True
    return bool(np.all(stds[finite] <= 1e-12))


def _make_imputer() -> SimpleImputer:
    """Median imputation; all-NaN columns are kept so the shape never changes.

    Kept as empty features (rather than dropped) because a fold that drops a
    different column than another fold would make the out-of-fold predictions
    incomparable.
    """
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:                             # pragma: no cover - old sklearn
        return SimpleImputer(strategy="median")


def _lda() -> LinearDiscriminantAnalysis:
    # 'lsqr' + automatic shrinkage is the only LDA setting that survives a
    # single feature and near-singular covariance, both of which happen here.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _make_clf(name: str, seed: int | None) -> Any:
    if name in ("lda", "riemann"):
        return _lda()
    if name in ("logreg", "logistic", "logistic_regression"):
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                  random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=seed)
    raise ValueError(f"unknown decoder model {name!r}; have {list(_MODEL_NAMES)}")


def _normalize_mode(name: str, decoder: dict[str, Any]) -> str:
    """Resolve ``normalize``.

    Default is z-scoring for the linear models -- band-power features are
    uncalibrated and their raw scale is arbitrary, so a Euclidean-distance
    classifier (LDA/SVM) without standardisation would be dominated by the
    noisiest high-variance feature. The Riemannian and EEGNet paths do their own
    centring (tangent space is centred at the mean; EEGNet z-scores each epoch
    internally), so their default is ``none``.
    """
    norm = decoder.get("normalize", None)
    if norm is None:
        return "none" if name in ("riemann", "eegnet") else "zscore"
    return str(norm).lower()


def build_pipeline(spec: Any, model: str | None = None,
                   seed: int | None = None) -> Any:
    """Assemble the sklearn estimator for a spec's ``decoder`` block.

    Exposed (in addition to :func:`train`) so evaluation can build a *fresh*
    clone per cross-validation fold without re-deriving the recipe.
    """
    decoder = dict(getattr(spec, "decoder", {}) or {})
    name = str(model or decoder.get("model", "lda")).lower()
    if seed is None:
        seed = decoder.get("seed", None)

    if name == "eegnet":
        return _make_eegnet(spec, seed)
    if name == "riemann":
        return Pipeline([
            ("riemann", RiemannianTangentSpace(reg=float(decoder.get("reg", 1e-6)))),
            ("clf", _make_clf("riemann", seed)),
        ])

    steps: list[tuple[str, Any]] = [("imputer", _make_imputer())]
    if _normalize_mode(name, decoder) in ("zscore", "standard", "standardize", "z"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", _make_clf(name, seed)))
    return Pipeline(steps)


def _make_eegnet(spec: Any, seed: int | None) -> "EEGNetClassifier":
    decoder = dict(getattr(spec, "decoder", {}) or {})
    classes = list(getattr(spec, "classes", []) or [])
    return EEGNetClassifier(
        n_classes=max(2, len(classes)),
        epochs=int(decoder.get("epochs", 80)),
        batch_size=int(decoder.get("batch_size", 32)),
        lr=float(decoder.get("lr", 1e-3)),
        dropout=float(decoder.get("dropout", 0.25)),
        weight_decay=float(decoder.get("weight_decay", 1e-4)),
        seed=int(seed if seed is not None else decoder.get("seed", 0) or 0),
    )


# ---------------------------------------------------------------- model object
@dataclass
class TrainedModel:
    """A fitted pipeline plus the provenance needed to report it honestly."""

    spec_id: str
    name: str
    feature_names: list[str]
    classes: list[str]
    pipeline: Any
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return bool(self.meta.get("skipped"))

    def _default(self) -> Any:
        maj = self.meta.get("majority", 0)
        try:
            arr = np.asarray([maj])
            return arr.dtype
        except Exception:                         # pragma: no cover
            return np.dtype(object)

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels; falls back to the majority class if training skipped."""
        n = _n_rows(X)
        if self.skipped or self.pipeline is None:
            maj = self.meta.get("majority", 0)
            return np.full(n, maj)
        return np.asarray(self.pipeline.predict(np.asarray(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities; one-hot of the majority class when skipped."""
        n = _n_rows(X)
        if self.skipped or self.pipeline is None:
            seen = list(self.meta.get("classes_seen") or self.classes or [0])
            k = max(1, len(seen))
            P = np.zeros((n, k), dtype=float)
            idx = int(self.meta.get("majority_index", 0) or 0)
            P[:, min(max(idx, 0), k - 1)] = 1.0
            return P
        if hasattr(self.pipeline, "predict_proba"):
            return np.asarray(self.pipeline.predict_proba(np.asarray(X)))
        pred = np.asarray(self.pipeline.predict(np.asarray(X))).reshape(-1)
        uniq = np.unique(pred)
        P = np.zeros((n, max(1, uniq.size)), dtype=float)
        for j, u in enumerate(uniq):
            P[pred == u, j] = 1.0
        return P

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable description (no estimator object)."""
        return {
            "spec_id": self.spec_id,
            "name": self.name,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "meta": dict(self.meta),
            "fitted": bool(self.pipeline is not None and not self.skipped),
        }


# ---------------------------------------------------------------- training
def train(spec: Any, X: Any, y: Any, groups: Any = None,
          feature_names: Sequence[str] | None = None,
          seed: int | None = None) -> TrainedModel:
    """Fit a decoder, or explain why fitting was not possible.

    ``groups`` is accepted for signature symmetry and for provenance (it is
    recorded in ``meta``); the actual grouped cross-validation happens in
    :mod:`neurobridge.pipeline.evaluate`, because a model trained on all epochs
    has no notion of held-out groups.

    Never raises for degenerate input: fewer than 8 epochs, a single class, or
    an all-constant feature matrix each produce a skipped model whose
    ``predict`` returns the majority class.
    """
    Xa = np.asarray(X)
    ya = np.asarray(y).reshape(-1)
    decoder = dict(getattr(spec, "decoder", {}) or {})
    name = str(decoder.get("model", "lda")).lower()
    classes = list(getattr(spec, "classes", []) or [])
    fnames = list(feature_names or [])
    n = _n_rows(Xa)
    meta: dict[str, Any] = {
        "decoder": name,
        "cv": str(decoder.get("cv", "leave_one_block_out")).lower(),
        "normalize": _normalize_mode(name, decoder),
        "n_samples": n,
        "n_features": int(_flatten(Xa).shape[1]) if Xa.ndim >= 2 else 0,
        "n_groups": int(len(np.unique(groups))) if groups is not None and np.size(groups) else 0,
        "classes": classes,
        "warnings": [],
    }
    if not classes:
        classes = [str(v) for v in np.unique(ya)]
    seen = np.unique(ya) if ya.size else np.asarray([])
    meta["classes_seen"] = [v.item() if hasattr(v, "item") else v for v in seen]
    maj = _majority(ya)
    meta["majority"] = maj.item() if hasattr(maj, "item") else maj
    try:
        meta["majority_index"] = int(np.flatnonzero(seen == maj)[0])
    except Exception:                             # pragma: no cover
        meta["majority_index"] = 0

    reasons: list[str] = []
    if ya.size != n:
        reasons.append(f"X/y length mismatch ({n} rows vs {ya.size} labels)")
    if n < _MIN_EPOCHS:
        reasons.append(f"fewer_than_{_MIN_EPOCHS}_epochs (n={n})")
    if seen.size < 2:
        reasons.append("only_one_class")
    if _all_constant(Xa):
        reasons.append("all_features_constant_or_nan")

    if reasons:
        meta["skipped"] = "; ".join(reasons)
        return TrainedModel(str(getattr(spec, "id", "")), name, fnames, classes,
                            None, meta)

    try:
        pipe = build_pipeline(spec, name, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xa, ya)
    except Exception as exc:                      # noqa: BLE001 - must not raise
        meta["skipped"] = f"fit_failed: {type(exc).__name__}: {exc}"
        return TrainedModel(str(getattr(spec, "id", "")), name, fnames, classes,
                            None, meta)

    meta["fitted_at"] = "train()"
    return TrainedModel(str(getattr(spec, "id", "")), name, fnames, classes,
                        pipe, meta)


# ---------------------------------------------------------------- persistence
def model_path(spec_id: str) -> Path:
    """``data/models/<spec_id>.joblib`` (directory created on demand)."""
    p = Path(MODEL_DIR) / f"{spec_id}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model: TrainedModel, path: str | Path | None = None) -> Path:
    """Persist a model with joblib.

    If the estimator itself cannot be serialised (a torch module on an exotic
    build), the provenance is still written, with ``save_error`` recorded: a
    reloadable record beats an exception during a long session teardown.
    """
    p = Path(path) if path is not None else model_path(model.spec_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_id": model.spec_id,
        "name": model.name,
        "feature_names": list(model.feature_names),
        "classes": list(model.classes),
        "pipeline": model.pipeline,
        "meta": dict(model.meta),
    }
    try:
        joblib.dump(payload, p)
    except Exception as exc:                      # noqa: BLE001
        payload["pipeline"] = None
        payload["meta"] = dict(model.meta)
        payload["meta"]["save_error"] = f"{type(exc).__name__}: {exc}"
        joblib.dump(payload, p)
    return p


def load_model(spec_id_or_path: str | Path) -> TrainedModel:
    """Load a model saved by :func:`save_model`.

    Accepts an explicit ``.joblib`` path or a spec id (resolved through
    :func:`model_path`).
    """
    raw = Path(spec_id_or_path)
    p = raw if raw.exists() and raw.is_file() else model_path(str(spec_id_or_path))
    if not p.exists():
        raise FileNotFoundError(f"no saved model at {p}")
    d = joblib.load(p)
    if isinstance(d, TrainedModel):               # tolerate a bare dataclass dump
        return d
    return TrainedModel(
        spec_id=str(d.get("spec_id", p.stem)),
        name=str(d.get("name", "")),
        feature_names=list(d.get("feature_names", []) or []),
        classes=list(d.get("classes", []) or []),
        pipeline=d.get("pipeline", None),
        meta=dict(d.get("meta", {}) or {}),
    )


# ---------------------------------------------------------------- Riemannian
def _sym(C: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(C, dtype=np.float64) + np.asarray(C, dtype=np.float64).T)


def _regularize(C: np.ndarray, reg: float) -> np.ndarray:
    """Make a covariance SPD: symmetrise, scrub NaN, add a ridge."""
    C = _sym(np.nan_to_num(np.asarray(C, dtype=np.float64), nan=0.0,
                           posinf=0.0, neginf=0.0))
    ch = C.shape[0]
    tr = float(np.trace(C)) / max(ch, 1)
    ridge = float(reg) * (tr if tr > 0 else 1.0)
    return C + ridge * np.eye(ch)


def _cov_of(x: np.ndarray, reg: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] < 2:
        C = np.eye(arr.shape[0]) * max(float(np.var(arr)), 0.0)
    else:
        C = np.atleast_2d(np.cov(arr, bias=False))
    return _regularize(C, reg)


class RiemannianTangentSpace(BaseEstimator, TransformerMixin):
    """Map SPD covariance matrices to the tangent space at their mean.

    Implements the standard Barachant-style pipeline without pyriemann:

    ``S_i = logm(M^-1/2 C_i M^-1/2)``, then vectorise the upper triangle
    (off-diagonal entries scaled by ``sqrt(2)`` so the Euclidean norm of the
    vector equals the Frobenius norm of the symmetric matrix).

    Accepted input shapes:

    * ``(n_samples, n_channels, n_time)`` — raw epochs; covariances are
      estimated internally;
    * ``(n_samples, n_channels, n_channels)`` — covariance matrices;
    * ``(n_samples, n_features)`` — either flattened symmetric matrices (when
      ``n_features`` is a perfect square) or a single-channel time series
      (otherwise). The single-channel case is exactly this montage, where a
      "covariance matrix" is 1x1 and the tangent projection reduces to
      log-variance about the mean variance — gain-invariant, which is what an
      uncalibrated ADC requires.

    The mean is the arithmetic mean of the covariances (see the module
    docstring for why that is an acceptable choice here).
    """

    def __init__(self, reg: float = 1e-6, vectorize: bool = True,
                 eps: float = 1e-10) -> None:
        self.reg = reg
        self.vectorize = vectorize
        self.eps = eps

    # -- internals --------------------------------------------------
    def _to_covs(self, X: Any) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        if A.ndim == 3:
            if A.shape[1] == A.shape[2]:
                return np.stack([_regularize(A[i], self.reg) for i in range(A.shape[0])]) \
                    if A.shape[0] else A.copy()
            return np.stack([_cov_of(A[i], self.reg) for i in range(A.shape[0])]) \
                if A.shape[0] else np.zeros((0, A.shape[1], A.shape[1]))
        if A.ndim == 2:
            n, nf = A.shape
            r = int(round(np.sqrt(nf)))
            if r > 0 and r * r == nf:
                return np.stack([_regularize(A[i].reshape(r, r), self.reg)
                                 for i in range(n)]) if n else np.zeros((0, r, r))
            return np.stack([_cov_of(A[i], self.reg) for i in range(n)]) \
                if n else np.zeros((0, 1, 1))
        if A.ndim == 1:
            return _cov_of(A, self.reg)[None, :, :]
        raise ValueError(f"RiemannianTangentSpace: unsupported input shape {A.shape}")

    def _sqrt_factors(self) -> None:
        w, V = sla.eigh(self.mean_)
        w = np.clip(w, self.eps, None)
        self._isqrt_ = (V * (w ** -0.5)) @ V.T
        self._sqrt_ = (V * (w ** 0.5)) @ V.T

    def _vectorize(self, S: np.ndarray) -> np.ndarray:
        ch = S.shape[0]
        iu = np.triu_indices(ch)
        v = S[iu].copy()
        off = iu[0] != iu[1]
        v[off] *= np.sqrt(2.0)
        return v

    def _unvectorize(self, v: np.ndarray) -> np.ndarray:
        ch = self.n_channels_
        S = np.zeros((ch, ch), dtype=np.float64)
        iu = np.triu_indices(ch)
        vals = np.array(v, dtype=np.float64, copy=True)
        off = iu[0] != iu[1]
        vals[off] /= np.sqrt(2.0)
        S[iu] = vals
        S = S + S.T - np.diag(np.diag(S))
        return S

    # -- sklearn API ------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> "RiemannianTangentSpace":
        C = self._to_covs(X)
        if C.shape[0] == 0:
            raise ValueError("RiemannianTangentSpace.fit: no samples")
        self.mean_ = _regularize(C.mean(axis=0), self.reg)
        self._sqrt_factors()
        self.n_channels_ = int(self.mean_.shape[0])
        self.n_features_in_ = int(np.prod(np.asarray(X).shape[1:])) if np.asarray(X).ndim > 1 else 1
        self.n_features_out_ = self.n_channels_ * (self.n_channels_ + 1) // 2
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("RiemannianTangentSpace.transform called before fit")
        C = self._to_covs(X)
        out = np.empty((C.shape[0], self.n_channels_ * (self.n_channels_ + 1) // 2),
                       dtype=np.float64)
        for i in range(C.shape[0]):
            W = _sym(self._isqrt_ @ C[i] @ self._isqrt_)
            L = sla.logm(W)
            L = _sym(np.real(L))
            out[i] = self._vectorize(L) if self.vectorize else L.reshape(-1)
        return out

    def fit_transform(self, X: Any, y: Any = None, **kw: Any) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, T: Any) -> np.ndarray:
        """Map tangent vectors back to SPD matrices (uses ``expm``).

        Provided because a tangent-space model is only interpretable if you can
        go back: the distance of a projected point from the origin *is* the
        Riemannian distance to the mean covariance.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError("inverse_transform called before fit")
        A = np.asarray(T, dtype=np.float64)
        if A.ndim == 1:
            A = A[None, :]
        out = np.empty((A.shape[0], self.n_channels_, self.n_channels_), dtype=np.float64)
        for i in range(A.shape[0]):
            S = self._unvectorize(A[i]) if self.vectorize else _sym(A[i].reshape(
                self.n_channels_, self.n_channels_))
            out[i] = _sym(self._sqrt_ @ np.real(sla.expm(S)) @ self._sqrt_)
        return out


# ---------------------------------------------------------------- EEGNet
class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around the compact 1-channel EEGNet.

    torch is imported inside ``fit`` only, so importing this module (and the
    whole pipeline) never requires torch. Each epoch is z-scored over time
    before it reaches the network: the headband is uncalibrated, so absolute
    amplitudes would otherwise vary by orders of magnitude between sessions.
    """

    def __init__(self, n_classes: int = 2, n_time: int | None = None,
                 n_ch: int = 1, epochs: int = 80, batch_size: int = 32,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 dropout: float = 0.25, seed: int = 0, verbose: bool = False,
                 F1: int = 8, D: int = 2, F2: int = 16) -> None:
        self.n_classes = n_classes
        self.n_time = n_time
        self.n_ch = n_ch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.verbose = verbose
        self.F1 = F1
        self.D = D
        self.F2 = F2

    # -- helpers ----------------------------------------------------
    @staticmethod
    def _as_3d(X: Any) -> np.ndarray:
        A = np.asarray(X, dtype=np.float32)
        if A.ndim == 3:
            return A
        if A.ndim == 2:
            return A[:, None, :]
        if A.ndim == 1:
            return A[None, None, :]
        raise ValueError(f"EEGNetClassifier: unsupported input shape {A.shape}")

    def _net(self, n_ch: int, n_time: int, n_classes: int) -> Any:
        cls = _build_eegnet_class()
        return cls(n_ch=n_ch, n_time=n_time, n_classes=n_classes,
                   F1=int(self.F1), D=int(self.D), F2=int(self.F2),
                   dropout=float(self.dropout))

    # -- sklearn API ------------------------------------------------
    def fit(self, X: Any, y: Any) -> "EEGNetClassifier":
        import torch

        X3 = self._as_3d(X)
        ya = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(ya)
        y_idx = np.searchsorted(self.classes_, ya).astype(np.int64)
        n_ch, n_time = int(X3.shape[1]), int(X3.shape[2])
        n_classes = int(self.n_classes or self.classes_.size)
        n_classes = max(n_classes, int(self.classes_.size))

        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed) % (2 ** 31))
        net = self._net(n_ch, n_time, n_classes)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr),
                               weight_decay=float(self.weight_decay))
        lossf = torch.nn.CrossEntropyLoss()

        Xt = torch.tensor(np.nan_to_num(X3, nan=0.0, posinf=0.0, neginf=0.0),
                          dtype=torch.float32)
        mu = Xt.mean(dim=2, keepdim=True)
        sd = Xt.std(dim=2, keepdim=True) + 1e-6
        Xt = (Xt - mu) / sd
        yt = torch.tensor(y_idx, dtype=torch.long)

        net.train()
        bs = max(1, int(self.batch_size))
        for ep in range(max(1, int(self.epochs))):
            perm = torch.randperm(Xt.shape[0])
            for i in range(0, Xt.shape[0], bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            if self.verbose:                      # pragma: no cover - debug only
                print(f"  eegnet epoch {ep + 1}/{self.epochs} loss={float(loss):.4f}")
        net.eval()
        self.net_ = net
        self.n_ch_ = n_ch
        self.n_time_ = n_time
        return self

    def _logits(self, X: Any) -> np.ndarray:
        import torch
        if not hasattr(self, "net_"):
            raise RuntimeError("EEGNetClassifier.predict called before fit")
        X3 = self._as_3d(X)
        Xt = torch.tensor(np.nan_to_num(X3, nan=0.0, posinf=0.0, neginf=0.0),
                          dtype=torch.float32)
        mu = Xt.mean(dim=2, keepdim=True)
        sd = Xt.std(dim=2, keepdim=True) + 1e-6
        Xt = (Xt - mu) / sd
        with torch.no_grad():
            out = self.net_(Xt)
        return out.detach().cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


_EEGNET_CLASS: Any = None


def _build_eegnet_class() -> Any:
    """Build (once) the ``nn.Module`` class and cache it at module level.

    Caching plus a stable ``__qualname__`` in this module's globals is what
    lets joblib pickle a trained EEGNet: pickle resolves the class by name, and
    the module ``__getattr__`` below returns the *same* cached object.
    """
    global _EEGNET_CLASS
    if _EEGNET_CLASS is not None:
        return _EEGNET_CLASS
    import torch
    from torch import nn

    class EEGNet(nn.Module):
        """Compact EEGNet for one channel and short epochs.

        Two convolution blocks (temporal -> depthwise spatial), then a
        separable convolution, then a dense layer. Compared with the published
        EEGNet this one keeps ``F1``/``F2`` small (8/16), uses
        ``AdaptiveAvgPool2d`` so any epoch length works, and applies the
        depthwise kernel over the single channel (kernel ``(n_ch, 1)``), which
        degenerates to a pointwise scaling -- the honest thing to do when there
        is no spatial dimension to exploit.
        """

        def __init__(self, n_ch: int = 1, n_time: int = 250, n_classes: int = 2,
                     F1: int = 8, D: int = 2, F2: int = 16,
                     dropout: float = 0.25) -> None:
            super().__init__()
            F1, D, F2 = int(F1), int(D), int(F2)
            k1 = int(min(64, max(3, n_time // 2)))
            if k1 % 2 == 0:
                k1 += 1
            k2 = int(min(16, max(3, n_time // 8)))
            if k2 % 2 == 0:
                k2 += 1
            p1 = 4 if n_time >= 16 else 2
            p2 = 4 if n_time // p1 >= 8 else 2
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU(),
                nn.AvgPool2d((1, p1)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, p2)),
                nn.Dropout(dropout),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, k2), padding=(0, k2 // 2), bias=False),
                nn.Conv2d(F2, F2, (1, 1), groups=F2, bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(F2 * 8, int(n_classes))

        def forward(self, x: Any) -> Any:
            if x.dim() == 2:                      # (n, time) -> (n, 1, 1, time)
                x = x[:, None, None, :]
            if x.dim() == 3:                      # (n, ch, time)
                x = x[:, :, None, :]
            x = self.block1(x)
            x = self.block2(x)
            x = self.separable(x)
            x = x.flatten(1)
            return self.head(x)

    EEGNet.__qualname__ = "EEGNet"
    EEGNet.__module__ = __name__
    _EEGNET_CLASS = EEGNet
    return EEGNet


def __getattr__(name: str) -> Any:
    """Lazily expose ``EEGNet`` (PEP 562) so torch stays optional at import."""
    if name == "EEGNet":
        return _build_eegnet_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
