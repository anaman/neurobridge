"""Evaluation: honest numbers for one subject, many short blocks.

What this module refuses to do
------------------------------
* **It does not report trial-level cross-validation as if it were a skill
  estimate.** Trials inside a block share drift, contact state and fatigue, so
  every fold here holds out *whole blocks*. That is the difference between "the
  decoder can tell two blocks apart" and "the decoder can tell two conditions
  apart".
* **It does not invent an accuracy for a test that has only one class.** Five
  specs (T002, T004, T005, T006, T107) are single-class characterisations whose
  gates are quality/N400-style metrics; for those, accuracy gates are marked
  *inapplicable* and the computable metrics decide the verdict.
* **It does not silently pass a gate it could not evaluate.** A missing metric
  is ``ok: false`` with ``reason: 'metric not computed'``; a genuinely
  inapplicable metric is ``applicable: false`` and is excluded from the verdict.
* **It does not raise.** A session with no epochs, one class, or a constant
  feature matrix produces a result dict that says so.

``passed`` is true only when every *evaluable* gate check passed and none were
skipped for a missing metric.
"""
from __future__ import annotations

import contextlib
import re
import warnings
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.base import clone
from sklearn.utils import check_random_state

from ..catalog.schema import TestSpec
from ..device.quality import alpha_reactivity
from .decoders import TrainedModel, build_pipeline, load_model, model_path, save_model, train
from .epochs import Epochs, epochs_from_session, windowed_epochs
from .features import DEFAULT_FEATURES, extract_epochs, feature_names
from .preprocess import preprocess_session

__all__ = [
    "cross_validate",
    "permutation_test",
    "information_transfer_rate",
    "apply_gates",
    "evaluate_session",
    "train_and_evaluate",
    "assert_signal_only",
    "timing_quality_of",
    "resolve_grouping",
    "epoching_mode",
    "build_epochs",
    "coverage_of",
    "false_activations_per_min",
]

#: Gates where a smaller value is the better outcome.
_LOWER_IS_BETTER = {
    "permutation_p", "p_value", "false_positive_rate", "artifact_fraction",
    "latency_jitter_ms", "saturation_pct", "flatline_pct", "drift_ratio",
    "mains_pct", "motion_jump_pct", "miss_rate",
}

#: Acceptance metric -> (source path in the result dict, human note).
_METRIC_SOURCES: dict[str, str] = {
    "balanced_accuracy": "cv.balanced_accuracy",
    "accuracy": "cv.balanced_accuracy",
    "ba": "cv.balanced_accuracy",
    "auc": "cv.auc",
    "roc_auc": "cv.auc",
    "f1": "cv.f1",
    "macro_f1": "cv.f1",
    "chance": "cv.chance",
    "n": "cv.n",
    "n_epochs": "cv.n",
    "permutation_p": "permutation.p_value",
    "p_value": "permutation.p_value",
    "p": "permutation.p_value",
    "itr": "itr_bits_per_min",
    "information_transfer_rate": "itr_bits_per_min",
    "artifact_fraction": "quality_metrics.artifact_fraction",
    "saturation_pct": "quality_metrics.saturation_pct",
    "flatline_pct": "quality_metrics.flatline_pct",
    "drift_ratio": "quality_metrics.drift_ratio",
    "band_delta_pct": "quality_metrics.band_delta_pct",
    "mains_pct": "quality_metrics.mains_pct",
    "motion_jump_pct": "quality_metrics.motion_jump_pct",
    "alpha_ratio_closed_open": "derived_metrics.alpha_ratio_closed_open",
    "response_snr": "derived_metrics.response_snr",
    "latency_jitter_ms": "derived_metrics.latency_jitter_ms",
    "latency_offset_ms": "derived_metrics.latency_offset_ms",
    "false_positive_rate": "derived_metrics.false_positive_rate",
    "target_recall": "derived_metrics.target_recall",
    "miss_rate": "derived_metrics.miss_rate",
}

#: Metrics that only make sense with a between-class contrast.
_ACCURACY_METRICS = ("balanced_accuracy", "accuracy", "ba", "auc", "roc_auc", "f1", "macro_f1")


# ---------------------------------------------------------------- helpers
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/Path/dataclass values into plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "as_dict"):
        return _jsonable(obj.as_dict())
    return str(obj)


@contextlib.contextmanager
def _quiet_fit():
    """Silence sklearn's degenerate-fold chatter (we report those folds ourselves).

    A fold whose training set has one sample per class makes Ledoit-Wolf warn and
    makes ``balanced_accuracy_score`` warn about a single label in a fold. Those
    folds are already named in ``cv['notes']``, so the warnings add noise without
    adding information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _is_trial_based(spec: TestSpec) -> bool:
    """True for discrete-trial designs (same rule the offline simulator uses)."""
    d = dict(getattr(spec, "design", {}) or {})
    return ("soa_s" in d and "trials" in d and float(d.get("soa_s") or 0) > 0)


# ---------------------------------------------------------------- signal-only guard
#: Feature families that are computed from the signal (and time) alone.
_SIGNAL_ONLY_FAMILIES = {
    "bandpower", "relbandpower", "ratio", "erp", "erp_peak", "hjorth",
    "entropy", "stats", "slope", "peakfreq", "aperiodic", "cov",
}

#: Tokens that would betray a *context* feature: anything the experimenter knew
#: before the signal arrived (which trial, which stimulus, what the participant
#: answered, ...). Checked against the family and against the non-component
#: parts of a feature name.
_FORBIDDEN_TOKENS = re.compile(
    r"\b(trial|test|test_id|session|session_id|block|block_id|filename|path|file|"
    r"stim|stim_id|stimulus|image|audio|clip|word|cue|answer|response|rt|rating|"
    r"label|class_id|requested|order|index|future|context|subject|participant|"
    r"task_label|reported_label|predicted_label)\b",
    re.IGNORECASE,
)


def assert_signal_only(names: Sequence[str]) -> bool:
    """Fail loudly if a feature name could carry experimental context.

    The Codex protocol (``docs/EEG_AGENT_TRAINING_PROTOCOL.md`` §9.2) forbids
    model inputs containing trial id, test id, stimulus identity, filename, cue
    word, answer, rating, response latency, requested class, trial order or
    future samples. Every feature NeuroBridge implements is derived from the
    signal and its own time base, so this guard should never fire -- which is
    exactly why it is worth asserting rather than assuming.

    A context-assisted model is a *different* model: it must be evaluated
    separately and compared against a context-only baseline, so this function
    refuses to let context leak into the signal-only feature matrix. Returns
    ``True``; raises :class:`ValueError` with the offending names otherwise.
    """
    bad: list[tuple[str, str]] = []
    for raw in names:
        name = str(raw)
        fam = name.split(":", 1)[0]
        if fam not in _SIGNAL_ONLY_FAMILIES:
            bad.append((name, "unknown feature family (context-assisted features are not "
                              "part of the signal-only model)"))
            continue
        parts = name.split(":")
        scan = ":".join([parts[0]] + parts[2:]) if parts[0] in ("erp", "erp_peak") else name
        if _FORBIDDEN_TOKENS.search(scan.replace("_", " ")):
            bad.append((name, "context-like token in the feature name"))
    if bad:
        detail = "; ".join(f"{n} ({why})" for n, why in bad)
        raise ValueError(
            f"signal-only guard failed: {detail}. Model inputs must not contain trial/test "
            "identity, stimulus identity, cue, answer, rating, latency or ordering "
            "information; evaluate context-assisted models separately against a "
            "context-only baseline.")
    return True


def timing_quality_of(session: Any, spec: TestSpec) -> tuple[str, list[str]]:
    """``verified`` | ``coarse`` | ``unverified`` for this session's timestamps.

    * ``verified`` -- the markers and samples share one monotonic clock *and* the
      output latency was measured (or the data is synthetic, where it is exact by
      construction);
    * ``coarse`` -- an imported packet log or a bridge-side software timestamp:
      fine for band power over long windows, not fine for ERP latency;
    * ``unverified`` -- nothing establishes the offset, so ERP latencies must not
      be reported as findings (amplitudes and patterns only).
    """
    notes: list[str] = []
    meta = dict(getattr(session, "meta", {}) or {})
    declared = str(meta.get("timing_quality", "") or "").lower()
    if declared in ("verified", "coarse", "unverified"):
        quality = declared
    elif meta.get("synthetic"):
        quality = "verified"
        notes.append("synthetic session: marker and sample times share one clock by construction")
    else:
        imported = any(k in meta for k in ("imported", "packet_file", "packets_file",
                                           "ble_packets", "imported_from", "source_file"))
        if str(meta.get("source", "")).lower() in ("import", "imported", "packets", "ble_file"):
            imported = True
        from ..config import get_settings
        proc = dict(getattr(spec, "processing", {}) or {})
        measured = bool(proc.get("latency_offset_s") or proc.get("latency_offset_ms")
                        or meta.get("output_latency_offset_s")
                        or get_settings().output_latency_offset_s)
        quality = "coarse" if imported else ("verified" if measured else "unverified")
    if quality != "verified":
        notes.append(
            f"timing_quality={quality}: ERP *latency* differences must not be presented as "
            "findings from this session; amplitude, pattern and band-power results only")
    return quality, notes


def resolve_grouping(spec: TestSpec, groups: Any, groups_fine: Any = None,
                     session_ids: Any = None) -> tuple[Any, str, str]:
    """Pick the coarsest available CV grouping: session > block > trial.

    Returns ``(array, group_by, source)``. ``spec.decoder['group_by']`` selects
    the *finest* unit that is still honest for the claim: ``block`` (default),
    ``trial`` (overlapping windows from one trial are one observation) or
    ``session`` (a held-out day). The source is reported so a reader can see
    that "accuracy" came with a grouping and which one.
    """
    decoder = dict(getattr(spec, "decoder", {}) or {})
    want = str(decoder.get("group_by", "block") or "block").lower()
    if want in ("trial", "trial_id", "fine"):
        if groups_fine is not None and np.size(groups_fine):
            return np.asarray(groups_fine).reshape(-1), "trial", "groups_fine"
        return (None if groups is None else np.asarray(groups).reshape(-1)), "trial", \
            "groups (no trial ids recorded)"
    if want in ("session", "day", "session_id"):
        if session_ids is not None and np.size(session_ids):
            return np.asarray(session_ids).reshape(-1), "session", "session_ids"
        return (None if groups is None else np.asarray(groups).reshape(-1)), "session", \
            "groups (single-session evaluation)"
    return (None if groups is None else np.asarray(groups).reshape(-1)), "block", "groups"


def epoching_mode(spec: TestSpec) -> str:
    """``'windowed'`` or ``'stimulus'``, from the spec (and its design shape).

    Declared ``processing.epoching`` wins. Without it, a spec that carries
    ``window_s`` *and* is not a discrete-trial design is a sustained-state test
    and gets windows; everything else is stimulus-locked.
    """
    proc = dict(getattr(spec, "processing", {}) or {})
    ep = str(proc.get("epoching", "") or "").lower()
    if ep in ("windowed", "window", "sustained"):
        return "windowed"
    if ep in ("stimulus", "stimulus_locked", "trial", "epoched", "erp"):
        return "stimulus"
    if "window_s" in proc and not _is_trial_based(spec):
        return "windowed"
    return "stimulus"


def _class_names(spec: TestSpec, y: np.ndarray) -> list[str]:
    """Canonical class names for the labels present in ``y``.

    Integer labels index ``spec.classes``; anything else is treated as a label
    itself. Only *present* classes are returned, because chance level (1/k) must
    reflect the classes the model can actually choose between.
    """
    ya = np.asarray(y).reshape(-1)
    present = [v for v in np.unique(ya)]
    classes = list(getattr(spec, "classes", []) or [])
    out: list[str] = []
    for v in present:
        if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(classes):
            out.append(str(classes[int(v)]))
        elif isinstance(v, (int, np.integer)):
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out


def _class_labels(spec: TestSpec, y: np.ndarray) -> np.ndarray:
    """Map labels to canonical strings for display / confusion ordering."""
    ya = np.asarray(y).reshape(-1)
    classes = list(getattr(spec, "classes", []) or [])
    if ya.dtype.kind in "iu":
        return np.array([str(classes[int(v)]) if 0 <= int(v) < len(classes) else str(int(v))
                         for v in ya], dtype=object)
    return np.array([str(v) for v in ya], dtype=object)


class _FixedSplits:
    """A splitter built from an explicit list of ``(train, test)`` index arrays.

    Used instead of a live sklearn splitter because folds whose *training* set is
    missing a class are dropped (with a note) before anything is fitted:
    a fold that cannot see a class cannot predict it, and silently averaging
    such a fold into the score would be dishonest.
    """

    def __init__(self, splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        self.splits = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
                       for a, b in splits]

    def split(self, X: Any = None, y: Any = None, groups: Any = None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for tr, te in self.splits:
            yield tr, te

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return len(self.splits)


def _cv_plan(spec: TestSpec, X: Any, y: np.ndarray, groups: Any,
             classes: Sequence[str], seed: int | None
             ) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Pick the splitter named by ``decoder.cv`` and enumerate its folds.

    Grouped CV is the default and the point: with one subject and short blocks,
    an ungrouped split would leak drift. ``leave_one_block_out`` is implemented
    as ``LeaveOneGroupOut`` over the block ids; ``stratified_group_kfold`` keeps
    class balance while never splitting a block; ``stratified_kfold`` is the
    honest fallback when a session carries no block information.
    """
    notes: list[str] = []
    cv_name = str((getattr(spec, "decoder", {}) or {}).get("cv", "leave_one_block_out")).lower()
    ya = np.asarray(y).reshape(-1)
    g = None if groups is None or np.size(groups) == 0 else np.asarray(groups).reshape(-1)
    n = int(ya.size)
    counts = {c: int(np.count_nonzero(ya == c)) for c in np.unique(ya)}
    min_count = min(counts.values()) if counts else 0
    n_groups = int(len(np.unique(g))) if g is not None else 0
    seed = int(seed or 0)

    splitter: Any = None
    if cv_name in ("leave_one_block_out", "lobo", "leave_one_group_out", "logo"):
        if g is not None and n_groups >= 2:
            splitter = LeaveOneGroupOut()
            notes.append(f"leave-one-block-out over {n_groups} blocks/groups")
        else:
            # No silent relaxation: a random split would look better and claim
            # less. The requested scheme cannot be formed, so say so.
            notes.append(
                "leave_one_block_out cannot be formed: this capture has "
                f"{n_groups} block(s) of block information; accuracy is not evaluable "
                "(a relaxed split would quietly weaken the claim)")
            return None, [], notes
    if splitter is None and cv_name in ("stratified_group_kfold", "sgkf", "group_kfold"):
        if g is not None and n_groups >= 2:
            n_splits = min(5, n_groups, min_count)
            if n_splits >= 2:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                                random_state=seed)
                notes.append(f"StratifiedGroupKFold({n_splits}) over {n_groups} groups")
            else:
                notes.append(
                    f"StratifiedGroupKFold cannot be formed ({n_groups} group(s), smallest "
                    f"class {min_count}); accuracy is not evaluable rather than split at random")
                return None, [], notes
        else:
            notes.append(
                "stratified_group_kfold cannot be formed: no block ids were recorded; "
                "accuracy is not evaluable rather than split at random")
            return None, [], notes
    if splitter is None:
        n_splits = min(5, min_count)
        if n_splits < 2:
            notes.append(f"cannot split {n} epochs into 2 folds with the smallest class "
                         f"having {min_count} members")
            return None, [], notes
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        notes.append(f"StratifiedKFold({n_splits}) as requested by the spec's decoder.cv")

    raw = list(splitter.split(np.zeros((n, 1)), ya, g))
    usable: list[tuple[np.ndarray, np.ndarray]] = []
    dropped = 0
    for tr, te in raw:
        if te.size == 0:
            continue
        if len(np.unique(ya[tr])) < 2:
            dropped += 1
            continue
        usable.append((np.asarray(tr), np.asarray(te)))
    if dropped:
        notes.append(f"{dropped} fold(s) dropped: their training set contained a single class "
                     "(that class occupies a single block, so no honest out-of-fold prediction "
                     "for it exists under this split)")
    if len(usable) < 2:
        notes.append("fewer than 2 usable folds after dropping degenerate ones: accuracy is "
                     "not evaluable for this capture")
        return None, [], notes
    return _FixedSplits(usable), usable, notes


def _estimator(spec: TestSpec, model: Any, seed: int | None) -> Any:
    """A fresh, unfitted estimator for one fold."""
    if isinstance(model, TrainedModel) and model.pipeline is not None:
        try:
            return clone(model.pipeline)
        except Exception:
            return model.pipeline
    name = str(model) if isinstance(model, str) else None
    return build_pipeline(spec, name, seed)


def _seed_of(spec: TestSpec, seed: int | None = None) -> int:
    if seed is not None:
        return int(seed)
    dec = dict(getattr(spec, "decoder", {}) or {})
    try:
        return int(dec.get("seed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _fold_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None]:
    try:
        with _quiet_fit():
            ba = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        ba = None
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        f1 = None
    return ba, f1


# ---------------------------------------------------------------- cross-validation
def _is_partition(splits: Sequence[tuple[np.ndarray, np.ndarray]], n: int) -> bool:
    """True when every sample is the test sample of exactly one fold."""
    seen = np.zeros(n, dtype=int)
    for _, te in splits:
        seen[te] += 1
    return bool(np.all(seen == 1))


def _manual_oof(spec: TestSpec, X: Any, y: np.ndarray,
                splits: Sequence[tuple[np.ndarray, np.ndarray]], model: Any,
                seed: int | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    """Out-of-fold predictions for an *incomplete* split.

    ``cross_val_predict`` refuses to run unless the folds partition the data, and
    they legitimately do not when a fold has to be dropped because its training
    set contained a single class (a class that lives in only one block). Rather
    than silently relaxing the split, we predict what can honestly be predicted
    and report exactly which epochs had no out-of-fold prediction.

    Returns ``(pred, proba, predicted_mask, notes)``.
    """
    Xa = np.asarray(X)
    ya = np.asarray(y).reshape(-1)
    n = int(ya.size)
    pred = np.empty(n, dtype=object)
    pred[:] = None
    predicted = np.zeros(n, dtype=bool)
    global_classes = np.unique(ya)
    proba = np.full((n, global_classes.size), np.nan, dtype=float)
    proba_ok = True
    notes: list[str] = []
    for tr, te in splits:
        est = _estimator(spec, model, seed)
        try:
            with _quiet_fit():
                est.fit(Xa[tr], ya[tr])
                pred[te] = est.predict(Xa[te])
            predicted[te] = True
        except Exception as exc:                  # noqa: BLE001
            notes.append(f"fold fit failed ({type(exc).__name__}: {exc}); "
                         f"{int(te.size)} epoch(s) left unpredicted")
            continue
        if not proba_ok:
            continue
        if not hasattr(est, "predict_proba"):
            proba_ok = False
            continue
        try:
            pp = np.asarray(est.predict_proba(Xa[te]), dtype=float)
            col = np.asarray(getattr(est, "classes_", []))
            if pp.ndim != 2 or col.size != pp.shape[1]:
                proba_ok = False
                continue
            for j, c in enumerate(col):
                gi = np.flatnonzero(global_classes == c)
                if gi.size != 1:
                    proba_ok = False
                    break
                proba[te, int(gi[0])] = pp[:, j]
        except Exception:                         # noqa: BLE001
            proba_ok = False
    return pred, (proba if proba_ok else None), predicted, notes


def _abstention(spec: TestSpec, X: Any, y: np.ndarray,
                splits: Sequence[tuple[np.ndarray, np.ndarray]], model: Any,
                seed: int | None, target: float = 0.8,
                predicted: np.ndarray | None = None
                ) -> tuple[np.ndarray | None, list[float], str]:
    """Freeze an abstention threshold on training folds, apply to test folds.

    Rule (fixed, so it cannot be tuned on held-out data): inside each fold, take
    the ``1 - target`` quantile of the training-fold maximum class probability
    as that fold's threshold; a test epoch is *covered* when its own maximum
    probability reaches it. The threshold therefore depends only on development
    data, which is what the protocol requires.

    Epochs with no out-of-fold probability (a dropped fold) are simply not
    covered; they are never guessed at. Returns ``(covered_mask, thresholds,
    note)``; ``covered_mask`` is ``None`` when probabilities are unavailable.
    """
    ya = np.asarray(y).reshape(-1)
    n = int(ya.size)
    Xa = np.asarray(X)
    conf = np.full(n, np.nan, dtype=float)
    tau = np.full(n, np.nan, dtype=float)
    thresholds: list[float] = []
    for tr, te in splits:
        est = _estimator(spec, model, seed)
        try:
            with _quiet_fit():
                est.fit(Xa[tr], ya[tr])
            if not hasattr(est, "predict_proba"):
                return None, thresholds, "abstention not computed: estimator has no predict_proba"
            ptr = np.asarray(est.predict_proba(Xa[tr]), dtype=float)
            pte = np.asarray(est.predict_proba(Xa[te]), dtype=float)
        except Exception as exc:                  # noqa: BLE001
            return None, thresholds, f"abstention not computed: {type(exc).__name__}"
        if ptr.ndim != 2 or pte.ndim != 2 or ptr.shape[0] != tr.size or pte.shape[0] != te.size:
            return None, thresholds, "abstention not computed: probability shape mismatch"
        t_f = float(np.quantile(ptr.max(axis=1), max(0.0, min(1.0, 1.0 - target))))
        thresholds.append(t_f)
        conf[te] = pte.max(axis=1)
        tau[te] = t_f
    valid = np.isfinite(conf) & np.isfinite(tau)
    if predicted is not None:
        valid &= np.asarray(predicted, dtype=bool)
    if not np.any(valid):
        return None, thresholds, "abstention not computed: no out-of-fold probabilities"
    covered = np.zeros(n, dtype=bool)
    covered[valid] = conf[valid] >= tau[valid]
    note = ""
    n_missing = n - int(np.count_nonzero(valid))
    if n_missing:
        note = (f"{n_missing} epoch(s) had no out-of-fold probability (dropped fold) "
                "and are not covered")
    return covered, thresholds, note


def cross_validate(spec: TestSpec, X: Any, y: Any, groups: Any = None,
                   model: Any = None, *, groups_fine: Any = None,
                   session_ids: Any = None) -> dict:
    """Out-of-fold evaluation with grouped splits.

    Returns the dict shape frozen in ``docs/INTERFACES.md``::

        {'folds': [...], 'balanced_accuracy': float|None, 'auc': float|None,
         'f1': float|None, 'confusion': [[...]], 'chance': float, 'n': int,
         'notes': [...]}

    plus the additions this project needs to keep the number honest:

    * ``grouping`` and ``groups_fine``/``session_ids`` handling -- the grouping
      unit is chosen by ``spec.decoder['group_by']`` (``block`` | ``trial`` |
      ``session``) and is always named in ``notes``, because "accuracy" without
      its grouping is meaningless. Overlapping windows from one trial must not
      be split across folds.
    * ``balanced_accuracy_covered`` and ``abstention_rate`` -- the score when the
      model is allowed to abstain, with the threshold frozen on training folds
      only (see :func:`_abstention`).
    * ``majority_class_baseline`` next to ``chance``.
    * ``oof_predictions`` / ``oof_true`` (canonical class names) so callers can
      compute their own rates (false activations per minute, per-class recall).

    ``auc`` is reported only for a two-class contrast (binary AUC via
    ``predict_proba``); for more classes it is ``None`` with a note, because a
    one-vs-rest average would silently change what the gate means.

    ``model`` may be a decoder name (overrides ``spec.decoder['model']``) or a
    fitted :class:`~neurobridge.pipeline.decoders.TrainedModel` whose pipeline is
    cloned per fold. The extra keyword arguments are additive; the frozen
    positional signature is unchanged.
    """
    Xa = np.asarray(X)
    ya = np.asarray(y).reshape(-1)
    classes = _class_names(spec, ya)
    k = len(classes)
    n = int(ya.size)
    out: dict[str, Any] = {
        "folds": [], "balanced_accuracy": None, "auc": None, "f1": None,
        "balanced_accuracy_covered": None, "abstention_rate": None,
        "majority_class_baseline": None,
        "confusion": [], "chance": (1.0 / k if k else None), "n": n,
        "n_epochs": n, "classes": classes, "notes": [],
        "cv": str((getattr(spec, "decoder", {}) or {}).get("cv", "leave_one_block_out")).lower(),
    }
    notes: list[str] = out["notes"]
    if n == 0:
        notes.append("no epochs to evaluate")
        return out
    if k < 2:
        out["confusion"] = [[n]]
        out["majority_class_baseline"] = 1.0
        notes.append(f"single class {classes}: balanced accuracy and AUC are undefined; "
                     "this test's gates must use quality metrics")
        return out
    counts = {c: int(np.count_nonzero(ya == c)) for c in np.unique(ya)}
    out["majority_class_baseline"] = float(max(counts.values()) / n)
    if n < 4:
        notes.append(f"only {n} epochs: too few for cross-validation")
        return out

    seed = _seed_of(spec)
    group_arr, group_by, group_src = resolve_grouping(spec, groups, groups_fine, session_ids)
    n_groups = int(len(np.unique(group_arr))) if group_arr is not None and np.size(group_arr) else 0
    out["grouping"] = {"group_by": group_by, "source": group_src, "n_groups": n_groups}
    notes.append(f"grouping by {group_by} over {n_groups} group(s) [{group_src}]: "
                 "the same accuracy under a different grouping is a different claim")
    if group_by != "block" and group_src.startswith("groups (no"):
        notes.append(f"group_by={group_by} was requested but that unit was not recorded; "
                     "the block grouping was used instead")

    splitter, splits, plan_notes = _cv_plan(spec, Xa, ya, group_arr, classes, seed)
    notes.extend(plan_notes)
    if splitter is None or not splits:
        notes.append("balanced accuracy is not evaluable: the requested cross-validation "
                     "scheme could not be formed for this capture")
        return out

    if max(counts.values()) / max(1, n) > 0.75:
        notes.append("class imbalance: the majority-class baseline is "
                     f"{max(counts.values())/n:.2f}, so balanced accuracy is the honest metric")

    est_name = (str(model) if isinstance(model, str)
                else str((getattr(spec, "decoder", {}) or {}).get("model", "lda")).lower())
    est = _estimator(spec, model, seed)
    global_classes = np.unique(ya)

    # --- out-of-fold predictions -------------------------------------
    # cross_val_predict is used when the folds partition the data. When a fold
    # had to be dropped the partition is incomplete and sklearn refuses; we then
    # predict the rest by hand and say which epochs carry no out-of-fold
    # prediction, rather than relaxing the split to make the number look nicer.
    complete = _is_partition(splits, n)
    predicted = np.ones(n, dtype=bool)
    proba = None
    if complete:
        try:
            with _quiet_fit():
                y_pred = np.asarray(cross_val_predict(est, Xa, ya, cv=splitter,
                                                      method="predict"))
        except Exception as exc:                  # noqa: BLE001 - never raise
            notes.append(f"cross-validation failed: {type(exc).__name__}: {exc}")
            return out
        try:
            with _quiet_fit():
                p = np.asarray(cross_val_predict(est, Xa, ya, cv=splitter,
                                                 method="predict_proba"))
            if p.ndim == 2 and p.shape[0] == n and p.shape[1] == global_classes.size:
                proba = p
            else:
                notes.append("predict_proba columns did not align with the class order; "
                             "AUC not reported")
        except Exception as exc:                  # noqa: BLE001
            notes.append(f"predict_proba unavailable ({type(exc).__name__}); AUC not reported")
    else:
        y_pred, proba, predicted, oof_notes = _manual_oof(spec, Xa, ya, splits, model, seed)
        notes.extend(oof_notes)
        notes.append(
            f"folds do not partition the epochs ({int(n - np.count_nonzero(predicted))} "
            "epoch(s) had no out-of-fold prediction because their fold's training set "
            "contained a single class); metrics cover the predicted epochs only")
    out["n_predicted"] = int(np.count_nonzero(predicted))

    # --- metrics on the predicted subset only ------------------------
    ev = predicted
    ya_eval = ya[ev]
    yp_eval = np.asarray(y_pred)[ev]
    if int(ev.sum()) == 0:
        notes.append("balanced accuracy is not evaluable: no epoch had an out-of-fold prediction")
        return out
    eval_classes = np.unique(ya_eval)
    out["evaluated_classes"] = _class_names(spec, ya_eval)
    if eval_classes.size < 2:
        names = ", ".join(str(v) for v in _class_labels(spec, eval_classes))
        notes.append(
            f"balanced accuracy is not evaluable: the held-out predictions contain a single "
            f"class ({names}); on this capture that class occupies a single block, so "
            "leave-one-block-out cannot form a two-class training set for its fold")
        out["chance"] = None
        return out
    out["chance"] = float(1.0 / eval_classes.size)

    with _quiet_fit():
        out["balanced_accuracy"] = float(balanced_accuracy_score(ya_eval, yp_eval))
        out["f1"] = float(f1_score(ya_eval, yp_eval, average="macro", zero_division=0))
        out["confusion"] = confusion_matrix(
            ya_eval, yp_eval, labels=list(eval_classes)).astype(int).tolist()
    out["oof_predictions"] = [str(v) for v in _class_labels(spec, yp_eval)]
    out["oof_true"] = [str(v) for v in _class_labels(spec, ya_eval)]

    if proba is not None:
        if eval_classes.size == 2:
            col = [int(np.flatnonzero(global_classes == eval_classes[1])[0])]
            y_bin = (ya_eval == eval_classes[1]).astype(int)
            if 0 < int(y_bin.sum()) < int(ya_eval.size):
                out["auc"] = float(roc_auc_score(y_bin, proba[ev][:, col[0]]))
        elif k == 2:
            notes.append("AUC not reported: the evaluated subset contains a single class")
        else:
            notes.append(f"AUC is reported only for binary contrasts; this test has "
                         f"{eval_classes.size} evaluated classes")

    # --- abstention, threshold frozen on training folds --------------
    target = float((getattr(spec, "decoder", {}) or {}).get("target_coverage", 0.8) or 0.8)
    covered, thresholds, ab_note = _abstention(spec, Xa, ya, splits, model, seed, target,
                                               predicted=predicted)
    out["abstention"] = {
        "target_coverage": target,
        "thresholds": thresholds,
        "abstention_rate": None,
        "balanced_accuracy_covered": None,
        "coverage": None,
        "note": ab_note or ("threshold = (1-target) quantile of the training-fold maximum "
                            "class probability, frozen per fold"),
    }
    if covered is not None:
        cov_valid = covered & predicted
        denom = int(np.count_nonzero((np.ones(n, dtype=bool) if predicted is None else predicted)))
        rate = float(1.0 - (int(np.count_nonzero(cov_valid)) / max(1, denom)))
        out["abstention_rate"] = rate
        out["abstention"]["abstention_rate"] = rate
        out["abstention"]["coverage"] = float(np.count_nonzero(cov_valid) / max(1, denom))
        if int(np.count_nonzero(cov_valid)) >= 4 and len(np.unique(ya[cov_valid])) >= 2:
            ba_cov = float(balanced_accuracy_score(ya[cov_valid], np.asarray(y_pred)[cov_valid]))
            out["balanced_accuracy_covered"] = ba_cov
            out["abstention"]["balanced_accuracy_covered"] = ba_cov
            out["abstention"]["n_covered"] = int(np.count_nonzero(cov_valid))
            notes.append(
                f"abstention: {100*rate:.0f}% of epochs abstained; covered balanced accuracy "
                f"{ba_cov:.3f} (all-epoch accuracy {out['balanced_accuracy']:.3f})")
        else:
            out["abstention"]["note"] += " (too few covered epochs per class to score)"
    else:
        notes.append(ab_note or "abstention not computed")

    # --- per-fold detail --------------------------------------------
    y_pred_arr = np.asarray(y_pred)
    for i, (tr, te) in enumerate(splits):
        te_pred = te[predicted[te]]
        if te_pred.size == 0:
            ba_f = f1_f = None
        else:
            ba_f, f1_f = _fold_metric(ya[te_pred], y_pred_arr[te_pred])
        auc_f = None
        if proba is not None and eval_classes.size == 2:
            col = int(np.flatnonzero(global_classes == eval_classes[1])[0])
            yb = (ya[te_pred] == eval_classes[1]).astype(int)
            if te_pred.size and 0 < int(yb.sum()) < int(te_pred.size):
                try:
                    auc_f = float(roc_auc_score(yb, proba[te_pred, col]))
                except Exception:
                    auc_f = None
        cov_f = None
        if covered is not None and te_pred.size:
            cov_f = float(np.count_nonzero(covered[te_pred]) / te_pred.size)
        gval = None
        if group_arr is not None and np.size(group_arr):
            gv = np.unique(np.asarray(group_arr).reshape(-1)[te])
            gval = int(gv[0]) if gv.size == 1 else [int(x) for x in gv]
        out["folds"].append({
            "fold": i,
            "group": gval,
            "n_train": int(tr.size),
            "n_test": int(te.size),
            "n_predicted": int(te_pred.size),
            "balanced_accuracy": ba_f,
            "f1": f1_f,
            "auc": auc_f,
            "coverage": cov_f,
        })
    folds_ba = [f["balanced_accuracy"] for f in out["folds"] if f["balanced_accuracy"] is not None]
    if folds_ba:
        notes.append(f"per-fold balanced accuracy {min(folds_ba):.2f}-{max(folds_ba):.2f} "
                     f"over {len(folds_ba)} folds")
    return out


def _cv_point_score(spec: TestSpec, X: Any, y: np.ndarray, groups: Any,
                    cv_name: str | None = None) -> float | None:
    """Cheap out-of-fold balanced accuracy, used by the permutation test.

    Only hard predictions are needed and the split plan is recomputed, so this is
    roughly half the cost of :func:`cross_validate`. Returns ``None`` when the
    score is not defined (no formable split, an unpredicted fold, or a
    single-class held-out subset) -- never a made-up number.
    """
    ya = np.asarray(y).reshape(-1)
    if np.unique(ya).size < 2:
        return None
    seed = _seed_of(spec)
    splitter, splits, _ = _cv_plan(spec, X, ya, groups, list(np.unique(ya)), seed)
    if splitter is None or not splits:
        return None
    Xa = np.asarray(X)
    pred = np.empty(ya.size, dtype=object)
    pred[:] = None
    for tr, te in splits:
        est = build_pipeline(spec, cv_name, seed)
        try:
            with _quiet_fit():
                est.fit(Xa[tr], ya[tr])
                pred[te] = est.predict(Xa[te])
        except Exception:
            return None
    mask = np.array([p is not None for p in pred])
    if int(mask.sum()) < 4 or len(np.unique(ya[mask])) < 2:
        return None
    try:
        with _quiet_fit():
            return float(balanced_accuracy_score(ya[mask], pred[mask].astype(ya.dtype)))
    except Exception:
        return None


def _permute_within_groups(y: np.ndarray, groups: Any, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels inside each group.

    Labels are permuted *within* a block so the null keeps each block's class
    proportions. That is the correct null when blocks differ in class balance
    (rare-class paradigms), and it is what makes the p-value a statement about
    the contrast rather than about block identity.
    """
    ya = np.asarray(y).reshape(-1)
    if groups is None or np.size(groups) == 0:
        return rng.permutation(ya)
    g = np.asarray(groups).reshape(-1)
    out = ya.copy()
    for gv in np.unique(g):
        idx = np.flatnonzero(g == gv)
        if idx.size > 1:
            out[idx] = rng.permutation(ya[idx])
    return out


def permutation_test(spec: TestSpec, X: Any, y: Any, groups: Any = None,
                     n: int = 200, seed: int = 0, *, groups_fine: Any = None,
                     session_ids: Any = None) -> dict:
    """Label-permutation test on the cross-validated balanced accuracy.

    ``p = (1 + #{null >= observed}) / (1 + n)``: the +1 keeps the p-value
    strictly positive, which matters because a p-value of exactly 0 is never
    justified with a finite number of permutations.

    Permutations happen **inside the chosen grouping unit** (``decoder.group_by``,
    resolved exactly as in :func:`cross_validate`): labels are shuffled within
    each block/trial/session, which keeps both the grouping structure and the
    per-group class *counts* fixed -- the assignment moves, the multiset does
    not. That is the correct null for a rare-class design, where a global
    shuffle would change each block's class balance.

    Cost is one CV per permutation; for a linear decoder on a few hundred epochs
    that is seconds, which is what makes 200 permutations the default. EEGNet is
    orders of magnitude more expensive, so its permutation count is capped
    (recorded in the result). The result always contains ``p_value``,
    ``observed`` and ``n_permutations``.
    """
    ya = np.asarray(y).reshape(-1)
    out = {"p_value": None, "observed": None, "null_mean": None, "null_std": None,
           "n": 0, "n_permutations": 0, "notes": [], "grouping": None}
    classes = _class_names(spec, ya)
    if ya.size == 0 or len(classes) < 2:
        out["notes"].append("permutation test needs at least two classes")
        return out

    group_arr, group_by, group_src = resolve_grouping(spec, groups, groups_fine, session_ids)
    out["grouping"] = {"group_by": group_by, "source": group_src}
    est_name = str((getattr(spec, "decoder", {}) or {}).get("model", "lda")).lower()
    n_req = int(n)
    if est_name == "eegnet" and n_req > 20:
        out["notes"].append(f"eegnet is expensive: permutation count capped at 20 (asked {n_req})")
        n_req = 20

    observed = _cv_point_score(spec, X, ya, group_arr)
    if observed is None:
        out["notes"].append("observed score could not be computed")
        return out
    rng = np.random.default_rng(int(seed))
    null: list[float] = []
    for _ in range(max(1, n_req)):
        yp = _permute_within_groups(ya, group_arr, rng)
        s = _cv_point_score(spec, X, yp, group_arr)
        if s is not None:
            null.append(float(s))
    if not null:
        out["notes"].append("every permuted fit failed")
        return out
    arr = np.asarray(null, dtype=float)
    n_ge = int(np.count_nonzero(arr >= observed - 1e-12))
    out.update({
        "p_value": float((1 + n_ge) / (1 + arr.size)),
        "observed": float(observed),
        "null_mean": float(arr.mean()),
        "null_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "n_permutations": int(arr.size),
    })
    out["notes"].append(
        f"{arr.size} permutations within {group_by} units"
        + ("" if group_src == "groups" else f" [{group_src}]")
        + f"; {n_ge} were at least as extreme as the observed score")
    return out


def information_transfer_rate(accuracy: float | None, n_classes: int, trial_s: float) -> float:
    """Wolpaw bits-per-minute for a discrete selection task.

    ``B = log2(N) + P log2(P) + (1-P) log2((1-P)/(N-1))`` bits per trial, times
    ``60 / trial_s``. Accuracy is clipped to ``[1/N, 1]`` because a below-chance
    decoder carries no information (and the formula's logarithm is undefined
    there). The honest use of this number on this hardware is as a *ceiling*
    estimate: tens of seconds per trial makes even a good two-class switch
    single-digit bits per minute.
    """
    k = int(n_classes)
    t = float(trial_s)
    if accuracy is None or k < 2 or t <= 0 or not np.isfinite(accuracy):
        return 0.0
    p = float(min(max(float(accuracy), 1.0 / k), 1.0))
    if p >= 1.0 - 1e-12:
        bits = float(np.log2(k))
    elif p <= 1.0 / k:
        bits = 0.0
    else:
        bits = (float(np.log2(k)) + p * float(np.log2(p))
                + (1.0 - p) * float(np.log2((1.0 - p) / (k - 1))))
    return float(max(0.0, bits) * 60.0 / t)


# ---------------------------------------------------------------- gates
def _lookup_metric(results: dict, metric: str) -> tuple[Any, str | None, bool]:
    """Resolve an acceptance metric to ``(value, source, found)``.

    ``results`` may be the full nested result dict or a flat one; both are
    accepted so callers can apply gates to a hand-built dict in a test.
    """
    key = metric.strip().lower()
    src = _METRIC_SOURCES.get(key)
    candidates = [src] if src else []
    candidates.append(key)
    for path in candidates:
        if not path:
            continue
        node: Any = results
        ok = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and node is not None:
            try:
                val = float(node)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val) or key in ("n", "n_epochs"):
                return val, path, True
    return None, src, False


def apply_gates(spec: TestSpec, results: dict) -> dict:
    """Compare every ``spec.acceptance`` entry against a result dict.

    Contract::

        {'passed': bool | None, 'evaluable': bool, 'checks': [
            {'metric', 'value', 'gate', 'ok', 'evaluated', 'source',
             'direction', 'applicable', 'reason'}]}

    Three states per check, and they are not the same thing:

    * **evaluated** -- a finite value was obtained; ``ok`` says whether it met the
      gate. This is the only thing that drives the verdict.
    * **applicable but not evaluated** -- the metric belongs to this spec but could
      not be computed now (e.g. no two-class training set exists under
      leave-one-block-out because a class occupies a single block). ``value`` is
      ``None``, ``evaluated`` is false, and the reason quotes the CV note.
      Never presented as a failed measurement.
    * **not applicable** -- the metric cannot apply to this spec at all (accuracy
      on a single-class test, binary AUC on a 3-class test). Excluded entirely.

    The overall verdict is three-valued:

    * ``passed: True`` -- every *evaluated* check passed;
    * ``passed: False`` -- at least one evaluated check failed;
    * ``passed: None``, ``evaluable: False`` -- nothing could be evaluated, so the
      honest answer is "insufficient data", not "failed".

    ``complete`` records whether the verdict rests on all of the spec's gates or
    only on the ones that could be evaluated here.

    For ``feasibility: control`` the accuracy gates stay minimum thresholds: a
    control test is *expected to fail them*, and that failure is the evidence
    that the pipeline is not hallucinating a difference (docs/METHOD.md §7).
    """
    acc = dict(getattr(spec, "acceptance", {}) or {})
    classes = list(getattr(spec, "classes", []) or [])
    cv = results.get("cv") or {}
    # Two different counts matter: what the *test* is (does an accuracy gate even
    # apply?) and what the *evaluation* actually covered (could it be computed?).
    k_spec = len(cv.get("classes") or classes) if (cv.get("classes") or classes) else 0
    k_eval = len(cv.get("evaluated_classes") or cv.get("classes") or classes) \
        if (cv.get("evaluated_classes") or cv.get("classes") or classes) else 0
    # The CV notes carry the *reason* a metric is unevaluable; quote them.
    cv_notes = [str(n) for n in (cv.get("notes") or [])]
    # Prefer the most specific explanation available.
    cv_reason = ""
    for needles in (("not evaluable",),
                    ("cannot be formed",),
                    ("cross-validation failed",),
                    ("dropped", "unpredicted"),
                    ("single class",),
                    ("too few", "no epochs to evaluate"),
                    ("only ",)):
        for n in cv_notes:
            if any(tok in n for tok in needles):
                cv_reason = n
                break
        if cv_reason:
            break
    checks: list[dict[str, Any]] = []

    for metric, gate in acc.items():
        key = str(metric).strip().lower()
        value, source, found = _lookup_metric(results, key)
        direction = "max" if key in _LOWER_IS_BETTER else "min"
        check: dict[str, Any] = {
            "metric": str(metric),
            "value": float(value) if (found and value is not None
                                      and np.isfinite(float(value))) else None,
            "gate": float(gate),
            "ok": False,
            "evaluated": False,
            "source": source,
            "direction": direction,
            "applicable": True,
        }
        # -- inapplicable metric, excluded from the verdict -----------
        if key in _ACCURACY_METRICS and k_spec < 2:
            check["applicable"] = False
            check["reason"] = (f"not applicable: this is a single-class test "
                               f"({classes or 'no classes'}); its quality gates decide")
        elif key in ("auc", "roc_auc") and k_spec > 2:
            check["applicable"] = False
            check["reason"] = (f"not applicable: AUC is reported for binary contrasts "
                               f"only and this test has {k_spec} classes")
        elif not found or value is None or not np.isfinite(float(value)):
            check["evaluated"] = False
            reason_text = (cv_reason
                           or ("the result contains no cross-validation block"
                               if not cv_notes else cv_notes[-1])
                           or "the metric is absent from the result dict")
            # drop a redundant prefix: the table already labels the check
            for prefix in ("balanced accuracy is not evaluable: ",
                           "not computed: ", "not evaluable: "):
                if reason_text.startswith(prefix):
                    reason_text = reason_text[len(prefix):]
            check["reason"] = reason_text
        else:
            check["evaluated"] = True
            if direction == "min":
                check["ok"] = bool(value >= check["gate"] - 1e-12)
            else:
                check["ok"] = bool(value <= check["gate"] + 1e-12)
            if key in _ACCURACY_METRICS and str(getattr(spec, "feasibility", "")) == "control":
                check["note"] = ("control test: passing this gate would mean the pipeline "
                                 "found a difference where the protocol contains none")
        checks.append(check)

    evaluated = [c for c in checks if c["evaluated"]]
    not_evaluated = [c for c in checks if c["applicable"] and not c["evaluated"]]
    inapplicable = [c for c in checks if not c["applicable"]]
    if evaluated:
        passed: bool | None = all(c["ok"] for c in evaluated)
    else:
        passed = None
    out: dict[str, Any] = {
        "passed": passed,
        "evaluable": bool(evaluated),
        "complete": bool(evaluated) and not not_evaluated,
        "checks": checks,
        "n_evaluated": len(evaluated),
        "n_evaluable": len(evaluated),
        "n_not_evaluated": len(not_evaluated),
        "n_inapplicable": len(inapplicable),
        "n_missing": len(not_evaluated),
    }
    out["verdict"] = ("PASS" if passed is True else
                      "FAIL" if passed is False else "INSUFFICIENT DATA")
    if not evaluated:
        out["reason"] = ("no gate could be evaluated for this capture"
                         + (f": {cv_reason}" if cv_reason else ""))
        out["note"] = "this is a design limitation, not a negative result"
        out["verdict_note"] = out["reason"] + "; " + out["note"]
    elif not_evaluated:
        out["reason"] = ("verdict rests on the " + str(len(evaluated))
                         + " evaluable gate(s); not evaluated: "
                         + ", ".join(c["metric"] for c in not_evaluated))
        out["verdict_note"] = out["reason"]
    return out


# ---------------------------------------------------------------- epoch building
def _extract(spec: TestSpec, epochs: Epochs) -> tuple[np.ndarray, list[str]]:
    if len(epochs) == 0:
        names = feature_names(getattr(spec, "features", None) or DEFAULT_FEATURES)
        return np.zeros((0, len(names)), dtype=float), names
    return extract_epochs(epochs, spec)


def build_epochs(session: Any, spec: TestSpec, signal: np.ndarray | None = None
                 ) -> tuple[Epochs, list[str]]:
    """Epoch a session according to the spec, with an honest fallback.

    Declared mode is honoured (``window_s`` for sustained-state tests,
    ``onset_markers`` for stimulus-locked ones). If a windowed spec yields fewer
    than 8 usable windows or only one class *and* stimulus-locked epoching yields
    more usable data, we fall back and say so in the notes: that happens when a
    recorded timeline turns out to be trial-based (short SOA) even though the
    spec expected long conditions, and refusing to analyse it would be worse than
    analysing it the way the data was actually collected.
    """
    notes: list[str] = []
    proc = dict(getattr(spec, "processing", {}) or {})
    policy = str(proc.get("reject_policy", "reject")).lower()
    hard_reject = policy == "reject"
    mode = epoching_mode(spec)

    if mode == "windowed":
        w = float(proc.get("window_s", 2.0) or 2.0)
        step = float(proc.get("step_s", 0.5) or 0.5)
        lf = str(proc.get("label_from", "state") or "state")
        ep = windowed_epochs(session, spec, window_s=w, step_s=step,
                             label_from=lf, signal=signal)
        ep.meta["epoching"] = "windowed"
        enough = len(ep) >= 8 and len(np.unique(ep.y)) >= 2 if len(ep) else False
        if not enough:
            stim = epochs_from_session(session, spec, signal=signal, reject=hard_reject)
            n_win, n_stim = len(ep), len(stim)
            better = n_stim > n_win or (
                n_stim and len(np.unique(stim.y)) > len(np.unique(ep.y)))
            if better:
                notes.append(
                    f"declared windowed epoching produced {n_win} usable window(s) "
                    f"(straddling class changes or too few conditions); fell back to "
                    f"stimulus-locked epochs ({n_stim})")
                stim.meta["epoching"] = "stimulus_fallback"
                stim.meta["windowed_attempt"] = {
                    "n": n_win, "skipped_straddle": ep.meta.get("n_skipped_straddle"),
                    "skipped_unlabelled": ep.meta.get("n_skipped_unlabelled"),
                }
                return stim, notes
        return ep, notes

    ep = epochs_from_session(session, spec, signal=signal, reject=hard_reject)
    ep.meta["epoching"] = "stimulus"
    return ep, notes


# ---------------------------------------------------------------- derived metrics
def _response_snr(epochs: Epochs, spec: TestSpec) -> float | None:
    """Grand-average ERP amplitude divided by the pre-stimulus noise.

    The amplitude is the largest absolute excursion of the *average* waveform in
    the post-stimulus window; the noise is the standard deviation of the
    pre-stimulus samples across trials. Averaging suppresses noise by
    ``sqrt(N)``, so this number grows with trial count -- which is exactly why a
    single-trial SNR would be the wrong thing to gate on.
    """
    if len(epochs) == 0:
        return None
    X = np.asarray(epochs.X, dtype=float)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    times = np.asarray(epochs.times, dtype=float)
    pre = times < 0
    post = times >= 0
    if not np.any(post):
        return None
    evoked = np.nanmean(X, axis=0)
    amp = float(np.nanmax(np.abs(evoked[post])))
    if np.any(pre):
        sigma = float(np.nanstd(X[:, pre]))
    else:
        # no baseline: use the trial-to-trial spread of the post-stimulus mean
        sigma = float(np.nanstd(np.nanmean(X[:, post], axis=1)))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return float(amp / sigma)


def _latency_jitter_ms(epochs: Epochs, spec: TestSpec) -> float | None:
    """Spread (SD) of single-trial ERP peak latencies, in milliseconds.

    High jitter means the response is present but its timing wanders: it lowers
    the amplitude of the average without meaning the effect is absent, so a
    report must show it next to any amplitude-based claim.
    """
    if len(epochs) < 2:
        return None
    X = np.asarray(epochs.X, dtype=float)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    times = np.asarray(epochs.times, dtype=float)
    post = times >= 0.0
    if not np.any(post) or X.shape[-1] != times.size:
        return None
    idx = np.flatnonzero(post)
    peak = np.nanargmax(np.abs(X[:, post]), axis=1)
    lat_ms = times[idx[peak]] * 1000.0
    if lat_ms.size < 2:
        return None
    return float(np.nanstd(lat_ms, ddof=1))


def _alpha_ratio(spec: TestSpec, epochs: Epochs) -> float | None:
    """Closed-eyes/eyes-open alpha ratio (the T003 validation).

    Uses the same ``device.quality.alpha_reactivity`` as the live quality check
    so the recorded session and the report cannot disagree. Requires both an
    eyes-open and an eyes-closed class; anything else is inapplicable.
    """
    names = [str(c) for c in (getattr(spec, "classes", []) or [])]
    open_i = closed_i = None
    for i, c in enumerate(names):
        low = c.lower()
        if "closed" in low or "shut" in low:
            closed_i = i
        elif "open" in low:
            open_i = i
    if open_i is None or closed_i is None or len(epochs) == 0:
        return None
    X = np.asarray(epochs.X, dtype=float)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    yo = X[np.asarray(epochs.y) == open_i]
    yc = X[np.asarray(epochs.y) == closed_i]
    if yo.size == 0 or yc.size == 0:
        return None
    try:
        res = alpha_reactivity(yo.reshape(-1), yc.reshape(-1), float(epochs.fs))
    except Exception:
        return None
    ratio = res.get("alpha_ratio")
    if ratio is None or not np.isfinite(ratio):
        return None
    return float(ratio)


def _false_positive_rate(spec: TestSpec, X: Any, y: np.ndarray, groups: Any,
                         n: int = 20, margin: float = 0.10,
                         seed: int = 0) -> tuple[float | None, dict[str, Any]]:
    """Empirical false-positive rate of this pipeline on a null contrast.

    Two flavours, both honest about what they measure:

    * **multi-class**: labels are permuted within blocks ``n`` times and the
      grouped CV is re-run each time; the rate is the fraction of null runs whose
      balanced accuracy exceeded ``chance + margin``. That is a direct estimate
      of the type-I error this pipeline would show at that threshold.
    * **single-class**: there is no label to permute, so each replicate splits
      the epochs into two arbitrary pseudo-classes (balanced within each block)
      and asks whether the pipeline can tell them apart. Any "success" is by
      construction a false positive.

    This is deliberately a *rate over replicates*, not a p-value: a control test
    needs to know how often the whole procedure fires when nothing is there.
    """
    ya = np.asarray(y).reshape(-1)
    classes = _class_names(spec, ya)
    k = len(classes)
    if ya.size < 8:
        return None, {"note": "not enough epochs for a false-positive estimate"}
    rng = np.random.default_rng(int(seed))
    hits = 0
    scores: list[float] = []
    single = k < 2
    for _ in range(max(1, int(n))):
        if single:
            yp = rng.permutation(np.arange(ya.size)) % 2   # pseudo-classes, balanced overall
            yp = np.asarray(yp)
        else:
            yp = _permute_within_groups(ya, groups, rng)
        score = _cv_point_score(spec, X, yp, groups)
        if score is None:
            continue
        scores.append(float(score))
        if score > (1.0 / max(2, k)) + margin:
            hits += 1
    if not scores:
        return None, {"note": "every replicate failed"}
    frac = hits / len(scores)
    return float(frac), {
        "n_replicates": len(scores),
        "margin": float(margin),
        "mean_null_accuracy": float(np.mean(scores)),
        "method": ("pseudo-class split (single-class spec)" if single
                   else "label permutation within blocks"),
    }


def _class_recall(cv: dict, name: str) -> float | None:
    """Recall of a named class from the confusion matrix, when present."""
    classes = [str(c) for c in (cv.get("classes") or [])]
    conf = cv.get("confusion") or []
    if name not in classes or not conf:
        return None
    i = classes.index(name)
    row = np.asarray(conf, dtype=float)
    if i >= row.shape[0] or row[i].sum() <= 0:
        return None
    return float(row[i, i] / row[i].sum())


#: Class names that mean "no command": false activations are counted here.
_IDLE_CLASS_TOKENS = ("no_command", "nocmd", "no_cmd", "idle", "rest", "cancel",
                      "timeout", "relax", "none", "quiet", "sham", "baseline")


def _is_idle_class(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in _IDLE_CLASS_TOKENS)


def coverage_of(epochs: Epochs) -> dict[str, Any]:
    """Retained epochs / candidate epochs, per class, with exclusion reasons.

    "Report performance with coverage so that rejecting most difficult trials
    cannot masquerade as a reliable interface" (protocol §8.4). A test that
    keeps 12 of 240 trials is not the same experiment as one that keeps 230, even
    if the surviving trials decode beautifully -- so the retention ratio and the
    reasons are part of the result, not a footnote.
    """
    counts = epochs.class_counts()
    meta = dict(epochs.meta or {})
    cand = dict(meta.get("candidate_counts") or {})
    rejected = dict(meta.get("rejected_by_label") or {})
    per_class: dict[str, Any] = {}
    for lab in epochs.labels:
        retained = int(counts.get(lab, 0))
        candidates = int(cand.get(lab, retained + int(rejected.get(lab, 0))))
        if candidates <= 0:
            candidates = retained
        per_class[lab] = {
            "retained": retained,
            "candidates": candidates,
            "retained_fraction": (float(retained) / candidates) if candidates else None,
        }
    total_cand = int(meta.get("n_candidates") or sum(v["candidates"] for v in per_class.values()))
    retained_n = int(len(epochs))
    return {
        "retained": retained_n,
        "candidates": total_cand,
        "retained_fraction": (float(retained_n) / total_cand) if total_cand else None,
        "per_class": per_class,
        "exclusions": dict(meta.get("exclusion_counts") or {}),
        "reject_policy": str(meta.get("reject_policy", "reject")),
        "hard_reject": bool(meta.get("hard_reject", False)),
        "flagged_kept": int(meta.get("n_flagged_kept", 0) or 0),
        "note": ("retained/candidate epochs per class; a low retained_fraction means the "
                 "headline accuracy describes an unrepresentative subset"),
    }


def false_activations_per_min(spec: TestSpec, cv: dict, epochs: Epochs) -> tuple[float | None, dict]:
    """Command-class detections per minute while the true class is idle.

    Frozen decision rule (there is no deployment smoother in the analysis path):
    ``argmax`` of the out-of-fold class probability, no smoothing and no
    refractory period. Idle duration is approximated as
    ``n_idle * unit_s``, where ``unit_s`` is the window step for windowed
    epoching and the epoch length otherwise -- for overlapping windows that is
    an *upper* bound on the covered time, so the reported rate is conservative
    (it can only understate the false-activation rate).

    Returns ``(rate, info)``; ``rate`` is ``None`` with an explanatory note when
    the design has no idle class or no distinct command class.
    """
    classes = [str(c) for c in (cv.get("classes") or getattr(spec, "classes", []) or [])]
    true = [str(v) for v in (cv.get("oof_true") or [])]
    pred = [str(v) for v in (cv.get("oof_predictions") or [])]
    info: dict[str, Any] = {"decision_rule": "argmax, no smoothing, no refractory"}
    if not classes:
        return None, {**info, "note": "no classes in the result"}
    idle = [c for c in classes if _is_idle_class(c)]
    commands = [c for c in classes if not _is_idle_class(c)]
    if not idle:
        return None, {**info, "note": "no idle/no_command/rest class in this design"}
    if not commands:
        return None, {**info, "note": "all classes are idle-like; there is no command to fire"}
    if not true or len(true) != len(pred):
        return None, {**info, "note": "out-of-fold predictions unavailable"}
    idle_mask = np.array([t in idle for t in true])
    n_idle = int(idle_mask.sum())
    if n_idle == 0:
        return None, {**info, "note": "no idle epochs were retained"}
    fa = int(sum(1 for t, p in zip(true, pred) if t in idle and p in commands))
    n_win = int(np.asarray(epochs.X).shape[-1]) if np.asarray(epochs.X).ndim >= 2 else 0
    unit_s = float(n_win / epochs.fs) if (n_win and epochs.fs) else 0.0
    if str(epochs.meta.get("epoching", "")) == "windowed":
        unit_s = float(epochs.meta.get("step_s", unit_s) or unit_s)
    minutes = n_idle * unit_s / 60.0 if unit_s > 0 else 0.0
    info.update({"idle_classes": idle, "command_classes": commands, "n_idle_epochs": n_idle,
                 "false_activations": fa, "idle_minutes": minutes,
                 "unit_s": unit_s})
    if minutes <= 0:
        return None, {**info, "note": "idle duration could not be estimated"}
    return float(fa / minutes), info


def _quality_metrics(pr: Any) -> dict[str, Any]:
    """Numbers that describe the *capture*, read from the shared quality code."""
    q = {}
    try:
        q.update({k: float(v) for k, v in (pr.quality.metrics or {}).items()})
    except Exception:
        pass
    art = dict(getattr(pr, "artifacts", {}) or {})
    for k in ("saturation_pct", "motion_pct", "flatline_pct", "bad_frac"):
        if k in art:
            q[k] = float(art[k])
    q["artifact_fraction"] = float(art.get("bad_frac", q.get("bad_frac", 0.0)))
    if "motion_jump_pct" not in q and "motion_pct" in q:
        q["motion_jump_pct"] = q["motion_pct"]
    q["n_samples"] = int(art.get("n_samples", getattr(pr, "n_samples", 0)) or 0)
    return q


def _derived_metrics(spec: TestSpec, session: Any, pr: Any, epochs: Epochs,
                     cv: dict, seed: int = 0, groups: Any = None) -> dict[str, Any]:
    """Metrics computed from the epochs, not from the raw capture."""
    d: dict[str, Any] = {}
    gp = groups if groups is not None and np.size(groups) else (
        epochs.groups if epochs.groups.size else None)
    d["latency_offset_ms"] = float(pr.meta.get("latency_offset_s", 0.0)) * 1000.0
    snr = _response_snr(epochs, spec)
    if snr is not None:
        d["response_snr"] = snr
    jit = _latency_jitter_ms(epochs, spec)
    if jit is not None:
        d["latency_jitter_ms"] = jit
    ratio = _alpha_ratio(spec, epochs)
    if ratio is not None:
        d["alpha_ratio_closed_open"] = ratio
    if len(epochs) >= 8:
        fpr, info = _false_positive_rate(spec, epochs.X, epochs.y, gp, seed=seed)
        if fpr is not None:
            d["false_positive_rate"] = fpr
            d["false_positive_rate_info"] = info
    fa_rate, fa_info = false_activations_per_min(spec, cv, epochs)
    d["false_activations_per_min"] = fa_rate
    d["false_activations_info"] = fa_info
    tr = _class_recall(cv, "target")
    if tr is not None:
        d["target_recall"] = tr
    miss = _class_recall(cv, "missed")
    if miss is not None:
        d["miss_rate"] = float(1.0 - miss)
    if epochs.onsets.size >= 2:
        d["median_ioi_s"] = float(np.median(np.diff(epochs.onsets)))
    # marker-derived event rates (informational; no gate uses them)
    try:
        events = list(session.events_for(test_id=spec.id))
        dur = float(getattr(session, "duration_s", 0.0)) or 0.0
        if dur > 0:
            nb = sum(1 for e in events if e.name in ("blink", "artifact_motion"))
            d["blink_rate_per_min"] = float(60.0 * nb / dur)
            nm = sum(1 for e in events if e.name in ("clench", "artifact_motion", "motion"))
            d["motion_events_per_min"] = float(60.0 * nm / dur)
    except Exception:
        pass
    return d


# ---------------------------------------------------------------- orchestration
def _result_summary(spec: TestSpec, session: Any, pr: Any, epochs: Epochs,
                    cv: dict, gates: dict, notes: list[str]) -> dict:
    return {
        "test_id": str(getattr(spec, "id", "")),
        "title": str(getattr(spec, "title", "")),
        "feasibility": str(getattr(spec, "feasibility", "")),
        "tier": int(getattr(spec, "tier", 0) or 0),
        "classes": list(getattr(spec, "classes", []) or []),
        "session": str(getattr(getattr(session, "path", None), "name", "")),
        "session_path": str(getattr(session, "path", "")),
        "synthetic": bool((getattr(session, "meta", {}) or {}).get("synthetic", False)),
        "fs": float(pr.fs),
        "n_samples": int(pr.n_samples),
        "duration_s": float(pr.meta.get("duration_s", 0.0)),
        "quality": pr.quality.as_dict(),
        "artifacts": dict(pr.artifacts),
        "preprocess": dict(pr.meta),
        "epochs": epochs.as_dict(),
        "cv": cv,
        "gates": gates,
        "notes": list(notes),
    }


def evaluate_session(session: Any, spec: TestSpec, model: Any = None,
                     train_if_needed: bool = True) -> dict:
    """Load -> preprocess -> epoch -> features -> CV/train, as one dict.

    The returned dict is JSON-serialisable and is the single object the CLI
    (``nb eval``), the reports and the agent bridge consume. It always contains
    ``cv`` (with ``balanced_accuracy: None`` and a note when no contrast exists),
    ``coverage``, ``quality_metrics``, ``derived_metrics``, ``timing_quality``,
    ``context_features`` and ``gates``.

    Never raises for data reasons: preprocessing, epoching, feature extraction
    and fitting each degrade to a documented note. The one deliberate exception
    is a spec that asks for context features -- :func:`assert_signal_only`
    raises, because a context-assisted model must be built and reported
    separately.
    """
    notes: list[str] = []
    pr = preprocess_session(session, spec)
    epochs, ep_notes = build_epochs(session, spec, signal=pr.signal)
    notes.extend(ep_notes)
    X, names = _extract(spec, epochs)

    # -- the signal-only claim is checked, not assumed ----------------
    assert_signal_only(names)
    context_features: list[str] = []

    group_arr, group_by, group_src = resolve_grouping(
        spec, epochs.groups if len(epochs) else None,
        epochs.groups_fine if len(epochs) else None, None)

    cv = cross_validate(spec, X, epochs.y, epochs.groups, groups_fine=epochs.groups_fine) \
        if len(epochs) else cross_validate(
            spec, np.zeros((0, max(1, len(names)))), np.zeros(0, dtype=int), None)

    # -- model --------------------------------------------------------
    trained: TrainedModel | None = None
    try:
        if isinstance(model, (str,)):
            model = load_model(model)
        if isinstance(model, TrainedModel):
            trained = model
        elif train_if_needed:
            trained = train(spec, X, epochs.y, epochs.groups, names, _seed_of(spec))
    except Exception as exc:                      # noqa: BLE001
        notes.append(f"model training skipped: {type(exc).__name__}: {exc}")

    # -- permutation test (only when the spec asks for it) -----------
    perm = None
    want_perm = "permutation_p" in [str(m) for m in (getattr(spec, "metrics", []) or [])]
    if want_perm and len(epochs) >= 8 and len(np.unique(epochs.y)) >= 2:
        try:
            perm = permutation_test(spec, X, epochs.y, group_arr, n=200,
                                    seed=_seed_of(spec))
        except Exception as exc:                  # noqa: BLE001
            notes.append(f"permutation test failed: {type(exc).__name__}: {exc}")

    quality_metrics = _quality_metrics(pr)
    derived = _derived_metrics(spec, session, pr, epochs, cv, seed=_seed_of(spec),
                               groups=group_arr)
    timing_quality, timing_notes = timing_quality_of(session, spec)
    notes.extend(timing_notes)
    coverage = coverage_of(epochs)

    k = len(cv.get("classes") or []) or len([c for c in (getattr(spec, "classes", []) or [])])
    trial_s = float(derived.get("median_ioi_s") or 0.0) or float(
        (getattr(spec, "design", {}) or {}).get("soa_s") or 0.0)
    itr = information_transfer_rate(cv.get("balanced_accuracy"), k, trial_s) if trial_s > 0 else 0.0

    results: dict[str, Any] = {
        **_result_summary(spec, session, pr, epochs, cv, {"passed": False, "checks": []}, notes),
        "features": list(names),
        "context_features": context_features,
        "signal_only_checked": True,
        "coverage": coverage,
        "quality_metrics": quality_metrics,
        "derived_metrics": derived,
        "model": trained.as_dict() if isinstance(trained, TrainedModel) else None,
        "permutation": perm,
        "itr_bits_per_min": itr,
        "timing_quality": timing_quality,
        "per_session": [{
            "session": str(getattr(getattr(session, "path", None), "name", "")),
            "n_epochs": int(len(epochs)),
            "balanced_accuracy": cv.get("balanced_accuracy"),
            "auc": cv.get("auc"),
        }],
        "timing": {
            "latency_offset_ms": float(pr.meta.get("latency_offset_s", 0.0)) * 1000.0,
            "n_onsets": int(epochs.onsets.size),
            "median_ioi_s": derived.get("median_ioi_s"),
            "marker_jitter_ms": (float(np.std(np.diff(epochs.onsets), ddof=1) * 1000.0)
                                 if epochs.onsets.size >= 3 else None),
            "timing_quality": timing_quality,
            "grouping": {"group_by": group_by, "source": group_src,
                         "n_groups": cv.get("grouping", {}).get("n_groups")},
        },
    }
    warnings: list[str] = []
    warnings.extend(pr.meta.get("warnings", []) or [])
    warnings.extend(epochs.meta.get("warnings", []) or [])
    if len(epochs) == 0:
        warnings.append("no usable epochs: the test is reported as not evaluated")
    if trained is not None and trained.skipped:
        warnings.append(f"model not trained: {trained.meta.get('skipped')}")
    if coverage.get("retained_fraction") is not None and coverage["retained_fraction"] < 0.5:
        warnings.append(f"only {100*coverage['retained_fraction']:.0f}% of candidate epochs "
                        "were retained; the headline accuracy describes a subset")
    if timing_quality != "verified":
        warnings.append(f"timing_quality={timing_quality}: no ERP latency claims from this "
                        "session")
    results["warnings"] = warnings
    results["gates"] = apply_gates(spec, results)
    return _jsonable(results)


def train_and_evaluate(session_paths: Any, spec: TestSpec, save: bool = True,
                       *, evaluate_each: bool = True) -> dict:
    """Pool several sessions and evaluate cross-session generalisation.

    This is the honest question for one subject with many short sessions: does
    the decoder survive a new day, a new electrode contact and a new block?
    Groups encode ``session * 100000 + block`` so that leave-one-block-out never
    mixes a block across sessions, and a separate leave-one-*session*-out pass
    reports the harder number.

    ``session_paths`` may be one path/session or an iterable of them.
    """
    if isinstance(session_paths, (str,)) or hasattr(session_paths, "eeg"):
        paths = [session_paths]
    else:
        paths = list(session_paths)
    notes: list[str] = []
    per_session: list[dict[str, Any]] = []
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    gfs: list[np.ndarray] = []
    session_of: list[np.ndarray] = []
    names: list[str] = []
    qualities: list[dict[str, Any]] = []
    derived_list: list[dict[str, Any]] = []
    timings: list[str] = []

    for si, item in enumerate(paths):
        try:
            session = item if hasattr(item, "eeg") else _load_session(item)
        except Exception as exc:                  # noqa: BLE001
            notes.append(f"session {item}: {type(exc).__name__}: {exc}")
            continue
        pr = preprocess_session(session, spec)
        ep, ep_notes = build_epochs(session, spec, signal=pr.signal)
        notes.extend(f"[session {si}] {n}" for n in ep_notes)
        X, fn = _extract(spec, ep)
        names = fn or names
        tq, tq_notes = timing_quality_of(session, spec)
        timings.append(tq)
        notes.extend(f"[session {si}] {n}" for n in tq_notes)
        qualities.append(_quality_metrics(pr))
        entry: dict[str, Any] = {
            "session": str(getattr(getattr(session, "path", None), "name", f"session{si}")),
            "path": str(getattr(session, "path", "")),
            "n_epochs": int(len(ep)),
            "class_counts": ep.class_counts(),
            "epoching": ep.meta.get("epoching"),
            "coverage": coverage_of(ep),
            "timing_quality": tq,
            "quality": pr.quality.as_dict(),
        }
        if evaluate_each and len(ep) >= 4:
            try:
                entry["cv"] = cross_validate(spec, X, ep.y, ep.groups,
                                             groups_fine=ep.groups_fine)
            except Exception as exc:              # noqa: BLE001
                entry["cv"] = None
                notes.append(f"[session {si}] cv failed: {type(exc).__name__}: {exc}")
        per_session.append(entry)
        if len(ep):
            cv_s = entry.get("cv") or cross_validate(spec, X, ep.y, ep.groups,
                                                     groups_fine=ep.groups_fine)
            derived_list.append(_derived_metrics(spec, session, pr, ep, cv_s or {},
                                                 seed=_seed_of(spec), groups=ep.groups))
            Xs.append(np.asarray(X))
            ys.append(np.asarray(ep.y))
            gs.append(np.asarray(ep.groups, dtype=int) + si * 100000)
            gfs.append(np.asarray(ep.groups_fine, dtype=int) + si * 100000)
            session_of.append(np.full(len(ep), si, dtype=int))

    if not Xs:
        results = {
            "test_id": str(getattr(spec, "id", "")),
            "title": str(getattr(spec, "title", "")),
            "n_sessions": 0,
            "per_session": per_session,
            "per_session_summary": [],
            "cv": {"balanced_accuracy": None, "auc": None, "f1": None, "confusion": [],
                   "chance": None, "n": 0, "n_epochs": 0, "notes": ["no epochs in any session"]},
            "notes": notes,
            "warnings": ["no usable epochs across the given sessions"],
        }
        results["quality_metrics"] = {}
        results["derived_metrics"] = {}
        results["model"] = None
        results["model_path"] = None
        results["cross_session"] = None
        results["coverage"] = coverage_of(Epochs(
            X=np.zeros((0, 1), dtype=np.float32), y=np.zeros(0, dtype=int),
            labels=list(getattr(spec, "classes", []) or []), groups=np.zeros(0, dtype=int),
            times=np.zeros(0), fs=float(getattr(spec, "fs", 250.0)),
            onsets=np.zeros(0)))
        results["context_features"] = []
        results["gates"] = apply_gates(spec, results)
        return _jsonable(results)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    g_all = np.concatenate(gs, axis=0)
    gf_all = np.concatenate(gfs, axis=0)
    s_all = np.concatenate(session_of, axis=0)
    assert_signal_only(names)
    group_arr, group_by, group_src = resolve_grouping(spec, g_all, gf_all, s_all)
    pooled = cross_validate(spec, X_all, y_all, g_all, groups_fine=gf_all,
                            session_ids=s_all)

    cross_session = None
    if len(paths) > 1 and len(np.unique(s_all)) >= 2:
        rows = []
        for si in np.unique(s_all):
            tr = np.flatnonzero(s_all != si)
            te = np.flatnonzero(s_all == si)
            if tr.size == 0 or te.size == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            try:
                est = build_pipeline(spec, None, _seed_of(spec))
                est.fit(X_all[tr], y_all[tr])
                pred = est.predict(X_all[te])
                ba = float(balanced_accuracy_score(y_all[te], pred))
            except Exception as exc:              # noqa: BLE001
                rows.append({"session": str(si), "balanced_accuracy": None,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append({"session": str(si), "n_test": int(te.size),
                         "balanced_accuracy": ba})
        good = [r["balanced_accuracy"] for r in rows if r.get("balanced_accuracy") is not None]
        cross_session = {
            "folds": rows,
            "balanced_accuracy": float(np.mean(good)) if good else None,
            "n_sessions": len(paths),
            "note": ("held-out whole sessions: the hardest and most honest estimate for one "
                     "subject recorded on different days"),
        }

    trained = train(spec, X_all, y_all, g_all, names, _seed_of(spec))
    saved: str | None = None
    if save and trained.pipeline is not None:
        try:
            saved = str(save_model(trained))
        except Exception as exc:                  # noqa: BLE001
            notes.append(f"could not save model: {type(exc).__name__}: {exc}")

    merged_q: dict[str, Any] = {}
    for q in qualities:
        for k2, v in q.items():
            if isinstance(v, (int, float)):
                merged_q.setdefault(k2, []).append(float(v))
    quality_metrics = {k2: float(np.mean(v)) for k2, v in merged_q.items() if v}
    derived: dict[str, Any] = {}
    for d in derived_list:
        for k2, v in d.items():
            if isinstance(v, (int, float)) and k2 not in derived:
                derived[k2] = float(v)

    k = len(pooled.get("classes") or []) or len(getattr(spec, "classes", []) or [])
    trial_s = float((getattr(spec, "design", {}) or {}).get("soa_s") or 0.0)
    per_session_summary = [{
        "session": e.get("session"),
        "n_epochs": e.get("n_epochs"),
        "balanced_accuracy": ((e.get("cv") or {}).get("balanced_accuracy")),
        "auc": ((e.get("cv") or {}).get("auc")),
        "coverage": (e.get("coverage") or {}).get("retained_fraction"),
        "timing_quality": e.get("timing_quality"),
    } for e in per_session]
    # pooled coverage: sum of per-session candidate/retained counts
    cov_ret = sum((e.get("coverage") or {}).get("retained", 0) or 0 for e in per_session)
    cov_cand = sum((e.get("coverage") or {}).get("candidates", 0) or 0 for e in per_session)
    per_class: dict[str, Any] = {}
    for e in per_session:
        for lab, d2 in ((e.get("coverage") or {}).get("per_class") or {}).items():
            slot = per_class.setdefault(lab, {"retained": 0, "candidates": 0})
            slot["retained"] += int(d2.get("retained", 0) or 0)
            slot["candidates"] += int(d2.get("candidates", 0) or 0)
    for slot in per_class.values():
        slot["retained_fraction"] = (slot["retained"] / slot["candidates"]
                                     if slot["candidates"] else None)
    pooled_coverage = {
        "retained": cov_ret,
        "candidates": cov_cand,
        "retained_fraction": (cov_ret / cov_cand) if cov_cand else None,
        "per_class": per_class,
        "per_session": [(e.get("session"), (e.get("coverage") or {}).get("retained_fraction"))
                        for e in per_session],
        "note": "pooled across sessions; see per_session for the per-day retention",
    }
    overall_timing = "verified" if timings and all(t == "verified" for t in timings) else (
        "coarse" if timings and any(t == "coarse" for t in timings) else "unverified")
    results = {
        "test_id": str(getattr(spec, "id", "")),
        "title": str(getattr(spec, "title", "")),
        "feasibility": str(getattr(spec, "feasibility", "")),
        "n_sessions": len(paths),
        "sessions": [p.get("path", "") if isinstance(p, dict) else str(p)
                     for p in per_session],
        "per_session": per_session,
        "per_session_summary": per_session_summary,
        "coverage": pooled_coverage,
        "context_features": [],
        "signal_only_checked": True,
        "timing_quality": overall_timing,
        "cv": pooled,
        "cross_session": cross_session,
        "quality_metrics": quality_metrics,
        "derived_metrics": derived,
        "features": list(names),
        "model": trained.as_dict(),
        "model_path": saved,
        "itr_bits_per_min": information_transfer_rate(pooled.get("balanced_accuracy"),
                                                      k, trial_s) if trial_s > 0 else 0.0,
        "notes": notes,
        "warnings": [],
    }
    if pooled.get("balanced_accuracy") is None:
        results["warnings"].append("no cross-validated contrast could be computed")
    if cross_session:
        results["warnings"].append(
            "cross-session accuracy is the headline number; within-session accuracy is "
            "optimistic")
    if overall_timing != "verified":
        results["warnings"].append(
            f"timing_quality={overall_timing}: no ERP latency claims across these sessions")
    results["gates"] = apply_gates(spec, results)
    return _jsonable(results)


def _load_session(path: Any) -> Any:
    from ..store import Session
    return Session.load(path)
