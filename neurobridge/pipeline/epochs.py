"""Epoching: turning a continuous session into labelled, grouped trials.

Why the details matter here
---------------------------
* **Markers are filtered by ``test_id`` before anything else.** The catalog
  deliberately reuses marker codes between tests in the same tier (1210 means a
  different class in T201 and T203), and one recorded session can hold several
  tests back to back. Resolving a code against the whole session would silently
  mix two tests' trials. Every helper here starts from
  ``session.events_for(test_id=spec.id)``.
* **Onsets come from the monotonic timestamps, not sample counts.** BLE packets
  arrive in bursts and occasionally arrive late, so ``sample_index == t * fs``
  is *not* reliable over a 10 minute session. We locate each onset with
  ``np.searchsorted`` on ``session.t`` and linearly interpolate the fractional
  sample position between the two bracketing samples.
* **Latency compensation shifts the onset, never the signal.**
* **Labels come from class markers, not from the onset marker.** The runner (and
  the offline simulator) write the class marker immediately *before*
  ``stim_onset`` at the same timestamp, so "most recent class marker at or
  before the onset" is the primary rule, with the onset's own code/name as a
  shortcut when the onset *is* the class marker.
* **T101/T108 back-date a ``lapse``/``missed`` marker** to an earlier probe
  timestamp, so events on disk are not guaranteed to be in insertion order.
  ``Session.load`` sorts by time and we re-sort defensively; nothing here
  assumes insertion order.
* **Groups are mandatory for honest cross-validation.** One subject, many short
  blocks: epochs from the same block share electrode drift, alertness and
  impedance. Random epoch-level CV would let the model memorise block-specific
  offsets. Grouped CV answers the question we actually care about.
* **Rejected epochs are counted and reported, not hidden.** A test that throws
  away 60% of trials is not the same experiment as one that throws away 2%.
* **No events must not be an exception.** Sessions get aborted, a marker name
  gets typo'd, or a test is run before its stimuli exist. Returning an empty
  :class:`Epochs` with a warning lets the CLI and the agent bridge degrade and
  say "not run" instead of crashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..store import MARKER_CODES
from .preprocess import (artifact_mask, artifact_ptp_limit, detect_artifacts,
                         latency_offset_s)

__all__ = [
    "Epochs",
    "epochs_from_arrays",
    "epochs_from_session",
    "windowed_epochs",
]

#: Markers that end the current block / sustained condition. After one of these
#: the "current state" is undefined until the next class marker: windows in the
#: inter-block rest must not inherit the previous condition's label.
_BOUNDARY_NAMES = ("block_end", "break_start", "break_end", "test_end", "session_end")
_BLOCK_START_NAMES = ("block_start", "test_start", "practice_start")

#: Default bound on how far back a class marker may sit and still label an onset.
_DEFAULT_LOOKBACK_S = 5.0


def _norm(label: Any) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def _class_lookup(classes: Sequence[str]) -> dict[str, str]:
    """normalised label -> canonical class string (as written in the spec)."""
    return {_norm(c): str(c) for c in classes}


def _reject_policy(spec: Any) -> str:
    proc = getattr(spec, "processing", {}) or {}
    return str(proc.get("reject_policy", "reject")).lower()


@dataclass
class Epochs:
    """A labelled, grouped set of single-channel windows.

    ``X`` is ``(n_epochs, n_samples)`` float32 for ordinary features, or
    ``(n_epochs, n_channels, n_channels)`` when the feature list is exactly
    ``cov:tangent`` (the Riemannian decoder wants matrices, not scalars).

    ``groups`` is the **block** id (existing consumers rely on it);
    ``groups_fine`` is the **trial / passage** id. Overlapping windows cut from
    one trial are not independent observations, so cross-validation must be able
    to group on the finer unit as well as the coarser one. When a session has no
    trial information, ``groups_fine`` falls back to ``groups`` (documented
    degradation, never a silent claim of independence).
    """

    X: np.ndarray
    y: np.ndarray
    labels: list[str]
    groups: np.ndarray
    times: np.ndarray
    fs: float
    onsets: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    groups_fine: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.groups_fine is None:
            self.groups_fine = np.asarray(self.groups, dtype=int)
        else:
            self.groups_fine = np.asarray(self.groups_fine, dtype=int)

    def __len__(self) -> int:
        return int(self.X.shape[0]) if self.X.ndim else 0

    def class_counts(self) -> dict[str, int]:
        """Number of epochs per class label (labels with zero epochs included)."""
        out = {lab: 0 for lab in self.labels}
        for yi in np.asarray(self.y).reshape(-1):
            if 0 <= int(yi) < len(self.labels):
                out[self.labels[int(yi)]] += 1
        return out

    def balance(self) -> "Epochs":
        """Subsample every class down to the smallest class size.

        Deliberately deterministic and order-preserving: we take the *earliest*
        ``k`` epochs of each class rather than a random subset. Random
        subsampling would (a) make results depend on a seed nobody recorded and
        (b) separate epochs from their block context when a block is only
        partially represented, which weakens the grouped CV that follows.

        The returned object keeps the original temporal order and the original
        group ids, so no group membership is ever "shuffled apart".
        """
        y = np.asarray(self.y).reshape(-1)
        if y.size == 0:
            return self
        counts = [int(np.count_nonzero(y == i)) for i in range(len(self.labels))]
        present = [c for c in counts if c > 0]
        if not present:
            return self
        k = min(present)
        keep: list[int] = []
        for i in range(len(self.labels)):
            idx = np.flatnonzero(y == i)[:k]
            keep.extend(int(j) for j in idx)
        keep_arr = np.array(sorted(keep), dtype=int)
        meta = dict(self.meta)
        meta["balanced_to"] = int(k)
        meta["balanced_from"] = int(y.size)
        return Epochs(
            X=self.X[keep_arr], y=y[keep_arr],
            labels=list(self.labels), groups=np.asarray(self.groups)[keep_arr],
            times=np.asarray(self.times), fs=float(self.fs),
            onsets=np.asarray(self.onsets)[keep_arr], meta=meta,
            groups_fine=np.asarray(self.groups_fine)[keep_arr],
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (never the raw arrays)."""
        return {
            "n": len(self),
            "class_counts": self.class_counts(),
            "n_groups": int(len(np.unique(self.groups))) if self.groups.size else 0,
            "n_fine_groups": (int(len(np.unique(self.groups_fine)))
                              if self.groups_fine is not None and self.groups_fine.size else 0),
            "groups": [int(g) for g in np.unique(self.groups)] if self.groups.size else [],
            "n_samples_per_epoch": int(self.X.shape[-1]) if self.X.ndim >= 2 else 0,
            "fs": float(self.fs),
            "tmin_s": float(self.times[0]) if self.times.size else None,
            "tmax_s": float(self.times[-1]) if self.times.size else None,
            "meta": dict(self.meta),
        }


# ---------------------------------------------------------------- event access
def session_events(session: Any, spec: Any) -> tuple[list[Any], str]:
    """Events belonging to ``spec``, filtered by ``test_id`` first.

    Returns ``(events, source)`` where source is ``'test_id'`` or
    ``'whole_session'`` (fallback for hand-built sessions that never tagged
    their markers). The fallback is recorded so a report can say the marker
    filter was unavailable.
    """
    tid = str(getattr(spec, "id", "") or "")
    events: list[Any] = []
    if tid and hasattr(session, "events_for"):
        try:
            events = list(session.events_for(test_id=tid))
        except Exception:
            events = []
    if events:
        return sorted(events, key=lambda e: float(getattr(e, "t_mono", 0.0))), "test_id"
    all_events = list(getattr(session, "events", []) or [])
    return sorted(all_events, key=lambda e: float(getattr(e, "t_mono", 0.0))), "whole_session"


def _event_label(ev: Any, classes: Sequence[str], lookup: dict[str, str],
                 lmap: dict[int, str]) -> str | None:
    """Class label carried by an event, or ``None``.

    Order: the spec's code table (authoritative, because codes are what the
    runner actually logs), then the marker's own name, then a label in ``meta``
    (the simulator writes ``meta['class']``). Names are normalised so
    ``Class A`` and ``class_a`` resolve to the same spec label.
    """
    lab = lmap.get(int(getattr(ev, "code", -1)))
    if lab in classes:
        return lab
    name = _norm(getattr(ev, "name", "") or "")
    if name in lookup:
        return lookup[name]
    meta = getattr(ev, "meta", None) or {}
    for key in ("class", "label", "condition", "state"):
        cand = meta.get(key)
        if cand is not None and _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


def _is_class_marker(ev: Any, classes: Sequence[str], lookup: dict[str, str]) -> bool:
    """True when the marker names a class (name or ``meta['class']``)."""
    name = _norm(getattr(ev, "name", "") or "")
    if name in lookup:
        return True
    meta = getattr(ev, "meta", None) or {}
    for key in ("class", "label"):
        cand = meta.get(key)
        if cand is not None and _norm(cand) in lookup:
            return True
    return False


def _is_boundary(ev: Any) -> bool:
    name = str(getattr(ev, "name", "") or "")
    code = int(getattr(ev, "code", -1))
    return name in _BOUNDARY_NAMES or code in {int(MARKER_CODES.get(n, -999))
                                               for n in _BOUNDARY_NAMES}


def _is_block_start(ev: Any) -> bool:
    name = str(getattr(ev, "name", "") or "")
    code = int(getattr(ev, "code", -1))
    return name in _BLOCK_START_NAMES or code == int(MARKER_CODES.get("block_start", -998))


# ---------------------------------------------------------------- index maths
def _epoch_indices(t: np.ndarray, onsets: Sequence[float], fs: float,
                   tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Map onset times to integer sample starts.

    Returns ``(starts, ok)`` where ``ok`` marks windows fully inside the
    recording. The fractional position comes from linear interpolation between
    the two samples that bracket the onset, so a 3 ms timing error becomes a
    sub-sample correction rather than a one-sample (4 ms at 250 Hz) jump.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n_ep = max(1, int(round((float(tmax) - float(tmin)) * fs)))
    starts = np.zeros(len(onsets), dtype=int)
    ok = np.zeros(len(onsets), dtype=bool)
    if t.size == 0:
        return starts, ok

    for i, onset in enumerate(onsets):
        onset = float(onset)
        j = int(np.searchsorted(t, onset, side="left"))
        j = min(max(j, 1), t.size - 1)
        t0, t1 = float(t[j - 1]), float(t[j])
        frac = (onset - t0) / (t1 - t0) if t1 > t0 else 0.0
        pos = (j - 1) + frac                       # fractional sample index
        start = int(round(pos + tmin * fs))
        end = start + n_ep
        starts[i] = start
        ok[i] = (start >= 0) and (end <= t.size)
    return starts, ok


def _apply_baseline(seg: np.ndarray, fs: float, tmin: float,
                    baseline: Sequence[float] | None) -> np.ndarray:
    """Subtract the mean of ``baseline`` (seconds relative to onset).

    Baseline correction is what makes single-trial ERP amplitudes comparable
    across a drifting session: without it the pre-stimulus offset (electrode
    drift) is added to every amplitude feature.
    """
    if baseline is None:
        return seg
    n = seg.shape[-1]
    lo, hi = float(baseline[0]), float(baseline[1])
    i0 = int(round((lo - tmin) * fs))
    i1 = int(round((hi - tmin) * fs))
    i0 = min(max(i0, 0), n)
    i1 = min(max(i1, i0 + 1), n)
    if i1 <= i0:
        return seg
    base = seg[..., i0:i1].mean(axis=-1, keepdims=True)
    return seg - base


def epochs_from_arrays(values: np.ndarray, fs: float, onsets: Sequence[float],
                       tmin: float, tmax: float,
                       baseline: Sequence[float] | None = None,
                       t: np.ndarray | None = None) -> np.ndarray:
    """Slice fixed windows out of an array; returns ``(n_epochs, n_samples)``.

    ``t`` optionally supplies per-sample timestamps aligned with ``values``
    (monotonic seconds). Without it, sample ``i`` is assumed to be at
    ``i / fs``. Windows that do not fit inside the recording are dropped,
    because a half-window epoch silently biases every ERP feature.

    The extra keyword ``t`` is an addition to the frozen signature; existing
    positional calls are unaffected.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim > 1:
        v = v.reshape(-1) if v.shape[0] == 1 else v[0]
    v = v.reshape(-1)
    fs = float(fs)
    tt = np.arange(v.size, dtype=np.float64) / fs if t is None else np.asarray(t, float)
    starts, ok = _epoch_indices(tt, onsets, fs, float(tmin), float(tmax))
    n_ep = max(1, int(round((float(tmax) - float(tmin)) * fs)))
    keep = np.flatnonzero(ok)
    if keep.size == 0:
        return np.zeros((0, n_ep), dtype=np.float32)
    out = np.empty((keep.size, n_ep), dtype=np.float64)
    for row, i in enumerate(keep):
        s = starts[i]
        seg = v[s: s + n_ep].copy()
        out[row] = _apply_baseline(seg, fs, float(tmin), baseline)
    return out.astype(np.float32)


def _empty(labels: Sequence[str], fs: float, tmin: float, tmax: float,
           meta: dict[str, Any]) -> Epochs:
    n_ep = max(1, int(round((float(tmax) - float(tmin)) * fs)))
    times = float(tmin) + np.arange(n_ep, dtype=np.float64) / max(fs, 1e-9)
    return Epochs(X=np.zeros((0, n_ep), dtype=np.float32),
                  y=np.zeros(0, dtype=int), labels=list(labels),
                  groups=np.zeros(0, dtype=int), times=times, fs=float(fs),
                  onsets=np.zeros(0, dtype=float), meta=meta)


# ---------------------------------------------------------------- stimulus-locked
def epochs_from_session(session: Any, spec: Any, signal: np.ndarray | None = None,
                        event_name: str = "stim_onset", reject: bool = True) -> Epochs:
    """Build :class:`Epochs` from a session, its spec and (optionally) a signal.

    Steps, in order:

    1. take only the events tagged with ``spec.id`` (see :func:`session_events`);
    2. onsets are the events named in ``processing.onset_markers`` (plus
       ``event_name``); if none exist, class markers are used as onsets so that
       synthetic/hand-built sessions still work;
    3. label each epoch from the onset's own code/name, otherwise from the most
       recent *class* marker at or before the onset within a bounded lookback
       (``processing.label_lookback_s``, default 5 s). ``meta['label_source']``
       records which rule was used and how often;
    4. slice, baseline-correct, and (if ``reject``) drop epochs touching bad
       samples or exceeding ``processing.artifact_ptp``.

    Callers should pass ``reject=False`` for ``reject_policy: flag`` specs
    (e.g. T701, where the blink artefact *is* the signal); the counts of what
    would have been rejected are always recorded.
    """
    labels = list(getattr(spec, "classes", []) or [])
    lookup = _class_lookup(labels)
    lmap = dict(getattr(spec, "label_map", lambda: {})() or {})
    tmin, tmax = getattr(spec, "epoch", (-0.2, 0.8))
    tmin, tmax = float(tmin), float(tmax)
    fs = float(getattr(spec, "fs", None) or getattr(session, "fs", 250.0))
    proc = dict(getattr(spec, "processing", {}) or {})
    baseline = proc.get("baseline", None)
    ptp = artifact_ptp_limit(spec)

    events, ev_source = session_events(session, spec)
    meta: dict[str, Any] = {
        "event_name": event_name,
        "event_filter": ev_source,
        "n_events": len(events),
        "tmin_s": tmin, "tmax_s": tmax,
        "baseline": list(baseline) if baseline is not None else None,
        "artifact_ptp": None if ptp is None else float(ptp),
        "reject": bool(reject),
        "reject_policy": _reject_policy(spec),
        "label_map": {str(k): v for k, v in lmap.items()},
        "warnings": [],
    }
    if ev_source == "whole_session":
        meta["warnings"].append(
            f"no events tagged test_id={spec.id!r}: using every event in the session")

    # -- signal ------------------------------------------------------
    if signal is None:
        from .preprocess import preprocess_session
        pr = preprocess_session(session, spec)
        sig = np.asarray(pr.signal, dtype=np.float64)
        artifacts = pr.artifacts
        meta["latency_offset_s"] = float(pr.meta.get("latency_offset_s", 0.0))
    else:
        sig = np.asarray(signal, dtype=np.float64).reshape(-1)
        artifacts = detect_artifacts(sig, fs, ptp_limit=ptp)
        meta["latency_offset_s"] = float(latency_offset_s(spec))
    offset = float(meta["latency_offset_s"])
    if offset:
        meta["latency_note"] = ("onsets shifted by +%.4f s (stimulus/acquisition "
                                "latency); signal unchanged" % offset)

    t = np.asarray(getattr(session, "t", np.zeros(0)), dtype=np.float64).reshape(-1)
    if t.size == 0:
        meta["warnings"].append("session has no sample timestamps; cannot epoch")
        return _empty(labels, fs, tmin, tmax, meta)

    onset_names = {str(event_name)}
    for nm in (proc.get("onset_markers") or []):
        onset_names.add(str(nm))
    lookback = float(proc.get("label_lookback_s", _DEFAULT_LOOKBACK_S))

    # -- single pass in time order: block counter, current class, onsets --
    cand: list[tuple[Any, str | None, str, int]] = []   # ev, label, source, group
    n_from_onset = 0
    n_from_last = 0
    block = 0
    cur_group = 0
    last_class: tuple[float, str] | None = None
    onset_seq: dict[int, int] = {}                      # per-block fallback trial counter

    def _fine(grp: int, ev: Any) -> int:
        """Fine grouping key: block * 100000 + trial (or onset ordinal).

        Encodes block+trial in one integer so ``LeaveOneGroupOut`` on
        ``groups_fine`` never merges two blocks' trial 3 into the same fold.
        """
        trial = getattr(ev, "trial", None)
        if trial is None:
            onset_seq[grp] = onset_seq.get(grp, 0) + 1
            trial = onset_seq[grp]
        return int(grp) * 100000 + int(trial)

    def _resolve(ev: Any) -> tuple[str | None, str]:
        """(label, source) for this event, using the class-marker fallback."""
        own = _event_label(ev, labels, lookup, lmap)
        if own is not None:
            return own, "onset_marker"
        if last_class is not None:
            lt, ll = last_class
            if 0.0 <= (float(ev.t_mono) - lt) <= lookback:
                return ll, "last_class_marker"
        return None, "none"

    for ev in events:
        if getattr(ev, "block", None) is not None:
            cur_group = int(ev.block)
        if _is_block_start(ev):
            block += 1
            if getattr(ev, "block", None) is None:
                cur_group = block
        if _is_class_marker(ev, labels, lookup):
            lab_here = _event_label(ev, labels, lookup, lmap)
            if lab_here is not None:
                last_class = (float(ev.t_mono), lab_here)
        if str(getattr(ev, "name", "")) in onset_names:
            lab, src = _resolve(ev)
            if src == "onset_marker":
                n_from_onset += 1
            elif src == "last_class_marker":
                n_from_last += 1
            cand.append((ev, lab, src, cur_group))

    if not cand:
        meta["warnings"].append(
            f"no onset events {sorted(onset_names)} for {spec.id}: "
            "falling back to class markers as onsets")
        block = 0
        cur_group = 0
        for ev in events:
            if getattr(ev, "block", None) is not None:
                cur_group = int(ev.block)
            if _is_block_start(ev):
                block += 1
                if getattr(ev, "block", None) is None:
                    cur_group = block
            if _is_class_marker(ev, labels, lookup):
                lab = _event_label(ev, labels, lookup, lmap)
                if lab is not None:
                    cand.append((ev, lab, "class_marker_onset", cur_group))
        meta["onset_source"] = "class_markers"
    else:
        meta["onset_source"] = "onset_markers"

    meta["n_onset_events"] = len(cand)
    meta["n_candidates"] = len(cand)
    meta["n_labelled_from_onset"] = n_from_onset
    meta["n_labelled_from_last_class"] = n_from_last

    if not cand:
        meta["label_source"] = "none"
        meta["warnings"].append(
            "no labelled events found; returning empty Epochs rather than raising")
        return _empty(labels, fs, tmin, tmax, meta)

    kept = [(ev, lab, src, grp) for ev, lab, src, grp in cand
            if lab is not None and lab in labels]
    n_unknown = len(cand) - len(kept)
    if n_unknown:
        meta["n_dropped_unknown_label"] = n_unknown
    if not kept:
        meta["label_source"] = "none"
        meta["warnings"].append("every onset had a label outside spec.classes")
        return _empty(labels, fs, tmin, tmax, meta)

    srcs = {src for _, _, src, _ in kept}
    if srcs == {"onset_marker"}:
        meta["label_source"] = "onset_marker"
    elif srcs == {"last_class_marker"}:
        meta["label_source"] = "last_class_marker"
    elif srcs == {"class_marker_onset"}:
        meta["label_source"] = "class_marker_onset"
    else:
        meta["label_source"] = "mixed:" + "+".join(sorted(srcs))

    onsets = np.array([float(ev.t_mono) + offset for ev, _, _, _ in kept], dtype=float)
    y_all = np.array([labels.index(lab) for _, lab, _, _ in kept], dtype=int)
    groups_all = np.array([int(g) for _, _, _, g in kept], dtype=int)
    fine_all = np.array([_fine(int(g), ev) for ev, _, _, g in kept], dtype=int)
    meta["candidate_counts"] = {lab: int(sum(1 for _, l, _, _ in kept if l == lab))
                                for lab in labels}
    meta["grouping"] = {"block_from": "event.block or running block counter",
                        "fine_from": "event.trial or per-block onset ordinal"}

    # -- slice --------------------------------------------------------
    starts, ok = _epoch_indices(t, onsets, fs, tmin, tmax)
    n_ep = max(1, int(round((tmax - tmin) * fs)))
    policy = _reject_policy(spec)
    hard_reject = bool(reject) and policy == "reject"

    keep_rows: list[int] = []
    n_out_of_range = 0
    n_bad = 0
    n_ptp = 0
    n_flagged = 0
    rej_bad_by_label: dict[str, int] = {lab: 0 for lab in labels}
    rej_ptp_by_label: dict[str, int] = {lab: 0 for lab in labels}
    bad_mask = artifact_mask(sig, fs, ptp_limit=ptp) if sig.size else np.zeros(0, bool)
    lab_of_row = [labels[int(v)] for v in y_all]
    for i in range(len(onsets)):
        if not ok[i]:
            n_out_of_range += 1
            continue
        s = starts[i]
        seg = sig[s: s + n_ep]
        bad_sample = bool(bad_mask.size >= s + n_ep and bad_mask[s: s + n_ep].any())
        over_ptp = bool(ptp is not None and seg.size and float(np.ptp(seg)) > float(ptp))
        if bad_sample or over_ptp:
            if hard_reject:
                if bad_sample:
                    n_bad += 1
                    rej_bad_by_label[lab_of_row[i]] += 1
                else:
                    n_ptp += 1
                    rej_ptp_by_label[lab_of_row[i]] += 1
                continue
            n_flagged += 1
        keep_rows.append(i)

    meta.update({
        "n_dropped_out_of_range": int(n_out_of_range),
        "n_rejected_bad_samples": int(n_bad),
        "n_rejected_ptp": int(n_ptp),
        "n_rejected": int(n_out_of_range + n_bad + n_ptp),
        "n_flagged_kept": int(n_flagged),
        "hard_reject": bool(hard_reject),
        "artifacts": artifacts,
        "exclusion_counts": {
            "unknown_label": int(n_unknown),
            "out_of_range": int(n_out_of_range),
            "artifact_bad_samples": int(n_bad),
            "artifact_ptp": int(n_ptp),
        },
        "rejected_by_label": {lab: int(rej_bad_by_label[lab] + rej_ptp_by_label[lab])
                              for lab in labels},
    })
    if not keep_rows:
        meta["warnings"].append("all epochs were rejected (artefact or out of range)")
        return _empty(labels, fs, tmin, tmax, meta)

    rows = np.array(keep_rows, dtype=int)
    X = np.empty((rows.size, n_ep), dtype=np.float64)
    for k, i in enumerate(rows):
        s = starts[i]
        seg = sig[s: s + n_ep].copy()
        X[k] = _apply_baseline(seg, fs, tmin, baseline)
    X = X.astype(np.float32)

    times = tmin + np.arange(n_ep, dtype=np.float64) / fs
    return Epochs(X=X, y=y_all[rows], labels=labels, groups=groups_all[rows],
                  times=times, fs=fs, onsets=onsets[rows], meta=meta,
                  groups_fine=fine_all[rows])


# ---------------------------------------------------------------- sustained state
def _state_label(ev: Any, classes: Sequence[str], lookup: dict[str, str],
                 label_from: str) -> str | None:
    """Class label for a sustained-state marker.

    ``label_from`` names the meta key the runner uses (``state`` by default).
    Falling back to ``class``/``label`` and then to the marker name keeps this
    usable with markers written by older code paths.
    """
    meta = getattr(ev, "meta", None) or {}
    keys = [label_from] if label_from else []
    keys += ["state", "class", "label", "condition"]
    for key in keys:
        cand = meta.get(key)
        if cand is not None and _norm(cand) in lookup:
            return lookup[_norm(cand)]
    name = _norm(getattr(ev, "name", "") or "")
    return lookup.get(name)


def windowed_epochs(session: Any, spec: Any, window_s: float = 2.0,
                    step_s: float = 0.5, label_from: str = "state",
                    signal: np.ndarray | None = None) -> Epochs:
    """Tile a session into fixed windows labelled by the current condition.

    Sustained-state tests (alpha reactivity, cognitive load, blink switch) have
    no single stimulus-locked response to average; the unit of analysis is a
    window of ongoing EEG whose label is whatever *class marker* was most
    recently written. Rules, in the order they are applied:

    * only events tagged with ``spec.id`` are considered;
    * the label comes from the most recent class marker at or before the window
      start (``<=``, in the stable time-sorted order, so a class marker written
      immediately before ``stim_onset`` at the same timestamp wins);
    * a window with a class *change* strictly inside it is skipped: it contains
      two conditions and no single label describes it. A repeated marker with the
      *same* label is a refresh, not a change, and does not split the window; a
      block/break boundary always does, so windows never span the inter-block
      rest;
    * a window starting before any class marker is skipped;
    * windows after a block/break boundary are skipped until the next class
      marker, so the inter-block rest is never mislabelled;
    * ``reject_policy: reject`` additionally drops windows touching bad samples
      or exceeding ``artifact_ptp``; ``flag`` keeps them and counts them.

    Groups follow the governing marker's ``block`` (or the running block
    counter), so a subsequent grouped CV respects the block structure.
    """
    labels = list(getattr(spec, "classes", []) or [])
    lookup = _class_lookup(labels)
    lmap = dict(getattr(spec, "label_map", lambda: {})() or {})
    proc = dict(getattr(spec, "processing", {}) or {})
    label_from = str(proc.get("label_from", label_from) or label_from)
    fs = float(getattr(spec, "fs", None) or getattr(session, "fs", 250.0))
    window_s = float(max(window_s, 1.0 / max(fs, 1.0)))
    step_s = float(max(step_s, 1.0 / max(fs, 1.0)))
    ptp = artifact_ptp_limit(spec)
    policy = _reject_policy(spec)
    hard_reject = policy == "reject"

    events, ev_source = session_events(session, spec)
    meta: dict[str, Any] = {
        "window_s": window_s, "step_s": step_s, "label_from": label_from,
        "event_filter": ev_source, "reject_policy": policy,
        "n_events": len(events),
        "label_source": "class_markers",
        "warnings": [],
    }
    if ev_source == "whole_session":
        meta["warnings"].append(
            f"no events tagged test_id={spec.id!r}: using every event in the session")

    if signal is None:
        from .preprocess import preprocess_session
        pr = preprocess_session(session, spec)
        sig = np.asarray(pr.signal, dtype=np.float64).reshape(-1)
        meta["artifacts"] = pr.artifacts
        meta["latency_offset_s"] = float(pr.meta.get("latency_offset_s", 0.0))
    else:
        sig = np.asarray(signal, dtype=np.float64).reshape(-1)
        meta["artifacts"] = detect_artifacts(sig, fs, ptp_limit=ptp)
        meta["latency_offset_s"] = float(latency_offset_s(spec))
    t = np.asarray(getattr(session, "t", np.zeros(0)), dtype=np.float64).reshape(-1)
    if sig.size == 0 or t.size == 0:
        meta["warnings"].append("empty session")
        return _empty(labels, fs, -window_s, 0.0, meta)

    n_win = max(1, int(round(window_s * fs)))
    times = -window_s + np.arange(n_win, dtype=np.float64) / fs
    bad_mask = artifact_mask(sig, fs, ptp_limit=ptp)

    # Pre-extract the marker information in stable time order.
    ev_times = np.array([float(e.t_mono) for e in events], dtype=float)
    ev_labels = [_state_label(e, labels, lookup, label_from) for e in events]
    ev_is_class = [_is_class_marker(e, labels, lookup) or ev_labels[i] is not None
                   for i, e in enumerate(events)]
    ev_boundary = [_is_boundary(e) for e in events]
    ev_trial = [getattr(e, "trial", None) for e in events]
    ev_group = [0] * len(events)
    # A *segment* is one contiguous run of the same condition. It is the natural
    # "trial"/"passage" unit for grouped CV on sustained tests: every overlapping
    # window cut from one 30 s condition must stay in one fold.
    ev_seg = [0] * len(events)
    run_block = 0
    seg_rank = 0
    cur_lab: str | None = None
    for i, e in enumerate(events):
        if _is_block_start(e):
            run_block += 1
        if getattr(e, "block", None) is not None:
            run_block = int(e.block)
        ev_group[i] = run_block
        if ev_is_class[i]:
            lab_i = ev_labels[i]
            if lab_i != cur_lab:
                seg_rank += 1
                cur_lab = lab_i
        if ev_boundary[i]:
            cur_lab = None
        ev_seg[i] = seg_rank

    def _fine(i: int) -> int:
        trial = ev_trial[i]
        if trial is None:
            trial = ev_seg[i]
        return int(ev_group[i]) * 100000 + int(trial)

    # A *change point* is either a class marker whose label differs from the
    # previous one, or a block/break boundary. Repeated markers of the same label
    # are condition *refreshes* (the runner re-announces a long condition every
    # ~20 s so a dropped marker cannot erase the state) and must NOT split a
    # window -- treating them as changes threw away most windows inside long
    # conditions and left an unrepresentative handful.
    change_times: list[float] = []
    prev_lab: str | None = None
    for i, e in enumerate(events):
        if ev_is_class[i]:
            lab_i = ev_labels[i]
            if lab_i != prev_lab:
                change_times.append(float(ev_times[i]))
                prev_lab = lab_i
        if ev_boundary[i]:
            change_times.append(float(ev_times[i]))
            prev_lab = None
    class_times = np.array(sorted(set(change_times))) if change_times else np.zeros(0)

    starts = np.arange(t[0], t[-1] - window_s + 1e-9, step_s)
    X_rows: list[np.ndarray] = []
    ys: list[int] = []
    gs: list[int] = []
    gfs: list[int] = []
    ons: list[float] = []
    n_straddle = 0
    n_unlabelled = 0
    n_bad = 0
    n_flagged = 0
    cand_by_label: dict[str, int] = {lab: 0 for lab in labels}
    rej_by_label: dict[str, int] = {lab: 0 for lab in labels}

    for s in starts:
        e = s + window_s
        # (a) class change inside the window?
        if class_times.size and np.any((class_times > s + 1e-9) & (class_times < e - 1e-9)):
            n_straddle += 1
            continue
        # (b) most recent event at or before the window start
        before = np.flatnonzero(ev_times <= s + 1e-9)
        if before.size == 0:
            n_unlabelled += 1
            continue
        j = int(before[-1])
        if not ev_is_class[j]:
            # a framing marker (block_end, cue, ...) is the most recent thing:
            # the current condition is undefined, so skip rather than guess.
            n_unlabelled += 1
            continue
        lab = ev_labels[j]
        if lab is None or lab not in labels:
            n_unlabelled += 1
            continue
        cand_by_label[lab] += 1
        s_idx = int(np.searchsorted(t, s, side="left"))
        e_idx = s_idx + n_win
        if e_idx > sig.size:
            n_unlabelled += 1
            continue
        if bad_mask.size >= e_idx and bool(bad_mask[s_idx:e_idx].any()):
            if hard_reject:
                n_bad += 1
                rej_by_label[lab] += 1
                continue
            n_flagged += 1
        seg = sig[s_idx:e_idx]
        if ptp is not None and seg.size and float(np.ptp(seg)) > float(ptp):
            if hard_reject:
                n_bad += 1
                rej_by_label[lab] += 1
                continue
            n_flagged += 1
        X_rows.append(seg.astype(np.float32))
        ys.append(labels.index(lab))
        gs.append(int(ev_group[j]))
        gfs.append(_fine(j))
        ons.append(float(s))

    meta.update({
        "n_windows_considered": int(starts.size),
        "n_candidates": int(starts.size),
        "n_skipped_straddle": int(n_straddle),
        "n_skipped_unlabelled": int(n_unlabelled),
        "n_rejected_bad_samples": int(n_bad),
        "n_flagged_kept": int(n_flagged),
        "hard_reject": bool(hard_reject),
        "candidate_counts": cand_by_label,
        "rejected_by_label": rej_by_label,
        "exclusion_counts": {
            "straddle_class_change": int(n_straddle),
            "unlabelled_or_out_of_range": int(n_unlabelled),
            "artifact_bad_samples": int(n_bad),
        },
        "grouping": {"block_from": "event.block or running block counter",
                     "fine_from": "event.trial or contiguous condition segment"},
    })
    if not X_rows:
        meta["warnings"].append("no window could be labelled and kept")
        return _empty(labels, fs, -window_s, 0.0, meta)

    return Epochs(
        X=np.stack(X_rows, axis=0).astype(np.float32),
        y=np.array(ys, dtype=int), labels=labels,
        groups=np.array(gs, dtype=int), times=times, fs=fs,
        onsets=np.array(ons, dtype=float), meta=meta,
        groups_fine=np.array(gfs, dtype=int),
    )
