"""Trial construction and phase scheduling for every design shape.

Why this module exists separately from the runner
--------------------------------------------------
The *content* of a protocol (how many trials, how classes are balanced, where
the rare class goes, how trials are split into blocks, how long a condition
lasts) is pure data.  Keeping it out of the acquisition loop means it can be
unit-tested offline, seeded and reproduced exactly, and inspected before a
single BLE packet is acquired.  :func:`build_trials` is deterministic given
``rng``.

Three design shapes, one trial model
------------------------------------
The catalog is not uniform.  It mixes:

* **trial-based** designs (``soa_s`` + ``trials``): one short stimulus per
  trial with a jittered inter-onset interval;
* **phase-based** designs (``phases_per_block`` + ``phase_duration_s``):
  alternating sustained windows inside a block;
* **block/condition-based** designs (``block_duration_s``,
  ``condition_duration_s``, ``run_s``, ``level_duration_s``,
  ``passage_duration_s``, ``task_window_s``, ``trials_per_block``, ...): a
  class-labelled window per block or per condition.

All three reduce to the same :class:`Trial`: a class label, an *onset marker
code*, a **body** (how long the stimulus/condition is held) and an **itv**
(onset-to-onset interval, >= body).  The runner never needs to know which shape
produced a trial -- which is what keeps one acquisition loop correct for 64
protocols.  The shape and the design keys that were consumed are reported by
:func:`plan_summary` so a session can be audited after the fact.

Timing conventions (derived from the catalog, not invented)
-----------------------------------------------------------
* With ``soa_s`` present, ``iti_jitter_s`` usually *is* the jittered SOA: T004
  writes ``soa_s: 1.0, iti_jitter_s: [0.85, 1.15]``.  So when the jitter band
  reaches the SOA we draw the interval directly from the band.
* When the band is clearly shorter than the SOA (T401: ``soa 1.2, jitter
  [0.15, 0.35]``) it is *added* after the SOA.
* Without ``soa_s`` the band is always extra jitter after the body: T701
  ``response_window_s 2.0, iti_jitter_s [1.8, 2.2]`` gives a ~4 s trial, which
  is exactly its ``trial_s``.
* ``p_target`` marks the **last** class as rare (catalogs list the frequent
  class first): T301 ``[standard, deviant]`` -> deviant, T101 ``[target,
  lapse]`` -> lapse.  A rare class is never scheduled twice in a row.

Jitter matters
--------------
If every stimulus arrives at a fixed offset from the previous one, ongoing
alpha is phase-locked to the stimulus train and the averaged ERP is
contaminated by entrainment.  :func:`jitter` reads the spec's bounds
defensively: a missing/None/scalar/nan value degrades to zero instead of
crashing a 12-minute session at trial 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

import numpy as np

from .. import store
from ..catalog.schema import TestSpec
from ..stimuli.base import Stimulus, StimulusSet

#: Fallback presentation length per stimulus kind, used when neither the
#: Stimulus nor ``stimuli.duration_s`` nor ``design.soa_s`` says anything.
DEFAULT_DURATION_S: dict[str, float] = {
    "image": 0.5,
    "text": 2.5,
    "audio": 1.0,
    "tone": 0.3,
    "none": 0.0,
}

#: Design keys that give a per-condition / per-trial window length, in the
#: priority order the integrator specified.  First numeric match wins.
WINDOW_KEYS: tuple[str, ...] = (
    "condition_duration_s", "run_s", "phase_duration_s", "level_duration_s",
    "passage_duration_s", "task_window_s", "imagery_window_s", "trial_s",
    "attempt_duration_s", "flicker_duration_s", "flash_duration_s",
    "picture_duration_s",
)

#: Classes that mean "the subject failed to respond".  When one is present a
#: trial is *outcome-gated*: its class is only known after the response window
#: and the marker is back-dated to the probe timestamp (T101, T108).
MISS_CLASSES: frozenset[str] = frozenset(
    {"lapse", "missed", "miss", "omission", "timeout", "unresponsive", "error"}
)

#: Bodies at least this long are treated as sustained windows and are
#: compressed by ``duration_scale``; shorter ones are discrete stimuli and are
#: presented at real speed (a 100 ms ERP tone cannot be played faster).
HOLD_THRESHOLD_S = 2.0


@dataclass
class Trial:
    """One stimulus presentation, cued action, or sustained class window."""

    index: int
    block: int
    label: str
    stim: Stimulus | None
    onset_code: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase:
    """A contiguous run of the protocol: instructions, practice, block, rest."""

    kind: str                     # instruction | practice | block | rest | end
    label: str
    trials: list[Trial] = field(default_factory=list)
    duration_s: float | None = None


# ---------------------------------------------------------------- rng helpers
def _rand(rng: Any) -> float:
    """Uniform [0,1) for either ``numpy.random.Generator`` or ``random.Random``."""
    if rng is None:
        rng = np.random.default_rng()
    try:
        return float(rng.random())
    except Exception:
        return float(np.random.default_rng().random())


def _shuffle(rng: Any, seq: list[Any]) -> None:
    try:
        rng.shuffle(seq)                 # both Generator and random.Random
    except Exception:
        pass


def jitter(rng: Any, lo: float | None, hi: float | None) -> float:
    """Uniform extra delay in ``[lo, hi]`` seconds, robust to bad spec values.

    Missing/None/NaN bounds collapse to 0.0; a reversed pair is swapped; a
    scalar bound is treated as a half-open ``[0, bound]``.
    """
    try:
        a = float(lo) if lo is not None else 0.0
        b = float(hi) if hi is not None else a
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = a
    if b < a:
        a, b = b, a
    a = max(0.0, a)
    b = max(0.0, b)
    if b <= 0.0:
        return 0.0
    return float(a + (b - a) * _rand(rng))


def iti_bounds(design: dict[str, Any] | None) -> tuple[float, float]:
    """Read ``design.iti_jitter_s`` safely.  Missing -> ``(0.0, 0.0)``."""
    if not design:
        return (0.0, 0.0)
    raw = design.get("iti_jitter_s")
    if raw is None:
        return (0.0, 0.0)
    if isinstance(raw, (int, float)):
        return (0.0, max(0.0, float(raw)))
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return (0.0, 0.0)
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return (0.0, 0.0)
    return (max(0.0, min(vals)), max(0.0, max(vals)))


# ---------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _label_key(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def onset_code_for(spec: TestSpec, label: str) -> int:
    """Marker code for a class/condition label.

    The spec's own ``markers`` block always wins: two specs in a tier routinely
    reuse a code (1210 means something else in T201 and T203), so the global
    name->code table must never be consulted first.  Only names the spec does
    not define fall back to ``store.MARKER_CODES``.
    """
    key = _label_key(label)
    try:
        return int(spec.code(key))
    except Exception:
        pass
    return int(store.MARKER_CODES.get(key, 0))


def rare_class(spec: TestSpec, p_target: float | None) -> str | None:
    """Which class is the rare one, or None when the design is balanced.

    The catalog lists the frequent class first (T301 ``[standard, deviant]``,
    T101 ``[target, lapse]``), so the *last* class is the rare one.  Name
    heuristics were deliberately removed: "target" is the *frequent* class in
    SART, so a name-based guess would invert T101.
    """
    classes = list(spec.classes or [])
    if p_target is None or not (0.0 < float(p_target) < 0.5) or len(classes) < 2:
        return None
    return classes[-1]


def miss_class(classes: Sequence[str]) -> str | None:
    for c in classes:
        if _label_key(c) in MISS_CLASSES:
            return c
    return None


def _even_counts(k: int, n: int) -> list[int]:
    if k <= 0:
        return []
    base, rem = divmod(int(n), int(k))
    return [base + (1 if i < rem else 0) for i in range(k)]


def _balanced_labels(classes: Sequence[str], n: int, rng: Any) -> list[str]:
    """Interleave classes so that every contiguous block is roughly balanced.

    Greedy "pick from the class with the most remaining quota", with random
    tie-breaking, is a prefix-balanced shuffled round-robin: at any point the
    counts seen so far differ by at most one, and the order is not a
    predictable ABABAB train (another entrainment source).
    """
    if not classes:
        return []
    if len(classes) == 1:
        return [classes[0]] * n
    remaining = dict(zip(classes, _even_counts(len(classes), n)))
    out: list[str] = []
    while len(out) < n:
        m = max(remaining.values())
        pool = [c for c in classes if remaining[c] == m]
        pick = pool[min(len(pool) - 1, int(_rand(rng) * len(pool)))]
        out.append(pick)
        remaining[pick] -= 1
    return out


def _rare_labels(classes: Sequence[str], rare: str, n: int, p: float,
                 rng: Any) -> list[str]:
    """Rare class at probability ``p``, never twice in a row.

    Rare slots are chosen by random rejection with a minimum gap of two trials,
    then sorted, so rare stimuli are spread through the run rather than
    clustered -- clustering would let a slow drift masquerade as a class effect.
    """
    n_rare = int(round(max(0.0, min(0.5, p)) * n))
    chosen: list[int] = []
    order = list(range(n))
    _shuffle(rng, order)
    for pos in order:
        if len(chosen) >= n_rare:
            break
        if all(abs(pos - c) >= 2 for c in chosen):
            chosen.append(pos)
    chosen.sort()
    rare_set = set(chosen)
    others = [c for c in classes if c != rare]
    deck: list[str] = []
    for cls, cnt in zip(others, _even_counts(len(others), n - len(rare_set))):
        deck.extend([cls] * cnt)
    _shuffle(rng, deck)
    it = iter(deck)
    labels: list[str] = []
    for i in range(n):
        if i in rare_set:
            labels.append(rare)
        else:
            try:
                labels.append(next(it))
            except StopIteration:
                labels.append(others[0] if others else rare)
    return labels


def _block_labels(blocks: int, classes: Sequence[str], design: dict[str, Any],
                  rng: Any) -> list[str | None]:
    """Per-block class labels from ``labelled_blocks`` / ``block_order`` / ``run_labels``.

    ``labelled_blocks`` (T102, T105) deliberately leaves some blocks
    *unlabelled*: those windows are recorded but carry no class, so they are
    excluded from supervised analysis instead of being mislabelled.
    """
    lb = design.get("labelled_blocks")
    if isinstance(lb, dict) and lb:
        out: list[str | None] = [None] * blocks
        for cls, ids in lb.items():
            if not isinstance(ids, (list, tuple)):
                continue
            for i in ids:
                try:
                    idx = int(i) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < blocks:
                    out[idx] = str(cls)
        if any(x is not None for x in out) or out:
            return out
    bo = design.get("block_order")
    if isinstance(bo, str) and bo.strip():
        mapping: dict[str, str] = {}
        seq: list[str] = []
        for ch in bo.strip():
            if ch.isspace():
                continue
            if ch not in mapping:
                if not classes:
                    break
                mapping[ch] = classes[len(mapping) % len(classes)]
            seq.append(mapping[ch])
        if seq:
            return [seq[i % len(seq)] for i in range(blocks)]
    rl = design.get("run_labels")
    if isinstance(rl, (list, tuple)) and rl:
        labs = [str(x) for x in rl]
        return [labs[i % len(labs)] for i in range(blocks)]
    if not classes:
        return [None] * blocks
    if len(classes) == 1:
        return [classes[0]] * blocks
    # deterministic interleave so consecutive blocks never repeat a class
    return [classes[i % len(classes)] for i in range(blocks)]


def _label_deck(classes: Sequence[str], n: int, design: dict[str, Any],
                rng: Any, rare: str | None = None) -> list[str]:
    """Class schedule: rare-class schedule when ``p_target`` names one."""
    p_raw = design.get("p_target")
    try:
        p = float(p_raw) if p_raw not in (None, "") else None
    except (TypeError, ValueError):
        p = None
    if rare is not None and p is not None and 0.0 < p < 0.5:
        return _rare_labels(classes, rare, n, p, rng)
    return _balanced_labels(classes, n, rng)


# ---------------------------------------------------------------- planning
def _first_window(design: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in WINDOW_KEYS:
        v = _num(design.get(key))
        if v is not None:
            return key, v
    return None, None


def _itv(body: float, base: float, design: dict[str, Any], rng: Any,
         soa_present: bool) -> float:
    """Onset-to-onset interval in nominal seconds."""
    lo, hi = iti_bounds(design)
    if hi <= 0.0:
        extra = 0.0
    elif soa_present and hi >= base:
        # The band *is* the jittered SOA (T004, T301, T101, T108, T007, ...).
        return max(body, jitter(rng, lo, hi))
    else:
        # The band is extra jitter after the body (T401, T701, T704, ...).
        extra = jitter(rng, lo, hi)
    return max(body, base) + extra


def resolve_design(spec: TestSpec) -> dict[str, Any]:
    """Classify a spec's design block and name the keys it relies on.

    Returns a small dict with ``shape``, ``keys`` (consumed design keys),
    ``blocks``, ``soa``, ``window`` and ``mode``.  Pure inspection: no RNG, no
    IO, so it can be logged in session meta and printed for the operator.
    """
    d = dict(spec.design or {})
    classes = list(spec.classes or [])
    stim = spec.stimuli or {}
    consumed: list[str] = []
    blocks = max(1, int(d.get("blocks") or 1))
    soa = _num(d.get("soa_s"))
    if soa is not None:
        consumed.append("soa_s")
    if d.get("iti_jitter_s") is not None:
        consumed.append("iti_jitter_s")
    if d.get("p_target") is not None:
        consumed.append("p_target")
    if d.get("instruction") is not None:
        consumed.append("instruction")
    if d.get("rest_between_blocks_s") is not None:
        consumed.append("rest_between_blocks_s")
    if d.get("break_duration_s") is not None:
        consumed.append("break_duration_s")
    if d.get("mid_session_break_s") is not None:
        consumed.append("mid_session_break_s")
    for key in ("feedback_update_s", "probe_every_s"):
        if _num(d.get(key)) is not None:
            consumed.append(key)

    shape = "trial"
    window_key, window = _first_window(d)
    if d.get("phases_per_block") is not None:
        shape = "phase"
        consumed.append("phases_per_block")
    elif d.get("levels_per_block") is not None:
        shape = "level"
        consumed.append("levels_per_block")
        for k in ("repeats_per_level", "level_order", "level_duration_s"):
            if d.get(k) is not None:
                consumed.append(k)
    elif isinstance(d.get("conditions_per_block"), list):
        shape = "condition-cue"
        consumed.append("conditions_per_block")
    elif d.get("conditions_per_block") is not None or window_key in (
            "condition_duration_s", "run_s"):
        shape = "condition"
        if d.get("conditions_per_block") is not None:
            consumed.append("conditions_per_block")
    elif d.get("trials_per_block") is not None:
        shape = "trial"
        consumed.append("trials_per_block")
    elif d.get("passages") is not None:
        shape = "passage"
        consumed.append("passages")
    elif d.get("block_duration_s") is not None or isinstance(d.get("block_durations_s"), list):
        shape = "block"
        if d.get("block_duration_s") is not None:
            consumed.append("block_duration_s")
        if isinstance(d.get("block_durations_s"), list):
            consumed.append("block_durations_s")
    elif soa is not None:
        shape = "trial"
    elif d.get("trials") is not None:
        shape = "block"
    if window_key is not None and window_key not in consumed:
        consumed.append(window_key)
    for k in ("labelled_blocks", "block_order", "run_labels", "runs_required",
              "marker_seed", "practice", "practice_trials"):
        if d.get(k) is not None:
            consumed.append(k)
    if spec.needs_stimuli and stim.get("generator"):
        consumed.append("generator")
    return {
        "shape": shape,
        "keys": list(dict.fromkeys(consumed)),
        "blocks": blocks,
        "soa_s": soa,
        "window_s": window,
        "window_key": window_key,
        "classes": classes,
        "has_stimuli": bool(spec.needs_stimuli),
        "generator": stim.get("generator"),
        "kind": stim.get("kind"),
        "stim_duration_s": _num(stim.get("duration_s")),
        "p_target": _num(d.get("p_target")),
        "miss_class": miss_class(classes),
    }


def plan_summary(spec: TestSpec, trials: Sequence[Trial] | None = None,
                 phases: Sequence[Phase] | None = None) -> dict[str, Any]:
    """Human-readable plan for session meta / the printed summary."""
    info = resolve_design(spec)
    out: dict[str, Any] = dict(info)
    if trials is not None:
        bodies = [float(t.meta.get("body_s") or 0.0) for t in trials]
        itvs = [float(t.meta.get("itv_s") or 0.0) for t in trials]
        out["n_trials"] = len(trials)
        out["body_s"] = round(float(np.mean(bodies)), 4) if bodies else 0.0
        out["itv_s"] = round(float(np.mean(itvs)), 4) if itvs else 0.0
        out["planned_s"] = round(float(np.sum(itvs)), 2)
        labels: dict[str, int] = {}
        for t in trials:
            labels[t.label or "(unlabelled)"] = labels.get(t.label or "(unlabelled)", 0) + 1
        out["label_counts"] = labels
        out["modes"] = sorted({str(t.meta.get("mode")) for t in trials})
    if phases is not None:
        out["rest_s"] = round(sum(float(p.duration_s or 0.0)
                                  for p in phases if p.kind == "rest"), 2)
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------- build
def _stim_pools(spec: TestSpec, stimset: StimulusSet | None, rng: Any) -> dict[str, list[Stimulus]]:
    pools: dict[str, list[Stimulus]] = {}
    if stimset is None:
        return pools
    by_label = stimset.by_label()
    for label, items in by_label.items():
        pool = [s for s in items if s.exists()]
        _shuffle(rng, pool)
        pools[label] = pool
    return pools


def _segments(spec: TestSpec, stimset: StimulusSet | None, rng: Any
              ) -> Iterator[tuple[int, str, float, float, str, Stimulus | None, dict[str, Any]]]:
    """Yield ``(block, label, body, itv, mode, stim, meta)`` for every trial."""
    d = dict(spec.design or {})
    info = resolve_design(spec)
    classes = list(spec.classes or [])
    blocks = int(info["blocks"])
    soa = info["soa_s"]
    kind = info["kind"]
    stim_dur = info["stim_duration_s"]
    if stim_dur is None:
        stim_dur = float(DEFAULT_DURATION_S.get(str(kind), 0.5))
    window = info["window_s"]
    p_target = info["p_target"]
    miss = info["miss_class"]
    response_window = _num(d.get("response_window_s"))
    prompt_s = _num(d.get("prompt_s")) or 0.0
    feedback_s = _num(d.get("feedback_s")) or 0.0
    inter_phase = _num(d.get("inter_phase_cue_s")) or 0.0
    mid_break = _num(d.get("mid_session_break_s"))

    pools = _stim_pools(spec, stimset, rng)
    ptr: dict[str, int] = {}

    def take(label: str) -> Stimulus | None:
        pool = pools.get(label) or []
        if not pool:
            return None
        k = ptr.get(label, 0)
        ptr[label] = k + 1
        s = pool[k % len(pool)]
        if (k + 1) % len(pool) == 0:
            _shuffle(rng, pool)
        return s

    def emit(block: int, label: str, body: float, base: float, mode: str,
             **meta: Any):
        itv = _itv(max(0.0, body), max(0.0, base), d, rng, soa is not None)
        stim = take(label) if label else None
        return (int(block), str(label), float(max(0.0, body)), float(itv),
                str(mode), stim, dict(meta))

    # -- shape: alternating phases inside each block (T003) -----------
    if info["shape"] == "phase":
        n_ph = max(1, int(d.get("phases_per_block") or len(classes) or 1))
        dur = _num(d.get("phase_duration_s")) or window or 20.0
        for b in range(blocks):
            for j in range(n_ph):
                lab = classes[(b * n_ph + j) % len(classes)] if classes else ""
                yield emit(b, lab, dur, dur + inter_phase, "hold",
                           phase=j + 1, of=n_ph, cue_s=inter_phase,
                           instruction=d.get("instruction"))
        return

    # -- shape: loudness/level ladder (T607) --------------------------
    if info["shape"] == "level":
        levels = max(1, int(d.get("levels_per_block") or 1))
        reps = max(1, int(d.get("repeats_per_level") or 1))
        dur = _num(d.get("level_duration_s")) or window or 30.0
        l2c = d.get("level_to_class")
        dbfs = d.get("levels_dbfs")
        for b in range(blocks):
            order = [lv for lv in range(1, levels + 1) for _ in range(reps)]
            _shuffle(rng, order)
            for lv in order:
                lab = None
                if isinstance(l2c, dict):
                    lab = l2c.get(lv) or l2c.get(str(lv))
                if lab is None and classes:
                    # Levels split into equal class bands in catalog order:
                    # "levels 1-2 -> quiet, levels 3-4 -> loud" (T607).
                    idx = min(len(classes) - 1, (lv - 1) * len(classes) // levels)
                    lab = classes[idx]
                meta: dict[str, Any] = {"level": lv}
                if isinstance(dbfs, (list, tuple)) and 0 < lv <= len(dbfs):
                    meta["level_db"] = _num(dbfs[lv - 1])
                yield emit(b, str(lab or ""), dur, dur, "hold", **meta)
        return

    # -- shape: cued action conditions inside each block (T007) -------
    if info["shape"] == "condition-cue":
        conds = [str(x) for x in d.get("conditions_per_block") or []]
        tpb = int(d.get("trials_per_block") or d.get("trials") or 0)
        per = max(1, int(round(tpb / max(1, len(conds))))) if tpb else 1
        body = response_window or 0.0
        base = body if body > 0 else (soa or 0.0)
        for b in range(blocks):
            for _ in range(per):
                for name in conds:
                    yield emit(b, name, body, base, "cued", cue=name)
        return

    # -- shape: one window per block (T002, T102, T104, T106, T107) ---
    if info["shape"] == "block":
        blabels = _block_labels(blocks, classes, d, rng)
        durs = d.get("block_durations_s")
        fallback = _num(d.get("block_duration_s")) or max(
            60.0, (float(spec.duration_min or 0) * 60.0) / max(1, blocks))
        for b in range(blocks):
            dur = window if window is not None else fallback
            if isinstance(durs, (list, tuple)) and b < len(durs):
                dur = _num(durs[b]) or dur
            lab = blabels[b] if b < len(blabels) else None
            mode = "seating" if _num(d.get("attempt_duration_s")) is not None else "hold"
            yield emit(b, str(lab or ""), float(dur), float(dur), mode,
                       block_label=lab, mid_break_s=mid_break)
        return

    # -- shape: passages as timed reading windows (T501) --------------
    if info["shape"] == "passage":
        n_pass = max(1, int(d.get("passages") or 1))
        per_block = max(1, int(round(n_pass / blocks)))
        dur = _num(d.get("passage_duration_s")) or window or 45.0
        lab = classes[0] if classes else ""
        for b in range(blocks):
            for j in range(per_block):
                yield emit(b, lab, dur, dur, "hold", passage=j + 1, cue=d.get("instruction"))
        return

    # -- shape: condition windows / runs (T201-T208, T601-T608, T707/8) --
    if info["shape"] == "condition":
        conds_raw = d.get("conditions_per_block")
        conds = ([str(x) for x in conds_raw] if isinstance(conds_raw, (list, tuple))
                 else list(classes))
        cond_dur = window or _num(d.get("block_duration_s")) or max(
            30.0, float(spec.duration_min or 0) * 60.0 / max(1, blocks * len(conds)))
        per_cond = max(1, int(round(cond_dur / soa))) if soa else 1
        tpb = d.get("trials_per_block")
        for b in range(blocks):
            if tpb is not None:
                # T707: a fixed number of windows per block, cycling conditions.
                k = max(1, int(tpb))
                seq = [conds[j % len(conds)] for j in range(k)]
            else:
                seq = [c for c in conds for _ in range(per_cond)]
            for name in seq:
                if soa is not None:
                    body = min(stim_dur, soa) if stim_dur else 0.0
                    mode = "stim" if body < HOLD_THRESHOLD_S else "hold"
                    yield emit(b, name, body, soa, mode,
                               condition=name, condition_duration_s=cond_dur)
                else:
                    yield emit(b, name, cond_dur, cond_dur, "hold", condition=name)
        return

    # -- shape: plain per-trial (soa_s / trials_per_block / trials) ---
    tpb = d.get("trials_per_block")
    if tpb is not None:
        n_total = int(tpb) * blocks
        per_block = int(tpb)
    elif d.get("trials") is not None:
        n_total = int(d.get("trials") or 0)
        per_block = None
    else:
        n_total = blocks
        per_block = 1

    # Per-trial window: response window (cued actions) beats trial_s beats
    # the modality default, because it is the part the subject actually does.
    if response_window is not None:
        body = response_window + prompt_s + feedback_s
        base = body
    elif window is not None:
        body = window
        base = window
    elif soa is not None:
        body = min(stim_dur, soa) if stim_dur else 0.0
        base = soa
    else:
        body = stim_dur
        base = stim_dur

    blabels = _block_labels(blocks, classes, d, rng) if (
        d.get("labelled_blocks") or d.get("run_labels") or d.get("block_order")) else None
    conds_raw = d.get("conditions_per_block")
    rare = rare_class(spec, p_target)

    # Build the label schedule block by block.
    schedule: list[tuple[int, str]] = []
    if conds_raw is not None and isinstance(conds_raw, (int, float)):
        conds = list(classes)
        k = max(1, int(conds_raw))
        for b in range(blocks):
            for j in range(k):
                schedule.append((b, conds[j % len(conds)]))
    elif per_block is not None:
        for b in range(blocks):
            if blabels is not None and b < len(blabels):
                lab = blabels[b]
                for _ in range(per_block):
                    schedule.append((b, str(lab) if lab else ""))
            else:
                deck = _label_deck(classes, per_block, d, rng, rare)
                for lab in deck:
                    schedule.append((b, lab))
    else:
        deck = _label_deck(classes, n_total, d, rng, rare)
        for i, lab in enumerate(deck):
            b = min(blocks - 1, (i * blocks) // max(1, n_total))
            schedule.append((b, lab))

    for block, lab in schedule:
        mode = "hold" if body >= HOLD_THRESHOLD_S else "stim"
        if lab and miss is not None and response_window is not None:
            # T101/T108: the class is only known after the response window, so
            # the runner back-dates the class marker to the probe timestamp.
            mode = "gated"
        yield emit(block, lab, body, base, mode)


def build_trials(spec: TestSpec, stimset: StimulusSet | None = None,
                 rng: Any = None) -> list[Trial]:
    """Turn a spec's ``design`` into a concrete, seeded trial list.

    See the module docstring for the timing conventions; the important
    invariants are:

    * ``onset_code`` comes from ``spec.markers[label]`` (never the global
      table), so two specs can reuse a code without ambiguity;
    * each trial carries ``body_s`` / ``itv_s`` / ``mode`` in ``meta`` --
      the runner's whole timing model;
    * ``source: none`` tests produce trials with ``stim=None`` but still a
      label and an onset marker: they are timed, class-marked phases, and the
      runner emits ``stim_onset`` for them (sustained-state analysis needs it).
    """
    rng = rng if rng is not None else np.random.default_rng()
    trials: list[Trial] = []
    for i, (block, label, body, itv, mode, stim, meta) in enumerate(
            _segments(spec, stimset, rng)):
        trials.append(Trial(
            index=i, block=block, label=label, stim=stim,
            onset_code=onset_code_for(spec, label) if label else 0,
            meta={**meta, "body_s": round(body, 6), "itv_s": round(itv, 6),
                  "mode": mode, "class": label or None},
        ))
    return trials


def make_phases(spec: TestSpec, trials: Sequence[Trial],
                include_practice: bool = True) -> list[Phase]:
    """Split trials into instruction / practice / block / rest / end phases.

    Breaks are real phases, not gaps: the runner marks ``break_start`` and
    ``break_end`` around them and keeps acquiring EEG throughout, so drift and
    drowsiness between blocks remain in the record.  A ``mid_session_break_s``
    (T502) is inserted after the middle block.
    """
    design = dict(spec.design or {})
    blocks = max(1, int(design.get("blocks") or 1))
    rest_s = _num(design.get("rest_between_blocks_s")) or 0.0
    if rest_s <= 0.0:
        rest_s = _num(design.get("break_duration_s")) or 0.0
    mid_break = _num(design.get("mid_session_break_s")) or 0.0
    phases: list[Phase] = [
        Phase(kind="instruction", label="instructions",
              duration_s=_num(design.get("instruction_s"))),
    ]

    by_block: dict[int, list[Trial]] = {}
    for t in trials:
        by_block.setdefault(int(t.block), []).append(t)

    if include_practice and design.get("practice"):
        n_practice = int(design.get("practice_trials", 4) or 4)
        head = list(trials)[:max(0, n_practice)]
        practice = [
            replace(t, index=-(i + 1), block=-1, meta={**t.meta, "practice": True})
            for i, t in enumerate(head)
        ]
        if practice:
            phases.append(Phase(kind="practice", label="practice", trials=practice))

    for k, b in enumerate(range(blocks)):
        ts = by_block.get(b, [])
        if not ts:
            continue
        phases.append(Phase(kind="block", label=f"block {b + 1}/{blocks}", trials=ts))
        is_last = (k == len([bb for bb in range(blocks) if by_block.get(bb)]) - 1)
        if not is_last:
            if mid_break > 0 and (b + 1) == (blocks // 2):
                phases.append(Phase(kind="rest", label="mid-session break",
                                    duration_s=mid_break))
            elif rest_s > 0:
                phases.append(Phase(kind="rest", label=f"break after block {b + 1}",
                                    duration_s=rest_s))
    phases.append(Phase(kind="end", label="finished"))
    return phases


def summarise(phases: Sequence[Phase]) -> dict[str, Any]:
    """Small dict for session meta: what was planned, in human units."""
    n_trials = sum(len(p.trials) for p in phases)
    return {
        "n_trials": n_trials,
        "blocks": sum(1 for p in phases if p.kind == "block"),
        "practice_trials": sum(len(p.trials) for p in phases if p.kind == "practice"),
        "rest_s": sum(float(p.duration_s or 0.0) for p in phases if p.kind == "rest"),
        "phases": [{"kind": p.kind, "label": p.label,
                    "trials": len(p.trials), "duration_s": p.duration_s}
                   for p in phases],
    }
