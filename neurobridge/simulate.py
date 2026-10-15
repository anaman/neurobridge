"""Offline synthetic sessions: the end-to-end self-test of the whole system.

``nb simulate T301`` builds a realistic session directory from the test spec
using the synthetic EEG source, then runs the ordinary analysis pipeline over
it. It exists to answer one question honestly: *if the brain produced the
effect this test is looking for, would this code find it?*

Two properties make it worth trusting:

* it does not use the live runner at all -- the timeline here is built
  independently from the spec, so a bug shared between the runner and the
  analysis cannot hide;
* ``--effect-size 0`` removes the injected response entirely, so the same code
  path also measures the pipeline's false-positive floor.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import store
from .config import ensure_dirs
from .device.fake import FakeSource

#: Specs whose design is a stream of single trials rather than alternating
#: conditions. They declare `trials` plus a trial period, which is called
#: `soa_s` in the ERP specs and `trial_s` in the control/switch specs.
TRIAL_KEYS = ("trials",)
PERIOD_KEYS = ("soa_s", "trial_s")

#: Keys that give the length of one sustained condition, most specific first.
#: The catalog really does use all of these (each workstream picked its own).
#: Presence of any of them means the design is *condition-run based*, even when
#: it also declares `soa_s` for the stimuli inside a run: T201 declares both
#: `soa_s: 1.2` (one letter per trial) and `run_s: 60` (one condition lasts a
#: minute). Treating that as a 1.2 s trial alternation made every 4 s analysis
#: window straddle both conditions and averaged the injected effect to nothing.
CONDITION_DURATION_KEYS = (
    "condition_duration_s", "run_s", "phase_duration_s", "level_duration_s",
    "passage_duration_s", "task_window_s", "imagery_window_s",
)

#: Keys that say how many conditions fit in one block.
CONDITIONS_PER_BLOCK_KEYS = (
    "conditions_per_block", "phases_per_block", "levels_per_block",
)


def has_condition_runs(spec) -> bool:
    """True when the design runs each condition for a sustained stretch."""
    d = spec.design or {}
    return any(isinstance(d.get(k), (int, float)) and float(d[k]) > 0
               for k in CONDITION_DURATION_KEYS)


def is_trial_based(spec) -> bool:
    """True when the design is a sequence of discrete trials (ERP/response).

    A declared condition-run duration wins over a declared SOA: several
    sustained-state specs carry both, and the run length is what decides how the
    analysis windows must be cut.
    """
    if has_condition_runs(spec):
        return False
    d = spec.design or {}
    return ("trials" in d and any(float(d.get(k) or 0) > 0 for k in PERIOD_KEYS))


def is_sustained(spec) -> bool:
    """True when the design alternates sustained conditions/phases."""
    return not is_trial_based(spec)


def condition_seconds(spec, blocks: int) -> float:
    """Length of one condition, from whichever timing key this spec happens to use."""
    d = spec.design or {}
    for key in CONDITION_DURATION_KEYS:
        v = d.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    cpb = conditions_per_block(spec)
    bd = d.get("block_duration_s")
    if isinstance(bd, (int, float)) and bd > 0:
        return float(bd) / cpb
    bds = d.get("block_durations_s")
    if isinstance(bds, list) and bds:
        return float(sum(float(x) for x in bds)) / len(bds) / cpb
    # last resort: spread the declared test duration evenly
    total = float(d.get("duration_min", 8) or 8) * 60.0
    return max(20.0, total / max(1, blocks) / cpb)


def conditions_per_block(spec) -> float:
    d = spec.design or {}
    for key in CONDITIONS_PER_BLOCK_KEYS:
        v = d.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return float(max(2, len(spec.classes)))


def build_offline_timeline(spec, seed: int = 20261013, scale: float = 1.0,
                           max_trials: int | None = None
                           ) -> tuple[list[dict[str, Any]], float]:
    """Build the marker timeline for a synthetic run of ``spec``.

    Returns ``(events, duration_s)``. Each event is a dict with
    ``t`` (seconds from session start), ``name``, ``code`` (resolved through
    :meth:`TestSpec.code`, so the spec's own marker table always wins), plus
    ``block``, ``trial``, ``class``, ``phase`` and ``meta``.

    The class marker is emitted immediately *before* ``stim_onset`` at the same
    timestamp: the analysis labels an epoch or a window from the most recent
    class marker, and the tiers differ in which anchor they rely on.
    """
    rng = random.Random(seed)
    classes = list(spec.classes) or ["rest"]
    blocks = max(1, int((spec.design or {}).get("blocks", 1) or 1))
    events: list[dict[str, Any]] = []

    def mark(t: float, name: str, block: int = 0, trial: int | None = None,
             label: str | None = None, phase: str = "trial",
             meta: dict[str, Any] | None = None) -> None:
        events.append({
            "t": round(max(0.0, float(t)), 4),
            "name": name,
            "code": spec.code(name, 0),
            "block": int(block),
            "trial": trial,
            "class": label,
            "phase": phase,
            "meta": dict(meta or {}),
        })

    mark(0.0, "session_start", phase="session")
    mark(0.3 * scale, "test_start", phase="test")
    t = 1.5 * scale                                    # settle before the first trial

    if is_trial_based(spec):
        trials = max(1, int(spec.design.get("trials") or 1))
        if max_trials:      # verification only: cap the trial count so a
            trials = min(trials, int(max_trials))   # 1600-trial MMN spec is affordable
        # The trial period is `soa_s` in the ERP specs and `trial_s` in the
        # switch/control specs; both may be absent in favour of an explicit
        # inter-trial interval. Jitter is *added* to the period, because the
        # designs use it to break phase-locking with ongoing alpha.
        jit = spec.design.get("iti_jitter_s") or [0.0, 0.0]
        j_lo, j_hi = float(jit[0]), float(jit[1])
        mean_jitter = (j_lo + j_hi) / 2.0 if j_hi > j_lo else 0.0
        base = next((float(spec.design[k]) for k in PERIOD_KEYS
                     if isinstance(spec.design.get(k), (int, float))
                     and float(spec.design[k]) > 0), 0.0)
        stim_dur = float(spec.stimuli.get("duration_s") or 0.0)
        if base <= 0:
            base = max(stim_dur, 0.3) + (mean_jitter or 2.5)
        period = base + mean_jitter
        resp_win = float(spec.design.get("response_window_s") or 0.0)
        p_target = float(spec.design.get("p_target", 0.0) or 0.0)
        stim_dur = stim_dur or min(0.5, period * 0.5)
        if stim_dur >= period:
            stim_dur = period * 0.5
        per_block = max(1, trials // blocks)
        for bi in range(blocks):
            mark(t, "block_start", bi, phase="block")
            t += 0.5 * scale
            for k in range(per_block):
                trial = bi * per_block + k
                if spec.design.get("random_labels"):
                    # null-floor designs (T008) must not have a predictable
                    # label sequence, or a decoder could learn the pattern
                    label = classes[rng.randrange(len(classes))]
                elif p_target and len(classes) >= 2:
                    label = classes[0] if rng.random() < p_target else classes[1]
                else:
                    label = classes[k % len(classes)]
                mark(t, label, bi, trial, label)                  # class marker FIRST
                mark(t, "stim_onset", bi, trial, label)
                t += stim_dur * scale
                mark(t, "stim_offset", bi, trial, label)
                if resp_win:                                      # response window
                    t += min(resp_win, max(0.0, period - stim_dur)) * scale
                step = period + (rng.uniform(j_lo, j_hi) - mean_jitter if j_hi > j_lo else 0.0)
                t += max(0.05, step - stim_dur - (min(resp_win, period) if resp_win else 0.0)) * scale
            mark(t, "block_end", bi, phase="block")
            t += 2.0 * scale
    else:
        cond_s = condition_seconds(spec, blocks)
        n_cond = int(conditions_per_block(spec))
        run_labels = spec.design.get("run_labels")
        for bi in range(blocks):
            mark(t, "block_start", bi, phase="block")
            t += 0.5 * scale
            for ci in range(n_cond):
                if isinstance(run_labels, list) and run_labels:
                    label = str(run_labels[ci % len(run_labels)])
                else:
                    label = classes[ci % len(classes)]
                mark(t, label, bi, label=label, phase="condition")
                mark(t, "stim_onset", bi, label=label)
                seg_end = t + cond_s * scale
                # refresh the class marker at most every 20 s: nothing in the
                # analysis should depend on a single marker surviving a drop
                while t + 20.0 * scale < seg_end:
                    t += 20.0 * scale
                    mark(t, label, bi, label=label, phase="condition")
                t = seg_end
            mark(t, "block_end", bi, phase="block")
            t += 2.0 * scale

    mark(t, "test_end", phase="test")
    mark(t + 0.3 * scale, "session_end", phase="session")
    return events, t + 0.5 * scale


def simulate_test(spec, effect_size: float = 1.0, seed: int = 20261013,
                  scale: float = 1.0, session_id: str | None = None,
                  seed_note: str | None = None, fs: float = 250.0,
                  max_trials: int | None = None) -> Path:
    """Generate one synthetic session for ``spec`` and write it to disk.

    ``scale`` multiplies every duration (a fast smoke test); it is recorded in
    the session metadata so a scaled run can never be mistaken for real data.
    """
    ensure_dirs()
    events, duration = build_offline_timeline(spec, seed=seed, scale=scale,
                                               max_trials=max_trials)

    # The synthetic source needs the class markers to know what to inject.
    src = FakeSource(fs=fs, seed=seed, effect_size=effect_size, realtime=False)
    markers = [(e["t"], e["name"]) for e in events]
    values, times, _ = src.generate(duration, markers)

    sid = session_id or f"sim-{spec.id}-{time.strftime('%Y%m%d-%H%M%S')}"
    writer = store.SessionWriter(session_id=sid, meta={
        "session_id": sid,
        "device": "fake",
        "device_info": src.info,
        "fs": fs,
        "channel_name": "Fpz",
        "montage": "synthetic single frontal channel",
        "synthetic": True,
        "effect_size": effect_size,
        "simulation_seed": seed,
        "duration_scale": scale,
        "note": seed_note or ("null-model run: no stimulus response was injected"
                              if effect_size == 0 else
                              "synthetic session generated by nb simulate"),
        "spec_id": spec.id,
        "spec_title": spec.title,
        "feasibility": spec.feasibility,
        "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    writer.event(store.MARKER_CODES["session_start"], name="session_start",
                 test_id=spec.id)
    writer.add_samples(values, times)
    t0 = float(times[0]) if times.size else 0.0
    for e in events:
        if e["name"] == "session_start":
            continue
        writer.event(e["code"], name=e["name"], test_id=spec.id, block=e.get("block"),
                     trial=e.get("trial"), t_mono=t0 + e["t"], phase=e.get("phase"),
                     meta=e.get("meta") or ({"class": e["class"]} if e.get("class") else {}))
    path = writer.close(summary={
        "synthetic": True,
        "effect_size": effect_size,
        "duration_scale": scale,
        "quality_verdict": "SYNTHETIC",
    })
    (path / "simulation.json").write_text(json.dumps({
        "spec_id": spec.id, "effect_size": effect_size, "seed": seed,
        "duration_scale": scale, "n_events": len(events), "duration_s": duration,
        "design_shape": "trial" if is_trial_based(spec) else "condition",
        "condition_seconds": None if is_trial_based(spec)
        else round(condition_seconds(spec, max(1, int(spec.design.get("blocks", 1) or 1))), 2),
    }, indent=2))
    return path


def estimate_duration_min(spec) -> tuple[float, float, float]:
    """(declared, mechanical, planning) minutes for one run of ``spec``.

    The two numbers disagree often, and for good reasons in both directions:

    * ``declared`` (``design.duration_min``) is the author's estimate and
      includes instructions, practice and setup that no formula knows about;
    * ``mechanical`` comes from actually building the timeline out of the
      spec's own trials/SOA/jitter/blocks/conditions.

    For planning we take the **larger** of the two: under-booking a 45 minute
    sitting is the failure mode that costs a session, and over-booking only
    costs time. The disagreement itself is surfaced by ``nb catalog audit``.
    """
    _, duration_s = build_offline_timeline(spec, scale=1.0)
    mechanical = duration_s / 60.0
    declared = float(spec.design.get("duration_min", 0) or 0)
    return declared, mechanical, max(declared, mechanical)


# ---------------------------------------------------------------- CLI helper
def _main(argv: list[str] | None = None) -> int:
    import argparse
    from .catalog import load_one
    from .pipeline import evaluate as E

    p = argparse.ArgumentParser(prog="python -m neurobridge.simulate",
                                description="generate a synthetic session and evaluate it")
    p.add_argument("id")
    p.add_argument("--effect-size", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args(argv)

    spec = load_one(a.id)
    path = simulate_test(spec, effect_size=a.effect_size, seed=a.seed, scale=a.scale)
    print(f"session: {path}")
    session = store.Session.load(path)
    res = E.evaluate_session(session, spec)
    cv = res.get("cv", {})
    print(f"balanced accuracy {cv.get('balanced_accuracy')} (chance {cv.get('chance')}) "
          f"auc {cv.get('auc')} n={cv.get('n')}")
    print("gates:", E.apply_gates(spec, res))
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(_main())
