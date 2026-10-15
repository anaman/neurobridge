"""Figures and Markdown reports.

Two rules shape this module:

1. **Nothing happens at import time.** matplotlib is imported inside a private
   helper, the Agg backend is selected there, and ``MPLCONFIGDIR`` is pointed at
   ``data/cache/matplotlib`` (created on first use) so importing the pipeline can
   never attempt to write into the user's home config directory. This machine
   actually has an unwritable ``~/.config/matplotlib``, so getting it wrong is
   not hypothetical.
2. **A failed gate is a normal outcome and the report says so.** The Markdown
   report leads with what was measured, compares it with chance *and* with the
   majority-class baseline, prints coverage/abstention, states the timing and
   quality caveats, and then gives a plain-language verdict: if the gate was not
   met, the most likely reason given this montage is named. No hype, no
   "breakthrough", no language model in the participant's voice.

Timing honesty: when ``timing_quality`` is not ``verified``, the report must not
present ERP *latency* differences as findings. :func:`_latency_allowed` gates
every latency mention, and the caveat is printed prominently instead.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import REPORT_DIR, DATA_DIR
from .epochs import Epochs
from .evaluate import apply_gates, false_activations_per_min, timing_quality_of
from .features import BANDS, band_powers

__all__ = [
    "plot_erp",
    "plot_bandpower",
    "plot_confusion",
    "plot_quality",
    "plot_timeline",
    "write_test_report",
    "write_session_report",
    "write_curriculum_report",
]

#: Matplotlib cache lives under the project data dir, never under $HOME.
MPL_DIR = DATA_DIR / "cache" / "matplotlib"
FIG_DIR = REPORT_DIR / "figures"

#: The 64 tests as a fallback table, used only when the catalog cannot be
#: loaded. Keeps `write_curriculum_report` a 64-row document even on a checkout
#: where the YAML files have not been generated yet.
_FALLBACK_TESTS: tuple[tuple[str, str, str], ...] = (
    ("T001", "Electrode seating & self-check", "established"),
    ("T002", "Resting baseline, eyes open", "established"),
    ("T003", "Alpha reactivity (eyes open vs closed)", "established"),
    ("T004", "Auditory latency calibration (click train)", "established"),
    ("T005", "Visual latency calibration (flash train)", "promising"),
    ("T006", "Mains, drift & saturation characterisation", "established"),
    ("T007", "Blink & jaw-clench artifact library", "established"),
    ("T008", "Sham marker control (null floor)", "control"),
    ("T101", "Sustained attention (SART)", "promising"),
    ("T102", "Drowsiness induction & detection", "promising"),
    ("T103", "Alpha entrainment to flicker", "exploratory"),
    ("T104", "Arousal from audio (calm vs exciting)", "promising"),
    ("T105", "Vigilance decrement over time-on-task", "promising"),
    ("T106", "Pre/post state comparison", "exploratory"),
    ("T107", "Artifact census during rest", "established"),
    ("T108", "Microsleep probe under eye closure", "exploratory"),
    ("T201", "N-back 0 vs 2 (working memory load)", "promising"),
    ("T202", "Mental arithmetic vs counting", "promising"),
    ("T203", "Stroop colour-word interference", "promising"),
    ("T204", "Visual search, easy vs hard", "promising"),
    ("T205", "Comprehension vs mindless reading", "promising"),
    ("T206", "Speech in noise (SNR ladder)", "promising"),
    ("T207", "Dual-task load", "exploratory"),
    ("T208", "Focused attention vs mind-wandering", "exploratory"),
    ("T301", "Auditory oddball (P300)", "established"),
    ("T302", "Environmental-sound oddball", "promising"),
    ("T303", "MMN, duration deviant", "promising"),
    ("T304", "MMN, passive frequency deviant", "established"),
    ("T305", "Speech-syllable oddball (/ba/ vs /da/)", "exploratory"),
    ("T306", "Roving-standard oddball", "promising"),
    ("T307", "Own name vs other name", "exploratory"),
    ("T308", "Auditory stream segregation", "exploratory"),
    ("T401", "Visual oddball with Openverse images", "promising"),
    ("T402", "RSVP natural vs man-made", "promising"),
    ("T403", "Face vs non-face (N170 surrogate)", "exploratory"),
    ("T404", "N400 sentence congruency", "promising"),
    ("T405", "N400 picture-word match", "exploratory"),
    ("T406", "Lexical decision (word vs pseudoword)", "promising"),
    ("T407", "Alpha blocking by image luminance", "promising"),
    ("T408", "Affective image valence", "exploratory"),
    ("T501", "Self-paced silent reading", "promising"),
    ("T502", "Reading aloud vs silent", "promising"),
    ("T503", "Word-frequency effect", "promising"),
    ("T504", "Garden-path vs simple sentences", "exploratory"),
    ("T505", "Coherent prose vs shuffled words", "promising"),
    ("T506", "Emotional text passages", "exploratory"),
    ("T507", "Readability difficulty ladder", "promising"),
    ("T508", "Inner speech vs fixation", "exploratory"),
    ("T601", "Music vs speech vs noise", "promising"),
    ("T602", "Beat/tempo tracking (60 vs 120 BPM)", "exploratory"),
    ("T603", "Consonant vs dissonant chords", "exploratory"),
    ("T604", "Familiar vs unfamiliar melody", "exploratory"),
    ("T605", "Narrative vs reversed speech", "exploratory"),
    ("T606", "Nature vs urban soundscapes", "promising"),
    ("T607", "Loudness/annoyance ladder", "exploratory"),
    ("T608", "Selective attention, two streams", "exploratory"),
    ("T701", "Blink-controlled switch", "established"),
    ("T702", "Jaw-clench switch", "established"),
    ("T703", "Eyebrow-raise switch", "promising"),
    ("T704", "Motor imagery left vs right", "exploratory"),
    ("T705", "Visual imagery vs perception", "exploratory"),
    ("T706", "Closed loop: blink-confirm handshake", "established"),
    ("T707", "Closed loop: load-adaptive agent pacing", "exploratory"),
    ("T708", "Closed loop: relaxation neurofeedback", "promising"),
)


# ---------------------------------------------------------------- setup
def _plt():
    """Import matplotlib with Agg + a writable cache dir, once."""
    MPL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _out_path(name: str, out: str | Path | None) -> Path:
    p = Path(out) if out is not None else (FIG_DIR / name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figures
def plot_erp(epochs: Epochs, spec: Any, out: str | Path | None = None) -> Path:
    """Butterfly of single epochs plus per-class mean with a 95% CI band.

    The butterfly is not decoration: on one dry frontal channel the single-trial
    spread is 1-2 orders of magnitude larger than the effect, and hiding that
    behind a smooth average is how a pipeline convinces itself of a P300 that is
    not there.
    """
    plt = _plt()
    p = _out_path(f"{getattr(spec, 'id', 'test')}_erp.png", out)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    X = np.asarray(epochs.X, dtype=float)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    ms = np.asarray(epochs.times, dtype=float) * 1000.0
    rest = X.mean(axis=0) - X.mean()
    for row in X[: min(len(X), 400)]:
        ax.plot(ms, row - X.mean(), color="0.75", lw=0.5, alpha=0.5, zorder=1)
    for i, lab in enumerate(epochs.labels):
        m = np.asarray(epochs.y) == i
        if not np.any(m):
            continue
        seg = X[m]
        mean = seg.mean(axis=0)
        sem = seg.std(axis=0, ddof=1) / np.sqrt(max(1, seg.shape[0]))
        ax.plot(ms, mean, lw=1.8, label=f"{lab} (n={int(m.sum())})", zorder=3)
        ax.fill_between(ms, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.18, zorder=2)
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("time relative to onset (ms)")
    ax.set_ylabel("amplitude (ADC counts, baseline-corrected)")
    ax.set_title(f"{getattr(spec, 'id', '')} {getattr(spec, 'title', '')} - ERPs")
    ax.legend(fontsize=8, loc="best")
    ax.text(0.01, 0.01, "amplitudes are uncalibrated counts; compare shapes, not volts",
            transform=ax.transAxes, fontsize=7, color="0.35")
    return _save(fig, p)


def _bandpower_table(epochs_or_feats: Any, spec: Any) -> tuple[dict[str, np.ndarray], np.ndarray, Epochs | None]:
    """Normalise the several accepted inputs into ``{band: (n,)}`` per epoch."""
    if isinstance(epochs_or_feats, Epochs):
        ep = epochs_or_feats
        X = np.asarray(ep.X, dtype=float)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        table = {b: np.array([band_powers(row, ep.fs)[b] for row in X])
                 for b in BANDS}
        return table, np.asarray(ep.y), ep
    if isinstance(epochs_or_feats, tuple) and len(epochs_or_feats) == 2:
        X, names = epochs_or_feats
        X = np.asarray(X, dtype=float)
        table = {}
        for b in BANDS:
            key = f"bandpower:{b}"
            if key in list(names) and X.ndim == 2:
                table[b] = X[:, list(names).index(key)]
        return table, np.zeros(X.shape[0], dtype=int), None
    if isinstance(epochs_or_feats, dict):
        table = {b: np.asarray(v, dtype=float)
                 for b, v in epochs_or_feats.items() if b in BANDS}
        n = max((v.size for v in table.values()), default=0)
        return table, np.zeros(n, dtype=int), None
    raise TypeError("plot_bandpower accepts Epochs, (X, names) or {band: values}")


def plot_bandpower(epochs_or_feats: Any, spec: Any, out: str | Path | None = None) -> Path:
    """Per-band log power, grouped by class, with per-window error bars.

    Error bars are the standard error across *windows/epochs*, because with
    overlapping windows the number of independent observations is much smaller
    than the number of windows -- the plot shows the spread rather than implying
    precision.
    """
    plt = _plt()
    p = _out_path(f"{getattr(spec, 'id', 'test')}_bandpower.png", out)
    table, y, ep = _bandpower_table(epochs_or_feats, spec)
    labels = list(getattr(spec, "classes", []) or (ep.labels if ep is not None else []))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bands = [b for b in BANDS if b in table]
    if not bands:
        ax.text(0.5, 0.5, "no band-power features in this spec", ha="center", va="center")
        ax.axis("off")
        return _save(fig, p)
    x = np.arange(len(bands))
    width = 0.8 / max(1, len([l for l in (labels or [0])]))
    for i, lab in enumerate(labels or ["all"]):
        m = (y == i) if labels else np.ones(len(next(iter(table.values()))), dtype=bool)
        means, errs = [], []
        for b in bands:
            vals = table[b][m] if np.size(m) == table[b].size else table[b]
            vals = vals[np.isfinite(vals)]
            means.append(float(vals.mean()) if vals.size else np.nan)
            errs.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        ax.bar(x + i * width, means, width, yerr=errs, capsize=2.5, label=str(lab))
    ax.set_xticks(x + width * (len(labels or ["all"]) - 1) / 2)
    ax.set_xticklabels(bands, rotation=20)
    ax.set_ylabel("log10 power (uncalibrated counts)")
    ax.set_title(f"{getattr(spec, 'id', '')} - band power by class (mean +/- SEM over windows)")
    ax.legend(fontsize=8)
    return _save(fig, p)


def plot_confusion(results: dict, spec: Any, out: str | Path | None = None) -> Path:
    """Confusion matrix heatmap from a ``cross_validate`` result."""
    plt = _plt()
    p = _out_path(f"{getattr(spec, 'id', 'test')}_confusion.png", out)
    cv = results.get("cv", results) if isinstance(results, dict) else {}
    conf = np.asarray(cv.get("confusion") or [], dtype=float)
    classes = [str(c) for c in (cv.get("classes") or getattr(spec, "classes", []) or [])]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if conf.size == 0:
        ax.text(0.5, 0.5, "no confusion matrix (no between-class contrast)",
                ha="center", va="center", wrap=True)
        ax.axis("off")
        return _save(fig, p)
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(conf.shape[1]))
    ax.set_yticks(range(conf.shape[0]))
    ax.set_xticklabels(classes[: conf.shape[1]], rotation=30, ha="right")
    ax.set_yticklabels(classes[: conf.shape[0]])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (out-of-fold)")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{getattr(spec, 'id', '')} confusion")
    return _save(fig, p)


def plot_quality(session: Any, out: str | Path | None = None) -> Path:
    """Raw trace (subsampled) plus Welch PSD with the analysis bands shaded."""
    from ..device.quality import assess
    plt = _plt()
    p = _out_path(f"{getattr(getattr(session, 'path', None), 'name', 'session')}_quality.png", out)
    v = np.asarray(getattr(session, "eeg", np.zeros(0)), dtype=float).reshape(-1)
    fs = float(getattr(session, "fs", 250.0))
    rep = assess(v, fs, contact=str(getattr(session, "meta", {}).get("contact", "unknown")))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    ax = axes[0]
    if v.size:
        step = max(1, v.size // 20000)
        t = np.asarray(getattr(session, "t", np.arange(v.size) / fs), dtype=float)
        tt = (t[: v.size: step] - (t[0] if t.size else 0.0))
        ax.plot(tt, v[::step], lw=0.6)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ADC counts")
    ax.set_title(f"raw signal - quality {rep.verdict} ({rep.score}/3)")
    for msg in rep.messages:
        ax.text(0.01, 0.02, " | ".join(rep.messages[:2]), transform=ax.transAxes,
                fontsize=7, color="0.35", va="bottom")
        break
    ax2 = axes[1]
    if v.size >= 8:
        from scipy import signal as sps
        nper = int(max(8, min(v.size, round(2 * fs))))
        f, P = sps.welch(v, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann")
        m = (f > 0.2) & (f <= 50)
        if np.any(m):
            ax2.semilogy(f[m], np.maximum(P[m], 1e-12), lw=0.8, color="k")
            top = float(np.max(P[m])) if np.max(P[m]) > 0 else 1.0
            for b, (lo, hi) in BANDS.items():
                ax2.axvspan(lo, hi, alpha=0.06)
                ax2.text((lo + hi) / 2, top, b, fontsize=6,
                         rotation=90, va="top", ha="center", color="0.4")
        ax2.set_xlabel("frequency (Hz)")
        ax2.set_ylabel("PSD (counts^2/Hz)")
        ax2.set_title(f"PSD - mains {rep.metrics.get('mains_pct', float('nan')):.1f}% of 1-45 Hz")
    return _save(fig, p)


def plot_timeline(session: Any, out: str | Path | None = None) -> Path:
    """Marker strip plot: one row per marker name, x = time, colour by test."""
    plt = _plt()
    p = _out_path(f"{getattr(getattr(session, 'path', None), 'name', 'session')}_timeline.png", out)
    events = list(getattr(session, "events", []) or [])
    fig, ax = plt.subplots(figsize=(10, max(2.4, 0.28 * max(1, len({e.name for e in events})))))
    if not events:
        ax.text(0.5, 0.5, "no markers in this session", ha="center", va="center")
        ax.axis("off")
        return _save(fig, p)
    t0 = float(getattr(session, "t", [0.0])[0]) if np.size(getattr(session, "t", [])) else 0.0
    names = sorted({e.name for e in events})
    tests = sorted({str(e.test_id or "") for e in events})
    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(tests)}
    for e in events:
        y = names.index(e.name)
        ax.plot([e.t_mono - t0], [y], marker="|", ms=14, mew=1.4,
                color=colors.get(str(e.test_id or ""), "k"))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("time in session (s)")
    ax.set_title(f"marker timeline - {len(events)} events, {len(tests)} test id(s)")
    if len(tests) > 1:
        handles = [plt.Line2D([], [], color=colors[t], lw=2, label=t or "(none)") for t in tests]
        ax.legend(handles=handles, fontsize=7, ncol=min(4, len(tests)))
    return _save(fig, p)


# ---------------------------------------------------------------- markdown
def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{nd}f}"
    return str(x)


def _latency_allowed(results: dict) -> tuple[bool, str]:
    tq = str(results.get("timing_quality")
             or (results.get("timing") or {}).get("timing_quality") or "unverified")
    if tq == "verified":
        return True, tq
    return False, tq


def _verdict_reason(spec: Any, results: dict, gates: dict, cv: dict,
                    coverage: dict) -> tuple[str, str]:
    """Plain-language headline + the most likely reason for the observed outcome.

    Honest and specific: "no effect" and "the analysis could not see an effect"
    are different statements and the second one is often the true one on a single
    dry frontal electrode.
    """
    fe = str(getattr(spec, "feasibility", "") or "").lower()
    ba = cv.get("balanced_accuracy")
    n = cv.get("n") or 0
    chance = cv.get("chance")
    frac = coverage.get("retained_fraction")
    passed = gates.get("passed")
    artifact = (results.get("quality_metrics") or {}).get("artifact_fraction")
    perm = (results.get("permutation") or {}).get("p_value")
    if passed is True:
        headline = ("This test passed its gate. The effect reproduces across held-out blocks "
                    "under the grouping that was used.")
    elif passed is None:
        headline = ("Insufficient data to evaluate this test's gates. No gate could be "
                    "computed for this capture; this is a design limitation, not a negative "
                    "result about the participant.")
    else:
        headline = "This test did not pass its gate."
    if passed is None:
        detail = " " + (str(gates.get("reason")
                            or "the pre-registered gate could not be computed from this "
                               "capture").rstrip(".")) + "."
        return headline, detail
    reasons = []
    if ba is None:
        reasons.append("no between-class contrast could be computed at all (single class or "
                       "no usable epochs), so an accuracy gate cannot be met by construction")
    if n < 40:
        reasons.append(f"only {n} accepted epochs: the average is too noisy for a stable "
                       "estimate, which is a data-quantity problem rather than a null result")
    if frac is not None and frac < 0.6:
        reasons.append(f"only {100*frac:.0f}% of candidate epochs survived artefact "
                       "rejection; the surviving subset is not representative")
    if artifact is not None and float(artifact) > 0.25:
        reasons.append("more than a quarter of samples were flagged as artefact "
                       "(dry-electrode motion/drift), which competes with any cortical effect")
    if perm is not None and float(perm) > 0.05 and ba is not None and chance is not None:
        reasons.append(f"the permutation test (p={float(perm):.3f}) says the observed "
                       f"{float(ba):.3f} is within the null distribution")
    if fe in ("exploratory",):
        reasons.append("the spec is marked exploratory: on one frontal channel the predicted "
                       "effect is weak, and a null result is the most likely outcome")
    if fe == "control":
        reasons.append("this is a control protocol: failing an accuracy gate is the expected "
                       "outcome and is what proves the pipeline is not hallucinating")
    if (results.get("timing_quality") or "") != "verified":
        reasons.append("timing is not verified for this session, so latency-dependent effects "
                       "cannot be assessed at all")
    if not reasons:
        reasons.append("the effect is expected to be modest at Fpz and the gate is set at the "
                       "top of the feasibility band; this is a normal null outcome, not a bug")
    detail = ""
    if passed is False and reasons:
        detail = " Most likely reason given the montage: " + "; ".join(reasons[:2]) + "."
    elif passed is True and not gates.get("complete", True):
        detail = (" The verdict rests on the gates that could be evaluated; the metric(s) "
                  "marked *not evaluable* above could not be computed for this capture.")
    return headline, detail


def _gate_table(gates: dict) -> list[str]:
    lines = ["| metric | value | gate | direction | result |",
             "|---|---:|---:|---|---|"]
    for c in gates.get("checks", []):
        if not c.get("applicable", True):
            res = f"n/a - {c.get('reason', 'inapplicable')}"
        elif not c.get("evaluated", False):
            res = f"**not evaluable** - {c.get('reason', 'metric absent')}"
        else:
            res = "PASS" if c.get("ok") else "FAIL"
        lines.append(f"| `{c['metric']}` | {_fmt(c.get('value'))} | {_fmt(c.get('gate'))} | "
                     f"{c.get('direction', '-')} | {res} |")
    return lines


def write_test_report(spec: Any, results: dict, session_path: str | Path | None = None,
                      out_dir: str | Path | None = None) -> Path:
    """Markdown report for one test: gates, chance, coverage, caveats, verdict.

    ``results`` is whatever :func:`~neurobridge.pipeline.evaluate.evaluate_session`
    or :func:`~neurobridge.pipeline.evaluate.train_and_evaluate` returned; missing
    keys degrade to "-" rather than raising, because a report that crashes is
    worse than a sparse one.
    """
    session_path = session_path or results.get("session_path") or results.get("sessions")
    out_root = Path(out_dir) if out_dir is not None else (REPORT_DIR / str(getattr(spec, "id", "test")))
    out_root.mkdir(parents=True, exist_ok=True)
    cv = results.get("cv") or {}
    gates = results.get("gates") or apply_gates(spec, results)
    coverage = results.get("coverage") or {}
    quality = results.get("quality_metrics") or {}
    derived = results.get("derived_metrics") or {}
    latent_ok, tq = _latency_allowed(results)

    headline, detail = _verdict_reason(spec, results, gates, cv, coverage)
    lines: list[str] = [
        f"# {getattr(spec, 'id', '')} - {getattr(spec, 'title', '')}",
        "",
        f"*feasibility:* `{getattr(spec, 'feasibility', '')}` &nbsp;&nbsp; "
        f"*classes:* {', '.join(str(c) for c in (getattr(spec, 'classes', []) or []))} "
        f"&nbsp;&nbsp; *agent use:* `{getattr(spec, 'agent_use', '')}`",
        "",
        f"*session:* `{session_path}`  ",
        f"*timing quality:* `{tq}`  ",
        f"*context features:* {results.get('context_features') or []} "
        f"(signal-only guard: {'checked' if results.get('signal_only_checked') else 'not recorded'})",
        "",
        "## Verdict",
        "",
        f"**{headline}**{detail}",
        "",
        "A failed gate is a normal research outcome. It means this session, this montage and "
        "this analysis did not demonstrate the effect at the pre-registered threshold; it does "
        "not mean the pipeline errored." if gates.get("passed") is False else
        ("This is not a claim that the effect is absent, and it is not a pass: the "
         "pre-registered gate could not be computed from this capture. Re-record with the "
         "missing condition spread across more blocks." if gates.get("passed") is None else
         "The gate was pre-registered and met. What it does *not* mean is that the underlying "
         "state has been read directly: on one frontal channel this is a correlate, and it "
         "should be re-checked on another day before it is relied on."),
        "",
        "## What was measured",
        "",
        f"- balanced accuracy (out-of-fold): **{_fmt(cv.get('balanced_accuracy'))}** "
        f"vs chance {_fmt(cv.get('chance'))} and majority-class baseline "
        f"{_fmt(cv.get('majority_class_baseline'))}",
        f"- AUC (binary only): {_fmt(cv.get('auc'))}",
        f"- macro F1: {_fmt(cv.get('f1'))}",
        f"- epochs: {cv.get('n')}  |  grouping: "
        f"{(cv.get('grouping') or {}).get('group_by', '-')} over "
        f"{(cv.get('grouping') or {}).get('n_groups', '-')} group(s)",
        f"- abstention: rate {_fmt(cv.get('abstention_rate'))}, covered balanced accuracy "
        f"{_fmt(cv.get('balanced_accuracy_covered'))}",
        f"- permutation p: {_fmt((results.get('permutation') or {}).get('p_value'))} "
        f"(n={(results.get('permutation') or {}).get('n_permutations', 0)})",
        f"- information transfer: {_fmt(results.get('itr_bits_per_min'), 2)} bits/min",
        "",
        f"## Acceptance gates",
        "",
        f"**Verdict: {gates.get('verdict', 'PASS' if gates.get('passed') else 'FAIL')}** "
        f"(gates evaluated: {gates.get('n_evaluated', gates.get('n_evaluable', '-'))}, "
        f"not evaluable: {gates.get('n_not_evaluated', gates.get('n_missing', 0))}, "
        f"not applicable: {gates.get('n_inapplicable', 0)})",
        "",
        *_gate_table(gates),
        "",
        f"Evaluated and passing: "
        f"{sum(1 for c in gates.get('checks', []) if c.get('evaluated') and c.get('ok'))} "
        f"of {gates.get('n_evaluated', gates.get('n_evaluable', '-'))}.",
    ]
    if not gates.get("complete", True):
        lines += ["",
                  "> A metric marked *not evaluable* was **not** a failed measurement: the "
                  "value could not be computed for this capture (see the reason in the table). "
                  "It is excluded from the verdict rather than counted against the test."]
    if gates.get("verdict_note"):
        lines += ["", f"> {gates['verdict_note']}"]

    lines += [
        "",
        "## Coverage (what was kept)",
        "",
        f"- retained {coverage.get('retained', '-')} of {coverage.get('candidates', '-')} "
        f"candidate epochs ({_fmt(coverage.get('retained_fraction'))})",
        f"- reject policy: `{coverage.get('reject_policy', '-')}` "
        f"(hard reject: {coverage.get('hard_reject', '-')})",
    ]
    for lab, d in (coverage.get("per_class") or {}).items():
        lines.append(f"- `{lab}`: {d.get('retained', '-')}/{d.get('candidates', '-')} "
                     f"({_fmt(d.get('retained_fraction'))})")
    if coverage.get("exclusions"):
        lines.append("- exclusions: " + ", ".join(
            f"{k}={v}" for k, v in coverage["exclusions"].items()))
    if not coverage:
        lines.append("- _coverage was not recorded for this result_")

    lines += ["", "## Signal quality", ""]
    if quality:
        for k in ("artifact_fraction", "saturation_pct", "drift_ratio", "band_delta_pct",
                  "mains_pct", "motion_jump_pct", "flatline_pct"):
            if k in quality:
                lines.append(f"- {k}: {_fmt(quality[k])}")
    q = results.get("quality") or {}
    if q.get("verdict"):
        lines.append(f"- verdict: **{q.get('verdict')}** ({q.get('score', '-')}/3)")
    for msg in (q.get("messages") or []):
        lines.append(f"  - {msg}")

    lines += ["", "## Derived metrics", ""]
    if derived:
        for k in sorted(derived):
            if k.endswith("_info") or isinstance(derived[k], dict):
                continue
            if "latency" in k and not latent_ok:
                lines.append(f"- {k}: {_fmt(derived[k])} - **not reportable as a finding** "
                             f"(timing_quality={tq})")
            else:
                lines.append(f"- {k}: {_fmt(derived[k])}")
    else:
        lines.append("- _none computed_")
    fa = derived.get("false_activations_per_min")
    fa_info = derived.get("false_activations_info") or {}
    if fa is None:
        lines.append(f"- false activations per minute: not measurable - "
                     f"{fa_info.get('note', 'no idle class')}")
    else:
        lines.append(f"- false activations per minute (idle epochs predicted as a command): "
                     f"{_fmt(fa)} across {fa_info.get('n_idle_epochs', '-')} idle epochs "
                     f"[{fa_info.get('decision_rule', '')}]")

    lines += ["", "## Caveats", ""]
    if not latent_ok:
        lines.append(f"- **timing_quality is {tq}**: stimulus onsets are not verified against "
                     "sample times, so ERP *latency* differences are not findings here. "
                     "Amplitude, shape and band-power results only.")
    lines.append("- single dry frontal electrode (Fpz): no topography, no lateralisation; "
                 "source localisation is not available and was not attempted.")
    lines.append("- amplitudes are uncalibrated 24-bit counts: only relative and "
                 "within-session comparisons are meaningful.")
    lines.append("- overlapping windows are not independent observations; the grouping "
                 "(block/session) reported above is what the accuracy refers to.")
    for w in (results.get("warnings") or [])[:8]:
        lines.append(f"- {w}")
    for note in (results.get("notes") or [])[:10]:
        lines.append(f"- {note}")
    lines.append("- risks declared for this test:")
    for r in (getattr(spec, "risks", []) or []):
        lines.append(f"  - {r}")

    out = out_root / f"{getattr(spec, 'id', 'test')}_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_session_report(session: Any, results_by_test: dict[str, dict]) -> Path:
    """One-page session summary linking every test's report."""
    out_root = Path(getattr(session, "path", REPORT_DIR)) / "reports"
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        from ..device.quality import assess
        rep = assess(np.asarray(getattr(session, "eeg", np.zeros(0)), dtype=float),
                     float(getattr(session, "fs", 250.0)))
        q_lines = ["", "## Signal quality", ""] + [f"- {m}" for m in rep.lines()]
    except Exception:
        q_lines = []
    lines = [
        f"# Session {getattr(getattr(session, 'path', None), 'name', '')}",
        "",
        f"- samples: {int(np.size(getattr(session, 'eeg', [])))}",
        f"- duration: {_fmt(float(getattr(session, 'duration_s', 0.0)), 1)} s",
        f"- events: {len(getattr(session, 'events', []) or [])}",
        f"- tests present: {', '.join(getattr(session, 'test_ids', lambda: [])() or []) or 'none'}",
        f"- timing quality: `{timing_quality_of(session, _StubSpec()) [0]}`",
        *q_lines,
        "",
        "## Tests",
        "",
        "| test | status | balanced accuracy | AUC | n | coverage | report |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for tid, res in (results_by_test or {}).items():
        res = res if isinstance(res, dict) else {}
        cv = res.get("cv") or {}
        gates = res.get("gates") or {}
        passed = res.get("passed", gates.get("passed"))
        status = "not run" if res == {} else ("passed" if passed else "failed")
        lines.append(
            f"| {tid} | {status} | {_fmt(cv.get('balanced_accuracy'))} | {_fmt(cv.get('auc'))} | "
            f"{cv.get('n', '-')} | {_fmt((res.get('coverage') or {}).get('retained_fraction'))} | "
            f"{'linked in ' + str(res.get('report_path')) if res.get('report_path') else '-'} |")
    lines += ["", "Curricula status is summarised in `curriculum_report.md`."]
    out = out_root / "session_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


class _StubSpec:
    """Minimal spec used only to ask for a session's timing quality."""

    id = ""
    processing: dict[str, Any] = {}
    decoder: dict[str, Any] = {}


def _catalog_rows() -> list[tuple[str, str, str]]:
    """(id, title, feasibility) for all 64 tests, from the catalog if possible."""
    try:
        from ..catalog import load_specs
        specs = load_specs()
        if specs:
            return [(s.id, s.title, s.feasibility) for s in sorted(specs, key=lambda s: s.id)]
    except Exception:
        pass
    return list(_FALLBACK_TESTS)


def write_curriculum_report(results_by_test: dict[str, dict]) -> Path:
    """The 64-row progress table: one row per test id, with its headline metric.

    ``results_by_test`` may be rich (:func:`evaluate_session` output) or as thin
    as ``{"T301": {"session": "..."}}`` -- the CLI's ``nb report --curriculum``
    passes the thin form, and a row must still be produced for every test id
    whether or not it has been run.
    """
    rows = _catalog_rows()
    seen = {str(k) for k in (results_by_test or {})}
    lines = [
        "# Curriculum progress",
        "",
        "| # | test | title | feasibility | status | balanced acc | AUC | n | coverage | "
        "grouping | headline |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    n_pass = n_fail = n_run = n_insufficient = 0
    for i, (tid, title, feas) in enumerate(rows, start=1):
        res = (results_by_test or {}).get(tid) or {}
        cv = res.get("cv") or {}
        gates = res.get("gates") or {}
        ba = cv.get("balanced_accuracy")
        auc = cv.get("auc")
        if res:
            n_run += 1
        passed = res.get("passed", gates.get("passed"))
        evaluated = bool(gates.get("n_evaluated") or gates.get("checks")) or ("cv" in res)
        if not res:
            status = "not-run"
        elif passed is None and evaluated:
            status = "insufficient data"
            n_insufficient += 1
        elif not evaluated:
            status = "recorded (not evaluated)"
        elif passed:
            status = "passed"
            n_pass += 1
        else:
            status = "failed"
            n_fail += 1
        headline_parts = []
        if ba is not None:
            headline_parts.append(f"bal-acc {float(ba):.3f} vs chance {_fmt(cv.get('chance'))}")
        elif auc is not None:
            headline_parts.append(f"AUC {float(auc):.3f}")
        elif gates.get("checks"):
            headline_parts.append("quality-gated: " + ", ".join(
                f"{c['metric']}={_fmt(c.get('value'))}" for c in gates["checks"][:2]))
        if not headline_parts:
            headline_parts.append("no metric yet")
        if gates.get("checks") and not all(c.get("applicable", True) for c in gates["checks"]):
            headline_parts.append("(some gates inapplicable)")
        lines.append(
            f"| {i} | {tid} | {title} | {feas} | {status} | {_fmt(ba)} | {_fmt(auc)} | "
            f"{cv.get('n', '-')} | {_fmt((res.get('coverage') or {}).get('retained_fraction'))} | "
            f"{(cv.get('grouping') or {}).get('group_by', '-')} | {'; '.join(headline_parts)} |")
    missing = sorted(set(seen) - {r[0] for r in rows})
    lines += [
        "",
        f"**{len(rows)} tests** in the catalog. Evaluated: {n_run} "
        f"(passed {n_pass}, failed {n_fail}, insufficient data {n_insufficient}, "
        f"not run {len(rows) - n_run}).",
        "",
        "A `failed` row is a normal outcome: the gate was pre-registered and the session did "
        "not reach it. Columns are out-of-fold under the stated grouping; coverage is retained "
        "epochs / candidate epochs.",
    ]
    if missing:
        lines.append("")
        lines.append("Results present for ids outside the 64-row table: " + ", ".join(missing))
    out = REPORT_DIR / "curriculum_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
