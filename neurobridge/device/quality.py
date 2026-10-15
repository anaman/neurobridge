"""Signal-quality assessment for a single dry frontal channel.

Reused logic from the original ``ble/`` reverse-engineering session
(``mne_pipeline.py``): the headband saturates at the 24-bit rail when the
electrodes are off the head, and dry electrodes drift, so a capture can look
like "brainwaves" while being pure artefact. Every session gets an explicit
verdict, and the pipeline refuses to train on POOR captures.

**Thresholds are calibrated against this device, not against a wet lab cap.**
The first version of this module inherited a wet-multi-channel intuition
(drift < 3, delta share < 60 %) and duly graded the participant's *own*
verified recording as MARGINAL: that capture had 0.00 % saturation, a drift
ratio of 7.1, a delta share of 81.2 % -- and an alpha ratio of 2.24 on eye
closure, i.e. unambiguous cortical signal. A dry frontal electrode on the
forehead genuinely carries most of its energy below 4 Hz, so those numbers are
normal, not artefactual.

Hence two threshold sets: ``dry_frontal`` (the default, matching this headband)
and ``wet_multichannel`` (the original, stricter one, for when this pipeline is
pointed at a different montage). The point of the verdict is to catch railing,
dropouts and movement, not to impose a spectral shape this hardware never had.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

SAT = 8388607.0            # 2**23 - 1: positive 24-bit rail
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)]

#: score -> verdict, for a four-point scale
VERDICTS = {0: "POOR", 1: "POOR", 2: "MARGINAL", 3: "FAIR", 4: "GOOD"}

#: Per-montage thresholds. Each entry is (pass_if_below, hard_fail_if_above).
#: ``None`` for hard-fail means "never hard-fail on this metric".
THRESHOLDS: dict[str, dict[str, tuple[float, float | None]]] = {
    # dry frontal single electrode (this headband): high delta share and a few
    # units of drift are the normal operating point
    "dry_frontal": {
        "saturation_pct": (0.5, 5.0),
        "drift_ratio": (15.0, None),
        "delta_pct": (90.0, None),
        "motion_pct": (3.0, None),
    },
    # wet multi-channel cap: the original, stricter expectations
    "wet_multichannel": {
        "saturation_pct": (0.5, 5.0),
        "drift_ratio": (3.0, None),
        "delta_pct": (60.0, None),
        "motion_pct": (1.0, None),
    },
}


@dataclass
class QualityReport:
    verdict: str = "POOR"
    score: int = 0
    messages: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    contact: str = "unknown"
    montage: str = "dry_frontal"
    max_score: int = 4

    @property
    def usable(self) -> bool:
        """Training only proceeds on FAIR/GOOD captures."""
        return self.score >= max(2, self.max_score - 1)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usable"] = self.usable
        return d

    def lines(self) -> list[str]:
        out = [f"SIGNAL QUALITY: {self.verdict} ({self.score}/{self.max_score}, "
               f"{self.montage} thresholds)"]
        out += [f"  - {m}" for m in self.messages]
        if self.contact != "unknown":
            out.append(f"  - electrode self-check: {self.contact}")
        return out


def assess(values: np.ndarray, fs: float = 250.0, contact: str = "unknown",
           montage: str = "dry_frontal", alpha_ratio: float | None = None
           ) -> QualityReport:
    """Judge whether a capture looks like real cortical EEG.

    ``alpha_ratio`` (from an eyes-open/eyes-closed protocol) is optional but
    valuable evidence: a ratio above ~1.5 means cortex is visible on this
    channel regardless of how much sub-4 Hz energy the dry electrode carries.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    rep = QualityReport(contact=contact, montage=montage,
                        max_score=4)
    thr = THRESHOLDS.get(montage, THRESHOLDS["dry_frontal"])
    if v.size < fs:
        rep.messages.append(f"only {v.size} samples - too short to judge")
        return rep
    hard_fail = False

    # 1. saturation at the 24-bit rail -> amplifier railed (electrodes off head)
    satpct = 100.0 * float(np.mean(np.abs(v) >= SAT - 2))
    rep.metrics["saturation_pct"] = satpct
    sat_pass, sat_fail = thr["saturation_pct"]
    if sat_fail is not None and satpct > sat_fail:
        rep.messages.append(f"SATURATION: {satpct:.2f}% of samples railed at the 24-bit "
                            f"limit (> {sat_fail}%) - electrodes are off the head or dry")
        hard_fail = True
    elif satpct <= sat_pass:
        rep.score += 1
        rep.messages.append(f"saturation {satpct:.3f}% (ok, <= {sat_pass}%)")
    else:
        rep.messages.append(f"saturation {satpct:.2f}% (above the {sat_pass}% comfort "
                            f"line but not railed)")

    # 2. drift: sub-1 Hz energy vs the cortical band
    x = v - v.mean()
    X = np.fft.rfft(x * np.hanning(x.size))
    fr = np.fft.rfftfreq(x.size, d=1.0 / fs)
    P = np.abs(X) ** 2
    slow = float(P[(fr > 0) & (fr < 1)].sum())
    fast = float(P[(fr >= 1) & (fr <= 45)].sum())
    ratio = slow / fast if fast > 0 else 999.0
    rep.metrics["drift_ratio"] = ratio
    rep.metrics["slow_power"] = slow
    rep.metrics["fast_power"] = fast
    drift_pass = thr["drift_ratio"][0]
    if ratio < drift_pass:
        rep.score += 1
        rep.messages.append(f"drift ratio {ratio:.2f} (ok, < {drift_pass})")
    else:
        rep.messages.append(f"drift ratio {ratio:.2f} - electrode drift dominates "
                            f"(>= {drift_pass}); reseat and clean the electrodes")

    # 3. delta share: railed/artefact captures are >60% delta
    band = {nm: float(P[(fr >= lo) & (fr < hi)].sum()) for nm, lo, hi in BANDS}
    tot = sum(band.values()) or 1.0
    for nm, _, _ in BANDS:
        rep.metrics[f"band_{nm}_pct"] = 100.0 * band[nm] / tot
    d = rep.metrics["band_delta_pct"]
    delta_pass = thr["delta_pct"][0]
    if d < delta_pass:
        rep.score += 1
        rep.messages.append(f"delta share {d:.1f}% (ok, < {delta_pass}%)")
    else:
        rep.messages.append(f"delta share {d:.1f}% (>= {delta_pass}%): either serious "
                            f"drift or a genuinely slow capture - check the raw trace")

    # 4. mains contamination (informational: the pipeline notches it out)
    mains = max(_band_pct(P, fr, 49.0, 51.0), _band_pct(P, fr, 59.0, 61.0))
    rep.metrics["mains_pct"] = mains
    if mains > 25:
        rep.messages.append(f"mains hum {mains:.1f}% of 1-45 Hz (notched, but check grounding)")

    # 5. motion: sample-to-sample steps far above the robust scale
    d1 = np.abs(np.diff(v))
    med = float(np.median(d1)) or 1.0
    frac_jump = float(np.mean(d1 > 20 * med))
    rep.metrics["motion_pct"] = 100 * frac_jump
    motion_pass = thr["motion_pct"][0]
    if 100 * frac_jump < motion_pass:
        rep.score += 1
        rep.messages.append(f"motion {100*frac_jump:.2f}% of samples jump >20x median "
                            f"(ok, < {motion_pass}%)")
    else:
        rep.messages.append(f"motion artefacts: {100*frac_jump:.2f}% of samples jump "
                            f">20x the median step - movement, jaw or electrode rub")

    # alpha reactivity is the strongest positive evidence available on this
    # montage: it says cortex, whatever the spectral shape looks like
    if alpha_ratio is not None:
        rep.metrics["alpha_ratio_closed_open"] = float(alpha_ratio)
        if alpha_ratio >= 1.5:
            rep.messages.append(f"alpha x{alpha_ratio:.2f} on eye closure - real cortical "
                                f"signal is visible on this channel")
        elif alpha_ratio < 0.9:
            rep.messages.append(f"alpha x{alpha_ratio:.2f} on eye closure (inverted) - "
                                f"treat this capture as artefact-dominated")

    if hard_fail:
        rep.score = 0
    rep.score = max(0, min(rep.score, rep.max_score))
    rep.verdict = VERDICTS.get(rep.score, "POOR")
    return rep


def _band_pct(P: np.ndarray, fr: np.ndarray, lo: float, hi: float) -> float:
    tot = float(P[(fr >= 1) & (fr <= 45)].sum()) or 1.0
    return 100.0 * float(P[(fr >= lo) & (fr <= hi)].sum()) / tot


def alpha_reactivity(eyes_open: np.ndarray, eyes_closed: np.ndarray,
                     fs: float = 250.0) -> dict[str, float]:
    """The classic validation: alpha power rises when the eyes close.

    This is the cheapest available proof that the channel carries real
    cortical signal rather than electrode artefact. Returns the ratio and a
    boolean verdict.
    """
    a_open = _alpha_pct(eyes_open, fs)
    a_shut = _alpha_pct(eyes_closed, fs)
    ratio = a_shut / a_open if a_open > 0 else float("inf")
    if ratio >= 1.5:
        verdict = "STRONG - this is real EEG"
    elif ratio >= 1.1:
        verdict = "CONSISTENT - alpha rises with eyes closed"
    elif ratio > 0.9:
        verdict = "WEAK - no clear alpha blockade; reseat the headband"
    else:
        verdict = "INVERTED - likely artefact dominated, do not trust the session"
    return {"alpha_open_pct": a_open, "alpha_closed_pct": a_shut,
            "alpha_ratio": ratio, "text": verdict}


def _alpha_pct(v: np.ndarray, fs: float) -> float:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size < fs:
        return 0.0
    x = v - v.mean()
    X = np.fft.rfft(x * np.hanning(x.size))
    fr = np.fft.rfftfreq(x.size, d=1.0 / fs)
    P = np.abs(X) ** 2
    tot = float(P[(fr >= 1) & (fr <= 45)].sum()) or 1.0
    return 100.0 * float(P[(fr >= 8) & (fr < 13)].sum()) / tot


def contact_from_selfcheck(payload: str) -> tuple[str, int]:
    """Parse the device self-check ASCII blob: four ``OK`` == good contact."""
    n_ok = payload.count("OK")
    if n_ok >= 4:
        return "good (4/4 electrodes OK)", n_ok
    if n_ok >= 2:
        return f"partial ({n_ok}/4 electrodes OK) - reseat the band", n_ok
    if n_ok >= 1:
        return f"poor ({n_ok}/4 electrodes OK) - clean electrodes with saline", n_ok
    return "no electrodes reporting OK - clean and reseat before recording", 0
