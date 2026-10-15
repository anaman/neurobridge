"""Synthetic EEG source.

This exists for three reasons:

1. **You can build and test the entire system before the headband is on.**
   ``nb simulate T301`` runs a full session, writes a real session directory,
   and the pipeline decodes it -- no hardware, no waiting.
2. **It makes the pipeline falsifiable.** The generator injects a known
   stimulus-locked response with known amplitude and jitter, so if the decoder
   cannot recover it the bug is in *our* code, not in the brain.
3. **It is the null model.** ``--effect-size 0`` produces data with no
   class difference at all; a decoder that still reports 80% on that is
   overfitting, and the evaluation harness will say so.

The signal model is a five-band oscillator bank plus 1/f noise, a slow DC
drift (dry electrodes always drift), and stimulus-locked responses whose
amplitude depends on the marker name. Everything is in raw 24-bit-ish ADC
counts so the same quality thresholds apply as for the real device.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .base import EEGSource, Packet

#: Default per-marker response policy.
#: amp  - peak amplitude in ADC counts (real headband: clean EEG p-p ~20k)
#: lat  - peak latency after the marker, seconds
#: wid  - half-width of the deflection, seconds
#: bands- multiplicative band-power gains applied from this marker on
#: kind - special artefact generators
DEFAULT_POLICY: dict[str, dict[str, Any]] = {
    # --- event-related potentials -----------------------------------
    "target":     {"amp": 9000, "lat": 0.30, "wid": 0.10},
    "deviant":    {"amp": 8200, "lat": 0.28, "wid": 0.09},
    "novel":      {"amp": 12000, "lat": 0.30, "wid": 0.13},
    "standard":   {"amp": 700, "lat": 0.20, "wid": 0.08},
    "distractor": {"amp": 3000, "lat": 0.26, "wid": 0.10},
    "class_a":    {"amp": 7500, "lat": 0.30, "wid": 0.10},
    "class_b":    {"amp": 900, "lat": 0.30, "wid": 0.10},
    "class_c":    {"amp": 4200, "lat": 0.30, "wid": 0.10},
    "class_d":    {"amp": 1800, "lat": 0.30, "wid": 0.10},
    "stim_onset": {"amp": 1500, "lat": 0.16, "wid": 0.07},
    "cue_onset":  {"amp": 2200, "lat": 0.22, "wid": 0.09},
    "question":   {"amp": 5200, "lat": 0.35, "wid": 0.13},
    "probe":      {"amp": 4800, "lat": 0.33, "wid": 0.12},
    "trial_onset": {"amp": 0},
    # --- classic ERP components for latency calibration -------------
    "latency_probe": {"amp": 14000, "lat": 0.05, "wid": 0.006},
    # --- artefacts / control channels -------------------------------
    "blink":      {"kind": "blink"},
    "clench":     {"kind": "clench"},
    "swallow":    {"kind": "swallow"},
    # --- sustained states -------------------------------------------
    "eyes_closed": {"kind": "bands", "bands": {"alpha": 3.0, "theta": 1.4, "beta": 0.8}},
    "eyes_open":   {"kind": "bands", "bands": {"alpha": 0.35, "theta": 1.0, "beta": 1.3}},
    # --- sustained cognitive / affective states ---------------------
    "load_high":   {"kind": "bands", "bands": {"theta": 1.7, "alpha": 0.8, "beta": 1.25}},
    "load_low":    {"kind": "bands", "bands": {"theta": 0.8, "alpha": 1.25, "beta": 0.95}},
    "relax":       {"kind": "bands", "bands": {"alpha": 2.2, "beta": 0.8}},
    "effort":      {"kind": "bands", "bands": {"alpha": 0.7, "beta": 1.4}},
}

BAND_FREQS = {"delta": 2.0, "theta": 6.0, "alpha": 10.4, "beta": 20.0, "gamma": 40.0}
BAND_BASE = {"delta": 2600.0, "theta": 900.0, "alpha": 1100.0, "beta": 700.0, "gamma": 260.0}


class FakeSource(EEGSource):
    """Deterministic-ish synthetic single-channel EEG."""

    name = "fake"
    fs = 250.0

    def __init__(self, fs: float = 250.0, seed: int = 20261013,
                 policy: dict[str, dict[str, Any]] | None = None,
                 effect_size: float = 1.0, noise: float = 900.0,
                 drift: float = 2200.0, realtime: bool = True,
                 contact: str = "", packet_samples: int = 10) -> None:
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)
        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self.effect_size = float(effect_size)
        self.noise = float(noise)
        self.drift_amp = float(drift)
        self.realtime = bool(realtime)
        # Mimic the real device's SELF_CHECK payload (four "OK" tokens) so that
        # contact_from_selfcheck() yields "good" and T001's seating protocol
        # exercises its good_contact path instead of always reporting poor
        # contact for a source that is by definition perfect.
        self.selfcheck_ascii = "\x00\x00OK\x00\x00OK\x00\x00OK\x00\x00OK10001050"
        self.contact = contact or "good (4/4 electrodes OK, synthetic)"
        self.packet_samples = int(packet_samples)

        self.band_gains = {b: 1.0 for b in BAND_FREQS}
        self.n_packets = 0
        self.n_samples = 0
        self.events: list[dict[str, Any]] = []      # scheduled responses
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._q: list[Packet] = []
        self._lock = threading.Lock()
        self._k0 = 0                                # global sample counter
        self._phase = {b: float(self.rng.uniform(0, 2 * np.pi)) for b in BAND_FREQS}
        self._lp = np.zeros(3)                      # 1/f noise filter state
        self._drift_phase = float(self.rng.uniform(0, 2 * np.pi))
        self._t_start = time.perf_counter()
        self.log: list[tuple[float, str]] = []

    # ------------------------------------------------------------ lifecycle
    def open(self) -> None:
        self._t_start = time.perf_counter()
        if self.realtime and self._thread is None:
            self._thread = threading.Thread(target=self._thread_main, name="fake-eeg", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _thread_main(self) -> None:
        """Emit packets at the nominal rate, catching up if the thread is late.

        A real amplifier does not slow down because the CPU is busy, so neither
        should the simulation: if this thread is starved (which happens when
        several analyses run at once) we generate the missing packets
        immediately rather than silently producing fewer samples than
        wall-clock time implies. Otherwise a "250 Hz" simulated session can come
        out at an effective 280 Hz or 60 Hz depending on system load, and marker
        alignment in a simulated run would be misleading.
        """
        period = self.packet_samples / self.fs
        next_t = time.perf_counter()
        max_catchup_packets = int(0.5 * self.fs / max(1, self.packet_samples))   # <=0.5 s
        while not self._stop.is_set():
            now = time.perf_counter()
            behind = now - next_t
            catchup = 0
            if behind > period:
                catchup = min(max_catchup_packets, int(behind / period))
            for _ in range(1 + catchup):
                vals, idxs, t = self._generate_packet()
                with self._lock:
                    self._q.append(Packet(values=vals, t_mono=t, indices=idxs))
                    self.n_packets += 1
                    self.n_samples += vals.size
                next_t += period
            if next_t < time.perf_counter():
                next_t = time.perf_counter()
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------ generation
    def _generate_packet(self) -> tuple[np.ndarray, np.ndarray, float]:
        n = self.packet_samples
        k = np.arange(self._k0, self._k0 + n, dtype=np.float64)
        self._k0 += n
        now = time.perf_counter()
        t = now - self._t_start + (k - k[-1]) / self.fs    # last sample ~= now

        sig = np.zeros(n, dtype=np.float64)
        for band, f0 in BAND_FREQS.items():
            f = f0 * (1.0 + 0.01 * self.rng.standard_normal())
            amp = BAND_BASE[band] * self.band_gains[band]
            ph = self._phase[band] + 2 * np.pi * f * (k / self.fs)
            sig += amp * np.sin(ph)
            self._phase[band] = float((self._phase[band] + 2 * np.pi * f * n / self.fs) % (2 * np.pi))

        # 1/f-ish noise: two cascaded one-pole lowpasses on white noise
        white = self.rng.standard_normal(n) * self.noise
        b = 0.55
        out = np.empty(n)
        s0, s1, s2 = self._lp
        for i in range(n):
            s0 = b * s0 + (1 - b) * white[i]
            s1 = b * s1 + (1 - b) * s0
            s2 = b * s2 + (1 - b) * s1
            out[i] = s2
        self._lp = np.array([s0, s1, s2])

        # slow electrode drift
        sig += self.drift_amp * np.sin(2 * np.pi * 0.05 * t + self._drift_phase)

        # scheduled stimulus-locked responses
        with self._lock:
            pending = []
            for ev in self.events:
                rel = t - ev["t0"]
                if rel[-1] < -3.0:
                    pending.append(ev)
                    continue
                if rel[-1] > 6.0:
                    continue
                sig += self._response(rel, ev)
            self.events = pending

        sig += out
        vals = np.clip(np.rint(sig), -8388608, 8388607).astype(np.int32)
        idxs = ((k.astype(np.int64)) % 256).astype(np.int32)
        return vals, idxs, float(now)

    def _response(self, rel: np.ndarray, ev: dict[str, Any]) -> np.ndarray:
        kind = ev.get("kind", "erp")
        amp = float(ev.get("amp", 0.0)) * self.effect_size
        lat = float(ev.get("lat", 0.2))
        wid = max(1e-4, float(ev.get("wid", 0.08)))

        if kind == "blink":
            # large, slow, monophasic deflection (~2 s) - frontal EOG bleed
            m = (rel > 0) & (rel < 2.0)
            out = np.zeros_like(rel)
            out[m] = 260_000 * np.exp(-((rel[m] - 0.25) ** 2) / (2 * 0.22 ** 2))
            return out
        if kind in ("clench", "swallow"):
            m = (rel > 0) & (rel < 1.0)
            out = np.zeros_like(rel)
            hf = np.sin(2 * np.pi * 32.0 * rel[m]) * (1 + 0.5 * np.sin(2 * np.pi * 7 * rel[m]))
            out[m] = 40_000 * hf * np.exp(-rel[m] / 0.35)
            return out
        if kind == "bands":
            return np.zeros_like(rel)

        # ERP: difference-of-Gaussians gives an N1-P3-ish shape
        if amp == 0:
            return np.zeros_like(rel)
        p3 = np.exp(-((rel - lat) ** 2) / (2 * wid ** 2))
        n1 = -0.35 * np.exp(-((rel - max(0.05, lat - 0.2)) ** 2) / (2 * (wid * 0.5) ** 2))
        return amp * (p3 + n1)

    # ------------------------------------------------------------ markers
    def notify_marker(self, code: int, name: str, t_mono: float,
                      meta: dict[str, Any] | None = None) -> None:
        """Schedule the synthetic response for a marker."""
        meta = meta or {}
        key = str(meta.get("synth") or name)
        pol = self.policy.get(key)
        if pol is None:
            pol = self.policy.get(name, {})
        if not pol:
            return
        kind = pol.get("kind", "erp")
        ev = {"t0": float(t_mono), "kind": kind}
        if kind == "bands":
            # sustained change: ramp band gains toward the target
            for b, g in (pol.get("bands") or {}).items():
                if b in self.band_gains:
                    self.band_gains[b] = float(g)
        else:
            jitter = 1.0 + 0.25 * float(self.rng.standard_normal())
            ev["amp"] = float(pol.get("amp", 0.0)) * max(0.2, jitter)
            ev["lat"] = float(pol.get("lat", 0.2)) * (1 + 0.05 * float(self.rng.standard_normal()))
            ev["wid"] = float(pol.get("wid", 0.08))
        with self._lock:
            self.events.append(ev)
        self.log.append((float(t_mono), key))

    def set_bands(self, **gains: float) -> None:
        """Directly set band gains (used by closed-loop/neurofeedback tests)."""
        for b, g in gains.items():
            if b in self.band_gains:
                self.band_gains[b] = float(g)

    # ------------------------------------------------------------ poll
    def poll(self, timeout_s: float = 0.0) -> list[Packet]:
        deadline = time.perf_counter() + timeout_s
        while True:
            with self._lock:
                out = self._q
                self._q = []
            if out or time.perf_counter() >= deadline:
                return out
            time.sleep(0.002)

    # ------------------------------------------------------------ batch mode
    def generate(self, duration_s: float, markers: list[tuple[float, str]] | None = None,
                 t0: float | None = None) -> tuple[np.ndarray, np.ndarray, list[tuple[float, str]]]:
        """Offline generation of a whole recording.

        Returns ``(values, times, markers)`` where times are monotonic-style
        seconds relative to ``t0``. Used by the test-suite and by
        ``nb simulate --offline``.
        """
        t0 = float(t0 if t0 is not None else 0.0)
        n = int(duration_s * self.fs)
        t = t0 + np.arange(n) / self.fs
        self._t_start = t0
        self._k0 = 0
        # Band-power changes are *sustained states*, not impulses: build a
        # per-band gain envelope so a block that switches from relaxed to
        # effortful really does change spectrum mid-recording. Multiplying a
        # continuous oscillator by a piecewise gain keeps phase continuity,
        # which matters because the analysis measures band power, not phase.
        ordered = sorted(markers or [], key=lambda x: x[0])
        gain_env = {b: np.ones(n) for b in BAND_FREQS}
        for tm, name in ordered:
            pol = self.policy.get(str(name), {})
            if pol.get("kind") != "bands":
                continue
            i0 = int(max(0.0, min(float(tm), float(duration_s))) * self.fs)
            for b, g in (pol.get("bands") or {}).items():
                if b in gain_env:
                    gain_env[b][i0:] = float(g)
        sig = np.zeros(n)
        for band, f0 in BAND_FREQS.items():
            sig += BAND_BASE[band] * gain_env[band] * np.sin(2 * np.pi * f0 * t + self._phase[band])
        # leave the generator in the final state so a later realtime run agrees
        self.band_gains = {b: float(gain_env[b][-1]) for b in BAND_FREQS}
        white = self.rng.standard_normal(n) * self.noise
        b = 0.55
        filt = np.empty(n)
        s0 = s1 = s2 = 0.0
        for i in range(n):
            s0 = b * s0 + (1 - b) * white[i]
            s1 = b * s1 + (1 - b) * s0
            s2 = b * s2 + (1 - b) * s1
            filt[i] = s2
        sig += filt
        sig += self.drift_amp * np.sin(2 * np.pi * 0.05 * t + self._drift_phase)
        # generate ERP-only events (bands handled above)
        self.events = []
        for tm, name in sorted(markers or [], key=lambda x: x[0]):
            pol = self.policy.get(name, {})
            if pol.get("kind") == "bands":
                continue
            amp = float(pol.get("amp", 0.0)) * self.effect_size
            if amp == 0 and pol.get("kind") is None:
                continue
            self.events.append({"t0": t0 + tm, "kind": pol.get("kind", "erp"),
                                "amp": amp, "lat": float(pol.get("lat", 0.2)),
                                "wid": float(pol.get("wid", 0.08))})
        for ev in self.events:
            sig += self._response(t - ev["t0"], ev)
        vals = np.clip(np.rint(sig), -8388608, 8388607).astype(np.int32)
        return vals, t, list(markers or [])

    @property
    def info(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "fs": self.fs,
            "contact": self.contact,
            "effect_size": self.effect_size,
            "montage": "synthetic single frontal channel",
        }

    def stats(self) -> dict[str, Any]:
        return {"packets": self.n_packets, "samples": self.n_samples,
                "scheduled_responses": len(self.events)}
