"""Smoke tests: everything that must work with no headband, no network, no screen.

    cd /path/to/neurobridge && .venv/bin/python -m pytest tests -q -p no:anyio

(The ``-p no:anyio`` disables an unrelated anyio pytest plugin that conflicts
with some system pytest installations; see the README.)

The important test is ``test_simulate_decodes_injected_erp``: it is the only
end-to-end guarantee that the acquisition timeline, the marker stream, the
epoching and the decoder agree with each other.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- catalog
def test_catalog_has_64_valid_specs():
    from neurobridge.catalog import load_specs, validate_catalog
    specs = load_specs()
    problems = validate_catalog()
    assert not problems, f"catalog problems: {problems}"
    assert len(specs) == 64, f"expected 64 specs, found {len(specs)}"
    ids = [s.id for s in specs]
    assert len(set(ids)) == len(ids), "duplicate test ids"
    tiers = {}
    for s in specs:
        tiers[s.tier] = tiers.get(s.tier, 0) + 1
    assert tiers == {t: 8 for t in range(8)}, f"expected 8 tests per tier, got {tiers}"


def test_every_spec_is_runnable():
    """A spec that cannot be turned into a timeline is a latent runtime error."""
    from neurobridge.catalog import load_specs
    from neurobridge.simulate import build_offline_timeline, is_sustained
    for spec in load_specs():
        events, duration = build_offline_timeline(spec, scale=0.05)
        names = {e["name"] for e in events}
        assert duration > 0, spec.id
        assert "session_start" in names and "test_end" in names, spec.id
        if is_sustained(spec):
            assert any(e["class"] for e in events), f"{spec.id}: no condition markers"
        else:
            onsets = [e for e in events if e["name"] in ("stim_onset", "deviant", "target",
                                                         "standard", "class_a", "blink")]
            assert onsets, f"{spec.id}: no trial markers"


# ---------------------------------------------------------------- device
def test_fake_source_shapes_and_quality():
    from neurobridge.device.fake import FakeSource
    from neurobridge.device.quality import assess
    src = FakeSource(seed=1, realtime=False)
    v, t, _ = src.generate(20.0, [(1.0, "standard"), (3.0, "deviant")])
    assert v.shape == t.shape == (5000,)
    assert v.min() > -8388608 and v.max() < 8388607
    rep = assess(v, 250.0)
    assert rep.verdict in ("FAIR", "GOOD"), rep.lines()
    assert rep.usable


def test_fake_source_injects_erp():
    """deviant > standard in the P300 window, otherwise nothing downstream matters."""
    from neurobridge.device.fake import FakeSource
    src = FakeSource(seed=2, realtime=False)
    markers = []
    t = 1.0
    for i in range(120):
        markers.append((t, "deviant" if i % 5 == 0 else "standard"))
        t += 1.0
    v, times, _ = src.generate(t + 1.0, markers)
    t0 = times[0]
    fs = 250
    dev, std = [], []
    for tm, nm in markers:
        i = int((tm - t0) * fs)
        w = v[i + int(0.25 * fs): i + int(0.45 * fs)]
        (dev if nm == "deviant" else std).append(float(w.mean()))
    assert np.mean(dev) > np.mean(std) + 1000, (np.mean(dev), np.mean(std))


def test_sustained_state_injection_is_time_locked():
    """A band-policy marker must only change the signal *after* it fires."""
    from neurobridge.device.fake import FakeSource
    src = FakeSource(seed=3, realtime=False)
    v, t, _ = src.generate(60.0, [(30.0, "eyes_open"), (45.0, "eyes_closed")])
    fs = 250
    open_win = v[int(32 * fs): int(38 * fs)]
    shut_win = v[int(50 * fs): int(56 * fs)]

    def alpha(x):
        x = x - x.mean()
        P = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        fr = np.fft.rfftfreq(x.size, 1 / fs)
        return float(P[(fr >= 8) & (fr < 13)].sum())

    assert alpha(shut_win) > 1.5 * alpha(open_win), (alpha(open_win), alpha(shut_win))


# ---------------------------------------------------------------- store
def test_session_round_trip(tmp_path):
    from neurobridge import store
    w = store.SessionWriter(session_id="s-test", meta={"fs": 250.0}, root=tmp_path)
    v = np.arange(1000, dtype=np.float32)
    w.event("session_start")
    w.add_samples(v, np.linspace(0, 4, 1000))
    w.event("trial_onset", test_id="T301", trial=0, block=0, meta={"class": "deviant"})
    p = w.close()
    s = store.Session.load(p)
    assert s.eeg.size == 1000
    assert s.duration_s > 3.9
    assert [e.name for e in s.events] == ["session_start", "trial_onset"]
    assert s.events_for(test_id="T301")[0].meta["class"] == "deviant"
    shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("test_id,gate", [("T301", 0.62), ("T201", 0.55)])
def test_simulate_decodes_injected_erp(test_id, gate, tmp_path, monkeypatch):
    """If the pipeline cannot recover a *known* injected effect, it is broken."""
    from neurobridge.catalog import load_one
    from neurobridge.simulate import simulate_test
    from neurobridge.pipeline import evaluate as E
    from neurobridge import store, config

    monkeypatch.setattr(config, "SESSION_DIR", tmp_path)
    spec = load_one(test_id)
    path = simulate_test(spec, effect_size=1.0, seed=7, scale=0.15)
    session = store.Session.load(path)
    assert session.eeg.size > 2500
    res = E.evaluate_session(session, spec)
    acc = res["cv"]["balanced_accuracy"]
    assert acc is not None and acc >= gate, f"{test_id}: balanced accuracy {acc} < {gate}"


def test_null_model_stays_near_chance(tmp_path, monkeypatch):
    """effect_size=0 must NOT decode: this is the pipeline's false-positive floor."""
    from neurobridge.catalog import load_one
    from neurobridge.simulate import simulate_test
    from neurobridge.pipeline import evaluate as E
    from neurobridge import store, config

    monkeypatch.setattr(config, "SESSION_DIR", tmp_path)
    spec = load_one("T301")
    path = simulate_test(spec, effect_size=0.0, seed=11, scale=0.15)
    res = E.evaluate_session(store.Session.load(path), spec)
    acc = res["cv"]["balanced_accuracy"]
    assert acc is not None and acc < 0.70, f"null model decoded at {acc}: leakage somewhere"


# ---------------------------------------------------------------- agent
def test_state_estimator_and_intents():
    from neurobridge.agent import IntentDetector, StateEstimator
    from neurobridge.device.fake import FakeSource
    src = FakeSource(seed=5, realtime=False)
    v, t, _ = src.generate(20.0, [(5.0, "blink"), (10.0, "clench"), (15.0, "blink")])
    est = StateEstimator(window_s=4.0)
    det = IntentDetector()
    intents = []
    for i in range(0, v.size, 250):
        est.push(v[i:i + 250], t[i:i + 250])
        intents += det.push(v[i:i + 250], t[i:i + 250])
    st = est.estimate()
    assert 0.0 <= st.confidence <= 1.0
    assert st.valence is None, "valence must be None on a single channel"
    kinds = {x["intent"] for x in intents}
    assert {"blink", "clench"} <= kinds, f"missed artefacts: {kinds}"


def test_bridge_app_builds():
    from neurobridge.agent import create_app
    app = create_app(device="fake")
    routes = {getattr(r, "path", None) for r in app.routes}
    for want in ("/health", "/state", "/schema", "/events", "/mark", "/state/stream", "/ws"):
        assert want in routes, f"missing route {want}"


# ---------------------------------------------------------------- agent link
def test_link_provenance_never_calls_a_generator_a_person():
    """The one mistake this project refuses to make, asserted directly."""
    from neurobridge.agent.link import Provenance

    fake = Provenance(device="fake", requested_device="auto", mode="live")
    assert fake.is_synthetic and not fake.is_live_human
    assert "SYNTHETIC" in fake.banner()

    # an unresolved alias is not evidence of a person either: fail closed
    alias = Provenance(device="auto", requested_device="auto", mode="live")
    assert alias.is_synthetic and not alias.is_live_human

    real = Provenance(device="th21a", requested_device="auto", mode="live")
    assert real.is_live_human and not real.is_synthetic

    # a replay is real EEG but not the live person, and must not be called either
    rep = Provenance(device="replay:s-1", requested_device="auto", mode="replay")
    assert rep.is_replay and not rep.is_live_human and not rep.is_synthetic


def test_link_parse_duration():
    from neurobridge.agent.link import parse_duration
    assert parse_duration("90s") == 90.0
    assert parse_duration("10m") == 600.0
    assert parse_duration("1h") == 3600.0
    assert parse_duration(45) == 45.0
    assert parse_duration(None) is None
    assert parse_duration("nonsense") is None


def test_link_sse_parser_ignores_heartbeats_and_junk():
    from neurobridge.agent.link import iter_sse
    lines = [": heartbeat", 'data: {"seq": 1}', "", 'data: {"seq": 2}', "",
             "data: {not json}", "", 'data: {"seq": 3}', ""]
    got = [o["seq"] for o in iter_sse(lines)]
    assert got == [1, 2, 3], f"SSE parser mangled the stream: {got}"


def _link_payload(seq: int, derivation: str = "heuristic_baseline", **kw):
    from neurobridge.agent.link import STATE_FIELDS
    p = {
        "seq": seq, "t_mono": 100.0 + seq * 0.5,
        "_recv_mono": 1000.0 + seq * 0.5, "_recv_wall": "2026-10-14T10:00:00",
        "_device": "th21a", "confidence": 0.8, "artifact_level": 0.01,
        "contact": "good (4/4)", "source": "th21a", "intents": [],
        "model_info": {"fields": {f: {"derivation": derivation} for f in STATE_FIELDS}},
    }
    for i, f in enumerate(STATE_FIELDS):
        p[f] = 0.5 + 0.01 * i * seq
    p.update(kw)
    return p


def test_link_summarise_refuses_to_report_pinned_fields_as_measurements():
    """uncalibrated_neutral pins every field at 0.5; that is not a measurement."""
    from neurobridge.agent.link import Provenance, summarize
    header = {"provenance": Provenance(device="th21a", mode="live").as_dict()}
    rows = [_link_payload(i, derivation="uncalibrated_neutral") for i in range(1, 13)]
    s = summarize(rows, header, expected_hz=2.0)
    assert s["provenance_label"] == "LIVE headband (th21a)"
    assert s["informative"] is False
    assert s["fields"]["attention"]["informative"] is False
    assert any("uncalibrated_neutral" in x for x in s["limitations"]), s["limitations"]


def test_link_summarise_flags_synthetic_and_low_confidence():
    from neurobridge.agent.link import Provenance, summarize
    header = {"provenance": Provenance(device="fake", requested_device="auto",
                                       mode="live").as_dict()}
    rows = [_link_payload(i, confidence=0.15) for i in range(1, 13)]
    s = summarize(rows, header)
    assert s["synthetic"] is True
    assert any("SYNTHETIC" in x for x in s["limitations"])
    assert any("confidence" in x for x in s["limitations"])


def test_link_summarise_tracks_a_real_trend_without_inventing_one():
    from neurobridge.agent.link import Provenance, summarize
    header = {"provenance": Provenance(device="th21a", mode="live").as_dict()}
    # a gentle ramp: enough to read as a trend, not enough to be a red flag
    rows = [_link_payload(i, attention=0.45 + 0.02 * i, load=0.5)
            for i in range(1, 13)]
    s = summarize(rows, header, expected_hz=2.0)
    assert s["informative"] is True
    att = s["fields"]["attention"]
    assert att["trend"]["direction"] == "rising", att["trend"]
    assert att["trend"]["delta"] > 0.05
    # a constant field must not be dressed up as a trend
    assert s["fields"]["load"]["trend"]["direction"] == "flat"
    assert any("baseline-relative" in x for x in s["limitations"])


def test_link_summarise_flags_fields_that_swing_across_the_whole_range():
    """A field crossing most of 0..1 in one window is noise, not a state.

    Nobody's attention goes floor-to-ceiling inside a minute, so the summariser
    must warn rather than let the trend column imply a real change.
    """
    from neurobridge.agent.link import Provenance, summarize
    header = {"provenance": Provenance(device="th21a", mode="live").as_dict()}
    rows = [_link_payload(i, attention=0.05 + 0.07 * i) for i in range(1, 13)]
    s = summarize(rows, header)
    assert s["fields"]["attention"]["max"] - s["fields"]["attention"]["min"] > 0.5
    assert any("swing across most of the 0..1 range" in x
               for x in s["limitations"]), s["limitations"]


def test_link_summarise_flags_a_high_artifact_window():
    """artifact_level is the one honest readout before a baseline exists."""
    from neurobridge.agent.link import Provenance, summarize
    header = {"provenance": Provenance(device="th21a", mode="live").as_dict()}
    rows = [_link_payload(i, derivation="uncalibrated_neutral", artifact_level=0.5)
            for i in range(1, 13)]
    s = summarize(rows, header)
    assert any("Artifact level averages" in x for x in s["limitations"]), s["limitations"]


def test_link_record_writes_a_self_describing_log(tmp_path):
    """A log read hours later must still say whether it described a person."""
    from neurobridge.agent.link import Provenance, load_log, record, summarize

    class _Stub:
        base_url = "stub://bridge"

        def health(self):
            return {"device": "fake", "requested_device": "auto", "mode": "live"}

        def provenance(self):
            return Provenance(device="fake", requested_device="auto", mode="live")

        def stream(self, stop=None, on_open=None):
            if on_open:
                on_open(self.provenance())
            for i in range(1, 5):
                yield _link_payload(i, derivation="uncalibrated_neutral")

    out = tmp_path / "link.jsonl"
    receipt = record(_Stub(), out_path=out)
    assert receipt["n_payloads"] == 4 and receipt["ok"]

    header, rows = load_log(out)
    assert header.get("_type") == "header"
    assert header["provenance"]["device"] == "fake"
    assert header["provenance"]["is_synthetic"] is True
    assert len(rows) == 4

    s = summarize(rows, header)
    assert s["synthetic"] is True and s["n_payloads"] == 4


def test_link_that_reports_a_frozen_bridge_is_not_live():
    """A wedged bridge still answers 200, so liveness needs two reads."""
    from neurobridge.agent.link import probe_liveness

    class _Frozen:
        def state(self, explain=False):
            return {"t_mono": 500.0, "seq": 7}

    live = probe_liveness(_Frozen(), gap_s=0.05)
    assert live["advanced"] is False and live["seq_delta"] == 0


def test_state_source_reports_the_resolved_device_never_the_alias():
    """`source` must name what produces samples, not the `--device` alias.

    The bridge is normally started with ``--device auto``; if that alias reached
    the payload, a synthetic fallback would be indistinguishable from a headband.
    """
    from neurobridge.agent import StateEstimator
    from neurobridge.device.fake import FakeSource

    src = FakeSource(seed=3, realtime=False)
    v, t, _ = src.generate(8.0, [])
    for name, expect in ((None, "unknown"), ("fake", "fake"), ("th21a", "th21a")):
        est = StateEstimator()
        est.source_name = name
        est.push(v, t)
        assert est.estimate().source == expect, (name, est.estimate().source)


def test_signal_quality_fails_closed_for_anything_but_the_headband():
    """A synthetic or unresolved capture must never be called `verified`."""
    from neurobridge.agent.bridge import BridgeState
    from neurobridge.agent.state import StateEstimate

    def est(source):
        return StateEstimate(t_mono=1.0, t_wall="x", attention=.5, arousal=.5, load=.5,
                             fatigue=.5, relaxation=.5, valence=None,
                             artifact_level=0.0, contact="good (4/4 electrodes OK)",
                             confidence=0.8, source=source, intents=[],
                             model_info={"quality": {"verdict": "GOOD"}})

    assert BridgeState._signal_quality(None, est("th21a")) == "verified"
    for src in ("fake", "auto", "unknown", "replay:s-20261013-1"):
        assert BridgeState._signal_quality(None, est(src)) == "coarse", src


def test_calibrate_route_exists_and_reports_provenance():
    """POST /calibrate is the only thing that establishes the personal baseline."""
    from neurobridge.agent import create_app
    app = create_app(device="fake")
    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/calibrate" in routes, "the live path has no way to set a baseline"


def test_calibrate_before_enough_data_says_why_and_flags_synthetic():
    from neurobridge.agent.bridge import BridgeState
    bs = BridgeState(device="fake", mode="live")          # feeder never started
    res = bs.calibrate()
    assert res["calibrated"] is False
    assert res["ok"] is False and res["reason"] == "not_enough_data"
    assert res["provenance"]["device"] == "fake"
    assert res["is_live_human"] is False
    assert "warning" in res, "a synthetic baseline must be flagged, not silent"


def test_device_selfcheck_reaches_the_state_payload():
    """`contact` must carry the electrode self-check, not a permanent "unknown".

    protocol.md promises this field to the agent and tells it to ask the wearer to
    reseat the band when it is not good; that instruction is dead if the device's
    own verdict never reaches the estimate.
    """
    from neurobridge.agent import StateEstimator
    from neurobridge.device.fake import FakeSource

    src = FakeSource(seed=7, realtime=False)
    v, t, _ = src.generate(8.0, [])

    est = StateEstimator()
    est.device_contact = "good (4/4 electrodes OK)"
    est.push(v, t)
    assert est.estimate().contact.startswith("good"), est.estimate().contact

    # an off-head self-check must cap confidence rather than be averaged in
    off = StateEstimator()
    off.device_contact = "no electrodes reporting OK"
    off.push(v, t)
    assert off.estimate().confidence <= 0.19, "off-head capture was not capped"


def test_no_data_state_does_not_claim_baseline_derivations():
    """With no window of data, fields must say no_data - not heuristic_baseline.

    `_model_info` defaults an omitted derivation to 'heuristic_baseline', which a
    consumer reads as '0.5 means at your baseline'. Before there is a single
    window there is no baseline and the fields mean nothing.
    """
    from neurobridge.agent import StateEstimator
    est = StateEstimator()
    info = est.estimate().model_info
    derivs = {v["derivation"] for v in info["fields"].values()}
    assert derivs == {"no_data"}, derivs
    assert info["status"] == "no_data"


# ---------------------------------------------------------------- monitor
def _tiny_session(root, n=1500, fs=250.0, session_id="unit-mon"):
    """A real native session, so replay tests exercise the real loader."""
    import numpy as np
    from neurobridge.store import SessionWriter
    w = SessionWriter(session_id=session_id, root=root, meta={"fs": fs})
    t = np.arange(n) / fs + 5000.0
    v = (np.sin(2 * np.pi * 10.0 * t) * 100).astype(np.float32)
    w.add_samples(v, t)
    w.close()
    return root / session_id


def test_monitor_records_csv_and_native_together(tmp_path):
    """CSV and the native format come from the same calls and must agree."""
    import numpy as np
    from neurobridge.agent.link import Provenance
    from neurobridge.monitor import MonitorRecorder
    from neurobridge.store import Session

    r = MonitorRecorder(out_dir=tmp_path, session_id="rec1",
                        provenance=Provenance(device="th21a", mode="live",
                                              calibrated=True))
    t = np.arange(500) / 250.0 + 900.0
    v = np.linspace(-100, 100, 500)
    r.add_samples(v, t)
    r.add_state(902.0, "2026-10-15T00:00:00",
                {"attention": .6, "arousal": .4, "load": .3, "fatigue": .2,
                 "relaxation": .7},
                confidence=.55, artifact_level=.1, contact="good (4/4 electrodes OK)",
                source="th21a", seq=1)
    r.event("unit_mark", {"k": "v"})
    rec = r.close()

    assert rec.n_samples == 500 and rec.n_state_rows == 1 and rec.n_events == 1
    d = tmp_path / "rec1"
    csv_rows = (d / "eeg.csv").read_text().strip().splitlines()
    assert csv_rows[0] == "sample,t_mono,t_rel_s,value"
    assert len(csv_rows) - 1 == 500, len(csv_rows)

    s = Session.load(d)                      # visible to the rest of the pipeline
    assert s.eeg.shape == (500,) and s.t.shape == (500,)
    assert abs(float(s.eeg[0]) - float(v[0])) < 1e-4
    assert len(s.events) == 1, "a monitor marker must reach the session timeline"

    saved = json.loads((d / "monitor.json").read_text())
    assert saved["provenance"]["device"] == "th21a"
    assert saved["provenance"]["is_live_human"] is True


def test_monitor_recording_says_so_when_it_is_synthetic(tmp_path):
    """The mistake this project refuses to make, asserted on the file itself."""
    from neurobridge.agent.link import Provenance
    from neurobridge.monitor import MonitorRecorder

    r = MonitorRecorder(out_dir=tmp_path, session_id="synth",
                        provenance=Provenance(device="fake", requested_device="auto",
                                              mode="live"))
    r.close()
    saved = json.loads((tmp_path / "synth" / "monitor.json").read_text())
    assert saved["provenance"]["is_synthetic"] is True
    assert saved["provenance"]["is_live_human"] is False
    assert "SYNTHETIC" in saved["banner"]


def test_monitor_replay_reads_a_native_session(tmp_path):
    from neurobridge.monitor import ReplaySource
    sess = _tiny_session(tmp_path)
    src = ReplaySource(sess, state_hz=2.0, speed=20.0)
    src.open()
    assert src.fs == 250.0 and src.position[1] == pytest.approx(6.0, abs=0.1)
    got = snaps = 0
    t0 = time.time()
    while time.time() - t0 < 1.5 and (got < 1500):
        rd = src.poll(0.01)
        got += rd.n_samples
        snaps += 1 if rd.snapshot else 0
    src.close()
    assert got == 1500, got
    assert snaps > 0, "replay produced no state snapshots"


def test_monitor_replay_accepts_its_own_csv(tmp_path):
    """A monitor eeg.csv is a first-class replay source, not just the native dir."""
    import numpy as np
    from neurobridge.monitor import MonitorRecorder, ReplaySource

    r = MonitorRecorder(out_dir=tmp_path, session_id="csvsrc",
                        meta={"fs": 250.0}, native_enabled=False)
    t = np.arange(300) / 250.0 + 700.0
    r.add_samples(np.ones(300) * 5.0, t)
    r.close()
    csv_path = tmp_path / "csvsrc" / "eeg.csv"

    src = ReplaySource(csv_path, speed=20.0)
    src.open()
    got = 0
    t0 = time.time()
    while time.time() - t0 < 1.5 and got < 300:
        got += src.poll(0.01).n_samples
    src.close()
    assert got == 300, got


def test_monitor_replay_transport_controls(tmp_path):
    from neurobridge.monitor import ReplaySource
    src = ReplaySource(_tiny_session(tmp_path), state_hz=2.0)
    src.open()
    src.pause()
    assert not src.playing
    src.set_speed(4.0)
    src.seek(3.0)
    pos, dur = src.position
    assert pos == pytest.approx(3.0, abs=0.1), (pos, dur)
    src.restart()
    assert src.playing and src.position[0] == pytest.approx(0.0, abs=0.1)


def test_bridge_sample_cursor_reports_a_gap_instead_of_splicing():
    """Falling behind must be visible: a silent splice invents a timeline."""
    import numpy as np
    from neurobridge.agent.bridge import BridgeState

    bs = BridgeState(device="fake", mode="live")
    t = np.arange(100) / 250.0 + 1000.0
    bs.estimator.push(np.arange(100.0), t)

    v, tt, gap = bs.samples_since(1000.0)          # exactly at the start
    assert v.size == 99 and gap is False, (v.size, gap)

    v2, t2, gap2 = bs.samples_since(0.0)           # far behind -> a real gap
    assert gap2 is True and v2.size == 100

    v3, t3, gap3 = bs.samples_since(float(t[-1]))
    assert v3.size == 0 and gap3 is False


def test_monitor_replay_reports_the_original_source(tmp_path):
    """A replay of generated data must never be described as real EEG.

    A replay looks identical to a live capture on screen, so the only honest
    description comes from the recording's own metadata.
    """
    import numpy as np
    from neurobridge.monitor import ReplaySource
    from neurobridge.store import SessionWriter

    def make(sid, source):
        w = SessionWriter(session_id=sid, root=tmp_path,
                          meta={"fs": 250.0, "device": {"source": source}})
        t = np.arange(300) / 250.0 + 100.0
        w.add_samples(np.arange(300, dtype=np.float32), t)
        w.close()
        return tmp_path / sid

    s = ReplaySource(make("syn", "fake"))
    s.open()
    assert "SYNTHETIC" in s.provenance().banner()
    s.close()

    s = ReplaySource(make("real", "th21a"))
    s.open()
    banner = s.provenance().banner()
    assert "th21a" in banner and "SYNTHETIC" not in banner, banner
    s.close()


def test_push_returns_the_timestamps_that_were_actually_stored():
    """A recorder must use the stored times, not the raw packet times.

    BLE packets overlap, so the ring nudges them forward to keep the buffer
    monotonic. Recording the un-nudged values yields a non-monotonic CSV whose
    timeline differs from the one the estimator analysed - which is what a
    25-minute live capture actually produced before this was fixed.
    """
    import numpy as np
    from neurobridge.agent import StateEstimator

    est = StateEstimator()
    t1 = np.arange(10) / 250.0 + 100.0
    stored1 = est.push(np.ones(10), t1)
    assert stored1.shape == (10,)
    assert np.allclose(stored1, t1)

    # a packet whose estimated times begin before the last stored time
    t2 = np.arange(10) / 250.0 + 100.0 + 0.02
    stored2 = est.push(np.ones(10), t2)
    assert stored2[0] > stored1[-1], (stored2[0], stored1[-1])
    assert bool((np.diff(stored2) > 0).all())

    joined = np.concatenate([stored1, stored2])
    assert bool((np.diff(joined) > 0).all()), "stored timeline is not monotonic"
