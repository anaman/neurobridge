#!/usr/bin/env python3
"""Import raw TH21A BLE packets (the format captured by the ``ble/`` tools) into
a NeuroBridge session directory.

    .venv/bin/python scripts/import_ble_packets.py ../ble/session_packets.txt --test-id T003

The headband's packets are ASCII-tagged hex lines. Files may contain
``# PHASE NAME`` headers, which become class markers: ``# EYES OPEN`` maps to
the ``eyes_open`` class, ``# EYES CLOSED`` to ``eyes_closed``, anything else to
a lowercased snake_case label. That makes an existing capture immediately
usable by the ordinary pipeline (``nb eval T003``) and comparable with
``device/quality.py``'s independent alpha-reactivity check.

Timestamps are synthesised at the nominal 250 Hz: the device does not timestamp
its samples, so for a file import the relative timing is all we can honestly
claim, and that is enough for band-power and quality analysis (not for ERP
latencies -- use a live session for those).
"""
from __future__ import annotations

import argparse
import json
import shutil
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurobridge import store  # noqa: E402
from neurobridge.config import SESSION_DIR  # noqa: E402
from neurobridge.device.quality import alpha_reactivity, assess  # noqa: E402
from neurobridge.device.th21a import HEADER_BYTES, parse_packet  # noqa: E402

PHASE_ALIASES = {
    "eyes open": "eyes_open", "eyes_open": "eyes_open", "open": "eyes_open",
    "eyes closed": "eyes_closed", "eyes_closed": "eyes_closed", "closed": "eyes_closed",
    "rest": "rest", "baseline": "baseline",
}

#: Marker codes for a phase-only import. They deliberately sit outside every
#: catalog tier band (which use 1000-1799) so an imported session can never be
#: confused with a real protocol run.
IMPORT_CODES = {"block_start": 9000, "block_end": 9001, "stim_onset": 9002,
                "eyes_open": 9010, "eyes_closed": 9011, "rest": 9012,
                "baseline": 9013, "imported": 9014}


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return PHASE_ALIASES.get(s.replace("_", " "), PHASE_ALIASES.get(s, s or "imported"))


def read_packets(path: Path) -> list[tuple[str, bytes]]:
    """-> [(phase_name, packet_bytes)] preserving order, skipping comments/blanks."""
    out: list[tuple[str, bytes]] = []
    phase = "imported"
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            phase = slug(line.lstrip("#"))
            continue
        try:
            out.append((phase, bytes.fromhex(line)))
        except ValueError:
            continue                      # tolerate any stray non-hex line
    return out


def decode(packets: list[tuple[str, bytes]], fs: float = 250.0):
    """-> (values, times, runs) using NeuroBridge's parser.

    ``runs`` maps a phase name to the list of *contiguous* sample spans carrying
    it. Consecutive packets with the same phase **must** be merged: the first
    version of this importer emitted one block per packet, so a 20 s eyes-open
    phase became 503 one-packet blocks and leave-one-block-out CV was
    meaningless. Grouping is what makes the imported session analysable.
    """
    values: list[np.ndarray] = []
    phases: list[str] = []
    for phase, pkt in packets:
        if not pkt.startswith(b"DATA\x00\x02"):
            continue                      # control/ack replies are not EEG
        v, _idx = parse_packet(pkt)
        if v.size:
            values.append(v.astype(np.float64))
            phases.append(phase)
    if not values:
        return np.zeros(0), np.zeros(0), {}
    v = np.concatenate(values)
    t = np.arange(v.size, dtype=np.float64) / fs

    runs: dict[str, list[tuple[int, int]]] = {}
    run_phase: str | None = None
    run_start = 0
    pos = 0
    for arr, ph in zip(values, phases):
        if ph != run_phase:
            if run_phase is not None:
                runs.setdefault(run_phase, []).append((run_start, pos))
            run_phase, run_start = ph, pos
        pos += arr.size
    if run_phase is not None:
        runs.setdefault(run_phase, []).append((run_start, pos))
    return v, t, runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packet_file", type=Path)
    ap.add_argument("--test-id", default="T003")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--keep-session", action="store_true")
    a = ap.parse_args(argv)

    packets = read_packets(a.packet_file)
    values, times, slices = decode(packets, a.fs)
    if values.size == 0:
        print(f"no EEG packets found in {a.packet_file}")
        return 1
    n_runs = sum(len(v) for v in slices.values())
    print(f"{a.packet_file.name}: {len(packets)} packet lines, {n_runs} contiguous "
          f"phase run(s), {values.size} samples ({values.size/a.fs:.1f}s at {a.fs:g} Hz)")
    print("runs per phase: " + ", ".join(
        f"{k} x{len(v)}" for k, v in slices.items()))

    sid = a.session_id or f"import-{a.packet_file.stem}-{a.test_id}"
    target = SESSION_DIR / sid
    # Re-importing must not append a second copy of the timeline to an existing
    # session, so an earlier import of the same file is replaced -- but only if
    # it really is one of ours.
    if target.exists():
        try:
            prev = json.loads((target / "meta.json").read_text())
        except Exception:
            prev = {}
        if prev.get("imported"):
            shutil.rmtree(target)
            print(f"replaced previous import at {target.name}")
        else:
            print(f"refusing to overwrite {target} (not an import); pass --session-id")
            return 1

    w = store.SessionWriter(session_id=sid, meta={
        "session_id": sid, "device": "th21a", "fs": a.fs, "channel_name": "Fpz",
        "montage": "single dry frontal electrode (Fpz)",
        "imported_from": str(a.packet_file.resolve()),
        "imported": True,
        # per the imported external protocol: a packet-file import has no measured
        # stimulus onset, so its timing is coarse for band power and unverified
        # for ERP latencies. Never let it masquerade as a live session.
        "timing_quality": "coarse",
        "note": ("imported from raw BLE packets; timestamps synthesised at the "
                 "nominal rate, so band-power and quality are valid but ERP "
                 "latencies are not"),
        "spec_id": a.test_id,
    })
    w.event(store.MARKER_CODES["session_start"], name="session_start", test_id=a.test_id)
    w.add_samples(values, times)
    bi = 0
    for phase, spans in slices.items():
        for (i0, i1) in spans:
            w.event(IMPORT_CODES["block_start"], name="block_start", test_id=a.test_id,
                    block=bi, t_mono=float(times[i0]))
            w.event(IMPORT_CODES.get(phase, IMPORT_CODES["imported"]), name=phase,
                    test_id=a.test_id, block=bi, t_mono=float(times[i0]),
                    meta={"class": phase})
            w.event(IMPORT_CODES["stim_onset"], name="stim_onset", test_id=a.test_id,
                    block=bi, t_mono=float(times[i0]), meta={"class": phase})
            w.event(IMPORT_CODES["block_end"], name="block_end", test_id=a.test_id,
                    block=bi, t_mono=float(times[i1 - 1]))
            bi += 1
    w.event(store.MARKER_CODES["test_end"], name="test_end", test_id=a.test_id)
    w.event(store.MARKER_CODES["session_end"], name="session_end", test_id=a.test_id)

    # independent checks before writing the summary, so the session records them
    alpha_r = None
    if "eyes_open" in slices and "eyes_closed" in slices:
        def gather(ph):
            return np.concatenate([values[i0:i1] for i0, i1 in slices[ph]])
        alpha_r = alpha_reactivity(gather("eyes_open"), gather("eyes_closed"), a.fs)
    rep = assess(values, a.fs, alpha_ratio=(alpha_r["alpha_ratio"] if alpha_r else None))
    summary = {"imported": True, "quality_verdict": rep.verdict,
               "quality_score": rep.score, "quality_max": rep.max_score,
               "saturation_pct": rep.metrics.get("saturation_pct"),
               "drift_ratio": rep.metrics.get("drift_ratio"),
               "band_delta_pct": rep.metrics.get("band_delta_pct"),
               "timing_quality": "coarse"}
    if alpha_r is not None:
        summary["alpha_ratio_closed_open"] = round(alpha_r["alpha_ratio"], 3)
        summary["alpha_open_pct"] = round(alpha_r["alpha_open_pct"], 2)
        summary["alpha_closed_pct"] = round(alpha_r["alpha_closed_pct"], 2)
        print(f"\nalpha: open {alpha_r['alpha_open_pct']:.2f}%  "
              f"closed {alpha_r['alpha_closed_pct']:.2f}%  "
              f"ratio {alpha_r['alpha_ratio']:.2f}")
        print(f"       -> {alpha_r['text']}")
    print("\nquality (NeuroBridge assess()):")
    for line in rep.lines():
        print("  " + line)

    path = w.close(summary=summary)
    print(f"\nsession written: {path}")
    print(f"evaluate with:   ./bin/nb eval {a.test_id} --session {path.name}")
    if not a.keep_session:
        pass                              # kept by default: real data is precious
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
