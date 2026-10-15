"""Agent-side link to the NeuroBridge state bridge (``nb serve``).

Why this module exists
----------------------
``bridge.py`` is the *server*: it publishes one honest frontal-EEG state stream
and never blocks on a slow consumer. An LLM agent is a different kind of
consumer from a GUI, and two properties of that difference shape this client:

* **A single ``GET /state`` is not monitoring.** Agent turns are seconds to
  minutes apart, so the useful question is rarely "what is the value *now*" but
  "what happened *while we were talking*". :func:`record` therefore streams
  payloads into a JSONL log (including while the agent is not running), and
  :func:`summarize` reduces a window of that log to trends and caveats.
* **Provenance has to be established before any number is rendered.** The bridge
  can serve three very different things - a live person, a replayed recording, or
  a synthetic generator - and ``/state``'s own ``source`` field carries the
  *requested* device alias (e.g. ``"auto"``), not what is actually producing
  samples. Ground truth is taken from ``GET /health``, where ``device`` is the
  real source name and ``requested_device`` is the alias the operator typed.

The rule this module enforces, in the bridge's own words: **never present
synthetic state to the human as their own.** Concretely that means every value
this module returns travels with a :class:`Provenance`, and
:func:`summarize` computes the caveats (synthetic source, no baseline, low
confidence) rather than leaving them to the caller to remember.

Nothing here opens the device or imports the web stack at import time; this is a
plain HTTP client that works against an already-running ``nb serve``.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..config import LOG_DIR, get_settings

__all__ = [
    "BridgeClient", "BridgeError", "BridgeUnreachable", "LinkError", "Provenance",
    "STATE_FIELDS", "SYNTHETIC_DEVICES", "default_log_path", "load_log",
    "parse_duration", "probe_liveness", "record", "summarize",
]

#: The 0..1 fields the bridge publishes. ``valence`` is deliberately absent: it
#: is always ``null`` on a single-channel montage and must never be rendered.
STATE_FIELDS = ("attention", "arousal", "load", "fatigue", "relaxation")

#: Device names that are *not* a person. ``replay`` is handled separately because
#: a replay is real recorded data, just not the live person.
SYNTHETIC_DEVICES = frozenset({"fake", "synthetic", "sim", "null", ""})

#: Derivations that mean "this field carries no information about anyone yet".
#: ``uncalibrated_neutral`` pins every field at 0.5 until a baseline exists, so
#: rendering it as a measurement is the exact failure this module exists to stop.
UNINFORMATIVE_DERIVATIONS = frozenset({"uncalibrated_neutral", "no_data", ""})


class LinkError(RuntimeError):
    """Base class for every link failure."""


class BridgeUnreachable(LinkError):
    """``nb serve`` is not answering on this host/port."""


class BridgeError(LinkError):
    """The bridge answered, but with an error payload or an unusable body."""


# --------------------------------------------------------------------- duration
def parse_duration(text: str | float | int | None) -> float | None:
    """``'90s'``/``'10m'``/``'1.5h'``/``600`` -> seconds. ``None`` -> ``None``.

    Used for ``--last``/``--seconds``, which humans type in mixed units. An
    unparseable string returns ``None`` rather than 0 so a caller can tell
    "no window given" from "an empty window".
    """
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    s = str(text).strip().lower()
    if not s:
        return None
    mult = {"s": 1.0, "m": 60.0, "h": 3600.0}
    if s[-1] in mult:
        try:
            return float(s[:-1]) * mult[s[-1]]
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------------------------- provenance
@dataclass(frozen=True)
class Provenance:
    """What is *actually* producing the samples behind a state payload.

    Built from ``GET /health``: ``device`` is the resolved source (``th21a``,
    ``fake``, ``replay:<session>``) while ``requested_device`` is the alias on the
    command line (``auto``). The whole point of this dataclass is that those two
    can differ, and that only the first one is safe to describe to a human.
    """

    device: str = "unknown"
    requested_device: str = ""
    mode: str = "unknown"
    version: str = ""
    session: str | None = None
    started_wall: str = ""
    #: Whether a personal baseline exists. ``None`` means /health did not say.
    #: This decides whether the 0..1 fields carry information at all, so the
    #: banner must not hard-code either answer.
    calibrated: bool | None = None
    #: For a replay: the device that made the original recording (``th21a``,
    #: ``fake``, or ``""`` when the recording did not say). Without this a replay
    #: of a synthetic capture is indistinguishable from a replay of a real
    #: session, and both would be described as "real EEG".
    original_device: str = ""

    @property
    def is_synthetic(self) -> bool:
        """True when the numbers are generated, never measured from a person."""
        d = (self.device or "").lower()
        return (not d.startswith("replay")) and (d in SYNTHETIC_DEVICES or d == "auto")

    @property
    def is_replay(self) -> bool:
        """True when real EEG is being replayed - real data, wrong moment."""
        return (self.device or "").lower().startswith("replay") or self.mode == "replay"

    @property
    def is_live_human(self) -> bool:
        """True only for a live, non-synthetic source: the one case worth acting on."""
        return (not self.is_synthetic) and (not self.is_replay) and self.mode == "live"

    @property
    def label(self) -> str:
        if self.is_synthetic:
            return f"SYNTHETIC ({self.device or 'unknown'})"
        if self.is_replay:
            return f"REPLAY ({self.device})"
        return f"LIVE headband ({self.device})"

    def banner(self) -> str:
        """One line a human must be shown before any state number is believed."""
        if self.is_synthetic:
            return ("SYNTHETIC SOURCE - these numbers are generated, not measured. "
                    "They do not describe any person's brain.")
        if self.is_replay:
            od = (self.original_device or "").strip()
            if od and od.lower() not in SYNTHETIC_DEVICES:
                return (f"REPLAYED RECORDING of the {od} headband - real EEG from an "
                        "earlier session, not the live person.")
            if od:
                return ("REPLAYED SYNTHETIC RECORDING - generated data, played back. "
                        "It never described a person.")
            return ("REPLAYED RECORDING - the original source was not recorded, so "
                    "treat this as historical data and never as this person's brain.")
        if self.is_live_human:
            if self.calibrated is True:
                state = "calibrated: the fields are relative to your own baseline"
            elif self.calibrated is False:
                state = ("NOT calibrated yet, so every 0..1 field is pinned at 0.5 "
                         "and carries no information (run `nb link calibrate`)")
            else:
                state = "baseline state not reported by the bridge"
            return (f"LIVE from the {self.device} headband, one dry frontal "
                    f"electrode; {state}.")
        return (f"UNKNOWN SOURCE ({self.device!r}) - do not present these numbers "
                "as anyone's brain state.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "requested_device": self.requested_device,
            "mode": self.mode,
            "version": self.version,
            "session": self.session,
            "started_wall": self.started_wall,
            "calibrated": self.calibrated,
            "original_device": self.original_device,
            "is_synthetic": self.is_synthetic,
            "is_replay": self.is_replay,
            "is_live_human": self.is_live_human,
            "label": self.label,
        }

    @classmethod
    def from_health(cls, health: dict[str, Any]) -> "Provenance":
        cal = health.get("calibrated")
        return cls(
            device=str(health.get("device") or "unknown"),
            requested_device=str(health.get("requested_device") or ""),
            mode=str(health.get("mode") or "unknown"),
            version=str(health.get("version") or ""),
            session=(str(health["session"]) if health.get("session") else None),
            started_wall=str(health.get("started_wall") or ""),
            calibrated=(None if cal is None else bool(cal)),
        )


# ----------------------------------------------------------------------- client
def default_base_url() -> str:
    s = get_settings()
    return f"http://{s.bridge_host}:{s.bridge_port}"


def default_log_path(when: float | None = None) -> Path:
    """``data/logs/agent-link-YYYYMMDD.jsonl`` - one file per day, append-only."""
    stamp = time.strftime("%Y%m%d", time.localtime(when or time.time()))
    return Path(LOG_DIR) / f"agent-link-{stamp}.jsonl"


def iter_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse an SSE line stream into the JSON objects carried by ``data:`` fields.

    Comment lines (``: heartbeat``) are ignored; a malformed payload is skipped
    rather than raised, because dropping one frame is better than tearing down a
    monitor that is otherwise working.
    """
    buf: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n") if isinstance(raw, str) else str(raw)
        if line == "":
            if buf:
                blob = "\n".join(buf)
                buf = []
                try:
                    obj = json.loads(blob)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
    if buf:                                     # stream ended without a blank line
        try:
            obj = json.loads("\n".join(buf))
            if isinstance(obj, dict):
                yield obj
        except Exception:
            pass


class BridgeClient:
    """Thin HTTP/SSE client for a running ``nb serve``.

    Deliberately small and synchronous: the agent-facing use cases are "ask once"
    and "stream in a background thread", and both are clearer without async in
    the caller's way.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 8.0) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.timeout = float(timeout)

    # -- plumbing ----------------------------------------------------------
    def _httpx(self):
        try:
            import httpx
        except Exception as exc:                       # pragma: no cover - dep missing
            raise LinkError(
                "the link needs httpx (a declared NeuroBridge dependency): "
                f"{exc}") from exc
        return httpx

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get_json(self, path: str, params: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        """GET one JSON object, mapping every transport failure to a LinkError."""
        httpx = self._httpx()
        url = self._url(path)
        try:
            r = httpx.get(url, params=params, timeout=timeout or self.timeout)
        except Exception as exc:
            raise BridgeUnreachable(
                f"no bridge at {self.base_url} ({type(exc).__name__}: {exc}). "
                f"Start one with `./bin/nb serve`.") from exc
        return self._decode(r, url)

    def post_json(self, path: str, payload: dict[str, Any],
                  timeout: float | None = None) -> dict[str, Any]:
        httpx = self._httpx()
        url = self._url(path)
        try:
            r = httpx.post(url, json=payload, timeout=timeout or self.timeout)
        except Exception as exc:
            raise BridgeUnreachable(
                f"no bridge at {self.base_url} ({type(exc).__name__}: {exc}).") from exc
        return self._decode(r, url)

    @staticmethod
    def _decode(r: Any, url: str) -> dict[str, Any]:
        try:
            body = r.json()
        except Exception as exc:
            raise BridgeError(
                f"{url} returned non-JSON (HTTP {r.status_code}): "
                f"{r.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise BridgeError(f"{url} returned {type(body).__name__}, expected object")
        if r.status_code >= 400 or body.get("ok") is False:
            detail = body.get("detail") or body.get("error") or r.status_code
            raise BridgeError(f"{url} -> {detail}")
        return body

    # -- endpoints ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self.get_json("/health")

    def provenance(self) -> Provenance:
        """Ground truth about the source, from ``/health`` (never ``/state``)."""
        return Provenance.from_health(self.health())

    def state(self, explain: bool = False) -> dict[str, Any]:
        return self.get_json("/state", params={"explain": str(bool(explain)).lower()})

    def schema(self) -> dict[str, Any]:
        return self.get_json("/schema")

    def candidate(self) -> dict[str, Any]:
        return self.get_json("/candidate")

    def metrics(self) -> dict[str, Any]:
        return self.get_json("/metrics")

    def events(self, since: float | None = None, limit: int = 100) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(limit)}
        if since is not None:
            params["since"] = float(since)
        return self.get_json("/events", params=params)

    def mark(self, name: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write an agent-side marker into the shared EEG timeline.

        The protocol asks the agent to bracket its own actions with markers so a
        later reader can align "what the agent did" with "what the signal did".
        """
        return self.post_json("/mark", {"name": str(name), "meta": dict(meta or {})})

    def confirm(self, candidate_seq: int, participant_response: str,
                note: str = "") -> dict[str, Any]:
        """Record the *participant's* answer about a candidate. Never invented."""
        return self.post_json("/confirm", {
            "candidate_seq": int(candidate_seq),
            "participant_response": str(participant_response),
            "note": str(note or ""),
        })

    def calibrate(self, force: bool = False) -> dict[str, Any]:
        """Ask the bridge to capture the personal baseline now.

        Deliberately does **not** go through :meth:`_decode`. "Not enough data
        yet" is a normal answer (HTTP 409) rather than a transport failure, and a
        caller needs to tell those apart: one means "wait and retry", the other
        means "the bridge is unreachable". So the payload is returned as-is and
        the caller decides.
        """
        httpx = self._httpx()
        url = self._url("/calibrate")
        try:
            r = httpx.post(url, json={"force": bool(force)}, timeout=self.timeout)
        except Exception as exc:
            raise BridgeUnreachable(
                f"no bridge at {self.base_url} ({type(exc).__name__}: {exc}).") from exc
        try:
            body = r.json()
        except Exception as exc:
            raise BridgeError(
                f"{url} returned non-JSON (HTTP {r.status_code}): "
                f"{r.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise BridgeError(f"{url} returned {type(body).__name__}, expected object")
        if r.status_code == 404 or ("status" not in body and body.get("detail")):
            # A bridge started before /calibrate existed answers 404 here. Saying
            # so beats rendering an empty "no baseline yet", which would send the
            # operator off to wait for data that can never produce a baseline.
            raise BridgeError(
                f"{url} -> this bridge does not expose /calibrate "
                f"(HTTP {r.status_code}: {body.get('detail')!r}). It is running an "
                "older build; restart it with `./bin/nb-serve` to pick up the "
                "current code.")
        return body

    # -- streaming ---------------------------------------------------------
    def stream(self, stop: threading.Event | None = None,
               on_open: Callable[[Provenance], None] | None = None,
               ) -> Iterator[dict[str, Any]]:
        """Yield state payloads from ``/state/stream`` until ``stop`` is set.

        Each yielded dict is the bridge payload plus three client-side keys:

        * ``_recv_mono`` / ``_recv_wall`` - when *this process* received it, which
          is the only timestamp a client can trust for its own gap detection
          (the server's ``t_mono`` is another process's ``perf_counter`` origin).
        * ``_device`` - the resolved source at open time, so a log written now is
          still self-describing when read later.
        """
        try:
            prov = self.provenance()
        except LinkError:
            raise
        if on_open is not None:
            on_open(prov)
        yield from self.stream_json("/state/stream", device=prov.device, stop=stop)

    def stream_json(self, path: str, params: dict[str, Any] | None = None,
                    device: str = "", stop: threading.Event | None = None,
                    ) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE JSON objects from any streaming endpoint.

        Shared by the state stream and the raw ``/samples/stream`` so both get
        the same gap/termination behaviour and the same client-side receive
        stamps rather than two subtly different parsers.
        """
        httpx = self._httpx()
        url = self._url(path)
        timeout = httpx.Timeout(connect=self.timeout, read=None,
                                write=self.timeout, pool=self.timeout)
        try:
            with httpx.stream("GET", url, params=params, timeout=timeout) as r:
                if r.status_code >= 400:
                    raise BridgeError(f"{url} -> HTTP {r.status_code}")
                for obj in iter_sse(r.iter_lines()):
                    if stop is not None and stop.is_set():
                        return
                    obj["_recv_mono"] = time.perf_counter()
                    obj["_recv_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    obj["_device"] = device
                    yield obj
        except (BridgeError, LinkError):
            raise
        except Exception as exc:
            raise BridgeUnreachable(
                f"stream {url} ended: {type(exc).__name__}: {exc}") from exc


# ------------------------------------------------------------------ liveness
def probe_liveness(client: BridgeClient, gap_s: float = 0.7) -> dict[str, Any]:
    """Read ``/state`` twice and report whether the server clock is advancing.

    A frozen bridge still answers HTTP 200, so "it responded" is not evidence of
    a live signal. ``t_mono`` is the server's own monotonic clock; if it does not
    move between two reads the estimator is wedged and the last value must not be
    trusted - which is exactly the staleness rule in ``protocol.md``.
    """
    a = client.state()
    time.sleep(max(0.05, float(gap_s)))
    b = client.state()
    dt = float(b.get("t_mono", 0.0)) - float(a.get("t_mono", 0.0))
    dseq = int(b.get("seq", 0)) - int(a.get("seq", 0))
    return {"advanced": dt > 0.0, "dt_mono": round(dt, 4), "seq_delta": dseq,
            "first": a, "second": b}


# ----------------------------------------------------------------------- record
def record(client: BridgeClient, out_path: Path | str | None = None,
           seconds: float | None = None, stop: threading.Event | None = None,
           on_payload: Callable[[dict[str, Any]], None] | None = None,
           on_open: Callable[[Provenance], None] | None = None) -> dict[str, Any]:
    """Stream state into a JSONL log; return a receipt describing what was logged.

    The first line is a self-describing header carrying the :class:`Provenance`.
    That matters because a log is read hours later, possibly by a different agent:
    without the header, replayed or synthetic data is indistinguishable from a
    real recording, which is the one mistake this project refuses to make.
    """
    path = Path(out_path) if out_path else default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    first_wall = last_wall = ""
    prov_box: dict[str, Any] = {}
    t_end = None if seconds is None else time.perf_counter() + float(seconds)
    stop = stop or threading.Event()

    def _capture(p: Provenance) -> None:
        prov_box.update(p.as_dict())
        if on_open is not None:
            on_open(p)

    with path.open("a", buffering=1) as fh:
        for payload in client.stream(stop=stop, on_open=_capture):
            if not prov_box:                              # header guard
                prov_box.update(Provenance.from_health(client.health()).as_dict())
            if n == 0:
                fh.write(json.dumps({
                    "_type": "header", "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "base_url": client.base_url, "provenance": prov_box,
                    "note": ("payloads follow, one JSON object per line; fields "
                             "prefixed with _ are added by the link client"),
                }) + "\n")
            fh.write(json.dumps(payload, default=str) + "\n")
            n += 1
            first_wall = first_wall or str(payload.get("_recv_wall") or "")
            last_wall = str(payload.get("_recv_wall") or "")
            if on_payload is not None:
                on_payload(payload)
            if t_end is not None and time.perf_counter() >= t_end:
                break
    return {"ok": n > 0, "path": str(path), "n_payloads": n,
            "first_wall": first_wall, "last_wall": last_wall,
            "provenance": prov_box}


# ------------------------------------------------------------------- log reading
def load_log(path: Path | str, last_s: float | None = None) -> tuple[dict[str, Any], list[dict]]:
    """Read a link JSONL log -> ``(header, payloads)``.

    ``last_s`` keeps only payloads received within the last N seconds (relative to
    the newest payload in the file, not to now), so ``--last 10m`` means "the last
    ten minutes we actually recorded" even if the recorder has since stopped. A
    malformed line is skipped: a truncated tail is normal for a live log.
    """
    p = Path(path)
    if not p.exists():
        raise LinkError(f"no log at {p}")
    header: dict[str, Any] = {}
    rows: list[dict] = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("_type") == "header":
            header = obj
        else:
            rows.append(obj)
    if last_s is not None and rows:
        newest = max(float(r.get("_recv_mono") or 0.0) for r in rows)
        rows = [r for r in rows if float(r.get("_recv_mono") or 0.0) >= newest - float(last_s)]
    return header, rows


# ------------------------------------------------------------------- summarising
def _nums(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append(float(v))
    return out


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4), "last": round(vals[-1], 4)}


def _trend(vals: list[float], thresh: float = 0.05) -> dict[str, Any]:
    """Compare the first and last third of a window; ignore single-sample noise.

    The protocol tells an agent to act on trends over >= 2 s, never one sample.
    Splitting the window and requiring a real delta is the cheapest honest way to
    express that: two correlated 0.02 wiggles are reported as ``flat``.
    """
    if len(vals) < 3:
        return {"direction": "unknown", "delta": None}
    k = max(1, len(vals) // 3)
    a = sum(vals[:k]) / k
    b = sum(vals[-k:]) / k
    d = b - a
    direction = "flat"
    if d >= thresh:
        direction = "rising"
    elif d <= -thresh:
        direction = "falling"
    return {"direction": direction, "delta": round(d, 4),
            "first_third_mean": round(a, 4), "last_third_mean": round(b, 4)}


def summarize(rows: list[dict], header: dict[str, Any] | None = None,
              expected_hz: float | None = None) -> dict[str, Any]:
    """Reduce logged payloads to trends plus the caveats an agent must not drop.

    Returns a plain dict (JSON-serialisable) so the caller decides presentation.
    The ``limitations`` list is computed, not boilerplate: it names the specific
    reasons *this* window cannot support a claim about a person.
    """
    header = dict(header or {})
    prov_d = dict(header.get("provenance") or {})
    if not prov_d and rows:
        prov_d = Provenance(device=str(rows[-1].get("_device") or "unknown")).as_dict()
    prov = Provenance(
        device=str(prov_d.get("device") or "unknown"),
        requested_device=str(prov_d.get("requested_device") or ""),
        mode=str(prov_d.get("mode") or "unknown"),
        version=str(prov_d.get("version") or ""),
        session=prov_d.get("session"),
        calibrated=(None if prov_d.get("calibrated") is None
                    else bool(prov_d.get("calibrated"))),
    )

    out: dict[str, Any] = {
        "ok": bool(rows), "n_payloads": len(rows),
        "provenance": prov.as_dict(), "provenance_label": prov.label,
        "synthetic": prov.is_synthetic, "replay": prov.is_replay,
    }
    if not rows:
        out["limitations"] = ["empty window: no payloads were recorded"]
        return out

    recv = [float(r.get("_recv_mono") or 0.0) for r in rows]
    duration = max(recv) - min(recv) if len(recv) > 1 else 0.0
    walls = [str(r.get("_recv_wall") or "") for r in rows]
    # inter-arrival gaps: the honest measure of whether the monitor actually ran
    gaps = [recv[i] - recv[i - 1] for i in range(1, len(recv))]
    max_gap = max(gaps) if gaps else 0.0
    out["window"] = {"first_wall": walls[0], "last_wall": walls[-1],
                     "duration_s": round(duration, 2)}
    out["rate_hz"] = round(len(rows) / duration, 3) if duration > 0 else None
    if expected_hz:
        out["expected_hz"] = float(expected_hz)

    seqs = [int(r["seq"]) for r in rows if isinstance(r.get("seq"), int)]
    dropped = 0
    gap_events = 0
    for i in range(1, len(seqs)):
        d = seqs[i] - seqs[i - 1]
        if d > 1:
            dropped += d - 1
            gap_events += 1
    out["seq"] = {"first": seqs[0] if seqs else None,
                  "last": seqs[-1] if seqs else None,
                  "gap_events": gap_events, "dropped": dropped}
    out["staleness"] = {"max_gap_s": round(max_gap, 3),
                        "gap_events_gt_3s": sum(1 for g in gaps if g > 3.0)}

    # per-field stats, with the derivation that decides whether to trust them
    fields: dict[str, Any] = {}
    any_informative = False
    for name in STATE_FIELDS:
        vals = _nums(rows, name)
        st = _stats(vals)
        derivs: set[str] = set()
        for r in rows:
            mi = (r.get("model_info") or {}).get("fields") or {}
            d = (mi.get(name) or {}).get("derivation")
            if d:
                derivs.add(str(d))
        informative = bool(derivs) and not derivs.issubset(UNINFORMATIVE_DERIVATIONS)
        if informative and vals:
            any_informative = True
        st.update({"derivation": sorted(derivs) or ["unknown"],
                   "informative": informative})
        if vals:
            st["trend"] = _trend(vals)
        fields[name] = st
    out["fields"] = fields
    out["informative"] = any_informative

    out["confidence"] = _stats(_nums(rows, "confidence"))
    out["artifact_level"] = _stats(_nums(rows, "artifact_level"))
    out["contact"] = sorted({str(r.get("contact")) for r in rows if r.get("contact")})
    out["sources_seen"] = sorted({str(r.get("source")) for r in rows if r.get("source")})

    by_kind: dict[str, int] = {}
    for r in rows:
        for it in (r.get("intents") or []):
            if isinstance(it, dict):
                k = str(it.get("intent") or "?")
                by_kind[k] = by_kind.get(k, 0) + 1
    out["intents"] = {"total": sum(by_kind.values()), "by_kind": by_kind}

    # ------------------------------------------------ computed honesty caveats
    lim: list[str] = []
    if prov.is_synthetic:
        lim.append("SYNTHETIC source: nothing in this window was measured from a "
                   "person. Never present it as anyone's brain state.")
    if prov.is_replay:
        lim.append("REPLAYED recording: real EEG, but not this conversation.")
    if not any_informative:
        lim.append("No field is informative: with no personal baseline every field "
                   "is pinned at 0.5 (uncalibrated_neutral) and carries no "
                   "information about anyone. Report signal quality instead.")
    art = out["artifact_level"]
    if art.get("n") and art["mean"] > 0.3:
        lim.append(f"Artifact level averages {art['mean']:.2f} (0 = clean, 1 = all "
                   "spoiled): a large share of this window is artefact rather than "
                   "signal, so the fields above sit on a noisy floor.")
    # A field that traverses most of 0..1 inside one window is noise, not a state:
    # nobody's attention swings from floor to ceiling in under a minute. Say so
    # rather than letting the trend column imply a real change.
    swing = [(n, st) for n, st in (out["fields"] or {}).items()
             if st.get("informative") and st.get("n")
             and (float(st.get("max", 0.0)) - float(st.get("min", 0.0))) > 0.5]
    if swing:
        detail = ", ".join(f"{n} {st['min']:.2f}-{st['max']:.2f}" for n, st in swing[:4])
        lim.append("Fields swing across most of the 0..1 range within this window "
                   f"({detail}). Treat single values as noise and require a "
                   "sustained movement before believing a change.")
    conf = out["confidence"]
    if conf.get("n"):
        if conf["mean"] < 0.2:
            lim.append(f"Mean confidence {conf['mean']:.2f} < 0.2: the capture is "
                       "saturated or off-head; every field above is meaningless.")
        elif conf["mean"] < 0.4:
            lim.append(f"Mean confidence {conf['mean']:.2f} < 0.4: below the "
                       "protocol's action gate. Do not act on state; ask a question.")
    if out["staleness"]["gap_events_gt_3s"]:
        lim.append(f"{out['staleness']['gap_events_gt_3s']} gap(s) over the 3 s "
                   "staleness limit: the stream stopped and restarted, so treat "
                   "the window as discontinuous.")
    if not lim:
        lim.append("No structural caveat detected in this window. The fields remain "
                   "baseline-relative heuristics, not measurements of thought.")
    out["limitations"] = lim
    return out
