"""NingNao / SereniBrain TH21A BLE EEG source.

Protocol recovered from the vendor SDK (see ``ble/README.md``):

* GATT ``8653000a-…`` custom service
    * ``8653000c-…``  write    command channel (host -> headset)
    * ``8653000b-…``  notify   data channel    (headset -> host)
* every command is an 8-byte ASCII-tagged frame beginning with ``CTRL``
* EEG packets: ``"DATA" 00 02 00 03 len 00 00 00`` (12 B header)
  followed by 10 x 7-byte samples ``[value:4B LE signed][index:2B BE][pad]``
  and a 10-byte trailer. 10 samples/packet x 25 packets/s = 250 Hz.

The BLE transport runs in its own thread with its own asyncio loop; the rest
of NeuroBridge stays synchronous and calls :meth:`poll`.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import numpy as np

from .base import EEGSource, Packet
from .quality import contact_from_selfcheck

#: The headband's BLE address. Each TH21A has its own *static random* address, so
#: there is no universal value to ship: leave it empty and the device is found by
#: name (see :func:`find_headband`), or set ``address:`` in ``neurobridge.yaml``
#: (or ``NB_ADDRESS``) to pin one unit. Find yours with ``nb doctor --scan``.
DEFAULT_ADDRESS = ""

#: Name prefix every TH21A advertises, used when no address is configured.
NAME_PREFIX = "TH21A"

RX_CHAR = "8653000c-43e6-47b7-9cb0-5fc21d4ae340"   # host -> device (write)
TX_CHAR = "8653000b-43e6-47b7-9cb0-5fc21d4ae340"   # device -> host (notify)

CTRL = b"CTRL"
CMD_INFO = CTRL + bytes([0, 3, 0, 5])
CMD_SELF_CHECK = CTRL + bytes([0, 4, 0, 1])
CMD_START = CTRL + bytes([0, 3, 0, 3])
CMD_STOP = CTRL + bytes([0, 3, 0, 4])

ACK_INFO = b"DATA\x00\x01\x00\x03"
ACK_SELF_CHECK = b"DATA\x00\x01\x00\x04"
ACK_STREAM = b"DATA\x00\x02"

SAMPLES_PER_PACKET = 10
PACKET_BYTES = 7
TRAILER_BYTES = 10
HEADER_BYTES = 12
NOMINAL_FS = 250.0


def parse_packet(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one ``DATA\\x00\\x02`` notification into (values, indices)."""
    body = payload[HEADER_BYTES:]
    n = max(0, (len(body) - TRAILER_BYTES) // PACKET_BYTES)
    vals = np.empty(n, dtype=np.int32)
    idxs = np.empty(n, dtype=np.int32)
    for i in range(n):
        g = body[i * PACKET_BYTES:(i + 1) * PACKET_BYTES]
        vals[i] = int.from_bytes(g[:4], "little", signed=True)
        idxs[i] = int.from_bytes(g[4:6], "big")
    return vals, idxs


async def find_headband(timeout: float = 15.0, address: str = DEFAULT_ADDRESS) -> Any:
    """Return the bleak device for the headband, or None.

    With an address configured the lookup is exact, and a sleeping band reports
    None rather than silently binding to a different unit. With no address, the
    scan falls back to matching ``NAME_PREFIX`` so a new user can power the band
    on and go, without first hunting for a MAC address that has no universal
    value anyway.
    """
    from bleak import BleakScanner
    if address:
        try:
            return await BleakScanner.find_device_by_address(address, timeout=timeout)
        except Exception:
            return None
    try:
        found = await BleakScanner.discover(timeout=timeout)
    except Exception:
        return None
    for dev in found or ():
        if str(getattr(dev, "name", "") or "").upper().startswith(NAME_PREFIX):
            return dev
    return None


class TH21A(EEGSource):
    """Live source for the headband."""

    name = "th21a"
    fs = NOMINAL_FS

    def __init__(self, address: str = DEFAULT_ADDRESS, fs: float = NOMINAL_FS,
                 connect_timeout_s: float = 30.0, handshake_tries: int = 12,
                 verbose: bool = True) -> None:
        self.address = address
        self.fs = float(fs)
        self.connect_timeout_s = float(connect_timeout_s)
        self.handshake_tries = int(handshake_tries)
        self.verbose = verbose

        self._q: "queue.Queue[Packet]" = queue.Queue(maxsize=20_000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self.contact = "unknown"
        self.device_info_hex: str = ""
        self.selfcheck_ascii: str = ""
        self.mtu: int = 0

        # diagnostics
        self.n_packets = 0
        self.n_samples = 0
        self.n_gaps = 0
        self.n_missing = 0
        self.packet_times: list[float] = []
        self._last_index: int | None = None
        self._first_packet_t: float | None = None

    # ------------------------------------------------------------ open
    def open(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="th21a-ble", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.connect_timeout_s + 20.0):
            raise TimeoutError("TH21A: timed out connecting; is the headband on?")
        if self._error:
            raise RuntimeError(f"TH21A: {self._error}")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None

    # ------------------------------------------------------------ poll
    def poll(self, timeout_s: float = 0.0) -> list[Packet]:
        out: list[Packet] = []
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                if not out and timeout_s > 0 and time.perf_counter() < deadline:
                    time.sleep(0.002)
                    continue
                break
        for p in out:
            self._track(p)
        return out

    def _track(self, p: Packet) -> None:
        self.n_packets += 1
        self.n_samples += p.n
        if self._first_packet_t is None:
            self._first_packet_t = p.t_mono
        self.packet_times.append(p.t_mono)
        if p.indices is not None and p.indices.size:
            idx = int(p.indices[-1])
            if self._last_index is not None:
                step = (idx - self._last_index) % 256
                if step != 1:
                    missing = (step - 1) % 256
                    self.n_gaps += 1
                    self.n_missing += missing
            self._last_index = idx

    # ------------------------------------------------------------ thread
    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as exc:                       # pragma: no cover - hardware path
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()

    async def _async_main(self) -> None:
        from bleak import BleakClient

        dev = await find_headband(timeout=self.connect_timeout_s, address=self.address)
        if dev is None:
            self._error = ("headband not advertising - long-press power for 3 s, "
                           "then re-run")
            self._ready.set()
            return

        client = BleakClient(dev, timeout=self.connect_timeout_s)
        await client.connect()
        self.mtu = int(getattr(client, "mtu_size", 0) or 0)

        #: EEG packets seen so far. START_STREAM is acknowledged by the *data*
        #: stream itself, and those frames are routed to the sample queue rather
        #: than to ``saw`` (see control_cb), so "did the stream start?" has to be
        #: answered from here.
        n_stream = [0]

        def cb(_sender: Any, data: bytearray) -> None:
            b = bytes(data)
            if not b.startswith(ACK_STREAM):
                return
            vals, idxs = parse_packet(b)
            if vals.size == 0:
                return
            n_stream[0] += 1
            t = time.perf_counter()
            pkt = Packet(values=vals, t_mono=t, indices=idxs)
            try:
                self._q.put_nowait(pkt)
            except queue.Full:                          # pragma: no cover
                pass
            if self.verbose:
                pass

        await client.start_notify(TX_CHAR, cb)
        await asyncio.sleep(0.8)

        saw: list[bytes] = []

        async def poll_cmd(frame: bytes, label: str, ack: bytes) -> bool:
            for _ in range(self.handshake_tries):
                if self._stop.is_set():
                    return False
                if any(p.startswith(ack) for p in saw):
                    return True
                try:
                    await client.write_gatt_char(RX_CHAR, frame, response=True)
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            return any(p.startswith(ack) for p in saw)

        async def poll_stream() -> bool:
            """Wait for real EEG packets after START_STREAM.

            Deliberately *not* ``poll_cmd(CMD_START, "START_STREAM", ACK_STREAM)``:
            ``control_cb`` routes every ``ACK_STREAM`` frame to the sample queue
            and only appends non-stream frames to ``saw``, so waiting for an
            ``ACK_STREAM`` frame *in saw* could never succeed. That made
            ``ok_stream`` permanently False, so the handshake always failed and
            every "live" session silently ran on the synthetic source even with
            the band connected and all four electrodes seated.
            """
            for _ in range(self.handshake_tries):
                if self._stop.is_set():
                    return False
                if n_stream[0] > 0:
                    return True
                try:
                    await client.write_gatt_char(RX_CHAR, CMD_START, response=True)
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            return n_stream[0] > 0

        # collect non-EEG replies for the handshake
        def control_cb(_sender: Any, data: bytearray) -> None:
            b = bytes(data)
            if not b.startswith(ACK_STREAM):
                saw.append(b)

        await client.start_notify(TX_CHAR, lambda s, d: (cb(s, d), control_cb(s, d)))

        ok_info = await poll_cmd(CMD_INFO, "GET_INFO", ACK_INFO)
        if ok_info:
            p = next((x for x in saw if x.startswith(ACK_INFO)), b"")
            self.device_info_hex = p[HEADER_BYTES:].hex()
        ok_self = await poll_cmd(CMD_SELF_CHECK, "SELF_CHECK", ACK_SELF_CHECK)
        if ok_self:
            p = next((x for x in saw if x.startswith(ACK_SELF_CHECK)), b"")
            self.selfcheck_ascii = p[HEADER_BYTES:].decode("ascii", "replace")
            self.contact, _ = contact_from_selfcheck(self.selfcheck_ascii)
        ok_stream = await poll_stream()
        if not (ok_info and ok_self and ok_stream):
            self._error = (f"handshake failed (info={ok_info} self={ok_self} "
                           f"stream={ok_stream}); electrode self-check: "
                           f"{self.selfcheck_ascii!r}")
            try:
                await client.disconnect()
            except Exception:
                pass
            self._ready.set()
            return

        self._ready.set()

        # stream until told to stop
        while not self._stop.is_set():
            await asyncio.sleep(0.1)

        try:
            await client.write_gatt_char(RX_CHAR, CMD_STOP, response=True)
        except Exception:
            pass
        try:
            await client.stop_notify(TX_CHAR)
        except Exception:
            pass
        try:
            await client.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------ info
    @property
    def info(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "address": self.address,
            "fs": self.fs,
            "mtu": self.mtu,
            "contact": self.contact,
            "selfcheck": self.selfcheck_ascii,
            "device_info_hex": self.device_info_hex,
            "montage": "single dry frontal electrode (Fpz), reference on band",
        }

    def stats(self) -> dict[str, Any]:
        dur = (self.packet_times[-1] - self.packet_times[0]) if len(self.packet_times) > 1 else 0.0
        return {
            "packets": self.n_packets,
            "samples": self.n_samples,
            "gaps": self.n_gaps,
            "missing_samples": self.n_missing,
            "duration_s": round(dur, 3),
            "fs_effective": round(self.n_samples / dur, 3) if dur > 0 else 0.0,
        }
