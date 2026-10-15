"""Paths, settings and directory layout for NeuroBridge.

Settings come from, in order of precedence:
  1. environment variables          (NB_DEVICE, FREESOUND_TOKEN, ...)
  2. ``<project>/neurobridge.yaml`` (optional user override file)
  3. the defaults below

The YAML file is never required: everything has a working default so the
system runs on a fresh checkout with no hardware attached.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- paths
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent                      # .../neurobridge
DATA_DIR = ROOT / "data"
STIM_DIR = DATA_DIR / "stimuli"
IMAGE_DIR = STIM_DIR / "images"
AUDIO_DIR = STIM_DIR / "audio"
TEXT_DIR = STIM_DIR / "text"
SESSION_DIR = DATA_DIR / "sessions"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"
CATALOG_DIR = PKG_DIR / "catalog"
TESTS_DIR = CATALOG_DIR / "tests"
MASTER_CATALOG = CATALOG_DIR / "tests.yaml"
DOCS_DIR = ROOT / "docs"
SETTINGS_FILE = ROOT / "neurobridge.yaml"
ATTRIBUTION_FILE = STIM_DIR / "ATTRIBUTION.md"

ALL_DIRS = (DATA_DIR, STIM_DIR, IMAGE_DIR, AUDIO_DIR, TEXT_DIR,
            SESSION_DIR, MODEL_DIR, REPORT_DIR, CACHE_DIR, LOG_DIR, TESTS_DIR)


def ensure_dirs() -> None:
    """Create the data tree if missing (idempotent)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    # keep pygame's "Hello from the pygame community" banner out of our own
    # output and out of the diagnostic bundles
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # matplotlib must never try to write under ~/.config (it is not always
    # writable, and a stray cache directory confuses offline runs).
    mpl = CACHE_DIR / "matplotlib"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    """Runtime settings. ``.env`` style env vars override the YAML file."""

    # --- device -----------------------------------------------------
    device: str = "auto"              # auto | th21a | fake
    #: Empty means "find the headband by name" (see device.th21a.find_headband).
    #: Set it (or NB_ADDRESS) to pin one specific unit.
    address: str = ""
    fs: float = 250.0
    channel_name: str = "Fpz"         # single dry frontal electrode
    connect_timeout_s: float = 30.0

    # --- presentation ----------------------------------------------
    # Fullscreen by default: a session should never show the desktop, the
    # panels or a changing wallpaper behind the stimuli, because that is a
    # visual distraction right where the experiment needs a clean field.
    # `nb run --windowed` opts out.
    fullscreen: bool = True
    display_index: int = 0
    screen_size: tuple[int, int] = (1280, 800)
    visual_backend: str = "auto"      # auto | pygame | terminal
    audio_backend: str = "auto"       # auto | sounddevice | aplay | null
    #: "key" waits for SPACE between segments; "timer" auto-advances.
    #: Automation (no display and no TTY) falls back to "timer" automatically.
    advance: str = "key"

    # --- timing -----------------------------------------------------
    # Constant added to every marker timestamp to correct for the
    # stimulus-output + acquisition latency measured by test T004/T005.
    output_latency_offset_s: float = 0.0
    lsl_enabled: bool = False         # publish EEG + markers on the lab stream bus
    marker_jitter_warn_ms: float = 25.0

    # --- stimuli ----------------------------------------------------
    openverse_client_id: str = ""
    openverse_client_secret: str = ""
    freesound_token: str = ""
    http_timeout_s: float = 30.0
    download_workers: int = 4
    max_items_per_test: int = 400

    # --- pipeline ---------------------------------------------------
    notch_hz: tuple[float, ...] = (50.0, 60.0)
    bandpass_hz: tuple[float, float] = (0.3, 45.0)
    artifact_ptp_counts: float = 400_000.0   # saturation-ish peak-to-peak reject
    seed: int = 20261013

    # --- agent bridge ----------------------------------------------
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 8731
    state_hz: float = 2.0             # how often the bridge recomputes state

    def save(self, path: Path | None = None) -> Path:
        import yaml
        path = Path(path or SETTINGS_FILE)
        path.write_text(yaml.safe_dump(self.as_dict(), sort_keys=True))
        return path

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        import yaml
        path = Path(path or SETTINGS_FILE)
        data: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: expected a mapping at top level")
            data.update(loaded)
        # environment wins
        for f in fields(cls):
            env = os.environ.get("NB_" + f.name.upper())
            if env is None:
                continue
            data[f.name] = env
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in known:
                continue                      # forward compatible: ignore unknowns
            kwargs[k] = _coerce(k, v)
        return cls(**kwargs)


def _coerce(name: str, value: Any) -> Any:
    """YAML/env values arrive as strings; coerce to the field's type."""
    if name in ("fs", "connect_timeout_s", "output_latency_offset_s",
                "http_timeout_s", "artifact_ptp_counts", "state_hz",
                "marker_jitter_warn_ms"):
        return float(value)
    if name in ("display_index", "download_workers", "max_items_per_test",
                "seed", "bridge_port"):
        return int(value)
    if name in ("fullscreen", "lsl_enabled"):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if name in ("notch_hz",):
        if isinstance(value, str):
            return tuple(float(x) for x in value.replace(",", " ").split())
        return tuple(float(x) for x in value)
    if name in ("bandpass_hz", "screen_size"):
        if isinstance(value, str):
            return tuple(float(x) for x in value.replace(",", " ").split())
        return tuple(float(x) for x in value)
    return value


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or reload:
        _settings = Settings.load()
    return _settings


def env_token(name: str) -> str:
    """Look up a credential in env, then in the YAML settings."""
    import os
    v = os.environ.get(name)
    if v:
        return v.strip()
    s = get_settings()
    key = name.lower()
    return str(getattr(s, key, "") or "").strip()


# ---------------------------------------------------------------- helpers
def session_dir(session_id: str) -> Path:
    return SESSION_DIR / session_id


def new_session_id(prefix: str = "s") -> str:
    """Sortable, human-readable session id: s-20261013-142530-ab12."""
    import datetime as _dt
    import secrets
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


def list_sessions() -> list[Path]:
    """All session directories, **oldest first by when the session started**.

    Sorting by directory *name* is wrong: session ids are timestamped for real
    sessions, but synthetic ones are prefixed ``sim-`` and imported ones
    ``import-``, so a plain lexicographic sort puts ``sim-*`` after every
    ``s-*`` no matter how old it is (``"sim-" > "s-"``). That made
    ``Session.latest()`` return a stale synthetic session during a live run.
    ``created_wall`` from ``meta.json`` is the honest key; mtime is the fallback.
    """
    if not SESSION_DIR.exists():
        return []
    dirs = [p for p in SESSION_DIR.iterdir() if (p / "meta.json").exists()]

    def key(p: Path) -> tuple[str, float]:
        try:
            meta = json.loads((p / "meta.json").read_text())
            created = str(meta.get("created_wall") or "")
        except Exception:
            created = ""
        try:
            mtime = (p / "meta.json").stat().st_mtime
        except OSError:
            mtime = 0.0
        return (created, mtime)

    return sorted(dirs, key=key)


def resolve_device_name(requested: str | None = None) -> str:
    """Resolve 'auto' to th21a when a headband is reachable, else fake.

    'auto' is intentionally conservative: scanning BLE takes ~10 s, so callers
    that cannot wait should pass an explicit device name.
    """
    import asyncio
    name = (requested or get_settings().device or "auto").lower()
    if name != "auto":
        return name
    try:
        from .device.th21a import TH21A, find_headband

        dev = asyncio.run(find_headband(timeout=8.0))
        return "th21a" if dev is not None else "fake"
    except Exception:
        return "fake"
