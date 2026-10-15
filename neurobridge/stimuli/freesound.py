"""Freesound client (https://freesound.org/apiv2/) — CC-licensed audio.

The token problem
-----------------
Unlike Openverse, Freesound refuses **even a search** without a token (verified
on this machine: anonymous ``GET /apiv2/search/text/?query=rain`` returns HTTP
401).  So this module is explicitly two-state:

* no token -> :attr:`FreesoundClient.available` is ``False`` and
  :meth:`FreesoundClient.fetch` raises :class:`NoTokenError` *immediately*,
  without touching the network.  The builder catches it and falls back to the
  deterministic synthesizer, and the warning it prints tells the operator
  exactly which variable to set.
* token present -> normal search/download.

Two auth styles, one heuristic
------------------------------
Freesound accepts either an OAuth2 bearer token (40+ chars) or a legacy API
key (32 hex chars, sent as the ``token`` query parameter).  The two are
indistinguishable by shape alone for some tokens, so the rule is explicit and
overridable:

* :func:`looks_like_oauth2` -- length >= 40 decides it by default.
* ``NB_FREESOUND_AUTH_STYLE`` / ``FREESOUND_AUTH_STYLE`` or the ``auth_style``
  argument forces ``"oauth2"`` or ``"apikey"`` when the guess is wrong.

OAuth2 tokens expire; :meth:`FreesoundClient.oauth2_token` performs the
documented client-credentials exchange so a long-running rig can refresh
instead of dying mid-block.

Why previews
------------
The ``previews`` field is the only stream that is reliably downloadable with
the same token (the original file needs per-sound OAuth2 download permission
for many uploads).  ``preview_hq`` is an mp3; we convert to 44.1 kHz mono WAV
with ffmpeg when it exists, and otherwise keep the mp3 and say so.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..config import get_settings
from .cache import CachedFile, StimulusCache, cache_key, freesound_attribution
from .http import retry_after_seconds as _retry_after_seconds
from .http import short_body as _short_body
from .http import sleep as _sleep
from .openverse import AuthError, NetworkError, NoResultsError

#: Preview quality order.  ``preview_hq`` is ~128 kbps mp3, plenty for an
#: auditory oddball and small enough to cache 400 of them.
PREVIEW_PREFERENCE = ("preview_hq", "preview_lq")

#: Freesound's OAuth2 tokens are long opaque strings; legacy API keys are 32
#: hex characters.  This threshold is the whole heuristic.
OAUTH2_MIN_LENGTH = 40

TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"

SEARCH_FIELDS = (
    "id,name,tags,license,username,duration,filesize,type,channels,samplerate,"
    "previews,download,url,description,created"
)


def looks_like_oauth2(token: str) -> bool:
    """Heuristic: >= 40 characters means OAuth2, otherwise a legacy API key.

    Deliberately simple and documented rather than clever: guessing wrong makes
    every request 401, so the guess must be explainable to the operator, who
    can override it with ``auth_style=`` or ``NB_FREESOUND_AUTH_STYLE``.
    """
    return len((token or "").strip()) >= OAUTH2_MIN_LENGTH


def resolve_auth_style(token: str, auth_style: str | None = None) -> str:
    """``"oauth2"`` or ``"apikey"``: explicit argument > env > heuristic."""
    explicit = (auth_style or os.environ.get("NB_FREESOUND_AUTH_STYLE", "")
                or os.environ.get("FREESOUND_AUTH_STYLE", "")).strip().lower()
    if explicit in ("oauth2", "bearer", "token"):
        return "oauth2"
    if explicit in ("apikey", "api_key", "key", "legacy"):
        return "apikey"
    return "oauth2" if looks_like_oauth2(token) else "apikey"


class NoTokenError(RuntimeError):
    """No Freesound token is configured, so no request can succeed."""


class FreesoundClient:
    """Minimal Freesound v2 client with auth-style detection and mp3->wav."""

    BASE = "https://freesound.org/apiv2"

    def __init__(self, token: str = "", timeout: float = 30.0,
                 auth_style: str | None = None) -> None:
        settings = get_settings()
        env = os.environ.get("NB_FREESOUND_TOKEN", "").strip()
        self.token = (token or env or settings.freesound_token or "").strip()
        self.timeout = float(timeout or settings.http_timeout_s or 30.0)
        self.auth_style = resolve_auth_style(self.token, auth_style)
        self._session: Any = None

    # ------------------------------------------------------------ auth
    @property
    def available(self) -> bool:
        """True when a token exists; the builder uses this to skip the network."""
        return bool(self.token)

    @property
    def uses_bearer(self) -> bool:
        return self.auth_style == "oauth2"

    def set_token(self, token: str, auth_style: str | None = None) -> None:
        """Replace the token (after a refresh) and re-resolve the auth style."""
        self.token = (token or "").strip()
        self.auth_style = resolve_auth_style(self.token, auth_style)

    def oauth2_token(self, client_id: str, client_secret: str,
                     timeout: float | None = None) -> str:
        """Documented OAuth2 client-credentials exchange; returns the token.

        Helper rather than automatic behaviour: a token grant is a write-ish
        operation against a third-party service, so it should happen when the
        caller explicitly asks (e.g. its first 401 of the day), not silently on
        an import or a probe.
        """
        import requests

        timeout = float(timeout or self.timeout)
        try:
            resp = requests.post(
                TOKEN_URL,
                data={"client_id": client_id, "client_secret": client_secret,
                      "grant_type": "client_credentials"},
                headers={"Accept": "application/json",
                         "User-Agent": "NeuroBridge/0.1"},
                timeout=timeout,
            )
        except Exception as exc:
            raise NetworkError(f"Freesound OAuth2 token request failed: {exc}") from exc
        if resp.status_code == 401:
            raise AuthError(
                "Freesound rejected the OAuth2 client credentials (HTTP 401). "
                "Check the client id/secret at freesound.org/apiv2/apply."
            )
        if resp.status_code >= 400:
            raise NetworkError(
                f"Freesound OAuth2 token endpoint returned HTTP {resp.status_code}: "
                f"{_short_body(resp)}"
            )
        try:
            token = str(resp.json().get("access_token", "") or "").strip()
        except Exception as exc:
            raise NetworkError(f"Freesound OAuth2 response was not JSON: {exc}") from exc
        if not token:
            raise AuthError("Freesound OAuth2 response contained no access_token")
        self.set_token(token, "oauth2")
        return token

    # ------------------------------------------------------------ plumbing
    def _client(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "NeuroBridge/0.1",
                                          "Accept": "application/json"})
        return self._session

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None

    def _auth_kwargs(self) -> tuple[dict[str, str], dict[str, str]]:
        if self.uses_bearer:
            return {"Authorization": f"Bearer {self.token}"}, {}
        return {}, {"token": self.token}

    def _request(self, url: str, *, params: dict | None = None,
                 stream: bool = False) -> Any:
        """GET with one 429 retry (honouring Retry-After) and one 5xx retry."""
        if not self.available:
            raise NoTokenError(
                "Freesound requires a token even to search, and none is configured. "
                "Set NB_FREESOUND_TOKEN (or freesound_token: in neurobridge.yaml) to "
                "use real recordings; the builder falls back to synthesized audio."
            )
        headers, extra = self._auth_kwargs()
        query = dict(params or {})
        query.update(extra)
        retries_429 = 2
        tried_5xx = False
        while True:
            try:
                resp = self._client().get(url, params=query, headers=headers,
                                          timeout=self.timeout, stream=stream)
            except Exception as exc:
                raise NetworkError(f"Freesound request to {url} failed: {exc}") from exc
            if resp.status_code == 429 and retries_429 > 0:
                delay = _retry_after_seconds(resp.headers, default=2.0)
                retries_429 -= 1
                _close(resp)
                _sleep(delay)
                continue
            if 500 <= resp.status_code < 600 and not tried_5xx:
                tried_5xx = True
                _close(resp)
                _sleep(1.0)
                continue
            if resp.status_code in (401, 403):
                raise AuthError(
                    f"Freesound returned HTTP {resp.status_code}. The token is missing, "
                    "expired or the wrong kind: legacy API keys go in the `token` query "
                    "parameter (32 hex chars), OAuth2 tokens go in the Authorization "
                    "header (40+ chars). Override the guess with "
                    "NB_FREESOUND_AUTH_STYLE=apikey|oauth2. "
                    f"{_short_body(resp)}"
                )
            if resp.status_code >= 400:
                raise NetworkError(
                    f"Freesound returned HTTP {resp.status_code} for {url}: "
                    f"{_short_body(resp)}"
                )
            return resp

    # ------------------------------------------------------------ API
    def search(self, query: str, n: int = 10, *, min_duration: float | None = None,
               max_duration: float | None = None, license: str | None = None,
               sort: str = "score") -> list[dict]:
        """Search sounds; returns raw Freesound result dicts (up to ``n``)."""
        query = (query or "").strip()
        if not query:
            raise NoResultsError("empty Freesound query")
        n = max(1, int(n))
        filters: list[str] = []
        if min_duration is not None:
            filters.append(f"duration:[{float(min_duration):g} TO *]")
        if max_duration is not None:
            filters.append(f"duration:[* TO {float(max_duration):g}]")
        if license:
            url = _license_url(license)
            if url:
                filters.append(f'license:"{url}"')
        params: dict[str, Any] = {
            "query": query,
            "page_size": min(n, 150),
            "fields": SEARCH_FIELDS,
            "sort": sort or "score",
        }
        if filters:
            params["filter"] = " ".join(filters)
        resp = self._request(f"{self.BASE}/search/text/", params=params)
        try:
            payload = resp.json()
        except Exception as exc:
            raise NetworkError(f"Freesound search returned non-JSON: {exc}") from exc
        results = payload.get("results")
        if results is None and isinstance(payload, list):
            results = payload
        results = [r for r in (results or []) if isinstance(r, dict)]
        if not results:
            raise NoResultsError(f"Freesound returned no sounds for {query!r}")
        return results[:n]

    def preview_url(self, item: dict, prefer: str = "preview_hq") -> tuple[str, str]:
        """Pick a preview URL; returns ``(url, preview_key)``.

        Falls back through :data:`PREVIEW_PREFERENCE` so a record that only has
        the low-quality preview still yields *something*.
        """
        previews = item.get("previews") or {}
        order = [prefer] + [p for p in PREVIEW_PREFERENCE if p != prefer]
        for key in order:
            url = str(previews.get(key, "") or "").strip()
            if url.startswith("http"):
                return url, key
        for key, url in previews.items():
            if str(url).startswith("http"):
                return str(url), str(key)
        return "", ""

    def download(self, item: dict, dest: Path, *, prefer: str = "preview_hq") -> Path:
        """Download a preview, decode it to 44.1 kHz mono WAV when ffmpeg exists.

        Returns the WAV path when conversion succeeded, otherwise the raw mp3
        (libsndfile >= 1.2 can read mp3, and the runner reports the format it
        actually got).  Raising here means "no usable audio for this item"; the
        caller skips it and tries the next result.
        """
        url, preview_key = self.preview_url(item, prefer)
        if not url:
            raise NetworkError(f"Freesound sound {item.get('id')} has no downloadable preview")
        resp = self._request(url, stream=True)
        max_bytes = 32 * 1024 * 1024
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise NetworkError(f"Freesound preview {url} exceeds {max_bytes} bytes")
                chunks.append(chunk)
        except NetworkError:
            raise
        except Exception as exc:
            raise NetworkError(f"Freesound download of {url} failed: {exc}") from exc
        finally:
            _close(resp)

        data = b"".join(chunks)
        if not data:
            raise NetworkError(f"Freesound preview {url} was empty")

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mp3 = dest.with_suffix(".mp3")
        mp3.write_bytes(data)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            wav = dest.with_suffix(".wav")
            if _ffmpeg_to_wav(mp3, wav, ffmpeg):
                mp3.unlink(missing_ok=True)
                return wav
        return mp3

    def fetch(self, query: str, n: int, cache: StimulusCache, *, force: bool = False) -> list[CachedFile]:
        """Cache-first fetch of ``n`` sounds for ``query``.

        Raises :class:`NoTokenError` before any network call when there is no
        token, so the builder's fallback path is instant offline.
        """
        if not self.available:
            raise NoTokenError(
                "Freesound needs NB_FREESOUND_TOKEN even to search; none configured."
            )
        n = max(1, int(n))
        key = cache_key(source="freesound", query=query, n=n, license="", sort="score")
        if not force:
            hit = cache.has(key)
            if hit is not None:
                return [hit]
        items = self.search(query, n)
        out: list[CachedFile] = []
        tmp = Path(tempfile.mkdtemp(prefix="nb-freesound-"))
        try:
            for item in items:
                if len(out) >= n:
                    break
                attr, licence = freesound_attribution(item)
                try:
                    path = self.download(item, tmp / f"fs{item.get('id', len(out))}")
                except NetworkError:
                    continue
                suffix = Path(path).suffix or ".mp3"
                preview_url = self.preview_url(item)[0]
                cf = cache.put_path(
                    path, kind="audio", source="freesound", suffix=suffix,
                    url=preview_url or str(item.get("url", "") or ""),
                    license=licence, attribution=attr,
                    title=str(item.get("name", "") or ""), query=query,
                    meta={
                        "key": key if len(out) == 0 else f"{key}:{len(out)}",
                        "freesound_id": item.get("id"),
                        "username": item.get("username"),
                        "duration_s": item.get("duration"),
                        "samplerate": item.get("samplerate"),
                        "channels": item.get("channels"),
                        "converted_to_wav": suffix.lower() == ".wav",
                        "original_preview": preview_url,
                        "page_url": item.get("url"),
                        "sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest(),
                    },
                )
                out.append(cf)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not out:
            raise NoResultsError(f"Freesound: no downloadable preview for {query!r}")
        return out


def _close(resp: Any) -> None:
    try:
        resp.close()
    except Exception:
        pass


def _license_url(license: str) -> str:
    """Map ``cc0`` / ``by`` / ``by-nc`` to the canonical CC URL Freesound filters on."""
    raw = (license or "").strip()
    if raw.startswith("http"):
        return raw.replace("http://", "https://")
    if not raw:
        return ""
    base = "https://creativecommons.org"
    if raw.lower() in ("cc0", "zero", "publicdomain", "public-domain"):
        return f"{base}/publicdomain/zero/1.0/"
    if raw.lower() in ("by", "by-sa", "by-nc", "by-nd", "by-nc-sa", "by-nc-nd"):
        return f"{base}/licenses/{raw.lower()}/4.0/"
    return ""


def _ffmpeg_to_wav(src: Path, dest: Path, ffmpeg: str) -> bool:
    """Decode ``src`` to 44.1 kHz mono 16-bit PCM WAV with ffmpeg.

    Mono + 44.1 kHz is chosen because the presentation path and every analysis
    assume one channel at the device sample rate's family; stereo previews would
    otherwise halve the perceived loudness when summed and break the loudness
    ladder in T607.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(dest)],
            capture_output=True, timeout=120,
        )
    except Exception:
        return False
    return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 1024


def ffmpeg_available() -> bool:
    """True when ffmpeg is on PATH (used for the fallback warning wording)."""
    return shutil.which("ffmpeg") is not None
