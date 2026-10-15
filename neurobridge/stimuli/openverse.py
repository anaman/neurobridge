"""Openverse client (https://api.openverse.org/v1/) — CC-licensed images.

Why Openverse
-------------
It aggregates Flickr, Wikimedia, museums and other CC sources behind one API,
and — verified on this machine — anonymous search works (HTTP 200 on
``/v1/images/?q=sunset&page_size=1``).  That matters because a research rig
should not require the operator to register an application before the first
session runs.

Auth model
----------
* **Anonymous** is the default and is fully supported for search.
* **OAuth2 client-credentials** is optional: set ``NB_OPENVERSE_CLIENT_ID`` /
  ``NB_OPENVERSE_CLIENT_SECRET`` (they may also live in ``neurobridge.yaml``).
  Anonymous callers are rate-limited much more aggressively, so a lab that
  fetches hundreds of images should register an application.  The token is
  fetched lazily on the first request that needs it and refreshed on 401.

Failure policy
--------------
Nothing here raises for "the network is down" or "the API said no": the fetcher
raises :class:`NetworkError`, and the *builder* catches it and falls back.  The
only exceptions that escape :meth:`OpenverseClient.fetch` are the documented
``NetworkError`` / ``NoResultsError``, and even those are caught one level up.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any

from ..config import get_settings
from .cache import CachedFile, StimulusCache, cache_key, openverse_attribution
from .http import retry_after_seconds as _retry_after_seconds, short_body as _short_body

#: Long edge cap for cached images.  1024 px keeps a 1280x800 presentation
#: window artifact-free at mild upscaling while keeping the cache small enough
#: to version-control metadata and to check out on a laptop.
MAX_LONG_EDGE = 1024
JPEG_QUALITY = 88

#: Openverse rejects requests without a descriptive UA; a bare python-requests
#: UA is treated as a bot and may be throttled harder.
DEFAULT_USER_AGENT = "NeuroBridge/0.1 (research EEG stimulus fetcher; +https://github.com/neurobridge)"
TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"

#: Process-wide memo for has_network(): a session calls it per test, and a
#: 2.5 s probe per test would add minutes over the 64-test curriculum.
_NETWORK_PROBE: dict[str, Any] = {"at": 0.0, "ok": False, "done": False}
NETWORK_TIMEOUT_S = 2.5
NETWORK_TTL_S = 300.0


class NetworkError(RuntimeError):
    """The request could not be completed (offline, DNS, 5xx after retries)."""


class NoResultsError(RuntimeError):
    """The API answered, but there is nothing usable for this query/licence."""


class AuthError(NetworkError):
    """Credentials are missing, wrong or rejected (401/403).

    Subclasses :class:`NetworkError` so the builder's single ``except`` clause
    still falls back to built-in stimuli — an expired token must never take a
    session down mid-curriculum.
    """


class OpenverseClient:
    """Small, dependency-light wrapper around the Openverse image API."""

    BASE = "https://api.openverse.org/v1"

    def __init__(self, client_id: str = "", client_secret: str = "",
                 timeout: float = 30.0, user_agent: str = "NeuroBridge/0.1") -> None:
        settings = get_settings()
        self.client_id = (client_id or settings.openverse_client_id or "").strip()
        self.client_secret = (client_secret or settings.openverse_client_secret or "").strip()
        self.timeout = float(timeout or settings.http_timeout_s or 30.0)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._token: str = ""
        self._session: Any = None
        self._last_status: int = 0

    # ------------------------------------------------------------ plumbing
    def _client(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            })
        return self._session

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _access_token(self, force: bool = False) -> str:
        """Fetch (and memoise) an OAuth2 client-credentials token.

        Anonymous access is the fallback: when no credentials are configured we
        return "" and the request goes out unauthenticated.
        """
        if not self.authenticated:
            return ""
        if self._token and not force:
            return self._token
        try:
            resp = self._client().post(
                TOKEN_URL,
                data={"grant_type": "client_credentials",
                      "client_id": self.client_id,
                      "client_secret": self.client_secret},
                timeout=self.timeout,
            )
        except Exception as exc:                     # offline, DNS, TLS, timeout
            raise NetworkError(f"Openverse token request failed: {exc}") from exc
        if resp.status_code in (400, 401, 403):
            raise AuthError(
                "Openverse rejected the OAuth2 credentials "
                f"(HTTP {resp.status_code}). Check NB_OPENVERSE_CLIENT_ID / "
                "NB_OPENVERSE_CLIENT_SECRET, or unset them to use anonymous access."
            )
        if resp.status_code >= 400:
            raise NetworkError(f"Openverse token endpoint returned {resp.status_code}")
        try:
            self._token = str(resp.json().get("access_token", "") or "")
        except Exception as exc:
            raise NetworkError(f"Openverse token response was not JSON: {exc}") from exc
        if not self._token:
            raise AuthError("Openverse token response contained no access_token")
        return self._token

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        token = self._access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _license_param(license_type: str) -> str:
        """Openverse wants a comma-separated list; normalise common spellings."""
        raw = (license_type or "all-cc").strip().lower()
        if raw in ("", "all", "all-cc", "all_cc"):
            return "all-cc"
        parts = [p.strip().replace("_", "-") for p in raw.replace("+", ",").split(",") if p.strip()]
        return ",".join(parts) if parts else "all-cc"

    def _request(self, method: str, url: str, *, params: dict | None = None,
                 stream: bool = False) -> Any:
        """GET/POST with 429 (Retry-After, <=2 retries) and one 5xx retry.

        Auth failures are re-raised as :class:`AuthError` with an actionable
        message; everything else that is not 2xx/404 becomes ``NetworkError``.
        """
        import requests

        client = self._client()
        headers = self._headers()
        retries_429 = 2
        tried_5xx = False
        while True:
            try:
                resp = client.request(method, url, params=params, headers=headers,
                                      timeout=self.timeout, stream=stream)
            except Exception as exc:
                raise NetworkError(f"Openverse request to {url} failed: {exc}") from exc
            self._last_status = int(resp.status_code)
            if resp.status_code == 429 and retries_429 > 0:
                delay = _retry_after_seconds(resp.headers, default=2.0)
                retries_429 -= 1
                try:
                    resp.close()
                except Exception:
                    pass
                time.sleep(delay)
                continue
            if 500 <= resp.status_code < 600 and not tried_5xx:
                tried_5xx = True
                try:
                    resp.close()
                except Exception:
                    pass
                time.sleep(1.0)
                continue
            if resp.status_code in (401, 403):
                detail = _short_body(resp)
                raise AuthError(
                    f"Openverse returned HTTP {resp.status_code} for {url}. "
                    "Anonymous access is heavily rate-limited and some queries need "
                    "credentials — set NB_OPENVERSE_CLIENT_ID / "
                    f"NB_OPENVERSE_CLIENT_SECRET and retry. {detail}"
                )
            if resp.status_code >= 400:
                raise NetworkError(
                    f"Openverse returned HTTP {resp.status_code} for {url}: {_short_body(resp)}"
                )
            return resp

    # ------------------------------------------------------------ API
    def search(self, query: str, n: int = 20, *, license_type: str = "all-cc",
               size: str | None = None, mature: bool = False,
               page_size: int = 20) -> list[dict]:
        """Search images; returns raw Openverse result dicts (up to ``n``).

        ``size`` is Openverse's ``small|medium|large`` bucket; ``mature``
        defaults to *False* because this is a research rig that may be used by
        people who did not choose the content.
        """
        query = (query or "").strip()
        if not query:
            raise NoResultsError("empty Openverse query")
        n = max(1, int(n))
        page_size = max(1, min(int(page_size or n), 100))
        params: dict[str, Any] = {
            "q": query,
            "page_size": page_size,
            "license_type": self._license_param(license_type),
            "mature": "true" if mature else "false",
        }
        if size:
            params["size"] = size
        out: list[dict] = []
        page = 1
        while len(out) < n and page <= 5:
            params["page"] = page
            resp = self._request("GET", f"{self.BASE}/images/", params=params)
            try:
                payload = resp.json()
            except Exception as exc:
                raise NetworkError(f"Openverse search returned non-JSON: {exc}") from exc
            results = payload.get("results") or []
            if not results:
                break
            out.extend([r for r in results if isinstance(r, dict)])
            if len(results) < page_size:
                break
            page += 1
        if not out:
            raise NoResultsError(
                f"Openverse returned no images for {query!r} (license_type="
                f"{params['license_type']})"
            )
        return out[:n]

    @staticmethod
    def _pick_url(item: dict) -> str:
        """The direct file URL, not the landing page.

        Some records carry several; ``url`` is the canonical one, but openverse
        also exposes ``thumbnail`` for tiny derivatives (too small for RSVP),
        so we only use the thumbnail as a last resort.
        """
        for key in ("url", "foreign_landing_url"):
            v = str(item.get(key, "") or "").strip()
            if v.startswith("http") and key == "url":
                return v
        return str(item.get("url", "") or "").strip()

    def download(self, item: dict, dest: Path) -> Path:
        """Download one record, verify it as an image, downscale, re-encode.

        Why re-encode: Openverse points at arbitrary formats (webp, huge
        TIFFs, occasionally an HTML error page).  Decoding with PIL proves the
        bytes are an image, and re-encoding to a bounded long edge makes
        presentation timing predictable (a 6000x4000 PNG decodes too slowly for
        RSVP).  PNG is kept when the image has alpha; otherwise JPEG q88.
        """
        from PIL import Image

        url = self._pick_url(item)
        if not url:
            raise NetworkError(f"Openverse record {item.get('id')} has no download url")
        resp = self._request("GET", url, stream=True)
        max_bytes = 32 * 1024 * 1024
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise NetworkError(f"Openverse file {url} exceeds {max_bytes} bytes")
                chunks.append(chunk)
        except NetworkError:
            raise
        except Exception as exc:
            raise NetworkError(f"Openverse download of {url} failed: {exc}") from exc
        finally:
            try:
                resp.close()
            except Exception:
                pass

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(io.BytesIO(b"".join(chunks)))
            img.load()
        except Exception as exc:
            raise NetworkError(f"Openverse file {url} did not decode as an image: {exc}") from exc

        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            out = img.convert("RGBA")
        else:
            out = img.convert("RGB")
        long_edge = max(out.size)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / float(long_edge)
            new = (max(1, int(round(out.size[0] * scale))),
                   max(1, int(round(out.size[1] * scale))))
            out = out.resize(new, Image.LANCZOS)

        suffix = ".png" if has_alpha else ".jpg"
        target = dest.with_suffix(suffix)
        try:
            if has_alpha:
                out.save(target, format="PNG", optimize=True)
            else:
                out.save(target, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        except Exception as exc:
            raise NetworkError(f"could not re-encode {url}: {exc}") from exc
        return target

    # ------------------------------------------------------------ cache-aware fetch
    def fetch(self, query: str, n: int, cache: StimulusCache, *, force: bool = False) -> list[CachedFile]:
        """Cache-first fetch of ``n`` images for ``query``.

        A cache hit is keyed on the request (query+n+licence), so two tests that
        ask for the same thing share the files.  Individual download failures
        are skipped: one dead Flickr URL must not fail the whole class.
        """
        n = max(1, int(n))
        key = cache_key(source="openverse", query=query, n=n,
                        license_type="all-cc", size="", mature=False)
        if not force:
            hit = cache.has(key)
            if hit is not None:
                return [hit]
        items = self.search(query, n)
        out: list[CachedFile] = []
        tmp = cache.root / "tmp"
        for item in items:
            if len(out) >= n:
                break
            attr, licence = openverse_attribution(item)
            try:
                path = self.download(item, tmp / str(item.get("id", "item")))
            except NetworkError:
                continue
            sha1 = hashlib.sha1(Path(path).read_bytes()).hexdigest()
            cf = cache.put_path(
                path, kind="image", source="openverse",
                url=self._pick_url(item), license=licence, attribution=attr,
                title=str(item.get("title", "") or ""), query=query,
                meta={
                    "key": key if len(out) == 0 else f"{key}:{len(out)}",
                    "openverse_id": item.get("id"),
                    "creator": item.get("creator"),
                    "creator_url": item.get("creator_url"),
                    "license_version": item.get("license_version"),
                    "foreign_landing_url": item.get("foreign_landing_url"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "sha1": sha1,
                },
            )
            try:
                path.unlink()
            except OSError:
                pass
            out.append(cf)
        if not out:
            raise NoResultsError(f"Openverse: no downloadable image for {query!r}")
        return out

    # ------------------------------------------------------------ connectivity
    def has_network(self) -> bool:
        """Short, memoised reachability probe (<= 3 s, cached per process).

        ``/v1/images/?q=test&page_size=1`` is used rather than a bare TCP
        connect: it proves the API is reachable *and* answering, which is what
        the caller actually cares about.  Anonymous 200 is enough.
        """
        global _NETWORK_PROBE
        now = time.monotonic()
        if _NETWORK_PROBE["done"] and (now - float(_NETWORK_PROBE["at"])) < NETWORK_TTL_S:
            return bool(_NETWORK_PROBE["ok"])
        ok = False
        try:
            import requests

            resp = requests.get(
                f"{self.BASE}/images/",
                params={"q": "test", "page_size": 1},
                headers={"User-Agent": self.user_agent},
                timeout=min(NETWORK_TIMEOUT_S, self.timeout),
            )
            # 429 still proves the API is reachable and answering.
            ok = resp.status_code < 500
        except Exception:
            ok = False
        _NETWORK_PROBE = {"at": now, "ok": ok, "done": True}
        return ok
