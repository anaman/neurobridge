"""Local content-addressed cache for stimulus files.

Why this module exists
----------------------
NeuroBridge must run a 64-test curriculum on a machine that may be offline,
may have no Freesound token, and may be rebooted mid-curriculum.  Re-fetching
an image or an audio clip for every session would be slow, rate-limited
(Openverse 429s), non-reproducible (search results change) and rude to the
free APIs we depend on.  So every downloaded or generated artefact is stored
once, addressed by the SHA-1 of its bytes, and looked up by a *stable* key
derived from the request parameters (query, class, n, license...).

Design choices worth remembering in six months
----------------------------------------------
* ``cache_key`` hashes *sorted* ``key=value`` pairs, so two callers that build
  the same logical request in a different order share one cache entry.
* The index is an append-only JSONL file.  Appending cannot corrupt earlier
  entries; a truncated final line (power loss) is skipped by
  :meth:`StimulusCache.index` instead of raising.
* Blob *names* are the SHA-1 of the content, so de-duplication is free and a
  partially-written blob can never be mistaken for a complete one (we write to
  ``*.part`` and ``os.replace`` atomically).
* ``ATTRIBUTION.md`` is a legal requirement, not decoration: CC-BY and CC-BY-SA
  items may be redistributed only with attribution, and the operator needs to
  be able to reconstruct what was presented.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ATTRIBUTION_FILE, CACHE_DIR

#: Sources we know how to attribute.  Kept as data so the builder can report
#: counts per source without importing the fetchers.
SOURCES = ("openverse", "freesound", "synth", "textbank", "builtin")


def cache_key(**parts: Any) -> str:
    """Stable SHA-1 of sorted ``key=value`` pairs.

    Values are coerced with ``str()``; sequences are normalised so that
    ``["a", "b"]`` and ``("a", "b")`` hash identically.  ``None`` and ``""``
    are normalised to the empty string so an omitted optional argument never
    creates a second cache lineage.
    """
    norm: dict[str, str] = {}
    for k in sorted(parts):
        v = parts[k]
        if v is None:
            norm[str(k)] = ""
        elif isinstance(v, (list, tuple, set)):
            norm[str(k)] = "|".join(sorted(str(x) for x in v))
        elif isinstance(v, bool):
            norm[str(k)] = "1" if v else "0"
        else:
            norm[str(k)] = str(v)
    blob = json.dumps(norm, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def sha1_bytes(data: bytes) -> str:
    """SHA-1 of a byte string (blob identity)."""
    return hashlib.sha1(data).hexdigest()


@dataclass
class CachedFile:
    """One stored artefact plus everything needed to attribute and re-find it."""

    sha1: str
    kind: str                                  # image | audio | text | tone
    path: Path
    source: str                                # openverse|freesound|synth|textbank|builtin
    url: str = ""
    license: str = ""                          # e.g. "cc0", "by-sa", "CC0-1.0"
    attribution: str = ""                      # human-readable credit string
    title: str = ""
    query: str = ""                            # the request that produced it
    meta: dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CachedFile":
        return cls(
            sha1=str(d.get("sha1", "")),
            kind=str(d.get("kind", "image")),
            path=Path(d.get("path", "")),
            source=str(d.get("source", "")),
            url=str(d.get("url", "") or ""),
            license=str(d.get("license", "") or ""),
            attribution=str(d.get("attribution", "") or ""),
            title=str(d.get("title", "") or ""),
            query=str(d.get("query", "") or ""),
            meta=dict(d.get("meta") or {}),
        )


class StimulusCache:
    """Content-addressed store under ``data/cache/stimuli`` by default.

    Layout::

        <root>/
            index.jsonl          # one JSON object per stored artefact (append-only)
            blobs/<kind>/<sha1><suffix>

    The cache never raises for a missing directory or a corrupt index line: a
    broken cache must degrade to a cold cache, never take a session down.
    """

    INDEX_NAME = "index.jsonl"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else (Path(CACHE_DIR) / "stimuli")
        self.blob_root = self.root / "blobs"
        self.index_path = self.root / self.INDEX_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        # Parsed-index memo.  ``has()`` runs once per query per class and a
        # curriculum run issues hundreds of lookups; re-reading and JSON-parsing
        # the whole log each time dominated the offline ``prepare()`` path.
        # Invalidated on every write, so it can never go stale.
        self._index_memo: list[CachedFile] | None = None

    # ------------------------------------------------------------ lookups
    def has(self, key: str) -> CachedFile | None:
        """Return the first stored artefact whose ``meta['key']`` == ``key``.

        ``key`` is the stable request key from :func:`cache_key`; ``sha1`` is
        also accepted so callers holding a blob id do not need a second method.
        """
        if not key:
            return None
        for cf in self.index():
            if cf.meta.get("key") == key or cf.sha1 == key:
                if cf.exists():
                    return cf
        return None

    def blob_path(self, sha1: str, suffix: str = "") -> Path:
        return self.blob_root / sha1[:2] / f"{sha1}{suffix}"

    def _write_blob(self, data: bytes, sha1: str, suffix: str) -> Path:
        dest = self.blob_path(sha1, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            self._atomic_write_bytes(dest, data)
        return dest

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    @staticmethod
    def _norm_suffix(suffix: str) -> str:
        if not suffix:
            return ""
        s = suffix if suffix.startswith(".") else "." + suffix
        return s.lower()

    @staticmethod
    def _slug(text: str, limit: int = 60) -> str:
        keep = [c.lower() if c.isalnum() else "-" for c in str(text)[:limit]]
        out = "".join(keep).strip("-")
        while "--" in out:
            out = out.replace("--", "-")
        return out[:limit]

    # ------------------------------------------------------------ writes
    def put_bytes(self, data: bytes, *, kind: str, suffix: str, source: str,
                  url: str = "", license: str = "", attribution: str = "",
                  title: str = "", query: str = "", meta: dict | None = None) -> CachedFile:
        """Store raw bytes; returns the index record (also appended to the log)."""
        if not data:
            raise ValueError("refusing to cache an empty byte string")
        suffix = self._norm_suffix(suffix)
        sha1 = sha1_bytes(data)
        dest = self._write_blob(data, sha1, suffix)
        cf = CachedFile(
            sha1=sha1, kind=kind, path=dest, source=source, url=url,
            license=license, attribution=attribution, title=title, query=query,
            meta=dict(meta or {}),
        )
        self._publish(cf)
        return cf

    def put_path(self, src: Path, **kw: Any) -> CachedFile:
        """Copy a file that already exists on disk into the cache.

        ``suffix`` defaults to the source suffix.  The source file is left in
        place: the synthesizer re-uses its own output directory, so moving
        would silently invalidate that.
        """
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(src)
        data = src.read_bytes()
        suffix = kw.pop("suffix", "") or src.suffix
        meta = dict(kw.pop("meta", None) or {})
        meta.setdefault("origin_path", str(src))
        kind = kw.pop("kind", "image")
        source = kw.pop("source", "builtin")
        return self.put_bytes(data, kind=kind, suffix=suffix, source=source,
                              meta=meta, **kw)

    def _publish(self, cf: CachedFile) -> None:
        """Append to the JSONL index and register attribution when needed."""
        self._index_memo = None                     # invalidate the parse memo
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(cf.as_dict(), sort_keys=True) + "\n")
        except OSError:
            # A read-only checkout still works: the file is on disk and the
            # builder returns it; only cross-session de-duplication is lost.
            pass
        if cf.source in ("openverse", "freesound") or cf.attribution:
            self.add_attribution(cf)

    # ------------------------------------------------------------ reads
    def index(self) -> list[CachedFile]:
        """Read ``index.jsonl``; last record per (sha1, key) wins.

        Corrupt or truncated lines are skipped rather than raised: a cache that
        cannot be parsed is treated as cold, and the caller re-fetches.  The
        parsed list is memoised per instance and invalidated by every write.
        """
        if self._index_memo is not None:
            return list(self._index_memo)
        out: list[CachedFile] = []
        seen: set[tuple[str, str]] = set()
        if not self.index_path.exists():
            self._index_memo = out
            return list(out)
        try:
            lines = self.index_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            self._index_memo = out
            return list(out)
        for line in reversed(lines):                    # newest first
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            cf = CachedFile.from_dict(d)
            if not cf.sha1:
                continue
            ident = (cf.sha1, str(cf.meta.get("key", "")))
            if ident in seen:
                continue
            seen.add(ident)
            out.append(cf)
        out.reverse()                                   # back to insertion order
        self._index_memo = out
        return list(out)

    def count(self) -> int:
        return len(self.index())

    # ------------------------------------------------------------ attribution
    def add_attribution(self, cf: CachedFile) -> None:
        """Append one deduplicated row to ``data/stimuli/ATTRIBUTION.md``.

        Deduplication is by URL (or SHA-1 when there is no URL), so repeated
        runs do not grow the file.  The row is written even when the licence
        is permissive: knowing *what* was presented is part of the dataset.
        """
        path = Path(ATTRIBUTION_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        ident = cf.url or f"sha1:{cf.sha1}"
        marker = f"<!-- {ident} -->"
        try:
            text = path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError:
            text = ""
        if marker in text:
            return
        if not text:
            text = (
                "# Stimulus attribution\n\n"
                "Every image, sound or text passage presented by NeuroBridge is listed\n"
                "here with its origin and licence.  Local `synth`, `textbank` and\n"
                "`builtin` items are generated by this repository (self-authored, no\n"
                "third-party rights); `openverse` and `freesound` rows are third-party\n"
                "works used under the licence named in the row.\n\n"
            )
        credit = cf.attribution or cf.title or cf.url or cf.sha1
        row = (f"- [{cf.source}] {credit}  \n"
               f"  licence: {cf.license or 'unknown'} | kind: {cf.kind} | "
               f"query: {cf.query or '-'} | sha1: {cf.sha1[:12]} | "
               f"source_url: {cf.url or '-'}\n{marker}\n")
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(row)
        except OSError:
            pass


def openverse_attribution(item: dict[str, Any]) -> tuple[str, str]:
    """Build the (attribution, licence) pair for an Openverse search result.

    Openverse already returns a ``attribution`` string that satisfies the
    licence's requirements, but it is not always populated (older records), so
    we fall back to ``creator``/``title``/``license``.
    """
    lic = str(item.get("license", "") or "").strip()
    ver = str(item.get("license_version", "") or "").strip()
    licence = f"{lic}-{ver}" if lic and ver else lic
    if not licence:
        licence = "unknown"
    attr = str(item.get("attribution", "") or "").strip()
    if not attr:
        creator = str(item.get("creator", "") or "").strip() or "unknown creator"
        title = str(item.get("title", "") or "").strip() or "untitled"
        attr = f'"{title}" by {creator} ({licence})'
    elif lic and lic.lower() not in attr.lower():
        attr = f"{attr} ({licence})"
    return attr, licence


def freesound_attribution(item: dict[str, Any]) -> tuple[str, str]:
    """Build the (attribution, licence) pair for a Freesound search result."""
    url = str(item.get("license", "") or "").strip()
    short = url.rstrip("/").rsplit("/", 1)[-1] if url else "unknown"
    name = str(item.get("name", "") or "").strip() or "untitled"
    user = str(item.get("username", "") or "").strip() or "unknown"
    sid = item.get("id", "?")
    attr = (f'"{name}" by {user} (freesound.org/s/{sid}) '
            f'licence {short}: {url or "see source"}')
    return attr, short
