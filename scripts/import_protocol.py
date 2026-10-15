#!/usr/bin/env python3
"""Import the external EEG-to-agent protocol document into NeuroBridge.

    .venv/bin/python scripts/import_protocol.py [path/to/EEG_AGENT_TRAINING_PROTOCOL.md]

The document (authored separately, 2026-10-13) is a *specification*: 64 test
cards in 8 modules (A-H, ids T01-T64) plus a media ledger of candidate
Openverse images and Freesound recordings. It is not code and it is not
evidence that any label is decodable -- it says so itself, repeatedly.

This importer does three things and nothing clever:

1. copies the document **verbatim** into ``docs/`` and writes a provenance
   sidecar (source path, size, sha256, import time) so a later reader can prove
   which revision was imported;
2. parses the 64 cards into ``neurobridge/catalog/protocol_codex.yaml``;
3. parses the media ledger into ``neurobridge/catalog/media_ledger.yaml`` so
   the exact candidate assets can be fetched and hashed.

Re-running it is idempotent and safe: it never edits the source document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Where the upstream protocol document lives on the machine that authored this
#: project. Override with ``--source PATH`` or ``NB_PROTOCOL_SOURCE``; the copy in
#: ``docs/`` is checked in, so re-running this script is only needed to refresh it
#: from a newer upstream revision.
DEFAULT_SOURCE = Path(
    os.environ.get("NB_PROTOCOL_SOURCE",
                   "EEG_AGENT_TRAINING_PROTOCOL.md"))

DOC_DEST = ROOT / "docs" / "EEG_AGENT_TRAINING_PROTOCOL.md"
PROVENANCE = ROOT / "docs" / "EEG_AGENT_TRAINING_PROTOCOL.provenance.json"
CARDS_DEST = ROOT / "neurobridge" / "catalog" / "protocol_codex.yaml"
LEDGER_DEST = ROOT / "neurobridge" / "catalog" / "media_ledger.yaml"

CARD_RE = re.compile(r"^####\s+(T\d{2})\.\s+(.*?)\s*$")
LEDGER_RE = re.compile(r"^####\s+`([A-Z]+_[A-Z0-9_]+)`\s*-\s*(.*?)\s*\.?\s*$")
BULLET_RE = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.*)$")
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
MODULE_RE = re.compile(r"^###\s+Module\s+([A-H]):\s*(.*?)\s*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")


def _clean(s: str) -> str:
    """Strip markdown emphasis, links and trailing sentence punctuation."""
    s = LINK_RE.sub(lambda m: m.group(2), s)
    s = s.replace("**", "").replace("`", "").strip()
    return s.rstrip(".").strip()


def _first_link(s: str) -> str:
    m = LINK_RE.search(s)
    return m.group(2) if m else ""


def _norm_key(s: str) -> str:
    """'Media URL' / 'Type / task duration' -> 'media_url' / 'type_task_duration'."""
    s = _clean(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _clean_label(s: str) -> str:
    """Like _clean, but keeps link *labels* instead of link targets."""
    s = LINK_RE.sub(lambda m: m.group(1), s)
    return s.replace("**", "").replace("`", "").strip().rstrip(".")


def _segments(val: str) -> dict[str, str]:
    """Split 'a; **b:** c' into {'': 'a', 'b': 'c'} (bare segments under '')."""
    out: dict[str, str] = {}
    for part in val.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out[_norm_key(k)] = v.strip()
        else:
            out.setdefault("", part)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- cards
def parse_cards(text: str) -> list[dict]:
    lines = text.splitlines()
    cards: list[dict] = []
    module = module_name = ""
    i = 0
    cur: dict | None = None
    mode = ""                      # '', 'stimulus', 'note'

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            cur["stimulus"] = cur.get("_stim", "").strip()
            cur["protocol_note"] = cur.get("_note", "").strip()
            cur.pop("_stim", None)
            cur.pop("_note", None)
            cards.append(cur)
        cur = None

    while i < len(lines):
        line = lines[i]
        m = MODULE_RE.match(line)
        if m:
            flush()
            module, module_name = m.group(1), _clean(m.group(2))
            i += 1
            continue
        c = CARD_RE.match(line)
        if c:
            flush()
            cur = {"id": c.group(1), "title": _clean(c.group(2)),
                   "module": module, "module_name": module_name}
            mode = ""
            i += 1
            continue
        if cur is not None:
            b = BULLET_RE.match(line)
            if b:
                key = _clean(b.group(1)).lower().replace(" / ", "_").replace(" ", "_")
                cur[key] = _clean(b.group(2))
                mode = ""
                i += 1
                continue
            if line.startswith("**Stimulus:**"):
                mode = "stimulus"
                i += 1
                continue
            if line.startswith("**Protocol note:**"):
                mode = "note"
                cur["_note"] = line.split("**Protocol note:**", 1)[1]
                i += 1
                continue
            if line.startswith("####") or line.startswith("###"):
                mode = ""
            elif mode == "stimulus":
                s = line.strip()
                if s.startswith(">"):
                    s = s.lstrip(">").strip()
                cur["_stim"] = (cur.get("_stim", "") + " " + s).strip()
            elif mode == "note":
                s = line.strip()
                if s and not s.startswith("-"):
                    cur["_note"] = (cur.get("_note", "") + " " + s).strip()
                elif s.startswith("-"):
                    mode = ""
        i += 1
    flush()
    return cards


# ---------------------------------------------------------------- ledger
def parse_ledger(text: str) -> list[dict]:
    lines = text.splitlines()
    entries: list[dict] = []
    kind = ""
    cur: dict | None = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            entries.append(cur)
        cur = None

    for line in lines:
        if line.startswith("### 12.1"):
            flush()
            kind = "image"
            continue
        if line.startswith("### 12.2"):
            flush()
            kind = "audio"
            continue
        if line.startswith("## 13"):
            flush()
            break
        m = LEDGER_RE.match(line)
        if m:
            flush()
            cur = {"id": m.group(1), "kind": kind, "title": _clean(m.group(2))}
            continue
        if cur is not None:
            b = BULLET_RE.match(line)
            if not b:
                continue
            # normalise "Media URL" / "Type / task duration" -> snake_case,
            # otherwise every multi-word field silently fails to match below
            key = _norm_key(b.group(1))
            val = b.group(2)
            if key == "role":
                seg = _segments(_clean(val))
                cur["role"] = seg.get("role") or seg.get("", "")
                cur["status"] = seg.get("status", "")
            elif key == "creator":
                seg = _segments(_clean(val))
                cur["creator"] = seg.get("creator") or seg.get("", "")
                cur["license"] = seg.get("indexed_license", "")
                cur["license_url"] = _first_link(b.group(2))
            elif key == "openverse_id":
                ids = BACKTICK_RE.search(b.group(2))
                cur["openverse_id"] = ids.group(1) if ids else _clean(val)
                cur["api_record"] = _first_link(b.group(2))
                m2 = re.search(r"/v1/(images|audio)/", cur["api_record"])
                if m2:
                    cur["openverse_endpoint"] = m2.group(1)
            elif key == "original_source":
                cur["source_page"] = _first_link(b.group(2))
                cur["source_name"] = _clean_label(val)
            elif key == "media_url":
                cur["media_url"] = _first_link(b.group(2))
                cur["media_label"] = _clean_label(val)
            elif key.startswith("indexed_source_duration") or key.startswith("source_duration"):
                dur = re.search(r"([\d.]+)\s*s", val)
                cur["source_duration_s"] = float(dur.group(1)) if dur else None
                ex = re.search(r"trial excerpt:?\s*([^;]+)", _clean_label(val))
                cur["trial_excerpt"] = ex.group(1).strip() if ex else ""
            elif key == "retain_attribution":
                cur["attribution"] = _clean(val)
            else:
                # never drop a field we did not anticipate: a future revision of
                # the document must not be able to lose information silently
                cur.setdefault("extra", {})[key] = _clean(val)
    flush()
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--force", action="store_true", help="re-copy even if unchanged")
    a = ap.parse_args(argv)

    src: Path = a.source
    if not src.exists():
        print(f"source document not found: {src}")
        return 1
    text = src.read_text()
    digest = sha256(src)

    # 1. verbatim copy + provenance
    import shutil
    DOC_DEST.parent.mkdir(parents=True, exist_ok=True)
    same = DOC_DEST.exists() and sha256(DOC_DEST) == digest
    if not same or a.force:
        shutil.copyfile(src, DOC_DEST)
    PROVENANCE.write_text(json.dumps({
        "source_path": str(src.resolve()),
        "sha256": digest,
        "bytes": src.stat().st_size,
        "lines": text.count("\n") + 1,
        "imported_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "imported_by": "scripts/import_protocol.py",
        "verbatim": True,
        "note": ("the copy is byte-identical to the source; edits belong in the "
                 "source document, and re-running this script refreshes the copy"),
    }, indent=2))
    print(f"document: {DOC_DEST}  ({'already current' if same and not a.force else 'copied'}, "
          f"sha256 {digest[:16]}...)")
    print(f"provenance: {PROVENANCE}")

    # 2. cards
    cards = parse_cards(text)
    by_module: dict[str, int] = {}
    for c in cards:
        by_module[c.get("module", "?")] = by_module.get(c.get("module", "?"), 0) + 1
    import yaml
    CARDS_DEST.write_text(yaml.safe_dump({
        "source": str(src.resolve()),
        "source_sha256": digest,
        "id_scheme": "codex T01..T64 (8 modules A-H); distinct from NeuroBridge T001..T708",
        "n_cards": len(cards),
        "cards": cards,
    }, sort_keys=False, allow_unicode=True, width=100))
    print(f"cards: {CARDS_DEST}  ({len(cards)} parsed, per module {by_module})")

    # 3. ledger
    ledger = parse_ledger(text)
    kinds: dict[str, int] = {}
    for e in ledger:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    LEDGER_DEST.write_text(yaml.safe_dump({
        "source": str(src.resolve()),
        "source_sha256": digest,
        "retrieved": "2026-10-13 (per the document's own header)",
        "caveat": ("licence statements are indexed metadata pending original-source "
                   "verification; images were composition-inspected, audio was not "
                   "auditioned; no hashes existed before this import"),
        "n_entries": len(ledger),
        "entries": ledger,
    }, sort_keys=False, allow_unicode=True, width=100))
    print(f"ledger: {LEDGER_DEST}  ({len(ledger)} entries, per kind {kinds})")

    missing_url = [e["id"] for e in ledger if not e.get("media_url")]
    if missing_url:
        print(f"  ! entries without a media URL: {missing_url}")

    # 4. module -> NeuroBridge tier crosswalk (headline only; the judgement
    #    lives in docs/PROTOCOL_MAPPING.md)
    print("\nmodule summary")
    for m, n in sorted(by_module.items()):
        name = next((c["module_name"] for c in cards if c.get("module") == m), "")
        print(f"  {m}: {n} cards  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
