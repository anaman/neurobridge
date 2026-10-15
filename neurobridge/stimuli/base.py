"""Stimulus model shared by the fetchers and the presentation runner.

A ``TestSpec`` describes *what* it needs (``stimuli: {source, queries, ...}``);
:func:`prepare` resolves that into concrete, cached, local files plus a text
bank. The runner then only ever deals with :class:`Stimulus` objects, so it
does not care whether an image came from Openverse, an audio clip from
Freesound, or a locally synthesized fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import AUDIO_DIR, IMAGE_DIR, TEXT_DIR, ensure_dirs

# ---------------------------------------------------------------- dataclasses
@dataclass
class Stimulus:
    """One presentable item."""

    kind: str                       # image | audio | text | tone | none
    label: str = ""                 # class label from the spec
    path: Path | None = None        # file on disk (image/audio)
    text: str = ""                  # inline text (text stimuli)
    duration_s: float | None = None # intended presentation duration
    source: str = "unknown"         # openverse | freesound | synth | textbank | builtin
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return str(self.path) if self.path is not None else (self.text[:40] or self.kind)

    def exists(self) -> bool:
        return self.path is None or Path(self.path).exists()

    def __repr__(self) -> str:
        where = self.path.name if self.path else (self.text[:28] + "…" if self.text else "-")
        return f"<Stimulus {self.kind} {self.label!r} {where}>"


@dataclass
class StimulusSet:
    """Everything one test needs, ready to present."""

    spec_id: str
    items: list[Stimulus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sources: dict[str, int] = field(default_factory=dict)
    #: True when the spec asks for no external stimuli (``source: none``), so an
    #: empty set is correct rather than a failure.  Set by ``builder.build``.
    device_only: bool = False

    def by_label(self) -> dict[str, list[Stimulus]]:
        out: dict[str, list[Stimulus]] = {}
        for s in self.items:
            out.setdefault(s.label, []).append(s)
        return out

    def labels(self) -> list[str]:
        seen: list[str] = []
        for s in self.items:
            if s.label not in seen:
                seen.append(s.label)
        return seen

    def of_kind(self, kind: str) -> list[Stimulus]:
        return [s for s in self.items if s.kind == kind]

    def check(self) -> list[str]:
        """Problems that would make a session useless (missing files, broken links).

        An empty set is a problem *only* when stimuli were expected.  A device-only
        test (``stimuli.source: none``, e.g. the blink switch T701 or the resting
        baseline T002) legitimately presents no external stimulus, so the builder
        marks such a set ``device_only`` and this method reports an empty set as
        an informational note instead of a failure.  Flagging those as broken was
        a false alarm that made every offline smoke test look like a defect.
        """
        problems: list[str] = []
        if not self.items:
            if not self.device_only:
                problems.append("no stimuli resolved at all")
        missing = [s for s in self.items if not s.exists()]
        if missing:
            problems.append(f"{len(missing)} stimulus files are missing from disk")
        counts = {k: len(v) for k, v in self.by_label().items()}
        if counts and min(counts.values()) == 0:
            problems.append(f"a class has no stimuli: {counts}")
        for w in self.warnings:
            problems.append(w)
        return problems

    def informational(self) -> list[str]:
        """Non-fatal remarks: device-only sets and deliberately empty classes."""
        notes = list(self.notes)
        if not self.items and self.device_only:
            notes.append("device-only test: no external stimuli by design")
        return notes

    def summary(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "n_items": len(self.items),
            "per_label": {k: len(v) for k, v in self.by_label().items()},
            "sources": self.sources,
            "warnings": self.warnings,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Stimulus]:
        return iter(self.items)


#: Where each stimulus kind is cached when downloaded or generated.
KIND_DIRS: dict[str, Path] = {
    "image": IMAGE_DIR,
    "audio": AUDIO_DIR,
    "tone": AUDIO_DIR,
    "text": TEXT_DIR,
}


def kind_dir(kind: str) -> Path:
    ensure_dirs()
    d = KIND_DIRS.get(kind, IMAGE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- entry point
def prepare(spec, allow_download: bool = True, force: bool = False,
            verbose: bool = True, limit: int | None = None) -> StimulusSet:
    """Resolve a spec's ``stimuli`` block into local, presentable items.

    Never raises for a missing API token or an offline network: it degrades to
    the spec's ``fallback`` (synthesized audio, a small built-in image set, or
    the text bank) and records exactly what happened in ``warnings``.
    """
    from .builder import build

    return build(spec, allow_download=allow_download, force=force,
                 verbose=verbose, limit=limit)
