"""Frozen development / locked-evaluation splits.

Adopted from the imported external protocol (docs/PROTOCOL_MAPPING.md, Module H).
The rule it enforces is the difference between a demo and a result:

    decide which recordings are for development and which are locked **before**
    training, write that decision down, and never fit on the locked ones.

Without this, "cross-validated accuracy" quietly becomes "accuracy on data we
looked at while choosing features". Three properties make the manifest worth
trusting:

* **Splits are by session, never by trial or window.** A session shares one
  electrode seating, one posture, one fatigue level and one drift trajectory.
  Splitting inside a session leaks all of that across the boundary. Where a
  later claim is about *new stimuli*, the grouping unit should be the source
  stimulus family, but the session boundary is the non-negotiable one.
* **The locked set is the newest sessions by default.** The honest question is
  whether the decoder survives a new day with a new contact, not whether it
  survives a random subset of today.
* **The manifest is frozen.** Re-creating an existing split requires an explicit
  ``force``, and a forced revision records itself as such, so a later reader can
  see that the lock was moved and when.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

from .config import SESSION_DIR, ensure_dirs, list_sessions

SPLIT_DIR = SESSION_DIR.parent / "splits"


class LockedSessionError(RuntimeError):
    """Raised when a caller tries to fit on a session marked locked."""


@dataclass
class Split:
    """A frozen development / locked-evaluation partition of sessions."""

    name: str = "default"
    spec_id: str | None = None
    created_wall: str = ""
    dev: list[str] = field(default_factory=list)
    locked: list[str] = field(default_factory=list)
    rule: str = ""
    note: str = ""
    revised_wall: str | None = None
    revision: int = 1

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def n_dev(self) -> int:
        return len(self.dev)

    @property
    def n_locked(self) -> int:
        return len(self.locked)

    def is_locked(self, session_id: str) -> bool:
        return session_id in self.locked

    def report_paths(self) -> dict[str, list[Path]]:
        return {"dev": [SESSION_DIR / s for s in self.dev],
                "locked": [SESSION_DIR / s for s in self.locked]}

    def to_markdown(self) -> str:
        lines = [f"# Split `{self.name}`", "",
                 f"- spec: `{self.spec_id or 'all tests'}`",
                 f"- created: {self.created_wall}"
                 + (f" (revised {self.revised_wall}, revision {self.revision})"
                    if self.revised_wall else ""),
                 f"- rule: {self.rule}",
                 f"- development sessions ({self.n_dev}): {', '.join(self.dev) or '—'}",
                 f"- **locked** sessions ({self.n_locked}): "
                 f"{', '.join(self.locked) or '—'}", ""]
        if self.note:
            lines += [self.note, ""]
        lines += ["Locked sessions are never used for fitting. `nb train` refuses "
                  "them unless `--force`; `nb eval --locked` is how they are used.", ""]
        return "\n".join(lines)


# ---------------------------------------------------------------- storage
def manifest_path(name: str = "default") -> Path:
    return SPLIT_DIR / f"{name}.json"


def list_splits() -> list[str]:
    if not SPLIT_DIR.exists():
        return []
    return sorted(p.stem for p in SPLIT_DIR.glob("*.json"))


def load_split(name: str = "default") -> Split:
    p = manifest_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"no split named {name!r}; existing: {list_splits() or 'none'} "
            f"(create one with `nb split create`)")
    return Split.from_dict(json.loads(p.read_text()))


def save_split(split: Split, force: bool = False) -> Path:
    """Write the manifest. Refuses to overwrite a frozen split without ``force``."""
    ensure_dirs()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    p = manifest_path(split.name)
    if p.exists() and not force:
        raise FileExistsError(
            f"split {split.name!r} already exists and is frozen: {p}. "
            f"Pass force=True (CLI: --force) to revise it; the revision is recorded.")
    if p.exists():
        old = Split.from_dict(json.loads(p.read_text()))
        split.revision = old.revision + 1
        split.revised_wall = time.strftime("%Y-%m-%dT%H:%M:%S")
    elif not split.created_wall:
        split.created_wall = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(split.as_dict(), indent=2))
    p.with_suffix(".md").write_text(split.to_markdown())
    return p


# ---------------------------------------------------------------- creation
def sessions_for(spec_id: str | None = None, root: Path | None = None) -> list[str]:
    """Session ids (oldest first) that contain ``spec_id`` (or all sessions)."""
    out: list[str] = []
    for d in (list_sessions() if root is None else sorted(root.iterdir())):
        if not (d / "meta.json").exists():
            continue
        if spec_id is None:
            out.append(d.name)
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            continue
        # cheap check first, then the authoritative event scan
        ids = meta.get("tests") or []
        if spec_id in ids:
            out.append(d.name)
            continue
        events = d / "events.jsonl"
        if events.exists():
            for line in events.read_text().splitlines():
                if f'"test_id": "{spec_id}"' in line:
                    out.append(d.name)
                    break
    return out


def create_split(spec_id: str | None = None, name: str = "default",
                 dev_frac: float = 0.7, seed: int | None = None,
                 session_ids: Sequence[str] | None = None,
                 lock_newest: bool = True, note: str = "",
                 force: bool = False) -> Split:
    """Partition sessions into development and locked sets.

    ``lock_newest`` (default) locks the most recent sessions, which is the only
    split that answers "does this survive a new day". With ``lock_newest=False``
    the locked set is drawn at random from the pool, which is a weaker claim and
    is recorded as such in ``rule``.
    """
    ids = list(session_ids) if session_ids is not None else sessions_for(spec_id)
    if len(ids) < 2:
        raise ValueError(
            f"need at least 2 sessions containing {spec_id or 'any test'} to split, "
            f"found {len(ids)}: {ids}. Record more sessions first "
            f"(`nb run {spec_id or '<test>'}` or `nb simulate ...`).")
    rng = random.Random(seed)
    if lock_newest:
        n_locked = max(1, min(len(ids) - 1, round(len(ids) * (1.0 - dev_frac))))
        dev, locked = ids[:-n_locked], ids[-n_locked:]
        rule = (f"development = oldest {len(dev)}, locked = newest {len(locked)} "
                f"(lock_newest=True); grouped by session, never by trial")
    else:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_dev = max(1, min(len(ids) - 1, round(len(ids) * dev_frac)))
        dev, locked = sorted(shuffled[:n_dev]), sorted(shuffled[n_dev:])
        rule = (f"random session split with seed={seed} (lock_newest=False); "
                f"weaker claim than a held-out day")
    split = Split(name=name, spec_id=spec_id, dev=dev, locked=locked, rule=rule, note=note,
                  created_wall=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_split(split, force=force)
    return split


# ---------------------------------------------------------------- enforcement
def check_not_locked(session_ids: Iterable[str | Path], split_name: str = "default",
                     force: bool = False) -> list[str]:
    """Raise if any id is in the locked set; returns the ids otherwise."""
    ids = [Path(s).name if not isinstance(s, str) else Path(s).name for s in session_ids]
    try:
        split = load_split(split_name)
    except FileNotFoundError:
        return ids                     # nothing frozen yet: nothing to violate
    offenders = [s for s in ids if split.is_locked(s)]
    if offenders and not force:
        raise LockedSessionError(
            f"{offenders} are in the LOCKED set of split {split_name!r}. Fitting on "
            f"them would invalidate every held-out number in the report. Use the "
            f"development set, or pass force=True and say so in the write-up.")
    return ids


def locked_sessions(split_name: str = "default") -> list[Path]:
    split = load_split(split_name)
    return [SESSION_DIR / s for s in split.locked]


def dev_sessions(split_name: str = "default") -> list[Path]:
    split = load_split(split_name)
    return [SESSION_DIR / s for s in split.dev]
