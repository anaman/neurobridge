"""Test-spec schema and catalog loader.

A *test* is one self-contained stimulation protocol with a known analysis
recipe. 64 tests live in ``catalog/tests/*.yaml`` (split) or in the single
master file ``catalog/tests.yaml``; both forms are accepted.

Feasibility vocabulary — this matters, because a single dry frontal channel
cannot do everything a 64-channel lab cap can:

``established``  replicated in the literature *and* plausible on one frontal
                 dry electrode (band-power states, alpha reactivity, blink and
                 jaw-clench artifact control, auditory oddball at Fz/Fpz).
``promising``    supported by literature, expected to work but with modest
                 effect size or dependence on good electrode contact.
``exploratory``  scientifically interesting, weak or unknown on this montage.
                 Recorded for the dataset, but the acceptance gate is lenient
                 and the report says so.
``control``      a null/sanity protocol with no expected signal; used to
                 measure the false-positive floor of the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import TESTS_DIR, MASTER_CATALOG

FEASIBILITY = ("established", "promising", "exploratory", "control")
MODALITIES = ("text", "image", "audio", "multimodal", "none")
TIERS = {
    0: "Commissioning & signal integrity",
    1: "Arousal, alertness & fatigue",
    2: "Attention & cognitive load",
    3: "Auditory event-related potentials",
    4: "Visual & semantic event-related potentials",
    5: "Reading & language",
    6: "Audio listening & music",
    7: "Imagery, control & closed-loop agent",
}

# Fields every spec must define. Validation errors are collected, not raised,
# so `nb catalog validate` can report every problem in one pass.
REQUIRED = ("id", "slug", "tier", "title", "modality", "feasibility", "goal",
            "classes", "stimuli", "design", "markers", "processing", "features",
            "decoder", "metrics", "acceptance", "risks", "agent_use")

REQUIRED_DESIGN = ("trials", "duration_min", "blocks")


@dataclass
class TestSpec:
    """One stimulation protocol + its analysis recipe."""

    id: str
    slug: str
    tier: int
    title: str
    modality: list[str]
    feasibility: str
    goal: str
    classes: list[str]
    stimuli: dict[str, Any]
    design: dict[str, Any]
    markers: dict[str, int]
    processing: dict[str, Any]
    features: list[Any]
    decoder: dict[str, Any]
    metrics: list[str]
    acceptance: dict[str, Any]
    risks: list[str]
    agent_use: str
    tier_name: str = ""
    notes: str = ""
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # -- derived helpers -------------------------------------------
    @property
    def duration_min(self) -> float:
        return float(self.design.get("duration_min", 0) or 0)

    @property
    def trials(self) -> int:
        return int(self.design.get("trials", 0) or 0)

    @property
    def fs(self) -> float:
        return float(self.processing.get("fs", 250.0))

    @property
    def epoch(self) -> tuple[float, float]:
        e = self.processing.get("epoch", [-0.2, 0.8])
        return float(e[0]), float(e[1])

    @property
    def needs_stimuli(self) -> bool:
        return bool(self.stimuli.get("source") not in (None, "none", "internal"))

    def onset_codes(self) -> list[int]:
        """Marker codes that define an epoch (stimulus-locked events)."""
        keys = self.processing.get("onset_markers") or ["stim_onset", "trial_onset"]
        return [self.markers[k] for k in keys if k in self.markers]

    def code(self, name: str, default: int | None = None) -> int:
        """Resolve a marker name to the code **this test** actually uses.

        The spec's own ``markers`` block always wins: two specs in the same
        tier routinely reuse a code (1210 means something different in T201 and
        T203), so a global name->code table would silently record the wrong
        code. Only names the spec does not define (session/test/block framing)
        fall back to ``store.MARKER_CODES``.
        """
        if name in self.markers:
            return int(self.markers[name])
        from ..store import MARKER_CODES
        if name in MARKER_CODES:
            return int(MARKER_CODES[name])
        if default is not None:
            return int(default)
        raise KeyError(
            f"test {self.id} does not define marker {name!r}; "
            f"add it to the spec's markers block (known here: {sorted(self.markers)[:8]}...) "
            f"or pass a default")


    def label_map(self) -> dict[int, str]:
        """marker code -> class label, using ``markers`` entries named after classes."""
        out: dict[int, str] = {}
        for cls in self.classes:
            key = cls.lower().replace(" ", "_").replace("-", "_")
            if key in self.markers:
                out[int(self.markers[key])] = cls
        return out

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.raw.items()}
        d.update({
            "id": self.id, "slug": self.slug, "tier": self.tier, "title": self.title,
            "modality": self.modality, "feasibility": self.feasibility, "goal": self.goal,
            "classes": self.classes, "stimuli": self.stimuli, "design": self.design,
            "markers": self.markers, "processing": self.processing,
            "features": self.features, "decoder": self.decoder, "metrics": self.metrics,
            "acceptance": self.acceptance, "risks": self.risks, "agent_use": self.agent_use,
        })
        if self.notes:
            d["notes"] = self.notes
        return d

    def one_line(self) -> str:
        return (f"{self.id}  [{self.feasibility:11s}] {self.title}  "
                f"({', '.join(self.modality)}, {self.duration_min:.0f} min)")

    def __repr__(self) -> str:
        return f"<TestSpec {self.id} {self.slug}>"


# ---------------------------------------------------------------- validation
def validate(raw: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems ([] == valid)."""
    errs: list[str] = []
    if not isinstance(raw, dict):
        return ["spec is not a mapping"]
    for k in REQUIRED:
        if k not in raw or raw[k] in (None, ""):
            errs.append(f"missing required field {k!r}")
    tid = str(raw.get("id", ""))
    if tid and not (len(tid) == 4 and tid[0] == "T" and tid[1:].isdigit()):
        errs.append(f"id {tid!r} must look like T301")
    mods = raw.get("modality") or []
    if isinstance(mods, str):
        mods = [mods]
    for m in mods:
        if m not in MODALITIES:
            errs.append(f"unknown modality {m!r} (allowed: {MODALITIES})")
    feas = raw.get("feasibility")
    if feas not in FEASIBILITY:
        errs.append(f"feasibility {feas!r} not in {FEASIBILITY}")
    tier = raw.get("tier")
    if not isinstance(tier, int) or tier not in TIERS:
        errs.append(f"tier {tier!r} must be one of {sorted(TIERS)}")
    design = raw.get("design") or {}
    for k in REQUIRED_DESIGN:
        if k not in design:
            errs.append(f"design missing {k!r}")
    if design.get("blocks") and design.get("trials"):
        try:
            if int(design["trials"]) < int(design["blocks"]):
                errs.append("design.trials must be >= design.blocks")
        except (TypeError, ValueError):
            errs.append("design.trials/blocks must be integers")
    classes = raw.get("classes") or []
    if len(classes) < 1:
        errs.append("classes needs at least one label")
    if len(classes) > 6:
        errs.append(f"{len(classes)} classes is unrealistic for 1 frontal channel")
    markers = raw.get("markers") or {}
    if not markers:
        errs.append("markers must define at least one code")
    if len(set(markers.values())) != len(markers):
        dupes = {v for v in markers.values() if list(markers.values()).count(v) > 1}
        errs.append(f"marker codes must be unique within a test (dupes: {sorted(dupes)})")
    for v in markers.values():
        if not isinstance(v, int) or not (0 < v < 100000):
            errs.append(f"marker code {v!r} must be an int in 1..99999")
    ep = (raw.get("processing") or {}).get("epoch")
    if ep is not None:
        try:
            a, b = float(ep[0]), float(ep[1])
            if not a < 0 < b:
                errs.append(f"epoch {ep} must straddle 0 (e.g. [-0.2, 0.8])")
            if b - a > 10:
                errs.append(f"epoch {ep} longer than 10 s is almost certainly wrong")
        except Exception:
            errs.append(f"epoch {ep!r} must be two numbers")
    acc = raw.get("acceptance") or {}
    if not isinstance(acc, dict) or not acc:
        errs.append("acceptance must define at least one metric gate")
    for k, v in acc.items():
        if not isinstance(v, (int, float)):
            errs.append(f"acceptance.{k} must be numeric, got {v!r}")
    if feas == "control" and acc and max(float(x) for x in acc.values()) > 0.75:
        errs.append("control tests must have a lenient gate (<= 0.75): they should NOT succeed")
    if not raw.get("risks"):
        errs.append("risks must list at least one honest failure mode")
    if raw.get("agent_use") not in ("none", "state", "command", "both"):
        errs.append("agent_use must be one of none|state|command|both")
    return errs


def from_dict(raw: dict[str, Any], source_path: Path | None = None) -> TestSpec:
    tier = int(raw.get("tier", 0))
    mods = raw.get("modality") or []
    if isinstance(mods, str):
        mods = [mods]
    return TestSpec(
        id=str(raw["id"]).strip(),
        slug=str(raw.get("slug") or raw["id"]).strip(),
        tier=tier,
        title=str(raw.get("title", "")).strip(),
        modality=list(mods),
        feasibility=str(raw.get("feasibility", "exploratory")),
        goal=str(raw.get("goal", "")).strip(),
        classes=list(raw.get("classes") or []),
        stimuli=dict(raw.get("stimuli") or {}),
        design=dict(raw.get("design") or {}),
        markers={k: int(v) for k, v in (raw.get("markers") or {}).items()},
        processing=dict(raw.get("processing") or {}),
        features=list(raw.get("features") or []),
        decoder=dict(raw.get("decoder") or {}),
        metrics=list(raw.get("metrics") or []),
        acceptance=dict(raw.get("acceptance") or {}),
        risks=list(raw.get("risks") or []),
        agent_use=str(raw.get("agent_use", "none")),
        tier_name=TIERS.get(tier, ""),
        notes=str(raw.get("notes", "") or ""),
        source_path=source_path,
        raw=dict(raw),
    )


# ---------------------------------------------------------------- loading
def _load_yaml(path: Path) -> Any:
    import yaml
    return yaml.safe_load(path.read_text())


def load_specs(directory: Path | None = None,
               master: Path | None = None) -> list[TestSpec]:
    """Load every spec, preferring split files over the master catalog."""
    directory = Path(directory or TESTS_DIR)
    master = Path(master or MASTER_CATALOG)
    specs: list[TestSpec] = []
    files = sorted(directory.glob("*.yaml")) if directory.exists() else []
    if files:
        for f in files:
            data = _load_yaml(f)
            if isinstance(data, list):
                specs += [from_dict(d, f) for d in data]
            elif isinstance(data, dict) and "tests" in data:
                specs += [from_dict(d, f) for d in data["tests"]]
            elif isinstance(data, dict):
                specs.append(from_dict(data, f))
    elif master.exists():
        data = _load_yaml(master)
        items = data.get("tests", data) if isinstance(data, dict) else data
        specs = [from_dict(d, master) for d in items]
    specs.sort(key=lambda s: s.id)
    return specs


def load_catalog(directory: Path | None = None, master: Path | None = None) -> dict[str, TestSpec]:
    """id -> TestSpec mapping."""
    return {s.id: s for s in load_specs(directory, master)}


def load_one(test_id: str, directory: Path | None = None,
             master: Path | None = None) -> TestSpec:
    """Fetch a spec by exact id (T301) or by slug/prefix match."""
    specs = load_catalog(directory, master)
    key = test_id.strip().upper()
    if key in specs:
        return specs[key]
    hits = [s for k, s in specs.items()
            if s.slug == test_id.strip() or k.startswith(key) or test_id.lower() in s.slug.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no test matches {test_id!r} (have {len(specs)} specs)")
    raise KeyError(f"{test_id!r} is ambiguous: {[h.id for h in hits]}")


def validate_catalog(directory: Path | None = None,
                     master: Path | None = None) -> dict[str, list[str]]:
    """Validate every spec; returns {test_id_or_file: [problems]} ({} == all good)."""
    directory = Path(directory or TESTS_DIR)
    master = Path(master or MASTER_CATALOG)
    problems: dict[str, list[str]] = {}
    files = sorted(directory.glob("*.yaml")) if directory.exists() else []
    if files:
        for f in files:
            data = _load_yaml(f)
            items = data if isinstance(data, list) else (
                data.get("tests", [data]) if isinstance(data, dict) else [])
            for d in items:
                errs = validate(d)
                if errs:
                    problems[str(d.get("id", f.name))] = errs
    elif master.exists():
        data = _load_yaml(master)
        items = data.get("tests", data) if isinstance(data, dict) else data
        for d in items:
            errs = validate(d)
            if errs:
                problems[str(d.get("id", "?"))] = errs
    seen: dict[str, str] = {}
    for s in load_specs(directory, master):
        if s.id in seen:
            problems.setdefault(s.id, []).append(f"duplicate id (also in {seen[s.id]})")
        seen[s.id] = str(s.source_path)
    return problems


def tier_summary(specs: Iterable[TestSpec]) -> dict[int, list[TestSpec]]:
    out: dict[int, list[TestSpec]] = {}
    for s in specs:
        out.setdefault(s.tier, []).append(s)
    return dict(sorted(out.items()))
