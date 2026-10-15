"""Resolution of a spec's ``stimuli`` block into presentable, cached items.

This is the only place that knows *both* the catalog vocabulary (``source``,
``queries``, ``fallback``, ``n_per_class``) and the acquisition back ends
(Openverse, Freesound, the synthesizer, the text bank, the built-in images).

The governing rule
------------------
**A session must always be able to run.**  Network, tokens and displays are all
optional in this project, so every remote path is wrapped: on any failure the
builder substitutes the spec's ``fallback``, records a warning that names the
cause *and* the substitution, and keeps going.  Nothing here raises for a
missing resource; the only exceptions that escape are programming errors
(``KeyError`` for a misspelled textbank category or synth generator), which must
fail loudly rather than silently produce an empty block.

Cache-first, always
-------------------
Every lookup checks :class:`StimulusCache` before touching the network, and a
cache hit is used even when ``allow_download=False``: "do not download" means
"do not use the network", not "ignore the 400 files you already have".

Query strings do double duty
----------------------------
The tier 6/7 specs are written as::

    source: freesound
    queries: {nature: ["forest rain", "noise_pink"], urban: ["city traffic", "noise_brown"]}

The *first* string is a Freesound search query and the *second* is the offline
synth generator to use when Freesound is unavailable.  So a list of remote
queries doubles as an ordered fallback list, which is why those specs can
promise a runnable test with no token at all.  A spec can also name a generator
*first* (``source: mixed`` with ``queries: {noise: ["noise_pink"]}``), and the
builder honours that for a mixed source.

Empty is not an error
---------------------
``queries: {focused: [], mind_wandering: []}`` (T208) and ``fixation: []``
(T508) are intentional: the class is present for marker/logging purposes but has
no external stimulus.  Those classes resolve to zero items **without a warning**;
the set records an informational note instead, so an operator can tell a
deliberate omission from a failed download.
"""
from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..catalog.schema import TestSpec, load_one
from ..config import get_settings
from . import builtin, synth, textbank
from .base import Stimulus, StimulusSet
from .cache import CachedFile, StimulusCache, cache_key
from .freesound import AuthError, FreesoundClient, NoTokenError
from .openverse import NetworkError, NoResultsError, OpenverseClient

#: Ordered default generators when nothing else is available.  Chosen so the
#: fallback is *acoustically plausible for the test family*: an oddball gets a
#: tone, a speech test gets a syllable or pseudo-speech, an affective test gets
#: the calm ambience rather than white noise.
DEFAULT_AUDIO_FALLBACK = ("tone_1000", "noise_pink", "pseudo_speech", "click")
SPEECH_HINTS = ("speech", "syllable", "word", "name", "narrative", "prosody",
                "voice", "talk", "story", "sentence")
NOISE_HINTS = ("noise", "static", "hiss", "rain", "traffic")
MUSIC_HINTS = ("music", "melody", "chord", "beat", "tempo", "song")
IMAGE_FALLBACK_DEFAULT = "standard"
_NUM_RE = _re.compile(r"\d+")
IMAGE_KINDS = ("image", "multimodal", "multimap", "picture")
AUDIO_KINDS = ("audio", "tone", "multimodal")


# ---------------------------------------------------------------- small helpers
def _as_list(value: Any) -> list[str]:
    """Normalise a queries entry (``str`` | ``list`` | ``None``) to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _hint_generator(label: str, queries: Sequence[str]) -> str:
    """A *sensible* default generator for a class with no usable fallback name.

    Without this, a class whose queries are all remote ("city traffic") would get
    an arbitrary reference tone.  The hints make the substitution defensible: a
    speech class gets pseudo-speech, a noise class gets noise.
    """
    blob = f"{label} {' '.join(queries)}".lower()
    if any(h in blob for h in SPEECH_HINTS):
        return "pseudo_speech"
    if any(h in blob for h in NOISE_HINTS):
        return "noise_pink"
    if any(h in blob for h in MUSIC_HINTS):
        return "melody_unfamiliar"
    if "tone" in blob or "hz" in blob or "sine" in blob:
        return "tone_1000"
    return DEFAULT_AUDIO_FALLBACK[0]


def _cached_chain(cache: StimulusCache, key: str, n: int) -> list[CachedFile]:
    """Cache hits for a request, honouring the fetchers' ``key``/``key:i`` scheme.

    The fetchers store the first item under the bare request key and later items
    under ``key:1``, ``key:2``...  Reproducing that here lets the builder use the
    cache without a network probe, which is what makes an offline run reuse a
    previously warmed cache.
    """
    out: list[CachedFile] = []
    if n <= 0:
        return out
    first = cache.has(key)
    if first is not None:
        out.append(first)
    i = 1
    while len(out) < n:
        hit = cache.has(f"{key}:{i}")
        if hit is None:
            break
        out.append(hit)
        i += 1
    return out


def _duration(spec: TestSpec) -> float | None:
    raw = spec.stimuli.get("duration_s")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _image_stimulus(cf: CachedFile, label: str, duration_s: float | None) -> Stimulus:
    return Stimulus(
        kind="image", label=label, path=Path(cf.path), duration_s=duration_s,
        source=cf.source,
        meta={"title": cf.title, "license": cf.license, "attribution": cf.attribution,
              "url": cf.url, "sha1": cf.sha1, "query": cf.query, **(cf.meta or {})},
    )


def _audio_stimulus(cf: CachedFile, label: str, duration_s: float | None,
                    generator: str = "") -> Stimulus:
    return Stimulus(
        kind="audio", label=label, path=Path(cf.path), duration_s=duration_s,
        source=cf.source,
        meta={"generator": generator, "license": cf.license, "attribution": cf.attribution,
              "url": cf.url, "sha1": cf.sha1, "query": cf.query, **(cf.meta or {})},
    )


def _count_sources(stimuli: Iterable[Stimulus]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in stimuli:
        out[s.source] = out.get(s.source, 0) + 1
    return dict(sorted(out.items()))


def _limit_per_class(spec: TestSpec, classes: Sequence[str], limit: int | None
                     ) -> tuple[int, list[str]]:
    """Items per class after honouring ``limit`` and ``max_items_per_test``."""
    warnings: list[str] = []
    want = int(spec.stimuli.get("n_per_class", 1) or 1)
    n_classes = max(1, len(classes))
    per = max(1, int(limit)) if limit is not None else max(1, want)
    cap = int(get_settings().max_items_per_test or 0)
    if cap:
        allowed = max(1, cap // n_classes)
        if per > allowed:
            warnings.append(
                f"n_per_class={per} x {n_classes} classes exceeds "
                f"max_items_per_test={cap}: truncated to {allowed} per class."
            )
            per = allowed
    return per, warnings


def _generator_hint(spec: TestSpec) -> str:
    """``stimuli.generator`` (T005: ``flash_frames``) mapped to a builtin category."""
    raw = str(spec.stimuli.get("generator", "") or "").strip().lower()
    if not raw:
        return ""
    for key in (raw, raw[:-len("_frames")] if raw.endswith("_frames") else raw):
        if key and builtin.is_known(key):
            return key
    return raw[:-len("_frames")] if raw.endswith("_frames") else raw


def _looks_like_generator(value: str) -> bool:
    return synth.is_generator(str(value or ""))


def _class_levels(spec: TestSpec, classes: Sequence[str]) -> dict[str, float]:
    """Map class -> target level in dBFS for a spec's loudness ladder.

    T607 writes ``levels_dbfs: [-30, -24, -18, -12]`` with
    ``level_to_class: "levels 1-2 -> quiet, levels 3-4 -> loud"``.  The ladder is
    applied as a *post-gain* on an otherwise identical noise generator, so the
    ``quiet`` and ``loud`` classes differ in level and in nothing else -- which is
    the entire point of a loudness/annoyance ladder.

    Returns ``{}`` when the spec has no ladder or no parseable mapping; classes
    are then presented at the generator's default level.
    """
    cfg = spec.stimuli if isinstance(spec.stimuli, dict) else {}
    levels = cfg.get("levels_dbfs")
    if not isinstance(levels, (list, tuple)) or not levels:
        return {}
    try:
        vals = [float(v) for v in levels]
    except (TypeError, ValueError):
        return {}
    mapping = cfg.get("level_to_class")
    if isinstance(mapping, dict):
        out_map = {str(k): float(v) for k, v in mapping.items()}
        return {c: out_map[c] for c in classes if c in out_map}
    if isinstance(mapping, str):
        out: dict[str, float] = {}
        for chunk in mapping.replace(";", ",").split(","):
            nums = [int(n) for n in _NUM_RE.findall(chunk)]
            label = chunk.split("->")[-1].strip() if "->" in chunk else ""
            if not nums or not label:
                continue
            chosen = [vals[n - 1] for n in nums if 0 < n <= len(vals)]
            if chosen:
                out[label] = sum(chosen) / len(chosen)
        return {c: out[c] for c in classes if c in out}
    return {}


def _synth_params(spec: TestSpec, label: str, names: Sequence[str],
                  cfg: dict[str, Any], class_levels: dict[str, float] | None = None
                  ) -> dict[str, Any]:
    """Spec-supplied synth overrides for one class.

    A spec may carry e.g. ``tone_duration_ms: {standard: 100, deviant: 50}``
    (T303, the duration deviant) either as a mapping keyed by class or as a
    single scalar applied to every class.

    ``stimuli.duration_s`` is passed as ``dur`` **only for generators whose
    length is a free parameter** (tones, noise, AM, silence, click trains).
    Anything with an intrinsic duration -- a melody, a chord, a metronome bar, an
    ambience -- keeps its own length: stretching the Ode to Joy to 30 s (or
    clipping it to 0.1 s) would destroy the very contrast the test is built on
    (T604 familiar vs unfamiliar melody).  For sustained-state tests the runner
    loops the item for the block duration instead.
    """
    kw: dict[str, Any] = {}
    for key, value in cfg.items():
        if not isinstance(value, dict):
            continue
        picked = value.get(label)
        if picked is None:
            continue
        if key in ("tone_duration_ms", "duration_ms", "dur_ms"):
            kw["tone_duration_ms"] = picked
        elif key in ("tone_freq_hz", "freq_hz", "frequency_hz", "freq"):
            kw["freq"] = picked
        elif key in ("amp", "amplitude"):
            kw["amp"] = picked
        elif key == "seed":
            kw["seed"] = picked
    dur = _duration(spec)
    if dur and "tone_duration_ms" not in kw and "dur" not in kw:
        token = synth.generator_token(str(names[0]) if names else "")
        if not synth.is_intrinsic(token):
            kw["dur"] = dur
    level = (class_levels or {}).get(label)
    if level is not None:
        kw["target_dbfs"] = float(level)
    return kw


#: Class labels that conventionally mean "no external stimulus": a rest,
#: fixation or blank period.  An *absent* textbank category for one of these is
#: treated as intentional rather than inferred, because inventing text for a
#: fixation cross would corrupt the baseline it exists to provide.
PLACEHOLDER_LABELS = ("rest", "fixation", "blank", "baseline", "quiet", "none",
                      "eyes_open", "eyes_closed")


def _text_category(spec: TestSpec, label: str, queries: dict[str, list[str]],
                   warnings: list[str]) -> str | None:
    """Pick a textbank category for a class whose ``queries`` entry is *absent*.

    T501 declares ``queries: {reading: passages, rest: []}``: the empty list is
    explicit, so ``rest`` means "no text" and resolves to zero items (a fixation
    period).  When the key is missing entirely we infer:

    1. an exact category name or alias equal to the label;
    2. a single unambiguous substring match (``prose`` for a ``prose`` label);
    3. the first category another class in this spec already uses;
    4. nothing (only for placeholder-like labels) -- the class stays empty.
    """
    key = label.strip().lower()
    valid = textbank.categories()
    if key in valid:
        return key
    if key in PLACEHOLDER_LABELS:
        return None
    if len(key) >= 3:
        close = [c for c in valid if key in c]
        if len(close) == 1:
            return close[0]
    for other in spec.classes or []:
        if str(other) == label:
            continue
        for cand in queries.get(str(other)) or []:
            if cand in valid:
                warnings.append(f"{spec.id}/{label}: no textbank category was named for "
                                f"this class -> re-used {cand!r} from class {other!r}.")
                return cand
    return None


def _text_items(spec: TestSpec, classes: Sequence[str], per: int,
                queries: dict[str, list[str]], warnings: list[str]) -> list[Stimulus]:
    """``kind: text`` -> one :class:`Stimulus` per textbank item (no files)."""
    out: list[Stimulus] = []
    for label in classes:
        if label in queries:
            cats = list(queries.get(label) or [])
            # An explicitly empty query list means "this class has no text"
            # (T508's fixation cross).  Respect it: no warning, no inference.
            if not cats and label.strip().lower() in PLACEHOLDER_LABELS:
                continue
        else:
            cats = []
        if not cats:
            fallback = _text_category(spec, label, queries, warnings)
            if fallback is None:
                continue
            cats = [fallback]
        got = 0
        for cat in cats:
            if got >= per:
                break
            try:
                bank = textbank.items(cat, per)
            except KeyError as exc:
                # A misspelled category is a *spec* bug: fail loudly rather than
                # silently present an empty block.
                raise KeyError(f"{spec.id}: stimuli.queries[{label!r}] -> {exc}") from exc
            if not bank:
                warnings.append(f"{spec.id}/{label}: textbank category {cat!r} is empty; "
                                "no items for this class.")
            for it in bank:
                if got >= per:
                    break
                out.append(Stimulus(
                    kind="text", label=label, text=it.text, duration_s=None,
                    source="textbank",
                    meta={"category": cat, "text_label": it.label, **(it.meta or {})},
                ))
                got += 1
    return out


def _synth_items(spec: TestSpec, classes: Sequence[str], per: int,
                 queries: dict[str, list[str]], warnings: list[str]) -> list[Stimulus]:
    """``source: synth`` -> ``queries[label]`` are generator names.

    Spec-supplied overrides (``tone_duration_ms``, ``duration_s``) are applied
    per class, which is how the MMN duration deviant (T303) gets a 50 ms file for
    ``deviant`` and a 100 ms file for ``standard`` from one generator name.
    """
    out: list[Stimulus] = []
    dur = _duration(spec)
    cfg = spec.stimuli if isinstance(spec.stimuli, dict) else {}
    levels = _class_levels(spec, classes)
    for label in classes:
        names = queries.get(label) or []
        if not names:
            continue                      # intentionally empty class, not a failure
        params = _synth_params(spec, label, names, cfg, levels)
        made = 0
        for name in names:
            if made >= per:
                break
            try:
                path = synth.generate(name, **params)
            except KeyError:
                continue                  # not a generator name: try the next entry
            out.append(Stimulus(kind="audio", label=label, path=Path(path),
                                duration_s=dur, source="synth",
                                meta={"generator": name, "license": "self-authored",
                                      "generator_params": params}))
            made += 1
        if made == 0:
            fallback = _hint_generator(label, names)
            if not synth.is_generator(fallback):
                fallback = DEFAULT_AUDIO_FALLBACK[0]
            path = synth.generate(fallback, **params)
            warnings.append(
                f"{spec.id}/{label}: none of {names} is a known synth generator -> "
                f"used synthesized {fallback!r} instead."
            )
            out.append(Stimulus(kind="audio", label=label, path=Path(path),
                                duration_s=dur, source="synth",
                                meta={"generator": fallback, "license": "self-authored",
                                      "generator_params": params,
                                      "substituted_for": list(names)}))
    return out


def _builtin_items(spec: TestSpec, classes: Sequence[str], per: int,
                   queries: dict[str, list[str]], fallback_hint: str,
                   warnings: list[str]) -> tuple[list[Stimulus], dict[str, int]]:
    """Local procedural images.  Accepts a missing or empty ``queries`` block."""
    out: list[Stimulus] = []
    counts: dict[str, int] = {}
    dur = _duration(spec)
    for label in classes:
        cands = list(queries.get(label) or [])
        if not cands:
            # No queries: use the generator hint, then the class label itself
            # (T005/T103 carry `generator: flash_frames`; T403 uses class names).
            base = label.split("_")[0] if "_" in label else label
            cands = [_generator_hint(spec) or fallback_hint, label, base]
        paths: list[Path] = []
        used = ""
        for cand in cands:
            keys = [str(cand).strip().lower()]
            if keys[0].endswith("_frames"):
                keys.append(keys[0][:-len("_frames")])
            for k in keys:
                if not k:
                    continue
                found = builtin.builtin_images(k)[:per]
                if found:
                    paths, used = found, k
                    break
            if paths:
                break
        counts[label] = len(paths)
        for p in paths:
            out.append(Stimulus(kind="image", label=label, path=Path(p), duration_s=dur,
                                source="builtin",
                                meta={"category": used,
                                      "license": "self-authored (procedural)"}))
    return out, counts


def _openverse_items(spec: TestSpec, classes: Sequence[str], per: int,
                     queries: dict[str, list[str]], client: OpenverseClient,
                     cache: StimulusCache, allow_download: bool, force: bool,
                     warnings: list[str]) -> tuple[list[Stimulus], dict[str, int]]:
    out: list[Stimulus] = []
    counts: dict[str, int] = {}
    dur = _duration(spec)
    lic = spec.stimuli.get("license")
    licence = lic if isinstance(lic, str) else ",".join(_as_list(lic or ["all-cc"]))
    for label in classes:
        qs = queries.get(label) or []
        if not qs:
            counts[label] = 0
            continue
        found: list[CachedFile] = []
        for q in qs:
            key = cache_key(source="openverse", query=q, n=per, license_type=licence,
                            size="", mature=False)
            found = _cached_chain(cache, key, per)
            if found:
                break
            if not allow_download:
                continue
            try:
                found = client.fetch(q, per, cache, force=force)
            except AuthError as exc:
                warnings.append(
                    f"{spec.id}/{label}: Openverse rejected the credentials ({exc}) -> "
                    "check NB_OPENVERSE_CLIENT_ID/_SECRET (or unset them for anonymous "
                    "access); built-in images will be used."
                )
                found = []
            except (NetworkError, NoResultsError) as exc:
                warnings.append(f"{spec.id}/{label}: Openverse {type(exc).__name__} for "
                                f"{q!r} ({exc}) -> tried the next query, then the fallback.")
                found = []
            if found:
                break
        counts[label] = len(found)
        for i, cf in enumerate(found[:per]):
            stim = _image_stimulus(cf, label, dur)
            stim.meta["index"] = i
            out.append(stim)
    return out, counts


def _freesound_items(spec: TestSpec, classes: Sequence[str], per: int,
                     queries: dict[str, list[str]], client: FreesoundClient,
                     cache: StimulusCache, allow_download: bool, force: bool,
                     warnings: list[str]) -> tuple[list[Stimulus], dict[str, int]]:
    out: list[Stimulus] = []
    counts: dict[str, int] = {}
    dur = _duration(spec)
    if not client.available:
        # Instant and offline: this is the common case on a fresh checkout.
        raise NoTokenError(
            "requires a token even to search and none is configured"
        )
    for label in classes:
        qs = queries.get(label) or []
        if not qs:
            counts[label] = 0
            continue
        found: list[CachedFile] = []
        for q in qs:
            key = cache_key(source="freesound", query=q, n=per, license="", sort="score")
            found = _cached_chain(cache, key, per)
            if found:
                break
            if not allow_download:
                continue
            try:
                found = client.fetch(q, per, cache, force=force)
            except NoTokenError:
                raise
            except AuthError as exc:
                warnings.append(f"{spec.id}/{label}: Freesound rejected the token ({exc}) "
                                "-> synthesized audio will be used.")
                found = []
            except (NetworkError, NoResultsError) as exc:
                warnings.append(f"{spec.id}/{label}: Freesound {type(exc).__name__} for "
                                f"{q!r} ({exc}) -> tried the next query, then the fallback.")
                found = []
            if found:
                break
        counts[label] = len(found)
        for cf in found[:per]:
            out.append(_audio_stimulus(cf, label, dur))
    return out, counts


def _synth_fallback(spec: TestSpec, labels: Sequence[str], per: int,
                    queries: dict[str, list[str]], warnings: list[str],
                    *, reason: str, warn: bool = True) -> list[Stimulus]:
    """Synthesize audio for ``labels``: named generators first, hints last.

    Tries every query string against the generator patterns -- that is how
    ``["forest rain", "noise_pink"]`` produces real pink noise with no token --
    then a class-appropriate hint, then a reference tone.

    ``warn=False`` is used when the spec *asked* for a generator by name (a
    ``mixed`` spec with ``queries: {noise: ["noise_pink"]}``): that is the
    intended outcome, not a fallback, and warning about it would train the
    operator to ignore the warning list.
    """
    out: list[Stimulus] = []
    dur = _duration(spec)
    cfg = spec.stimuli if isinstance(spec.stimuli, dict) else {}
    levels = _class_levels(spec, labels)
    for label in labels:
        qs = list(queries.get(label) or [])
        cands = [q for q in qs if synth.is_generator(q)]
        name = cands[0] if cands else _hint_generator(label, qs)
        if not synth.is_generator(name):
            name = DEFAULT_AUDIO_FALLBACK[0]
        params = _synth_params(spec, label, [name] + qs, cfg, levels)
        try:
            path = synth.generate(name, **params)
        except KeyError:
            name = DEFAULT_AUDIO_FALLBACK[0]
            path = synth.generate(name, **params)
        out.append(Stimulus(kind="audio", label=label, path=Path(path), duration_s=dur,
                            source="synth",
                            meta={"generator": name, "license": "self-authored",
                                  "generator_params": params,
                                  "fallback_reason": reason, "substituted_for": qs}))
        if warn:
            tried = f" (query strings tried: {qs})" if qs else ""
            warnings.append(f"{spec.id}/{label}: {reason} -> used synthesized {name!r} "
                            f"instead{tried}.")
    return out


def _remote_then_synth(label: str, queries: list[str], spec: TestSpec, per: int,
                       *, want_image: bool, allow_download: bool,
                       client_ov: OpenverseClient, cache: StimulusCache,
                       warnings: list[str]) -> list[Stimulus]:
    """One class of a ``mixed`` spec: remote first, then synth or builtin."""
    if not queries:
        return []                            # intentionally empty class
    made: list[Stimulus] = []
    if any(_looks_like_generator(q) for q in queries):
        # A named generator is used *directly* and is not a degradation, so this
        # path does not warn: only the offline fallback of a remote query does.
        return _synth_fallback(spec, [label], per, {label: queries}, warnings,
                               reason="synth generator named for this class",
                               warn=False)
    if want_image:
        got, counts = _openverse_items(spec, [label], per, {label: queries}, client_ov,
                                       cache, allow_download, False, warnings)
        if counts.get(label, 0):
            return got
        fb, _ = _builtin_items(spec, [label], per, {label: [label]},
                               IMAGE_FALLBACK_DEFAULT, warnings)
        if fb:
            warnings.append(f"{spec.id}/{label}: Openverse had no image for {queries} -> "
                            f"used the built-in {label!r} image instead.")
        return fb
    try:
        got, counts = _freesound_items(spec, [label], per, {label: queries},
                                       FreesoundClient(), cache, allow_download, False,
                                       warnings)
    except NoTokenError:
        got, counts = [], {label: 0}
        warnings.append(f"{spec.id}/{label}: Freesound requires a token even to search "
                        "and none is configured.")
    if counts.get(label, 0):
        return got
    return _synth_fallback(spec, [label], per, {label: queries}, warnings,
                           reason="no remote query succeeded")


# ---------------------------------------------------------------- builder
def build(spec: TestSpec, allow_download: bool = True, force: bool = False,
          verbose: bool = True, limit: int | None = None) -> StimulusSet:
    """Resolve ``spec.stimuli`` into a ready-to-present :class:`StimulusSet`.

    Never raises for a missing token, an offline network, an empty query list or
    a class with no stimuli.  Raises ``KeyError`` only for a genuinely unknown
    textbank category or synth generator name, which is a spec bug that must be
    fixed rather than papered over.
    """
    settings = get_settings()
    stimuli_cfg = spec.stimuli if isinstance(spec.stimuli, dict) else {}
    source = str(stimuli_cfg.get("source", "none") or "none").strip().lower()
    kind = str(stimuli_cfg.get("kind", "image") or "image").strip().lower()
    classes = [str(c) for c in (spec.classes or [])]
    warnings: list[str] = []
    notes: list[str] = []
    per, limit_warnings = _limit_per_class(spec, classes, limit)
    warnings += limit_warnings

    # Normalise queries: {label: [q, ...]}; a bare list is zipped with classes.
    raw_queries = stimuli_cfg.get("queries") or {}
    if isinstance(raw_queries, (list, tuple)):
        raw_queries = {label: q for label, q in zip(classes, raw_queries)}
    queries: dict[str, list[str]] = {}
    if isinstance(raw_queries, dict):
        for i, (k, v) in enumerate(raw_queries.items()):
            key = str(k)
            if key not in classes and i < len(classes) and not key:
                key = classes[i]
            queries[key] = _as_list(v)
    for label in classes:
        queries.setdefault(label, [])

    fallback = str(stimuli_cfg.get("fallback", "") or "").strip().lower()
    cache = StimulusCache()
    client_ov = OpenverseClient(timeout=float(settings.http_timeout_s or 30.0))

    def say(msg: str) -> None:
        if verbose:
            print(f"  [stimuli] {spec.id}: {msg}")

    # ---------------------------------------------------------- device-only
    if source in ("", "none", "internal") or not classes:
        reason = ("stimuli.source: none" if source in ("", "none", "internal")
                  else "the spec declares no classes")
        notes.append(f"{spec.id} is device-only ({reason}): no external stimuli to "
                     "prepare; the runner presents instructions and countdowns only.")
        say(f"device-only ({reason}) -> no stimuli")
        return StimulusSet(spec_id=spec.id, notes=notes, device_only=True)

    items: list[Stimulus] = []

    if kind == "text" or source == "textbank":
        items = _text_items(spec, classes, per, queries, warnings)

    elif source == "synth":
        items = _synth_items(spec, classes, per, queries, warnings)

    elif source == "builtin":
        hint = _generator_hint(spec) or IMAGE_FALLBACK_DEFAULT
        items, counts = _builtin_items(spec, classes, per, queries, hint, warnings)
        if not items:
            # Last resort: the class labels themselves.
            items, counts = _builtin_items(spec, classes, per,
                                           {c: [c] for c in classes}, hint, warnings)
            if items:
                warnings.append(f"{spec.id}: stimuli.source is 'builtin' but no category "
                                "was named -> the class labels were used as categories.")

    elif source == "openverse":
        items, counts = _openverse_items(spec, classes, per, queries, client_ov, cache,
                                         allow_download, force, warnings)
        missing = [c for c in classes if counts.get(c, 0) == 0 and queries.get(c)]
        if missing:
            fb, _ = _builtin_items(spec, classes, per, queries, IMAGE_FALLBACK_DEFAULT,
                                   warnings)
            fb = [s for s in fb if s.label in set(missing)]
            items += fb
            warnings.append(
                f"{spec.id}: Openverse produced no image for {', '.join(missing)} -> "
                "used built-in procedurally drawn images instead (fallbacks, not real "
                "photographs)."
            )

    elif source == "freesound":
        try:
            items, counts = _freesound_items(spec, classes, per, queries, FreesoundClient(),
                                             cache, allow_download, force, warnings)
        except NoTokenError as exc:
            items, counts = [], {c: 0 for c in classes}
            warnings.append(f"{spec.id}: Freesound {exc} -> used synthesized audio "
                            "instead (generator names listed after a search query are "
                            "used directly).")
        missing = [c for c in classes if counts.get(c, 0) == 0 and queries.get(c)]
        if missing:
            items += _synth_fallback(spec, missing, per, queries, warnings,
                                     reason="Freesound unavailable")

    elif source == "mixed":
        want_image = kind in IMAGE_KINDS or fallback == "builtin"
        for label in classes:
            items += _remote_then_synth(label, queries.get(label) or [], spec, per,
                                        want_image=want_image,
                                        allow_download=allow_download,
                                        client_ov=client_ov, cache=cache,
                                        warnings=warnings)

    else:
        # Unknown source: degrade rather than crash, and say exactly what happened.
        warnings.append(
            f"{spec.id}: unknown stimuli.source={source!r} (expected "
            "none|internal|textbank|synth|builtin|openverse|freesound|mixed) -> no "
            "stimuli prepared for this test."
        )
        notes.append(f"{spec.id}: unknown source {source!r}; the runner will present "
                     "instructions only.")

    # ---------------------------------------------------------- optional extras
    if stimuli_cfg.get("word_queries"):
        items = attach_words(spec, items, stimuli_cfg, warnings)
    if stimuli_cfg.get("secondary_source"):
        items = attach_secondary(spec, items, stimuli_cfg, allow_download, force, warnings)

    result = StimulusSet(spec_id=spec.id, items=items, warnings=warnings,
                         notes=notes, sources=_count_sources(items))
    for lab in classes:
        if not any(s.label == lab for s in result.items) and queries.get(lab):
            result.notes.append(
                f"class {lab!r} resolved to zero stimuli (its queries were empty or it "
                "cannot be satisfied offline); the runner will skip it."
            )
    if verbose:
        per_label = {lab: sum(1 for s in result.items if s.label == lab) for lab in classes}
        say(f"source={source} kind={kind} -> {len(result.items)} items {per_label} "
            f"sources={result.sources}")
        for w in warnings:
            say(f"warning: {w}")
    return result


# ---------------------------------------------------------------- optional streams
def attach_words(spec: TestSpec, items: list[Stimulus], cfg: dict[str, Any],
                 warnings: list[str]) -> list[Stimulus]:
    """Attach a textbank word to every image item (T405 picture-word matching).

    ``word_queries`` is not part of the frozen contract; it exists because the
    N400 picture-word design needs *both* a picture and a word while the runner
    presents one stimulus kind.  The picture stays the item (``kind == 'image'``,
    ``path`` set) and the word travels in ``meta['word']`` so the runner can echo
    it and the analysis can use it.  The word is drawn deterministically from the
    named category by class and index, so a rerun presents the same pairs.
    """
    spec_words = cfg.get("word_queries") or {}
    by_label: dict[str, int] = {}
    out: list[Stimulus] = []
    for stim in items:
        label = stim.label
        if isinstance(spec_words, dict):
            words = _as_list(spec_words.get(label))
        else:
            words = _as_list(spec_words)
        if not words:
            words = ["words_high_freq"]
        cat = words[by_label.get(label, 0) % len(words)]
        try:
            pool = textbank.items(cat)
        except KeyError as exc:
            raise KeyError(f"{spec.id}: stimuli.word_queries[{label!r}] -> {exc}") from exc
        if not pool:
            raise KeyError(f"{spec.id}: stimuli.word_queries[{label!r}] -> textbank "
                           f"category {cat!r} has no items")
        idx = by_label.get(label, 0)
        chosen = pool[idx % len(pool)]
        by_label[label] = idx + 1
        stim.meta["word"] = chosen.text
        stim.meta["word_category"] = cat
        stim.meta["word_index"] = idx
        out.append(stim)
    warnings.append(
        f"{spec.id}: stimuli.word_queries present -> every image carries "
        "meta['word'] from the named textbank categories (the runner must echo the "
        "word while presenting the picture)."
    )
    return out


def attach_secondary(spec: TestSpec, items: list[Stimulus], cfg: dict[str, Any],
                     allow_download: bool, force: bool, warnings: list[str]
                     ) -> list[Stimulus]:
    """Resolve ``secondary_source`` and attach it per item (T207 dual task).

    Implemented rather than merely warned about: the dual-task design needs the
    verbal stream *paired* with the audio stream, so each primary item gets
    ``meta['secondary'] = {'kind', 'label', 'text'|'path', 'source'}``.  If the
    secondary stream cannot be resolved, the primary items are returned unchanged
    and a warning states that the dual-task variant needs manual attention.
    """
    secondary_source = str(cfg.get("secondary_source", "") or "").strip().lower()
    secondary = dict(cfg)
    secondary["source"] = secondary_source
    secondary["queries"] = cfg.get("secondary_queries") or {}
    for key in ("secondary_source", "secondary_queries", "word_queries", "generator"):
        secondary.pop(key, None)
    probe = TestSpec(
        id=f"{spec.id}:secondary", slug=f"{spec.slug}-secondary", tier=spec.tier,
        title=f"{spec.title} (secondary stream)", modality=list(spec.modality),
        feasibility=spec.feasibility, goal=spec.goal, classes=list(spec.classes),
        stimuli=secondary, design=dict(spec.design), markers=dict(spec.markers),
        processing=dict(spec.processing), features=list(spec.features),
        decoder=dict(spec.decoder), metrics=list(spec.metrics),
        acceptance=dict(spec.acceptance), risks=list(spec.risks), agent_use=spec.agent_use,
    )
    try:
        sub = build(probe, allow_download=allow_download, force=force, verbose=False,
                    limit=None)
    except Exception as exc:                        # never fail the primary stream
        warnings.append(
            f"{spec.id}: secondary_source={secondary_source!r} could not be resolved "
            f"({type(exc).__name__}: {exc}) -> the dual-task variant was NOT assembled "
            "and needs manual attention."
        )
        return items
    pool: dict[str, list[Stimulus]] = {}
    for s in sub.items:
        pool.setdefault(s.label, []).append(s)
    used: dict[str, int] = {}
    attached = 0
    for stim in items:
        group = pool.get(stim.label) or []
        if not group:
            continue
        i = used.get(stim.label, 0)
        sec = group[i % len(group)]
        used[stim.label] = i + 1
        stim.meta["secondary"] = {
            "kind": sec.kind,
            "label": sec.label,
            "text": sec.text or "",
            "path": str(sec.path) if sec.path else "",
            "source": sec.source,
        }
        attached += 1
    if attached:
        warnings.append(f"{spec.id}: secondary stream ({secondary_source}) resolved and "
                        f"attached to {attached} items as meta['secondary'] "
                        "(the runner must present or cue it).")
    else:
        warnings.append(
            f"{spec.id}: secondary_source={secondary_source!r} produced no items -> the "
            "dual-task variant was NOT assembled and needs manual attention."
        )
    return items


# ---------------------------------------------------------------- CLI
def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m neurobridge.stimuli.builder T601 --limit 2``.

    Prints exactly what the builder would resolve for a test id -- the fastest
    way to debug "why is this test falling back to the synthesizer?" without
    starting a session or opening a display.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m neurobridge.stimuli.builder",
        description="Resolve a test spec's stimuli block and report what would be presented.",
    )
    parser.add_argument("test_id", help="test id (T601), slug, or id prefix")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items per class (smoke tests)")
    parser.add_argument("--force", action="store_true", help="bypass the cache")
    parser.add_argument("--no-download", action="store_true",
                        help="never touch the network (cache and local generators only)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        spec = load_one(args.test_id)
    except KeyError as exc:
        print(f"error: {exc}")
        return 2

    stimset = build(spec, allow_download=not args.no_download, force=args.force,
                    verbose=not args.quiet, limit=args.limit)
    payload = {
        "spec_id": spec.id,
        "title": spec.title,
        "source": spec.stimuli.get("source"),
        "kind": spec.stimuli.get("kind"),
        "limit": args.limit,
        "n_items": len(stimset.items),
        "per_label": {lab: sum(1 for s in stimset.items if s.label == lab)
                      for lab in spec.classes},
        "sources": stimset.sources,
        "warnings": stimset.warnings,
        "notes": stimset.notes,
        "problems": stimset.check(),
        "items": [
            {"label": s.label, "kind": s.kind,
             "path": str(s.path) if s.path else None,
             "text": s.text[:80] if s.text else "",
             "source": s.source,
             "meta": {k: v for k, v in s.meta.items()
                      if k in ("generator", "category", "word", "license", "url",
                               "secondary", "fallback_reason")}}
            for s in stimset.items
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    print(f"\n{spec.id}  {spec.title}")
    print(f"  source={payload['source']} kind={payload['kind']} limit={payload['limit']}")
    print(f"  items={payload['n_items']} per_label={payload['per_label']}")
    print(f"  sources={payload['sources']}")
    for w in payload["warnings"]:
        print(f"  warning: {w}")
    for nt in payload["notes"]:
        print(f"  note: {nt}")
    if payload["problems"]:
        print(f"  problems: {payload['problems']}")
    print("  resolved items:")
    for it in payload["items"][:24]:
        extra = it["meta"].get("generator") or it["meta"].get("word") or ""
        where = it["path"] or (it["text"][:40] + "..." if it["text"] else "-")
        print(f"    {it['label']:<14} {it['kind']:<6} {it['source']:<9} "
              f"{str(where):<52} {extra}")
    if len(payload["items"]) > 24:
        print(f"    ... {len(payload['items']) - 24} more")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_main())
