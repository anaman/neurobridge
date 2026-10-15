"""Content acquisition: cache, fetchers, synthesizer, text bank, image fallbacks.

This package owns everything that turns a spec's ``stimuli`` block into local
files or text.  The public entry point is :func:`prepare` (re-exported from
``base``), and the design rule for the whole package is that **the network,
credentials and displays are optional**: every acquisition path degrades to a
local generator and reports what happened in ``StimulusSet.warnings``.

Importing this package never touches the network, a token or a device: the
process must start cleanly on an offline laptop with no headband attached, which
is also the condition every smoke test runs under.

``build`` and the two optional-stream helpers are re-exported lazily (PEP 562)
rather than imported here, so that ``python -m neurobridge.stimuli.builder T601``
does not import the ``builder`` module twice.  ``runpy`` pre-registers the target
module in ``sys.modules`` before executing it, so an eager import here makes
CPython print "found in sys.modules after import of package ... but prior to
execution" over the debugging output the operator is trying to read.
"""
from __future__ import annotations

from typing import Any

from . import builtin, synth, textbank          # noqa: F401  (re-exported submodules)
from .base import Stimulus, StimulusSet, kind_dir, prepare
from .cache import CachedFile, StimulusCache, cache_key
from .freesound import (
    AuthError,
    FreesoundClient,
    NoTokenError,
    ffmpeg_available,
    looks_like_oauth2,
    resolve_auth_style,
)
from .openverse import NetworkError, NoResultsError, OpenverseClient

__all__ = [
    # frozen contract
    "prepare", "Stimulus", "StimulusSet", "StimulusCache", "CachedFile", "cache_key",
    "OpenverseClient", "FreesoundClient", "NetworkError", "NoResultsError",
    "NoTokenError", "AuthError",
    "synth", "textbank", "builtin",
    # convenience
    "build", "kind_dir", "ffmpeg_available", "looks_like_oauth2", "resolve_auth_style",
    "attach_words", "attach_secondary",
]

#: Names resolved on first access from :mod:`neurobridge.stimuli.builder`.
_LAZY = {
    "build": "build",
    "attach_words": "attach_words",
    "attach_secondary": "attach_secondary",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy re-export of the builder helpers."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import builder

    return getattr(builder, target)


def __dir__() -> list[str]:
    return sorted(set(__all__))
