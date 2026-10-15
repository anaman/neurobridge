"""Presentation and acquisition runner.

``run_test`` runs one catalog spec end to end (stimuli, markers, EEG, quality,
timing report) and returns the session directory.  ``run_sequence`` runs several
specs in one session over one BLE connection.

The module is import-safe on a machine with no headband, no display and no
sound card: hardware and pygame are only touched inside ``Runner.setup`` and the
presenters' ``__init__``.
"""
from __future__ import annotations

from .present_audio import AudioPresenter
from .present_visual import VisualPresenter
from .session import RunConfig, Runner, run_sequence, run_test
from .timeline import (Phase, Trial, build_trials, iti_bounds, jitter,
                       make_phases, plan_summary, resolve_design, summarise)

__all__ = [
    "RunConfig", "run_test", "run_sequence", "Runner",
    "AudioPresenter", "VisualPresenter",
    "build_trials", "make_phases", "jitter",
    "Trial", "Phase", "iti_bounds", "plan_summary", "resolve_design", "summarise",
]
