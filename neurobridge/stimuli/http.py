"""Shared HTTP retry/error-formatting helpers for the stimulus fetchers.

Kept in one place so Openverse and Freesound apply the *same* rate-limit
politeness rules: a 429 is always honoured via ``Retry-After`` (bounded), a
single 5xx is retried once, and auth failures get an actionable message that
names the environment variable the operator must set.  Free APIs that are
abused by research scripts end up blocking whole institutions, so this is as
much an ethical decision as a technical one.
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any

#: Never sleep longer than this, even if a server asks: a curriculum run must
#: not appear hung.  Any longer wait means the caller should fall back.
MAX_RETRY_SLEEP_S = 30.0


def retry_after_seconds(headers: Any, default: float = 2.0) -> float:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date), clamped to [0, 30]."""
    raw = ""
    try:
        raw = str(headers.get("Retry-After", "") or "").strip()
    except Exception:
        raw = ""
    if raw:
        try:
            return max(0.0, min(float(raw), MAX_RETRY_SLEEP_S))
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(raw)
            now = _dt.datetime.now(when.tzinfo) if when.tzinfo else _dt.datetime.now()
            return max(0.0, min((when - now).total_seconds(), MAX_RETRY_SLEEP_S))
        except Exception:
            pass
    return max(0.0, min(float(default), MAX_RETRY_SLEEP_S))


def short_body(resp: Any, limit: int = 200) -> str:
    """Whitespace-collapsed first ``limit`` characters of a response body."""
    try:
        text = resp.text or ""
    except Exception:
        return ""
    return " ".join(text.split())[:limit]


def sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch without touching ``time.sleep``."""
    if seconds > 0:
        time.sleep(seconds)
