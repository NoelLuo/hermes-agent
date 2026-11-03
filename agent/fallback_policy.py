"""Opt-in cross-provider fallback guard; never infer quota from HTTP status alone."""
from __future__ import annotations

import re
from collections.abc import Mapping

_CODES = re.compile(r"\b(?:usage_limit_reached|insufficient_quota|quota_exceeded|credits_exhausted|insufficient_balance)\b", re.I)
_EXHAUSTED = re.compile(
    r"\b(?:quota|credits?|balance|subscription (?:usage|allowance)|usage (?:limit|allowance))\b"
    r"[^.\n]{0,65}\b(?:exhausted|depleted|exceeded|reached|insufficient|zero)\b"
    r"|\b(?:exhausted|depleted|insufficient)\b[^.\n]{0,40}\b(?:quota|credits?|balance)\b"
    r"|\byou(?:'ve| have) (?:hit|reached) your usage limit\b", re.I)


def explicit_quota_exhaustion(error) -> bool:
    """Inspect only this failure, not cached health, reset timestamps or old errors."""
    if error is None:
        return False
    if isinstance(error, Mapping):
        return any(explicit_quota_exhaustion(error.get(k)) for k in
                   ("error", "code", "type", "message", "error_description"))
    if isinstance(error, BaseException):
        return explicit_quota_exhaustion(getattr(error, "body", None)) or explicit_quota_exhaustion(str(error))
    if not isinstance(error, str):
        return False
    if re.search(r"\b(?:not|never)\s+(?:exhausted|exceeded|depleted|reached)\b", error, re.I):
        return False
    return bool(_CODES.search(error) or _EXHAUSTED.search(error))


def fallback_allowed(error=None, config=None) -> bool:
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            # Unknown policy must not authorize a potentially billable provider hop.
            return False
    section = config.get("fallback", {}) if isinstance(config, Mapping) else {}
    enabled = isinstance(section, Mapping) and section.get("quota_only") is True
    return not enabled or explicit_quota_exhaustion(error)


def activate_for_error(agent, error, **kwargs):
    """Pass evidence only for the duration of one activation (no stale error latch)."""
    if not fallback_allowed(error):
        return False
    previous = getattr(agent, "_fallback_quota_evidence", None)
    agent._fallback_quota_evidence = error
    try:
        return agent._try_activate_fallback(**kwargs)
    finally:
        agent._fallback_quota_evidence = previous
