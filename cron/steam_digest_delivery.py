"""Local, admin-approved delivery contract for the existing Steam digest job.

This is deliberately NOT a general cron privilege or stdout artifact grant. A
new job, destination, profile or output path requires a separate admin review.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.request_security import RequestSecurityPolicy, current_policy, policy_scope
from hermes_constants import get_hermes_home, profile_name_for_home


def _approved_destination(job, chat_id):
    """Read the admin-owned allowlist, never job/stdout-supplied authority."""
    from hermes_cli.config_effective import load_user_config_effective

    cron = load_user_config_effective().get("cron", {})
    approved = cron.get("steam_digest_delivery", {}) if isinstance(cron, dict) else {}
    if not isinstance(approved, dict):
        return False
    job_id, targets = approved.get("job_id"), approved.get("targets")
    return (isinstance(job_id, str) and bool(job_id) and job.get("id") == job_id
            and isinstance(targets, list)
            and all(isinstance(target, str) and bool(target) for target in targets)
            and chat_id in targets)


def poster_delivery_policy(job, platform, chat_id, media_path):
    """Grant only today's independently derived poster, never a stdout-selected file."""
    if (current_policy() is not None or "request_policy" in job
            or job.get("public_context_only") is True
            or job.get("script") != "steam_deals_digest.py"
            or job.get("no_agent") is not True
            or getattr(platform, "value", platform) != "qqbot"):
        return None
    home = get_hermes_home()
    if profile_name_for_home(home) != "default" or not _approved_destination(job, chat_id):
        return None
    day = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    expected = home / "cache" / "images" / f"steam-cn-deals-{day}.png"
    try:
        # Reject links/junctions escaping the precise cache path, even when stdout
        # names the expected file. Path policy still runs in the caller first.
        canonical = expected.resolve(strict=True)
        if canonical != expected.absolute() or Path(media_path).resolve(strict=True) != canonical:
            return None
        with canonical.open("rb") as image:
            if image.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
        policy = RequestSecurityPolicy(
            subject_id=f"cron:{job['id']}", platform="qqbot", chat_id=chat_id,
            allowed_tools=frozenset(), allowed_commands=frozenset(), scope_id="default")
        return policy if policy.grant_artifact(str(canonical)) else None
    except (OSError, RuntimeError, ValueError):
        return None


async def send_poster_with_policy(policy, adapter, chat_id, media_path, metadata):
    """Bind in the gateway task, not merely while creating its coroutine."""
    with policy_scope(policy):
        return await adapter.send_image_file(
            chat_id=chat_id, image_path=media_path, metadata=metadata)
