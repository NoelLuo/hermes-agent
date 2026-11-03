"""Only the admin-approved Steam poster may cross the cron/live QQ boundary."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from threading import Thread
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from agent.request_security import RequestSecurityPolicy, current_policy, policy_scope
from cron.scheduler_delivery import _send_media_via_adapter
from gateway.config import Platform
from gateway.platforms.base import SendResult
from plugins.platforms.qqbot.adapter import QQAdapter

TARGETS = (
    'A' * 32,
    'B' * 32,
    'C' * 32,
)


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: tmp_path)
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'cron': {'steam_digest_delivery': {'job_id': 'f91fc03b9f08', 'targets': list(TARGETS)}}
    }), encoding='utf-8')
    day = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    poster = tmp_path / 'cache' / 'images' / f'steam-cn-deals-{day}.png'
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b'\x89PNG\r\n\x1a\nposter-test')
    adapter = QQAdapter.__new__(QQAdapter)
    adapter.config = NS(extra={'role_policy': {'enabled': True}})
    adapter.platform = Platform.QQBOT
    adapter._chat_type_map = dict.fromkeys(TARGETS, 'group')
    adapter._ensure_connected = AsyncMock(return_value=True)
    adapter._upload_local_file = AsyncMock(return_value={'file_info': 'test-upload'})
    adapter._post_message = AsyncMock(return_value=SendResult(success=True, message_id='test-id'))
    adapter._next_msg_seq = lambda chat: 1
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever)
    thread.start()
    job = {'id': 'f91fc03b9f08', 'script': 'steam_deals_digest.py', 'no_agent': True}
    def send(target=TARGETS[0], *, path=None, changes=None, platform=Platform.QQBOT):
        return _send_media_via_adapter(adapter, target, [(str(path or poster), False)], None,
                                       loop, {**job, **(changes or {})}, platform)
    yield NS(send=send, poster=poster, adapter=adapter, loop=loop, home=tmp_path)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.mark.parametrize('target', TARGETS)
def test_approved_poster_reaches_real_qq_policy_and_upload(delivery, target):
    assert delivery.send(target) == []
    delivery.adapter._upload_local_file.assert_awaited_once()
    delivery.adapter._post_message.assert_awaited_once()
    assert current_policy() is None
    assert asyncio.run_coroutine_threadsafe(_get_policy(), delivery.loop).result(5) is None


async def _get_policy():
    return current_policy()


@pytest.mark.parametrize('changes', [
    {'id': 'other-job'}, {'script': 'other.py'}, {'no_agent': False},
    {'request_policy': None}, {'public_context_only': True},
])
def test_other_jobs_and_public_jobs_stay_denied(delivery, changes):
    assert delivery.send(changes=changes)
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_other_destination_stays_denied(delivery):
    assert delivery.send('other-room')
    delivery.adapter._upload_local_file.assert_not_awaited()


@pytest.mark.parametrize('approved', [
    {}, None, [], {'job_id': '', 'targets': list(TARGETS)},
    {'job_id': 'f91fc03b9f08', 'targets': TARGETS[0]},
    {'job_id': 'f91fc03b9f08', 'targets': [None, TARGETS[0]]},
])
def test_missing_or_malformed_local_allowlist_denies_delivery(delivery, approved):
    (delivery.home / 'config.yaml').write_text(json.dumps({
        'cron': {'steam_digest_delivery': approved}
    }), encoding='utf-8')
    assert delivery.send()
    delivery.adapter._upload_local_file.assert_not_awaited()


@pytest.mark.parametrize('filename', ['private.png', 'steam-cn-deals-2000-01-01.png'])
def test_stdout_cannot_select_another_file(delivery, filename):
    path = delivery.poster.with_name(filename)
    path.write_bytes(delivery.poster.read_bytes())
    assert delivery.send(path=path)
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_other_profile_stays_denied(delivery, monkeypatch):
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: delivery.home / 'other-default')
    assert delivery.send()
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_guest_request_cannot_borrow_cron_grant(delivery):
    guest = RequestSecurityPolicy('guest', 'qqbot', TARGETS[0])
    with policy_scope(guest):
        assert delivery.send()
        assert current_policy() is guest
    assert not guest.allows_media(str(delivery.poster))
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_private_file_outside_cache_stays_denied(delivery):
    path = delivery.home / delivery.poster.name
    path.write_bytes(delivery.poster.read_bytes())
    assert delivery.send(path=path)
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_grant_is_non_admin_and_binds_file_identity_and_hash(delivery):
    from cron.steam_digest_delivery import poster_delivery_policy
    job = {'id': 'f91fc03b9f08', 'script': 'steam_deals_digest.py', 'no_agent': True}
    policy = poster_delivery_policy(job, Platform.QQBOT, TARGETS[0], str(delivery.poster))
    assert policy is not None and not policy.is_admin and policy.public_context
    assert not policy.allows_tool('terminal')
    assert not policy.allows_tool('execute_code')
    assert not policy.allows_command('restart')
    assert policy.allows_media(str(delivery.poster))
    delivery.poster.write_bytes(b'\x89PNG\r\n\x1a\nchanged-test')
    assert not policy.allows_media(str(delivery.poster))
    delivery.poster.unlink()
    delivery.poster.write_bytes(b'\x89PNG\r\n\x1a\nposter-test')
    assert not policy.allows_media(str(delivery.poster))


def test_invalid_png_stays_denied(delivery):
    delivery.poster.write_bytes(b'not-a-poster')
    assert delivery.send()
    delivery.adapter._upload_local_file.assert_not_awaited()


def test_upload_failure_does_not_leak_policy(delivery):
    delivery.adapter._upload_local_file.side_effect = RuntimeError('test transport failure')
    errors = delivery.send()
    assert len(errors) == 1 and 'test transport failure' in errors[0]
    assert current_policy() is None
    assert asyncio.run_coroutine_threadsafe(_get_policy(), delivery.loop).result(5) is None
    delivery.adapter._post_message.assert_not_awaited()
