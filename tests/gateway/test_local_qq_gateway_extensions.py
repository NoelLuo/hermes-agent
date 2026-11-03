"""Local QQ extensions survive the plugin migration without changing other lanes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import gateway.run as gateway_run
from gateway.config import Platform, GatewayConfig
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from gateway.session import SessionStore


@pytest.mark.parametrize('effort,expected', [('low', {'enabled': True, 'effort': 'low'}),
                                            (False, {'enabled': False})])
def test_platform_reasoning(monkeypatch, effort, expected):
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {
        'agent': {'platforms': {'qqbot': {'reasoning_effort': effort}}}})
    runner = GatewayConfigLoadersMixin()
    runner._resolve_session_key_or_none = lambda source, key: None
    runner._load_reasoning_config = lambda model: {'enabled': True, 'effort': 'high'}
    source = SimpleNamespace(platform=Platform('qqbot'), chat_type='group')
    assert runner._resolve_session_reasoning_config(source=source) == expected


def test_session_reasoning_wins(monkeypatch):
    runner = GatewayConfigLoadersMixin()
    runner._resolve_session_key_or_none = lambda source, key: 'session'
    runner._peek_session_state = lambda key: SimpleNamespace(conversation=SimpleNamespace(
        reasoning_override={'enabled': True, 'effort': 'xhigh'}))
    assert runner._resolve_session_reasoning_config() == {'enabled': True, 'effort': 'xhigh'}


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,chat_type,use_hook', [
    ('qqbot', 'group', True), ('qqbot', 'dm', False), ('telegram', 'group', False)])
async def test_history_hook_only_for_qq_group(platform, chat_type, use_hook):
    runner = GatewayTurnPrepareMixin()
    runner.async_session_store = SimpleNamespace(load_transcript=AsyncMock(return_value=['normal']))
    adapter = SimpleNamespace(prepare_group_turn_history=AsyncMock(return_value=['prepared']))
    runner._delivery_adapter_for = Mock(return_value=adapter)
    source = SimpleNamespace(platform=Platform(platform), chat_type=chat_type)
    event = SimpleNamespace(source=source)
    result = await runner._hmwa_load_turn_history(event, source, 'session')
    assert result == (['prepared'] if use_hook else ['normal'])
    if use_hook:
        adapter.prepare_group_turn_history.assert_awaited_once_with(runner.async_session_store, 'session', event)
        runner.async_session_store.load_transcript.assert_not_awaited()
    else:
        adapter.prepare_group_turn_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_qq_adapter_uses_normal_history():
    runner = GatewayTurnPrepareMixin()
    runner.async_session_store = SimpleNamespace(load_transcript=AsyncMock(return_value=['normal']))
    runner._delivery_adapter_for = lambda source: None
    source = SimpleNamespace(platform=Platform('qqbot'), chat_type='group')
    assert await runner._hmwa_load_turn_history(SimpleNamespace(), source, 'session') == ['normal']


def test_raw_transcript_option_reaches_db(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', tmp_path / 'state.db')
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    db = store._db
    db.create_session(session_id='local-qq-test', source='test')
    db.append_message('local-qq-test', 'user', 'hello')
    db.append_message('local-qq-test', 'assistant', 'world')
    calls = []
    original = db.get_messages_as_conversation

    def tracked(session_id, **kwargs):
        calls.append(kwargs['repair_alternation'])
        return original(session_id, **kwargs)

    monkeypatch.setattr(db, 'get_messages_as_conversation', tracked)
    assert len(store.load_transcript('local-qq-test', repair_alternation=False)) == 2
    assert len(store.load_transcript('local-qq-test')) == 2
    assert calls == [False, True]
