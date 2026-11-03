from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from agent.fallback_policy import fallback_allowed, activate_for_error
# Load runtime bootstrap during collection, before the real-home I/O guard.
import agent.conversation_loop
from tui_gateway import server as tui_server
import cli

STRICT = {"fallback": {"quota_only": True}}


@pytest.mark.parametrize("error", ["usage_limit_reached", "insufficient_quota", "Quota exhausted", "subscription usage allowance exhausted", "Insufficient balance", "You have hit your usage limit", {"error": {"code": "insufficient_quota"}}])
def test_quota_allowed(error):
    assert fallback_allowed(error, STRICT)


@pytest.mark.parametrize("error", [None, "", "connection failure", "timeout", "429 Too many requests", "401 Unauthorized", "500 Internal Server Error", "rate_limit", "billing", "quota not exhausted"])
def test_other_errors_denied(error):
    assert not fallback_allowed(error, STRICT)


@pytest.mark.parametrize("config", [{}, {"fallback": {"quota_only": False}}])
def test_default_unchanged(config):
    assert fallback_allowed(None, config)


def test_evidence_not_stale(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    agent = SimpleNamespace(_try_activate_fallback=Mock(return_value=True))
    assert activate_for_error(agent, "usage_limit_reached")
    assert agent._fallback_quota_evidence is None
    assert not activate_for_error(agent, "timeout")
    assert agent._try_activate_fallback.call_count == 1


def test_main_gate_has_no_side_effects(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    from agent.chat_completion_helpers import try_activate_fallback
    agent = SimpleNamespace(_fallback_index=0, _fallback_chain=[{}])
    assert not try_activate_fallback(agent)
    assert agent._fallback_index == 0


@pytest.mark.parametrize("message,allowed", [("usage_limit_reached", True), ("401 Unauthorized", False), ("429 Too many requests", False), ("Connection failure", False)])
def test_pre_agent_resolution(monkeypatch, message, allowed):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError
    primary = AuthError(message)
    resolver = Mock(side_effect=[primary, {"provider": "deepseek"}])
    monkeypatch.setattr(rp, "resolve_runtime_provider", resolver)
    config = {**STRICT, "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}]}
    if allowed:
        runtime, entry = rp.resolve_runtime_with_fallback(config)
        assert runtime["provider"] == "deepseek"
        assert entry["model"] == "deepseek-chat"
        assert resolver.call_count == 2
    else:
        with pytest.raises(AuthError):
            rp.resolve_runtime_with_fallback(config)
        assert resolver.call_count == 1


@pytest.mark.parametrize("code,message,expected", [
    ("usage_limit_reached", "You've hit your usage limit.", "break"),
    ("rate_limit_exceeded", "Too many requests", "continue"),
    ("server_error", "Internal Server Error", "continue"),
])
def test_codex_http200_quota_forwarding(monkeypatch, code, message, expected):
    from tests.agent.test_codex_soft_failure_pool_rotation import _Agent, _run, _soft_failure, _entry
    from agent.credential_pool import CredentialPool
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart", lambda *args: None)
    agent = _Agent(CredentialPool("openai-codex", [_entry(0)]))
    agent._recover_with_credential_pool = Mock(return_value=(False, False))
    agent._fallback_chain = [{"provider": "deepseek", "model": "deepseek-flash"}]
    agent._fallback_quota_evidence = None
    agent._try_activate_fallback = Mock(return_value=True)
    verdict = _run(agent, _soft_failure(code, message))
    assert verdict.action == expected
    assert agent._try_activate_fallback.call_count == (1 if expected == "break" else 0)
    assert agent._fallback_quota_evidence is None


def test_evidence_restored_on_activation_error(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    agent = SimpleNamespace(_fallback_quota_evidence="previous", _try_activate_fallback=Mock(side_effect=ValueError("failed")))
    with pytest.raises(ValueError, match="failed"):
        activate_for_error(agent, "insufficient_quota")
    assert agent._fallback_quota_evidence == "previous"


def test_unreadable_policy_is_fail_closed(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", Mock(side_effect=OSError("unreadable")))
    assert not fallback_allowed("insufficient_quota")


def test_exception_body_is_current_quota_evidence():
    error = RuntimeError("HTTP 429")
    error.body = {"error": {"code": "insufficient_quota"}}
    assert fallback_allowed(error, STRICT)


@pytest.mark.parametrize("message,allowed", [("usage_limit_reached", True), ("401 Unauthorized", False), ("429 Too many requests", False)])
def test_tui_relocated_runtime_gate(monkeypatch, message, allowed):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    chain = Mock(return_value=[{"provider": "deepseek", "model": "deepseek-chat"}])
    monkeypatch.setattr(tui_server, "_load_fallback_model", chain)
    resolver = Mock(side_effect=[AuthError(message), {"provider": "deepseek"}])
    monkeypatch.setattr(rp, "resolve_runtime_provider", resolver)
    if allowed:
        resolution = tui_server._resolve_runtime_with_fallback()
        assert resolution.runtime["provider"] == "deepseek"
        assert resolver.call_count == 2
    else:
        with pytest.raises(AuthError):
            tui_server._resolve_runtime_with_fallback()
        assert resolver.call_count == 1
        chain.assert_not_called()


@pytest.mark.parametrize("message,allowed", [("Quota exhausted", True), ("401 Unauthorized", False), ("429 Too many requests", False)])
def test_cron_resolution_gate(monkeypatch, message, allowed):
    from cron import scheduler
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError
    chain = Mock(return_value=[{"provider": "deepseek", "model": "deepseek-chat"}])
    monkeypatch.setattr(scheduler, "_job_fallback_chain", chain)
    resolver = Mock(side_effect=[AuthError(message), {"provider": "deepseek"}])
    monkeypatch.setattr(rp, "resolve_runtime_provider", resolver)
    jc = SimpleNamespace(model="primary", cron_default_provider=None, cfg=STRICT, model_cfg={})
    if allowed:
        runtime, model = scheduler._resolve_job_runtime({}, "test-job", jc)
        assert (runtime["provider"], model) == ("deepseek", "deepseek-chat")
        assert resolver.call_count == 2
    else:
        with pytest.raises(RuntimeError):
            scheduler._resolve_job_runtime({}, "test-job", jc)
        assert resolver.call_count == 1
        chain.assert_not_called()


@pytest.mark.parametrize("message,allowed", [("Codex provider quota exhausted (429); retry after 120s.", True), ("429 Too many requests", False), ("timeout", False)])
def test_codex_app_server_gate(monkeypatch, message, allowed):
    from agent.turn_recovery import activate_codex_app_server_fallback
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    agent = SimpleNamespace(provider="openai-codex", model="gpt-test", _has_pending_fallback=Mock(return_value=True),
                            _try_activate_fallback=Mock(return_value=True), _buffer_diagnostic_status=Mock())
    assert activate_codex_app_server_fallback(agent, {"error": message}) is allowed
    assert agent._try_activate_fallback.call_count == int(allowed)
    if not allowed:
        agent._buffer_diagnostic_status.assert_not_called()


def test_aux_unavailable_and_quarantine_cannot_discover_other_providers(monkeypatch):
    from agent import auxiliary_client as ac
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    configured = Mock()
    monkeypatch.setattr(ac, "_try_configured_fallback_chain", configured)
    assert ac._try_configured_fallback_for_unavailable_client("compression", "openai") == (None, None, "")
    assert ac._next_fallback_after_quarantine("compression", "auto", True, None, None, None) == (None, None, "")
    configured.assert_not_called()


def test_aux_auto_still_uses_primary_but_never_discovers_on_unavailability(monkeypatch):
    from agent import auxiliary_client as ac
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    monkeypatch.setattr(ac, "_normalize_main_runtime", lambda _r: {"provider": "openai"})
    monkeypatch.setattr(ac, "_main_route_target", lambda *_a: ("openai", "test", None, None, None))
    primary = Mock(return_value=("primary-client", "test", "openai"))
    monkeypatch.setattr(ac, "_try_main_provider_route", primary)
    discovery = Mock()
    monkeypatch.setattr(ac, "_try_configured_fallback_chain", discovery)
    assert ac._resolve_auto_route(task="compression") == ("primary-client", "test", "openai")
    primary.return_value = None
    assert ac._resolve_auto_route(task="compression") == (None, None, "")
    discovery.assert_not_called()


@pytest.mark.parametrize("message", ["429 Too many requests", "connection failure", "401 Unauthorized"])
def test_aux_ladder_denies_non_quota_before_route_side_effects(monkeypatch, message):
    from agent import auxiliary_client as ac
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    # A missing route is deliberate: policy must return before inspecting/mutating it.
    assert list(ac._ladder_provider_fallback(RuntimeError(message), None)) == []


def test_cli_resolution_guard_precedes_fallback_access(monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    assert CLIAgentSetupMixin._resolve_fallback_runtime(SimpleNamespace(), AuthError("401 Unauthorized")) is None


def test_aux_vision_unavailable_does_not_try_auto(monkeypatch):
    from agent import auxiliary_client as ac
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    resolve = Mock(return_value=("openai", None, "vision-model"))
    monkeypatch.setattr(ac, "resolve_vision_provider_client", resolve)
    with pytest.raises(RuntimeError):
        ac._resolve_call_client(
            "vision", provider="openai", model="vision-model", base_url=None, api_key=None,
            resolved_provider="openai", resolved_model="vision-model", resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False)
    assert resolve.call_count == 1


def test_aux_quota_still_allows_first_fallback(monkeypatch):
    from agent import auxiliary_client as ac
    from tests.agent.test_auxiliary_fallback_multihop import _client, _quota_429, _ok_response, _walk_patches
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: STRICT)
    primary = _client("http://127.0.0.1:1/v1", Mock(side_effect=_quota_429("A")))
    fallback = _client("http://127.0.0.1:2/v1", Mock(return_value=_ok_response("OK")))
    patches = _walk_patches(primary, [(fallback, "modelB", "custom")])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        response = ac.call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])
    assert response.choices[0].message.content == "OK"
    assert fallback.chat.completions.create.call_count == 1
