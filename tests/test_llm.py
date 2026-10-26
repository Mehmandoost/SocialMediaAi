import json
from unittest.mock import MagicMock, patch

import pytest

from socialmediaai import Config, Llm
from socialmediaai.errors import MissingCredentialError


@pytest.fixture
def llm(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAI_HOME", str(tmp_path))
    return Llm(Config())


def test_save_api_key_maps_provider(llm: Llm):
    env = llm.save_api_key("deepseek", "sk-deep")
    assert env == "DEEPSEEK_API_KEY"
    assert llm.config.get_secret("DEEPSEEK_API_KEY") == "sk-deep"


def test_set_model_custom_stores_env_reference(llm: Llm):
    llm.set_model(
        "custom",
        "my-model",
        base_url="https://api.example.com/v1",
        api_key="secret-key",
    )
    raw = llm.config.config_path.read_text()
    assert "${SMAI_CUSTOM_" in raw
    assert llm.config.get_secret("SMAI_CUSTOM_API_EXAMPLE_COM_API_KEY") == "secret-key"


def test_complete_missing_key(llm: Llm, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    llm.set_model("deepseek", "deepseek-chat")
    with pytest.raises(MissingCredentialError) as exc:
        llm.complete("hi")
    assert exc.value.env_var == "DEEPSEEK_API_KEY"


@patch("socialmediaai.transports.chat_completions", return_value="ok-chat")
def test_complete_chat_completions(mock_chat, llm: Llm):
    llm.save_api_key("deepseek", "sk-x")
    llm.set_model("deepseek", "deepseek-chat")
    assert llm.complete("hello") == "ok-chat"
    mock_chat.assert_called_once()


@patch("socialmediaai.transports.anthropic_messages", return_value="ok-anthropic")
def test_complete_anthropic(mock_anthropic, llm: Llm):
    llm.save_api_key("anthropic", "sk-ant")
    llm.set_model("anthropic", "claude-sonnet-4-6")
    assert llm.complete("hello") == "ok-anthropic"
    mock_anthropic.assert_called_once()


@patch("socialmediaai.transports.codex_responses", return_value="ok-responses")
def test_complete_codex_responses(mock_responses, llm: Llm):
    llm.save_api_key("xai", "xai-key")
    llm.set_model("xai", "grok-4")
    assert llm.complete("hello") == "ok-responses"
    mock_responses.assert_called_once()


@patch("socialmediaai.transports.bedrock_converse", return_value="ok-bedrock")
def test_complete_bedrock(mock_bedrock, llm: Llm):
    llm.set_model("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert llm.complete("hello") == "ok-bedrock"
    mock_bedrock.assert_called_once()


@patch("socialmediaai.transports.copilot_http", return_value="ok-copilot")
def test_complete_copilot(mock_copilot, llm: Llm):
    llm.config.save_secret("COPILOT_GITHUB_TOKEN", "gho_test")
    llm.set_model("copilot", "gpt-4o")
    assert llm.complete("hello") == "ok-copilot"
    mock_copilot.assert_called_once()


@patch("socialmediaai.transports.copilot_acp", return_value="ok-acp")
def test_complete_copilot_acp(mock_acp, llm: Llm):
    llm.set_model("copilot-acp", "default")
    assert llm.complete("hello") == "ok-acp"
    mock_acp.assert_called_once()


@patch("socialmediaai.transports.vertex_chat", return_value="ok-vertex")
def test_complete_vertex(mock_vertex, llm: Llm):
    llm.set_model("vertex", "gemini-2.0-flash")
    assert llm.complete("hello") == "ok-vertex"
    mock_vertex.assert_called_once()


def test_oauth_bearer_from_auth_json(llm: Llm, tmp_path):
    auth = {"openai-codex": {"access_token": "oauth-token"}}
    (llm.config.home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    llm.set_model("openai-codex", "gpt-5.4")
    with patch("socialmediaai.transports.codex_responses", return_value="oauth-ok") as mock_resp:
        assert llm.complete("hi") == "oauth-ok"
        assert mock_resp.call_args.kwargs["api_key"] == "oauth-token"
