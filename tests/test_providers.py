from socialmediaai.providers.registry import get_provider, list_providers, primary_env_var, resolve_base_url
from socialmediaai import Config


def test_catalog_lists_hermes_providers():
    names = {p.name for p in list_providers()}
    expected = {
        "openrouter",
        "deepseek",
        "anthropic",
        "gemini",
        "xai",
        "bedrock",
        "copilot",
        "copilot-acp",
        "kimi-coding",
        "lmstudio",
        "ollama",
        "custom",
    }
    assert expected.issubset(names)
    assert len(names) >= 45


def test_alias_resolution():
    assert get_provider("or").name == "openrouter"
    assert get_provider("deep-seek").name == "deepseek"


def test_every_provider_has_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAI_HOME", str(tmp_path))
    config = Config()
    for profile in list_providers():
        assert profile.name
        assert profile.api_mode
        assert profile.auth_type
        if profile.auth_type == "api_key" and profile.env_vars:
            assert primary_env_var(profile)
        if profile.base_url or profile.name in ("custom", "azure-foundry", "solstice"):
            continue
        if profile.auth_type in ("oauth_external", "oauth_device_code", "aws_sdk", "external_process", "copilot", "vertex"):
            continue
        # Most API-key cloud providers ship a default base URL in the Hermes catalog.
        assert profile.base_url or profile.name in {
            "router",
            "commandcode",
            "meta-ai",
        }


def test_resolve_base_url_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAI_HOME", str(tmp_path))
    config = Config()
    profile = get_provider("lmstudio")
    config.save_secret("LM_BASE_URL", "http://127.0.0.1:9999/v1")
    assert resolve_base_url(profile, config) == "http://127.0.0.1:9999/v1"
