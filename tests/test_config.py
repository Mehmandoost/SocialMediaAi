import os
from pathlib import Path

import pytest

from socialmediaai import Config
from socialmediaai.errors import ConfigError


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAI_HOME", str(tmp_path))
    return Config()


def test_save_and_get_secret(config_home: Config):
    config_home.save_secret("DEEPSEEK_API_KEY", "sk-test")
    assert config_home.get_secret("DEEPSEEK_API_KEY") == "sk-test"
    assert (config_home.env_path.stat().st_mode & 0o777) == 0o600


def test_no_duplicate_env_lines(config_home: Config):
    config_home.save_secret("OPENROUTER_API_KEY", "first")
    config_home.save_secret("OPENROUTER_API_KEY", "second")
    text = config_home.env_path.read_text()
    assert text.count("OPENROUTER_API_KEY=") == 1
    assert "second" in text


def test_process_env_overrides_dotenv(config_home: Config, monkeypatch):
    config_home.save_secret("OPENAI_API_KEY", "from-file")
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")
    assert config_home.get_secret("OPENAI_API_KEY") == "from-process"


def test_set_secret_key_routes_to_env(config_home: Config):
    config_home.set("OPENROUTER_API_KEY", "sk-or-test")
    assert config_home.get_secret("OPENROUTER_API_KEY") == "sk-or-test"
    assert not config_home.config_path.exists() or "sk-or" not in config_home.config_path.read_text()


def test_yaml_model_round_trip(config_home: Config):
    config_home.set("model.provider", "openrouter")
    config_home.set("model.default", "openai/gpt-4o-mini")
    loaded = config_home.load()
    assert loaded["model"]["provider"] == "openrouter"


def test_reject_raw_api_key_in_yaml(config_home: Config):
    with pytest.raises(ConfigError):
        config_home.save({"model": {"api_key": "sk-raw-secret"}})


def test_expand_env_ref(config_home: Config):
    config_home.save_secret("MY_KEY", "resolved-value")
    config_home.save({"model": {"api_key": "${MY_KEY}"}})
    loaded = config_home.load()
    assert loaded["model"]["api_key"] == "resolved-value"
