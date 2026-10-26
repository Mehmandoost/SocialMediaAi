from __future__ import annotations

import json
import urllib.parse
from typing import Any

from socialmediaai.config import Config, custom_endpoint_key_env, is_secret_key
from socialmediaai.errors import MissingCredentialError
from socialmediaai.providers.registry import (
    ProviderProfile,
    get_provider,
    list_providers,
    primary_env_var,
    resolve_base_url,
)
from socialmediaai.transports import complete_with_profile


class Llm:
    """LLM access: provider catalog, credential storage, and completion calls."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def config(self) -> Config:
        return self._config

    def providers(self) -> list[ProviderProfile]:
        return list_providers()

    def save_api_key(self, provider_or_env: str, value: str) -> str:
        try:
            profile = get_provider(provider_or_env)
        except KeyError:
            key = provider_or_env.upper()
            if not is_secret_key(key):
                raise
            self._config.save_secret(key, value)
            return key
        env_name = primary_env_var(profile)
        if not env_name:
            raise ValueError(f"Provider {profile.name} has no API key environment variable.")
        self._config.save_secret(env_name, value)
        return env_name

    def set_model(
        self,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        profile = get_provider(provider)
        data = self._config.load()
        model_section: dict[str, Any] = dict(data.get("model") or {})
        model_section["provider"] = profile.name
        model_section["default"] = model
        if base_url:
            model_section["base_url"] = base_url.rstrip("/")
        elif profile.name == "custom" and not model_section.get("base_url"):
            raise ValueError("custom provider requires base_url")

        if api_key:
            effective_url = base_url or model_section.get("base_url") or profile.base_url
            parsed = urllib.parse.urlparse(effective_url)
            identity = parsed.hostname or "custom"
            if parsed.port:
                identity = f"{identity}_{parsed.port}"
            env_name = custom_endpoint_key_env(identity)
            self._config.save_secret(env_name, api_key)
            model_section["api_key"] = f"${{{env_name}}}"

        data["model"] = model_section
        self._config.save(data)

    def _active_model_config(self) -> dict[str, Any]:
        cfg = self._config.load().get("model") or {}
        return cfg if isinstance(cfg, dict) else {}

    def _resolve_api_key(self, profile: ProviderProfile, override: str | None = None) -> str | None:
        if override:
            return override
        model_cfg = self._active_model_config()
        api_key_field = model_cfg.get("api_key")
        if isinstance(api_key_field, str) and api_key_field.startswith("${") and api_key_field.endswith("}"):
            env_name = api_key_field[2:-1]
            val = self._config.get_secret(env_name)
            if val:
                return val

        if profile.auth_type == "copilot":
            for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
                val = self._config.get_secret(name)
                if val:
                    return val
            return None

        if profile.auth_type in ("oauth_external", "oauth_device_code"):
            token = self._oauth_bearer(profile)
            if token:
                return token

        for name in profile.env_vars:
            if name.upper().endswith("_BASE_URL") or name.upper().endswith("_URL"):
                continue
            val = self._config.get_secret(name)
            if val:
                return val
        return None

    def _oauth_bearer(self, profile: ProviderProfile) -> str | None:
        auth_path = self._config.home / "auth.json"
        if not auth_path.exists():
            return None
        try:
            data = json.loads(auth_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        entry = data.get(profile.name) or data.get(profile.name.replace("-", "_"))
        if isinstance(entry, dict):
            for key in ("access_token", "token", "api_key"):
                if entry.get(key):
                    return str(entry[key])
        return None

    def complete(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> str:
        model_cfg = self._active_model_config()
        provider_id = provider or model_cfg.get("provider") or "openrouter"
        model_id = model or model_cfg.get("default") or "openai/gpt-4o-mini"
        profile = get_provider(str(provider_id))

        resolved_key = self._resolve_api_key(profile, api_key)
        base_url = resolve_base_url(profile, self._config)

        if profile.name == "custom" and not base_url:
            raise ValueError("custom provider requires model.base_url in config.yaml")

        extras: dict[str, Any] = {}
        vertex_sa = self._config.get("vertex.service_account")
        if vertex_sa:
            extras["vertex_service_account"] = vertex_sa

        return complete_with_profile(
            profile,
            api_key=resolved_key,
            base_url=base_url,
            model=str(model_id),
            prompt=prompt,
            config_extras=extras,
        )
