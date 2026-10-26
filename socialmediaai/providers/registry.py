from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

from socialmediaai.config import Config


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    api_mode: str = "chat_completions"
    auth_type: str = "api_key"
    base_url: str = ""
    env_vars: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    signup_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderProfile:
        return cls(
            name=data["name"],
            api_mode=data.get("api_mode", "chat_completions"),
            auth_type=data.get("auth_type", "api_key"),
            base_url=data.get("base_url", "") or "",
            env_vars=tuple(data.get("env_vars") or ()),
            aliases=tuple(data.get("aliases") or ()),
            display_name=data.get("display_name", "") or data["name"],
            signup_url=data.get("signup_url", "") or "",
        )


def _load_catalog() -> list[ProviderProfile]:
    raw = resources.files("socialmediaai.providers").joinpath("catalog.json").read_text(encoding="utf-8")
    items = json.loads(raw)
    return [ProviderProfile.from_dict(item) for item in items]


@lru_cache(maxsize=1)
def _by_name() -> dict[str, ProviderProfile]:
    mapping: dict[str, ProviderProfile] = {}
    for profile in _load_catalog():
        mapping[profile.name] = profile
        for alias in profile.aliases:
            mapping[alias] = profile
    return mapping


def list_providers() -> list[ProviderProfile]:
    seen: set[str] = set()
    result: list[ProviderProfile] = []
    for profile in _load_catalog():
        if profile.name in seen:
            continue
        seen.add(profile.name)
        result.append(profile)
    return result


def get_provider(provider_id: str) -> ProviderProfile:
    key = provider_id.strip().lower()
    profile = _by_name().get(key)
    if profile is None:
        raise KeyError(f"Unknown provider: {provider_id}")
    return profile


def _is_url_env(name: str) -> bool:
    upper = name.upper()
    return upper.endswith("_BASE_URL") or upper.endswith("_URL")


def primary_env_var(profile: ProviderProfile) -> str | None:
    for name in profile.env_vars:
        if _is_url_env(name):
            continue
        return name
    return None


def resolve_base_url(profile: ProviderProfile, config: Config) -> str:
    model_cfg = config.load().get("model") or {}
    if isinstance(model_cfg, dict):
        override = model_cfg.get("base_url")
        if override:
            return str(override).rstrip("/")
    for name in profile.env_vars:
        if _is_url_env(name):
            val = config.get_secret(name)
            if val:
                return val.rstrip("/")
    return profile.base_url.rstrip("/")
