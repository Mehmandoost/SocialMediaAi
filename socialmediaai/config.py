from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from socialmediaai.envfile import (
    read_env_file,
    remove_env_value,
    validate_env_var_name,
    write_env_value,
)
from socialmediaai.errors import ConfigError

_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")


def default_home() -> Path:
    override = os.environ.get("SMAI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".socialmediaai"


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    if "." in key:
        leaf = key.rsplit(".", 1)[-1].upper()
        return leaf.endswith(_SECRET_SUFFIXES) or leaf in {"API_KEY", "KEY", "TOKEN"}
    return upper.endswith(_SECRET_SUFFIXES)


def custom_endpoint_key_env(identity: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", str(identity or "").upper()).strip("_")
    return f"SMAI_CUSTOM_{slug}_API_KEY" if slug else "SMAI_CUSTOM_API_KEY"


class Config:
    """Project-wide configuration: secrets in .env, settings in config.yaml."""

    def __init__(self, home: Path | str | None = None) -> None:
        self._home = Path(home).expanduser() if home is not None else default_home()
        self._env_path = self._home / ".env"
        self._config_path = self._home / "config.yaml"
        self.ensure_home()

    @property
    def home(self) -> Path:
        return self._home

    @property
    def env_path(self) -> Path:
        return self._env_path

    @property
    def config_path(self) -> Path:
        return self._config_path

    def ensure_home(self) -> None:
        self._home.mkdir(parents=True, exist_ok=True)
        try:
            self._home.chmod(0o700)
        except OSError:
            pass

    def save_secret(self, key: str, value: str) -> None:
        write_env_value(self._env_path, key, value)
        os.environ[key] = value

    def get_secret(self, key: str) -> str | None:
        env_val = os.environ.get(key)
        if env_val:
            return env_val
        file_val = read_env_file(self._env_path).get(key)
        if file_val:
            return file_val
        return None

    def remove_secret(self, key: str) -> bool:
        removed = remove_env_value(self._env_path, key)
        if key in os.environ:
            del os.environ[key]
        return removed

    def load(self) -> dict[str, Any]:
        if not self._config_path.exists():
            return {}
        with open(self._config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return self._expand_env_refs(data)

    def save(self, data: dict[str, Any]) -> None:
        self._reject_raw_secrets(data)
        self.ensure_home()
        fd_path = self._config_path
        content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        fd_path.write_text(content, encoding="utf-8")
        try:
            fd_path.chmod(0o600)
        except OSError:
            pass

    def get(self, dotted: str) -> Any:
        if is_secret_key(dotted) and "." not in dotted:
            return self.get_secret(dotted)
        return _get_nested(self.load(), dotted)

    def set(self, dotted: str, value: Any) -> None:
        if is_secret_key(dotted) and "." not in dotted:
            validate_env_var_name(dotted)
            self.save_secret(dotted, str(value))
            return
        data = self.load()
        _set_nested(data, dotted, value)
        self.save(data)

    def _expand_env_refs(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._expand_env_refs(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand_env_refs(v) for v in value]
        if isinstance(value, str):
            match = _ENV_REF_RE.match(value.strip())
            if match:
                resolved = self.get_secret(match.group(1))
                return resolved if resolved is not None else value
        return value

    def _reject_raw_secrets(self, data: dict[str, Any]) -> None:
        def walk(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    full = f"{path}.{key}" if path else key
                    if key == "api_key" and isinstance(val, str) and val and not _ENV_REF_RE.match(val.strip()):
                        raise ConfigError(
                            "Raw API keys cannot be stored in config.yaml; use save_secret and a ${VAR} reference."
                        )
                    walk(val, full)

        walk(copy.deepcopy(data))


def _get_nested(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = data
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value
