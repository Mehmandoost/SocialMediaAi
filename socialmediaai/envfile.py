from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ENV_DENYLIST = frozenset(
    {
        "SMAI_HOME",
        "PATH",
        "HOME",
        "USER",
        "SHELL",
    }
)

_ENV_DENY_PREFIXES = ("LD_", "DYLD_", "PYTHON", "GIT_CONFIG_")


def validate_env_var_name(key: str) -> None:
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    upper = key.upper()
    if upper in _ENV_DENYLIST or upper.startswith(_ENV_DENY_PREFIXES):
        raise ValueError(
            f"Environment variable {key!r} cannot be persisted via the env writer."
        )


def _quote_env_value(value: str) -> str:
    if not value:
        return '""'
    if re.search(r'[\s#"$\\]', value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _env_line_defines_key(line: str, key: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return False
    name = stripped.split("=", 1)[0].strip()
    return name == key


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                continue
            name, raw = stripped.split("=", 1)
            name = name.strip()
            value = raw.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            result[name] = value
    return result


def write_env_value(path: Path, key: str, value: str) -> None:
    validate_env_var_name(key)
    value = value.replace("\n", "").replace("\r", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    lines: list[str] = []
    if path.exists():
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            lines = handle.readlines()

    serialized = f"{key}={_quote_env_value(value)}\n"
    idx = next(
        (i for i, line in enumerate(lines) if _env_line_defines_key(line, key)),
        None,
    )
    if idx is not None:
        lines[idx] = serialized
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(serialized)

    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        os.replace(tmp_name, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def remove_env_value(path: Path, key: str) -> bool:
    validate_env_var_name(key)
    if not path.exists():
        return False
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        lines = handle.readlines()
    new_lines = [line for line in lines if not _env_line_defines_key(line, key)]
    if len(new_lines) == len(lines):
        return False
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(new_lines)
        os.replace(tmp_name, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return True
