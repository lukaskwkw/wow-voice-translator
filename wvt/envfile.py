"""Minimal ``.env`` loader: KEY=VALUE lines from the git-ignored .env next to the project.

Secrets (ELEVENLABS_API_KEY, MONGODB_URI) live only there. Values are put into
``os.environ`` and are never logged; callers may log *which* keys were found.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import ROOT

ENV_PATH = ROOT / ".env"
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env(path: Path = ENV_PATH, override: bool = False) -> list[str]:
    """Load ``path`` into os.environ (existing variables win unless override). Returns the key names."""
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def secret(name: str) -> str:
    """Value of an environment secret ('' when missing)."""
    return os.environ.get(name, "").strip()


def set_secret(name: str, value: str, path: Path = ENV_PATH) -> None:
    """Set KEY=VALUE in .env, preserving comments and other lines. Never log the value."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid secret name: {name!r}")
    if not value:
        raise ValueError("secret value must not be empty")
    if "\n" in value or "\r" in value:
        raise ValueError("secret value must be a single line")

    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    key_re = re.compile(rf"^(?:export\s+)?{re.escape(name)}\s*=")
    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if not replaced and key_re.match(stripped):
            prefix = "export " if stripped.startswith("export ") else ""
            out.append(f"{prefix}{name}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{name}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    os.environ[name] = value
