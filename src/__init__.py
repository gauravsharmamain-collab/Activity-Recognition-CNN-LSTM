"""Activity Recognition CNN-LSTM package."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    """Parse a YAML scalar string into a native Python value."""
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None
    if text.startswith("[") and text.endswith("]"):
        return ast.literal_eval(text)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Parse basic indentation-based YAML when PyYAML is unavailable."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = stripped.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            item_val = _parse_scalar(content[2:])
            if isinstance(parent, list):
                parent.append(item_val)
            continue

        if ":" in content:
            key, val_part = content.split(":", 1)
            key = key.strip()
            val_str = val_part.strip()
            if val_str:
                parent[key] = _parse_scalar(val_str)
            else:
                # Look ahead to determine whether nested block is list or dict
                container: Any = {}
                for next_line in lines[idx + 1 :]:
                    next_clean = next_line.split("#", 1)[0].strip()
                    if next_clean:
                        if next_clean.startswith("- "):
                            container = []
                        break
                parent[key] = container
                stack.append((indent, container))
    return root


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration from disk.

    Uses ``PyYAML`` if installed, falling back to a lightweight parser for
    standard key-value/list YAML files.
    """
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _minimal_yaml_load(text)
