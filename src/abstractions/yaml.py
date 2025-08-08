"""
This module provides a simple interface for loading and saving YAML files.
"""
import re
from pathlib import Path
from io import StringIO
from typing import Any, Union
import yaml

__all__ = ["load_yaml", "save_yaml"]


# ------------------------- Dumping (Emitter) -------------------------

def save_yaml(p: Union[str, Path], data: Any) -> None:
    """Save data as YAML with nice formatting for strings.

    Features:
    - Multi-line strings and strings with YAML‑significant characters are
      emitted using literal block scalars (|), avoiding quotes.
    - Other strings are emitted as plain scalars (unquoted) when possible.
    - Lists and dicts are written in block style with two‑space indentation.
    - Insertion order of dict keys is preserved.

    This emitter focuses on readability over complete YAML spec coverage.
    """
    path = Path(p)
    with path.open("w", encoding="utf-8") as f:
        _emit_value(data, indent=0, into=f)


def _emit_value(value: Any, indent: int, into) -> None:
    if isinstance(value, dict):
        for index, (key, v) in enumerate(value.items()):
            _emit_key_value(key, v, indent, into)
    elif isinstance(value, list):
        for item in value:
            _emit_list_item(item, indent, into)
    else:
        # Top-level scalar
        into.write(_scalar_to_yaml(value) + "\n")


def _emit_key_value(key: Any, value: Any, indent: int, into) -> None:
    key_str = str(key)
    pad = " " * indent
    if isinstance(value, dict):
        into.write(f"{pad}{key_str}:\n")
        _emit_value(value, indent=indent + 2, into=into)
    elif isinstance(value, list):
        into.write(f"{pad}{key_str}:\n")
        _emit_value(value, indent=indent + 2, into=into)
    elif isinstance(value, str) and _should_use_literal_block(value):
        into.write(f"{pad}{key_str}: |-\n")
        _emit_literal_block(value, indent + 2, into)
    else:
        into.write(f"{pad}{key_str}: {_scalar_to_yaml(value)}\n")


def _emit_list_item(item: Any, indent: int, into) -> None:
    pad = " " * indent
    dash = f"{pad}- "
    if isinstance(item, dict):
        into.write(dash + "\n")
        _emit_value(item, indent=indent + 2, into=into)
    elif isinstance(item, list):
        into.write(dash + "\n")
        _emit_value(item, indent=indent + 2, into=into)
    elif isinstance(item, str) and _should_use_literal_block(item):
        into.write(dash.rstrip() + " |-\n")
        _emit_literal_block(item, indent + 2, into)
    else:
        into.write(dash + _scalar_to_yaml(item) + "\n")


def _emit_literal_block(text: str, indent: int, into) -> None:
    indent_str = " " * indent
    lines = text.split("\n")
    for line in lines:
        into.write(f"{indent_str}{line}\n")


def _should_use_literal_block(s: str) -> bool:
    if "\n" in s:
        return True
    # If the string contains characters that tend to trigger quoting in YAML,
    # emit it as a literal block for readability and to avoid quoting.
    if re.search(r"[:#\[\]\{\},&\*]", s):
        return True
    return False


def _scalar_to_yaml(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    # Fallback to string representation
    return str(v)


def load_yaml(p: Union[str, Path]) -> Any:
    """
    Read a YAML file.

    Uses sensible defaults for safe loading.
    """
    path = Path(p)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def load_yaml_string(s: str) -> Any:
    """
    Read a YAML string.

    Uses sensible defaults for safe loading.
    """
    with StringIO(s) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)