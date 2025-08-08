import io
import textwrap
from pathlib import Path

import pytest

from abstractions.yaml import load_yaml, load_yaml_string, save_yaml


def test_load_yaml_string_scalars_and_structures():
    data = load_yaml_string(
        textwrap.dedent(
            """
            a: 1
            b:
              - x
              - y
            c:
              k1: v1
              k2: v2
            """
        )
    )
    assert data == {"a": 1, "b": ["x", "y"], "c": {"k1": "v1", "k2": "v2"}}


def test_save_yaml_multiline_strings_literal_block(tmp_path: Path):
    content = """Line one\nLine two\nLine three"""
    data = {"description": content}
    out = tmp_path / "out.yaml"
    save_yaml(out, data)
    txt = out.read_text(encoding="utf-8")

    # Expect literal block scalar for multi-line strings
    assert "description: |" in txt
    assert "Line one" in txt and "Line two" in txt and "Line three" in txt

    # Round-trip should preserve content
    loaded = load_yaml(out)
    assert loaded == data


def test_save_yaml_special_characters_unquoted_using_block(tmp_path: Path):
    # Strings with YAML-significant characters should use literal block to avoid quoting
    s = "value: with: colons, #hash, [brackets], {braces}, &ampersand*"
    data = {"note": s}
    path = tmp_path / "special.yaml"
    save_yaml(path, data)
    text = path.read_text(encoding="utf-8")
    assert "note: |" in text
    assert s in text
    assert "'" not in text and '"' not in text

    # Round-trip
    assert load_yaml(path) == data


def test_plain_scalars_remain_unquoted(tmp_path: Path):
    data = {"title": "Hello world", "count": 3}
    path = tmp_path / "plain.yaml"
    save_yaml(path, data)
    text = path.read_text(encoding="utf-8")
    # Ensure title is not quoted and not a block
    assert "title: |" not in text
    assert "title: \"Hello world\"" not in text
    assert "title: 'Hello world'" not in text

    assert load_yaml(path) == data


def test_order_preservation_and_flow_disabled(tmp_path: Path):
    data = {"z": 1, "a": 2, "list": [1, 2, 3]}
    path = tmp_path / "order.yaml"
    save_yaml(path, data)
    text = path.read_text(encoding="utf-8")

    # Flow disabled implies lists are written one item per line
    assert "- 1" in text and "- 2" in text and "- 3" in text

    # Keys should appear in insertion order (not sorted)
    z_index = text.index("z:")
    a_index = text.index("a:")
    assert z_index < a_index


def test_load_yaml_file(tmp_path: Path):
    p = tmp_path / "in.yaml"
    p.write_text("name: test\n", encoding="utf-8")
    assert load_yaml(p) == {"name": "test"}
