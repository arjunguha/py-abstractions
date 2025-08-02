import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple, AsyncIterator, Awaitable, Callable

import pytest

from abstractions.storage import create_or_resume_jsonl_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_value_generator(key_name: str):
    """Create a ValueGenerator that yields an *awaitable* which returns a row.

    The implementation matches the contract expected by ``create_or_resume_jsonl_file``
    (i.e. it produces an ``AsyncIterator[Awaitable[dict]]``).
    """

    async def _generator(
        new_value_counts: List[Tuple[str, int]],
    ) -> AsyncIterator[Awaitable[dict]]:
        for value, count in new_value_counts:
            for _ in range(count):

                async def _produce(v=value):
                    return {key_name: v}

                # ``_produce()`` returns a coroutine, which is awaitable. Yield it so that
                # the caller will ``await`` it.
                yield _produce()

    return _generator


def _read_counts(path: Path, key_name: str):
    with path.open() as fp:
        return Counter(json.loads(line)[key_name] for line in fp)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_jsonl_file(tmp_path):
    """When the file does not exist, the helper should create it from scratch."""
    file_path = tmp_path / "data.jsonl"
    key_name = "letter"
    keys = ["a", "b", "c"]
    key_count = 2

    value_generator = make_value_generator(key_name)

    await create_or_resume_jsonl_file(
        file_path,
        key_name,
        key_count,
        iter(keys),
        value_generator,
        on_error="raise",
    )

    # Ensure file was created with exactly the required number of rows per key.
    counts = _read_counts(file_path, key_name)
    assert counts == {"a": 2, "b": 2, "c": 2}


@pytest.mark.asyncio
async def test_resume_existing_file(tmp_path):
    """If the JSONL file already exists, the helper should append only missing rows."""
    file_path = tmp_path / "data.jsonl"
    key_name = "letter"
    keys = ["a", "b", "c"]
    key_count = 2

    # Pre-populate the file with a partial run:
    initial_rows = [{"letter": "a"}, {"letter": "a"}, {"letter": "b"}]
    with file_path.open("w") as fp:
        for row in initial_rows:
            json.dump(row, fp)
            fp.write("\n")

    value_generator = make_value_generator(key_name)

    await create_or_resume_jsonl_file(
        file_path,
        key_name,
        key_count,
        iter(keys),
        value_generator,
        on_error="raise",
    )

    counts = _read_counts(file_path, key_name)
    assert counts == {"a": 2, "b": 2, "c": 2}


@pytest.mark.asyncio
async def test_invalid_key_raises(tmp_path):
    """Rows whose key value is not in *key_generator* should raise ValueError."""
    file_path = tmp_path / "data.jsonl"
    key_name = "letter"
    key_count = 1

    # Write a row with an unexpected key "z".
    with file_path.open("w") as fp:
        json.dump({"letter": "z"}, fp)
        fp.write("\n")

    value_generator = make_value_generator(key_name)

    with pytest.raises(ValueError):
        await create_or_resume_jsonl_file(
            file_path,
            key_name,
            key_count,
            iter(["a", "b", "c"]),
            value_generator,
            on_error="raise",
        )
