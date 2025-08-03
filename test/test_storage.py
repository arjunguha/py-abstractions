import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple, AsyncIterator, Awaitable, Callable
import asyncio
import pytest

from abstractions.storage import create_or_resume_jsonl_file, map_by_key_jsonl_file
import tempfile

DATA_DIR = Path(__file__).parent / "test_data"
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


def assert_permutation(path1: Path, path2: Path):
    lines1 = path1.read_text().splitlines()
    lines2 = path2.read_text().splitlines()
    if not Counter(lines1) == Counter(lines2):
        print(f"Lines of {path1}:")
        for line in lines1:
            print(f"  {line}")
        print(f"Lines of {path2}:")
        for line in lines2:
            print(f"  {line}")
        assert False


@pytest.mark.asyncio
async def test_map_jsonl_trivial(tmp_path):
    async def task(row):
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        DATA_DIR / "in_trivial.jsonl",
        tmp_path / "out.jsonl",
        task,
        key="key",
        keep_columns=["other"],
        num_concurrent=1,
        on_error="raise",
    )
    assert_permutation(tmp_path / "out.jsonl", DATA_DIR / "out_trivial.jsonl")


@pytest.mark.asyncio
async def test_map_jsonl_out_of_order_finish(tmp_path):
    # WARNING: This test would deadlock if num_concurrent=1. Do not write
    # code like this, which has an inter-row dependency.

    second_row_finished = asyncio.Event()

    async def task(row):
        if row["key"] == 10:
            await second_row_finished.wait()
        if row["key"] == 20:
            second_row_finished.set()
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        DATA_DIR / "in_trivial.jsonl",
        tmp_path / "out.jsonl",
        task,
        key="key",
        keep_columns=["other"],
        num_concurrent=2,
        on_error="raise",
    )
    assert_permutation(tmp_path / "out.jsonl", DATA_DIR / "out_trivial.jsonl")


@pytest.mark.asyncio
async def test_map_jsonl_count_calls(tmp_path):
    key_counts = Counter()

    async def task(row):
        key_counts[row["key"]] += 1
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        DATA_DIR / "in_trivial.jsonl",
        tmp_path / "out.jsonl",
        task,
        key="key",
        keep_columns=["other"],
        num_concurrent=1,
        on_error="raise",
    )
    assert_permutation(tmp_path / "out.jsonl", DATA_DIR / "out_trivial.jsonl")

    assert key_counts == {10: 1, 20: 1}


@pytest.mark.asyncio
async def test_resume_does_not_recompute(tmp_path):
    src = DATA_DIR / "in_trivial.jsonl"
    dst = tmp_path / "out.jsonl"

    # Pretend key=10 was already done.
    precomputed = {"other": "A", "key": 10, "result": 11}
    dst.write_text(json.dumps(precomputed) + "\n")

    call_counter = Counter()

    async def task(row):
        call_counter[row["key"]] += 1
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        src,
        dst,
        task,
        key="key",
        num_concurrent=1,
        keep_columns=["other"],
        on_error="raise",
    )

    # 10 should **not** be recomputed; 20 should be computed once.
    assert call_counter == {20: 1}
    assert_permutation(dst, DATA_DIR / "out_trivial.jsonl")


@pytest.mark.asyncio
async def test_empty_keep_columns(tmp_path):
    dst = tmp_path / "out.jsonl"

    async def task(row):
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        DATA_DIR / "in_trivial.jsonl",
        dst,
        task,
        key="key",
        num_concurrent=1,
        keep_columns=[],  # ← empty
        on_error="raise",
    )

    with dst.open() as fp:
        for line in fp:
            data = json.loads(line)
            assert set(data.keys()) == {"key", "result"}


@pytest.mark.asyncio
async def test_f_error_print(tmp_path):
    """If `f` raises and on_error='print', rows for that key are skipped."""
    dst = tmp_path / "out.jsonl"

    async def task(row):
        if row["key"] == 10:
            raise RuntimeError("boom")
        return {"result": row["key"] + 1, "key": row["key"]}

    await map_by_key_jsonl_file(
        DATA_DIR / "in_trivial.jsonl",
        dst,
        task,
        key="key",
        num_concurrent=2,
        keep_columns=["other"],
        on_error="print",
    )

    # Only the key 20 row should remain.
    with dst.open() as fp:
        rows = [json.loads(line) for line in fp]
    assert len(rows) == 1 and rows[0]["key"] == 20


@pytest.mark.asyncio
async def test_f_error_raise(tmp_path):
    """If `f` raises and on_error='raise', the wrapper should propagate."""
    dst = tmp_path / "out.jsonl"

    async def task(row):
        raise RuntimeError("boom")

    with pytest.raises(ExceptionGroup):
        await map_by_key_jsonl_file(
            DATA_DIR / "in_trivial.jsonl",
            dst,
            task,
            key="key",
            num_concurrent=1,
            keep_columns=["other"],
            on_error="raise",
        )


@pytest.mark.asyncio
async def test_map_jsonl_error_raise_writes_prior_rows(tmp_path):
    """
    If `f` fails for the second key (20) with on_error='raise', the row for the
    first key (10) should already be in the output file before the exception
    propagates.
    """
    dst = tmp_path / "out.jsonl"

    async def task(row):
        if row["key"] == 20:
            raise RuntimeError("boom")
        return {"result": row["key"] + 1, "key": row["key"]}

    # Expect ValueError from _error(..., on_error="raise")
    with pytest.raises(ExceptionGroup):
        await map_by_key_jsonl_file(
            DATA_DIR / "in_trivial.jsonl",
            dst,
            task,
            key="key",
            num_concurrent=1,  # sequential ensures 10 finishes before 20
            keep_columns=["other"],
            on_error="raise",
        )

    # The file should contain only rows whose key is 10.
    with dst.open() as fp:
        rows = [json.loads(line) for line in fp]

    assert rows, "Row for key 10 should have been written before failure"
    assert all(r["key"] == 10 for r in rows)
