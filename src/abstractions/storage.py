"""
This module contains abstractions for data storage.
"""

import json
import sys
from typing import Protocol, Iterator, AsyncIterator, Tuple, List, Awaitable, Literal
from pathlib import Path
from abstractions.async_abstractions import run_bounded


class RowGenerator(Protocol):
    def __call__(
        self, new_value_counts: List[Tuple[str, int]]
    ) -> AsyncIterator[Awaitable[dict]]: ...


OnError = Literal["print", "raise"]


def _error(on_error: OnError, message: str):
    if on_error == "print":
        print(message, file=sys.stderr)
    elif on_error == "raise":
        raise ValueError(message)


async def create_or_resume_jsonl_file(
    file_name: Path,
    key_name: str,
    key_count: int,
    key_generator: Iterator[str],
    value_generator: RowGenerator,
    *,
    on_error: OnError,
):
    """
    An abstraction to help persist generated data to a JSONL file that supports
    resuming from an interrupted run.

    The goal is to produce a JSONL file where each line has the shape:

    ```
    { key_name: value, ... }
    ```

    And each `value` appears exactly `key_count` times. To use this function,
    the caller must be able to generate the list of expected keys with
    `key_generator`, and then produce each row with `value_generator`.

    The `value_generator` receives a list of `(value, count)` tuples, and must
    produce a row with the shape `{ key_name: value, ... }` exactly `count` times.
    """
    if not file_name.exists():
        # Handle the trivial case with trivial code.
        with file_name.open("wt") as f:
            all_values = [(k, key_count) for k in key_generator]
            async for value in value_generator(all_values):
                json.dump(await value, f)
                f.write("\n")
                f.flush()
        return

    # Pass through the file: we compute how many keys need to be generated.
    values_needed = {k: key_count for k in key_generator}
    with file_name.open("rt") as f:
        for line in f:
            data = json.loads(line)
            this_value = data[key_name]
            if this_value not in values_needed:
                _error(
                    on_error,
                    f"{file_name} has {this_value}, but key_generator does not",
                )
                continue

            this_value_count = values_needed[this_value]
            if this_value_count == 0:
                _error(
                    on_error,
                    f"{file_name} has more entries for {this_value} than key_generator demands",
                )
                continue

            values_needed[this_value] = values_needed[this_value] - 1

    # Not significant, but note that all keys_needed may map to 0, in which case
    # the loop below will be trivial.
    with file_name.open("at") as f:
        all_values = [(k, n) for k, n in values_needed.items() if n > 0]
        async for value in value_generator(all_values):
            json.dump(await value, f)
            f.write("\n")
            f.flush()


async def run_bounded_create_or_resume_jsonl_file(
    file_name: Path,
    key_name: str,
    key_count: int,
    key_generator: Iterator[str],
    value_generator: RowGenerator,
    *,
    limit: int,
    on_error: OnError,
):
    """
    Encapsulates the boilerplate needed to compose `create_or_resume_jsonl_file`
    with `run_bounded`.
    """

    async def parallel_value_generator(
        new_value_counts: List[Tuple[str, int]],
    ) -> AsyncIterator[Awaitable[dict]]:
        async for value in run_bounded(value_generator(new_value_counts), limit=limit):
            yield value

    await create_or_resume_jsonl_file(
        file_name=file_name,
        key_name=key_name,
        key_count=key_count,
        key_generator=key_generator,
        value_generator=parallel_value_generator,
        on_error=on_error,
    )
