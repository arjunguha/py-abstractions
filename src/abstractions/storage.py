"""
This module contains abstractions for data storage.
"""

import json
import sys
from collections import Counter
from typing import (
    Protocol,
    Iterator,
    AsyncIterator,
    Tuple,
    List,
    Awaitable,
    Literal,
    Callable,
    Dict,
    Optional,
)
from pathlib import Path
from abstractions.async_abstractions import run_bounded
from collections import Counter
import asyncio


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


def _num_lines(file_name: Path) -> int:
    if not file_name.exists():
        return 0
    with file_name.open("rt") as f:
        return sum(1 for _ in f)


async def map_by_key_jsonl_file(
    src: Path,
    dst: Path,
    f: Callable[[dict], Awaitable[dict]],
    *,
    key: str,
    num_concurrent: int,
    keep_columns: List[str],
    on_error: OnError,
    progress: Optional[Callable[[],None]] = None,
):
    """
    Apply an async transformation to exactly one representative row from each
    equivalence class in *src* (rows that share *key*), writing a row for every
    line in *src* to *dst*.

    ### Parameters
     
     - `src, dst : Path`: source and destination JSONL files.
     - `f : Callable[[dict], Awaitable[dict]]`: async function invoked once per distinct value of *key*.
     - `key : str`: column whose value defines the equivalence classes.
     - `num_concurrent : int`: maximum number of concurrent invocations of *f*.
     - `keep_columns : List[str]`: columns to copy verbatim from *src* to each output row.
     - `on_error : Literal["print", "raise"]`: how to handle inconsistencies while resuming.
     - `progress`: an optional function that is is called after each row is processed, even if *f* does not receive
       the row. You can use this to display a progress bar.

    ### Behaviour

    - The first time a key is encountered, that row is passed to *f*; its result
       is cached and duplicated for every row in the same class.

    - If *dst* already exists, rows are read to determine which keys are
      complete so the computation can resume without repeating work.

    ### Example

     Consider the following input file:

     ```jsonl
     { "key": 10, "other": "A", "discarded": "X" }
     { "key": 20, "other": "B", "discarded": "Y" }
     { "key": 10, "other": "C", "discarded": "Z" }
     ```

     Consider the following application (inside an async context):

     ```python
     async def compute(row):
         return { "result": row["key"] + 1, "key": row["key"] }

     await map_by_key_jsonl_file(
         src,
         dst,
         f=compute,
         key="key",
         keep_columns=["other"],
         on_error="raise",
         num_concurrent=1,
     )
     ```

     Because the file contains two rows whose key is 10 and one whose key is 20,
     *f* is called twice: once with the row with *key* 10 and once with the row
     with *key* 20.

     The output file *dst* will be a permutation of:

     ```jsonl
     { "key": 10, "result": 11, "other": "A" }
     { "key": 20, "result": 21, "other": "B" }
     { "key": 10, "result": 11, "other": "C" }
     ```

     It omits the *discarded* column. We always emit the *key* column. We
     emit the column *other* because it was specified in the *keep_columns*
     argument. Finally, we include all the columns from *f*'s output.
    """

    # The assumption is that src may be enormous, and we don't want to read
    # all of it into memory.
    MAX_SRC_LINES_TO_HOLD_IN_MEMORY = 100

    f_args_buffer = asyncio.Queue(MAX_SRC_LINES_TO_HOLD_IN_MEMORY)

    # f_results[key] is a future that will hold the result of applying f to the
    # representative row for key. We will eventually hold all results from f
    # in memory. So, f should return a compact result.
    f_results: Dict[str, asyncio.Future[dict]] = {}

    dst_rows_buffer = asyncio.Queue()

    async def read_src_proc(skip_rows: int):
        with src.open("rt") as f:
            for line_num, line in enumerate(f):
                if line_num < skip_rows:
                    continue
                row = json.loads(line)
                row_key = row[key]

                # If row_key not in f_results, this is the representative row.
                # So, we add it to f_args_buffer.
                if row_key not in f_results:
                    f_results[row_key] = asyncio.Future()
                    # The await below stops us from keeping too many full rows
                    # in memory if f is slow.
                    await f_args_buffer.put(row)

                if progress is not None:
                    progress()

                partial_dst_row = {k: row[k] for k in keep_columns}
                partial_dst_row[key] = row_key
                # The await below stops us from keeping too many output rows
                # in memory if f is slow.
                await dst_rows_buffer.put(partial_dst_row)
            # The Nones below signal end of input to the other processes.
            await dst_rows_buffer.put(None)
            for _ in range(num_concurrent):
                await f_args_buffer.put(None)

    async def write_dst_proc():
        with dst.open("at") as dst_f:
            while True:
                # It is partial because we don't know the result of f yet.
                partial_dst_row = await dst_rows_buffer.get()
                if partial_dst_row is None:
                    break
                try:
                    f_result = await f_results[partial_dst_row[key]]
                except:
                    # If f had failed, we skip it on output. We would have either
                    # printed a warning once, or raised an exception earlier that
                    # would have aborted the whole task group.
                    continue

                dst_row = {**partial_dst_row, **f_result}
                json.dump(dst_row, dst_f)
                dst_f.write("\n")
                dst_f.flush()

    async def apply_f_proc():
        while True:
            row = await f_args_buffer.get()
            if row is None:
                break
            row_key = row[key]
            f_slot = f_results[row_key]
            try:
                result = await f(row)
                f_slot.set_result(result)
            except Exception as e:
                f_slot.set_exception(e)
                _error(on_error, f"Error applying f to {row}: {e}")

    def initialize_f_results(dst_f):
        skip_rows = 0
        for line in dst_f:
            skip_rows = skip_rows + 1
            row = json.loads(line)
            row_key = row[key]
            if row_key not in f_results:
                fut = asyncio.Future()
                fut.set_result(
                    {
                        k: row[k]
                        for k in row.keys()
                        if k != key and k not in keep_columns
                    }
                )
                f_results[row_key] = fut
            if progress is not None:
                progress()
        return skip_rows

    async with asyncio.TaskGroup() as tg:
        if dst.exists():
            with dst.open("rt") as dst_f:
                skip_rows = initialize_f_results(dst_f)
        else:
            skip_rows = 0

        tg.create_task(read_src_proc(skip_rows))
        tg.create_task(write_dst_proc())
        for _ in range(num_concurrent):
            tg.create_task(apply_f_proc())


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
