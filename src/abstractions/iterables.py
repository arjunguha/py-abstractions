"""
This module contains abstractions for iterables.
"""
from typing import Callable, Iterable, Iterator, List, TypeVar


T = TypeVar("T")


def batches(
    iterable: Iterable[T], *, batch_size: int, epochs: int
) -> Iterator[List[T]]:
    """
    Yields batches of items, for the given number of epochs.

    The final item may not be of length `batch_size. At the epoch boundary,
    a batch may have items from two successive epochs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")

    batch = []
    dataset_len = None
    for i in range(epochs):
        for j, item in enumerate(iterable):
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if dataset_len is None:
            dataset_len = j + 1
        elif j + 1 < dataset_len:
            raise ValueError(
                f"First epoch had {dataset_len} items, but second epoch had {j + 1} items"
            )

    if batch:
        yield batch


def recv_dict_vec(out_keys: List[str], f: Callable):
    """
    The purpose of this abstraction is to make it easier to process batches of
    items with the Hugging Face datasets library.

    This function wraps `f` as follows. It receives a dictionary whose items are
    lists, e.g.,:

    ```python
    { "key_1": [v_11, v_12, v_13], "key_2": [v_21, v_22, v_23], ... }
    ```

    We assume that the dictionary has at least one key.

    It calls `f` on each item:

    ```python
    f({"key_1": v_11, "key_2": v_21})
    f({"key_1": v_12, "key_2": v_22})
    f({"key_1": v_13, "key_2": v_23})
    ...
    ```

    `f` must optionally return a dictionary with exactly the keys in `out_keys`.

    The wrapped function will accumulate all results from `f` into a dictionary
    of lists, similar to the input.

    This means that you can do this:

    ```python
    import datasets

    the_dataset = datasets.load_dataset("openai/openai_humaneval")

    def f(item):
        if len(item["canonical_solution"]) > 50:
            return None
        return {
         "canonical_solution": item["canonical_solution"],
         "prompt": item["prompt"],
        }

    the_dataset = the_dataset.map(
        recv_dict_vec(["prompt", "canonical_solution"], f),
        batched=True,
        batch_size=10
        num_proc=2
    )

    ```
    """
    def wrapper(batch):
        result = { k: [] for k in out_keys }
        # Get num_items from any key in the batch (they should all have the same length)
        batch_keys = list(batch.keys())
        num_items = len(batch[batch_keys[0]])
        for ix in range(num_items):
            reconstructed_item = { k: batch[k][ix] for k in batch_keys }
            f_result = f(reconstructed_item)
            if f_result is None:
                continue
            for k in out_keys:
                result[k].append(f_result[k])
        return result
    return wrapper