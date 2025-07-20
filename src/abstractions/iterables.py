"""
This module contains abstractions for iterables.
"""
from typing import Iterable, Iterator, List, TypeVar


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
