from abstractions.iterables import batches
import pytest


def test_batches_with_list_multiple_epochs():
    data = list(range(6))
    result = list(batches(data, batch_size=2, epochs=2))

    single_epoch_expected = [[0, 1], [2, 3], [4, 5]]
    expected = single_epoch_expected + single_epoch_expected

    assert result == expected


def test_batches_with_list_non_exact_division():
    data = list(range(5))
    result = list(batches(data, batch_size=2, epochs=1))

    assert result == [[0, 1], [2, 3], [4]]


def test_batches_with_generator_single_consumption():
    # Generators are exhausted after one pass; check that behaviour.
    generator_data = (i for i in range(5))
    result = list(batches(generator_data, batch_size=2, epochs=2))

    # Only the first epoch will yield items because the generator is exhausted afterwards.
    assert result == [[0, 1], [2, 3], [4]]


def test_batches_with_list_leftover_across_epochs():
    """When the iterable length is not divisible by batch_size, the leftover from the
    first epoch should be filled by items from the next epoch before being yielded."""
    data = list(range(5))  # length 5, batch_size 2 -> leftover of 1 each epoch
    result = list(batches(data, batch_size=2, epochs=2))

    # After the code change, leftover `4` from epoch-1 combines with `0` from epoch-2.
    expected = [[0, 1], [2, 3], [4, 0], [1, 2], [3, 4]]

    assert result == expected


class VariableIterable:
    """An iterable that yields a different sequence each time it's iterated over."""
    def __init__(self, sequences):
        self._sequences = sequences
        self._iter_count = 0

    def __iter__(self):
        seq = self._sequences[self._iter_count]
        self._iter_count += 1
        return iter(seq)


def test_batches_dataset_length_mismatch_raises():
    """
    This test may seem contrived, but I'm concerned about a user who tries to
    use this function on a generator, and just exactly one epoch when asking
    for more.
    """
    var_iterable = VariableIterable([[0, 1, 2], [3, 4]])  # lengths 3 and 2

    with pytest.raises(ValueError):
        list(batches(var_iterable, batch_size=2, epochs=2))
