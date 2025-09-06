from abstractions.iterables import batches, recv_dict_vec
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


def test_recv_dict_vec_basic_functionality():
    """Test basic functionality of recv_dict_vec with a simple transformation."""
    batch = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    }

    def process_item(item):
        return {
            "name": item["name"].upper(),
            "age": item["age"] + 1
        }

    result = recv_dict_vec(["name", "age"], process_item)(batch)

    expected = {
        "name": ["ALICE", "BOB", "CHARLIE"],
        "age": [26, 31, 36]
    }

    assert result == expected


def test_recv_dict_vec_with_none_returns():
    """Test that items returning None are skipped in the output."""
    batch = {
        "value": [1, 2, 3, 4, 5]
    }

    def filter_even(item):
        if item["value"] % 2 == 0:
            return None
        return {"value": item["value"] * 10}

    result = recv_dict_vec(["value"], filter_even)(batch)

    expected = {
        "value": [10, 30, 50]  # 2 and 4 were filtered out
    }

    assert result == expected


def test_recv_dict_vec_single_item():
    """Test with a batch containing a single item."""
    batch = {
        "text": ["hello"]
    }

    def add_exclamation(item):
        return {"text": item["text"] + "!"}

    result = recv_dict_vec(["text"], add_exclamation)(batch)

    expected = {
        "text": ["hello!"]
    }

    assert result == expected


def test_recv_dict_vec_empty_lists():
    """Test with a batch containing empty lists."""
    batch = {
        "data": []
    }

    def process_item(item):
        return {"data": item["data"]}

    result = recv_dict_vec(["data"], process_item)(batch)

    expected = {
        "data": []
    }

    assert result == expected
