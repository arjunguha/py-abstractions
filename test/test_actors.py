"""Tests for the simplified subprocess actor abstraction."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from abstractions.actors import ActorRef, actor, _exposed_method_names


@actor
class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self._secret = "hidden"

    async def increment(self) -> int:
        self.value += 1
        return self.value

    async def add(self, n: int) -> int:
        self.value += n
        return self.value

    async def get(self) -> int:
        return self.value

    def sync_method(self) -> int:
        return self.value

    async def _private(self) -> int:
        return self.value

    @staticmethod
    async def static_async() -> int:
        return 1

    @classmethod
    async def class_async(cls) -> int:
        return 2


@actor
class Greeter:
    def __init__(self, name: str) -> None:
        self._name = name

    async def greet(self, whom: str) -> str:
        return f"{self._name} says hello to {whom}"


@actor
class Caller:
    """Actor that receives and invokes other actor refs."""

    async def bump(self, counter: ActorRef[Any], times: int) -> int:
        last = 0
        for _ in range(times):
            last = await counter.increment()
        return last

    async def bounce(self, counter: ActorRef[Any]) -> ActorRef[Any]:
        await counter.increment()
        return counter

    async def greet_via(self, greeter: ActorRef[Any], whom: str) -> str:
        return await greeter.greet(whom)


@actor
class Failing:
    async def boom(self) -> None:
        raise ValueError("nope")

    async def ok(self) -> str:
        return "ok"


@actor
class Slow:
    async def sleep_and_return(self, delay: float, value: int) -> int:
        await asyncio.sleep(delay)
        return value


def test_exposed_method_names_filters_correctly() -> None:
    names = _exposed_method_names(Counter.__actor_class__)
    assert "increment" in names
    assert "add" in names
    assert "get" in names
    assert "sync_method" not in names
    assert "_private" not in names
    assert "static_async" not in names
    assert "class_async" not in names
    assert "value" not in names
    assert "_secret" not in names


@pytest.mark.asyncio
async def test_construction_returns_actor_ref() -> None:
    counter = Counter()
    assert isinstance(counter, ActorRef)
    await counter.close()


@pytest.mark.asyncio
async def test_basic_method_calls() -> None:
    async with Counter(10) as counter:
        assert await counter.get() == 10
        assert await counter.increment() == 11
        assert await counter.add(4) == 15
        assert await counter.get() == 15


@pytest.mark.asyncio
async def test_fields_not_exposed() -> None:
    async with Counter(0) as counter:
        with pytest.raises(AttributeError):
            _ = counter.value  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            _ = counter._secret  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sync_methods_not_exposed() -> None:
    async with Counter(0) as counter:
        with pytest.raises(AttributeError):
            await counter.sync_method()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_private_methods_not_exposed() -> None:
    async with Counter(0) as counter:
        with pytest.raises(AttributeError):
            await counter._private()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unknown_method_not_exposed() -> None:
    async with Counter(0) as counter:
        with pytest.raises(AttributeError):
            await counter.does_not_exist()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_static_and_class_methods_not_exposed() -> None:
    async with Counter(0) as counter:
        with pytest.raises(AttributeError):
            await counter.static_async()  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            await counter.class_async()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_remote_exception_propagates() -> None:
    async with Failing() as failing:
        with pytest.raises(ValueError, match="nope"):
            await failing.boom()
        # Actor still usable after a failure.
        assert await failing.ok() == "ok"


@pytest.mark.asyncio
async def test_pass_actor_ref_between_actors() -> None:
    async with Counter(0) as counter, Caller() as caller:
        result = await caller.bump(counter, 5)
        assert result == 5
        assert await counter.get() == 5


@pytest.mark.asyncio
async def test_actor_ref_returned_from_actor() -> None:
    async with Counter(0) as counter, Caller() as caller:
        returned = await caller.bounce(counter)
        assert isinstance(returned, ActorRef)
        assert await returned.increment() == 2
        assert await counter.get() == 2


@pytest.mark.asyncio
async def test_multiple_actor_types() -> None:
    async with Greeter("Ada") as greeter, Caller() as caller:
        assert await greeter.greet("Bob") == "Ada says hello to Bob"
        assert await caller.greet_via(greeter, "Bob") == "Ada says hello to Bob"


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_actor() -> None:
    async with Counter(0) as counter:
        results = await asyncio.gather(
            *(counter.increment() for _ in range(20))
        )
        assert sorted(results) == list(range(1, 21))
        assert await counter.get() == 20


@pytest.mark.asyncio
async def test_concurrent_calls_across_actors() -> None:
    async with Slow() as a, Slow() as b:
        results = await asyncio.gather(
            a.sleep_and_return(0.1, 1),
            b.sleep_and_return(0.1, 2),
        )
        assert set(results) == {1, 2}


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    counter = Counter()
    await counter.close()
    await counter.close()


@pytest.mark.asyncio
async def test_call_after_close_raises() -> None:
    counter = Counter()
    await counter.close()
    with pytest.raises(RuntimeError, match="closed"):
        await counter.increment()


@pytest.mark.asyncio
async def test_non_owning_ref_close_does_not_kill_actor() -> None:
    async with Counter(0) as counter, Caller() as caller:
        returned = await caller.bounce(counter)
        await returned.close()
        # Original owner still works.
        assert await counter.increment() == 2


@pytest.mark.asyncio
async def test_state_is_isolated_per_actor() -> None:
    async with Counter(0) as a, Counter(100) as b:
        await a.increment()
        await b.add(5)
        assert await a.get() == 1
        assert await b.get() == 105


@pytest.mark.asyncio
async def test_kwargs_and_args() -> None:
    async with Greeter(name="Eve") as greeter:
        assert await greeter.greet(whom="Frank") == "Eve says hello to Frank"
