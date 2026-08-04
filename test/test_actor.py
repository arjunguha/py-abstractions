import asyncio
import gc
import inspect
import os
import pickle
from typing import Any, cast

import pytest

from abstractions.actor import ActorError, ActorRef, actor


@actor
class Counter:
    class_value = 40

    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    async def add(self, amount: int = 1) -> int:
        self.value += amount
        return self.value

    async def process_id(self) -> int:
        return os.getpid()

    async def fail(self, message: str) -> None:
        raise ValueError(message)

    async def _private(self) -> str:
        return "private"

    def synchronous(self) -> str:
        return "sync"

    @property
    def doubled(self) -> int:
        return self.value * 2

    @staticmethod
    async def static_value(value: int) -> int:
        return value + 1

    @classmethod
    async def class_method(cls) -> int:
        return cls.class_value + 2


@actor
class Forwarder:
    async def add_via(self, target: ActorRef[Any], amount: int) -> int:
        result = await target.add(amount)
        return cast(int, result)


@actor
class SlowCounter:
    def __init__(self) -> None:
        self.value = 0

    async def increment(self) -> int:
        old_value = self.value
        await asyncio.sleep(0.02)
        self.value = old_value + 1
        return self.value


@actor
class BadResult:
    async def lock(self) -> object:
        import threading

        return threading.Lock()


def _collect_actor(ref: ActorRef[Any]) -> None:
    del ref
    gc.collect()


@pytest.mark.asyncio
async def test_construction_and_async_method_calls() -> None:
    counter = Counter(10)
    try:
        assert inspect.iscoroutinefunction(counter.add)
        assert await counter.add() == 11
        assert await counter.add(4) == 15
        assert await counter.process_id() != os.getpid()
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_only_public_async_methods_are_exposed() -> None:
    counter = Counter()
    try:
        assert await counter.static_value(3) == 4
        assert await counter.class_method() == 42

        with pytest.raises(AttributeError):
            getattr(counter, "value")
        with pytest.raises(AttributeError):
            getattr(counter, "doubled")
        with pytest.raises(AttributeError):
            getattr(counter, "_private")
        with pytest.raises(AttributeError):
            getattr(counter, "synchronous")
        with pytest.raises(AttributeError):
            getattr(counter, "missing")
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_actor_calls_are_serialized() -> None:
    counter = SlowCounter()
    try:
        results = await asyncio.gather(*(counter.increment() for _ in range(8)))
        assert sorted(results) == list(range(1, 9))
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_actor_references_can_be_passed_to_actors() -> None:
    counter = Counter(5)
    forwarder = Forwarder()
    try:
        assert await forwarder.add_via(counter, 7) == 12
        assert await counter.add() == 13
    finally:
        _collect_actor(forwarder)
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_actor_reference_is_pickleable() -> None:
    counter = Counter(2)
    copied = cast(ActorRef[Any], pickle.loads(pickle.dumps(counter)))
    try:
        assert await copied.add(3) == 5
        assert await counter.add() == 6
    finally:
        del copied
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_remote_exceptions_are_raised_by_await() -> None:
    counter = Counter()
    try:
        with pytest.raises(ValueError, match="deliberate"):
            await counter.fail("deliberate")
        assert await counter.add() == 1
    finally:
        _collect_actor(counter)


def test_constructor_exceptions_are_raised_immediately() -> None:
    @actor
    class Broken:
        def __init__(self) -> None:
            raise RuntimeError("broken constructor")

        async def unused(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="broken constructor"):
        Broken()


@pytest.mark.asyncio
async def test_local_actor_classes_work_with_spawn() -> None:
    @actor
    class Local:
        async def echo(self, value: str) -> str:
            return value

    local = Local()
    try:
        assert await local.echo("hello") == "hello"
    finally:
        _collect_actor(local)


@pytest.mark.asyncio
async def test_unserializable_results_raise_actor_error() -> None:
    actor_ref = BadResult()
    try:
        with pytest.raises(ActorError, match="Could not serialize actor response"):
            await actor_ref.lock()
    finally:
        _collect_actor(actor_ref)
