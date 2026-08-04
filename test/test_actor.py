import asyncio
import gc
import inspect
import os
import pickle
import threading
import time
from typing import Any, cast

import pytest

from abstractions import ActorDiedError, ActorError, ActorRef, actor, terminate


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

    async def echo(self, value: object) -> object:
        return value

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
        self.active = 0
        self.both_started = asyncio.Event()

    async def rendezvous(self) -> int:
        self.active += 1
        if self.active == 2:
            self.both_started.set()
        await asyncio.wait_for(self.both_started.wait(), timeout=1)
        return self.active


@actor
class MixedMethods:
    def __init__(self) -> None:
        self.async_active = 0
        self.sync_active = False
        self.overlapped = False

    async def async_work(self, delay: float) -> bool:
        if self.sync_active:
            self.overlapped = True
        self.async_active += 1
        await asyncio.sleep(delay)
        if self.sync_active:
            self.overlapped = True
        self.async_active -= 1
        return self.overlapped

    def sync_work(self, delay: float) -> bool:
        if self.sync_active or self.async_active != 0:
            self.overlapped = True
        self.sync_active = True
        time.sleep(delay)
        if self.async_active != 0:
            self.overlapped = True
        self.sync_active = False
        return self.overlapped


@actor
class BadResult:
    async def lock(self) -> object:
        import threading

        return threading.Lock()


@actor
class Spawner:
    async def spawn_counter(self, initial: int) -> ActorRef[Any]:
        return Counter(initial)

    async def process_id(self) -> int:
        return os.getpid()


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
async def test_only_public_instance_methods_are_exposed() -> None:
    counter = Counter()
    try:
        assert inspect.iscoroutinefunction(counter.synchronous)
        assert await counter.synchronous() == "sync"
        with pytest.raises(AttributeError):
            getattr(counter, "value")
        with pytest.raises(AttributeError):
            getattr(counter, "doubled")
        with pytest.raises(AttributeError):
            getattr(counter, "_private")
        with pytest.raises(AttributeError):
            getattr(counter, "missing")
        with pytest.raises(AttributeError):
            getattr(counter, "static_value")
        with pytest.raises(AttributeError):
            getattr(counter, "class_method")
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_async_actor_calls_run_concurrently() -> None:
    counter = SlowCounter()
    try:
        results = await asyncio.gather(counter.rendezvous(), counter.rendezvous())
        assert results == [2, 2]
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_sync_actor_calls_are_exclusive_with_all_methods() -> None:
    mixed = MixedMethods()
    try:
        results = await asyncio.gather(
            mixed.async_work(0.05),
            mixed.sync_work(0.05),
            mixed.async_work(0.05),
            mixed.sync_work(0.05),
        )
        assert results == [False, False, False, False]
    finally:
        _collect_actor(mixed)


@pytest.mark.asyncio
async def test_sync_actor_call_does_not_block_callers_event_loop() -> None:
    mixed = MixedMethods()
    heartbeat_ran = False

    async def heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0.01)
        heartbeat_ran = True

    try:
        await asyncio.gather(mixed.sync_work(0.05), heartbeat())
        assert heartbeat_ran
    finally:
        _collect_actor(mixed)


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


@pytest.mark.asyncio
async def test_unserializable_arguments_raise_actor_error() -> None:
    counter = Counter()
    try:
        with pytest.raises(ActorError, match="Could not serialize actor request"):
            await counter.echo(threading.Lock())
        assert await counter.add() == 1
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_transferred_ref_survives_original_ref_deletion() -> None:
    original = Counter(10)
    transferred = cast(ActorRef[Any], pickle.loads(pickle.dumps(original)))

    del original
    gc.collect()

    assert await transferred.add(2) == 12


@pytest.mark.asyncio
async def test_actor_can_spawn_an_actor_and_owns_its_lifetime() -> None:
    spawner = Spawner()
    child = cast(ActorRef[Any], await spawner.spawn_counter(20))

    assert await child.add(2) == 22
    assert await child.process_id() != await spawner.process_id()

    await terminate(spawner)

    with pytest.raises(ActorDiedError, match="is not running"):
        await spawner.process_id()
    with pytest.raises(ActorDiedError, match="is not running"):
        await child.add()


@pytest.mark.asyncio
async def test_terminate_is_idempotent_and_calls_raise_actor_died_error() -> None:
    counter = Counter()

    await terminate(counter)
    await terminate(counter)

    with pytest.raises(ActorDiedError, match="is not running"):
        await counter.add()
