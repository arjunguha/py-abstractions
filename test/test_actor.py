import asyncio
import gc
import inspect
import multiprocessing
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

    def value_now(self) -> int:
        return self.value

    def replace(self, value: int) -> int:
        self.value = value
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
class ConcurrentProbe:
    def __init__(self) -> None:
        self.current = 0
        self.max_current = 0

    async def work(self) -> int:
        self.current += 1
        self.max_current = max(self.max_current, self.current)
        await asyncio.sleep(0.05)
        self.current -= 1
        return self.max_current

    async def max_seen(self) -> int:
        return self.max_current


@actor
class MixedProbe:
    def __init__(self) -> None:
        self.in_sync = False
        self.in_async = 0
        self.overlap = False

    def sync_work(self) -> bool:
        if self.in_async:
            self.overlap = True
        self.in_sync = True
        time.sleep(0.05)
        self.in_sync = False
        return self.overlap

    async def async_work(self) -> bool:
        if self.in_sync:
            self.overlap = True
        self.in_async += 1
        await asyncio.sleep(0.05)
        self.in_async -= 1
        return self.overlap

    def saw_overlap(self) -> bool:
        return self.overlap


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
async def test_sync_methods_are_exposed_and_block() -> None:
    counter = Counter(10)
    try:
        assert not inspect.iscoroutinefunction(counter.synchronous)
        assert counter.synchronous() == "sync"
        assert counter.value_now() == 10
        assert counter.replace(42) == 42
        assert await counter.add() == 43
        assert counter.value_now() == 43
    finally:
        _collect_actor(counter)


@pytest.mark.asyncio
async def test_only_public_instance_methods_are_exposed() -> None:
    counter = Counter()
    try:
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
async def test_async_methods_run_concurrently() -> None:
    probe = ConcurrentProbe()
    try:
        results = await asyncio.gather(*(probe.work() for _ in range(8)))
        assert max(results) > 1
        assert await probe.max_seen() > 1
    finally:
        _collect_actor(probe)


@pytest.mark.asyncio
async def test_sync_methods_exclude_other_operations() -> None:
    probe = MixedProbe()
    try:
        results = await asyncio.gather(
            asyncio.to_thread(probe.sync_work),
            probe.async_work(),
            probe.async_work(),
            asyncio.to_thread(probe.sync_work),
            probe.async_work(),
        )
        assert results == [False, False, False, False, False]
        assert probe.saw_overlap() is False
    finally:
        _collect_actor(probe)


@pytest.mark.asyncio
async def test_async_call_does_not_block_caller_while_sync_holds_lock() -> None:
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()

    @actor
    class HoldProbe:
        def __init__(self, entered_event: Any, release_event: Any) -> None:
            self._entered = entered_event
            self._release = release_event
            self.ran_during_hold = False

        def hold(self) -> None:
            self._entered.set()
            self._release.wait(timeout=5)

        async def marker(self) -> str:
            if not self._release.is_set():
                self.ran_during_hold = True
            return "done"

        def saw_run_during_hold(self) -> bool:
            return self.ran_during_hold

    probe = HoldProbe(entered, release)
    try:
        hold_task = asyncio.create_task(asyncio.to_thread(probe.hold))
        assert await asyncio.to_thread(entered.wait, 5)

        marker_task = asyncio.create_task(probe.marker())
        # Sending the async call must not block this event loop: we can still
        # release the sync method while the async call is waiting for the lock.
        await asyncio.sleep(0.05)
        assert not marker_task.done()
        release.set()

        await hold_task
        assert await marker_task == "done"
        assert probe.saw_run_during_hold() is False
    finally:
        release.set()
        _collect_actor(probe)


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
        assert copied.value_now() == 6
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

        def tag(self) -> str:
            return "local"

    local = Local()
    try:
        assert await local.echo("hello") == "hello"
        assert local.tag() == "local"
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
