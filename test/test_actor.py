import asyncio
import gc
import inspect
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from abstractions.actor import (
    ActorDiedError, ActorError, ActorRef, actor, terminate,
)


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



@actor
class BrokenConstructor:
    def __init__(self) -> None:
        raise RuntimeError("broken constructor")


class ConstructorError(Exception):
    def __init__(self) -> None:
        super().__init__("constructor failed")
        self.lock = threading.Lock()


@actor
class UnserializableConstructor:
    def __init__(self) -> None:
        raise ConstructorError()


@actor
class ClassWithLock:
    lock = threading.Lock()

    async def has_lock(self) -> bool:
        return self.lock.acquire(blocking=False)


@actor
class BusyChild:
    def __init__(self, completed: Path) -> None:
        self.completed = completed
        self.started = asyncio.Event()

    async def work(self) -> None:
        self.started.set()
        await asyncio.sleep(0.3)
        self.completed.touch()

    async def wait_started(self) -> None:
        await self.started.wait()


@actor
class Parent:
    async def spawn(self, completed: Path) -> ActorRef[Any]:
        return BusyChild(completed)


@actor
class Terminator:
    async def stop(self, ref: ActorRef[Any]) -> None:
        await terminate(ref)


@actor
class SelfMessaging:
    __slots__ = ("ref",)
    as_actor: Callable[[], ActorRef[Any]]

    def __init__(self) -> None:
        self.ref = self.as_actor()

    async def append(self, values: list[int]) -> list[int]:
        values.append(1)
        return values

    async def send(self) -> tuple[list[int], list[int]]:
        values: list[int] = []
        result = await self.ref.append(values)
        return values, result

    def reference(self) -> ActorRef[Any]:
        return self.as_actor()


@actor
class SelfStopping:
    as_actor: Callable[[], ActorRef[Any]]

    async def stop(self) -> str:
        ref = self.as_actor()
        copied = pickle.loads(pickle.dumps(ref))
        await copied.terminate()
        await ref.terminate()
        await terminate(ref)
        await asyncio.sleep(0.01)
        return "finished"

    def stop_sync(self) -> str:
        asyncio.run(self.as_actor().terminate())
        return "finished"


@actor
class SelfStoppingParent:
    as_actor: Callable[[], ActorRef[Any]]

    async def spawn_and_stop(self) -> ActorRef[Any]:
        child = Counter()
        await self.as_actor().terminate()
        return child


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
    with pytest.raises(RuntimeError, match="broken constructor"):
        BrokenConstructor()


def test_local_actor_classes_are_rejected() -> None:
    with pytest.raises(ActorError, match="module scope"):
        @actor
        class Local:
            async def echo(self, value: str) -> str:
                return value


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


@pytest.mark.parametrize("keyword", [False, True])
def test_unserializable_constructor_arguments_raise_actor_error(keyword: bool) -> None:
    with pytest.raises(ActorError, match="Could not serialize actor constructor arguments"):
        if keyword:
            Counter(initial=threading.Lock())
        else:
            Counter(threading.Lock())


@pytest.mark.asyncio
async def test_actor_class_is_imported_without_pickling_its_attributes() -> None:
    ref = ClassWithLock()
    try:
        assert await ref.has_lock()
    finally:
        await ref.terminate()


def test_unserializable_constructor_exception_raises_actor_error() -> None:
    with pytest.raises(ActorError, match="Could not serialize actor response"):
        UnserializableConstructor()


@pytest.mark.asyncio
async def test_cancelled_sync_waiter_unblocks_async_calls() -> None:
    from abstractions.actor import _MethodLock

    lock = _MethodLock()
    await lock.acquire_async()
    writer = asyncio.create_task(lock.acquire_sync())
    await asyncio.sleep(0)
    reader = asyncio.create_task(lock.acquire_async())
    await asyncio.sleep(0)
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    try:
        await asyncio.wait_for(reader, timeout=0.5)
        await lock.release_async()
    finally:
        await lock.release_async()


@pytest.mark.asyncio
async def test_transferred_termination_waits_for_owned_actors(tmp_path: Path) -> None:
    completed = tmp_path / "completed"

    parent = Parent()
    terminator = Terminator()
    child = await parent.spawn(completed)
    work = asyncio.create_task(child.work())
    try:
        await child.wait_started()
        await terminator.stop(parent)
        assert completed.exists(), "termination returned before the child finished"
        with pytest.raises(ActorDiedError):
            await child.wait_started()
    finally:
        await work
        await terminate(parent)
        await terminate(terminator)


@pytest.mark.asyncio
async def test_self_reference_sends_messages_and_can_be_transferred() -> None:
    ref = SelfMessaging()
    try:
        # A message serializes its arguments; a direct self.append call would
        # instead mutate the original list.
        assert await asyncio.wait_for(ref.send(), timeout=5) == ([], [1])
        with pytest.raises(AttributeError):
            getattr(ref, "as_actor")
        transferred = await ref.reference()
        assert await transferred.append([2]) == [2, 1]
    finally:
        await ref.terminate()


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.asyncio
async def test_self_termination_finishes_current_call(synchronous: bool) -> None:
    ref = SelfStopping()
    try:
        method = ref.stop_sync if synchronous else ref.stop
        assert await asyncio.wait_for(method(), timeout=5) == "finished"
    finally:
        await ref.terminate()
    with pytest.raises(ActorDiedError):
        await ref.stop()


@pytest.mark.asyncio
async def test_self_termination_shuts_down_children() -> None:
    ref = SelfStoppingParent()
    child = await asyncio.wait_for(ref.spawn_and_stop(), timeout=5)
    await ref.terminate()
    with pytest.raises(ActorDiedError):
        await child.add()


def test_terminate_is_reserved_on_actor_references() -> None:
    with pytest.raises(ActorError, match="reserved"):
        @actor
        class Conflicting:
            async def terminate(self) -> None:
                pass


@pytest.mark.parametrize("inherited", [False, True])
def test_as_actor_does_not_replace_existing_attributes(inherited: bool) -> None:
    class Existing:
        def as_actor(self) -> str:
            return "existing method"

    class Child(Existing):
        pass

    cls = Child if inherited else Existing
    with pytest.raises(ActorError, match="'as_actor' is reserved"):
        actor(cls)
    assert cls().as_actor() == "existing method"


def _reject_executor(*args: object, **kwargs: object) -> Any:
    raise AssertionError("Actor communication must not use an executor")


@actor
class TransportProbe:
    as_actor: Callable[[], ActorRef[Any]]

    def __init__(self) -> None:
        # This patch runs inside the actor subprocess, including for its
        # outgoing self messages. It also catches implicit DNS executor use.
        setattr(asyncio.BaseEventLoop, "run_in_executor", _reject_executor)
        self.started = asyncio.Event()

    async def echo(self, value: object) -> object:
        return value

    async def fanout(self, count: int) -> list[int]:
        ref = self.as_actor()
        return await asyncio.gather(*(ref.echo(i) for i in range(count)))

    async def recursive(self, depth: int) -> int:
        if depth == 0:
            return 0
        return 1 + await self.as_actor().recursive(depth - 1)

    async def threads(self) -> list[int | None]:
        return [thread.ident for thread in threading.enumerate()]

    def sync_threads(self) -> tuple[int, list[int | None]]:
        return threading.get_ident(), [thread.ident for thread in threading.enumerate()]

    async def work_during_shutdown(self) -> str:
        self.started.set()
        await asyncio.sleep(0.2)
        return await self.as_actor().echo("finished")

    async def wait_started(self) -> None:
        await self.started.wait()

    async def self_stop_and_message(self) -> str:
        await self.as_actor().terminate()
        await asyncio.sleep(0.01)
        return await self.as_actor().echo("finished")

    def fail_sync(self) -> None:
        raise ValueError("sync failure")


@pytest.mark.asyncio
async def test_many_self_messages_without_executors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", _reject_executor)
    ref = TransportProbe()
    try:
        assert await asyncio.wait_for(ref.fanout(96), timeout=10) == list(range(96))
        assert await asyncio.wait_for(ref.recursive(80), timeout=10) == 80
        assert len(await ref.threads()) == 1
    finally:
        await ref.terminate()


@pytest.mark.asyncio
async def test_sync_methods_share_exactly_one_auxiliary_thread() -> None:
    ref = TransportProbe()
    try:
        main_threads = await ref.threads()
        assert len(main_threads) == 1
        results = await asyncio.wait_for(
            asyncio.gather(*(ref.sync_threads() for _ in range(40))), timeout=10
        )
        worker_ids = {worker for worker, _ in results}
        assert len(worker_ids) == 1
        assert worker_ids.isdisjoint(main_threads)
        for _, threads in results:
            assert set(threads) == set(main_threads) | worker_ids
        with pytest.raises(ValueError, match="sync failure"):
            await ref.fail_sync()
        assert set(await ref.threads()) == set(main_threads) | worker_ids
    finally:
        await ref.terminate()


@pytest.mark.asyncio
async def test_shutdown_allows_active_calls_to_send_self_messages() -> None:
    ref = TransportProbe()
    work = asyncio.create_task(ref.work_during_shutdown())
    await ref.wait_started()
    result, _ = await asyncio.wait_for(
        asyncio.gather(work, ref.terminate()), timeout=5
    )
    assert result == "finished"
    with pytest.raises(ActorDiedError):
        await ref.echo(None)


@pytest.mark.asyncio
async def test_self_shutdown_allows_current_call_to_send_messages() -> None:
    ref = TransportProbe()
    try:
        assert await asyncio.wait_for(ref.self_stop_and_message(), timeout=5) == "finished"
    finally:
        await ref.terminate()


@pytest.mark.asyncio
async def test_large_messages_and_cancelled_call_do_not_break_transport() -> None:
    ref = TransportProbe()
    try:
        payload = b"x" * (2 * 1024 * 1024)
        assert await ref.echo(payload) == payload
        work = asyncio.create_task(ref.work_during_shutdown())
        await ref.wait_started()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert await ref.echo("still running") == "still running"
    finally:
        await ref.terminate()


@pytest.mark.asyncio
async def test_concurrent_termination_is_idempotent() -> None:
    ref = Counter()
    await asyncio.wait_for(
        asyncio.gather(*(ref.terminate() for _ in range(10))), timeout=5
    )
    with pytest.raises(ActorDiedError):
        await ref.add()


@pytest.mark.asyncio
async def test_termination_closes_idle_connections_before_waiting_for_server() -> None:
    from abstractions.actor import _owned_processes

    ref = Counter()
    _, writer = await asyncio.open_connection(*ref._address)
    # An incomplete request keeps a second connection open during shutdown.
    # Python 3.12's Server.wait_closed() waits for both this connection and
    # the termination connection to close.
    writer.write(b"\x00")
    await writer.drain()
    try:
        await asyncio.wait_for(ref.terminate(), timeout=3)
    finally:
        writer.close()
        await writer.wait_closed()
        # Keep a regression from hanging pytest's process-exit cleanup.
        process = _owned_processes.pop(ref._address, None)
        if process is not None:
            if process.is_alive():
                process.kill()
            process.join()
