import asyncio
import os
import pickle
import threading
from typing import Any, Callable

import pytest

from abstractions.actor import ActorDiedError, ActorError, ActorRef, actor, local_actor


@local_actor
class LocalCounter:
    __slots__ = ("value", "constructed_on", "ref", "lock")
    as_actor: Callable[[], ActorRef[Any]]

    def __init__(self, initial: int = 0, lock: object = None) -> None:
        self.value = initial
        self.lock = lock
        self.constructed_on = (os.getpid(), threading.get_ident())
        self.ref = self.as_actor()

    async def add(self, amount: int = 1) -> int:
        self.value += amount
        return self.value

    def identity(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.constructed_on, (os.getpid(), threading.get_ident())

    async def async_identity(self) -> tuple[int, int]:
        return os.getpid(), threading.get_ident()

    async def reference(self) -> ActorRef[Any]:
        return self.as_actor()

    async def round_trip(self, peer: ActorRef[Any], amount: int) -> int:
        return await peer.add_via(self.as_actor(), amount)

    async def echo(self, value: object) -> object:
        return value

    async def stop(self) -> str:
        await self.as_actor().terminate()
        await self.as_actor().terminate()
        return await self.as_actor().echo("finished")

    async def stop_other(self, other: ActorRef[Any]) -> None:
        await other.terminate()

    async def fail(self) -> None:
        raise ValueError("local failure")


@actor
class RemotePeer:
    async def add_via(self, target: ActorRef[Any], amount: int) -> int:
        return await target.add(amount)

    async def echo(self, value: object) -> object:
        return value

    async def stop(self, target: ActorRef[Any]) -> None:
        await target.terminate()

    async def create_local(self) -> ActorRef[Any]:
        return LocalCounter()


@local_actor
class LocalParent:
    def children_sync(self) -> tuple[ActorRef[Any], ActorRef[Any]]:
        return LocalCounter(), RemotePeer()

    async def children(self) -> tuple[ActorRef[Any], ActorRef[Any]]:
        return self.children_sync()


@pytest.mark.asyncio
async def test_local_methods_and_constructor_run_on_main_thread() -> None:
    identity = (os.getpid(), threading.main_thread().ident)
    threads = {thread.ident for thread in threading.enumerate()}
    # Local construction can keep non-pickleable resources in the main process.
    ref = LocalCounter(3, threading.Lock())
    try:
        assert await ref.identity() == (identity, identity)
        assert await ref.async_identity() == identity
        assert await ref.add(2) == 5
        assert {thread.ident for thread in threading.enumerate()} == threads
    finally:
        await ref.terminate()


@pytest.mark.asyncio
async def test_local_references_are_pickleable_and_interoperate() -> None:
    local = LocalCounter(4)
    peer = RemotePeer()
    copied = pickle.loads(pickle.dumps(local))
    try:
        assert await copied.add(1) == 5
        assert await peer.add_via(copied, 2) == 7
        # local -> subprocess -> local, while the original local call awaits.
        assert await local.round_trip(peer, 3) == 10
        returned = await peer.echo(local)
        assert await returned.add(1) == 11
        assert (await local.reference())._address == local._address
        with pytest.raises(AttributeError):
            getattr(local, "as_actor")
        with pytest.raises(ValueError, match="local failure"):
            await copied.fail()
        assert await local.add(1) == 12
    finally:
        await local.terminate()
        await peer.terminate()


@pytest.mark.asyncio
async def test_multiple_local_actors_have_distinct_self_references() -> None:
    first, second = LocalCounter(1), LocalCounter(10)
    try:
        assert (await first.reference())._address == first._address
        assert (await second.reference())._address == second._address
        assert await first.stop() == "finished"
        await first.terminate()
        with pytest.raises(ActorDiedError):
            await first.add()
        assert await second.add() == 11
    finally:
        await first.terminate()
        await second.terminate()


@pytest.mark.asyncio
async def test_local_actor_can_terminate_another_local_actor() -> None:
    first, second = LocalCounter(), LocalCounter()
    try:
        await asyncio.wait_for(first.stop_other(second), timeout=5)
        with pytest.raises(ActorDiedError):
            await second.add()
        assert await first.add() == 1
    finally:
        await first.terminate()
        await second.terminate()


@pytest.mark.asyncio
async def test_remote_actor_can_terminate_local_actor() -> None:
    local, peer = LocalCounter(), RemotePeer()
    try:
        await asyncio.wait_for(peer.stop(local), timeout=5)
        with pytest.raises(ActorDiedError):
            await local.add()
        assert await peer.echo("alive") == "alive"
    finally:
        await local.terminate()
        await peer.terminate()


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.asyncio
async def test_local_termination_stops_only_its_own_children(synchronous: bool) -> None:
    parent = LocalParent()
    sibling_local, sibling_remote = LocalCounter(), RemotePeer()
    try:
        method = parent.children_sync if synchronous else parent.children
        child_local, child_remote = await method()
        await asyncio.wait_for(parent.terminate(), timeout=5)
        with pytest.raises(ActorDiedError):
            await child_local.add()
        with pytest.raises(ActorDiedError):
            await child_remote.echo(None)
        assert await sibling_local.add() == 1
        assert await sibling_remote.echo("alive") == "alive"
    finally:
        await parent.terminate()
        await sibling_local.terminate()
        await sibling_remote.terminate()


def test_local_construction_requires_running_loop() -> None:
    with pytest.raises(ActorError, match="running asyncio"):
        LocalCounter()


def test_local_construction_rejects_auxiliary_thread() -> None:
    errors: list[BaseException] = []

    async def attempt() -> None:
        try:
            LocalCounter()
        except ActorError as exception:
            errors.append(exception)

    thread = threading.Thread(target=lambda: asyncio.run(attempt()))
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "main thread" in str(errors[0])


@pytest.mark.asyncio
async def test_local_construction_rejects_subprocess() -> None:
    peer = RemotePeer()
    try:
        with pytest.raises(ActorError, match="main process"):
            await peer.create_local()
    finally:
        await peer.terminate()


@pytest.mark.asyncio
async def test_local_actor_can_be_function_local_and_keeps_constructor_state() -> None:
    state: list[str] = []

    @local_actor
    class Capturing:
        def record(self, value: str) -> None:
            state.append(value)

    ref = Capturing()
    try:
        await ref.record("called")
        assert state == ["called"]
    finally:
        await ref.terminate()


def test_loop_shutdown_stops_local_actor_and_its_children() -> None:
    async def create() -> tuple[ActorRef[Any], ActorRef[Any], ActorRef[Any]]:
        parent = LocalParent()
        local, remote = await parent.children()
        return parent, local, remote

    refs = asyncio.run(create())

    async def verify() -> None:
        for ref, name in zip(refs, ("children", "add", "echo")):
            with pytest.raises(ActorDiedError):
                await getattr(ref, name)()

    asyncio.run(verify())


def test_loop_shutdown_before_local_server_starts_closes_listener() -> None:
    async def create() -> ActorRef[Any]:
        return LocalCounter()

    ref = asyncio.run(create())

    async def verify() -> None:
        with pytest.raises(ActorDiedError):
            await asyncio.wait_for(ref.add(), timeout=2)

    asyncio.run(verify())


@pytest.mark.asyncio
async def test_local_constructor_failure_propagates_and_cleans_up_children() -> None:
    children: list[ActorRef[Any]] = []

    @local_actor
    class Broken:
        def __init__(self) -> None:
            children.extend((LocalCounter(), RemotePeer()))
            raise ValueError("constructor failed")

    with pytest.raises(ValueError, match="constructor failed"):
        Broken()

    async def wait_for_cleanup() -> None:
        from abstractions.actor import _local_actors, _owned_processes

        while any(
            ref._address in _local_actors or ref._address in _owned_processes
            for ref in children
        ):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_for_cleanup(), timeout=5)


@pytest.mark.asyncio
async def test_local_messages_copy_arguments_and_reject_unpickleable_values() -> None:
    state: list[list[int]] = []

    @local_actor
    class Mutating:
        def append(self, values: list[int]) -> list[int]:
            values.append(3)
            state.append(values)
            return values

    ref = Mutating()
    try:
        original = [1, 2]
        assert await ref.append(original) == [1, 2, 3]
        assert original == [1, 2]
        assert state == [[1, 2, 3]]
        with pytest.raises(ActorError, match="serialize actor request"):
            await ref.append(threading.Lock())
    finally:
        await ref.terminate()
