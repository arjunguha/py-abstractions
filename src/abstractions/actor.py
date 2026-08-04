"""Small, subprocess-backed actors.

Decorating a class with :func:`actor` replaces the class with a callable that
starts an instance in a spawned subprocess. Public instance methods are
available on the returned :class:`ActorRef`.
"""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from multiprocessing.context import SpawnProcess
from multiprocessing.util import Finalize
from typing import Any, Generic, Protocol, TypeVar, cast

import cloudpickle
from typing_extensions import override


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
AsyncMethod = Callable[..., Awaitable[Any]]
SyncMethod = Callable[..., Any]
Address = tuple[str, int]
_owned_processes: dict[Address, SpawnProcess] = {}


class ActorClass(Protocol[T_co]):
    """The callable produced by :func:`actor`."""

    __name__: str
    __qualname__: str
    __doc__: str | None
    __module__: str

    def __call__(self, *args: object, **kwargs: object) -> ActorRef[T_co]: ...


class ActorError(RuntimeError):
    """Raised when an actor cannot start or communicate."""


class ActorDiedError(ActorError):
    """Raised when a method call targets an actor that is no longer running."""


@dataclass(frozen=True)
class _Call:
    method: str
    args: tuple[object, ...]
    kwargs: Mapping[str, object]


@dataclass(frozen=True)
class _Result:
    value: object


@dataclass(frozen=True)
class _Failure:
    exception: BaseException


@dataclass(frozen=True)
class _Terminate:
    pass


_Request = _Call | _Terminate
_Response = _Result | _Failure


class _ActorGate:
    """Reader-writer gate for actor method execution.

    Async methods take shared access and may run concurrently. Sync methods take
    exclusive access and exclude both sync and async methods while they run.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._shared = 0
        self._exclusive = False

    async def acquire_shared(self) -> None:
        async with self._condition:
            while self._exclusive:
                await self._condition.wait()
            self._shared += 1

    async def release_shared(self) -> None:
        async with self._condition:
            self._shared -= 1
            if self._shared == 0:
                self._condition.notify_all()

    async def acquire_exclusive(self) -> None:
        async with self._condition:
            while self._exclusive or self._shared > 0:
                await self._condition.wait()
            self._exclusive = True

    async def release_exclusive(self) -> None:
        async with self._condition:
            self._exclusive = False
            self._condition.notify_all()


def _serialize(value: object) -> bytes:
    return cloudpickle.dumps(value)


def _deserialize(data: bytes) -> object:
    return cloudpickle.loads(data)


def _send(connection: Connection, value: object) -> None:
    connection.send_bytes(_serialize(value))


def _receive(connection: Connection) -> object:
    return _deserialize(connection.recv_bytes())


def _exposed_methods(cls: type[object]) -> tuple[frozenset[str], frozenset[str]]:
    async_methods: set[str] = set()
    sync_methods: set[str] = set()
    for name, descriptor in inspect.getmembers_static(cls):
        if name.startswith("_"):
            continue
        if isinstance(descriptor, (staticmethod, classmethod)):
            continue
        if inspect.iscoroutinefunction(descriptor):
            async_methods.add(name)
        elif inspect.isfunction(descriptor):
            sync_methods.add(name)
    return frozenset(async_methods), frozenset(sync_methods)


def _terminate_at_address(address: Address) -> None:
    try:
        with Client(address) as connection:
            _send(connection, _Terminate())
            try:
                connection.recv_bytes()
            except EOFError:
                pass
    except OSError:
        pass


def _terminate_owned_processes() -> None:
    owned_processes = tuple(_owned_processes.items())
    for address, process in owned_processes:
        if process.is_alive():
            _terminate_at_address(address)
    for _, process in owned_processes:
        process.join()
    _owned_processes.clear()


_process_finalizer = Finalize(
    None,
    _terminate_owned_processes,
    exitpriority=10,
)


def _register_owned_process(address: Address, process: SpawnProcess) -> None:
    _owned_processes[address] = process


class ActorRef(Generic[T_co]):
    """A pickleable reference to an object living in another process.

    Actor methods are resolved dynamically. Async methods remain async callables
    and may run concurrently in the actor process. Sync methods remain
    synchronous callables and run exclusively.
    """

    def __init__(
        self,
        address: Address,
        async_methods: frozenset[str],
        sync_methods: frozenset[str],
    ) -> None:
        self._address = address
        self._async_methods = async_methods
        self._sync_methods = sync_methods

    def __getattr__(self, name: str) -> AsyncMethod | SyncMethod:
        if name in self._async_methods:

            async def async_invoke(*args: object, **kwargs: object) -> Any:
                return await asyncio.to_thread(self._call, name, args, kwargs)

            return async_invoke

        if name in self._sync_methods:

            def sync_invoke(*args: object, **kwargs: object) -> Any:
                return self._call(name, args, kwargs)

            return sync_invoke

        raise AttributeError(
            f"{type(self).__name__!s} has no exposed method {name!r}"
        )

    @override
    def __getstate__(
        self,
    ) -> tuple[Address, frozenset[str], frozenset[str]]:
        return self._address, self._async_methods, self._sync_methods

    def __setstate__(
        self, state: tuple[Address, frozenset[str], frozenset[str]]
    ) -> None:
        self._address, self._async_methods, self._sync_methods = state

    @override
    def __repr__(self) -> str:
        host, port = self._address
        return f"ActorRef(address={host}:{port})"

    def _call(
        self,
        method: str,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Any:
        try:
            request_data = _serialize(_Call(method, args, kwargs))
        except Exception as exception:
            raise ActorError("Could not serialize actor request") from exception

        try:
            with Client(self._address) as connection:
                connection.send_bytes(request_data)
                response = cast(_Response, _receive(connection))
        except (EOFError, OSError) as exception:
            raise ActorDiedError(
                f"Actor at {self._address[0]}:{self._address[1]} is not running"
            ) from exception

        if isinstance(response, _Failure):
            raise response.exception
        if not isinstance(response, _Result):
            raise ActorError("Actor returned an invalid response")
        return response.value


def _actor_process(
    class_data: bytes,
    args_data: bytes,
    kwargs_data: bytes,
    async_methods: frozenset[str],
    sync_methods: frozenset[str],
    startup: Connection,
) -> None:
    listener: Listener | None = None
    try:
        cls = cast(type[object], _deserialize(class_data))
        args = cast(tuple[object, ...], _deserialize(args_data))
        kwargs = cast(dict[str, object], _deserialize(kwargs_data))
        instance = cls(*args, **kwargs)
        listener = Listener(
            ("127.0.0.1", 0),
            family="AF_INET",
            backlog=128,
        )
        address = cast(Address, listener.address)
        _send(startup, _Result(address))
    except BaseException as exception:
        exception.add_note("The actor failed during initialization")
        _send(startup, _Failure(exception))
        return
    finally:
        startup.close()

    assert listener is not None
    asyncio.run(_serve_actor(listener, instance, async_methods, sync_methods))


async def _serve_actor(
    listener: Listener,
    instance: object,
    async_methods: frozenset[str],
    sync_methods: frozenset[str],
) -> None:
    methods = async_methods | sync_methods
    gate = _ActorGate()
    shutting_down = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()

    async def handle_connection(connection: Connection) -> None:
        response: _Response | None = None
        try:
            request = cast(_Request, await asyncio.to_thread(_receive, connection))
            if isinstance(request, _Terminate):
                shutting_down.set()
                listener.close()
                return
            if not isinstance(request, _Call) or request.method not in methods:
                raise ActorError("Received a call to an unexposed actor method")

            method = getattr(instance, request.method)
            if request.method in async_methods:
                await gate.acquire_shared()
                try:
                    value = await cast(Callable[..., Awaitable[object]], method)(
                        *request.args, **request.kwargs
                    )
                finally:
                    await gate.release_shared()
            else:
                await gate.acquire_exclusive()
                try:
                    value = await asyncio.to_thread(
                        cast(Callable[..., object], method),
                        *request.args,
                        **request.kwargs,
                    )
                finally:
                    await gate.release_exclusive()
            response = _Result(value)
        except BaseException as exception:
            exception.add_note(
                "Remote actor traceback:\n"
                + "".join(traceback.format_exception(exception))
            )
            response = _Failure(exception)
        finally:
            if response is not None:
                try:
                    await asyncio.to_thread(_send, connection, response)
                except BaseException as exception:
                    fallback = ActorError(
                        f"Could not serialize actor response: {exception}"
                    )
                    try:
                        await asyncio.to_thread(_send, connection, _Failure(fallback))
                    except BaseException:
                        pass
            await asyncio.to_thread(connection.close)

    async def accept_loop() -> None:
        while not shutting_down.is_set():
            try:
                connection = await asyncio.to_thread(listener.accept)
            except (OSError, EOFError):
                return

            if shutting_down.is_set():
                await asyncio.to_thread(connection.close)
                return

            task = asyncio.create_task(handle_connection(connection))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    accept_task = asyncio.create_task(accept_loop())
    await shutting_down.wait()
    try:
        listener.close()
    except OSError:
        pass
    await accept_task
    if tasks:
        await asyncio.wait(tasks)


async def terminate(actor_ref: ActorRef[object]) -> None:
    """Terminate an actor and all actors that it owns.

    The operation is idempotent. Calls through any reference to the terminated
    actor subsequently raise :class:`ActorDiedError`.
    """

    await asyncio.to_thread(_terminate_ref, actor_ref)


def _terminate_ref(actor_ref: ActorRef[object]) -> None:
    address = actor_ref._address
    _terminate_at_address(address)
    process = _owned_processes.pop(address, None)
    if process is not None:
        process.join()


def actor(cls: type[T]) -> ActorClass[T]:
    """Run instances of ``cls`` as actors in spawned subprocesses.

    Public instance methods are exposed. Async methods may run concurrently;
    sync methods run exclusively. Construction starts the subprocess immediately
    and returns once ``cls.__init__`` has completed.
    """

    async_methods, sync_methods = _exposed_methods(cls)
    class_data = _serialize(cls)

    def create_actor(*args: object, **kwargs: object) -> ActorRef[T]:
        context = multiprocessing.get_context("spawn")
        parent_startup, child_startup = context.Pipe(duplex=False)
        process = context.Process(
            target=_actor_process,
            args=(
                class_data,
                _serialize(args),
                _serialize(kwargs),
                async_methods,
                sync_methods,
                child_startup,
            ),
            name=f"{cls.__name__}Actor",
        )
        process.start()
        child_startup.close()

        try:
            startup_response = cast(_Response, _receive(parent_startup))
        except (EOFError, OSError) as exception:
            process.join()
            raise ActorError(f"Actor {cls.__name__} failed to start") from exception
        finally:
            parent_startup.close()

        if isinstance(startup_response, _Failure):
            process.join()
            raise startup_response.exception
        if not isinstance(startup_response, _Result):
            process.terminate()
            process.join()
            raise ActorError("Actor returned an invalid startup response")

        address = cast(Address, startup_response.value)
        _register_owned_process(address, process)
        return ActorRef(address, async_methods, sync_methods)

    create_actor.__name__ = cls.__name__
    create_actor.__qualname__ = cls.__qualname__
    create_actor.__doc__ = cls.__doc__
    create_actor.__module__ = cls.__module__
    return cast(ActorClass[T], create_actor)
