"""Small, subprocess-backed actors.

Decorating a class with :func:`actor` replaces the class with a callable that
starts an instance in a spawned subprocess. Public async methods are available
on the returned :class:`ActorRef`.
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


def _serialize(value: object) -> bytes:
    return cloudpickle.dumps(value)


def _deserialize(data: bytes) -> object:
    return cloudpickle.loads(data)


def _send(connection: Connection, value: object) -> None:
    connection.send_bytes(_serialize(value))


def _receive(connection: Connection) -> object:
    return _deserialize(connection.recv_bytes())


def _exposed_methods(cls: type[object]) -> frozenset[str]:
    exposed: set[str] = set()
    for name, descriptor in inspect.getmembers_static(cls):
        if name.startswith("_"):
            continue
        if isinstance(descriptor, (staticmethod, classmethod)):
            continue
        if not inspect.iscoroutinefunction(descriptor):
            continue
        exposed.add(name)
    return frozenset(exposed)


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

    Actor methods are resolved dynamically. Every resolved method is an async
    callable and every call executes on the actor's single subprocess.
    """

    def __init__(
        self,
        address: Address,
        methods: frozenset[str],
    ) -> None:
        self._address = address
        self._methods = methods

    def __getattr__(self, name: str) -> AsyncMethod:
        if name not in self._methods:
            raise AttributeError(
                f"{type(self).__name__!s} has no exposed method {name!r}"
            )

        async def invoke(*args: object, **kwargs: object) -> Any:
            return await asyncio.to_thread(self._call, name, args, kwargs)

        return invoke

    @override
    def __getstate__(self) -> tuple[Address, frozenset[str]]:
        return self._address, self._methods

    def __setstate__(
        self, state: tuple[Address, frozenset[str]]
    ) -> None:
        self._address, self._methods = state

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
    methods: frozenset[str],
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
    asyncio.run(_serve_actor(listener, instance, methods))


async def _serve_actor(
    listener: Listener,
    instance: object,
    methods: frozenset[str],
) -> None:
    with listener:
        while True:
            try:
                connection = await asyncio.to_thread(listener.accept)
            except (OSError, EOFError):
                return

            with connection:
                try:
                    request = cast(_Request, _receive(connection))
                    if isinstance(request, _Terminate):
                        return
                    if not isinstance(request, _Call) or request.method not in methods:
                        raise ActorError("Received a call to an unexposed actor method")
                    method = cast(
                        Callable[..., Awaitable[object]],
                        getattr(instance, request.method),
                    )
                    value = await method(*request.args, **request.kwargs)
                    response: _Response = _Result(value)
                except BaseException as exception:
                    exception.add_note(
                        "Remote actor traceback:\n"
                        + "".join(traceback.format_exception(exception))
                    )
                    response = _Failure(exception)

                try:
                    _send(connection, response)
                except BaseException as exception:
                    fallback = ActorError(
                        f"Could not serialize actor response: {exception}"
                    )
                    _send(connection, _Failure(fallback))


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

    Only public async instance methods are exposed. Construction starts the
    subprocess immediately and returns once ``cls.__init__`` has completed.
    """

    methods = _exposed_methods(cls)
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
                methods,
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
        return ActorRef(address, methods)

    create_actor.__name__ = cls.__name__
    create_actor.__qualname__ = cls.__qualname__
    create_actor.__doc__ = cls.__doc__
    create_actor.__module__ = cls.__module__
    return cast(ActorClass[T], create_actor)
