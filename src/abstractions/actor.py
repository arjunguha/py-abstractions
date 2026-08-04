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
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from multiprocessing.context import SpawnProcess
from typing import Any, Generic, Protocol, TypeVar, cast

import cloudpickle


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
AsyncMethod = Callable[..., Awaitable[Any]]
Address = tuple[str, int]


class ActorClass(Protocol[T_co]):
    """The callable produced by :func:`actor`."""

    __name__: str
    __qualname__: str
    __doc__: str | None
    __module__: str

    def __call__(self, *args: object, **kwargs: object) -> ActorRef[T_co]: ...


class ActorError(RuntimeError):
    """Raised when an actor cannot start or communicate."""


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

        function: object
        if isinstance(descriptor, (staticmethod, classmethod)):
            function = descriptor.__func__
        else:
            function = descriptor

        if not inspect.iscoroutinefunction(function):
            continue
        exposed.add(name)
    return frozenset(exposed)


def _terminate(process: SpawnProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join()


class ActorRef(Generic[T_co]):
    """A pickleable reference to an object living in another process.

    Actor methods are resolved dynamically. Every resolved method is an async
    callable and every call executes on the actor's single subprocess.
    """

    def __init__(
        self,
        address: Address,
        authkey: bytes,
        methods: frozenset[str],
        process: SpawnProcess | None = None,
    ) -> None:
        self._address = address
        self._authkey = authkey
        self._methods = methods
        self._finalizer: weakref.finalize | None = None
        if process is not None:
            self._finalizer = weakref.finalize(self, _terminate, process)

    def __getattr__(self, name: str) -> AsyncMethod:
        if name not in self._methods:
            raise AttributeError(
                f"{type(self).__name__!s} has no exposed method {name!r}"
            )

        async def invoke(*args: object, **kwargs: object) -> Any:
            return await asyncio.to_thread(self._call, name, args, kwargs)

        return invoke

    def __getstate__(self) -> tuple[Address, bytes, frozenset[str]]:
        return self._address, self._authkey, self._methods

    def __setstate__(
        self, state: tuple[Address, bytes, frozenset[str]]
    ) -> None:
        self._address, self._authkey, self._methods = state
        self._finalizer = None

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
            with Client(self._address, authkey=self._authkey) as connection:
                _send(connection, _Call(method, args, kwargs))
                response = cast(_Response, _receive(connection))
        except (EOFError, OSError) as exception:
            raise ActorError(
                f"Could not call actor at {self._address[0]}:{self._address[1]}"
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
    authkey: bytes,
    startup: Connection,
) -> None:
    listener: Listener | None = None
    try:
        cls = cast(type[object], _deserialize(class_data))
        args = cast(tuple[object, ...], _deserialize(args_data))
        kwargs = cast(dict[str, object], _deserialize(kwargs_data))
        instance = cls(*args, **kwargs)
        listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
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
                    request = cast(_Call, _receive(connection))
                    if not isinstance(request, _Call) or request.method not in methods:
                        raise ActorError("Received a call to an unexposed actor method")
                    method = cast(Callable[..., Awaitable[object]], getattr(instance, request.method))
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


def actor(cls: type[T]) -> ActorClass[T]:
    """Run instances of ``cls`` as actors in spawned subprocesses.

    Only public async methods are exposed. Construction starts the
    subprocess immediately and returns once ``cls.__init__`` has completed.
    """

    methods = _exposed_methods(cls)
    class_data = _serialize(cls)

    def create_actor(*args: object, **kwargs: object) -> ActorRef[T]:
        context = multiprocessing.get_context("spawn")
        parent_startup, child_startup = context.Pipe(duplex=False)
        authkey = bytes(multiprocessing.current_process().authkey)
        process = cast(
            SpawnProcess,
            context.Process(
            target=_actor_process,
            args=(
                class_data,
                _serialize(args),
                _serialize(kwargs),
                methods,
                authkey,
                child_startup,
            ),
            name=f"{cls.__name__}Actor",
            ),
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
            _terminate(process)
            raise ActorError("Actor returned an invalid startup response")

        return ActorRef(cast(Address, startup_response.value), authkey, methods, process)

    create_actor.__name__ = cls.__name__
    create_actor.__qualname__ = cls.__qualname__
    create_actor.__doc__ = cls.__doc__
    create_actor.__module__ = cls.__module__
    return cast(ActorClass[T], create_actor)
