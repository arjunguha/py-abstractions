"""Local and subprocess actors communicating through asynchronous socket I/O.

Subprocess actors use one auxiliary thread for synchronous methods. Local
actors run all methods on the main thread. Network I/O uses no worker threads.
"""

from __future__ import annotations

import asyncio
import inspect
import importlib
import multiprocessing
import pickle
import queue
import socket
import struct
import threading
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from multiprocessing.util import Finalize
from typing import Any, Generic, Protocol, TypeVar, cast

__all__ = [
    "ActorClass",
    "ActorDiedError",
    "ActorError",
    "ActorRef",
    "actor",
    "local_actor",
    "terminate",
]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
AsyncMethod = Callable[..., Coroutine[Any, Any, Any]]
Address = tuple[str, int]
_FRAME_SIZE = struct.Struct("!Q")
_owned_processes: dict[Address, SpawnProcess] = {}
_actor_context: _ActorContext | None = None
_current_actor: ContextVar[_ActorContext | None] = ContextVar("current_actor", default=None)
_actor_instances: dict[int, _ActorContext] = {}
_local_actors: dict[Address, asyncio.Task[None]] = {}


class ActorClass(Protocol[T_co]):
    """The callable produced by actor or local_actor."""

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


_Response = _Result | _Failure


@dataclass(frozen=True)
class _ExposedMethods:
    all: frozenset[str]
    asynchronous: frozenset[str]


@dataclass
class _ActorContext:
    reference: ActorRef[object]
    stopping: bool = False
    server: _ActorServer | None = None
    instance: object | None = None
    children: dict[Address, ActorRef[object]] = field(default_factory=dict)


def _as_actor(self: object) -> ActorRef[Any]:
    """Return the reference for the actor instance receiving this method."""
    context = _actor_instances.get(id(self))
    if context is None:
        # Construction runs before the instance can be registered.
        context = _current_actor.get()
        if context is not None and context.instance is not None:
            context = None
    if context is None:
        raise ActorError("as_actor() must be called inside an actor")
    return context.reference


def _serialize(value: object) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(data: bytes) -> object:
    try:
        return pickle.loads(data)
    except Exception as exception:
        raise ActorError("Could not deserialize actor message") from exception


def _serialize_response(response: _Response) -> bytes:
    try:
        return _serialize(response)
    except BaseException as exception:
        return _serialize(_Failure(ActorError(
            f"Could not serialize actor response: {exception}"
        )))


async def _read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(_FRAME_SIZE.size)
    size, = _FRAME_SIZE.unpack(header)
    return await reader.readexactly(size)


async def _write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(_FRAME_SIZE.pack(len(data)))
    writer.write(data)
    await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass


async def _exchange(address: Address, request: object) -> object:
    try:
        data = _serialize(request)
    except Exception as exception:
        raise ActorError("Could not serialize actor request") from exception
    try:
        # Actor addresses are numeric loopback addresses, so opening the
        # connection does not need an executor for DNS resolution.
        reader, writer = await asyncio.open_connection(*address)
        try:
            await _write_frame(writer, data)
            return _deserialize(await _read_frame(reader))
        finally:
            await _close_writer(writer)
    except (OSError, asyncio.IncompleteReadError) as exception:
        raise ActorDiedError(
            f"Actor at {address[0]}:{address[1]} is not running"
        ) from exception


class ActorRef(Generic[T_co]):
    """A pickleable reference whose method calls send asynchronous messages."""

    def __init__(self, address: Address, methods: frozenset[str]) -> None:
        self._address = address
        self._methods = methods

    async def terminate(self) -> None:
        """Shut down this actor; self-termination only requests shutdown."""
        await terminate(self)

    def __getattr__(self, name: str) -> AsyncMethod:
        if name not in self._methods:
            raise AttributeError(
                f"{type(self).__name__!s} has no exposed method {name!r}"
            )

        async def invoke(*args: object, **kwargs: object) -> Any:
            response = await _exchange(self._address, _Call(name, args, kwargs))
            if isinstance(response, _Failure):
                raise response.exception
            if not isinstance(response, _Result):
                raise ActorError("Actor returned an invalid response")
            return response.value

        return invoke

    # Python 3.11 has no typing.override; keep this module standard-library-only.
    def __getstate__(self) -> tuple[Address, frozenset[str]]:  # ty: ignore[missing-override-decorator]
        return self._address, self._methods

    def __setstate__(self, state: tuple[Address, frozenset[str]]) -> None:
        self._address, self._methods = state

    def __repr__(self) -> str:  # ty: ignore[missing-override-decorator]
        host, port = self._address
        return f"ActorRef(address={host}:{port})"


class _SyncWorker:
    """The actor's only auxiliary thread, used exclusively for sync methods."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._jobs: queue.SimpleQueue[
            tuple[Callable[[], object], asyncio.Future[_Response]] | None
        ] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    async def call(self, method: Callable[[], object]) -> _Response:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="actor-sync")
            self._thread.start()
        future: asyncio.Future[_Response] = self._loop.create_future()
        context = copy_context()
        self._jobs.put((lambda: context.run(method), future))
        return await future

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            method, future = job
            try:
                response: _Response = _Result(method())
            except BaseException as exception:
                response = _remote_failure(exception)
            self._loop.call_soon_threadsafe(self._deliver, future, response)

    @staticmethod
    def _deliver(future: asyncio.Future[_Response], response: _Response) -> None:
        if not future.done():
            future.set_result(response)

    async def close(self) -> None:
        if self._thread is not None:
            self._jobs.put(None)
            while self._thread.is_alive():
                await asyncio.sleep(0.001)
            self._thread.join()


def _remote_failure(exception: BaseException) -> _Failure:
    exception.add_note(
        "Remote actor traceback:\n" + "".join(traceback.format_exception(exception))
    )
    return _Failure(exception)


class _MethodLock:
    """Favor waiting synchronous methods over newly arriving async methods."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_async = 0
        self._sync_active = False
        self._waiting_sync = 0

    async def acquire_async(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._sync_active and self._waiting_sync == 0
            )
            self._active_async += 1

    async def release_async(self) -> None:
        async with self._condition:
            self._active_async -= 1
            if self._active_async == 0:
                self._condition.notify_all()

    async def acquire_sync(self) -> None:
        async with self._condition:
            self._waiting_sync += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._sync_active and self._active_async == 0
                )
                self._sync_active = True
            finally:
                self._waiting_sync -= 1
                if self._waiting_sync == 0:
                    self._condition.notify_all()

    async def release_sync(self) -> None:
        async with self._condition:
            self._sync_active = False
            self._condition.notify_all()


class _ActorServer:
    def __init__(
        self, instance: object, methods: _ExposedMethods,
        context: _ActorContext, *, local: bool = False,
    ) -> None:
        self.loop = asyncio.get_running_loop()
        self._instance = instance
        self._methods = methods
        self._context = context
        context.server = self
        self._lock = _MethodLock()
        self._worker = None if local else _SyncWorker()
        self._shutdown = asyncio.Event()
        self._stopped = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._accepting_calls = True
        self._calls: set[asyncio.Task[None]] = set()
        self._connections: set[asyncio.Task[None]] = set()
        self._reading: set[asyncio.Task[None]] = set()

    def request_stop(self) -> None:
        self._shutdown.set()

    def _connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def run(
        self, listener: socket.socket, startup: Connection | None = None,
    ) -> None:
        # Service tasks are not method calls, even if their creator was one.
        token = _current_actor.set(None)
        server: asyncio.Server | None = None
        try:
            server = await asyncio.start_server(self._connect, sock=listener)
            if self._context.stopping:
                self.request_stop()
            if startup is not None:
                startup.send_bytes(_serialize(_Result(self._context.reference._address)))
                startup.close()
            await self._shutdown.wait()
            # Keep receiving while active calls finish: those calls may need
            # more self messages to produce their results.
            while self._calls:
                await self._idle.wait()
            self._accepting_calls = False
            server.close()
            # On Python 3.12+, wait_closed() also waits for client connections.
            # Finish cleanup and release termination waiters before awaiting
            # it in finally, or their open connections would deadlock shutdown.
            await _shutdown_children(self._context)
            if self._worker is not None:
                await self._worker.close()
            self._stopped.set()
            # An idle client that never sends a request must not hold shutdown
            # open. Termination waiters, in contrast, receive a final response.
            for task in tuple(self._reading):
                task.cancel()
            while self._connections:
                await asyncio.gather(*self._connections, return_exceptions=True)
        finally:
            self._accepting_calls = False
            if server is not None:
                server.close()
            listener.close()
            # Also clean up when the local actor's event loop shuts down.
            for task in tuple(self._connections):
                task.cancel()
            if self._connections:
                await asyncio.gather(*self._connections, return_exceptions=True)
            await _shutdown_children(self._context)
            if self._worker is not None:
                await self._worker.close()
            if server is not None:
                await server.wait_closed()
            _actor_instances.pop(id(self._instance), None)
            _current_actor.reset(token)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = cast(asyncio.Task[None], asyncio.current_task())
        try:
            self._reading.add(task)
            try:
                request = _deserialize(await _read_frame(reader))
            finally:
                self._reading.discard(task)
            if isinstance(request, _Terminate):
                self.request_stop()
                await self._stopped.wait()
                response: _Response = _Result(None)
            elif not isinstance(request, _Call):
                response = _Failure(ActorError("Received an invalid actor request"))
            elif not self._accepting_calls:
                response = _Failure(ActorDiedError("Actor is not running"))
            else:
                self._calls.add(task)
                self._idle.clear()
                response = await self._invoke(request)
            await _write_frame(writer, _serialize_response(response))
        except (OSError, asyncio.IncompleteReadError):
            pass
        except ActorError as exception:
            try:
                await _write_frame(writer, _serialize_response(_Failure(exception)))
            except OSError:
                pass
        finally:
            self._calls.discard(task)
            if not self._calls:
                self._idle.set()
            await _close_writer(writer)

    async def _invoke(self, request: _Call) -> _Response:
        token = _current_actor.set(self._context)
        try:
            if request.method not in self._methods.all:
                raise ActorError("Received a call to an unexposed actor method")
            method = cast(Callable[..., object], getattr(self._instance, request.method))
            if request.method in self._methods.asynchronous:
                await self._lock.acquire_async()
                try:
                    async_method = cast(Callable[..., Awaitable[object]], method)
                    return _Result(await async_method(*request.args, **request.kwargs))
                finally:
                    await self._lock.release_async()
            await self._lock.acquire_sync()
            try:
                if self._worker is None:
                    return _Result(method(*request.args, **request.kwargs))
                return await self._worker.call(lambda: method(*request.args, **request.kwargs))
            finally:
                await self._lock.release_sync()
        except BaseException as exception:
            return _remote_failure(exception)
        finally:
            _current_actor.reset(token)


async def _join_process(process: SpawnProcess) -> None:
    # Polling process exit works on both Windows and Unix without a helper
    # thread. join() is nonblocking once the process has exited.
    while process.is_alive():
        await asyncio.sleep(0.01)
    process.join()


async def terminate(actor_ref: ActorRef[object]) -> None:
    """Terminate an actor and its children, finishing accepted calls first.

    External callers wait for shutdown. Self-termination only requests it,
    allowing the current method to finish. Repeated termination is harmless.
    """
    context = _current_actor.get() or _actor_context
    if context is not None and actor_ref._address == context.reference._address:
        context.stopping = True
        if context.server is not None:
            context.server.loop.call_soon_threadsafe(context.server.request_stop)
        return
    try:
        response = await _exchange(actor_ref._address, _Terminate())
        if not isinstance(response, _Result):
            raise ActorError("Actor returned an invalid termination response")
    except ActorDiedError:
        pass
    process = _owned_processes.get(actor_ref._address)
    if process is not None:
        await _join_process(process)
        _owned_processes.pop(actor_ref._address, None)
    local_task = _local_actors.get(actor_ref._address)
    if local_task is not None:
        # Cancelling a caller must not cancel the actor's service task.
        await asyncio.shield(local_task)


async def _shutdown_children(context: _ActorContext) -> None:
    results = await asyncio.gather(*(
        terminate(ref) for ref in tuple(context.children.values())
    ), return_exceptions=True)
    context.children.clear()
    # At loop shutdown, local children's service tasks may already be
    # cancelled. Still wait for every subprocess child to exit.
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


def _register_child(reference: ActorRef[object]) -> None:
    context = _current_actor.get() or _actor_context
    if context is not None:
        context.children[reference._address] = reference


def _construct_instance(
    cls: type[object], args: tuple[object, ...], kwargs: dict[str, object],
    context: _ActorContext,
) -> object:
    token = _current_actor.set(context)
    try:
        instance = cls(*args, **kwargs)
        context.instance = instance
        _actor_instances[id(instance)] = context
        return instance
    finally:
        _current_actor.reset(token)


def _terminate_owned_processes() -> None:
    """Blocking cleanup at process exit, when no event loop is available."""
    for address, process in tuple(_owned_processes.items()):
        if process.is_alive():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                    connection.connect(address)
                    data = _serialize(_Terminate())
                    connection.sendall(_FRAME_SIZE.pack(len(data)) + data)
                    while connection.recv(65536):
                        pass
            except OSError:
                pass
        process.join()
    _owned_processes.clear()


_process_finalizer = Finalize(None, _terminate_owned_processes, exitpriority=10)


def _exposed_methods(cls: type[object]) -> _ExposedMethods:
    exposed: set[str] = set()
    asynchronous: set[str] = set()
    for name, descriptor in inspect.getmembers_static(cls):
        if name.startswith("_") or isinstance(descriptor, (staticmethod, classmethod)):
            continue
        if inspect.isfunction(descriptor):
            exposed.add(name)
            if inspect.iscoroutinefunction(descriptor):
                asynchronous.add(name)
    return _ExposedMethods(frozenset(exposed), frozenset(asynchronous))


def _actor_process(
    class_module: str,
    class_name: str,
    args_data: bytes,
    kwargs_data: bytes,
    methods: _ExposedMethods,
    startup: Connection,
) -> None:
    global _actor_context
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            definition: object = importlib.import_module(class_module)
            for part in class_name.split("."):
                definition = getattr(definition, part)
            cls = cast(type[object], getattr(definition, "_actor_class"))
            args = cast(tuple[object, ...], _deserialize(args_data))
            kwargs = cast(dict[str, object], _deserialize(kwargs_data))
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.setblocking(False)
            address = cast(Address, listener.getsockname())
            _actor_context = _ActorContext(ActorRef(address, methods.all))
            instance = _construct_instance(cls, args, kwargs, _actor_context)

            async def serve() -> None:
                await _ActorServer(instance, methods, _actor_context).run(listener, startup)

            asyncio.run(serve())
        except BaseException as exception:
            if startup.closed:
                raise
            exception.add_note("The actor failed during initialization")
            startup.send_bytes(_serialize_response(_Failure(exception)))
        finally:
            startup.close()


def _prepare_actor(cls: type[object], *, local: bool) -> _ExposedMethods:
    methods = _exposed_methods(cls)
    if "terminate" in methods.all:
        raise ActorError("'terminate' is reserved for ActorRef.terminate()")
    if any("as_actor" in base.__dict__ for base in cls.__mro__):
        raise ActorError("'as_actor' is reserved for the injected actor reference method")
    if not local and "<locals>" in cls.__qualname__:
        raise ActorError("Actor classes must be defined at module scope, not inside functions")
    # Discover exposed methods first: as_actor is a local helper, not a message.
    setattr(cls, "as_actor", _as_actor)
    return methods


def actor(cls: type[T]) -> ActorClass[T]:
    """Run instances of cls in spawned subprocesses, exposing public methods.

    Construction returns once initialization finishes and the server starts.
    The method name 'terminate' is reserved for ActorRef's shutdown operation.
    The decorator injects as_actor(), which returns the instance's ActorRef
    and is available during construction. That name must not already exist.
    """
    methods = _prepare_actor(cls, local=False)

    def create_actor(*args: object, **kwargs: object) -> ActorRef[T]:
        try:
            args_data, kwargs_data = _serialize(args), _serialize(kwargs)
        except Exception as exception:
            raise ActorError("Could not serialize actor constructor arguments") from exception
        context = multiprocessing.get_context("spawn")
        parent_startup, child_startup = context.Pipe(duplex=False)
        process = context.Process(
            target=_actor_process,
            args=(
                cls.__module__, cls.__qualname__, args_data, kwargs_data,
                methods, child_startup,
            ),
            name=f"{cls.__name__}Actor",
        )
        try:
            process.start()
        except BaseException:
            parent_startup.close()
            raise
        finally:
            child_startup.close()
        try:
            try:
                response = _deserialize(parent_startup.recv_bytes())
            except (EOFError, OSError) as exception:
                raise ActorError(f"Actor {cls.__name__} failed to start") from exception
            if isinstance(response, _Failure):
                process.join()
                raise response.exception
            if not isinstance(response, _Result):
                raise ActorError("Actor returned an invalid startup response")
        except BaseException:
            if process.is_alive():
                process.terminate()
            process.join()
            raise
        finally:
            parent_startup.close()
        address = cast(Address, response.value)
        _owned_processes[address] = process
        reference: ActorRef[T] = ActorRef(address, methods.all)
        _register_child(reference)
        return reference

    setattr(create_actor, "_actor_class", cls)
    create_actor.__name__ = cls.__name__
    create_actor.__qualname__ = cls.__qualname__
    create_actor.__doc__ = cls.__doc__
    create_actor.__module__ = cls.__module__
    return cast(ActorClass[T], create_actor)


def local_actor(cls: type[T]) -> ActorClass[T]:
    """Run instances on the main process's running asyncio loop, in its main thread.

    Both sync and async methods run on that thread. Blocking synchronous
    methods therefore block the loop. References use the same pickleable TCP
    protocol as subprocess actors. The loop must run for messages to be served.
    """
    methods = _prepare_actor(cls, local=True)

    def create_local_actor(*args: object, **kwargs: object) -> ActorRef[T]:
        if (threading.current_thread() is not threading.main_thread()
                or multiprocessing.parent_process() is not None):
            raise ActorError("Local actors must be created in the main process's main thread")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exception:
            raise ActorError("Local actors require a running asyncio event loop") from exception

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        context: _ActorContext | None = None
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.setblocking(False)
            address = cast(Address, listener.getsockname())
            reference: ActorRef[T] = ActorRef(address, methods.all)
            context = _ActorContext(reference)
            instance = _construct_instance(cls, args, kwargs, context)
            server = _ActorServer(instance, methods, context, local=True)
            task = loop.create_task(server.run(listener))
        except BaseException:
            listener.close()
            if context is not None and context.children:
                loop.create_task(_shutdown_children(context))
            raise

        _local_actors[address] = task

        def finished(task: asyncio.Task[None]) -> None:
            _local_actors.pop(address, None)
            _actor_instances.pop(id(instance), None)
            listener.close()
            if not task.cancelled() and task.exception() is not None:
                loop.call_exception_handler({
                    "message": "Local actor server failed",
                    "exception": task.exception(),
                    "task": task,
                })

        task.add_done_callback(finished)
        _register_child(reference)
        return reference

    create_local_actor.__name__ = cls.__name__
    create_local_actor.__qualname__ = cls.__qualname__
    create_local_actor.__doc__ = cls.__doc__
    create_local_actor.__module__ = cls.__module__
    return cast(ActorClass[T], create_local_actor)
