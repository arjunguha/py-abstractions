"""
A vastly simplified actor abstraction inspired by Ray actors.

Decorating a class with ``@actor`` makes construction spawn a subprocess that
holds the instance. Construction returns an :class:`ActorRef`. Public ``async``
instance methods are exposed on the ref as ordinary asynchronous calls. Fields,
methods with leading underscores, and non-async methods are not exposed.

Actor refs are picklable, so they can be passed as arguments to (and returned
from) other actors' methods.

Inter-process communication uses :mod:`multiprocessing` manager queues.
"""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import sys
import threading
import weakref
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager
from multiprocessing.process import BaseProcess
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    TypeVar,
    cast,
    final,
)
from typing_extensions import override

T = TypeVar("T")

_CTX: BaseContext = multiprocessing.get_context("spawn")

# Sentinel placed on an actor's request queue to ask it to shut down.
_SHUTDOWN: str = "__abstractions_actor_shutdown__"

# Server manager for the process that creates actors. Child / daemon processes
# must connect() to this server rather than starting their own (daemons cannot
# spawn children).
_manager: SyncManager | None = None
_manager_lock = threading.Lock()

# Cache of connected managers keyed by address, for processes that reconnect.
_connected_managers: dict[Any, SyncManager] = {}
_connected_lock = threading.Lock()


def _start_manager() -> SyncManager:
    """Start a manager server in this process. Caller must not be daemonic."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = _CTX.Manager()
        return _manager


def _manager_for(address: Any, authkey: bytes) -> SyncManager:
    """Return a manager for *address*, connecting if it is not local."""
    global _manager
    if _manager is not None and _manager.address == address:
        return _manager
    with _connected_lock:
        existing = _connected_managers.get(address)
        if existing is not None:
            return existing
        mgr = SyncManager(address=address, authkey=authkey)
        mgr.connect()
        _connected_managers[address] = mgr
        return mgr


def _exposed_method_names(cls: type[object]) -> frozenset[str]:
    """Return names of public async instance methods on *cls*."""
    names: set[str] = set()
    for name, _member in inspect.getmembers(cls, predicate=inspect.iscoroutinefunction):
        if name.startswith("_"):
            continue
        raw: object | None = None
        for base in cls.__mro__:
            if name in base.__dict__:
                raw = base.__dict__[name]
                break
        if isinstance(raw, (staticmethod, classmethod)):
            continue
        names.add(name)
    return frozenset(names)


def _run_actor(
    cls: type[object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    request_queue: Any,
) -> None:
    """Subprocess entry point: construct the actor and serve method calls."""
    instance = cls(*args, **kwargs)
    asyncio.run(_serve(instance, request_queue))


async def _serve(
    instance: object,
    request_queue: Any,
) -> None:
    """Sequentially handle requests until a shutdown sentinel arrives."""
    while True:
        message: object = await asyncio.to_thread(request_queue.get)
        if message == _SHUTDOWN:
            return
        method_name, args, kwargs, response_queue = cast(
            tuple[str, tuple[Any, ...], dict[str, Any], Any],
            message,
        )
        try:
            method = getattr(instance, method_name)
            result: object = await method(*args, **kwargs)
            await asyncio.to_thread(response_queue.put, (True, result))
        except BaseException as exc:
            await asyncio.to_thread(response_queue.put, (False, exc))


@final
class ActorRef(Generic[T]):
    """
    Handle to an actor instance of type ``T`` running in a subprocess.

    Only public async instance methods of ``T`` are reachable. Calling such a
    method returns an awaitable that resolves to the method's result (or raises
    the remote exception).
    """

    def __init__(
        self,
        cls: type[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        _request_queue: Any | None = None,
        _process: BaseProcess | None = None,
        _owns_process: bool = True,
        _exposed: frozenset[str] | None = None,
        _manager_address: Any | None = None,
        _manager_authkey: bytes | None = None,
    ) -> None:
        self._cls_name: str = cls.__name__
        self._exposed: frozenset[str] = (
            _exposed
            if _exposed is not None
            else _exposed_method_names(cast(type[object], cls))
        )
        self._owns_process: bool = _owns_process
        self._closed: bool = False

        if _request_queue is not None:
            assert _manager_address is not None
            assert _manager_authkey is not None
            self._request_queue: Any = _request_queue
            self._process = _process
            self._manager_address = _manager_address
            self._manager_authkey = _manager_authkey
        else:
            mgr = _start_manager()
            self._manager_address = mgr.address
            self._manager_authkey = bytes(multiprocessing.current_process().authkey)
            self._request_queue = mgr.Queue()
            # BaseContext.Process is provided by concrete contexts (spawn/fork).
            process_factory = cast(
                Callable[..., BaseProcess],
                getattr(_CTX, "Process"),
            )
            self._process = process_factory(
                target=_run_actor,
                args=(cls, args, kwargs, self._request_queue),
                name=f"actor-{cls.__name__}",
                daemon=True,
            )
            self._process.start()

        self._finalizer: weakref.finalize | None
        if self._owns_process and self._process is not None:
            self._finalizer = weakref.finalize(
                self,
                ActorRef._force_kill,
                self._process,
                self._request_queue,
            )
        else:
            self._finalizer = None

    @staticmethod
    def _force_kill(process: BaseProcess, request_queue: Any) -> None:
        try:
            request_queue.put_nowait(_SHUTDOWN)
        except Exception:
            pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    @override
    def __getstate__(
        self,
    ) -> tuple[Any, frozenset[str], str, Any, bytes]:
        if self._closed:
            raise RuntimeError(f"ActorRef for {self._cls_name} is closed")
        # Copies never own the subprocess; they only share the request queue.
        return (
            self._request_queue,
            self._exposed,
            self._cls_name,
            self._manager_address,
            self._manager_authkey,
        )

    def __setstate__(
        self, state: tuple[Any, frozenset[str], str, Any, bytes]
    ) -> None:
        request_queue, exposed, cls_name, address, authkey = state
        self._request_queue = request_queue
        self._exposed = exposed
        self._cls_name = cls_name
        self._manager_address = address
        self._manager_authkey = authkey
        self._owns_process = False
        self._process = None
        self._closed = False
        self._finalizer = None

    @override
    def __repr__(self) -> str:
        ownership = "owner" if self._owns_process else "ref"
        status = "closed" if self._closed else "alive"
        return f"<ActorRef {self._cls_name} ({ownership}, {status})>"

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._exposed:
            raise AttributeError(
                f"'{self._cls_name}' actor has no exposed method '{name}'"
            )

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await self._call(name, args, kwargs)

        call.__name__ = name
        call.__qualname__ = f"ActorRef.{name}"
        return call

    async def _call(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self._closed:
            raise RuntimeError(f"ActorRef for {self._cls_name} is closed")
        mgr = _manager_for(self._manager_address, self._manager_authkey)
        response_queue: Any = mgr.Queue()
        await asyncio.to_thread(
            self._request_queue.put,
            (method_name, args, kwargs, response_queue),
        )
        ok: bool
        value: Any
        ok, value = await asyncio.to_thread(response_queue.get)
        if ok:
            return value
        raise value

    async def close(self) -> None:
        """Shut down the actor subprocess. Only the owning ref has an effect."""
        if self._closed:
            return
        self._closed = True
        if not self._owns_process:
            return
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        await asyncio.to_thread(self._request_queue.put, _SHUTDOWN)
        if self._process is not None:
            await asyncio.to_thread(self._process.join, 5.0)
            if self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 1.0)
                if self._process.is_alive():
                    self._process.kill()
                    await asyncio.to_thread(self._process.join, 1.0)

    async def __aenter__(self) -> ActorRef[T]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.close()


@final
class Actor(Generic[T]):
    """
    Callable factory produced by :func:`actor`.

    Calling an ``Actor`` spawns the actor subprocess and returns an
    :class:`ActorRef`.
    """

    def __init__(self, cls: type[T]) -> None:
        self.__actor_class__: type[T] = cls
        self.__name__: str = cls.__name__
        self.__qualname__: str = cls.__qualname__
        self.__module__: str = cls.__module__
        self.__doc__: str | None = cls.__doc__
        self._exposed: frozenset[str] = _exposed_method_names(
            cast(type[object], cls)
        )

    def __call__(self, *args: Any, **kwargs: Any) -> ActorRef[T]:
        return ActorRef(
            self.__actor_class__,
            args,
            kwargs,
            _exposed=self._exposed,
        )

    @override
    def __repr__(self) -> str:
        return f"<Actor {self.__module__}.{self.__qualname__}>"


def actor(cls: type[T]) -> Actor[T]:
    """
    Decorator that turns a class into an actor factory.

    Calling the decorated class constructs the instance in a new subprocess and
    returns an :class:`ActorRef`. No runtime initialization is required.

    Example::

        @actor
        class Counter:
            def __init__(self) -> None:
                self._value = 0

            async def increment(self) -> int:
                self._value += 1
                return self._value

            def not_exposed(self) -> int:
                return self._value

        counter = Counter()
        assert await counter.increment() == 1
        await counter.close()
    """
    # Keep the implementation class importable under a private name so that
    # multiprocessing spawn can unpickle it after the public name is replaced
    # by the Actor factory.
    original_name = cls.__name__
    original_qualname = cls.__qualname__
    impl_name = f"_ActorImpl_{original_name}"
    if "." in original_qualname:
        prefix = original_qualname.rsplit(".", 1)[0]
        impl_qualname = f"{prefix}.{impl_name}"
    else:
        impl_qualname = impl_name
    cls.__name__ = impl_name
    cls.__qualname__ = impl_qualname
    module = sys.modules[cls.__module__]
    setattr(module, impl_name, cls)

    factory = Actor(cls)
    factory.__name__ = original_name
    factory.__qualname__ = original_qualname
    return factory
