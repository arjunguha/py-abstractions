# Actors

The `actor` decorator runs each decorated object in its own spawned subprocess.
Constructing the decorated class starts the subprocess immediately and returns
an `ActorRef`; there is no separate start or initialization step.

Define actor classes at module scope so the spawned subprocess can import them.
Classes defined inside functions are not supported. Guard application startup
with `if __name__ == "__main__":`, as in the example below.

The actor implementation uses only the Python standard library. Network I/O and
async methods run on the actor's event loop. Subprocess actors' synchronous methods share one
dedicated auxiliary thread, created on the first synchronous call. There are no
thread pools or threads waiting for message responses.

```python
import asyncio
import os

from abstractions.actor import actor


@actor
class Counter:
    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    async def add(self, amount: int = 1) -> int:
        self.value += amount
        return self.value

    async def process_id(self) -> int:
        return os.getpid()


async def main() -> None:
    counter = Counter(10)

    assert await counter.add() == 11
    assert await counter.add(4) == 15
    assert await counter.process_id() != os.getpid()


if __name__ == "__main__":
    asyncio.run(main())
```

## Actors on the main thread

Use `@local_actor` for objects that must stay in the main process and on its
main thread. Construct them inside a running asyncio loop on that thread.
Construction runs immediately; the server starts on the loop, and calls through
the returned `ActorRef` use the same TCP protocol as subprocess actors.

```python
from abstractions.actor import actor, local_actor, ActorRef


@local_actor
class Results:
    def __init__(self) -> None:
        self.values = []

    async def add(self, value: int) -> None:
        self.values.append(value)

    def snapshot(self) -> list[int]:
        return list(self.values)


@actor
class Worker:
    async def deliver(self, results: ActorRef, value: int) -> None:
        await results.add(value)


async def main() -> None:
    results = Results()
    worker = Worker()
    try:
        await worker.deliver(results, 42)
        assert await results.snapshot() == [42]
    finally:
        await worker.terminate()
        await results.terminate()
```

Local references are pickleable and can be passed to or returned from any
actor. Local actors can also send messages to subprocess actors and to
themselves using `self.as_actor()`. Multiple local actors keep separate state,
self references, and child ownership.

Both synchronous and asynchronous local methods run on the main thread, with
the same method-exclusion rules described below. No auxiliary thread is used.
A blocking synchronous method blocks the main event loop, including other
local actors; use async methods for operations that need to wait.

Local actor classes may be defined inside functions. Their constructor
arguments are passed directly, so they can hold resources that cannot be
pickled. Message arguments, results, and exceptions still use `pickle`, even
for calls from the same process; a message is not a direct Python method call.

Local actors require their creating loop to keep running. Terminating one
drains its calls and stops the actors it created, without stopping unrelated
actors in the same process. When the loop shuts down, remaining local actors
close their connections, cancel unfinished calls, and clean up their children.
For graceful completion, explicitly await `ref.terminate()` before leaving the
loop. Constructing a local actor in an auxiliary thread or a subprocess raises
`ActorError`.

## Exposed methods

An actor exposes only methods that:

- are instance methods declared with `def` or `async def`; and
- do not begin with an underscore.

Fields, properties, private methods, static methods, and class methods are not
available through the actor reference. Looking one up raises `AttributeError`.
The name `as_actor` is reserved for the injected local reference helper.
The name `terminate` is reserved for the reference's shutdown operation; an
actor class cannot expose an instance method with that name.
Calls through an `ActorRef` are always asynchronous, including calls backed by
a synchronous method, so every exposed method is called with `await`.

```python
@actor
class Example:
    field = "not exposed"

    async def exposed(self) -> str:
        return "yes"

    def synchronous(self) -> str:
        return "also exposed"

    async def _private(self) -> str:
        return "not exposed"
```

Calls to async methods on one actor can run concurrently. A synchronous method
executes exclusively: it waits for running async methods to finish and prevents
other sync or async methods from running until it returns. A subprocess actor continues
to receive calls while a synchronous method holds the lock, so submitting an
async call does not block the caller's event loop.
Scheduling favors synchronous calls: once one is waiting, new async calls wait
behind it, even while earlier async calls are still running.

```python
counter = Counter()

first_result, second_result = await asyncio.gather(
    counter.add(2),
    counter.add(3),
)
```

## Sending messages to yourself

`@actor` injects an `as_actor()` method into the decorated class.
`self.as_actor()` returns the actor's `ActorRef`, including in constructors and
synchronous methods. The reference can be stored or passed to other actors just
like any other reference. `as_actor` is reserved: the decorator rejects classes
that already define or inherit that name. This helper is local to the instance;
it is not exposed as a message on `ActorRef`.

```python
from abstractions.actor import actor


@actor
class Example:
    async def echo(self, value: str) -> str:
        return value

    async def round_trip(self) -> str:
        return await self.as_actor().echo("hello")

    async def stop(self) -> str:
        await self.as_actor().terminate()
        return "goodbye"
```

For static type checkers that do not infer injected methods, you can declare
`as_actor: Callable[[], ActorRef[Any]]` in the class body, importing `Callable`
and `Any` from `typing` and `ActorRef` from `abstractions.actor`. This annotation
does not define or replace the injected method.

`await self.as_actor().echo(...)` sends a serialized message through the actor's normal
connection and dispatch path. `await self.echo(...)` is an ordinary Python
method call. Self messages follow the same concurrency rules as other messages:
async methods may overlap, while synchronous methods require exclusive access.
Do not wait for a self message while holding exclusive access, or await a
synchronous self message from an active async method: the message would wait
for the current method to finish. Constructors may obtain the reference but
must finish before messages can execute.

Awaiting an **async** self message can also deadlock if a synchronous call is
queued. For example, `work()` is running, `save()` (synchronous) queues, and
`work()` then awaits an async self message to `echo()`. Now `work()` waits for
`echo()`, `echo()` waits behind `save()`, and `save()` waits for `work()` to
finish. This is a limitation of favoring synchronous calls, so avoid awaiting
self messages when synchronous calls can be queued on the same actor. If a
message is not needed, use a normal Python method call such as
`await self.echo(...)` instead.

Self-termination is special: `await self.as_actor().terminate()` requests
shutdown and returns without waiting for the current method to finish. The
actor drains accepted calls and shuts down its children before exiting. Finish
the current method after requesting shutdown. While calls remain active, the
actor continues receiving messages, including self messages needed by those
calls. Once no calls remain, it closes its listener. Additional traffic during
this drain period can delay shutdown. `await terminate(self.as_actor())` has the
same behavior.

## Passing actor references

`ActorRef` objects are pickleable and can be arguments or return values of actor
methods. This allows one actor to call another directly.

```python
from typing import Any, cast

from abstractions.actor import ActorRef, actor


@actor
class Forwarder:
    async def add_via(
        self,
        target: ActorRef[Any],
        amount: int,
    ) -> int:
        result = await target.add(amount)
        return cast(int, result)


async def main() -> None:
    counter = Counter(5)
    forwarder = Forwarder()

    assert await forwarder.add_via(counter, 7) == 12
```

The caller must be able to reach the process that owns the target actor. Actor
references contain a loopback TCP endpoint, not the actor
object itself. The process that creates an actor owns its subprocess and shuts
it down when that creating process exits; deleting or transferring an
individual reference does not stop the actor.

## Creating actors from actors

Actors can construct and return other actors. The newly created actor is owned
by the actor process that created it.

```python
@actor
class Spawner:
    async def make_counter(self, initial: int) -> ActorRef[Any]:
        return Counter(initial)


async def main() -> None:
    spawner = Spawner()
    counter = await spawner.make_counter(10)

    assert await counter.add(5) == 15
```

## Lifetime and termination

An actor remains alive until either:

- `await terminate(actor_ref)` explicitly terminates it; or
- the process that created it exits.

Termination is graceful: the actor finishes its accepted method calls, closes
its listener, and terminates actors that it created. Ownership is recursive,
so terminating the `Spawner` above also terminates its counter. `terminate` is
idempotent and accepts any reference to the actor. `await actor_ref.terminate()`
is equivalent to `await terminate(actor_ref)`. External callers wait for shutdown;
self-termination only requests shutdown, allowing the current method to return.

```python
from abstractions.actor import ActorDiedError, terminate


await terminate(counter)

try:
    await counter.add()
except ActorDiedError:
    print("the actor is no longer running")
```

Deleting an `ActorRef` does not terminate its actor. This is necessary because
other processes may still hold transferred copies of that reference.

## Errors and serialization

Constructor errors are raised immediately when the decorated class is called.
Exceptions from actor methods are raised when the method call is awaited and
include the remote traceback as an exception note.

A method call to an explicitly terminated, crashed, or otherwise unreachable
actor raises `ActorDiedError`, which is a subclass of `ActorError`.

Arguments, return values, exceptions, and actor references are serialized with
the standard-library `pickle` module. Values must be pickleable, and any custom
types must be importable in the receiving process. Lambdas and function-local
definitions are not supported. A value that cannot be serialized or
deserialized causes an `ActorError`.

Actor classes are located by their module and qualified name in the subprocess;
their code is not serialized. Define them with `@actor` in an importable module,
outside the application's main guard. Changes made only to class attributes in
the creating process are not copied into the subprocess; pass per-instance
state as constructor arguments instead.

::: abstractions.actor
