# Actors

The `actor` decorator runs each decorated object in its own spawned subprocess.
Constructing the decorated class starts the subprocess immediately and returns
an `ActorRef`; there is no separate start or initialization step.

```python
import asyncio
import os

from abstractions import actor


@actor
class Counter:
    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    async def add(self, amount: int = 1) -> int:
        self.value += amount
        return self.value

    def value_now(self) -> int:
        return self.value

    async def process_id(self) -> int:
        return os.getpid()


async def main() -> None:
    counter = Counter(10)

    assert await counter.add() == 11
    assert counter.value_now() == 11
    assert await counter.add(4) == 15
    assert await counter.process_id() != os.getpid()


if __name__ == "__main__":
    asyncio.run(main())
```

## Exposed methods

An actor exposes only methods that:

- are public instance methods (declared with `def` or `async def`); and
- do not begin with an underscore.

Fields, properties, and private methods are not available through the actor
reference. Static methods and class methods are also not exposed. Looking one up
raises `AttributeError`.

Async methods on the actor remain async on the reference. Sync methods remain
sync and block the caller until the remote call completes.

```python
@actor
class Example:
    field = "not exposed"

    async def exposed_async(self) -> str:
        return "yes"

    def exposed_sync(self) -> str:
        return "yes"

    async def _private(self) -> str:
        return "not exposed"
```

## Concurrency

Async methods on one actor may run concurrently with each other. The actor
process accepts calls continuously and runs each async method in its own task,
so overlapping `await`s on the same actor can execute at the same time:

```python
first = Counter()
second = Counter()

# Concurrent calls to different actors, and concurrent async calls to one actor,
# are both allowed:
await asyncio.gather(first.add(2), first.add(3), second.add(4))
```

Because async methods can overlap, they must tolerate concurrent access to the
actor's state (or leave mutation to sync methods).

Sync methods take an exclusive lock: while a sync method runs, no other sync or
async method runs on that actor. In-flight async methods finish before the sync
method starts, and new async methods wait for the lock without blocking the
actor from accepting further calls. Submitting an async call while the actor is
locked therefore does not block the caller from making progress on other work;
only awaiting that call waits for the sync method to finish.

```python
@actor
class Mixed:
    def __init__(self) -> None:
        self.value = 0

    def replace(self, value: int) -> int:
        self.value = value
        return self.value

    async def read(self) -> int:
        return self.value
```

## Passing actor references

`ActorRef` objects are pickleable and can be arguments or return values of actor
methods. This allows one actor to call another directly.

```python
from typing import Any, cast

from abstractions import ActorRef, actor


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
references contain a local multiprocessing connection endpoint, not the actor
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

Termination is graceful: the actor finishes in-flight method calls, closes its
listener, and terminates actors that it created. Ownership is recursive, so
terminating the `Spawner` above also terminates its counter. `terminate` is
idempotent and accepts any reference to the actor.

```python
from abstractions import ActorDiedError, terminate


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
Exceptions from actor methods are raised when the method call returns (for sync
methods) or when it is awaited (for async methods) and include the remote
traceback as an exception note.

A method call to an explicitly terminated, crashed, or otherwise unreachable
actor raises `ActorDiedError`, which is a subclass of `ActorError`.

Arguments, return values, actor classes, and actor references are serialized
with `cloudpickle`. A value that cannot be serialized causes an `ActorError`.

::: abstractions.actor
