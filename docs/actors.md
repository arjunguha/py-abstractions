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

    async def process_id(self) -> int:
        return os.getpid()


async def main() -> None:
    counter = Counter(10)

    assert await counter.add() == 11
    assert await counter.add(4) == 15
    assert await counter.process_id() != os.getpid()


asyncio.run(main())
```

## Exposed methods

An actor exposes only methods that:

- are declared with `async def`; and
- do not begin with an underscore.

Fields, properties, synchronous methods, and private methods are not available
through the actor reference. Looking one up raises `AttributeError`.

```python
@actor
class Example:
    field = "not exposed"

    async def exposed(self) -> str:
        return "yes"

    def synchronous(self) -> str:
        return "not exposed"

    async def _private(self) -> str:
        return "not exposed"
```

Calls to one actor execute serially in its subprocess. Calling methods on
different actors can run concurrently:

```python
first = Counter()
second = Counter()

first_result, second_result = await asyncio.gather(
    first.add(2),
    second.add(3),
)
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
object itself.

## Errors and serialization

Constructor errors are raised immediately when the decorated class is called.
Exceptions from actor methods are raised when the method call is awaited and
include the remote traceback as an exception note.

Arguments, return values, actor classes, and actor references are serialized
with `cloudpickle`. A value that cannot be serialized causes an `ActorError`.

::: abstractions.actor
