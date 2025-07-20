"""
This module contains abstractions for async programming.
"""

import asyncio
from typing import Awaitable, AsyncIterator, TypeVar

# cspell:ignore sema

T = TypeVar("T")


async def run_bounded(
    tasks: AsyncIterator[Awaitable[T]], limit: int
) -> AsyncIterator[Awaitable[T]]:
    """
    There are many situations where we want to run lots of tasks with a bounded
    amount of concurrency. For example, we may want to make thousands of
    requests to an API, but not not run more than a few requests concurrently.
    We may also want to run thousands of containerized jobs on a machine that
    has multiple cores, but not run too many containers concurrently.

    The simplest way to achieve this is with a semaphore:

    ```python
    sema = asyncio.Semaphore(limit)

    async def the_task(t):
        async with semaphore:
            # Do the task.

    asyncio.gather(*(the_task(t) for t in tasks))
    ```

    However, this approach does not address another problem that often occurs
    in these situations, which is that preparing tasks can take a lot of time
    and memory. For example, if each task is sourced from a file, and the
    file has thousands of lines, we are forced to read the entire file into
    memory before we can start running tasks.

    The `run_bounded` function addresses this problem. You can use it to
    write code such as the following:

    ```python
    async def task(line):
        # Do the task.

    async def task_gen():
        with open("tasks.txt") as f:
            for line in f:
                yield task(line)

    async for result in run_bounded(task_gen(), limit=10):
        print(result)
    ```

    The code above will run 10 tasks concurrently and also stream the file
    line-by-line. The results will be **out of order**.
    """

    # A unique object that we put on a queue. When a task receives this, it
    # will shut down and stop receiving more items. In Python,
    # ({ } is { }) == False.
    complete_sentinel = {}

    task_q = asyncio.Queue(limit)
    result_q = asyncio.Queue()
    # Why do we need a semaphore, when the .put and .get methods on the queue
    # already block? We want to not pull the next task from tasks until we have
    # a consumer ready to run it immediately.
    #
    # This semaphore tracks the number of consumers waiting to get tasks from
    # the queue. The producer acquires (decrement) the semaphore immediately
    # before fetching a new task from the tasks generator. Each consumer
    # releases (increments) the semaphore immediately before it waits to .get()
    # a task from the queue. We initialize it to zero because at this point, no
    # consumers have been created.
    consumer_sema = asyncio.Semaphore(0)

    async def producer():
        # Fetch tasks and put them on the queue. Blocks when the queue hits
        # limit, which limits the rate at which we fetch tasks, and also ensures
        # that we do not fetch all tasks at once.
        await consumer_sema.acquire()
        async for coroutine in tasks:
            await task_q.put(coroutine)
            await consumer_sema.acquire()  # For the next iteration

        # Each consumer will receive a sentinel and shut down.
        for _ in range(limit):
            await task_q.put(complete_sentinel)

    async def consumer():
        # We create _limit_ consumers, which limits how many tasks run
        # concurrently.
        while True:
            consumer_sema.release()
            coroutine = await task_q.get()
            if coroutine is complete_sentinel:
                break
            fut = asyncio.Future()
            try:
                fut.set_result(await coroutine)
            except Exception as exn:
                fut.set_exception(exn)
            await result_q.put(fut)

    async def gen_results():
        while True:
            r = await result_q.get()
            if r is complete_sentinel:
                break
            yield r

    # Main body
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        workers = [tg.create_task(consumer()) for _ in range(limit)]

        # Send a sentinel to shut down gen_results after all workers finish.
        async def send_complete():
            await asyncio.gather(*workers)
            await result_q.put(complete_sentinel)

        tg.create_task(send_complete())

        async for item in gen_results():
            yield item
