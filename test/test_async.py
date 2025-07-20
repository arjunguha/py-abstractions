import pytest
import asyncio
from abstractions.async_abstractions import run_bounded

@pytest.mark.asyncio
async def test_run_bounded_with_no_concurrency():
    async def task():
        print(f"Task started")
        await asyncio.sleep(0.1)
        print(f"Task completed")

    async def gen_tasks():
        yield task()
        yield task()
        yield task()

    async for r in run_bounded(gen_tasks(), 1):
        await r


@pytest.mark.asyncio
async def test_run_bounded():
    checkpoint0 = {i: False for i in range(6)}
    checkpoint1 = {i: False for i in range(6)}

    async def task(i):
        print(f"Task {i} started")
        if i >= 3:
            assert checkpoint1[0], f"Task 0 should have completed for Task {i}"
            assert checkpoint1[1], f"Task 1 should have completed for Task {i}"
            assert checkpoint1[2], f"Task 2 should have completed for Task {i}"

        checkpoint0[i] = True
        # All three first tasks will sleep together.
        await asyncio.sleep(0.1)
        checkpoint1[i] = True
        if i < 3:
            assert checkpoint0[0]
            assert checkpoint0[1]
            assert checkpoint0[2]
            assert not checkpoint0[3]
            assert not checkpoint0[4]
            assert not checkpoint0[5]
        print(f"Task {i} completed")

    async def gen_tasks():
        for i in range(6):
            yield task(i)

    async for r in run_bounded(gen_tasks(), 3):
        await r
