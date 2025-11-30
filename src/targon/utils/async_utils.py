import asyncio
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

T = TypeVar("T")
V = TypeVar("V")

class _ErrorWrapper:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


_STOP_INPUT = object()
_STOP_RESULT = object()

async def async_map(
    input_gen: AsyncGenerator[T, None],
    mapper: Callable[[T], Awaitable[V]],
    concurrency: int,
    queue_size: int | None = None,
) -> AsyncGenerator[V, None]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    queue_size = queue_size or concurrency * 2
    work_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_size)
    result_queue: asyncio.Queue[object] = asyncio.Queue()

    async def producer() -> None:
        try:
            async for item in input_gen:
                await work_queue.put(item)
        finally:
            for _ in range(concurrency):
                await work_queue.put(_STOP_INPUT)

    async def worker() -> None:
        while True:
            item = await work_queue.get()
            if item is _STOP_INPUT:
                await result_queue.put(_STOP_RESULT)
                return
            try:
                result = await mapper(item)  
            except Exception as exc:
                await result_queue.put(_ErrorWrapper(exc))
            else:
                await result_queue.put(result)

    producer_task = asyncio.create_task(producer())
    worker_tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    finished_workers = 0

    try:
        while finished_workers < concurrency:
            result = await result_queue.get()
            if result is _STOP_RESULT:
                finished_workers += 1
                continue
            if isinstance(result, _ErrorWrapper):
                raise result.error
            yield result
        await producer_task
        await asyncio.gather(*worker_tasks)
    finally:
        if not producer_task.done():
            producer_task.cancel()
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)

