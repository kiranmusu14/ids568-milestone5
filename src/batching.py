"""
Dynamic request batching for LLM inference.

Strategy: Hybrid (whichever fires first)
  - Size trigger:   process immediately once max_batch_size requests accumulate.
  - Timeout trigger: process whatever has accumulated after batch_timeout_ms ms,
                     even if the batch is only partially full.

This balances throughput (larger batches amortize GPU overhead) against
per-request latency (requests don't wait indefinitely for a full batch).

Concurrency safety:
  - asyncio.Queue is intrinsically safe for multiple concurrent producers.
  - A single background consumer task owns all batch state, eliminating the
    need for additional locks around batch formation logic.
  - asyncio.Future objects are used to return results to each waiting coroutine.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from src.config import ServerConfig


@dataclass
class InferenceRequest:
    prompt: str
    model_name: str
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class InferenceResult:
    text: str
    tokens_generated: int
    inference_latency_ms: float
    batch_size: int


@dataclass
class BatchStats:
    total_requests: int = 0
    total_batches: int = 0
    batch_size_histogram: dict = field(default_factory=dict)

    @property
    def avg_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_requests / self.total_batches


# ---------------------------------------------------------------------------
# Dynamic batcher
# ---------------------------------------------------------------------------

class DynamicBatcher:
    """
    Collects incoming inference requests and dispatches them in batches.

    Usage:
        batcher = DynamicBatcher(config)
        await batcher.start()          # call once, e.g. in FastAPI lifespan
        result = await batcher.submit(request)
        await batcher.stop()           # graceful shutdown
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[Tuple[InferenceRequest, asyncio.Future]] = (
            asyncio.Queue()
        )
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = BatchStats()
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._processor_task = asyncio.create_task(
            self._batch_processor(), name="batch-processor"
        )

    async def stop(self) -> None:
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: InferenceRequest) -> InferenceResult:
        """
        Enqueue a request and await its result.
        Returns once the batch containing this request has been processed.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((request, future))
        return await future

    async def get_stats(self) -> dict:
        return {
            "total_requests": self._stats.total_requests,
            "total_batches": self._stats.total_batches,
            "avg_batch_size": round(self._stats.avg_batch_size, 2),
            "batch_size_histogram": self._stats.batch_size_histogram,
            "queue_depth": self._queue.qsize(),
            "max_batch_size": self._config.max_batch_size,
            "batch_timeout_ms": self._config.batch_timeout_ms,
        }

    # ------------------------------------------------------------------
    # Internal batch processor
    # ------------------------------------------------------------------

    async def _batch_processor(self) -> None:
        """
        Single consumer loop — owns all batch formation logic.

        Algorithm:
          1. Block until the first request arrives (no CPU spin).
          2. Open a time window of batch_timeout_ms.
          3. Keep pulling requests from the queue until either:
               a. max_batch_size is reached, OR
               b. the time window expires.
          4. Dispatch the batch as a non-blocking task so the processor can
             immediately start collecting the next batch while the current
             one is being processed (pipeline parallelism).
        """
        while self._running:
            batch: List[Tuple[InferenceRequest, asyncio.Future]] = []

            # --- Step 1: block on first item ---
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return
            batch.append(item)

            # --- Step 2 & 3: fill batch within timeout window ---
            deadline = (
                asyncio.get_event_loop().time()
                + self._config.batch_timeout_ms / 1000.0
            )
            while len(batch) < self._config.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # --- Step 4: dispatch without awaiting (pipeline) ---
            asyncio.create_task(self._process_batch(batch))

    async def _process_batch(
        self, batch: List[Tuple[InferenceRequest, asyncio.Future]]
    ) -> None:
        requests = [req for req, _ in batch]
        futures = [fut for _, fut in batch]

        # Update stats
        n = len(batch)
        self._stats.total_requests += n
        self._stats.total_batches += 1
        self._stats.batch_size_histogram[str(n)] = (
            self._stats.batch_size_histogram.get(str(n), 0) + 1
        )

        try:
            from src.inference import simulate_inference
            results = await simulate_inference(requests, self._config)
            for fut, result in zip(futures, results):
                if not fut.done():
                    fut.set_result(result)
        except Exception as exc:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)
