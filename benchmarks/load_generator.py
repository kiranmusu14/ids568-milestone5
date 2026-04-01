"""
Synthetic load generator for the LLM inference server.

Generates a realistic mix of unique and repeated prompts so both
batching benefits (concurrent unique requests) and caching benefits
(repeated requests return cache hits) can be measured.
"""
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


# ---------------------------------------------------------------------------
# Prompt corpus
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "What are the main advantages of {topic}?",
    "Describe the history and evolution of {topic}.",
    "How does {topic} compare to its alternatives?",
    "What are common use cases for {topic}?",
    "Summarise the key challenges in {topic}.",
    "Provide a step-by-step guide to {topic}.",
    "What recent advances have been made in {topic}?",
]

_TOPICS = [
    "machine learning", "neural networks", "transformer architecture",
    "attention mechanisms", "gradient descent", "backpropagation",
    "reinforcement learning", "natural language processing",
    "computer vision", "federated learning", "model quantisation",
    "knowledge distillation", "transfer learning", "fine-tuning",
    "prompt engineering", "in-context learning", "chain-of-thought reasoning",
    "retrieval-augmented generation", "vector databases", "embeddings",
]


def _generate_prompts(n: int, unique_ratio: float) -> List[str]:
    """
    Generate `n` prompts.  `unique_ratio` fraction are distinct; the rest
    are randomly drawn from a small pool to trigger cache hits.
    """
    unique_count = max(1, int(n * unique_ratio))
    pool_size = max(1, int(unique_count * 0.2))  # repeated prompts come from 20% pool

    unique_prompts = [
        random.choice(_PROMPT_TEMPLATES).format(topic=random.choice(_TOPICS))
        for _ in range(unique_count)
    ]
    repeated_pool = random.sample(unique_prompts, min(pool_size, len(unique_prompts)))

    prompts: List[str] = []
    for _ in range(n):
        if len(prompts) < unique_count:
            prompts.append(unique_prompts[len(prompts)])
        else:
            prompts.append(random.choice(repeated_pool))

    random.shuffle(prompts)
    return prompts


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str
    latency_ms: float
    cached: bool
    batch_size: int
    tokens_generated: int
    status_code: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.status_code == 200


@dataclass
class LoadTestSummary:
    scenario: str
    total_requests: int
    successful: int
    failed: int
    duration_seconds: float
    latencies_ms: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    batch_sizes: List[int] = field(default_factory=list)

    @property
    def throughput_rps(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.successful / self.duration_seconds

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def p99_ms(self) -> float:
        return _percentile(self.latencies_ms, 99)

    @property
    def mean_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "duration_seconds": round(self.duration_seconds, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_ms": {
                "mean": round(self.mean_ms, 2),
                "p50": round(self.p50_ms, 2),
                "p95": round(self.p95_ms, 2),
                "p99": round(self.p99_ms, 2),
                "min": round(min(self.latencies_ms, default=0), 2),
                "max": round(max(self.latencies_ms, default=0), 2),
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hit_rate, 4),
            },
            "avg_batch_size": (
                round(sum(self.batch_sizes) / len(self.batch_sizes), 2)
                if self.batch_sizes else 0
            ),
        }


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# Async load runner
# ---------------------------------------------------------------------------

class LoadGenerator:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    async def _single_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
    ) -> RequestResult:
        try:
            t0 = time.perf_counter()
            resp = await client.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, "max_tokens": 128, "temperature": 0.7},
                timeout=30.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                return RequestResult(
                    prompt=prompt,
                    latency_ms=latency_ms,
                    cached=data.get("cached", False),
                    batch_size=data.get("batch_size", 1),
                    tokens_generated=data.get("tokens_generated", 0),
                    status_code=200,
                )
            return RequestResult(
                prompt=prompt, latency_ms=latency_ms, cached=False,
                batch_size=0, tokens_generated=0, status_code=resp.status_code,
                error=f"HTTP {resp.status_code}",
            )
        except Exception as exc:
            return RequestResult(
                prompt=prompt, latency_ms=0.0, cached=False,
                batch_size=0, tokens_generated=0, status_code=0,
                error=str(exc),
            )

    async def run(
        self,
        scenario: str,
        num_requests: int,
        concurrency: int,
        unique_ratio: float = 1.0,
    ) -> LoadTestSummary:
        """
        Send `num_requests` total requests with `concurrency` in-flight at once.
        `unique_ratio` controls the fraction of unique prompts (remainder repeats).
        """
        prompts = _generate_prompts(num_requests, unique_ratio)
        semaphore = asyncio.Semaphore(concurrency)
        results: List[RequestResult] = []

        async def bounded_request(prompt: str) -> RequestResult:
            async with semaphore:
                return await self._single_request(client, prompt)

        t_start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            tasks = [bounded_request(p) for p in prompts]
            results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_start

        summary = LoadTestSummary(
            scenario=scenario,
            total_requests=num_requests,
            successful=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success),
            duration_seconds=duration,
        )
        for r in results:
            if r.success:
                summary.latencies_ms.append(r.latency_ms)
                summary.batch_sizes.append(r.batch_size)
                if r.cached:
                    summary.cache_hits += 1
                else:
                    summary.cache_misses += 1

        return summary
