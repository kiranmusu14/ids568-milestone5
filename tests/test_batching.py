"""Unit tests for DynamicBatcher."""
import asyncio
import pytest

from src.batching import DynamicBatcher, InferenceRequest
from src.config import ServerConfig


@pytest.fixture
def config():
    return ServerConfig(
        max_batch_size=4,
        batch_timeout_ms=50.0,
        base_inference_latency_ms=10.0,
        per_request_latency_ms=5.0,
        batch_amortization_factor=0.4,
    )


@pytest.mark.asyncio
async def test_single_request(config):
    batcher = DynamicBatcher(config)
    await batcher.start()
    req = InferenceRequest(prompt="Hello", model_name="simulated-llm-7b")
    result = await batcher.submit(req)
    await batcher.stop()
    assert result.batch_size == 1
    assert result.tokens_generated > 0
    assert "Hello" in result.text


@pytest.mark.asyncio
async def test_concurrent_requests_batched(config):
    batcher = DynamicBatcher(config)
    await batcher.start()
    requests = [
        InferenceRequest(prompt=f"Prompt {i}", model_name="simulated-llm-7b")
        for i in range(4)
    ]
    results = await asyncio.gather(*[batcher.submit(r) for r in requests])
    await batcher.stop()
    assert len(results) == 4
    batch_sizes = {r.batch_size for r in results}
    # All requests in one batch means all share the same batch_size
    assert max(batch_sizes) > 1


@pytest.mark.asyncio
async def test_stats_tracked(config):
    batcher = DynamicBatcher(config)
    await batcher.start()
    req = InferenceRequest(prompt="Test stats", model_name="simulated-llm-7b")
    await batcher.submit(req)
    stats = await batcher.get_stats()
    await batcher.stop()
    assert stats["total_requests"] == 1
    assert stats["total_batches"] == 1
    assert stats["avg_batch_size"] == 1.0


@pytest.mark.asyncio
async def test_batch_size_does_not_exceed_max(config):
    batcher = DynamicBatcher(config)
    await batcher.start()
    requests = [
        InferenceRequest(prompt=f"Prompt {i}", model_name="simulated-llm-7b")
        for i in range(8)
    ]
    results = await asyncio.gather(*[batcher.submit(r) for r in requests])
    await batcher.stop()
    for r in results:
        assert r.batch_size <= config.max_batch_size
