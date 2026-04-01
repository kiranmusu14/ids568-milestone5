"""Unit tests for InferenceCache."""
import asyncio
import pytest

from src.caching import InferenceCache


@pytest.fixture
def cache():
    return InferenceCache(max_entries=3, ttl_seconds=60.0)


@pytest.mark.asyncio
async def test_cache_miss_on_empty(cache):
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get(cache):
    await cache.set("key1", {"text": "hello"})
    result = await cache.get("key1")
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_cache_hit_rate(cache):
    await cache.set("key1", "value1")
    await cache.get("key1")  # hit
    await cache.get("missing")  # miss
    stats = await cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_lru_eviction(cache):
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)
    # Access 'a' to make it recently used
    await cache.get("a")
    # Insert 'd' — should evict 'b' (LRU)
    await cache.set("d", 4)
    assert await cache.get("b") is None
    assert await cache.get("a") == 1
    assert await cache.get("d") == 4
    stats = await cache.get_stats()
    assert stats["evictions_lru"] == 1


@pytest.mark.asyncio
async def test_ttl_expiry(cache):
    short_cache = InferenceCache(max_entries=10, ttl_seconds=0.01)
    await short_cache.set("key", "val")
    await asyncio.sleep(0.05)
    result = await short_cache.get("key")
    assert result is None
    stats = await short_cache.get_stats()
    assert stats["evictions_ttl"] == 1


@pytest.mark.asyncio
async def test_clear(cache):
    await cache.set("x", 1)
    await cache.set("y", 2)
    await cache.clear()
    assert await cache.get("x") is None
    stats = await cache.get_stats()
    assert stats["current_size"] == 0


@pytest.mark.asyncio
async def test_make_key_deterministic():
    k1 = InferenceCache.make_key("hello", "model-7b", max_tokens=256, temperature=0.7)
    k2 = InferenceCache.make_key("hello", "model-7b", max_tokens=256, temperature=0.7)
    k3 = InferenceCache.make_key("world", "model-7b", max_tokens=256, temperature=0.7)
    assert k1 == k2
    assert k1 != k3
