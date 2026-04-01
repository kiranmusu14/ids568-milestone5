"""End-to-end API tests for the LLM inference server."""
import pytest
from httpx import AsyncClient, ASGITransport

from src.server import app, batcher, cache


@pytest.fixture(autouse=True)
async def reset_server():
    """Start batcher and clear cache before each test."""
    await batcher.start()
    await cache.clear()
    yield
    await batcher.stop()


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_generate_returns_response():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/generate",
            json={"prompt": "What is 2+2?", "max_tokens": 64, "temperature": 0.0},
        )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["cached"] is False
    assert data["tokens_generated"] > 0


@pytest.mark.asyncio
async def test_generate_cached_on_repeat():
    payload = {"prompt": "Repeat me", "max_tokens": 64, "temperature": 0.0}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.post("/generate", json=payload)
        r2 = await client.post("/generate", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is True
    assert r1.json()["response"] == r2.json()["response"]


@pytest.mark.asyncio
async def test_stats_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/generate", json={"prompt": "Stats test"})
        response = await client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "batching" in data
    assert "caching" in data


@pytest.mark.asyncio
async def test_cache_clear_endpoint():
    payload = {"prompt": "Cache me"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/generate", json=payload)
        clear_resp = await client.post("/cache/clear")
        r2 = await client.post("/generate", json=payload)
    assert clear_resp.json()["status"] == "cache cleared"
    assert r2.json()["cached"] is False


@pytest.mark.asyncio
async def test_generate_invalid_prompt():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/generate",
            json={"prompt": "", "max_tokens": 64},
        )
    assert response.status_code == 422
