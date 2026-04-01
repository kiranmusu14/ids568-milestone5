"""
Main LLM inference server.

Endpoints:
  POST /generate  — submit an inference request (batched + cached)
  GET  /health    — liveness probe
  GET  /stats     — real-time batching and caching metrics
  POST /cache/clear — flush the cache (admin use)

Concurrency model:
  - FastAPI + uvicorn run on an asyncio event loop.
  - The DynamicBatcher runs a single background consumer task; multiple
    concurrent /generate requests safely share the asyncio.Queue.
  - The InferenceCache uses asyncio.Lock internally; no external locking needed.
"""
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.batching import DynamicBatcher, InferenceRequest
from src.caching import InferenceCache
from src.config import config


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    response: str
    cached: bool
    batch_size: int
    latency_ms: float
    tokens_generated: int
    model: str


# ---------------------------------------------------------------------------
# Application state (module-level so tests can import without side effects)
# ---------------------------------------------------------------------------

batcher: DynamicBatcher = DynamicBatcher(config)
cache: InferenceCache = InferenceCache(
    max_entries=config.cache_max_entries,
    ttl_seconds=config.cache_ttl_seconds,
)


# ---------------------------------------------------------------------------
# Lifespan: start/stop background tasks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    await batcher.start()
    yield
    await batcher.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Inference Server",
    description=(
        "Production-ready LLM inference API with dynamic request batching "
        "and privacy-preserving response caching."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Submit a prompt for inference.

    Flow:
      1. Compute privacy-preserving cache key (SHA-256 of prompt + params).
      2. Return cached response immediately on hit.
      3. On miss: enqueue to DynamicBatcher and await batch result.
      4. Store result in cache before returning.
    """
    t_start = time.perf_counter()

    cache_key = InferenceCache.make_key(
        request.prompt,
        config.model_name,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    # --- Cache lookup ---
    cached_value = await cache.get(cache_key)
    if cached_value is not None:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return GenerateResponse(
            response=cached_value["text"],
            cached=True,
            batch_size=cached_value["batch_size"],
            latency_ms=latency_ms,
            tokens_generated=cached_value["tokens_generated"],
            model=config.model_name,
        )

    # --- Batched inference ---
    inference_req = InferenceRequest(
        prompt=request.prompt,
        model_name=config.model_name,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    result = await batcher.submit(inference_req)

    # --- Store in cache ---
    await cache.set(
        cache_key,
        {
            "text": result.text,
            "batch_size": result.batch_size,
            "tokens_generated": result.tokens_generated,
        },
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return GenerateResponse(
        response=result.text,
        cached=False,
        batch_size=result.batch_size,
        latency_ms=latency_ms,
        tokens_generated=result.tokens_generated,
        model=config.model_name,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": config.model_name}


@app.get("/stats")
async def stats() -> dict:
    return {
        "batching": await batcher.get_stats(),
        "caching": await cache.get_stats(),
        "config": {
            "max_batch_size": config.max_batch_size,
            "batch_timeout_ms": config.batch_timeout_ms,
            "cache_ttl_seconds": config.cache_ttl_seconds,
            "cache_max_entries": config.cache_max_entries,
            "model_name": config.model_name,
        },
    }


@app.post("/cache/clear")
async def clear_cache() -> dict:
    await cache.clear()
    return {"status": "cache cleared"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
