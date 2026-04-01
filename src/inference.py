"""
Model loading and inference for the LLM inference server.

This module provides the simulated GPU inference backend used by the
DynamicBatcher. In a production deployment, replace _simulate_inference
with a real model call (e.g., via HuggingFace Transformers or vLLM).

Latency model (mirrors real GPU behaviour):
    total_batch_time = base_latency
                     + sum_i(per_request_latency * amortisation_factor)

  - base_latency: fixed GPU kernel launch + attention matrix setup cost
    shared across ALL requests in the batch.
  - per_request_latency * factor: marginal cost per request, reduced by
    the amortisation factor to model parallel GPU execution.
"""
import asyncio
import random
import time
from typing import List

from src.batching import InferenceRequest, InferenceResult
from src.config import ServerConfig


async def simulate_inference(
    requests: List[InferenceRequest],
    config: ServerConfig,
) -> List[InferenceResult]:
    """
    Simulate GPU-based LLM inference with realistic batching amortisation.

    Per-request latency decreases as batch size grows — the core throughput
    benefit of batching.
    """
    n = len(requests)
    base_ms = config.base_inference_latency_ms
    per_ms = config.per_request_latency_ms
    factor = config.batch_amortization_factor

    total_ms = base_ms + n * per_ms * factor
    # ±10 % jitter to model realistic variance
    jitter = random.uniform(0.90, 1.10)
    total_ms *= jitter

    t0 = time.perf_counter()
    await asyncio.sleep(total_ms / 1000.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_request_ms = elapsed_ms / n

    results = []
    for req in requests:
        tokens = max(10, int(len(req.prompt.split()) * 2.5))
        results.append(
            InferenceResult(
                text=(
                    f"[{req.model_name}] Response to: \"{req.prompt[:60]}"
                    f"{'...' if len(req.prompt) > 60 else ''}\" "
                    f"— {tokens} tokens generated."
                ),
                tokens_generated=tokens,
                inference_latency_ms=round(per_request_ms, 2),
                batch_size=n,
            )
        )
    return results
