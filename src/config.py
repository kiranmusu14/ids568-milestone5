"""
Configuration management for the LLM inference server.
All settings can be overridden via environment variables prefixed with LLM_
(e.g., LLM_MAX_BATCH_SIZE=16) or via a .env file.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    # --------------- Batching ---------------
    max_batch_size: int = Field(
        default=8,
        description="Maximum number of requests grouped into a single batch.",
    )
    batch_timeout_ms: float = Field(
        default=50.0,
        description=(
            "Maximum time (ms) to wait for a batch to fill before processing "
            "whatever requests have accumulated (hybrid: size OR timeout first)."
        ),
    )

    # --------------- Caching ---------------
    cache_ttl_seconds: float = Field(
        default=300.0,
        description="Time-to-live for cache entries in seconds.",
    )
    cache_max_entries: int = Field(
        default=1000,
        description="Maximum number of entries; oldest (LRU) are evicted when full.",
    )

    # --------------- Simulated model ---------------
    model_name: str = Field(
        default="simulated-llm-7b",
        description="Model identifier (used in cache keys and response metadata).",
    )
    base_inference_latency_ms: float = Field(
        default=100.0,
        description=(
            "Base GPU compute time per batch (ms). Models the fixed overhead of "
            "loading a batch onto the GPU (kernel launch, attention base cost)."
        ),
    )
    per_request_latency_ms: float = Field(
        default=30.0,
        description=(
            "Marginal latency added per request within a batch (ms). "
            "Amortized by the batch_amortization_factor."
        ),
    )
    batch_amortization_factor: float = Field(
        default=0.4,
        description=(
            "GPU parallelism factor (0-1). A value of 0.4 means each additional "
            "request in a batch adds only 40%% of the per_request_latency_ms."
        ),
    )

    # --------------- Server ---------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")


# Module-level singleton used by server, batching, and caching modules.
config = ServerConfig()
