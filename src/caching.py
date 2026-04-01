"""
In-process LRU + TTL cache for LLM inference results.

Privacy design:
  - Cache keys are SHA-256 hashes of (prompt + model_name + generation_params).
  - No plaintext user identifiers, IP addresses, or session tokens are stored.
  - Only the anonymised request hash and the model response are persisted.

Thread safety:
  - All mutations are guarded by asyncio.Lock to prevent race conditions under
    concurrent async request handling.
"""
import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    value: Any
    expires_at: float  # Unix timestamp


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions_lru: int = 0
    evictions_ttl: int = 0
    current_size: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


class InferenceCache:
    """
    LRU cache with per-entry TTL expiration.

    Eviction policy:
      1. TTL expiration is checked on every get() call.
      2. LRU eviction triggers when inserting into a full cache.
    """

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(prompt: str, model_name: str, **generation_params: Any) -> str:
        """
        Build a privacy-preserving cache key.

        The key is the hex digest of SHA-256 applied to the JSON-serialised
        combination of prompt, model name, and generation parameters.
        No user identifier, IP address, or session token is included.
        """
        payload = json.dumps(
            {"prompt": prompt, "model": model_name, **generation_params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Return cached value or None on miss/expiry."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if time.time() > entry.expires_at:
                # TTL expired — remove and report miss
                del self._store[key]
                self._stats.evictions_ttl += 1
                self._stats.misses += 1
                self._stats.current_size = len(self._store)
                return None

            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any) -> None:
        """Insert or update a cache entry, evicting LRU if at capacity."""
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self.max_entries:
                    # Evict least-recently-used (front of OrderedDict)
                    self._store.popitem(last=False)
                    self._stats.evictions_lru += 1

            self._store[key] = CacheEntry(
                value=value,
                expires_at=time.time() + self.ttl_seconds,
            )
            self._stats.current_size = len(self._store)

    async def invalidate(self, key: str) -> bool:
        """Explicitly remove a single entry. Returns True if it existed."""
        async with self._lock:
            existed = key in self._store
            if existed:
                del self._store[key]
                self._stats.current_size = len(self._store)
            return existed

    async def clear(self) -> None:
        """Flush all entries (e.g., for testing or manual invalidation)."""
        async with self._lock:
            self._store.clear()
            self._stats.current_size = 0

    async def get_stats(self) -> dict:
        async with self._lock:
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": round(self._stats.hit_rate, 4),
                "total_requests": self._stats.total_requests,
                "evictions_lru": self._stats.evictions_lru,
                "evictions_ttl": self._stats.evictions_ttl,
                "current_size": self._stats.current_size,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }
