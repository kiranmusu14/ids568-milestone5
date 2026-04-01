# LLM Inference Server — Milestone 5

Production-ready LLM inference API with **dynamic request batching** and **privacy-preserving response caching**.

---

## Quick Start (under 5 minutes)

```bash
# 1. Clone and enter the repo
cd ids568-milestone5

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the inference server (defaults: port 8000, batch_size=8, TTL=300s)
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

# 5. Test the server
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformer attention mechanisms", "max_tokens": 128}'
```

---

## Repository Structure

```
ids568-milestone5/
├── src/
│   ├── server.py          # FastAPI inference server (batching + caching)
│   ├── batching.py        # Dynamic request batcher (hybrid size+timeout)
│   ├── caching.py         # In-process LRU+TTL cache (SHA-256 keys, no PII)
│   └── config.py          # Pydantic configuration (env-var overridable)
├── benchmarks/
│   ├── run_benchmarks.py  # Benchmark orchestration (--help supported)
│   ├── load_generator.py  # Async load generator (configurable concurrency)
│   └── results/           # Raw JSON benchmark data
├── analysis/
│   ├── generate_reports.py     # Generates charts + PDFs from results
│   ├── performance_report.pdf  # 4-page performance analysis
│   ├── governance_memo.pdf     # 1-page governance considerations
│   └── visualizations/         # PNG charts (6 figures)
├── requirements.txt
└── README.md
```

---

## Configuration

All parameters can be set via environment variables (prefix `LLM_`) or a `.env` file.

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `LLM_BATCH_TIMEOUT_MS` | `50.0` | Max wait (ms) to fill a batch |
| `LLM_CACHE_TTL_SECONDS` | `300.0` | Cache entry time-to-live |
| `LLM_CACHE_MAX_ENTRIES` | `1000` | Max LRU cache entries |
| `LLM_MODEL_NAME` | `simulated-llm-7b` | Model identifier |
| `LLM_HOST` | `0.0.0.0` | Bind address |
| `LLM_PORT` | `8000` | Listen port |

**Example — tune for low-latency workload:**
```bash
LLM_MAX_BATCH_SIZE=4 LLM_BATCH_TIMEOUT_MS=25 python -m uvicorn src.server:app
```

**Example — tune for high-throughput workload:**
```bash
LLM_MAX_BATCH_SIZE=16 LLM_BATCH_TIMEOUT_MS=100 python -m uvicorn src.server:app
```

---

## API Endpoints

### `POST /generate`
Submit a prompt for inference (batched + cached).

**Request:**
```json
{
  "prompt": "Explain transformer attention mechanisms",
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "[simulated-llm-7b] Response to: ...",
  "cached": false,
  "batch_size": 4,
  "latency_ms": 37.2,
  "tokens_generated": 128,
  "model": "simulated-llm-7b"
}
```

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

### `GET /stats`
Real-time batching and caching metrics:
```json
{
  "batching": {"total_requests": 500, "avg_batch_size": 5.2, ...},
  "caching":  {"hits": 312, "misses": 188, "hit_rate": 0.624, ...}
}
```

### `POST /cache/clear`
Flush all cache entries (admin use / GDPR erasure).

---

## Running Benchmarks

The benchmark suite requires the server to be running:

```bash
# Terminal 1: start server
source venv/bin/activate
python -m uvicorn src.server:app --port 8000 --log-level warning

# Terminal 2: run all scenarios
source venv/bin/activate
python benchmarks/run_benchmarks.py --scenario all

# Run a specific scenario
python benchmarks/run_benchmarks.py --scenario cache
python benchmarks/run_benchmarks.py --scenario throughput --server-url http://localhost:8000
```

**Scenarios:**

| Scenario | Description |
|---|---|
| `single` | Baseline sequential latency (no batching) |
| `batch` | Batching amortisation — varied concurrency levels |
| `cache` | Cold vs. warm cache latency comparison |
| `throughput` | Req/s at low (c=4), medium (c=16), high (c=32) load |
| `all` | Run all scenarios (default) |

Results are saved as JSON to `benchmarks/results/`.

---

## Generating Analysis Reports

After running benchmarks:

```bash
python analysis/generate_reports.py
```

This produces:
- `analysis/visualizations/fig1_*.png` through `fig6_*.png`
- `analysis/performance_report.pdf` (4-page analysis)
- `analysis/governance_memo.pdf` (1-page governance memo)

---

## Design Decisions

### Batching Strategy: Hybrid (Size OR Timeout)
Requests are dispatched when **either** `max_batch_size` requests accumulate **or** `batch_timeout_ms` elapses — whichever fires first. This balances:
- **Throughput**: larger batches amortise the GPU base cost across more requests.
- **Latency**: no request waits longer than `batch_timeout_ms` for a batch to fill.

### Cache Key Design (Privacy-Preserving)
Cache keys are `SHA-256(prompt + model_name + generation_params)`. No user identifier, IP address, or session token is ever stored. See `src/caching.py:InferenceCache.make_key`.

### Concurrency Safety
- The `DynamicBatcher` uses `asyncio.Queue` (producer-safe for multiple concurrent coroutines) and a single background consumer task that owns all batch-formation state — no additional locks needed for batching logic.
- The `InferenceCache` guards all mutations with `asyncio.Lock` to prevent race conditions on concurrent cache reads/writes.

### Simulated Inference Backend
The server uses a simulated inference model (`asyncio.sleep` with realistic latency math) so benchmarks run without GPU hardware. The latency model accurately represents GPU batching amortisation:

```
total_batch_time = base_latency + N × per_request_latency × amortization_factor
per_request_time = total_batch_time / N
```

At `N=1`: ~112 ms/request. At `N=8`: ~25 ms/request (4.5× improvement).

---

## Automated Sanity Checks

```bash
# File existence
test -f src/server.py && echo "OK" || echo "MISSING"
test -f src/batching.py && echo "OK" || echo "MISSING"
test -f src/caching.py && echo "OK" || echo "MISSING"
test -f src/config.py && echo "OK" || echo "MISSING"

# Python syntax
python -m py_compile src/server.py && echo "syntax OK"
python -m py_compile src/batching.py && echo "syntax OK"
python -m py_compile src/caching.py && echo "syntax OK"
python -m py_compile src/config.py && echo "syntax OK"

# Server import
python -c "from src.server import app; print('imports OK')"

# Benchmark --help
python benchmarks/run_benchmarks.py --help

# No PII in caching module
grep -n "user_id\|user_name\|email\|username" src/caching.py || echo "No PII found"

# Asyncio patterns present
grep -n "asyncio.Lock\|async def\|await" src/server.py src/batching.py
```
