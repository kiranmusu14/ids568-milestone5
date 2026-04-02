# LLM Inference Server

High-throughput LLM inference API with dynamic batching and caching.

---

## Features

- **Dynamic batching** — groups concurrent requests into GPU batches using a hybrid size-or-timeout trigger, reducing per-request latency as batch size grows.
- **Response caching** — LRU + TTL in-memory cache eliminates redundant inference calls for repeated prompts.
- **Privacy-preserving cache keys** — cache keys are SHA-256 hashes of `(prompt + model + params)`. No user identifiers, IPs, or session tokens are ever stored.

---

## Quick Start

### Prerequisites

- Python 3.9+
- Redis (optional — default cache is in-memory)
- GPU recommended for production; simulated backend works on CPU

### Installation

```bash
# 1. Clone and enter the repo
cd ids568-milestone5

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env to match your setup
```

### Configuration via `.env`

All tunable parameters are read from environment variables with the `LLM_` prefix. Copy `.env.example` to `.env` and adjust as needed. No code changes required per environment.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `simulated-llm-7b` | Model identifier |
| `LLM_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `LLM_BATCH_TIMEOUT_MS` | `50.0` | Max wait (ms) to fill a batch |
| `LLM_CACHE_TTL_SECONDS` | `300.0` | Cache entry time-to-live |
| `LLM_CACHE_MAX_ENTRIES` | `1000` | Max LRU cache entries |
| `LLM_BASE_INFERENCE_LATENCY_MS` | `100.0` | Fixed GPU kernel cost per batch (ms) |
| `LLM_PER_REQUEST_LATENCY_MS` | `30.0` | Marginal latency added per request (ms) |
| `LLM_BATCH_AMORTIZATION_FACTOR` | `0.4` | GPU parallelism factor (0–1) |
| `LLM_HOST` | `0.0.0.0` | Bind address |
| `LLM_PORT` | `8000` | Listen port |
| `LLM_LOG_LEVEL` | `info` | Uvicorn log level |

---

## Running the Server

### Development mode (auto-reload on file changes)

```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

### Production mode

```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

> Note: use `--workers 1` — the batcher and cache are in-process singletons; multiple workers would not share state.

---

## API Usage

### `POST /generate` — submit a prompt for inference

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformer attention", "max_tokens": 128, "temperature": 0.7}'
```

### `GET /health` — liveness probe

```bash
curl http://localhost:8000/health
```

### `GET /stats` — real-time batching and caching metrics

```bash
curl http://localhost:8000/stats
```

### `POST /cache/clear` — flush all cache entries

```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## Running Benchmarks

The benchmark suite requires the server to be running:

```bash
# Terminal 1 — start the server
python -m uvicorn src.server:app --port 8000 --log-level warning

# Terminal 2 — run all benchmark scenarios
python benchmarks/run_benchmarks.py --scenario all

# Run a specific scenario
python benchmarks/run_benchmarks.py --scenario batch
python benchmarks/run_benchmarks.py --scenario cache
python benchmarks/run_benchmarks.py --scenario throughput

# Custom load test (example: 32 concurrent users, 200 total requests)
python benchmarks/load_generator.py --concurrency 32 --num-requests 200
```

Results are saved as JSON to `benchmarks/results/`.

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/
```

---

## Project Structure

```
ids568-milestone5/
├── src/
│   ├── __init__.py
│   ├── server.py          # FastAPI application entrypoint
│   ├── batching.py        # Dynamic batching logic
│   ├── caching.py         # Cache implementation (in-memory LRU+TTL)
│   ├── inference.py       # Model loading and inference
│   ├── config.py          # Configuration management
│   └── models.py          # Pydantic request/response schemas
├── benchmarks/
│   ├── run_benchmarks.py  # Benchmark orchestration script
│   ├── load_generator.py  # Synthetic load generation
│   └── results/           # Raw benchmark data
├── tests/
│   ├── test_batching.py   # Unit tests for batcher
│   ├── test_caching.py    # Unit tests for cache
│   └── test_integration.py # End-to-end API tests
├── analysis/
│   ├── performance_report.pdf  # Main analysis document
│   ├── governance_memo.pdf     # Governance considerations
│   └── visualizations/         # Charts and graphs
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment File Template

```bash
# .env.example - Copy to .env and customize

# Batching
LLM_MAX_BATCH_SIZE=8
LLM_BATCH_TIMEOUT_MS=50.0

# Caching
LLM_CACHE_TTL_SECONDS=300.0
LLM_CACHE_MAX_ENTRIES=1000

# Simulated model
LLM_MODEL_NAME=simulated-llm-7b
LLM_BASE_INFERENCE_LATENCY_MS=100.0
LLM_PER_REQUEST_LATENCY_MS=30.0
LLM_BATCH_AMORTIZATION_FACTOR=0.4

# Server
LLM_HOST=0.0.0.0
LLM_PORT=8000
LLM_LOG_LEVEL=info
```

---

## Common Pitfalls

**Hardcoded configuration.** Use environment variables for all tunable parameters. This enables different settings per environment without code changes.

**Missing `__init__.py`.** Required for Python to recognize directories as packages. Without it, imports will fail.

**No type hints.** Type annotations improve readability and enable better IDE support. They also help catch bugs before runtime.

**Undocumented dependencies.** Always keep `requirements.txt` updated with pinned versions. This ensures reproducible deployments.

---

## License

MIT
