"""
Benchmark orchestration script.

Scenarios
---------
single      Single-request latency (no batching, no caching)
batch       Batching benefit: varied concurrency levels
cache       Caching benefit: cold vs. warm cache comparison
throughput  Throughput (req/s) at low / medium / high load
all         Run all scenarios (default)

Usage
-----
    python benchmarks/run_benchmarks.py --help
    python benchmarks/run_benchmarks.py --scenario all
    python benchmarks/run_benchmarks.py --scenario cache --server-url http://localhost:8000
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Allow running from repo root: `python benchmarks/run_benchmarks.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.load_generator import LoadGenerator

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{url}/health", timeout=3.0)
                if resp.status_code == 200:
                    print(f"  Server ready at {url}")
                    return
        except Exception:
            pass
        await asyncio.sleep(1.0)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout}s")


async def _clear_cache(url: str) -> None:
    async with httpx.AsyncClient() as client:
        await client.post(f"{url}/cache/clear", timeout=5.0)


def _save(filename: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / filename
    out.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {out}")
    return out


def _print_summary(label: str, summary) -> None:
    d = summary.to_dict()
    lat = d["latency_ms"]
    print(
        f"    {label:30s} | "
        f"rps={d['throughput_rps']:6.1f} | "
        f"p50={lat['p50']:7.1f}ms | "
        f"p95={lat['p95']:7.1f}ms | "
        f"hit_rate={d['cache']['hit_rate']:.0%} | "
        f"ok={d['successful']}/{d['total_requests']}"
    )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

async def run_single(gen: LoadGenerator, url: str) -> dict:
    """Baseline: sequential single requests with no batching benefit."""
    print("\n[Scenario: single] Sequential single-request latency...")
    await _clear_cache(url)

    results = []
    # 20 sequential requests (concurrency=1 → no batching)
    for i in range(20):
        summary = await gen.run(
            scenario=f"single_{i}",
            num_requests=1,
            concurrency=1,
            unique_ratio=1.0,
        )
        results.append(summary.to_dict())

    # Aggregate
    all_lat = [r["latency_ms"]["mean"] for r in results]
    agg = {
        "scenario": "single",
        "description": "Sequential single requests — baseline, no batching amortisation",
        "num_samples": len(results),
        "latency_ms": {
            "mean": round(sum(all_lat) / len(all_lat), 2),
            "min": round(min(all_lat), 2),
            "max": round(max(all_lat), 2),
        },
        "individual_runs": results,
    }
    _print_summary("single (mean across 20)", type("S", (), {"to_dict": lambda s: {
        "throughput_rps": 1 / (agg["latency_ms"]["mean"] / 1000 + 1e-9),
        "latency_ms": agg["latency_ms"] | {"p50": agg["latency_ms"]["mean"], "p95": agg["latency_ms"]["max"]},
        "cache": {"hit_rate": 0},
        "successful": 20,
        "total_requests": 20,
    }})())
    return agg


async def run_batch(gen: LoadGenerator, url: str) -> dict:
    """Batching benefit: vary concurrency to form different batch sizes."""
    print("\n[Scenario: batch] Measuring batching amortisation...")
    await _clear_cache(url)

    concurrency_levels = [1, 2, 4, 8, 16]
    results = []
    for c in concurrency_levels:
        await _clear_cache(url)
        summary = await gen.run(
            scenario=f"batch_concurrency_{c}",
            num_requests=40,
            concurrency=c,
            unique_ratio=1.0,
        )
        d = summary.to_dict()
        d["concurrency"] = c
        results.append(d)
        _print_summary(f"concurrency={c}", summary)

    return {
        "scenario": "batch",
        "description": "Batching benefit: per-request latency vs. concurrency level",
        "results": results,
    }


async def run_cache(gen: LoadGenerator, url: str) -> dict:
    """Cold-cache vs. warm-cache performance comparison."""
    print("\n[Scenario: cache] Cold-cache vs. warm-cache comparison...")

    # Cold cache: all unique prompts, cache is empty
    await _clear_cache(url)
    cold = await gen.run(
        scenario="cache_cold",
        num_requests=50,
        concurrency=8,
        unique_ratio=1.0,
    )
    _print_summary("cold cache (50 unique)", cold)

    # Warm cache: same prompts repeated → all should be cache hits
    warm = await gen.run(
        scenario="cache_warm",
        num_requests=50,
        concurrency=8,
        unique_ratio=0.0,   # all repeat from prior run pool
    )
    _print_summary("warm cache (50 repeated)", warm)

    # Mixed: realistic 30% unique / 70% repeat
    mixed = await gen.run(
        scenario="cache_mixed",
        num_requests=100,
        concurrency=8,
        unique_ratio=0.3,
    )
    _print_summary("mixed (30% unique)", mixed)

    return {
        "scenario": "cache",
        "description": "Cold vs. warm cache latency comparison",
        "cold": cold.to_dict(),
        "warm": warm.to_dict(),
        "mixed": mixed.to_dict(),
    }


async def run_cache_hit_rate_over_time(gen: LoadGenerator, url: str) -> dict:
    """Track cache hit rate as the cache warms up over time."""
    print("\n[Scenario: cache_hitrate] Hit-rate buildup over time...")
    await _clear_cache(url)

    snapshots = []
    cumulative_hits = 0
    cumulative_total = 0
    for batch_num in range(10):
        # 60% repeat ratio so cache warms gradually
        summary = await gen.run(
            scenario=f"hitrate_batch_{batch_num}",
            num_requests=20,
            concurrency=4,
            unique_ratio=0.4,
        )
        d = summary.to_dict()
        cumulative_hits += d["cache"]["hits"]
        cumulative_total += d["successful"]
        snapshots.append({
            "batch": batch_num + 1,
            "requests_so_far": cumulative_total,
            "hit_rate_batch": d["cache"]["hit_rate"],
            "hit_rate_cumulative": round(cumulative_hits / max(cumulative_total, 1), 4),
            "latency_p50": d["latency_ms"]["p50"],
        })
        print(
            f"    Batch {batch_num+1:2d}: hit_rate={d['cache']['hit_rate']:.0%}  "
            f"cumulative={cumulative_hits/max(cumulative_total,1):.0%}  "
            f"p50={d['latency_ms']['p50']:.1f}ms"
        )

    return {"scenario": "cache_hitrate", "snapshots": snapshots}


async def run_throughput(gen: LoadGenerator, url: str) -> dict:
    """Throughput under low / medium / high load levels."""
    print("\n[Scenario: throughput] Throughput at multiple load levels...")
    await _clear_cache(url)

    load_levels = [
        ("low",    10,  4),
        ("medium", 50, 16),
        ("high",  100, 32),
    ]
    results = []
    for label, num_req, concurrency in load_levels:
        await _clear_cache(url)
        summary = await gen.run(
            scenario=f"throughput_{label}",
            num_requests=num_req,
            concurrency=concurrency,
            unique_ratio=0.7,
        )
        d = summary.to_dict()
        d["load_level"] = label
        results.append(d)
        _print_summary(f"load={label} (n={num_req}, c={concurrency})", summary)

    return {
        "scenario": "throughput",
        "description": "Throughput (req/s) at low / medium / high concurrency",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    url = args.server_url.rstrip("/")
    print(f"\nConnecting to inference server at {url}...")
    await _wait_for_server(url)

    gen = LoadGenerator(base_url=url)
    all_results: dict = {}

    run_all = args.scenario == "all"

    if run_all or args.scenario == "single":
        all_results["single"] = await run_single(gen, url)
        _save("single_latency.json", all_results["single"])

    if run_all or args.scenario == "batch":
        all_results["batch"] = await run_batch(gen, url)
        _save("batch_performance.json", all_results["batch"])

    if run_all or args.scenario == "cache":
        all_results["cache"] = await run_cache(gen, url)
        _save("cache_comparison.json", all_results["cache"])
        all_results["cache_hitrate"] = await run_cache_hit_rate_over_time(gen, url)
        _save("cache_hitrate_over_time.json", all_results["cache_hitrate"])

    if run_all or args.scenario == "throughput":
        all_results["throughput"] = await run_throughput(gen, url)
        _save("throughput.json", all_results["throughput"])

    if run_all:
        _save("all_results.json", all_results)

    print("\nBenchmarks complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmarks.py",
        description="Benchmark suite for the LLM inference server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=["single", "batch", "cache", "throughput", "all"],
        default="all",
        help="Which benchmark scenario to run (default: all).",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Base URL of the inference server (default: http://localhost:8000).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
