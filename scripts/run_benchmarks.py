#!/usr/bin/env python3
"""Script to run benchmarks and generate a performance report."""

from datetime import datetime
import asyncio
from tests.benchmark_classifier import run_benchmarks, save_benchmark_results
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


async def main():
    """Run benchmarks and generate report."""
    # Create results directory
    results_dir = project_root / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    try:
        # Run benchmarks
        print("Starting benchmark suite...")
        results = await run_benchmarks()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"benchmark_results_{timestamp}.json"
        save_benchmark_results(results, results_file)

        print(f"\nBenchmark results saved to: {results_file}")

    except Exception as e:
        print(f"Error during benchmark: {e}")


if __name__ == "__main__":
    asyncio.run(main())
