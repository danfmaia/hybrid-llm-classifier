"""Performance benchmarking for the hybrid classifier system."""

import asyncio
import time
from typing import List, Dict, Any
import statistics
from dataclasses import dataclass
import json
import aiohttp
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

# Test documents of varying sizes
SMALL_DOC = """This contract agreement is made between Party A and Party B."""

MEDIUM_DOC = """LEGAL OPINION
In the matter of corporate restructuring...
[Content truncated for brevity - 1KB of legal text]
"""

LARGE_DOC = """REGULATORY COMPLIANCE REPORT
Comprehensive analysis of regulatory requirements...
[Content truncated for brevity - 2KB of legal text]
"""


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    avg_response_time: float
    p95_response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    success_rate: float
    category_distribution: Dict[str, int]


async def benchmark_single_document(
    client: aiohttp.ClientSession,
    base_url: str,
    token: str,
    document: str
) -> Dict[str, float]:
    """Benchmark single document classification."""
    start_time = time.perf_counter()
    print(f"\nTesting document of size: {len(document)} chars")

    try:
        async with client.post(
            f"{base_url}/api/v1/classify",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": document}
        ) as response:
            latency = time.perf_counter() - start_time
            print(
                f"Request completed in {latency:.2f}s with status {response.status}")

            response_data = await response.json()
            success = response.status == 200

            if not success:
                print(
                    f"Error response: {response_data.get('detail', 'Unknown error')}")
            else:
                print(f"Classification: {response_data.get('category')} "
                      f"(confidence: {response_data.get('confidence', 0.0):.2f})")

            return {
                "latency": latency,
                "success": success,
                "status_code": response.status,
                "category": response_data.get("category", "unknown"),
                "confidence": response_data.get("confidence", 0.0),
                "error": response_data.get("detail") if not success else None
            }
    except Exception as e:
        latency = time.perf_counter() - start_time
        print(f"Error during request: {str(e)}")
        return {
            "latency": latency,
            "success": False,
            "status_code": 0,
            "error": str(e)
        }


async def benchmark_batch_classification(
    client: aiohttp.ClientSession,
    base_url: str,
    token: str,
    batch_size: int
) -> Dict[str, Any]:
    """Benchmark batch document classification."""
    # Reduced batch sizes for initial testing
    documents = [
        SMALL_DOC,
        MEDIUM_DOC,
        LARGE_DOC
    ] * (batch_size // 3 + 1)
    documents = documents[:batch_size]

    start_time = time.perf_counter()
    print(f"\nTesting batch classification with {batch_size} documents")

    try:
        async with client.post(
            f"{base_url}/api/v1/classify/batch",
            headers={"Authorization": f"Bearer {token}"},
            json=[{"text": doc} for doc in documents]
        ) as response:
            results = await response.json()
            latency = time.perf_counter() - start_time
            print(f"Batch request completed in {latency:.2f}s")
            return {
                "latency": latency,
                "success": response.status == 200,
                "results": results,
                "throughput": batch_size / latency
            }
    except Exception as e:
        print(f"Error during batch request: {str(e)}")
        return {
            "latency": time.perf_counter() - start_time,
            "success": False,
            "error": str(e)
        }


async def simulate_concurrent_users(
    base_url: str,
    token: str,
    num_users: int,
    requests_per_user: int
) -> List[Dict[str, Any]]:
    """Simulate concurrent users making classification requests."""
    print(
        f"\nSimulating {num_users} concurrent users with {requests_per_user} requests each")
    async with aiohttp.ClientSession() as client:
        tasks = []
        for user_id in range(num_users):
            for req_id in range(requests_per_user):
                # Randomly select document size
                doc = np.random.choice([SMALL_DOC, MEDIUM_DOC, LARGE_DOC])
                print(
                    f"User {user_id}, Request {req_id}: Document size {len(doc)} chars")
                tasks.append(
                    benchmark_single_document(client, base_url, token, doc)
                )

        return await asyncio.gather(*tasks)


async def run_benchmarks(
    base_url: str = "http://localhost:8001",
    auth: Dict[str, str] = {"username": "testuser", "password": "testpass"}
) -> BenchmarkResult:
    """Run complete benchmark suite with reduced test parameters."""
    async with aiohttp.ClientSession() as client:
        print("\nStarting benchmark suite with reduced parameters...")

        # Get auth token
        print("\nGetting authentication token...")
        try:
            async with client.post(
                f"{base_url}/api/v1/auth/token",
                data=auth,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            ) as response:
                if response.status != 200:
                    print(
                        f"Authentication failed with status {response.status}")
                    response_data = await response.json()
                    print(
                        f"Error: {response_data.get('detail', 'Unknown error')}")
                    return BenchmarkResult(
                        avg_response_time=0.0,
                        p95_response_time=0.0,
                        throughput=0.0,
                        error_rate=1.0,
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        success_rate=0.0,
                        category_distribution={}
                    )

                token_data = await response.json()
                token = token_data["access_token"]
                print("Authentication successful")
        except Exception as e:
            print(f"Failed to authenticate: {str(e)}")
            return BenchmarkResult(
                avg_response_time=0.0,
                p95_response_time=0.0,
                throughput=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                success_rate=0.0,
                category_distribution={}
            )

        # 1. Single Document Tests (reduced)
        print("\nRunning single document tests...")
        single_results = []
        for doc in [SMALL_DOC, MEDIUM_DOC]:  # Skip large doc initially
            results = await asyncio.gather(*[
                benchmark_single_document(client, base_url, token, doc)
                for _ in range(2)  # Reduced from 10 to 2 requests per document
            ])
            single_results.extend(results)

        # 2. Batch Classification Tests (reduced)
        print("\nRunning batch classification tests...")
        batch_sizes = [5, 10]  # Reduced batch sizes
        batch_results = []
        for size in batch_sizes:
            result = await benchmark_batch_classification(
                client, base_url, token, size
            )
            batch_results.append(result)

        # 3. Concurrent User Tests (reduced)
        print("\nRunning concurrent user tests...")
        concurrent_configs = [
            (2, 2),    # 2 users, 2 requests each
            (5, 1),    # 5 users, 1 request each
        ]

        concurrent_results = []
        for num_users, requests_per_user in concurrent_configs:
            print(f"Testing with {num_users} concurrent users...")
            results = await simulate_concurrent_users(
                base_url, token, num_users, requests_per_user
            )
            concurrent_results.append(results)

        # Calculate aggregate metrics
        all_latencies = [r["latency"] for r in single_results if r["success"]]
        successful_requests = sum(1 for r in single_results if r["success"])
        total_requests = len(single_results)

        # Collect error statistics
        error_types = {}
        for result in single_results:
            if not result["success"] and "error" in result:
                error_type = result.get("error", "Unknown error")
                error_types[error_type] = error_types.get(error_type, 0) + 1

        if not all_latencies:
            print("\nNo successful requests were made.")
            print("\nError distribution:")
            for error, count in error_types.items():
                print(f"  {error}: {count} occurrences")
            return BenchmarkResult(
                avg_response_time=0.0,
                p95_response_time=0.0,
                throughput=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                success_rate=0.0,
                category_distribution={}
            )

        result = BenchmarkResult(
            avg_response_time=statistics.mean(all_latencies),
            p95_response_time=np.percentile(all_latencies, 95),
            throughput=len(all_latencies) / sum(all_latencies),
            error_rate=1 - (successful_requests / total_requests),
            cpu_usage=0.0,  # Would need system metrics integration
            memory_usage=0.0,  # Would need system metrics integration
            success_rate=successful_requests / total_requests,
            category_distribution={
                cat: sum(1 for r in single_results
                         if r.get("category") == cat)
                for cat in set(r.get("category") for r in single_results
                               if "category" in r)
            }
        )

        print("\nBenchmark Results:")
        print(f"Average Response Time: {result.avg_response_time:.2f}s")
        print(
            f"95th Percentile Response Time: {result.p95_response_time:.2f}s")
        print(f"Throughput: {result.throughput:.2f} requests/second")
        print(f"Success Rate: {result.success_rate * 100:.1f}%")
        print("\nCategory Distribution:", result.category_distribution)

        if error_types:
            print("\nError Distribution:")
            for error, count in error_types.items():
                print(f"  {error}: {count} occurrences")

        return result


def save_benchmark_results(results: BenchmarkResult, output_file: str) -> None:
    """Save benchmark results to a file."""
    with open(output_file, 'w') as f:
        json.dump({
            "avg_response_time": results.avg_response_time,
            "p95_response_time": results.p95_response_time,
            "throughput": results.throughput,
            "error_rate": results.error_rate,
            "success_rate": results.success_rate,
            "category_distribution": results.category_distribution
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
