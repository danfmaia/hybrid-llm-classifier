"""Unit tests for benchmarking infrastructure."""

import pytest
import json
from unittest.mock import patch, AsyncMock, mock_open
import aiohttp
from pathlib import Path
import numpy as np
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

from tests.benchmark_classifier import (
    benchmark_single_document,
    benchmark_batch_classification,
    simulate_concurrent_users,
    run_benchmarks,
    save_benchmark_results,
    BenchmarkResult,
    SMALL_DOC,
    MEDIUM_DOC,
    LARGE_DOC
)


class MockResponse:
    """Mock aiohttp.ClientResponse."""

    def __init__(self, data: Any, status: int = 200):
        self.data = data
        self.status = status

    async def json(self):
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    """Mock aiohttp.ClientSession."""

    def __init__(self, mock_response: Any, status: int = 200):
        self.mock_response = mock_response
        self.status = status

    def post(self, url: str, *args, **kwargs) -> MockResponse:
        """Mock post method that returns a MockResponse that can be used as a context manager."""
        return MockResponse(self.mock_response, self.status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class ErrorClientSession(MockClientSession):
    """Mock client session that raises errors."""

    def post(self, *args, **kwargs):
        raise Exception("Test error")


@pytest.mark.asyncio
async def test_benchmark_single_document():
    """Test single document benchmarking."""
    mock_response = {
        "category": "Contract",
        "confidence": 0.85,
        "validation_score": 0.75
    }

    async with MockClientSession(mock_response) as client:
        result = await benchmark_single_document(
            client,
            "http://test",
            "test_token",
            SMALL_DOC
        )

        assert result["success"] is True
        assert result["category"] == "Contract"
        assert result["confidence"] == 0.85
        assert "latency" in result
        assert result["status_code"] == 200


@pytest.mark.asyncio
async def test_benchmark_single_document_error():
    """Test error handling in single document benchmarking."""
    async with ErrorClientSession(None) as client:
        result = await benchmark_single_document(
            client,
            "http://test",
            "test_token",
            SMALL_DOC
        )

        assert result["success"] is False
        assert result["error"] == "Test error"
        assert "latency" in result
        assert result["status_code"] == 0


@pytest.mark.asyncio
async def test_benchmark_batch_classification():
    """Test batch classification benchmarking."""
    mock_response = [
        {
            "category": "Contract",
            "confidence": 0.85,
            "validation_score": 0.75
        }
    ] * 5  # 5 identical results

    async with MockClientSession(mock_response) as client:
        result = await benchmark_batch_classification(
            client,
            "http://test",
            "test_token",
            5
        )

        assert result["success"] is True
        assert "latency" in result
        assert "throughput" in result
        assert len(result["results"]) == 5


@pytest.mark.asyncio
async def test_simulate_concurrent_users():
    """Test concurrent user simulation."""
    mock_response = {
        "category": "Contract",
        "confidence": 0.85,
        "validation_score": 0.75
    }

    async with MockClientSession(mock_response) as client:
        with patch("aiohttp.ClientSession", return_value=client):
            results = await simulate_concurrent_users(
                "http://test",
                "test_token",
                2,  # num_users
                2   # requests_per_user
            )

            assert len(results) == 4  # 2 users * 2 requests
            assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_run_benchmarks():
    """Test complete benchmark suite execution."""
    mock_auth_response = {"access_token": "test_token"}
    mock_classify_response = {
        "category": "Contract",
        "confidence": 0.85,
        "validation_score": 0.75
    }

    class DynamicMockClientSession(MockClientSession):
        def post(self, url: str, *args, **kwargs) -> MockResponse:
            if "auth/token" in url:
                return MockResponse(mock_auth_response)
            return MockResponse(mock_classify_response)

    async with DynamicMockClientSession(None) as client:
        with patch("aiohttp.ClientSession", return_value=client):
            result = await run_benchmarks()
            assert result.success_rate > 0
            assert result.avg_response_time > 0
            assert result.throughput > 0


def test_save_benchmark_results(tmp_path):
    """Test saving benchmark results to file."""
    result = BenchmarkResult(
        avg_response_time=1.5,
        p95_response_time=2.0,
        throughput=10.0,
        error_rate=0.1,
        cpu_usage=50.0,
        memory_usage=1000.0,
        success_rate=0.9,
        category_distribution={"Contract": 5}
    )

    output_file = tmp_path / "test_results.json"
    save_benchmark_results(result, output_file)

    # Verify file contents
    with open(output_file) as f:
        saved_data = json.load(f)
        assert saved_data["avg_response_time"] == 1.5
        assert saved_data["p95_response_time"] == 2.0
        assert saved_data["throughput"] == 10.0
        assert saved_data["error_rate"] == 0.1
        assert saved_data["success_rate"] == 0.9
        assert saved_data["category_distribution"] == {"Contract": 5}
