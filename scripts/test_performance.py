#!/usr/bin/env python3
"""Simple performance test for the optimized model."""

from src.app.models.classifier import HybridClassifier
import asyncio
import time
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


# Test document (legal opinion)
TEST_DOC = """
LEGAL OPINION

In the matter of corporate restructuring for XYZ Corporation, we have reviewed the proposed merger agreement
and related documentation. Based on our analysis of applicable state and federal laws, we conclude that
the proposed transaction complies with relevant regulatory requirements.

Key considerations include:
1. Antitrust implications
2. Securities law compliance
3. Corporate governance requirements
4. Shareholder approval procedures

This opinion is subject to the assumptions and qualifications set forth herein.
"""


async def main():
    """Run a simple performance test."""
    print("\nInitializing classifier...")
    classifier = HybridClassifier(
        ollama_base_url="http://localhost:11434",
        model_name="legal-classifier",  # Using our optimized model
        embedding_dim=384
    )

    print("\nStarting classification test...")
    start_time = time.perf_counter()

    try:
        result = await classifier.classify(TEST_DOC)
        end_time = time.perf_counter()

        print(f"\nClassification completed in {end_time - start_time:.2f}s")
        print(f"Category: {result.category}")
        print(f"Confidence: {result.confidence:.2f}")

        if result.performance_metrics:
            print("\nPerformance Metrics:")
            print(
                f"LLM Latency: {result.performance_metrics.llm_latency:.2f}s")
            print(
                f"Embedding Latency: {result.performance_metrics.embedding_latency:.2f}s")
            print(
                f"Validation Latency: {result.performance_metrics.validation_latency:.2f}s")
            print(
                f"Total Latency: {result.performance_metrics.total_latency:.2f}s")

    except Exception as e:
        print(f"\nError during classification: {e}")

if __name__ == "__main__":
    asyncio.run(main())
