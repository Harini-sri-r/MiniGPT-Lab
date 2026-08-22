"""Demo for Commerce Phase 11 – Marketplace Integration Foundation.

Run with:
    python -m experiments.commerce_phase11_demo

The demo parses a sample query, runs the orchestrator, and prints a concise
summary including which providers were queried, how many results each returned,
and a data‑source notice.
"""

import os

from commerce.models import UserRequirements
from commerce.orchestrator import orchestrate_search


def main():
    # Sample requirement – a typical user query translated into UserRequirements.
    req = UserRequirements(
        query="I need a laptop under Rs. 60000 for programming",
        category="Laptop",
        budget=60000,
        features=("16GB RAM", "SSD"),
        use_case="programming",
    )

    # Orchestrate search using environment configuration (default mock mode).
    result = orchestrate_search(req)

    # Display a high‑level summary.
    print("=== Commerce Phase 11 Demo ===")
    print(f"Query: {req.query}")
    print(f"Data source summary: {result.data_source_summary}")
    print(f"Notice: {result.data_notice}\n")
    print("Provider results:")
    for prov, info in result.provider_results.items():
        status = "OK" if info.success else f"FAIL ({info.error})"
        print(f" - {prov}: {status}, {info.result_count} products")

    # Show first few normalized products.
    print("\nSample products (up to 3):")
    for prod in result.products[:3]:
        print(
            f"[{prod.data_source}] {prod.title} – Rs.{prod.price} | Rating: {prod.rating} "
            f"(retrieved {prod.retrieved_at})"
        )
    print("\nNote: Current marketplace data is synthetic/offline. Live providers are not configured.")


if __name__ == "__main__":
    # Ensure the workspace root is in PYTHONPATH for imports when running as a module.
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    main()
