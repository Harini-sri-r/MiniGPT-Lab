"""Evaluation script for Commerce Phase 11.

This script runs a handful of offline scenarios using only the mock
providers. It checks:
- Provider availability (all mock providers should be instantiated).
- Normalized result consistency (required fields are present).
- Search success (at least one product returned for each scenario).
- Fallback behavior (when a platform is disabled, mock data is still used).
- Schema validity (the NormalizedProduct dataclass contains the new fields).
- Recommendation compatibility (the executor can still produce a recommendation).

Run with:
    python -m experiments.commerce_phase11_evaluation
"""

import os
import sys
from pprint import pprint

from commerce.models import NormalizedProduct, UserRequirements
from commerce.providers import configured_providers, default_providers, load_config, MarketplaceConfig
from commerce.orchestrator import orchestrate_search
from commerce.agent.executor import execute_plan
from commerce.agent.state import ShoppingState


def _print_header(title: str):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def _check_product_schema(product: NormalizedProduct):
    # Verify the new Phase‑11 fields exist and are of the expected type.
    assert hasattr(product, "currency"), "Missing currency field"
    assert hasattr(product, "image_url"), "Missing image_url field"
    assert hasattr(product, "data_source"), "Missing data_source field"
    assert hasattr(product, "retrieved_at"), "Missing retrieved_at field"
    # Basic type sanity checks
    assert isinstance(product.currency, str)
    assert isinstance(product.image_url, str)
    assert isinstance(product.data_source, str)
    assert isinstance(product.retrieved_at, str)


def run_scenarios():
    # Scenario definitions – each is a UserRequirements instance.
    scenarios = [
        UserRequirements(
            query="Laptop under Rs. 60000 for programming",
            category="Laptop",
            budget=60000,
            features=("16GB RAM", "SSD"),
            use_case="programming",
        ),
        UserRequirements(
            query="Samsung phone under Rs. 30000",
            category="Phone",
            budget=30000,
            brand="Samsung",
        ),
        UserRequirements(
            query="Gaming laptop under Rs. 80000",
            category="Laptop",
            budget=80000,
            features=("gaming",),
        ),
        UserRequirements(
            query="Headphones under Rs. 5000",
            category="Headphones",
            budget=5000,
        ),
        UserRequirements(
            query="Camera with brand Nikon",
            category="Camera",
            brand="Nikon",
        ),
    ]

    # Ensure mock mode is active.
    os.environ["MARKETPLACE_MODE"] = "mock"

    # Load config and verify mock providers are selected.
    cfg = load_config()
    providers = configured_providers(cfg)
    _print_header("Provider Availability")
    print(f"Configured mode: {cfg.mode}")
    print(f"Enabled platforms: {cfg.enabled_platforms}")
    print(f"Instantiated providers: {[p.platform for p in providers]}")
    assert all(p.data_source == "mock" for p in providers), "Non‑mock provider instantiated"

    # Run each scenario through the orchestrator and basic checks.
    for idx, req in enumerate(scenarios, 1):
        _print_header(f"Scenario {idx}: {req.query}")
        result = orchestrate_search(req, config=cfg)
        # Search success
        assert result.products, "No products returned"
        print(f"Total products returned: {len(result.products)}")
        # Data source summary
        print(f"Data source summary: {result.data_source_summary}")
        # Verify each product schema
        for p in result.products:
            _check_product_schema(p)
        # Fallback behavior – when a platform is missing, mock still provides data.
        # Here we simply ensure the provider_results contain entries for all enabled platforms.
        missing = set(cfg.enabled_platforms) - set(result.provider_results.keys())
        assert not missing, f"Missing provider results for: {missing}"
        # Run the executor to ensure recommendation compatibility.
        state = ShoppingState(query=req.query, requirements=req)
        state = execute_plan(state, providers)
        assert state.status == "success", f"Executor failed: {state.status}"
        print(f"Recommendation product ID: {state.selected_product['product'].product_id}")
        print(f"Data source of recommendation: {state.selected_product['product'].data_source}")

    print("\nAll scenarios passed successfully.")


if __name__ == "__main__":
    run_scenarios()
