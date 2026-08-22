import os
import pytest
from commerce.models import NormalizedProduct
from commerce.providers import default_providers, configured_providers, load_config
from commerce.orchestrator import orchestrate_search
from commerce.models import UserRequirements

def test_normalized_product_fields():
    prod = NormalizedProduct(
        product_id="p1",
        platform="Amazon",
        title="Test Product",
        brand="Brand",
        category="Laptop",
        price=50000,
        rating=4.5,
        review_count=10,
        features={"ram": "16GB"},
        product_url="http://example.com",
        availability="In stock",
        currency="INR",
        image_url="",
        data_source="mock",
        retrieved_at="2023-01-01T00:00:00",
    )
    assert prod.currency == "INR"
    assert prod.data_source == "mock"
    assert prod.retrieved_at == "2023-01-01T00:00:00"

def test_default_providers_mock():
    providers = default_providers()
    assert len(providers) == 3
    for p in providers:
        assert p.data_source == "mock"
        assert hasattr(p, "search_products")

def test_config_load_defaults(monkeypatch):
    monkeypatch.delenv("MARKETPLACE_MODE", raising=False)
    monkeypatch.delenv("MARKETPLACE_PLATFORMS", raising=False)
    cfg = load_config()
    assert cfg.mode == "mock"
    assert set(cfg.enabled_platforms) == {"Amazon", "Flipkart", "Meesho"}
    assert cfg.timeout_seconds == 10
    assert cfg.max_results == 20

def test_orchestrator_with_mock(monkeypatch):
    monkeypatch.setenv("MARKETPLACE_MODE", "mock")
    req = UserRequirements(query="test", category=None)
    result = orchestrate_search(req)
    sources = {p.data_source for p in result.products}
    assert sources == {"mock"}
    assert result.data_source_summary == "mock"
    assert "mock" in result.data_notice.lower()
    assert set(result.provider_results.keys()) == {"Amazon", "Flipkart", "Meesho"}
    for info in result.provider_results.values():
        assert info.success
        assert info.result_count >= 0
