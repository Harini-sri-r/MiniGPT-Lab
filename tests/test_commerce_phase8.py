"""Phase 8 tests: offline provider architecture and explainable assistant behavior."""

from commerce.assistant import build_assistant_result, score_normalized_product
from commerce.matching import match_products
from commerce.parser import parse_requirements
from commerce.providers import AmazonMockProvider, FlipkartMockProvider, MarketplaceProvider, MeeshoMockProvider, default_providers


def test_provider_interface_and_normalized_schema():
    provider = AmazonMockProvider()
    assert isinstance(provider, MarketplaceProvider)
    products = provider.search_products(parse_requirements("laptop under Rs. 60000"))
    assert products and products[0].platform == "Amazon"
    assert products[0].title and products[0].features and provider.get_product(products[0].product_id) == products[0]


def test_all_mock_providers_are_offline_and_searchable():
    requirements = parse_requirements("Samsung phone under 30000")
    providers = default_providers()
    assert [provider.platform for provider in providers] == ["Amazon", "Flipkart", "Meesho"]
    assert all(provider.search_products(requirements) for provider in providers)
    assert MeeshoMockProvider().get_product("missing") is None


def test_rich_query_parser_extracts_structured_requirements():
    parsed = parse_requirements("I want a Samsung phone under 30,000 with good camera and at least 4.2 rating")
    assert parsed.category == "smartphones" and parsed.budget == 30000 and parsed.brand == "Samsung"
    assert parsed.minimum_rating == 4.2 and "good camera" in parsed.required_features
    coding = parse_requirements("I need a laptop for coding under 70k with 16GB RAM")
    assert coding.use_case == "programming" and coding.budget == 70000 and "16 GB RAM" in coding.required_features


def test_product_matching_returns_high_confidence_for_equivalent_mock_listings():
    requirements = parse_requirements("Lenovo laptop")
    amazon = AmazonMockProvider().search_products(requirements)[0]
    flipkart = FlipkartMockProvider().search_products(requirements)[0]
    match = match_products(amazon, flipkart)
    assert match["likely_match"] and match["confidence"] >= 0.9


def test_transparent_recommendation_score_has_all_components():
    product = AmazonMockProvider().search_products(parse_requirements("laptop under 60000"))[0]
    score = score_normalized_product(product, parse_requirements("laptop under 60000 for programming"))
    assert set(score) == {"price_score", "rating_score", "review_score", "feature_score", "requirement_score", "overall_score"}
    assert 0 <= score["overall_score"] <= 100


def test_structured_comparison_explanation_alternatives_and_empty_result():
    result = build_assistant_result("Best laptop under Rs. 60000 for programming")
    assert result["best_overall"] and result["alternatives"] and result["comparison_table"]
    assert result["best_overall"]["why"] and "Offline synthetic" in result["data_notice"]
    empty = build_assistant_result("Apple laptop under Rs. 1000")
    assert empty["found_count"] == 0 and empty["best_overall"] is None


def test_cli_rendering_and_invalid_query_behavior():
    from app.commerce_cli import render_result
    rendered = render_result(build_assistant_result("phone under 20000 with good camera"))
    assert "Commerce Assistant" in rendered and "Data source: Offline synthetic" in rendered
    invalid = render_result(build_assistant_result("something unavailable under Rs. 1"))
    assert "No matching mock products" in invalid
