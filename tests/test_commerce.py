"""Tests for the deterministic, synthetic-only Phase 7 commerce foundation."""

from commerce.catalog import PLATFORMS, load_mock_products
from commerce.comparison import compare_product_group, group_products
from commerce.models import Product, UserRequirements
from commerce.normalize import normalize_product_name, products_match
from commerce.parser import parse_requirements
from commerce.recommender import recommend_products, score_product
from commerce.search import search_products
from commerce.service import build_comparison_response


def test_product_schema_and_mock_catalog_size():
    products = load_mock_products()
    assert len(products) >= 100
    assert {product.platform for product in products} == set(PLATFORMS)
    assert len({product.category for product in products}) == 7
    assert all(isinstance(product, Product) and product.product_url.startswith("https://mock.example/") for product in products)


def test_search_and_all_supported_filters():
    products = load_mock_products()
    results = search_products("wireless headphones", products=products, category="headphones", maximum_price=3000, minimum_rating=4.2, brand="Sony", platform="Flipkart")
    assert len(results) == 1
    assert results[0].product_name == "Sony WH-CH520"
    assert results[0].price <= 3000 and results[0].rating >= 4.2


def test_normalization_and_cross_platform_grouping():
    assert normalize_product_name("Apple AirPods (3rd Generation)") == "apple airpods 3"
    assert products_match("Apple AirPods 3rd Gen", "AirPods 3")
    groups = group_products(load_mock_products())
    airpods = groups["apple airpods 3"]
    assert len(airpods) == 3


def test_comparison_detects_cheapest_highest_rated_value_and_differences():
    sony = next(group for group in group_products(load_mock_products()).values() if group[0].product_name == "Sony WH-CH520")
    comparison = compare_product_group(sony)
    assert comparison["cheapest"].platform == "Flipkart"
    assert comparison["highest_rated"].platform == "Amazon"
    assert comparison["best_value"] in sony
    assert comparison["price_difference"] > 0
    assert comparison["rating_difference"] > 0


def test_recommendation_scoring_and_budget_requirement_match():
    products = search_products("wireless headphones", products=load_mock_products(), category="headphones", maximum_price=3000)
    requirements = parse_requirements("I need wireless headphones under ₹3000 with good battery life")
    recommendations = recommend_products(products, requirements)
    assert recommendations and recommendations[0]["score"] <= 100
    assert score_product(recommendations[0]["product"], requirements) >= 0


def test_requirement_parser_extracts_category_budget_features_and_use_case():
    headphone_query = parse_requirements("I need wireless headphones under ₹3,000 with good battery life.")
    assert headphone_query.category == "headphones"
    assert headphone_query.budget == 3000
    assert headphone_query.features == ("wireless", "good battery life")
    laptop_query = parse_requirements("Best laptop under Rs. 60000 for programming")
    assert laptop_query.category == "laptops"
    assert laptop_query.budget == 60000
    assert laptop_query.use_case == "programming"


def test_final_comparison_output_is_structured_and_explicitly_mock():
    response = build_comparison_response("I need wireless headphones under ₹3000")
    assert response["found_count"] > 0
    assert response["recommendations"]
    assert response["comparisons"]
    assert "synthetic mock data" in response["data_notice"]
