"""End-to-end response assembly for the Phase 7 mock commerce demo."""

from .catalog import load_mock_products
from .comparison import compare_product_group, group_products
from .parser import parse_requirements
from .recommender import recommend_products
from .search import search_products


def build_comparison_response(query: str, *, products=None, limit: int = 3) -> dict:
    """Build a structured, clearly mock-only comparison response for one query."""
    requirements = parse_requirements(query)
    catalog = list(products) if products is not None else load_mock_products()
    results = search_products(query, products=catalog, category=requirements.category, maximum_price=requirements.budget)
    recommendations = recommend_products(results, requirements, limit=limit)
    comparisons = [compare_product_group(group) for group in group_products(results).values() if len(group) > 1]
    return {
        "data_notice": "All prices, listings, ratings, and availability are synthetic mock data; no marketplace is queried live.",
        "requirements": requirements,
        "found_count": len(results),
        "recommendations": recommendations,
        "comparisons": comparisons,
    }
