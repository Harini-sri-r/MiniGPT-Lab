"""Deterministic, mock-data commerce comparison foundation for MiniGPT Phase 7."""

from .catalog import load_mock_products
from .comparison import compare_product_group, group_products
from .parser import parse_requirements
from .recommender import recommend_products
from .search import search_products
from .service import build_comparison_response

__all__ = [
    "build_comparison_response",
    "compare_product_group",
    "group_products",
    "load_mock_products",
    "parse_requirements",
    "recommend_products",
    "search_products",
]
