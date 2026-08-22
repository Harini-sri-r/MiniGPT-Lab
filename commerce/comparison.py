"""Cross-platform grouping and comparison for mock listings."""

from collections import defaultdict
from collections.abc import Iterable

from .models import Product
from .normalize import normalize_product_name


def group_products(products: Iterable[Product]) -> dict[str, list[Product]]:
    groups: dict[str, list[Product]] = defaultdict(list)
    for product in products:
        groups[normalize_product_name(product.product_name)].append(product)
    return dict(groups)


def compare_product_group(products: Iterable[Product]) -> dict:
    """Compare one normalized product group across platforms."""
    listings = sorted(products, key=lambda product: product.price)
    if not listings:
        raise ValueError("Cannot compare an empty product group")
    cheapest = listings[0]
    highest_rated = max(listings, key=lambda product: (product.rating, product.review_count))
    best_value = max(listings, key=lambda product: (product.rating * 20) + min(product.review_count / 1000, 10) - product.price / 1000)
    return {
        "normalized_name": normalize_product_name(listings[0].product_name),
        "listings": listings,
        "cheapest": cheapest,
        "highest_rated": highest_rated,
        "best_value": best_value,
        "price_difference": max(item.price for item in listings) - cheapest.price,
        "rating_difference": round(max(item.rating for item in listings) - min(item.rating for item in listings), 1),
    }
