"""Search and filter the synthetic product catalog."""

from collections.abc import Iterable

from .models import Product


def search_products(
    query: str = "",
    *,
    products: Iterable[Product],
    category: str | None = None,
    maximum_price: int | None = None,
    minimum_rating: float | None = None,
    brand: str | None = None,
    platform: str | None = None,
) -> list[Product]:
    """Return matching mock products, ordered by relevance then price."""
    terms = set(query.lower().replace("-", " ").split())
    filtered: list[tuple[int, Product]] = []
    for product in products:
        searchable = " ".join((product.product_name, product.brand, product.category, *product.specifications.values())).lower()
        requested_category = category.lower() if category else None
        if requested_category and not requested_category.endswith("s"):
            requested_category += "s"
        if requested_category and product.category.lower() != requested_category:
            continue
        if maximum_price is not None and product.price > maximum_price:
            continue
        if minimum_rating is not None and product.rating < minimum_rating:
            continue
        if brand and product.brand.lower() != brand.lower():
            continue
        if platform and product.platform.lower() != platform.lower():
            continue
        matches = sum(term in searchable for term in terms)
        if terms and not matches:
            continue
        filtered.append((matches, product))
    return [product for _, product in sorted(filtered, key=lambda item: (-item[0], -item[1].rating, item[1].price))]
