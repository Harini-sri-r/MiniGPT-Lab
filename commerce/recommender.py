"""Explainable deterministic recommendation scoring for mock products."""

from collections.abc import Iterable, Mapping

from .models import Product, UserRequirements

DEFAULT_WEIGHTS = {
    "price": 0.35,
    "rating": 0.30,
    "review": 0.15,
    "requirement": 0.20,
}


def score_product(product: Product, requirements: UserRequirements, weights: Mapping[str, float] | None = None) -> float:
    """Score a product from 0 to 100 using documented configurable weights."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if round(sum(weights.values()), 8) != 1.0:
        raise ValueError("Recommendation weights must sum to 1.0")
    budget = requirements.budget or max(product.price, 1)
    price_score = min(1.0, budget / product.price) if product.price else 1.0
    rating_score = product.rating / 5.0
    review_score = min(1.0, product.review_count / 10000)
    haystack = " ".join((product.product_name, product.category, *product.specifications.keys(), *product.specifications.values())).lower()
    requirement_score = 1.0 if not requirements.features else sum(feature.lower() in haystack for feature in requirements.features) / len(requirements.features)
    return round(100 * (
        weights["price"] * price_score + weights["rating"] * rating_score +
        weights["review"] * review_score + weights["requirement"] * requirement_score
    ), 2)


def recommend_products(products: Iterable[Product], requirements: UserRequirements, weights: Mapping[str, float] | None = None, limit: int = 3) -> list[dict]:
    """Rank mock listings and provide a concise deterministic reason."""
    ranked = sorted(
        ((score_product(product, requirements, weights), product) for product in products),
        key=lambda entry: (-entry[0], -entry[1].rating, entry[1].price),
    )
    recommendations = []
    for score, product in ranked[:limit]:
        reason = "Best overall"
        if requirements.budget and product.price <= requirements.budget * 0.65:
            reason = "Best budget"
        elif "battery" in " ".join(requirements.features).lower() and "battery_life" in product.specifications:
            reason = "Best battery match"
        recommendations.append({"product": product, "score": score, "reason": reason})
    return recommendations
