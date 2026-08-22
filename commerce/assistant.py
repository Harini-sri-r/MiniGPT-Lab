"""Phase 8 orchestration for the offline, explainable shopping assistant."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .matching import match_products
from .models import NormalizedProduct, UserRequirements
from .parser import parse_requirements
from .providers import MarketplaceProvider, default_providers

DEFAULT_ASSISTANT_WEIGHTS = {"price": 0.30, "rating": 0.25, "reviews": 0.15, "feature": 0.20, "requirement": 0.10}


def score_normalized_product(product: NormalizedProduct, requirements: UserRequirements, weights: Mapping[str, float] | None = None) -> dict:
    """Return transparent score components (each 0..1) and an overall 0..100 score."""
    weights = {**DEFAULT_ASSISTANT_WEIGHTS, **(weights or {})}
    if round(sum(weights.values()), 8) != 1.0:
        raise ValueError("Assistant recommendation weights must sum to 1.0")
    haystack = " ".join((product.title, product.category, *product.features.keys(), *product.features.values())).lower()
    requested = requirements.required_features or requirements.features
    feature_score = 1.0 if not requested else sum(feature.lower() in haystack for feature in requested) / len(requested)
    requirement_checks = [requirements.category is None or product.category == requirements.category, requirements.brand is None or product.brand.lower() == requirements.brand.lower(), requirements.budget is None or product.price <= requirements.budget, requirements.minimum_rating is None or product.rating >= requirements.minimum_rating]
    requirement_score = sum(requirement_checks) / len(requirement_checks)
    # Programming/coding favours a laptop with 16 GB RAM; gaming favours an appropriate listed device.
    if requirements.use_case in {"programming", "gaming"}:
        suitable = product.category == "laptops" and (requirements.use_case != "programming" or product.features.get("ram") == "16 GB")
        requirement_score = (requirement_score + float(suitable)) / 2
    components = {"price_score": min(1.0, (requirements.budget or product.price) / product.price), "rating_score": product.rating / 5, "review_score": min(1.0, product.review_count / 10000), "feature_score": feature_score, "requirement_score": requirement_score}
    overall = 100 * (weights["price"] * components["price_score"] + weights["rating"] * components["rating_score"] + weights["reviews"] * components["review_score"] + weights["feature"] * components["feature_score"] + weights["requirement"] * components["requirement_score"])
    return {**{key: round(value, 3) for key, value in components.items()}, "overall_score": round(overall, 2)}


def build_assistant_result(query: str, providers: Iterable[MarketplaceProvider] | None = None, limit: int = 3) -> dict:
    """Search all offline providers and return explainable recommendations and alternatives."""
    requirements = parse_requirements(query)
    active_providers = list(providers or default_providers())
    by_provider = {provider.platform: provider.search_products(requirements) for provider in active_providers}
    listings = [product for results in by_provider.values() for product in results]
    ranked = sorted(((score_normalized_product(product, requirements), product) for product in listings), key=lambda entry: (-entry[0]["overall_score"], -entry[1].rating, entry[1].price))
    groups: dict[str, list[NormalizedProduct]] = defaultdict(list)
    for product in listings:
        groups[f"{product.brand.lower()}::{product.title.lower().replace('(3rd generation)', '3rd gen')}"].append(product)
    best = ranked[0] if ranked else None
    result = {"data_notice": "Offline synthetic marketplace dataset only. Amazon, Flipkart, and Meesho are not queried live.", "requirements": requirements, "provider_counts": {platform: len(found) for platform, found in by_provider.items()}, "found_count": len(listings), "best_overall": _recommendation(best, groups) if best else None, "alternatives": [_recommendation(entry, groups) for entry in ranked[1:limit]], "comparison_table": build_comparison_table(groups), "match_confidences": _group_confidences(groups)}
    return result


def _recommendation(entry, groups: Mapping[str, list[NormalizedProduct]]) -> dict:
    scores, product = entry
    equivalents = next(group for group in groups.values() if product in group)
    lowest = min(equivalents, key=lambda item: item.price)
    reasons = []
    if scores["requirement_score"] >= 0.75: reasons.append("Matches your structured requirements")
    if product.price == lowest.price: reasons.append("Lowest offline mock price among matched listings")
    if product.rating >= 4.3: reasons.append(f"Strong {product.rating:.1f}/5 rating")
    if scores["feature_score"] > 0: reasons.append("Matches requested product features")
    if not reasons: reasons.append("Strong transparent score across price, rating, reviews, and features")
    return {"product": product, "scores": scores, "why": reasons, "matched_listings": equivalents, "price_difference_to_lowest": product.price - lowest.price}


def _group_confidences(groups: Mapping[str, list[NormalizedProduct]]) -> list[float]:
    confidences = []
    for group in groups.values():
        if len(group) > 1:
            confidences.extend(match_products(group[0], item)["confidence"] for item in group[1:])
    return confidences


def build_comparison_table(groups: Mapping[str, list[NormalizedProduct]]) -> list[dict]:
    """Create terminal-friendly rows: Product | Amazon | Flipkart | Meesho | Rating | Best Value."""
    rows = []
    for listings in groups.values():
        if len(listings) < 2: continue
        prices = {item.platform: item.price for item in listings}
        cheapest = min(listings, key=lambda item: item.price)
        rows.append({"product": listings[0].title, "Amazon": prices.get("Amazon"), "Flipkart": prices.get("Flipkart"), "Meesho": prices.get("Meesho"), "rating": max(item.rating for item in listings), "best_value": cheapest.platform})
    return rows
