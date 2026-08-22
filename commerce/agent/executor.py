"""Step-by-step executor for deterministic offline commerce decisions."""

from collections import defaultdict
from dataclasses import replace

from ..assistant import score_normalized_product
from ..matching import match_products
from ..models import NormalizedProduct
from ..providers import MarketplaceProvider
from .state import ShoppingState

_SOFT_FEATURES = {"good battery life", "battery life", "good camera", "camera", "gaming"}


def validate_requirements(state: ShoppingState) -> tuple[str, str]:
    req = state.requirements
    if not state.query.strip() or req is None or req.category is None:
        return "invalid_request", "Please specify a product category, such as laptop or phone."
    if req.budget is not None and req.budget <= 0:
        return "invalid_request", "Budget must be greater than zero."
    return "success", "Requirements are valid."


def hard_constraint_failures(product: NormalizedProduct, state: ShoppingState) -> list[str]:
    req = state.requirements
    assert req is not None
    failures = []
    if product.category != req.category: failures.append("category")
    if req.budget is not None and product.price > req.budget: failures.append("budget")
    if req.brand and product.brand.lower() != req.brand.lower(): failures.append("brand")
    if req.minimum_rating is not None and product.rating < req.minimum_rating: failures.append("minimum_rating")
    haystack = " ".join((*product.features.keys(), *product.features.values())).lower()
    hard_features = [feature for feature in req.required_features if feature.lower() not in _SOFT_FEATURES]
    failures.extend(f"feature:{feature}" for feature in hard_features if not _feature_matches(feature, product, haystack))
    return failures


def execute_plan(state: ShoppingState, providers: list[MarketplaceProvider]) -> ShoppingState:
    """Execute a stated plan, recording one structured outcome for every step.

    For Phase 11 we use the orchestrator to aggregate results across providers and
    capture data‑source metadata. The ``providers`` argument is retained for backward
    compatibility but is ignored; the orchestrator determines providers based on the
    environment configuration.
    """
    status, message = validate_requirements(state)
    state.record("understand_requirements", "success", state.requirements)
    state.record("validate_requirements", status, message)
    if status != "success":
        state.status = status
        return state

    # Phase 11: Use orchestrator to perform the search.
    from ..orchestrator import orchestrate_search
    from ..providers import configured_providers

    # Obtain providers according to configuration (mock by default).
    providers = configured_providers()
    # Orchestrate the search.
    search_result = orchestrate_search(state.requirements, config=None)

    # Record per‑provider outcomes.
    for prov, info in search_result.provider_results.items():
        state.record(
            f"search_{prov.lower()}",
            "success" if info.success else "failure",
            {"count": info.result_count, "error": info.error},
        )

    # Record overall data source summary.
    state.record("data_source_summary", "info", {
        "summary": search_result.data_source_summary,
        "notice": search_result.data_notice,
    })

    all_results: list[NormalizedProduct] = search_result.products
    state.search_results = all_results
    # Continue with existing filtering, matching, and recommendation logic.
    valid = [product for product in all_results if not hard_constraint_failures(product, state)]
    state.filtered_results = valid
    state.record("filter_hard_constraints", "success" if valid else "no_results", {"before": len(all_results), "after": len(valid)})
    if not valid:
        state.status = "no_results"
        state.relaxed_constraints = _closest_relaxed_constraints(all_results, state)
        state.record("select_recommendation", "no_results", "No product satisfies all hard requirements.")
        return state
    groups = _group_equivalents(valid)
    state.matched_products = list(groups.values())
    state.record("match_equivalent_products", "success", {"groups": len(groups), "confidence": _mean_confidence(state.matched_products)})
    state.comparisons = [_comparison(group) for group in state.matched_products]
    state.record("compare_candidates", "success", {"comparisons": len(state.comparisons)})
    state.recommendations = sorted((_recommendation(product, state) for product in valid), key=lambda item: (-item["scores"]["overall_score"], -item["product"].rating, item["product"].price))
    state.record("score_valid_candidates", "success", {"candidates": len(state.recommendations)})
    state.selected_product = state.recommendations[0]
    state.record("select_recommendation", "success", state.selected_product["product"].product_id)
    state.selected_product["why"] = _explain(state.selected_product, state)
    state.record("generate_explanation", "success", state.selected_product["why"])
    state.alternatives = _alternatives(state)
    state.record("select_alternatives", "success", {key: item["product"].product_id for key, item in state.alternatives.items()})
    state.status = "success"
    return state


def _group_equivalents(products: list[NormalizedProduct]) -> dict[str, list[NormalizedProduct]]:
    groups: dict[str, list[NormalizedProduct]] = defaultdict(list)
    for product in products:
        groups[f"{product.brand.lower()}::{product.title.lower().replace('(3rd generation)', '3rd gen')}"] .append(product)
    return dict(groups)


def _comparison(group: list[NormalizedProduct]) -> dict:
    lowest = min(group, key=lambda product: product.price)
    highest_rated = max(group, key=lambda product: product.rating)
    return {"title": group[0].title, "listings": group, "cheapest": lowest, "highest_rated": highest_rated, "price_difference": max(item.price for item in group) - lowest.price, "rating_difference": round(highest_rated.rating - min(item.rating for item in group), 1)}


def _recommendation(product: NormalizedProduct, state: ShoppingState) -> dict:
    return {"product": product, "scores": score_normalized_product(product, state.requirements), "why": []}


def _explain(recommendation: dict, state: ShoppingState) -> list[str]:
    product, req = recommendation["product"], state.requirements
    lines = []
    if req.budget is not None: lines.append(f"Price Rs. {product.price:,} is within the Rs. {req.budget:,} budget.")
    if req.use_case: lines.append(f"It is evaluated for your {req.use_case} use case.")
    lines.append(f"Rating is {product.rating:.1f}/5 from {product.review_count:,} synthetic reviews.")
    matched = [item for item in req.required_features if _feature_matches(item, product)]
    if matched: lines.append("Matches required features: " + ", ".join(matched) + ".")
    cheaper = min(state.recommendations, key=lambda item: item["product"].price)
    if cheaper["product"] != product:
        lines.append(f"Trade-off: {cheaper['product'].title} on {cheaper['product'].platform} is Rs. {product.price - cheaper['product'].price:,} cheaper but scores lower overall.")
    lower_ranked = next((item for item in state.recommendations[1:] if item["product"] != product), None)
    if lower_ranked:
        lines.append(f"Alternative ranks lower with a {lower_ranked['scores']['overall_score']}/100 score versus {recommendation['scores']['overall_score']}/100.")
    lines.append(f"Transparent score: {recommendation['scores']['overall_score']}/100 among valid candidates.")
    return lines


def _alternatives(state: ShoppingState) -> dict[str, dict]:
    candidates = state.recommendations
    return {"best_overall": candidates[0], "cheapest_valid": min(candidates, key=lambda item: item["product"].price), "highest_rated_valid": max(candidates, key=lambda item: (item["product"].rating, item["product"].review_count)), "best_feature_match": max(candidates, key=lambda item: item["scores"]["feature_score"])}


def _closest_relaxed_constraints(products: list[NormalizedProduct], state: ShoppingState) -> list[str]:
    if not products:
        return []
    nearest = min(products, key=lambda product: len(hard_constraint_failures(product, state)))
    failures = hard_constraint_failures(nearest, state)
    # Ensure at least one constraint is reported for user feedback.
    return failures if failures else ["unsatisfied_constraints"]
    if not products: return []
    nearest = min(products, key=lambda product: len(hard_constraint_failures(product, state)))
    return hard_constraint_failures(nearest, state)


def _feature_matches(feature: str, product: NormalizedProduct, haystack: str | None = None) -> bool:
    normalized = feature.lower().replace(" ", "")
    if normalized in {"16gbram", "8gbram"}:
        return normalized.removesuffix("ram") in product.features.get("ram", "").lower().replace(" ", "")
    return feature.lower() in (haystack or " ".join((*product.features.keys(), *product.features.values())).lower())


def _mean_confidence(groups: list[list[NormalizedProduct]]) -> float:
    values = [match_products(group[0], item)["confidence"] for group in groups for item in group[1:]]
    return round(sum(values) / len(values), 2) if values else 1.0
