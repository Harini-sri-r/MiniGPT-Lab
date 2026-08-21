"""Concise factual reporting for Phase 10; this is not hidden reasoning."""

from ..assistant import DEFAULT_ASSISTANT_WEIGHTS
from .state import ShoppingState


def score_breakdown(recommendation: dict) -> dict:
    """Expose deterministic score components and configured weights."""
    scores = recommendation["scores"]
    return {"weights": dict(DEFAULT_ASSISTANT_WEIGHTS), "price_score": scores["price_score"], "rating_score": scores["rating_score"], "review_score": scores["review_score"], "feature_score": scores["feature_score"], "requirement_score": scores["requirement_score"], "final_score": scores["overall_score"]}


def rich_comparisons(state: ShoppingState, limit: int = 3) -> list[dict]:
    """Return factual rows for the strongest matched product groups."""
    rows = []
    for comparison in state.comparisons[:limit]:
        listings = comparison["listings"]
        highest_rating = max(item.rating for item in listings)
        for product in listings:
            recommendation = next(item for item in state.recommendations if item["product"] == product)
            req = state.requirements
            feature_values = {key: value for key, value in product.features.items() if key in {"ram", "processor", "battery_life", "battery", "camera", "storage", "refresh_rate"}}
            rows.append({"product_name": product.title, "brand": product.brand, "category": product.category, "platform": product.platform, "price": product.price, "rating": product.rating, "review_count": product.review_count, "relevant_features": feature_values, "requirement_match": recommendation["scores"]["requirement_score"], "price_difference": product.price - comparison["cheapest"].price, "rating_difference": round(highest_rating - product.rating, 1), "value_score": recommendation["scores"]["overall_score"]})
    return rows


def decision_trace(state: ShoppingState) -> list[dict]:
    """Expose concise action metadata, never private chain-of-thought."""
    labels = {"understand_requirements": "Understand request", "validate_requirements": "Validate requirements", "filter_hard_constraints": "Apply hard constraints", "match_equivalent_products": "Match equivalent products", "compare_candidates": "Compare candidates", "score_valid_candidates": "Calculate recommendation scores", "select_recommendation": "Select recommendation", "generate_explanation": "Generate factual explanation", "select_alternatives": "Select alternatives"}
    trace = []
    for index, item in enumerate(state.reasoning_steps, 1):
        label = labels.get(item["step"], "Search provider" if item["step"].startswith("search_") else item["step"])
        trace.append({"step_number": index, "action": label, "status": item["status"], "metadata": item["result"] if isinstance(item["result"], (str, int, float, dict)) else None})
    return trace


def no_result_guidance(state: ShoppingState) -> list[str]:
    """Generate only useful actions grounded in failed hard constraints."""
    guidance = []
    for constraint in state.relaxed_constraints:
        if constraint == "budget": guidance.append("Increase your budget.")
        elif constraint == "brand": guidance.append("Remove or change the brand restriction.")
        elif constraint == "minimum_rating": guidance.append("Relax the minimum rating.")
        elif constraint.startswith("feature:"): guidance.append(f"Remove or relax {constraint.removeprefix('feature:')}.")
    return guidance or ["Try a broader category or a higher budget."]
