"""Deterministic planner that makes the agent workflow visible before execution."""

from ..models import UserRequirements


def create_plan(requirements: UserRequirements) -> list[str]:
    """Return an explicit, stable shopping plan for a valid structured request."""
    plan = ["understand_requirements", "validate_requirements", "search_amazon", "search_flipkart", "search_meesho", "filter_hard_constraints", "match_equivalent_products", "compare_candidates", "score_valid_candidates", "select_recommendation", "generate_explanation", "select_alternatives"]
    if requirements.category is None:
        return plan[:2]
    return plan
