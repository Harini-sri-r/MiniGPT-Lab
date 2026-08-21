"""Phase 10 tests for factual reporting and polished offline agent behavior."""

from app.commerce_agent_cli import render_state
from commerce.agent import CommerceDecisionAgent
from commerce.agent.reporting import decision_trace, no_result_guidance, rich_comparisons, score_breakdown
from commerce.providers import AmazonMockProvider


def _state():
    return CommerceDecisionAgent(trace=True).run("laptop under Rs. 60000 for programming")


def test_rich_comparison_exposes_required_factual_fields():
    state = _state()
    rows = rich_comparisons(state)
    assert rows and {"product_name", "brand", "category", "platform", "price", "rating", "review_count", "relevant_features", "requirement_match", "price_difference", "rating_difference", "value_score"} <= set(rows[0])
    assert len(rows) >= 3


def test_score_breakdown_is_complete_deterministic_and_weighted():
    state = _state()
    first, second = score_breakdown(state.selected_product), score_breakdown(state.selected_product)
    assert first == second and round(sum(first["weights"].values()), 8) == 1.0
    assert {"price_score", "rating_score", "review_score", "feature_score", "requirement_score", "final_score"} <= set(first)


def test_explanation_is_factual_contains_tradeoff_and_alternative_reason():
    state = _state()
    explanation = " ".join(state.selected_product["why"])
    assert "within the Rs. 60,000 budget" in explanation and f"Rating is {state.selected_product['product'].rating:.1f}/5" in explanation
    assert "Alternative ranks lower" in explanation


def test_decision_trace_is_concise_structured_metadata():
    trace = decision_trace(_state())
    assert trace[0]["action"] == "Understand request"
    assert any(item["action"] == "Search provider" for item in trace)
    assert all({"step_number", "action", "status", "metadata"} == set(item) for item in trace)


def test_follow_ups_only_brand_budget_feature_and_cheapest_preserve_state():
    agent = CommerceDecisionAgent()
    agent.run("laptop under Rs. 60000")
    assert agent.follow_up("Only HP").requirements.brand == "HP"
    assert agent.follow_up("Make the budget 70000").requirements.budget == 70000
    assert "16 GB RAM" in agent.follow_up("What if I need 16GB RAM?").requirements.required_features
    state = agent.follow_up("Which is cheapest?")
    assert state.selected_product == state.alternatives["cheapest_valid"]


def test_no_result_has_factual_guidance_without_invalid_recommendation():
    state = CommerceDecisionAgent().run("Apple laptop under Rs. 1000")
    assert state.status == "no_results" and state.selected_product is None
    assert no_result_guidance(state)


def test_provider_contract_availability_price_and_cli_rendering():
    provider = AmazonMockProvider()
    product = provider.search_products(_state().requirements)[0]
    assert provider.availability(product.product_id) == product.availability and provider.price(product.product_id) == product.price
    rendered = render_state(_state())
    assert "MiniGPT" not in rendered and "Score:" in rendered and "Top comparison rows:" in rendered


def test_phase10_evaluation_metrics_are_reported():
    from experiments.commerce_phase10_evaluation import evaluate
    metrics = evaluate()["metrics"]
    assert len(evaluate()["states"]) == 20
    assert metrics["follow_up_consistency"] == 1.0 and metrics["no_result_correctness"] == 1.0
