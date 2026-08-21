"""Tests for explicit Phase 9 offline shopping-agent planning and decisions."""

from commerce.agent import CommerceDecisionAgent, ShoppingState
from commerce.agent.executor import hard_constraint_failures
from commerce.agent.planner import create_plan
from commerce.parser import parse_requirements


def test_state_and_explicit_plan():
    state = ShoppingState(query="laptop under Rs. 60000")
    assert state.status == "new" and state.reasoning_steps == []
    plan = create_plan(parse_requirements(state.query))
    assert plan[0] == "understand_requirements" and "search_amazon" in plan and plan[-1] == "select_alternatives"


def test_agent_lifecycle_records_stepwise_execution_and_trace():
    agent = CommerceDecisionAgent(trace=True)
    state = agent.run("I need a laptop under Rs. 60000 for programming")
    assert state.status == "success" and state.selected_product and len(state.reasoning_steps) >= 10
    assert any("search_amazon" in line for line in agent.trace_lines())


def test_validation_invalid_and_empty_requests_fail_gracefully():
    agent = CommerceDecisionAgent()
    assert agent.run("I need a phone under Rs. 0").status == "invalid_request"
    assert agent.run("I need something").status == "invalid_request"
    assert agent.run("").status == "invalid_request"


def test_hard_constraints_are_not_silently_violated():
    agent = CommerceDecisionAgent()
    state = agent.run("Samsung phone under Rs. 30000")
    assert state.status == "success"
    assert all(not hard_constraint_failures(item, state) for item in state.filtered_results)
    impossible = agent.run("Apple laptop under Rs. 1000")
    assert impossible.status == "no_results" and impossible.relaxed_constraints


def test_required_ram_is_hard_and_soft_camera_affects_explanation():
    agent = CommerceDecisionAgent()
    ram = agent.run("laptop with 16GB RAM")
    assert ram.status == "success" and all(item.features.get("ram") == "16 GB" for item in ram.filtered_results)
    camera = agent.run("phone under Rs. 20000 with good camera")
    assert camera.status == "success" and camera.selected_product["scores"]["feature_score"] > 0


def test_alternatives_and_explanation_are_data_based():
    state = CommerceDecisionAgent().run("laptop under Rs. 60000")
    assert set(state.alternatives) == {"best_overall", "cheapest_valid", "highest_rated_valid", "best_feature_match"}
    assert "Rating is" in " ".join(state.selected_product["why"])


def test_follow_up_budget_brand_and_cheapest_state_updates():
    agent = CommerceDecisionAgent()
    agent.run("laptop under Rs. 60000 for programming")
    hp = agent.follow_up("What about HP?")
    assert hp.requirements.brand == "HP" and hp.status == "success"
    updated = agent.follow_up("Increase budget to 70000")
    assert updated.requirements.budget == 70000
    cheapest = agent.follow_up("Which one is cheapest?")
    assert cheapest.selected_product == cheapest.alternatives["cheapest_valid"]


def test_determinism_and_no_result_trace():
    first = CommerceDecisionAgent().run("laptop under Rs. 60000")
    second = CommerceDecisionAgent().run("laptop under Rs. 60000")
    assert first.selected_product["product"].product_id == second.selected_product["product"].product_id
    none = CommerceDecisionAgent().run("laptop under Rs. 1")
    assert none.status == "no_results" and any(step["status"] == "no_results" for step in none.reasoning_steps)
