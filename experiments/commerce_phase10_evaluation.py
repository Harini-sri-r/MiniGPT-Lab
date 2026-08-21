"""Twenty deterministic offline scenarios for Phase 10 decision quality checks."""

from commerce.agent import CommerceDecisionAgent
from commerce.agent.reporting import decision_trace

SCENARIOS = (
    "laptop under Rs. 60000", "Samsung phone under Rs. 30000", "phone under Rs. 20000 with good camera",
    "laptop with 16GB RAM", "laptop at least 4.3 rating", "HP laptop for students", "gaming laptop under Rs. 80000",
    "laptop under Rs. 1", "Apple laptop under Rs. 1000", "phone under Rs. 0", "", "best rated laptop",
    "laptop under Rs. 60000 for programming", "Only HP", "Make the budget 70000", "What if I need 16GB RAM?",
    "Which is cheapest?", "Compare the best 3 laptops for programming under Rs. 60000", "best value laptop", "Samsung phone at least 4.3 rating",
)


def evaluate() -> dict:
    agent = CommerceDecisionAgent(trace=True)
    states = []
    follow_ups = {"Only HP", "Make the budget 70000", "What if I need 16GB RAM?", "Which is cheapest?"}
    for query in SCENARIOS:
        states.append(agent.follow_up(query) if query in follow_ups else agent.run(query))
    success = [state for state in states if state.status == "success"]
    metrics = {
        "parsing_success": sum(state.requirements.category is not None for state in states if state.query) / max(1, sum(bool(state.query) for state in states)),
        "hard_constraint_satisfaction": sum(all(product.price <= state.requirements.budget for product in state.filtered_results) for state in success if state.requirements.budget is not None) / max(1, sum(state.requirements.budget is not None for state in success)),
        "recommendation_validity": sum(state.selected_product is not None for state in success) / max(1, len(success)),
        "follow_up_consistency": float(states[13].requirements.brand == "HP" and states[14].requirements.budget == 70000 and "16 GB RAM" in states[15].requirements.required_features and states[16].selected_product == states[16].alternatives["cheapest_valid"]),
        "explanation_factual_consistency": sum(bool(state.selected_product and state.selected_product["why"] and all("Rs." in line or "Rating" in line or "evaluated" in line or "Matches" in line or "score" in line or "Trade-off" in line or "Alternative" in line for line in state.selected_product["why"])) for state in success) / max(1, len(success)),
        "no_result_correctness": sum(state.status in {"no_results", "invalid_request"} for state in (states[7], states[8], states[9], states[10])) / 4,
        "trace_completion": sum(bool(decision_trace(state)) for state in states) / len(states),
    }
    return {"states": states, "metrics": metrics}


def main() -> None:
    report = evaluate()
    for index, state in enumerate(report["states"], 1): print(f"{index}. {state.query!r}: {state.status}; candidate_count={len(state.filtered_results)}")
    print("Offline synthetic metrics (not real-world marketplace accuracy):", report["metrics"])


if __name__ == "__main__": main()
