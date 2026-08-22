"""Fifteen deterministic offline scenarios for Commerce Phase 9."""

from commerce.agent import CommerceDecisionAgent

SCENARIOS = (
    "laptop under Rs. 60000", "phone under Rs. 30000", "Samsung phone", "gaming laptop", "programming laptop", "phone with good camera", "laptop under Rs. 1", "laptop with 16GB RAM", "Apple laptop under Rs. 1000", "", "laptop under Rs. 60000 for programming", "What about HP?", "Which one is cheapest?", "best rated laptop", "Increase budget to 70000",
)


def evaluate() -> dict:
    agent = CommerceDecisionAgent(trace=True)
    states = []
    for scenario in SCENARIOS:
        state = agent.follow_up(scenario) if scenario in {"What about HP?", "Which one is cheapest?", "Increase budget to 70000"} else agent.run(scenario)
        states.append(state)
    executable = [state for state in states if state.status == "success"]
    metrics = {"planning_success_rate": sum(bool(state.plan) for state in states) / len(states), "execution_success_rate": sum(state.status in {"success", "no_results", "invalid_request"} for state in states) / len(states), "hard_constraint_satisfaction_rate": sum(not any("filter_hard_constraints" == step["step"] and step["status"] != "success" for step in state.reasoning_steps) for state in executable) / max(1, len(executable)), "recommendation_validity_rate": sum(state.selected_product is not None for state in executable) / max(1, len(executable)), "follow_up_consistency": float(states[11].requirements.brand == "HP" and states[12].selected_product == states[12].alternatives.get("cheapest_valid")), "explanation_factual_consistency": sum(bool(state.selected_product and state.selected_product["why"]) for state in executable) / max(1, len(executable))}
    return {"scenarios": states, "metrics": metrics}


def main() -> None:
    report = evaluate()
    for index, state in enumerate(report["scenarios"], 1): print(f"{index}. {state.query!r}: {state.status}; selected={state.selected_product['product'].title if state.selected_product else None}")
    print("Offline synthetic metrics (not real-world accuracy):", report["metrics"])


if __name__ == "__main__": main()
