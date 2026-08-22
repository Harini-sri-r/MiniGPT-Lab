"""Interactive CLI for the explicit offline Commerce Decision Agent."""

import argparse

from commerce.agent import CommerceDecisionAgent
from commerce.agent.reporting import decision_trace, no_result_guidance, rich_comparisons, score_breakdown


def render_state(state) -> str:
    if state.status == "invalid_request": return f"Agent: {state.reasoning_steps[-1]['result']}"
    if state.status == "no_results": return "Agent: No product satisfies all your hard requirements.\nSuggestions:\n" + "\n".join(f"- {item}" for item in no_result_guidance(state))
    selected = state.selected_product
    product = selected["product"]
    breakdown = score_breakdown(selected)
    lines = ["Understanding your requirements...", "Searching offline Amazon, Flipkart and Meesho mock providers...", f"Found {len(state.filtered_results)} valid candidates.", "", f"Recommendation: {product.title}", f"Rs. {product.price:,} - {product.platform}", f"Rating: {product.rating:.1f}/5", f"Score: {breakdown['final_score']}/100", "Reason:"]
    lines.extend(f"- {reason}" for reason in selected["why"])
    lines += ["Alternatives:"]
    for label, item in state.alternatives.items(): lines.append(f"- {label}: {item['product'].title} ({item['product'].platform}, Rs. {item['product'].price:,})")
    lines += ["", "Top comparison rows:"]
    lines.extend(f"- {row['product_name']} | {row['platform']} | Rs. {row['price']:,} | rating {row['rating']:.1f} | value {row['value_score']}" for row in rich_comparisons(state, limit=1))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    agent = CommerceDecisionAgent(trace=args.trace)
    print("MiniGPT Commerce Agent\n=======================\nOffline synthetic data; type exit to quit")
    while True:
        try: query = input("\nYou: ").strip()
        except EOFError: break
        if query.lower() in {"exit", "quit"}: break
        if query.lower() == "trace on": agent.trace = True; print("Trace enabled."); continue
        if query.lower() == "trace off": agent.trace = False; print("Trace disabled."); continue
        state = agent.run(query) if agent.state.status in {"new", "invalid_request"} else agent.follow_up(query)
        print("\nAgent:\n" + render_state(state))
        if agent.trace: print("\nDecision trace:\n" + "\n".join(f"[{item['step_number']}] {item['action']}: {item['status']}" for item in decision_trace(state)))


if __name__ == "__main__": main()
