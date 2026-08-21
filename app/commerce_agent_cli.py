"""Interactive CLI for the explicit offline Commerce Decision Agent."""

import argparse

from commerce.agent import CommerceDecisionAgent


def render_state(state) -> str:
    if state.status == "invalid_request": return f"Agent: {state.reasoning_steps[-1]['result']}"
    if state.status == "no_results": return "Agent: No product satisfies all your hard requirements. Closest alternatives would relax: " + ", ".join(state.relaxed_constraints or ["no supported constraint"])
    selected = state.selected_product
    product = selected["product"]
    lines = [f"Recommendation: {product.title}", f"Rs. {product.price:,} - {product.platform}", f"Rating: {product.rating:.1f}/5", "Reason:"]
    lines.extend(f"- {reason}" for reason in selected["why"])
    lines += ["Alternatives:"]
    for label, item in state.alternatives.items(): lines.append(f"- {label}: {item['product'].title} ({item['product'].platform}, Rs. {item['product'].price:,})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    agent = CommerceDecisionAgent(trace=args.trace)
    print("Commerce AI Agent (offline synthetic data; type exit to quit)")
    while True:
        try: query = input("\nYou: ").strip()
        except EOFError: break
        if query.lower() in {"exit", "quit"}: break
        if query.lower() == "trace on": agent.trace = True; print("Trace enabled."); continue
        if query.lower() == "trace off": agent.trace = False; print("Trace disabled."); continue
        state = agent.run(query) if agent.state.status in {"new", "invalid_request"} else agent.follow_up(query)
        print("\nAgent:\n" + render_state(state))
        if agent.trace: print("\n" + "\n".join(agent.trace_lines()))


if __name__ == "__main__": main()
