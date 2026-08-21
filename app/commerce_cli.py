"""Interactive terminal UI for the Phase 8 offline commerce assistant."""

from commerce.assistant import build_assistant_result


def render_result(result: dict) -> str:
    requirements = result["requirements"]
    lines = ["Commerce Assistant", "=" * 18, "Data source: Offline synthetic marketplace dataset.", "", "Parsed requirements:", f"Category: {requirements.category or 'Not specified'}", f"Budget: Rs. {requirements.budget:,}" if requirements.budget else "Budget: Not specified", f"Brand: {requirements.brand or 'Not specified'}", f"Use case: {requirements.use_case or 'Not specified'}", "", "Marketplace results:"]
    lines.extend(f"{platform}: {count} products" for platform, count in result["provider_counts"].items())
    best = result["best_overall"]
    if not best:
        return "\n".join(lines + ["", "No matching mock products found. Try a broader request."])
    product = best["product"]
    lines += ["", "Best Match:", product.title, f"Price: Rs. {product.price:,}", f"Platform: {product.platform}", f"Rating: {product.rating:.1f}/5", f"Score: {best['scores']['overall_score']}", "Why this product?"]
    lines.extend(f"- {reason}" for reason in best["why"])
    if result["alternatives"]:
        lines += ["", "Alternatives:"]
        lines.extend(f"{index}. {item['product'].title} | Rs. {item['product'].price:,} | {item['product'].platform}" for index, item in enumerate(result["alternatives"], 1))
    return "\n".join(lines)


def main() -> None:
    print("Commerce Assistant (offline mock data; type 'quit' to exit)")
    while True:
        try: query = input("\nShopping request: ").strip()
        except EOFError: break
        if query.lower() in {"quit", "exit"}: break
        if not query:
            print("Please enter a shopping request.")
            continue
        print(render_result(build_assistant_result(query)))


if __name__ == "__main__": main()
