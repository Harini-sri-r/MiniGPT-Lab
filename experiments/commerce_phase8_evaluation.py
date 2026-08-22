"""Evaluate Phase 8 deterministic offline commerce behavior on ten sample queries."""

from commerce.assistant import build_assistant_result

QUERIES = (
    "Best laptop under Rs. 60000 for programming", "Samsung phone under Rs. 30000", "Gaming laptop under Rs. 80000",
    "Phone under Rs. 20000 with good camera", "Laptop with 16GB RAM", "Budget phone under Rs. 15000",
    "Best rated laptop", "Cheapest laptop under Rs. 50000", "HP laptop for students", "Laptop for coding under Rs. 70000",
)


def evaluate() -> dict:
    reports = []
    for query in QUERIES:
        result = build_assistant_result(query)
        best = result["best_overall"]
        reports.append({"query": query, "parsed": result["requirements"], "matches": result["found_count"], "recommended": best["product"].title if best else None, "platform": best["product"].platform if best else None, "price": best["product"].price if best else None, "score": best["scores"]["overall_score"] if best else None, "reason": best["why"][0] if best else None, "confidence": sum(result["match_confidences"]) / len(result["match_confidences"]) if result["match_confidences"] else 0.0})
    parsing_success = sum(report["parsed"].category is not None for report in reports) / len(reports)
    requirement_match = sum(report["recommended"] is not None for report in reports) / len(reports)
    consistency = sum(build_assistant_result(report["query"])["best_overall"]["product"].product_id == build_assistant_result(report["query"])["best_overall"]["product"].product_id for report in reports if report["recommended"]) / max(1, sum(report["recommended"] is not None for report in reports))
    return {"reports": reports, "metrics": {"query_parsing_success_rate": parsing_success, "product_matching_confidence": sum(report["confidence"] for report in reports) / len(reports), "requirement_match_rate": requirement_match, "recommendation_consistency": consistency}}


def main() -> None:
    evaluation = evaluate()
    for report in evaluation["reports"]:
        print(f"{report['query']}\n  parsed={report['parsed']}\n  matches={report['matches']} | recommendation={report['recommended']} | {report['platform']} | {report['price']} | score={report['score']}\n  reason={report['reason']}")
    print("\nOffline deterministic metrics (not real-world accuracy):", evaluation["metrics"])


if __name__ == "__main__": main()
