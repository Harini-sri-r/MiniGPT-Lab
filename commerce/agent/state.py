"""Explicit, inspectable state for one offline shopping decision session."""

from dataclasses import dataclass, field

from ..models import NormalizedProduct, UserRequirements


@dataclass
class ShoppingState:
    query: str = ""
    requirements: UserRequirements | None = None
    plan: list[str] = field(default_factory=list)
    search_results: list[NormalizedProduct] = field(default_factory=list)
    filtered_results: list[NormalizedProduct] = field(default_factory=list)
    matched_products: list[list[NormalizedProduct]] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    selected_product: dict | None = None
    alternatives: dict[str, dict] = field(default_factory=dict)
    reasoning_steps: list[dict] = field(default_factory=list)
    status: str = "new"
    relaxed_constraints: list[str] = field(default_factory=list)

    def record(self, step: str, status: str, result=None) -> None:
        self.reasoning_steps.append({"step": step, "status": status, "result": result})
