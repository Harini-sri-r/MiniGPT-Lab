"""Public lifecycle and follow-up interface for the Commerce Decision Agent."""

from dataclasses import replace
import re

from ..parser import parse_requirements
from ..providers import MarketplaceProvider, default_providers
from .executor import execute_plan
from .planner import create_plan
from .state import ShoppingState


class CommerceDecisionAgent:
    """A deterministic offline shopping decision agent; it cannot purchase anything."""

    def __init__(self, providers: list[MarketplaceProvider] | None = None, trace: bool = False):
        self.providers = providers or default_providers()
        self.trace = trace
        self.state = ShoppingState()

    def run(self, query: str) -> ShoppingState:
        requirements = parse_requirements(query)
        self.state = ShoppingState(query=query, requirements=requirements, plan=create_plan(requirements), status="planned")
        return execute_plan(self.state, self.providers)

    def follow_up(self, query: str) -> ShoppingState:
        """Update prior requirements and re-run the same decision process explicitly."""
        if self.state.requirements is None:
            return self.run(query)
        update = query.lower().strip()
        if update in {"which one is cheapest?", "which is cheapest?", "cheapest"}:
            if self.state.alternatives: self.state.selected_product = self.state.alternatives["cheapest_valid"]
            self.state.record("follow_up_cheapest", "success", self.state.selected_product["product"].product_id if self.state.selected_product else None)
            return self.state
        parsed = parse_requirements(query)
        old = self.state.requirements
        amount = re.search(r"(?:increase|change|update)\s+(?:the\s+)?budget\s+(?:to|of)?\s*(?:rs\.?\s*)?([\d,]+)\s*(k)?", update)
        follow_up_budget = int(amount.group(1).replace(",", "")) * (1000 if amount and amount.group(2) else 1) if amount else None
        budget = follow_up_budget if follow_up_budget is not None else (parsed.budget if parsed.budget is not None else old.budget)
        brand = parsed.brand if parsed.brand is not None else old.brand
        minimum_rating = parsed.minimum_rating if parsed.minimum_rating is not None else old.minimum_rating
        category = parsed.category if parsed.category is not None else old.category
        use_case = parsed.use_case if parsed.use_case is not None else old.use_case
        features = old.required_features
        if update.startswith("remove "):
            feature = update.removeprefix("remove ").strip()
            features = tuple(item for item in features if item.lower() != feature)
        elif parsed.required_features:
            features = tuple(dict.fromkeys((*features, *parsed.required_features)))
        requirements = replace(old, query=query, category=category, budget=budget, brand=brand, minimum_rating=minimum_rating, use_case=use_case, features=features, required_features=features)
        self.state = ShoppingState(query=query, requirements=requirements, plan=create_plan(requirements), status="planned")
        self.state.record("apply_follow_up", "success", {"budget": budget, "brand": brand, "features": features})
        return execute_plan(self.state, self.providers)

    def trace_lines(self) -> list[str]:
        return [f"[{index}] {step['step']}: {step['status']}" for index, step in enumerate(self.state.reasoning_steps, 1)]
