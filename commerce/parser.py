"""Rule-based user-requirement parser; it does not use an LLM or external API."""

import re

from .models import UserRequirements

_CATEGORIES = {
    "headphone": "headphones", "headphones": "headphones", "earphone": "headphones", "earbuds": "headphones",
    "smartphone": "smartphones", "phone": "smartphones", "mobile": "smartphones",
    "laptop": "laptops", "notebook": "laptops",
    "smartwatch": "smartwatches", "watch": "smartwatches",
    "keyboard": "keyboards", "mouse": "mice", "mice": "mice", "monitor": "monitors", "display": "monitors",
}
_FEATURE_PATTERNS = ("wireless", "good battery life", "battery life", "mechanical", "gaming", "amoled", "rgb")
_USE_CASES = ("programming", "gaming", "work", "study")


def parse_requirements(query: str) -> UserRequirements:
    """Extract a category, INR budget, simple feature phrases, and use case."""
    lowered = query.lower()
    category = next((value for keyword, value in _CATEGORIES.items() if re.search(rf"\b{keyword}s?\b", lowered)), None)
    budget_match = re.search(r"(?:under|below|less than|within)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)", lowered)
    if not budget_match:
        budget_match = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)", lowered)
    budget = int(budget_match.group(1).replace(",", "")) if budget_match else None
    features = tuple(pattern for pattern in _FEATURE_PATTERNS if pattern in lowered and not (pattern == "battery life" and "good battery life" in lowered))
    use_case = next((item for item in _USE_CASES if item in lowered), None)
    return UserRequirements(query=query, category=category, budget=budget, features=features, use_case=use_case)
