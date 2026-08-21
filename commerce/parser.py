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
_FEATURE_PATTERNS = ("wireless", "good battery life", "battery life", "good camera", "camera", "16gb ram", "16 gb ram", "mechanical", "gaming", "amoled", "rgb")
_USE_CASES = ("programming", "coding", "gaming", "students", "student", "work", "study")
_BRANDS = ("Acer", "Apple", "ASUS", "BenQ", "boAt", "Dell", "Fastrack", "HP", "JBL", "Lenovo", "LG", "Logitech", "Motorola", "MSI", "Noise", "Razer", "Realme", "Redragon", "Samsung", "Sony", "Xiaomi", "Zebronics")


def parse_requirements(query: str) -> UserRequirements:
    """Extract a category, INR budget, simple feature phrases, and use case."""
    lowered = query.lower()
    category = next((value for keyword, value in _CATEGORIES.items() if re.search(rf"\b{keyword}s?\b", lowered)), None)
    budget_match = re.search(r"(?:under|below|less than|within)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)", lowered)
    if not budget_match:
        budget_match = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)", lowered)
    budget = int(budget_match.group(1).replace(",", "")) if budget_match else None
    shorthand = re.search(r"(?:under|below|within)\s*(\d+)k\b", lowered)
    if shorthand:
        budget = int(shorthand.group(1)) * 1000
    raw_features = tuple(item for item in _FEATURE_PATTERNS if item in lowered and not (item == "battery life" and "good battery life" in lowered))
    features = tuple("16 GB RAM" if item in {"16gb ram", "16 gb ram"} else item for item in raw_features)
    use_case = next(("programming" if item == "coding" else "study" if item in {"students", "student"} else item for item in _USE_CASES if item in lowered), None)
    brand = next((brand for brand in _BRANDS if re.search(rf"\b{re.escape(brand.lower())}\b", lowered)), None)
    rating_match = re.search(r"(?:at least|min(?:imum)?|above)\s*([0-5](?:\.\d)?)\s*(?:star|rating)", lowered)
    minimum_rating = float(rating_match.group(1)) if rating_match else None
    return UserRequirements(query=query, category=category, budget=budget, features=features, required_features=features, use_case=use_case, brand=brand, minimum_rating=minimum_rating)
