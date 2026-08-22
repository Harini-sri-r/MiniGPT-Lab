"""Deterministic product-name normalization and matching."""

import re

_STOP_WORDS = {"the", "wireless", "headphones", "headphone", "generation", "gen"}
_ALIASES = {
    "airpods 3": "apple airpods 3",
    "apple airpods 3rd": "apple airpods 3",
    "apple airpods 3rd generation": "apple airpods 3",
}


def normalize_product_name(name: str) -> str:
    """Produce a stable, simple grouping key without ML or external services."""
    text = name.lower()
    text = re.sub(r"\bthird\b", "3", text)
    text = re.sub(r"\b3rd\b", "3", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = " ".join(token for token in text.split() if token not in _STOP_WORDS)
    return _ALIASES.get(text, text)


def products_match(left_name: str, right_name: str) -> bool:
    """Determine if two listings are the same after deterministic normalization."""
    return normalize_product_name(left_name) == normalize_product_name(right_name)
