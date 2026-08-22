"""Deterministic, explainable likely-equivalence matching for Phase 8."""

import re

from .models import NormalizedProduct
from .normalize import normalize_product_name


def match_products(left: NormalizedProduct, right: NormalizedProduct) -> dict:
    """Return heuristic confidence; it does not claim perfect product matching."""
    same_brand, same_category = left.brand.lower() == right.brand.lower(), left.category == right.category
    title_score = 1.0 if normalize_product_name(left.title) == normalize_product_name(right.title) else _token_similarity(left.title, right.title)
    keys = set(left.features) | set(right.features)
    feature_score = sum(left.features.get(key) == right.features.get(key) for key in keys) / len(keys) if keys else 0.0
    confidence = round(0.35 * same_brand + 0.20 * same_category + 0.30 * title_score + 0.15 * feature_score, 2)
    return {"likely_match": confidence >= 0.7, "confidence": confidence, "same_brand": same_brand, "title_similarity": round(title_score, 2), "feature_similarity": round(feature_score, 2)}


def _token_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = set(re.findall(r"[a-z0-9]+", left.lower())), set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0
