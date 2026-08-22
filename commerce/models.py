"""Typed product and response models used by the mock commerce assistant."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Product:
    """A product listing from the synthetic Phase 7 catalog; never live data."""

    product_id: str
    product_name: str
    platform: str
    category: str
    price: int
    original_price: int
    rating: float
    review_count: int
    brand: str
    specifications: Mapping[str, str]
    product_url: str
    availability: str


@dataclass(frozen=True)
class UserRequirements:
    query: str
    category: str | None = None
    budget: int | None = None
    features: tuple[str, ...] = ()
    use_case: str | None = None
    brand: str | None = None
    platform: str | None = None
    minimum_rating: float | None = None
    required_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedProduct:
    """Provider-neutral representation of a synthetic marketplace listing."""

    product_id: str
    platform: str
    title: str
    brand: str
    category: str
    price: int
    rating: float
    review_count: int
    features: Mapping[str, str]
    product_url: str
    availability: str
    currency: str = "INR"
    image_url: str = ""
    data_source: str = "mock"
    retrieved_at: str = ""
