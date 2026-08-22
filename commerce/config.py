from dataclasses import dataclass
import os
from typing import Tuple


@dataclass(frozen=True)
class MarketplaceConfig:
    """Configuration for marketplace providers.

    Attributes:
        mode: "mock" (default) or "live" – determines whether to attempt live providers.
        enabled_platforms: Tuple of platform names to query. Empty means all default mock providers.
        timeout_seconds: Per‑provider network timeout (future live calls).
        max_results: Maximum number of results to return per provider.
    """
    mode: str = "mock"
    enabled_platforms: Tuple[str, ...] = ("Amazon", "Flipkart", "Meesho")
    timeout_seconds: int = 10
    max_results: int = 20


def load_config() -> MarketplaceConfig:
    """Load configuration from environment variables.

    Environment variables:
        MARKETPLACE_MODE – "mock" or "live" (default: "mock").
        MARKETPLACE_PLATFORMS – comma‑separated list of platforms (default: all three mocks).
        MARKETPLACE_TIMEOUT – integer seconds (default: 10).
        MARKETPLACE_MAX_RESULTS – integer (default: 20).
    """
    mode = os.getenv("MARKETPLACE_MODE", "mock").lower()
    platforms = os.getenv("MARKETPLACE_PLATFORMS")
    if platforms:
        enabled_platforms = tuple(p.strip() for p in platforms.split(",") if p.strip())
    else:
        enabled_platforms = ("Amazon", "Flipkart", "Meesho")
    try:
        timeout = int(os.getenv("MARKETPLACE_TIMEOUT", "10"))
    except ValueError:
        timeout = 10
    try:
        max_results = int(os.getenv("MARKETPLACE_MAX_RESULTS", "20"))
    except ValueError:
        max_results = 20
    return MarketplaceConfig(
        mode=mode,
        enabled_platforms=enabled_platforms,
        timeout_seconds=timeout,
        max_results=max_results,
    )
