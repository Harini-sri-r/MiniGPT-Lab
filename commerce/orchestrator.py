"""Search orchestrator for Commerce Phase 11.

It queries a list of providers (mock and live) and aggregates results while
preserving per‑provider status and data‑source metadata.
"""

from dataclasses import dataclass
from typing import List, Dict
import datetime

from .models import NormalizedProduct
from .providers import MarketplaceProvider, configured_providers


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    success: bool
    error: str | None = None
    result_count: int = 0


@dataclass(frozen=True)
class SearchResult:
    products: List[NormalizedProduct]
    provider_results: Dict[str, ProviderResult]
    data_source_summary: str
    data_notice: str


def _attach_metadata(product: NormalizedProduct, source: str) -> NormalizedProduct:
    """Return a copy of ``product`` with ``data_source`` and ``retrieved_at`` set.

    ``product`` is a dataclass; we recreate it with the extra fields. This avoids
    modifying the original provider implementations.
    """
    now = datetime.datetime.utcnow().isoformat()
    from dataclasses import replace
    return replace(product, data_source=source, retrieved_at=now)


def orchestrate_search(requirements, config=None) -> SearchResult:
    """Execute a search across configured providers.

    Parameters
    ----------
    requirements: UserRequirements
        Parsed user query requirements.
    config: MarketplaceConfig | None
        Optional configuration; if omitted the environment is consulted.
    """
    providers = configured_providers(config)
    all_products: List[NormalizedProduct] = []
    provider_results: Dict[str, ProviderResult] = {}
    for provider in providers:
        try:
            raw = provider.search_products(requirements)
            enriched = [_attach_metadata(p, getattr(provider, "data_source", "unknown")) for p in raw]
            all_products.extend(enriched)
            provider_results[provider.platform] = ProviderResult(
                provider=provider.platform,
                success=True,
                result_count=len(enriched),
            )
        except Exception as exc:  # pragma: no cover – defensive guard.
            provider_results[provider.platform] = ProviderResult(
                provider=provider.platform,
                success=False,
                error=str(exc),
                result_count=0,
            )
    sources = {getattr(p, "data_source", "unknown") for p in all_products}
    if not sources:
        summary = "none"
        notice = "No products found."
    elif sources == {"mock"}:
        summary = "mock"
        notice = "Results are from offline mock providers."
    elif sources == {"live"}:
        summary = "live"
        notice = "Results are from live marketplace providers."
    else:
        summary = "mixed"
        notice = "Results combine mock and live data sources."
    return SearchResult(
        products=all_products,
        provider_results=provider_results,
        data_source_summary=summary,
        data_notice=notice,
    )
