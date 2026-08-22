"""Provider abstraction for marketplace data (mock and future live)."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import os
import datetime
from typing import Tuple

from .catalog import load_mock_products
from .models import NormalizedProduct, Product, UserRequirements
from .search import search_products
from .config import MarketplaceConfig, load_config


class MarketplaceProvider(ABC):
    """Common interface future compliant data providers can implement."""

    platform: str
    data_source: str  # "mock" or "live"

    @abstractmethod
    def search_products(self, requirements: UserRequirements) -> list[NormalizedProduct]: ...

    @abstractmethod
    def get_product(self, product_id: str) -> NormalizedProduct | None: ...

    @abstractmethod
    def availability(self, product_id: str) -> str | None: ...

    @abstractmethod
    def price(self, product_id: str) -> int | None: ...

    @staticmethod
    def normalize_product(product: Product, source: str = "mock") -> NormalizedProduct:
        """Convert a raw Product into a NormalizedProduct, adding metadata.

        The ``source`` argument indicates whether the data originated from a mock
        or live provider. ``retrieved_at`` records the UTC timestamp of conversion.
        """
        now = datetime.datetime.utcnow().isoformat()
        return NormalizedProduct(
            product.product_id,
            product.platform,
            product.product_name,
            product.brand,
            product.category,
            product.price,
            product.rating,
            product.review_count,
            product.specifications,
            product.product_url,
            product.availability,
            currency="INR",
            image_url="",
            data_source=source,
            retrieved_at=now,
        )


class MockMarketplaceProvider(MarketplaceProvider):
    """Reusable offline adapter; it never makes network calls or returns live prices."""

    data_source = "mock"

    def __init__(self, platform: str, products: Iterable[Product] | None = None):
        self.platform = platform
        self._products = [product for product in (products or load_mock_products()) if product.platform == platform]

    def search_products(self, requirements: UserRequirements) -> list[NormalizedProduct]:
        found = search_products(
            "",
            products=self._products,
            category=requirements.category,
            maximum_price=requirements.budget,
            minimum_rating=requirements.minimum_rating,
            brand=requirements.brand,
            platform=self.platform,
        )
        return [self.normalize_product(p, source=self.data_source) for p in found]

    def get_product(self, product_id: str) -> NormalizedProduct | None:
        product = next((p for p in self._products if p.product_id == product_id), None)
        return self.normalize_product(product, source=self.data_source) if product else None

    def availability(self, product_id: str) -> str | None:
        prod = self.get_product(product_id)
        return prod.availability if prod else None

    def price(self, product_id: str) -> int | None:
        prod = self.get_product(product_id)
        return prod.price if prod else None

    def search(self, requirements: UserRequirements) -> list[NormalizedProduct]:
        """Alias retained for compatibility with existing callers."""
        return self.search_products(requirements)


class AmazonMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None):
        super().__init__("Amazon", products)


class FlipkartMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None):
        super().__init__("Flipkart", products)


class MeeshoMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None):
        super().__init__("Meesho", products)


class NotConfiguredError(RuntimeError):
    """Raised when a live provider is requested but required configuration is missing."""
    pass


class LiveMarketplaceProvider(MarketplaceProvider):
    """Abstract stub for future authorized live providers.

    Subclasses must implement the concrete API calls. The base class checks for
    required environment variables, raising :class:`NotConfiguredError` if they
    are absent.
    """

    data_source = "live"

    def __init__(self, platform: str, required_env_vars: Tuple[str, ...] | None = None):
        self.platform = platform
        required_env_vars = required_env_vars or []
        missing = [var for var in required_env_vars if not os.getenv(var)]
        if missing:
            raise NotConfiguredError(
                f"Missing required env vars for live provider {platform}: {', '.join(missing)}"
            )

    # Subclasses must implement abstract methods.


def default_providers() -> list[MarketplaceProvider]:
    """Return the standard mock providers used by earlier phases."""
    return [AmazonMockProvider(), FlipkartMockProvider(), MeeshoMockProvider()]


def configured_providers(config: MarketplaceConfig | None = None) -> list[MarketplaceProvider]:
    """Instantiate providers according to the supplied configuration.

    * If ``config`` is ``None``, the environment variables are loaded via
      :func:`load_config`.
    * In ``mock`` mode only the mock adapters are instantiated.
    * In ``live`` mode the function attempts to create a live provider for each
      enabled platform; if instantiation fails (e.g., missing credentials) the
      mock fallback is used.
    """
    if config is None:
        config = load_config()
    providers: list[MarketplaceProvider] = []
    enabled = set(config.enabled_platforms)
    mock_map = {
        "Amazon": AmazonMockProvider,
        "Flipkart": FlipkartMockProvider,
        "Meesho": MeeshoMockProvider,
    }
    if config.mode == "live":
        for platform in enabled:
            live_cls = globals().get(f"Live{platform}Provider")
            if live_cls:
                try:
                    providers.append(live_cls())
                except NotConfiguredError:
                    providers.append(mock_map[platform]())
            else:
                providers.append(mock_map[platform]())
    else:
        for platform in enabled:
            if platform in mock_map:
                providers.append(mock_map[platform]())
    return providers
