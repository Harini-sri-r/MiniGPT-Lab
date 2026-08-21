"""Provider abstraction for offline synthetic marketplace data only."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .catalog import load_mock_products
from .models import NormalizedProduct, Product, UserRequirements
from .search import search_products


class MarketplaceProvider(ABC):
    """Common interface future compliant data providers can implement."""

    platform: str

    @abstractmethod
    def search_products(self, requirements: UserRequirements) -> list[NormalizedProduct]: ...

    @abstractmethod
    def get_product(self, product_id: str) -> NormalizedProduct | None: ...

    @abstractmethod
    def availability(self, product_id: str) -> str | None: ...

    @abstractmethod
    def price(self, product_id: str) -> int | None: ...

    @staticmethod
    def normalize_product(product: Product) -> NormalizedProduct:
        return NormalizedProduct(product.product_id, product.platform, product.product_name, product.brand, product.category, product.price, product.rating, product.review_count, product.specifications, product.product_url, product.availability)


class MockMarketplaceProvider(MarketplaceProvider):
    """Reusable offline adapter; it never makes network calls or returns live prices."""

    def __init__(self, platform: str, products: Iterable[Product] | None = None):
        self.platform = platform
        self._products = [product for product in (products or load_mock_products()) if product.platform == platform]

    def search_products(self, requirements: UserRequirements) -> list[NormalizedProduct]:
        found = search_products("", products=self._products, category=requirements.category, maximum_price=requirements.budget, minimum_rating=requirements.minimum_rating, brand=requirements.brand, platform=self.platform)
        return [self.normalize_product(product) for product in found]

    def get_product(self, product_id: str) -> NormalizedProduct | None:
        product = next((item for item in self._products if item.product_id == product_id), None)
        return self.normalize_product(product) if product else None

    def availability(self, product_id: str) -> str | None:
        product = self.get_product(product_id)
        return product.availability if product else None

    def price(self, product_id: str) -> int | None:
        product = self.get_product(product_id)
        return product.price if product else None

    def search(self, requirements: UserRequirements) -> list[NormalizedProduct]:
        """Short alias retained for future provider adapters."""
        return self.search_products(requirements)


class AmazonMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None): super().__init__("Amazon", products)


class FlipkartMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None): super().__init__("Flipkart", products)


class MeeshoMockProvider(MockMarketplaceProvider):
    def __init__(self, products: Iterable[Product] | None = None): super().__init__("Meesho", products)


def default_providers() -> list[MarketplaceProvider]:
    return [AmazonMockProvider(), FlipkartMockProvider(), MeeshoMockProvider()]
