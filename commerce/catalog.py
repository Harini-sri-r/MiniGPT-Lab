"""A deterministic synthetic catalog for Phase 7; it contains no live listings."""

from .models import Product

PLATFORMS = ("Amazon", "Flipkart", "Meesho")

# Each template is deliberately represented on all three mock platforms.
# Prices, ratings, URLs, and inventory below are fabricated test fixtures.
_TEMPLATES = (
    ("Sony WH-CH520", "Sony", "headphones", 2850, 4.4, 18500, {"wireless": "yes", "battery_life": "50 hours", "driver": "30 mm"}),
    ("JBL Tune 510BT", "JBL", "headphones", 2750, 4.3, 14200, {"wireless": "yes", "battery_life": "40 hours", "driver": "32 mm"}),
    ("boAt Rockerz 450", "boAt", "headphones", 1600, 4.1, 31000, {"wireless": "yes", "battery_life": "15 hours", "driver": "40 mm"}),
    ("Apple AirPods 3rd Gen", "Apple", "headphones", 16500, 4.5, 9200, {"wireless": "yes", "battery_life": "30 hours", "driver": "custom"}),
    ("Samsung Galaxy M35", "Samsung", "smartphones", 18500, 4.3, 8700, {"ram": "8 GB", "storage": "128 GB", "battery": "6000 mAh", "camera": "50 MP"}),
    ("Redmi Note 13", "Xiaomi", "smartphones", 15500, 4.2, 12100, {"ram": "6 GB", "storage": "128 GB", "battery": "5000 mAh", "camera": "108 MP"}),
    ("Moto G64", "Motorola", "smartphones", 14500, 4.2, 6800, {"ram": "8 GB", "storage": "128 GB", "battery": "6000 mAh", "camera": "50 MP"}),
    ("Apple iPhone 15", "Apple", "smartphones", 67500, 4.6, 15300, {"ram": "6 GB", "storage": "128 GB", "battery": "3349 mAh", "camera": "48 MP"}),
    ("Lenovo IdeaPad Slim 3", "Lenovo", "laptops", 54500, 4.3, 4300, {"processor": "Intel Core i5", "ram": "16 GB", "storage": "512 GB SSD", "display": "15.6 inch"}),
    ("HP 15s", "HP", "laptops", 57500, 4.2, 3900, {"processor": "Intel Core i5", "ram": "16 GB", "storage": "512 GB SSD", "display": "15.6 inch"}),
    ("ASUS Vivobook 15", "ASUS", "laptops", 59500, 4.4, 5100, {"processor": "AMD Ryzen 5", "ram": "16 GB", "storage": "512 GB SSD", "display": "15.6 inch"}),
    ("Acer Aspire Lite", "Acer", "laptops", 48500, 4.1, 2700, {"processor": "AMD Ryzen 5", "ram": "8 GB", "storage": "512 GB SSD", "display": "15.6 inch"}),
    ("Samsung Galaxy Watch 6", "Samsung", "smartwatches", 22500, 4.4, 3500, {"display": "AMOLED", "battery_life": "40 hours", "gps": "yes"}),
    ("Noise ColorFit Pro 5", "Noise", "smartwatches", 4300, 4.2, 8200, {"display": "AMOLED", "battery_life": "7 days", "gps": "yes"}),
    ("boAt Wave Sigma", "boAt", "smartwatches", 2100, 4.0, 9600, {"display": "LCD", "battery_life": "7 days", "gps": "no"}),
    ("Apple Watch SE", "Apple", "smartwatches", 28500, 4.5, 2300, {"display": "Retina", "battery_life": "18 hours", "gps": "yes"}),
    ("Logitech K380", "Logitech", "keyboards", 2800, 4.4, 11200, {"wireless": "yes", "switch": "scissor", "battery_life": "24 months"}),
    ("HP 330 Wireless Keyboard", "HP", "keyboards", 1700, 4.1, 5000, {"wireless": "yes", "switch": "membrane", "battery_life": "12 months"}),
    ("Redragon K552", "Redragon", "keyboards", 3500, 4.3, 6100, {"wireless": "no", "switch": "mechanical", "backlight": "RGB"}),
    ("Dell KM3322W", "Dell", "keyboards", 1900, 4.2, 7200, {"wireless": "yes", "switch": "membrane", "battery_life": "36 months"}),
    ("Logitech M331", "Logitech", "mice", 1200, 4.4, 15800, {"wireless": "yes", "dpi": "1000", "battery_life": "18 months"}),
    ("HP Z3700", "HP", "mice", 1100, 4.2, 6800, {"wireless": "yes", "dpi": "1200", "battery_life": "16 months"}),
    ("Razer DeathAdder Essential", "Razer", "mice", 1800, 4.5, 7900, {"wireless": "no", "dpi": "6400", "sensor": "optical"}),
    ("Dell MS3320W", "Dell", "mice", 1350, 4.1, 4100, {"wireless": "yes", "dpi": "1600", "battery_life": "36 months"}),
    ("LG 24MP60G", "LG", "monitors", 10500, 4.3, 3300, {"size": "24 inch", "resolution": "Full HD", "refresh_rate": "75 Hz"}),
    ("Samsung Odyssey G3", "Samsung", "monitors", 16500, 4.4, 2100, {"size": "24 inch", "resolution": "Full HD", "refresh_rate": "165 Hz"}),
    ("Acer Nitro VG240Y", "Acer", "monitors", 13500, 4.3, 2900, {"size": "24 inch", "resolution": "Full HD", "refresh_rate": "180 Hz"}),
    ("BenQ GW2490", "BenQ", "monitors", 11500, 4.2, 1800, {"size": "24 inch", "resolution": "Full HD", "refresh_rate": "100 Hz"}),
    ("Sony MDR-EX155AP", "Sony", "headphones", 950, 4.1, 7300, {"wireless": "no", "driver": "9 mm", "type": "in-ear"}),
    ("Realme P1", "Realme", "smartphones", 16500, 4.2, 7100, {"ram": "8 GB", "storage": "128 GB", "battery": "5000 mAh", "camera": "50 MP"}),
    ("MSI Modern 14", "MSI", "laptops", 52500, 4.3, 1900, {"processor": "Intel Core i5", "ram": "16 GB", "storage": "512 GB SSD", "display": "14 inch"}),
    ("Fastrack Limitless FS1", "Fastrack", "smartwatches", 1900, 4.0, 6400, {"display": "LCD", "battery_life": "7 days", "gps": "no"}),
    ("Cosmic Byte CB-GK-18", "Cosmic Byte", "keyboards", 2400, 4.1, 3700, {"wireless": "no", "switch": "mechanical", "backlight": "RGB"}),
    ("Zebronics AC32FHD", "Zebronics", "monitors", 8500, 4.0, 1300, {"size": "32 inch", "resolution": "Full HD", "refresh_rate": "75 Hz"}),
)


def load_mock_products() -> list[Product]:
    """Return 102 fabricated listings (34 products x 3 platforms)."""
    products: list[Product] = []
    price_offsets = (-51, -151, 39)
    rating_offsets = (0.0, -0.1, -0.2)
    for template_index, (name, brand, category, price, rating, reviews, specs) in enumerate(_TEMPLATES, 1):
        for platform_index, platform in enumerate(PLATFORMS):
            listing_name = name
            if name == "Apple AirPods 3rd Gen" and platform == "Flipkart":
                listing_name = "Apple AirPods (3rd Generation)"
            elif name == "Apple AirPods 3rd Gen" and platform == "Meesho":
                listing_name = "AirPods 3"
            listing_price = price + price_offsets[platform_index]
            products.append(Product(
                product_id=f"mock-{template_index:02d}-{platform.lower()}",
                product_name=listing_name,
                platform=platform,
                category=category,
                price=listing_price,
                original_price=listing_price + 500 + template_index * 10,
                rating=round(max(3.5, rating + rating_offsets[platform_index]), 1),
                review_count=reviews - platform_index * (reviews // 10),
                brand=brand,
                specifications=specs,
                product_url=f"https://mock.example/{platform.lower()}/product/{template_index}",
                availability="in_stock" if platform_index != 2 or template_index % 5 else "limited_stock",
            ))
    return products
