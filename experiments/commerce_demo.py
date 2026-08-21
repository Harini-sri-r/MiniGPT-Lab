"""Run deterministic Phase 7 commerce examples against synthetic mock data."""

from commerce.comparison import compare_product_group, group_products
from commerce.catalog import load_mock_products
from commerce.service import build_comparison_response


def show_recommendations(query: str) -> None:
    response = build_comparison_response(query)
    print(f"\nQuery: {query}\nFound {response['found_count']} mock listings.")
    for index, item in enumerate(response["recommendations"], 1):
        product = item["product"]
        print(f"{index}. {product.product_name} - {item['reason']} | Rs. {product.price} | {product.rating} | {product.platform} | score {item['score']}")
    print(response["data_notice"])


def main() -> None:
    show_recommendations("What are the best wireless headphones under Rs. 3000?")
    show_recommendations("Find me a laptop under Rs. 60000 for programming.")
    products = load_mock_products()
    sony = next(group for group in group_products(products).values() if group[0].product_name == "Sony WH-CH520")
    comparison = compare_product_group(sony[:2])
    print(f"\nSony WH-CH520: cheapest is {comparison['cheapest'].platform} at Rs. {comparison['cheapest'].price}; highest-rated is {comparison['highest_rated'].platform}.")


if __name__ == "__main__":
    main()
