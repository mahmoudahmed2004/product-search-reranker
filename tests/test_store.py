from pathlib import Path

import pytest

pytest.importorskip("fastembed")

from product_reranker.store import complement_allowed, read_catalog  # noqa: E402

CATALOG = Path(__file__).parents[1] / "storefront" / "catalog.json"


def test_device_specific_complements_are_mutual_and_use_exact_fit():
    products = {product.id: product for product in read_catalog(CATALOG)}
    phone = products[123]
    battery = products[105]
    assert complement_allowed(phone, battery)
    assert complement_allowed(battery, phone)
    assert not complement_allowed(phone, products[108])
    assert not complement_allowed(products[108], phone)
    assert not complement_allowed(products[133], battery)


def test_laptop_setup_uses_distinct_available_product_types():
    products = {product.id: product for product in read_catalog(CATALOG)}
    laptop = products[78]
    assert complement_allowed(laptop, products[100])
    assert complement_allowed(laptop, products[47])
    assert not complement_allowed(laptop, products[79])
    assert not complement_allowed(laptop, products[100].model_copy(update={"stock": 0}))
