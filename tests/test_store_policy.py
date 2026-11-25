import pytest

models = pytest.importorskip("qdrant_client.models")
pytest.importorskip("fastembed")

from product_reranker.store import Product, candidate_filter, complement_allowed  # noqa: E402


def _product(
    pid: int,
    slug: str,
    title: str,
    kind: str,
    collection: str = "accessories",
    **overrides,
) -> Product:
    return Product(
        id=pid,
        slug=slug,
        title=title,
        description=f"{title} description",
        price_cents=4999,
        stock=overrides.pop("stock", 10),
        collection=collection,
        kind=kind,
        image=f"/img/{slug}.jpg",
        images=[f"/img/{slug}.jpg"],
        **overrides,
    )


IPHONE_13_PRO = _product(
    1,
    "iphone-13-pro",
    "Apple iPhone 13 Pro",
    "phone",
    "phones",
    brand="Apple",
    device="iphone-13-pro",
    tags=["magsafe", "ios"],
)
IPHONE_12 = _product(
    2,
    "iphone-12",
    "Apple iPhone 12",
    "phone",
    "phones",
    brand="Apple",
    device="iphone-12",
    tags=["magsafe", "ios"],
)
ANDROID_PHONE = _product(
    3,
    "galaxy-s23",
    "Samsung Galaxy S23",
    "phone",
    "phones",
    brand="Samsung",
    device="galaxy-s23",
    tags=["android"],
)
UNKNOWN_PHONE = _product(4, "mystery-phone", "Unlocked Smartphone", "phone", "phones")
LAPTOP = _product(
    5,
    "thinkpad-x1",
    "ThinkPad X1 Carbon",
    "laptop",
    "computers",
    brand="Lenovo",
    device="thinkpad-x1",
)
MAGSAFE_BATTERY = _product(
    10,
    "apple-magsafe-battery",
    "Apple MagSafe Battery Pack",
    "battery",
    brand="Apple",
    tags=["magsafe", "battery"],
    compatible_with=["iphone-12", "iphone-13-pro"],
)
MAGSAFE_BATTERY_OOS = _product(
    11,
    "apple-magsafe-battery-oos",
    "Apple MagSafe Battery Pack",
    "battery",
    stock=0,
    brand="Apple",
    tags=["magsafe", "battery"],
    compatible_with=["iphone-12", "iphone-13-pro"],
)
IPHONE_12_CASE = _product(
    12,
    "iphone-12-case",
    "iPhone 12 Silicone Case",
    "case",
    brand="Apple",
    compatible_with=["iphone-12"],
)
GENERIC_AUDIO = _product(
    13,
    "wireless-headphones",
    "Studio Wireless Headphones",
    "personal-audio",
    "audio",
    brand="Sony",
    tags=["bluetooth"],
)


@pytest.mark.parametrize(
    ("anchor", "candidate", "expected"),
    [
        pytest.param(IPHONE_13_PRO, MAGSAFE_BATTERY, True, id="magsafe-allowed-iphone13pro"),
        pytest.param(ANDROID_PHONE, MAGSAFE_BATTERY, False, id="magsafe-blocked-android"),
        pytest.param(UNKNOWN_PHONE, MAGSAFE_BATTERY, False, id="magsafe-blocked-unknown-phone"),
        pytest.param(IPHONE_13_PRO, IPHONE_12_CASE, False, id="iphone12-case-blocked-13pro"),
        pytest.param(LAPTOP, GENERIC_AUDIO, True, id="generic-audio-allowed-laptop"),
        pytest.param(IPHONE_13_PRO, IPHONE_13_PRO, False, id="self-rejected"),
        pytest.param(IPHONE_13_PRO, IPHONE_12, False, id="same-kind-rejected"),
        pytest.param(IPHONE_13_PRO, MAGSAFE_BATTERY_OOS, False, id="out-of-stock-rejected"),
        pytest.param(IPHONE_12_CASE, IPHONE_12, True, id="inverse-case-exact-fit-phone"),
        pytest.param(IPHONE_12_CASE, IPHONE_13_PRO, False, id="inverse-case-wrong-phone"),
        pytest.param(IPHONE_12_CASE, UNKNOWN_PHONE, False, id="inverse-case-unknown-phone"),
    ],
)
def test_complement_allowed(anchor: Product, candidate: Product, expected: bool) -> None:
    assert complement_allowed(anchor, candidate) is expected


@pytest.mark.parametrize("ids", [[1], [2, 5, 9]])
def test_candidate_filter(ids: list[int]) -> None:
    flt = candidate_filter(ids)
    assert isinstance(flt, models.Filter)
    assert flt.must is not None and len(flt.must) == 1
    cond = flt.must[0]
    assert isinstance(cond, models.HasIdCondition)
    assert set(cond.has_id) == set(ids)
    assert len(cond.has_id) == len(ids)
