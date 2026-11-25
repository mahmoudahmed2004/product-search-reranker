"""Refresh the curated storefront snapshot from the pinned public catalog."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

REVISION = "3d78fed5f35b7f1087db1c162edc412f2a736722"
SOURCE = f"https://raw.githubusercontent.com/Ovi/DummyJSON/{REVISION}/database/products.json"
ARCHIVE = Path("storefront/catalog.json")

CATEGORY = {
    "laptops": ("computers", "laptop"),
    "smartphones": ("phones", "phone"),
    "tablets": ("tablets", "tablet"),
    "fragrances": ("beauty", "fragrance"),
    "mens-watches": ("style", "analog-watch"),
    "womens-watches": ("style", "analog-watch"),
    "sunglasses": ("style", "eyewear"),
}
ITEM_TYPES = {
    14: ("workspace", "chair"),
    47: ("workspace", "lamp"),
    99: ("audio", "speaker"),
    100: ("audio", "personal-audio"),
    101: ("audio", "personal-audio"),
    103: ("audio", "speaker"),
    104: ("accessories", "charger"),
    105: ("accessories", "battery"),
    106: ("accessories", "watch"),
    107: ("audio", "personal-audio"),
    108: ("accessories", "case"),
    109: ("accessories", "camera-support"),
    110: ("accessories", "camera-light"),
    111: ("accessories", "camera-support"),
    112: ("accessories", "camera-support"),
}
# Products with different jobs keep distinct kinds, even within one department.
GROUPS = {
    "kitchen": {
        "turner": (48, 72),
        "drinkware": (49, 59),
        "whisk": (50,),
        "blender": (51, 61),
        "pan": (52, 68),
        "cutting-board": (53,),
        "juicer": (54,),
        "slicer": (55, 77),
        "stove": (56,),
        "strainer": (57, 63),
        "flatware": (58, 74),
        "grater": (60,),
        "ice-tray": (62,),
        "kitchen-knife": (64,),
        "food-storage": (65,),
        "microwave": (66,),
        "organizer": (67, 73),
        "plate": (69,),
        "tongs": (70,),
        "cooking-pot": (71,),
        "serving-tray": (75,),
        "rolling-pin": (76,),
    },
    "sports": {
        "american-football": (137,),
        "baseball": (138,),
        "baseball-glove": (139,),
        "basketball": (140,),
        "basketball-rim": (141,),
        "cricket-ball": (142,),
        "cricket-bat": (143,),
        "cricket-helmet": (144,),
        "cricket-wicket": (145,),
        "shuttlecock": (146,),
        "soccer-ball": (147,),
        "golf-ball": (148,),
        "golf-club": (149,),
        "baseball-bat": (150,),
        "tennis-ball": (151,),
        "tennis-racket": (152,),
        "volleyball": (153,),
    },
    "home": {
        "bed": (11,),
        "sofa": (12,),
        "side-table": (13,),
        "bathroom-vanity": (15,),
        "swing": (43,),
        "photo-frame": (44,),
        "artificial-plant": (45,),
        "planter": (46,),
    },
    "beauty": {
        "mascara": (1,),
        "eyeshadow": (2,),
        "face-powder": (3,),
        "lipstick": (4,),
        "nail-polish": (5,),
        "hand-soap": (118,),
        "body-wash": (119,),
        "body-lotion": (120,),
    },
    "pantry": {
        "fruit": (16, 30, 31, 33, 40),
        "meat": (17, 19),
        "oil": (20,),
        "vegetable": (21, 25, 26, 35, 37),
        "eggs": (23,),
        "seafood": (24,),
        "honey": (27,),
        "ice-cream": (28,),
        "juice": (29,),
        "milk": (32,),
        "coffee": (34,),
        "protein-powder": (36,),
        "rice": (38,),
        "soft-drink": (39,),
        "water": (42,),
    },
    "pets": {"cat-food": (18,), "dog-food": (22,)},
}
ITEM_TYPES.update(
    (item_id, (collection, kind))
    for collection, types in GROUPS.items()
    for kind, ids in types.items()
    for item_id in ids
)

# A device-specific accessory without a confirmed fit has an empty allowlist.
DEVICE = {
    121: "iphone-5s",
    122: "iphone-6",
    123: "iphone-13-pro",
    124: "iphone-x",
    159: "ipad-mini-2021",
}
COMPATIBLE = {105: ["iphone-13-pro"], 108: ["iphone-12"]}
FEATURED = [52, 152, 119, 78, 12, 123, 98, 34, 101, 14, 160, 47]


def select_products(rows: list[dict]) -> list[dict]:
    products = []
    seen_slugs = set()
    for row in rows:
        item_id = row["id"]
        taxon = ITEM_TYPES.get(item_id, CATEGORY.get(row["category"]))
        if taxon is None:
            continue
        collection, kind = taxon
        slug = re.sub(r"[^a-z0-9]+", "-", row["title"].lower()).strip("-")
        if slug in seen_slugs:
            slug = f"{slug}-{item_id}"
        seen_slugs.add(slug)
        products.append(
            {
                "id": item_id,
                "slug": slug,
                "title": row["title"],
                "description": row["description"],
                "brand": row.get("brand", ""),
                "price_cents": int(Decimal(str(row["price"])) * 100),
                "stock": row["stock"],
                "collection": collection,
                "kind": kind,
                "tags": row["tags"],
                "image": row["thumbnail"],
                "images": row["images"],
                "device": DEVICE.get(item_id, ""),
                "compatible_with": COMPATIBLE.get(item_id, []),
                "featured": FEATURED.index(item_id) + 1 if item_id in FEATURED else 999,
            }
        )
    return products


def main() -> None:
    with urlopen(SOURCE, timeout=45) as response:
        items = select_products(json.load(response))
    payload = {"source": SOURCE, "revision": REVISION, "currency": "USD", "products": items}
    ARCHIVE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} products to {ARCHIVE}")


if __name__ == "__main__":
    main()
