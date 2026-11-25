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
# A device-specific accessory without a confirmed fit has an empty allowlist.
DEVICE = {
    121: "iphone-5s",
    122: "iphone-6",
    123: "iphone-13-pro",
    124: "iphone-x",
    159: "ipad-mini-2021",
}
COMPATIBLE = {105: ["iphone-13-pro"], 108: ["iphone-12"]}
FEATURED = [78, 123, 101, 14, 160, 82, 100, 47]


def select_products(rows: list[dict]) -> list[dict]:
    products = []
    for row in rows:
        item_id = row["id"]
        taxon = ITEM_TYPES.get(item_id, CATEGORY.get(row["category"]))
        if taxon is None:
            continue
        collection, kind = taxon
        slug = re.sub(r"[^a-z0-9]+", "-", row["title"].lower()).strip("-")
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
