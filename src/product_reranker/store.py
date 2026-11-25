"""Hybrid search and product-to-product recommendations for the demo catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from .core import Candidate, normalize_query
from .model import Ranker
from .relations import MODEL, PairProduct, PairRankers, pair_features

DIMENSION = 384
INDEX_SCHEMA = "assembly-v2-granite-bm25"
COLLECTIONS = {
    "computers": "Computers",
    "phones": "Phones",
    "tablets": "Tablets",
    "audio": "Audio",
    "accessories": "Accessories",
    "workspace": "Workspace",
}
DEVICE_SPECIFIC = {"case", "battery", "charger", "watch"}
HANDHELD = {"phone", "tablet"}
COMPANIONS = {
    "laptop": {
        "personal-audio": "For calls and listening at your desk",
        "lamp": "Task lighting for your workspace",
        "chair": "Seating for your desk setup",
        "speaker": "For music around the workspace",
    },
    "phone": {
        "battery": "Confirmed fit for this phone",
        "case": "Made to fit this phone",
        "charger": "Confirmed fit for this phone",
        "watch": "Supported phone pairing",
        "personal-audio": "Wireless listening on the go",
    },
    "tablet": {"personal-audio": "For listening and video calls"},
    "personal-audio": {
        "phone": "For listening away from your desk",
        "tablet": "For music, video, and calls",
        "laptop": "For work calls and listening",
    },
    "speaker": {"lamp": "For a listening space after dark", "chair": "For your room setup"},
    "chair": {"lamp": "Task lighting for the same setup", "laptop": "For your workspace"},
    "lamp": {"chair": "Seating for the same workspace", "laptop": "For a desk setup"},
    "battery": {"phone": "Confirmed fit for this battery pack"},
    "charger": {"phone": "Confirmed fit for this charger"},
    "case": {"phone": "Confirmed fit for this case"},
    "watch": {"phone": "Supported phone pairing"},
    "camera-support": {"camera-light": "Lighting for a shooting setup"},
    "camera-light": {"camera-support": "Support for a shooting setup"},
}


class Product(BaseModel):
    id: int = Field(gt=0)
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    description: str
    brand: str = ""
    price_cents: int = Field(ge=0)
    stock: int = Field(ge=0)
    collection: str
    kind: str
    image: str
    images: list[str]
    tags: list[str] = Field(default_factory=list)
    device: str = ""
    compatible_with: list[str] = Field(default_factory=list)
    featured: int = 999

    def pair(self) -> PairProduct:
        return PairProduct(self.title, self.description, self.brand)

    def text(self) -> str:
        kind = self.kind.replace("-", " ")
        return " ".join((self.title, self.brand, kind, " ".join(self.tags), self.description))


class Catalog(BaseModel):
    products: list[Product] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Recommendation:
    product: Product
    reason: str


def read_catalog(path: Path) -> list[Product]:
    products = Catalog.model_validate_json(path.read_bytes()).products
    for key in ("id", "slug"):
        if len({getattr(product, key) for product in products}) != len(products):
            raise ValueError(f"Catalog {key} values must be unique.")
    if any(product.collection not in COLLECTIONS for product in products):
        raise ValueError("The catalog has an unknown storefront collection.")
    return products


def complement_allowed(anchor: Product, target: Product) -> bool:
    if anchor.id == target.id or target.stock == 0:
        return False
    if target.kind not in COMPANIONS.get(anchor.kind, {}):
        return False
    if anchor.kind in HANDHELD and target.kind in DEVICE_SPECIFIC:
        return bool(anchor.device) and anchor.device in target.compatible_with
    if target.kind in HANDHELD and anchor.kind in DEVICE_SPECIFIC:
        return bool(target.device) and target.device in anchor.compatible_with
    return True


def candidate_filter(ids: Sequence[int]) -> models.Filter:
    return models.Filter(must=[models.HasIdCondition(has_id=list(ids))])


class StoreEngine:
    def __init__(
        self,
        catalog_path: Path,
        index_dir: Path,
        cache_dir: Path,
        ranker: Ranker | None,
        prediction_lock: Lock,
        relations: PairRankers | None = None,
    ) -> None:
        products = read_catalog(catalog_path)
        self.products = {product.id: product for product in products}
        self.by_slug = {product.slug: product for product in products}
        self.ranker = ranker
        self.relations = relations
        self.prediction_lock = prediction_lock
        self.lock = Lock()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.dense = TextEmbedding(MODEL, cache_dir=str(cache_dir), threads=2)
        self.sparse = SparseTextEmbedding("Qdrant/bm25", cache_dir=str(cache_dir), threads=2)
        checksum = hashlib.sha256(
            INDEX_SCHEMA.encode() + MODEL.encode() + catalog_path.read_bytes(),
        ).hexdigest()[:12]
        self.index_name = f"catalog_{checksum}"
        index_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(index_dir), force_disable_check_same_thread=True)
        try:
            self._ensure_index(products)
        except Exception:
            self.client.close()
            raise

    def close(self) -> None:
        self.client.close()

    def _ensure_index(self, products: Sequence[Product]) -> None:
        if not self.client.collection_exists(self.index_name):
            self.client.create_collection(
                self.index_name,
                vectors_config={
                    name: models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE)
                    for name in ("dense", "pair")
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        if self.client.count(self.index_name, exact=True).count == len(products):
            return

        texts = [product.text() for product in products]
        points = []
        product_vectors = self.dense.embed([product.pair().text() for product in products])
        for product, dense, sparse, pair in zip(
            products,
            self.dense.embed(texts),
            self.sparse.embed(texts),
            product_vectors,
            strict=True,
        ):
            points.append(
                models.PointStruct(
                    id=product.id,
                    vector={
                        "dense": dense.tolist(),
                        "pair": pair.tolist(),
                        "bm25": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                )
            )
        self.client.upsert(self.index_name, points, wait=True)

    def catalog(self, collection: str = "") -> list[Product]:
        return sorted(
            (p for p in self.products.values() if not collection or p.collection == collection),
            key=lambda p: (p.featured, p.id),
        )

    def _point_ids(self, query, *, using=None, where=None, prefetch=None, limit=20) -> list[int]:
        response = self.client.query_points(
            self.index_name,
            query=query,
            using=using,
            query_filter=where,
            prefetch=prefetch,
            limit=limit,
            with_payload=False,
        )
        return [int(hit.id) for hit in response.points]

    def search(
        self,
        query: str,
        collection: str = "",
        strategy: Literal["bm25", "dense", "hybrid", "reranked"] = "reranked",
    ) -> list[Product]:
        normalized = normalize_query(query)
        if not normalized:
            return self.catalog(collection)
        eligible = [p.id for p in self.catalog(collection) if p.stock]
        if not eligible:
            return []
        where = candidate_filter(eligible)
        with self.lock:
            dense = None
            lexical = None
            if strategy != "bm25":
                dense = next(self.dense.query_embed(query)).tolist()
            if strategy != "dense":
                sparse = next(self.sparse.query_embed(query))
                lexical = models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                )
            if strategy == "bm25":
                ids = self._point_ids(lexical, using="bm25", where=where)
            elif strategy == "dense":
                ids = self._point_ids(dense, using="dense", where=where)
            else:
                prefetch = [models.Prefetch(query=dense, using="dense", limit=20, filter=where)]
                if lexical.indices:
                    prefetch.append(
                        models.Prefetch(query=lexical, using="bm25", limit=20, filter=where)
                    )
                ids = self._point_ids(
                    models.FusionQuery(fusion=models.Fusion.RRF), prefetch=prefetch, where=where
                )
            return self._rerank(query, normalized, ids, strategy)

    def _rerank(self, query: str, normalized: str, ids: list[int], strategy: str) -> list[Product]:
        products = [self.products[item_id] for item_id in ids]
        if strategy != "reranked" or self.ranker is None or not products:
            return products
        candidates = [
            Candidate(str(product.id), product.title, product.description, product.brand)
            for product in products
        ]
        with self.prediction_lock:
            learned = {
                int(row.product_id): row.lambdarank_rank
                for row in self.ranker.rank(query, candidates)
            }
        retrieval = {item_id: index for index, item_id in enumerate(ids, 1)}
        return sorted(
            products,
            key=lambda p: (
                normalize_query(p.title) != normalized,
                -(1 / (4 + retrieval[p.id]) + 1 / (4 + learned[p.id])),
                retrieval[p.id],
            ),
        )

    def _rerank_pairs(self, anchor: Product, ids: list[int], mode: str) -> list[int]:
        if self.relations is None or len(ids) < 2:
            return ids
        booster = self.relations.similar if mode == "similar" else self.relations.complements
        matrix = pair_features(anchor.pair(), [self.products[item_id].pair() for item_id in ids])
        with self.prediction_lock:
            scores = booster.predict(matrix, num_threads=1)
        positions = {item_id: rank for rank, item_id in enumerate(ids, 1)}
        ordered = sorted(ids, key=lambda item_id: -scores[positions[item_id] - 1])
        learned = {item_id: rank for rank, item_id in enumerate(ordered, 1)}
        return sorted(
            ids,
            key=lambda item_id: -(1 / (4 + positions[item_id]) + 1 / (4 + learned[item_id])),
        )

    def recommendations(self, anchor: Product) -> tuple[list[Recommendation], list[Recommendation]]:
        available = [p for p in self.products.values() if p.stock and p.id != anchor.id]
        substitutes = [p.id for p in available if p.kind == anchor.kind]
        linked = [p.id for p in available if complement_allowed(anchor, p)]
        reasons = COMPANIONS.get(anchor.kind, {})
        with self.lock:
            similar_ids = (
                self._point_ids(
                    anchor.id, using="pair", where=candidate_filter(substitutes), limit=12
                )
                if substitutes
                else []
            )
            similar_ids = self._rerank_pairs(anchor, similar_ids, "similar")
            complement_ids = []
            if linked:
                intent = f"Products to use alongside {anchor.title} in a setup: " + ", ".join(
                    reasons[kind]
                    for kind in dict.fromkeys(self.products[item_id].kind for item_id in linked)
                )
                vector = next(self.dense.query_embed(intent)).tolist()
                complement_ids = self._point_ids(
                    vector, using="dense", where=candidate_filter(linked), limit=len(linked)
                )
                complement_ids = self._rerank_pairs(anchor, complement_ids, "complements")

        similar = [
            Recommendation(self.products[item_id], "Same product type")
            for item_id in similar_ids[:4]
        ]
        ordered = [self.products[item_id] for item_id in complement_ids]
        # Keep verified device-specific matches visible alongside more general equipment.
        ordered.sort(
            key=lambda product: (
                not (
                    anchor.device in product.compatible_with
                    or product.device in anchor.compatible_with
                )
            )
        )
        distinct = {}
        for product in ordered:
            distinct.setdefault(product.kind, product)
        first = list(distinct.values())
        primary_ids = {product.id for product in first}
        balanced = first + [product for product in ordered if product.id not in primary_ids]
        return similar, [Recommendation(product, reasons[product.kind]) for product in balanced[:4]]
