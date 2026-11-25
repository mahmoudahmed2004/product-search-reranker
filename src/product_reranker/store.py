"""Hybrid search and product-to-product recommendations for the demo catalog."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from py_rust_stemmers import SnowballStemmer
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from .core import Candidate, normalize_query, tokens
from .home import Hit, preserve_query_evidence, rank_topic_hits, select_topic
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
    "kitchen": "Kitchen & dining",
    "sports": "Sports & outdoors",
    "home": "Home & living",
    "beauty": "Beauty & care",
    "style": "Style",
    "pantry": "Pantry",
    "pets": "Pets",
}
CONTEXT_FILLER = {
    "a",
    "an",
    "and",
    "at",
    "best",
    "buy",
    "for",
    "from",
    "gear",
    "in",
    "of",
    "on",
    "practice",
    "shop",
    "the",
    "to",
    "under",
    "with",
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
    "tennis-racket": {"tennis-ball": "Tennis balls for the same game"},
    "tennis-ball": {"tennis-racket": "A racket for the same game"},
    "basketball": {"basketball-rim": "A rim for shooting practice"},
    "basketball-rim": {"basketball": "A ball for shooting practice"},
    "baseball": {"baseball-bat": "A bat for hitting practice", "baseball-glove": "For fielding"},
    "baseball-bat": {"baseball": "Baseballs for hitting practice"},
    "baseball-glove": {"baseball": "Baseballs for catching practice"},
    "cricket-bat": {"cricket-ball": "For cricket practice", "cricket-wicket": "Set up a wicket"},
    "cricket-ball": {"cricket-bat": "For batting practice"},
    "golf-club": {"golf-ball": "Golf balls for practice"},
    "golf-ball": {"golf-club": "A club for the same sport"},
    "pan": {"turner": "For flipping and serving", "tongs": "For turning food while cooking"},
    "turner": {"pan": "Cookware for the same task"},
    "cutting-board": {"kitchen-knife": "For food prep on the board"},
    "kitchen-knife": {"cutting-board": "A surface for food prep"},
    "coffee": {"drinkware": "For serving coffee"},
    "sofa": {"side-table": "A surface beside the sofa", "lamp": "Lighting beside the seating"},
    "side-table": {"lamp": "Lighting for the table", "sofa": "Seating beside the table"},
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
        product_vectors = self.dense.embed(
            [product.pair().text() for product in products], batch_size=16
        )
        for product, dense, sparse, pair in zip(
            products,
            self.dense.embed(texts, batch_size=16),
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

    def _query_points(
        self,
        query,
        *,
        using=None,
        where=None,
        prefetch=None,
        limit=20,
    ) -> list[models.ScoredPoint]:
        return self.client.query_points(
            self.index_name,
            query=query,
            using=using,
            query_filter=where,
            prefetch=prefetch,
            limit=limit,
            with_payload=False,
        ).points

    def _point_ids(self, query, **options) -> list[int]:
        return [int(hit.id) for hit in self._query_points(query, **options)]

    def _keyword_vector(self, query: str) -> models.SparseVector:
        sparse = next(self.sparse.query_embed(query))
        return models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())

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

    def home_search(
        self,
        query: str,
        collection: str = "",
        *,
        ranking: Literal["retrieval", "learned", "guarded"] = "guarded",
    ) -> tuple[str | None, list[Product]]:
        normalized = normalize_query(query)
        available = [product for product in self.catalog(collection) if product.stock]
        if not normalized or not available:
            return None, []

        titles = [
            p
            for p in available
            if normalized in {normalize_query(p.title), normalize_query(f"{p.brand} {p.title}")}
        ]
        exact = titles[0] if len(titles) == 1 else None
        if exact and any(
            p.collection != exact.collection and normalize_query(p.brand) == normalized
            for p in available
        ):
            return None, []

        where = candidate_filter([product.id for product in available])
        with self.lock:
            vector = next(self.dense.query_embed(query)).tolist()
            dense = self._query_points(vector, using="dense", where=where, limit=16)
            keyword = self._keyword_vector(query)
            sparse = (
                self._query_points(keyword, using="bm25", where=where, limit=16)
                if keyword.indices
                else []
            )

        def to_hit(point: models.ScoredPoint) -> Hit:
            item_id = int(point.id)
            product = self.products[item_id]
            return Hit(item_id, product.collection, point.score, product.kind)

        semantic_hits = [to_hit(point) for point in dense]
        keyword_hits = [to_hit(point) for point in sparse]
        evidence = self._title_evidence(query, available)
        corroborating = [hit for hit in keyword_hits if evidence[hit.id] > 0]
        topic = select_topic(semantic_hits, corroborating, exact.collection if exact else None)
        if not topic:
            return None, []
        if exact:
            same_task = [
                product
                for product in available
                if product.collection == topic
                and product.id != exact.id
                and (product.kind == exact.kind or complement_allowed(exact, product))
            ]
            ranked = rank_topic_hits(topic, semantic_hits, keyword_hits)
            order = {item_id: index for index, item_id in enumerate(ranked)}
            same_task.sort(key=lambda product: (order.get(product.id, len(order)), product.id))
            if ranking != "retrieval":
                same_task = self._rerank(
                    query, normalized, [product.id for product in same_task], "reranked"
                )
            return topic, [exact, *same_task[:3]]
        ranked = rank_topic_hits(topic, semantic_hits, keyword_hits)
        best_evidence = max((evidence[item_id] for item_id in ranked), default=0.0)
        if best_evidence:
            ranked = [item_id for item_id in ranked if evidence[item_id] >= 0.45 * best_evidence]
        elif ranked:
            best = next(hit for hit in semantic_hits if hit.collection == topic)
            ranked = [
                hit.id
                for hit in semantic_hits
                if hit.collection == topic
                and hit.kind == best.kind
                and hit.score >= best.score - 0.06
            ]
        if ranking != "retrieval":
            ids = [product.id for product in self._rerank(query, normalized, ranked, "reranked")]
            ranked = preserve_query_evidence(ranked, ids, evidence) if ranking == "guarded" else ids
        return topic, [self.products[item_id] for item_id in ranked[:4]]

    @staticmethod
    def _title_evidence(query: str, products: Sequence[Product]) -> dict[int, float]:
        stemmer = SnowballStemmer("english")
        terms = {stemmer.stem_word(term) for term in set(tokens(query)) - CONTEXT_FILLER}
        if not terms:
            return dict.fromkeys((product.id for product in products), 0.0)
        matches = {}
        for product in products:
            text = " ".join(
                (product.title, product.kind, *product.tags, COLLECTIONS[product.collection])
            )
            matches[product.id] = terms.intersection(
                stemmer.stem_word(term) for term in tokens(text)
            )
        weights = {
            term: math.log1p(len(products) / (1 + sum(term in hits for hits in matches.values())))
            for term in terms
        }
        return {
            item_id: sum(weights[term] for term in sorted(hits))
            for item_id, hits in matches.items()
        }

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
