"""Feature extraction shared by offline evaluation and live ranking."""

from __future__ import annotations

import gzip
import html
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
TAG = re.compile(r"<[^>]{1,200}>")
FEATURE_NAMES = (
    "bm25",
    "bm25_title",
    "bm25_description",
    "title_coverage",
    "description_coverage",
    "bullet_coverage",
    "title_precision",
    "ordered_query_coverage",
    "title_phrase",
    "exact_title",
    "brand_coverage",
    "color_coverage",
    "type_coverage",
    "number_coverage",
    "title_length",
)


def tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(unicodedata.normalize("NFKC", text).casefold()))


def normalize_query(query: str) -> str:
    return " ".join(tokens(query))


def product_tokens(text: str, limit: int) -> tuple[str, ...]:
    return tokens(TAG.sub(" ", html.unescape(text[:12000])))[:limit]


@dataclass(frozen=True, slots=True, eq=False)
class Candidate:
    product_id: str
    title: str
    description: str = ""
    brand: str = ""
    color: str = ""
    bullet: str = ""


@dataclass(frozen=True, slots=True)
class ProductText:
    title: tuple[str, ...]
    description: tuple[str, ...]
    bullet: tuple[str, ...]
    brand: frozenset[str]
    color: frozenset[str]
    product_type: frozenset[str]
    title_counts: Counter[str]
    description_counts: Counter[str]


@lru_cache(maxsize=8192)
def prepare(product: Candidate) -> ProductText:
    title = product_tokens(product.title, 64)
    description = product_tokens(product.description, 256)
    bullet = product_tokens(product.bullet, 96)
    brand = frozenset(product_tokens(product.brand, 16))
    color = frozenset(product_tokens(product.color, 16))
    # Opening title terms often identify the item before attribute-heavy free text.
    type_tokens = frozenset(term for term in title[:8] if term not in brand and term not in color)
    return ProductText(
        title,
        description,
        bullet,
        brand,
        color,
        type_tokens,
        Counter(title),
        Counter(description),
    )


@dataclass(frozen=True, slots=True)
class BM25Field:
    idf: dict[str, float]
    average_length: float

    @classmethod
    def from_counts(cls, df: Counter[str], total_length: int, count: int) -> BM25Field:
        return cls(
            {term: math.log1p((count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()},
            max(total_length / count, 1.0),
        )

    def score(self, terms: Sequence[str], frequencies: Counter[str], length: int) -> float:
        length_norm = 1.2 * (0.25 + 0.75 * length / self.average_length)
        return sum(
            self.idf.get(term, 0.0) * freq * 2.2 / (freq + length_norm)
            for term in terms
            if (freq := frequencies[term])
        )


@dataclass(frozen=True, slots=True)
class Featurizer:
    title: BM25Field
    description: BM25Field

    @classmethod
    def fit(cls, products: Iterable[Candidate]) -> Featurizer:
        seen = set()
        title_df: Counter[str] = Counter()
        description_df: Counter[str] = Counter()
        title_length = description_length = 0
        for product in products:
            if product.product_id in seen:
                continue
            seen.add(product.product_id)
            prepared = prepare(product)
            title_df.update(prepared.title_counts.keys())
            description_df.update(prepared.description_counts.keys())
            title_length += len(prepared.title)
            description_length += len(prepared.description)
        count = len(seen)
        if not count:
            raise ValueError("BM25 needs at least one training product.")
        return cls(
            BM25Field.from_counts(title_df, title_length, count),
            BM25Field.from_counts(description_df, description_length, count),
        )

    def transform(self, query: str, products: Sequence[Candidate]) -> np.ndarray:
        query_terms = tokens(query)
        query_set = set(query_terms)
        query_pairs = set(zip(query_terms, query_terms[1:], strict=False))
        number_terms = {term for term in query_set if any(char.isdigit() for char in term)}
        phrase = " " + " ".join(query_terms) + " "
        denominator = max(len(query_set), 1)
        rows = np.empty((len(products), len(FEATURE_NAMES)), dtype=np.float32)
        unique_terms = sorted(query_set)
        for index, product in enumerate(products):
            text = prepare(product)
            title_score = self.title.score(unique_terms, text.title_counts, len(text.title))
            description_score = self.description.score(
                unique_terms, text.description_counts, len(text.description)
            )
            title_set = set(text.title)
            matches = len(query_set & title_set)
            ordered = zip(text.title, text.title[1:], strict=False)
            rows[index] = (
                2 * title_score + description_score,
                title_score,
                description_score,
                matches / denominator,
                len(query_set.intersection(text.description)) / denominator,
                len(query_set.intersection(text.bullet)) / denominator,
                matches / max(len(title_set), 1),
                len(query_pairs.intersection(ordered)) / max(len(query_pairs), 1),
                bool(query_terms) and phrase in (" " + " ".join(text.title) + " "),
                bool(query_terms) and query_terms == text.title,
                len(query_set & text.brand) / denominator,
                len(query_set & text.color) / denominator,
                len(query_set & text.product_type) / denominator,
                len(number_terms & title_set) / max(len(number_terms), 1),
                len(text.title),
            )
        return rows

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as target:
            json.dump(
                {
                    "version": 1,
                    "features": FEATURE_NAMES,
                    "title": {"idf": self.title.idf, "average_length": self.title.average_length},
                    "description": {
                        "idf": self.description.idf,
                        "average_length": self.description.average_length,
                    },
                },
                target,
                separators=(",", ":"),
            )

    @classmethod
    def load(cls, path: Path) -> Featurizer:
        if path.exists():
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
        else:
            with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as source:
                data = json.load(source)
        if data["version"] != 1 or tuple(data["features"]) != FEATURE_NAMES:
            raise ValueError("The feature artifact does not match this package.")
        return cls(title=BM25Field(**data["title"]), description=BM25Field(**data["description"]))


def rank_indices(scores: Sequence[float], products: Sequence[Candidate]) -> list[int]:
    return sorted(
        range(len(products)), key=lambda index: (-scores[index], products[index].product_id)
    )
