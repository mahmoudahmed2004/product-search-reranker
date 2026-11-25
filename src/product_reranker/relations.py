"""Product-pair feature extraction shared by training and storefront inference."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .core import tokens

MODEL = "ibm-granite/granite-embedding-small-english-r2"
PAIR_NAMES = (
    "title_char_cosine",
    "title_jaccard",
    "anchor_title_coverage",
    "candidate_title_coverage",
    "numeric_overlap",
    "brand_match",
    "anchor_accessory",
    "candidate_accessory",
    "candidate_description_coverage",
    "length_ratio",
)
ACCESSORY_TERMS = frozenset(
    {
        "adapter",
        "battery",
        "cable",
        "case",
        "charger",
        "cover",
        "holder",
        "mount",
        "protector",
        "refill",
        "replacement",
        "strap",
    }
)


@dataclass(frozen=True, slots=True)
class PairProduct:
    title: str
    description: str
    brand: str

    def text(self) -> str:
        return " ".join((self.title, self.brand, self.description[:420])).strip()


def character_grams(terms: Sequence[str]) -> set[str]:
    text = " " + " ".join(terms) + " "
    return {text[index : index + 3] for index in range(len(text) - 2)}


def pair_features(anchor: PairProduct, candidates: Sequence[PairProduct]) -> np.ndarray:
    anchor_words = tokens(anchor.title)
    anchor_terms = set(anchor_words)
    anchor_grams = character_grams(anchor_words)
    anchor_numbers = {term for term in anchor_terms if any(char.isdigit() for char in term)}
    anchor_brand = tokens(anchor.brand)
    rows = []
    for candidate in candidates:
        words = tokens(candidate.title)
        terms = set(words)
        grams = character_grams(words)
        similarity = len(anchor_grams & grams) / math.sqrt(max(len(anchor_grams) * len(grams), 1))
        shared = len(anchor_terms & terms)
        content = set(tokens(candidate.description[:420]))
        rows.append(
            (
                similarity,
                shared / max(len(anchor_terms | terms), 1),
                shared / max(len(anchor_terms), 1),
                shared / max(len(terms), 1),
                len(anchor_numbers & terms) / max(len(anchor_numbers), 1),
                bool(anchor_brand) and anchor_brand == tokens(candidate.brand),
                bool(anchor_terms & ACCESSORY_TERMS),
                bool(terms & ACCESSORY_TERMS),
                len(anchor_terms & content) / max(len(anchor_terms), 1),
                min(len(anchor_terms), len(terms)) / max(len(anchor_terms), len(terms), 1),
            )
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(PAIR_NAMES))


@dataclass(frozen=True, slots=True)
class PairRankers:
    similar: lgb.Booster
    complements: lgb.Booster

    @classmethod
    def load(cls, directory: Path) -> PairRankers:
        boosters = []
        for name in ("similar", "complements"):
            path = directory / f"v2-{name}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            booster = lgb.Booster(model_file=str(path))
            if tuple(booster.feature_name()) != PAIR_NAMES:
                raise ValueError(f"Invalid product-pair feature schema in {path}")
            boosters.append(booster)
        return cls(*boosters)
