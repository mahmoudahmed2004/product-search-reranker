"""Training and inference for the shared candidate-ranking pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .core import FEATURE_NAMES, Candidate, Featurizer, rank_indices
from .dataset import GAINS, LABELS, QueryGroup


@dataclass(frozen=True, slots=True)
class RankedProduct:
    product_id: str
    title: str
    bm25_rank: int
    lambdarank_rank: int


@dataclass(frozen=True, slots=True)
class Ranker:
    features: Featurizer
    booster: lgb.Booster

    def rank(self, query: str, products: Sequence[Candidate]) -> list[RankedProduct]:
        matrix = self.features.transform(query, products)
        predictions = self.booster.predict(matrix, num_threads=1)
        baseline = rank_indices(matrix[:, 0], products)
        learned = rank_indices(predictions, products)
        original = {index: rank for rank, index in enumerate(baseline, 1)}
        return [
            RankedProduct(products[index].product_id, products[index].title, original[index], rank)
            for rank, index in enumerate(learned, 1)
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.features.save(directory / "features.json")
        self.booster.save_model(str(directory / "ranker.txt"))

    @classmethod
    def load(cls, directory: Path) -> Ranker:
        model_file = directory / "ranker.txt"
        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        features = Featurizer.load(directory / "features.json")
        booster = lgb.Booster(model_file=str(model_file))
        if tuple(booster.feature_name()) != FEATURE_NAMES:
            raise ValueError("The trained model does not match the feature schema.")
        return cls(features, booster)


def build_matrix(
    groups: Sequence[QueryGroup], features: Featurizer
) -> tuple[np.ndarray, list[int]]:
    sizes = [len(group.candidates) for group in groups]
    matrix = np.empty((sum(sizes), len(FEATURE_NAMES)), dtype=np.float32)
    offset = 0
    for group, size in zip(groups, sizes, strict=True):
        matrix[offset : offset + size] = features.transform(group.query, group.candidates)
        offset += size
    return matrix, sizes


def training_dataset(
    groups: Sequence[QueryGroup], features: Featurizer, reference: lgb.Dataset | None = None
) -> lgb.Dataset:
    matrix, sizes = build_matrix(groups, features)
    labels = [LABELS.index(label) for group in groups for label in group.labels]
    return lgb.Dataset(
        matrix,
        label=labels,
        group=sizes,
        feature_name=list(FEATURE_NAMES),
        reference=reference,
    )


def train_ranker(train: Sequence[QueryGroup], validation: Sequence[QueryGroup]) -> Ranker:
    if not train or not validation:
        raise ValueError("Training and validation both need labeled query groups.")
    features = Featurizer.fit(product for group in train for product in group.candidates)
    train_data = training_dataset(train, features)
    validation_data = training_dataset(validation, features, reference=train_data)
    booster = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "label_gain": list(GAINS),
            "lambdarank_truncation_level": 13,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "num_threads": 4,
            "seed": 42,
            "deterministic": True,
            "force_col_wise": True,
        },
        train_data,
        num_boost_round=400,
        valid_sets=[validation_data],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return Ranker(features, booster)
