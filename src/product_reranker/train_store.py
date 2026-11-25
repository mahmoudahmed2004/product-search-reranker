"""Train and evaluate two supervised item-to-item rankers using ESCI projections."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fastembed import TextEmbedding

from .core import Candidate
from .dataset import REVISION, QueryGroup, load_groups, query_hash, split_groups
from .relations import MODEL, PAIR_NAMES, PairProduct, pair_features

GRADES = {
    "similar": {"I": 0, "C": 0, "S": 1, "E": 2},
    "complements": {"I": 0, "C": 1, "S": 0, "E": 0},
}


@dataclass(frozen=True, slots=True)
class PairGroup:
    anchor: Candidate
    candidates: tuple[Candidate, ...]
    labels: tuple[str, ...]


def sample_queries(table: pa.Table, count: int) -> pa.Table:
    # Sample whole original query IDs before reading the large product catalog.
    metadata = table.select(["query_id", "split", "esci_label"])
    exact = metadata.filter(pc.equal(metadata["esci_label"], "E"))
    partitions = {
        split: set(exact.filter(pc.equal(exact["split"], split))["query_id"].to_pylist())
        for split in ("train", "test")
    }
    conditions = []
    for split, keys in partitions.items():
        limit = count if split == "train" else max(count // 3, 1)
        selected = sorted(keys, key=lambda key: query_hash(f"pairs-{split}-{key}"))[:limit]
        conditions.append(
            pc.and_(
                pc.equal(table["split"], split),
                pc.is_in(table["query_id"], value_set=pa.array(selected, type=pa.int64())),
            )
        )
    return table.filter(pc.or_(*conditions))


def project(groups: Sequence[QueryGroup]) -> list[PairGroup]:
    projected = []
    for group in groups:
        exact = [index for index, label in enumerate(group.labels) if label == "E"]
        if len(group.candidates) < 2 or not exact:
            continue
        selected = min(
            exact,
            key=lambda index: query_hash(group.query_id + group.candidates[index].product_id),
        )
        candidates = group.candidates[:selected] + group.candidates[selected + 1 :]
        labels = group.labels[:selected] + group.labels[selected + 1 :]
        projected.append(PairGroup(group.candidates[selected], candidates, labels))
    return projected


def disjoint_anchors(
    train: list[PairGroup], validation: list[PairGroup], test: list[PairGroup]
) -> tuple[list[PairGroup], list[PairGroup], list[PairGroup]]:
    def without_overlap(groups: list[PairGroup], held: list[PairGroup]) -> list[PairGroup]:
        held_anchors = {group.anchor.product_id for group in held}
        held_pairs = {
            (product.product_id, group.anchor.product_id)
            for group in held
            for product in group.candidates
        }
        result = []
        for group in groups:
            anchor_id = group.anchor.product_id
            if anchor_id in held_anchors:
                continue
            keep = [
                (product, label)
                for product, label in zip(group.candidates, group.labels, strict=True)
                if (anchor_id, product.product_id) not in held_pairs
            ]
            if keep:
                result.append(
                    PairGroup(
                        group.anchor, tuple(p for p, _ in keep), tuple(label for _, label in keep)
                    )
                )
        return result

    validation = without_overlap(validation, test)
    train = without_overlap(train, validation + test)
    return train, validation, test


def catalog_vectors(
    parts: Sequence[Sequence[PairGroup]], cache: Path
) -> tuple[dict[str, PairProduct], dict[str, np.ndarray]]:
    products = {}
    for groups in parts:
        for group in groups:
            for item in (group.anchor, *group.candidates):
                products[item.product_id] = PairProduct(item.title, item.description, item.brand)
    encoder = TextEmbedding(MODEL, cache_dir=str(cache), threads=4)
    keys = sorted(products, key=lambda key: len(products[key].text()))
    vectors = {}
    total = len(keys)
    print(f"Encoding {total:,} unique products on CPU", flush=True)
    for index, vector in enumerate(
        encoder.embed((products[key].text() for key in keys), batch_size=32)
    ):
        vectors[keys[index]] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if (index + 1) % 1500 == 0:
            print(f"Encoded {index + 1:,}/{total:,} products", flush=True)
    return products, vectors


def as_pair(item: Candidate) -> PairProduct:
    return PairProduct(item.title, item.description, item.brand)


def prepare(groups: Sequence[PairGroup]) -> list[np.ndarray]:
    return [
        pair_features(as_pair(group.anchor), [as_pair(product) for product in group.candidates])
        for group in groups
    ]


def ranking_data(groups: Sequence[PairGroup], matrices: Sequence[np.ndarray], grades: dict):
    labels = []
    features = []
    counts = []
    for group, matrix in zip(groups, matrices, strict=True):
        values = [grades[label] for label in group.labels]
        if len(set(values)) < 2:
            continue
        labels.extend(values)
        features.append(matrix)
        counts.append(len(values))
    return np.concatenate(features), labels, counts


def ndcg5(labels: Sequence[int], scores: Sequence[float]) -> float:
    def dcg(ordered: Sequence[int]) -> float:
        return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ordered[:5]))

    ideal = dcg(sorted(labels, reverse=True))
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    return dcg([labels[index] for index in order]) / ideal if ideal else 0.0


def fit_one(name: str, parts, matrices, output_dir: Path) -> dict:
    grades = GRADES[name]
    train_x, train_y, train_q = ranking_data(parts[0], matrices[0], grades)
    val_x, val_y, val_q = ranking_data(parts[1], matrices[1], grades)
    training = lgb.Dataset(train_x, label=train_y, group=train_q, feature_name=list(PAIR_NAMES))
    validation = lgb.Dataset(val_x, label=val_y, group=val_q, reference=training)
    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "num_leaves": 15,
            "min_data_in_leaf": 24,
            "learning_rate": 0.05,
            "verbosity": -1,
            "num_threads": 4,
            "deterministic": True,
            "force_col_wise": True,
            "seed": 42,
        },
        training,
        300,
        valid_sets=[validation],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    model.save_model(str(output_dir / f"v2-{name}.txt"))
    outcomes = []
    for group, matrix in zip(parts[2], matrices[2], strict=True):
        labels = [grades[label] for label in group.labels]
        if len(set(labels)) < 2:
            continue
        outcomes.append(
            (ndcg5(labels, matrix[:, 0]), ndcg5(labels, model.predict(matrix, num_threads=1)))
        )
    baseline, learned = np.asarray(outcomes).mean(axis=0).tolist()
    return {
        "train_groups": len(train_q),
        "train_pairs": len(train_y),
        "validation_groups": len(val_q),
        "test_groups": len(outcomes),
        "title_char_cosine_ndcg5": baseline,
        "learned_ndcg5": learned,
        "delta": learned - baseline,
        "iterations": model.best_iteration,
    }


def compare_dense(test: list[PairGroup], cache: Path, output_dir: Path) -> dict:
    subset = sorted(test, key=lambda group: query_hash("dense-audit-" + group.anchor.product_id))
    sampled = {
        name: [group for group in subset if len({grades[label] for label in group.labels}) > 1][:14]
        for name, grades in GRADES.items()
    }
    _, vectors = catalog_vectors(list(sampled.values()), cache)
    report = {}
    for name, groups in sampled.items():
        booster = lgb.Booster(model_file=str(output_dir / f"v2-{name}.txt"))
        scores = []
        for group, matrix in zip(groups, prepare(groups), strict=True):
            labels = [GRADES[name][label] for label in group.labels]
            embeddings = np.stack([vectors[p.product_id] for p in group.candidates])
            dense = embeddings @ vectors[group.anchor.product_id]
            learned = booster.predict(matrix, num_threads=1)
            scores.append((ndcg5(labels, dense), ndcg5(labels, learned)))
        dense_score, learned_score = np.asarray(scores).mean(axis=0).tolist()
        report[name] = {
            "groups": len(groups),
            "dense_ndcg5": dense_score,
            "learned_ndcg5": learned_score,
        }
    return report


def train_pairs(data: Path, cache: Path, output_dir: Path, count: int) -> None:
    split = split_groups(load_groups(data, lambda table: sample_queries(table, count)))
    parts = disjoint_anchors(project(split.train), project(split.validation), project(split.test))
    print(f"Projected groups after anchor isolation: {[len(part) for part in parts]}", flush=True)
    matrices = [prepare(part) for part in parts]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {name: fit_one(name, parts, matrices, output_dir) for name in GRADES}
    report = {
        "dataset": "Amazon ESCI US Task 1",
        "revision": REVISION,
        "encoder": MODEL,
        "train_sample_budget": count,
        "projected_groups": dict(
            zip(("train", "validation", "test"), (len(part) for part in parts), strict=True)
        ),
        "method": (
            "Pair labels projected from shopping-query judgments using one Exact product per query."
        ),
        "metrics": results,
    }
    (output_dir / "v2-training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    report["dense_review"] = compare_dense(parts[2], cache, output_dir)
    (output_dir / "v2-training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Dense comparison: " + json.dumps(report["dense_review"], indent=2), flush=True)
