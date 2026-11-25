"""Held-out candidate ranking and the benchmark report used by the demo."""

from __future__ import annotations

import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np

from .core import normalize_query, product_tokens, rank_indices
from .dataset import GAINS, LABELS, REVISION, QueryGroup, QuerySplits
from .model import Ranker, build_matrix

GAIN_BY_LABEL = dict(zip(LABELS, GAINS, strict=True))
DISCOUNTS = tuple(1 / math.log2(index + 2) for index in range(10))


def ndcg_at_10(labels: Sequence[str]) -> float:
    gains = [GAIN_BY_LABEL[label] for label in labels]
    dcg = sum(gain * discount for gain, discount in zip(gains, DISCOUNTS, strict=False))
    ideal = sum(
        gain * discount
        for gain, discount in zip(sorted(gains, reverse=True), DISCOUNTS, strict=False)
    )
    return dcg / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class Outcome:
    group: QueryGroup
    bm25_order: list[int]
    learned_order: list[int]
    bm25: float
    learned: float
    normalized: str
    exact_title: bool

    @property
    def short(self) -> bool:
        return 1 <= len(self.normalized.split()) <= 2

    @property
    def delta(self) -> float:
        return self.learned - self.bm25


def evaluate(ranker: Ranker, queries: Sequence[QueryGroup]) -> list[Outcome]:
    matrix, sizes = build_matrix(queries, ranker.features)
    predictions = ranker.booster.predict(matrix, num_threads=4)
    offset = 0
    outcomes = []
    for group, size in zip(queries, sizes, strict=True):
        stop = offset + size
        baseline = rank_indices(matrix[offset:stop, 0], group.candidates)
        learned = rank_indices(predictions[offset:stop], group.candidates)
        normalized = normalize_query(group.query)
        exact = (
            any(
                normalized == " ".join(product_tokens(product.title, 64))
                for product in group.candidates
            )
            if normalized
            else False
        )
        outcomes.append(
            Outcome(
                group,
                baseline,
                learned,
                ndcg_at_10([group.labels[index] for index in baseline]),
                ndcg_at_10([group.labels[index] for index in learned]),
                normalized,
                exact,
            )
        )
        offset = stop
    return outcomes


def summarize(outcomes: Sequence[Outcome]) -> dict:
    if not outcomes:
        return {"queries": 0, "bm25": None, "lambdarank": None, "delta": None}
    count = len(outcomes)
    baseline = math.fsum(outcome.bm25 for outcome in outcomes) / count
    learned = math.fsum(outcome.learned for outcome in outcomes) / count
    improvements = sum(outcome.delta > 1e-12 for outcome in outcomes)
    regressions = sum(outcome.delta < -1e-12 for outcome in outcomes)
    return {
        "queries": count,
        "bm25": baseline,
        "lambdarank": learned,
        "delta": learned - baseline,
        "improved": improvements,
        "regressed": regressions,
        "tied": count - improvements - regressions,
    }


def paired_interval(outcomes: Sequence[Outcome]) -> dict:
    # Resample normalized query clusters to preserve related held-out query IDs.
    clusters: dict[str, list[float]] = {}
    for outcome in outcomes:
        clusters.setdefault(outcome.normalized, []).append(outcome.delta)
    sums = np.array([math.fsum(values) for values in clusters.values()])
    sizes = np.array([len(values) for values in clusters.values()])
    rng = np.random.default_rng(42)
    estimates = np.empty(2000)
    for index in range(len(estimates)):
        draw = rng.integers(len(clusters), size=len(clusters))
        estimates[index] = sums[draw].sum() / sizes[draw].sum()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "resamples": len(estimates)}


def export_case(outcome: Outcome) -> dict:
    baseline = {index: rank for rank, index in enumerate(outcome.bm25_order, 1)}
    learned = {index: rank for rank, index in enumerate(outcome.learned_order, 1)}
    group = outcome.group
    products = [
        {
            "product_id": product.product_id,
            "title": product.title,
            "label": group.labels[index],
            "bm25_rank": baseline[index],
            "lambdarank_rank": learned[index],
        }
        for index, product in enumerate(group.candidates)
        if min(baseline[index], learned[index]) <= 10
    ]
    return {
        "query": group.query,
        "query_id": group.query_id,
        "candidate_count": len(group.candidates),
        "bm25": outcome.bm25,
        "lambdarank": outcome.learned,
        "products": products,
    }


def pick_cases(outcomes: Sequence[Outcome]) -> list[dict]:
    usable = [o for o in outcomes if o.normalized and len(o.group.candidates) >= 10]
    if not usable:
        usable = list(outcomes)
    picked = []
    seen = set()
    ordered = sorted(usable, key=lambda o: (o.delta, o.normalized, o.group.query_id))
    pools = [
        reversed(ordered),
        iter(ordered),
        (o for o in reversed(ordered) if o.short),
        (o for o in reversed(ordered) if o.exact_title),
        (o for o in reversed(ordered) if len(o.normalized.split()) > 2),
        (o for o in ordered if o.exact_title),
    ]
    for pool in pools:
        choice = next((o for o in pool if o.normalized not in seen), None)
        if choice:
            picked.append(export_case(choice))
            seen.add(choice.normalized)
    return picked


def benchmark(ranker: Ranker, split: QuerySplits, mode: str) -> dict:
    outcomes = evaluate(ranker, split.test)
    if not outcomes:
        raise ValueError("The held-out split contains no queries.")
    parts = {"train": split.train, "validation": split.validation, "test": split.test}
    return {
        "dataset": {"name": "Amazon ESCI · US Task 1", "revision": REVISION, "mode": mode},
        "splits": {
            name: {"queries": len(groups), "pairs": sum(len(g.candidates) for g in groups)}
            for name, groups in parts.items()
        },
        "overlap_dropped_train_queries": split.overlap_dropped,
        "gains": GAIN_BY_LABEL,
        "iterations": ranker.booster.best_iteration,
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("lightgbm", "numpy", "pyarrow")},
        },
        "slices": {
            "all": summarize(outcomes),
            "short": summarize([outcome for outcome in outcomes if outcome.short]),
            "exact_title": summarize([outcome for outcome in outcomes if outcome.exact_title]),
        },
        "bootstrap": paired_interval(outcomes),
        "cases": pick_cases(outcomes),
    }
