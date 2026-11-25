"""Audit the storefront demo against a fixed set of catalog judgments."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from threading import Lock

from .model import Ranker
from .relations import MODEL, PairRankers
from .store import StoreEngine, complement_allowed

STRATEGIES = ("bm25", "dense", "hybrid", "reranked")


def ndcg(order: Sequence[int], grades: dict[int, int]) -> float:
    def gain(values: Sequence[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values[:5]))

    optimum = gain(sorted(grades.values(), reverse=True))
    return gain([grades.get(item_id, 0) for item_id in order]) / optimum if optimum else 0.0


def audit(engine: StoreEngine, judgments: dict) -> dict:
    scores = {strategy: [] for strategy in STRATEGIES}
    cases = []
    for case in judgments["search"]:
        query = case["query"]
        grades = {int(item_id): grade for item_id, grade in case["grades"].items()}
        row = {}
        for strategy in STRATEGIES:
            ids = [product.id for product in engine.search(query, strategy=strategy)]
            score = ndcg(ids, grades)
            scores[strategy].append(score)
            row[strategy] = {"ndcg_at_5": score, "top5": ids[:5]}
        cases.append({"query": query, "runs": row})

    judged = {case["anchor"]: case for case in judgments["relationships"]}
    seen = {"similar": [], "complements": []}
    violations = []
    reviewed = []
    size = len(engine.products)
    for anchor in engine.products.values():
        similar, complements = engine.recommendations(anchor)
        if any(
            row.product.id == anchor.id or not row.product.stock or row.product.kind != anchor.kind
            for row in similar
        ):
            violations.append({"anchor": anchor.id, "error": "invalid similar result"})
        if any(not complement_allowed(anchor, row.product) for row in complements):
            violations.append({"anchor": anchor.id, "error": "invalid complement"})
        if anchor.id not in judged:
            continue
        case = judged[anchor.id]
        by_mode = {
            "similar": [row.product.id for row in similar],
            "complements": [row.product.id for row in complements],
        }
        if set(by_mode["complements"]) & set(case["forbidden"]):
            violations.append({"anchor": anchor.id, "error": "manually forbidden complement"})
        for mode, ids in by_mode.items():
            expected = case[mode]
            hit = bool(set(ids) & set(expected)) if expected else not ids
            seen[mode].append(hit)
        reviewed.append({"anchor": anchor.id, **by_mode})

    return {
        "catalog_products": size,
        "encoder": MODEL,
        "search": {
            "judged_queries": len(cases),
            "mean_ndcg_at_5": {name: math.fsum(part) / len(part) for name, part in scores.items()},
            "cases": cases,
        },
        "recommendations": {
            "catalog_products_checked": size,
            "judged_anchors": len(reviewed),
            "intent_checks_passed": {mode: sum(values) for mode, values in seen.items()},
            "violations": violations,
            "cases": reviewed,
        },
    }


def run_audit(
    storefront: Path, index_dir: Path, cache_dir: Path, model_dir: Path, output: Path
) -> None:
    engine = StoreEngine(
        storefront / "catalog.json",
        index_dir,
        cache_dir,
        Ranker.load(model_dir),
        Lock(),
        PairRankers.load(model_dir),
    )
    try:
        judgments = json.loads((storefront / "judgments.json").read_text(encoding="utf-8"))
        report = audit(engine, judgments)
    finally:
        engine.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "catalog_products": report["catalog_products"],
                "ndcg_at_5": report["search"]["mean_ndcg_at_5"],
                "recommendations": {
                    key: value for key, value in report["recommendations"].items() if key != "cases"
                },
            },
            indent=2,
        )
    )
