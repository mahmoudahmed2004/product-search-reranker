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


def ndcg(order: Sequence[int], grades: dict[int, int], cutoff: int = 5) -> float:
    def gain(values: Sequence[int]) -> float:
        return sum(
            (2**value - 1) / math.log2(index + 2) for index, value in enumerate(values[:cutoff])
        )

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


def audit_home_context(engine: StoreEngine, judgments: list[dict]) -> dict:
    rows = []
    for case in judgments:
        query = case["query"]
        topic, items = engine.home_search(query)
        _, gated_retrieval = engine.home_search(query, ranking="retrieval")
        _, unguarded_learned = engine.home_search(query, ranking="learned")
        baseline = engine.search(query, strategy="hybrid")[:4]
        target = case["collection"]
        ids = {product.id for product in items}
        expected = set(case.get("expected", []))
        grades = dict.fromkeys(expected, 1)
        rows.append(
            {
                "query": query,
                "expected_collection": target,
                "selected_collection": topic,
                "context_ids": [product.id for product in items],
                "hybrid_ids": [product.id for product in baseline],
                "gated_retrieval_ids": [product.id for product in gated_retrieval],
                "unguarded_learned_ids": [product.id for product in unguarded_learned],
                "unguarded_learned_ndcg_at_4": ndcg(
                    [product.id for product in unguarded_learned], grades, cutoff=4
                ),
                "context_ndcg_at_4": ndcg([product.id for product in items], grades, cutoff=4),
                "gated_retrieval_ndcg_at_4": ndcg(
                    [product.id for product in gated_retrieval], grades, cutoff=4
                ),
                "context_in_department": sum(product.collection == target for product in items),
                "hybrid_in_department": sum(product.collection == target for product in baseline),
                "context_expected_hit": bool(ids & expected),
                "hybrid_expected_hit": any(product.id in expected for product in baseline),
            }
        )

    positive = [row for row in rows if row["expected_collection"] is not None]
    negative = [row for row in rows if row["expected_collection"] is None]
    metrics = {
        "matching_queries": len(positive),
        "ambiguous_or_out_of_catalog_queries": len(negative),
        "correct_topics": sum(
            row["expected_collection"] == row["selected_collection"] and bool(row["context_ids"])
            for row in positive
        ),
        "abstentions_on_unmatched_queries": sum(not row["context_ids"] for row in negative),
        "hybrid_abstentions_on_unmatched_queries": sum(not row["hybrid_ids"] for row in negative),
        "homepage_reranker": {
            key: math.fsum(row[key] for row in positive) / len(positive)
            for key in (
                "gated_retrieval_ndcg_at_4",
                "unguarded_learned_ndcg_at_4",
                "context_ndcg_at_4",
            )
        },
    }
    for mode in ("context", "hybrid"):
        count = sum(len(row[f"{mode}_ids"]) for row in positive)
        correct = sum(row[f"{mode}_in_department"] for row in positive)
        metrics[mode] = {
            "returned_items": count,
            "in_department": correct,
            "department_precision": correct / count if count else 0.0,
            "expected_product_hit_queries": sum(row[f"{mode}_expected_hit"] for row in positive),
        }
    return {"metrics": metrics, "cases": rows}


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
        context_path = storefront / "home-judgments.json"
        if context_path.is_file():
            cases = json.loads(context_path.read_text(encoding="utf-8"))
            report["home_context"] = audit_home_context(engine, cases)
    finally:
        engine.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "catalog_products": report["catalog_products"],
                "home_context": report.get("home_context", {}).get("metrics"),
                "ndcg_at_5": report["search"]["mean_ndcg_at_5"],
                "recommendations": {
                    key: value for key, value in report["recommendations"].items() if key != "cases"
                },
            },
            indent=2,
        )
    )
