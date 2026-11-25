"""Command-line entry points for reproducing and serving the benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import normalize_query
from .dataset import QueryGroup, QuerySplits, download, load_groups, query_hash, split_groups


def smoke_sample(split: QuerySplits, size: int) -> QuerySplits:
    def pick(groups: tuple[QueryGroup, ...]) -> tuple[QueryGroup, ...]:
        return tuple(
            sorted(
                groups,
                key=lambda group: (
                    query_hash(normalize_query(group.query)),
                    group.query_id,
                ),
            )[:size]
        )

    return QuerySplits(
        pick(split.train),
        pick(split.validation),
        pick(split.test),
        split.overlap_dropped,
    )


def train(data_dir: Path, output_dir: Path, smoke_queries: int) -> None:
    from .evaluate import benchmark
    from .model import train_ranker

    print(f"Loading US Task 1 candidate lists from {data_dir}", flush=True)
    split = split_groups(load_groups(data_dir))
    if smoke_queries:
        split = smoke_sample(split, smoke_queries)
    print(
        f"Training on {len(split.train):,} queries with {len(split.validation):,} for validation",
        flush=True,
    )
    ranker = train_ranker(split.train, split.validation)
    print(f"Evaluating {len(split.test):,} held-out queries", flush=True)
    report = benchmark(ranker, split, "smoke" if smoke_queries else "full")
    ranker.save(output_dir)
    report_path = output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scores = report["slices"]["all"]
    print(
        f"nDCG@10  BM25 {scores['bm25']:.4f}  LambdaRank {scores['lambdarank']:.4f}  "
        f"delta {scores['delta']:+.4f}"
    )
    print(f"Saved the model and report in {output_dir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Fetch and verify the pinned Amazon ESCI data.")
    fetch.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    fit = commands.add_parser("train", help="Train LambdaRank and evaluate BM25 and LambdaRank.")
    fit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    fit.add_argument("--smoke-queries", type=int, default=0, help="Limit each split to N queries.")
    for command, help_text in (
        ("serve", "Serve the V1 benchmark and inference API."),
        ("serve-store", "Serve the storefront and product recommendations."),
    ):
        serve = commands.add_parser(command, help=help_text)
        serve.add_argument("--model-dir", type=Path, default=Path("models"))
        serve.add_argument("--site-dir", type=Path, default=Path("docs"))
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8000)
    audit = commands.add_parser("audit-store", help="Evaluate search and recommendation examples.")
    audit.add_argument("--model-dir", type=Path, default=Path("models"))
    audit.add_argument("--output", type=Path, default=Path("artifacts/store-audit.json"))
    for command in (serve, audit):
        command.add_argument("--storefront", type=Path, default=Path("storefront"))
        command.add_argument("--index-dir", type=Path, default=Path("data/store-index"))
        command.add_argument("--cache-dir", type=Path, default=Path("data/store-models"))
    pairs = commands.add_parser("train-store", help="Train V2 item-to-item rerankers on ESCI.")
    pairs.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    pairs.add_argument("--cache-dir", type=Path, default=Path("data/store-models"))
    pairs.add_argument("--output-dir", type=Path, default=Path("artifacts/store-training"))
    pairs.add_argument("--queries", type=int, default=780)
    args = parser.parse_args()

    if args.command == "train" and args.smoke_queries < 0:
        parser.error("--smoke-queries must be zero or a positive integer.")
    try:
        if args.command == "download":
            download(args.data_dir)
        elif args.command == "train":
            train(args.data_dir, args.output_dir, args.smoke_queries)
        elif args.command == "train-store":
            from .train_store import train_pairs

            if args.queries < 60:
                parser.error("--queries must be at least 60.")
            train_pairs(args.data_dir, args.cache_dir, args.output_dir, args.queries)
        elif args.command == "audit-store":
            from .store_eval import run_audit

            run_audit(args.storefront, args.index_dir, args.cache_dir, args.model_dir, args.output)
        else:
            import uvicorn

            from .web import create_app

            options = None
            if args.command == "serve-store":
                from .store_web import StoreOptions

                options = StoreOptions(args.storefront, args.index_dir, args.cache_dir)
            uvicorn.run(
                create_app(args.model_dir, args.site_dir, options), host=args.host, port=args.port
            )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
