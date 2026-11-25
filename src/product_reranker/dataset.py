"""Pinned Amazon ESCI data and leakage-safe query splits."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .core import Candidate, normalize_query

REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
DATASET_URL = (
    f"https://media.githubusercontent.com/media/amazon-science/esci-data/{REVISION}"
    "/shopping_queries_dataset/"
)
EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
CHECKSUMS = {
    EXAMPLES: "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    PRODUCTS: "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
}
LABELS = ("I", "C", "S", "E")
GAINS = (0, 1, 10, 100)


@dataclass(frozen=True, slots=True)
class QueryGroup:
    query_id: str
    query: str
    candidates: tuple[Candidate, ...]
    labels: tuple[str, ...]
    official_split: str


@dataclass(frozen=True, slots=True)
class QuerySplits:
    train: tuple[QueryGroup, ...]
    validation: tuple[QueryGroup, ...]
    test: tuple[QueryGroup, ...]
    overlap_dropped: int


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        path = data_dir / name
        if path.exists() and file_digest(path) == checksum:
            print(f"Verified {path}", flush=True)
            continue

        print(f"Downloading {name}", flush=True)
        with NamedTemporaryFile(
            dir=data_dir, prefix=name + ".", suffix=".part", delete=False
        ) as out:
            temporary = Path(out.name)
        try:
            with urlopen(DATASET_URL + name, timeout=180) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if file_digest(temporary) != checksum:
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def query_hash(normalized: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(normalized.encode(), digest_size=8, person=b"esci-ml03").digest(), "big"
    )


def split_groups(groups: Sequence[QueryGroup]) -> QuerySplits:
    if any(group.official_split not in {"train", "test"} for group in groups):
        raise ValueError("ESCI examples must use the official train or test split.")
    test = tuple(group for group in groups if group.official_split == "test")
    test_texts = {normalize_query(group.query) for group in test}
    train = []
    validation = []
    dropped = 0
    for group in groups:
        if group.official_split == "test":
            continue
        normalized = normalize_query(group.query)
        if normalized in test_texts:
            dropped += 1
        elif query_hash(normalized) % 100 < 15:
            validation.append(group)
        else:
            train.append(group)
    return QuerySplits(tuple(train), tuple(validation), test, dropped)


def _load_products(path: Path, ids: set[str]) -> dict[str, Candidate]:
    columns = [
        "product_id",
        "product_locale",
        "product_title",
        "product_description",
        "product_brand",
        "product_color",
        "product_bullet_point",
    ]
    products = {}
    values = pa.array(sorted(ids), type=pa.string())
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
        matches = pc.and_(
            pc.equal(batch.column(1), "us"),
            pc.is_in(batch.column(0), value_set=values),
        )
        for row in batch.filter(matches).to_pylist():
            product_id = row["product_id"]
            if product_id in products:
                raise ValueError(f"Duplicate US catalog entry: {product_id}")
            products[product_id] = Candidate(
                product_id,
                row["product_title"] or "",
                (row["product_description"] or "")[:12000],
                row["product_brand"] or "",
                row["product_color"] or "",
                (row["product_bullet_point"] or "")[:12000],
            )
    if missing := ids.difference(products):
        raise ValueError(f"Missing {len(missing)} products from the US catalog.")
    return products


def load_groups(
    data_dir: Path, select: Callable[[pa.Table], pa.Table] | None = None
) -> tuple[QueryGroup, ...]:
    for filename in CHECKSUMS:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run python -m product_reranker download.")
    examples = pq.read_table(
        data_dir / EXAMPLES,
        columns=["query_id", "query", "product_id", "esci_label", "split"],
        filters=[("small_version", "=", 1), ("product_locale", "=", "us")],
    )
    if select is not None:
        examples = select(examples)
    examples = examples.sort_by(
        [("split", "ascending"), ("query_id", "ascending"), ("product_id", "ascending")]
    )
    products = _load_products(data_dir / PRODUCTS, set(examples["product_id"].to_pylist()))
    groups = []
    rows = (row for batch in examples.to_batches() for row in batch.to_pylist())
    for (split, query_id), matches in groupby(rows, lambda row: (row["split"], row["query_id"])):
        items = list(matches)
        query = items[0]["query"]
        ids = [item["product_id"] for item in items]
        labels = tuple(item["esci_label"] for item in items)
        if (
            any(item["query"] != query for item in items)
            or len(set(ids)) != len(ids)
            or not set(labels).issubset(LABELS)
        ):
            raise ValueError(f"Invalid candidate list for query {query_id}")
        groups.append(
            QueryGroup(
                str(query_id),
                query,
                tuple(products[product_id] for product_id in ids),
                labels,
                split,
            )
        )
    return tuple(groups)
