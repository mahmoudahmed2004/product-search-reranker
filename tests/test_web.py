from pathlib import Path
from shutil import copyfile

import pytest
from fastapi.testclient import TestClient

from product_reranker.core import Candidate
from product_reranker.dataset import QueryGroup
from product_reranker.model import train_ranker
from product_reranker.web import create_app


def training_groups(start: int, stop: int) -> list[QueryGroup]:
    groups = []
    for index in range(start, stop):
        color = ("black", "white", "emerald")[index % 3]
        query = f"{color} mechanical keyboard {index}"
        groups.append(
            QueryGroup(
                str(index),
                query,
                (
                    Candidate(f"{index:03d}-i", "Ivory insulated thermos"),
                    Candidate(f"{index:03d}-s", "Silver mechanical keyboard"),
                    Candidate(f"{index:03d}-e", query, query, color=color),
                ),
                ("I", "S", "E"),
                "train",
            )
        )
    return groups


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("ranker_model")
    ranker = train_ranker(training_groups(0, 40), training_groups(40, 50))
    ranker.save(model_dir)
    return model_dir


@pytest.fixture
def site_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "site"
    directory.mkdir()
    (directory / "index.html").write_text(
        "<!doctype html><html><body><h1>Product Reranker</h1></body></html>",
        encoding="utf-8",
    )
    return directory


def test_root_serves_site_index(trained_model_dir: Path, site_dir: Path) -> None:
    app = create_app(model_dir=trained_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Product Reranker" in response.text


@pytest.mark.parametrize("partial", [False, True])
def test_unavailable_model_reports_not_ready_and_returns_503(
    tmp_path: Path,
    site_dir: Path,
    trained_model_dir: Path,
    partial: bool,
) -> None:
    missing_model_dir = tmp_path / "missing-model"
    if partial:
        missing_model_dir.mkdir()
        copyfile(trained_model_dir / "features.json", missing_model_dir / "features.json")
    app = create_app(model_dir=missing_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        health = client.get("/api/health")
        rank = client.post(
            "/api/rank",
            json={
                "query": "black mechanical keyboard",
                "candidates": [
                    {"title": "Apex Black Mechanical Keyboard"},
                    {"title": "Lumen Ivory Insulated Thermos"},
                ],
            },
        )

    assert health.status_code == 200
    assert health.json() == {"ready": False}
    assert rank.status_code == 503


def test_health_ready_when_model_loaded(trained_model_dir: Path, site_dir: Path) -> None:
    app = create_app(model_dir=trained_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        health = client.get("/api/health")

    assert health.status_code == 200
    assert health.json() == {"ready": True}


def test_rank_real_inference_order_title_parity_and_permutations(
    trained_model_dir: Path,
    site_dir: Path,
) -> None:
    exact_title = "Lumen Emerald Mechanical Keyboard"
    substitute_title = "Orbis Silver Mechanical Keyboard"
    irrelevant_title = "Boreal Ivory Insulated Thermos"
    payload = {
        "query": "  emerald Lumen mechanical keyboard  ",
        "candidates": [
            {"title": irrelevant_title},
            {
                "title": substitute_title,
                "brand": "Orbis",
                "color": "silver",
            },
            {
                "title": exact_title,
                "description": (
                    "Official Lumen mechanical keyboard finished in emerald "
                    "for everyday performance."
                ),
                "brand": "Lumen",
                "color": "emerald",
                "bullet": "Genuine Lumen keyboard with mechanical build and emerald exterior.",
            },
        ],
    }

    app = create_app(model_dir=trained_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        response = client.post("/api/rank", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "emerald Lumen mechanical keyboard"
    assert body["candidate_count"] == 3

    products = body["products"]
    assert len(products) == 3
    for product in products:
        assert {"product_id", "title", "bm25_rank", "lambdarank_rank"} <= set(product)

    returned_titles = [product["title"] for product in products]
    assert set(returned_titles) == {exact_title, substitute_title, irrelevant_title}
    assert {product["product_id"] for product in products} == {"001", "002", "003"}
    assert [product["lambdarank_rank"] for product in products] == [1, 2, 3]
    assert sorted(product["bm25_rank"] for product in products) == [1, 2, 3]


@pytest.mark.parametrize("candidate_count", [2, 40])
def test_rank_boundary_candidate_counts(
    trained_model_dir: Path,
    site_dir: Path,
    candidate_count: int,
) -> None:
    titles = [f"Model {index} mechanical keyboard" for index in range(candidate_count)]
    payload = {
        "query": "Apex mechanical keyboard",
        "candidates": [{"title": title} for title in titles],
    }

    app = create_app(model_dir=trained_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        response = client.post("/api/rank", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Apex mechanical keyboard"
    assert body["candidate_count"] == candidate_count

    products = body["products"]
    expected_ranks = list(range(1, candidate_count + 1))
    assert len(products) == candidate_count
    assert {product["title"] for product in products} == set(titles)
    assert {product["product_id"] for product in products} == {
        f"{rank:03d}" for rank in expected_ranks
    }
    assert [product["lambdarank_rank"] for product in products] == expected_ranks
    assert sorted(product["bm25_rank"] for product in products) == expected_ranks


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "query": "",
                "candidates": [{"title": "Alpha Desk Lamp"}, {"title": "Beta Floor Lamp"}],
            },
            id="empty-query",
        ),
        pytest.param(
            {
                "query": "   \t  ",
                "candidates": [{"title": "Alpha Desk Lamp"}, {"title": "Beta Floor Lamp"}],
            },
            id="whitespace-query",
        ),
        pytest.param(
            {
                "query": "!!! --- ???",
                "candidates": [{"title": "Alpha Desk Lamp"}, {"title": "Beta Floor Lamp"}],
            },
            id="punctuation-only-query",
        ),
        pytest.param(
            {"query": "desk lamp", "candidates": []},
            id="zero-candidates",
        ),
        pytest.param(
            {"query": "desk lamp", "candidates": [{"title": "Alpha Desk Lamp"}]},
            id="one-candidate",
        ),
        pytest.param(
            {
                "query": "desk lamp",
                "candidates": [{"title": f"Model {i} mechanical keyboard"} for i in range(41)],
            },
            id="too-many-candidates",
        ),
        pytest.param(
            {
                "query": "desk lamp",
                "candidates": [{"title": "Alpha Desk Lamp"}, {"title": "Alpha Desk Lamp"}],
            },
            id="duplicate-exact-title",
        ),
        pytest.param(
            {
                "query": "desk lamp",
                "candidates": [
                    {"title": "Alpha Desk Lamp"},
                    {"title": "Beta Floor Lamp"},
                    {"title": "  alpha   desk  lamp  "},
                ],
            },
            id="duplicate-normalized-title",
        ),
    ],
)
def test_rank_validation_rejects_invalid_payloads(
    trained_model_dir: Path,
    site_dir: Path,
    payload: dict[str, object],
) -> None:
    app = create_app(model_dir=trained_model_dir, site_dir=site_dir)
    with TestClient(app) as client:
        response = client.post("/api/rank", json=payload)

    assert response.status_code == 422
