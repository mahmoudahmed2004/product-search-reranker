"""Serve static benchmark cases and live predictions from a trained model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .core import Candidate, normalize_query
from .model import RankedProduct, Ranker

if TYPE_CHECKING:
    from .store_web import StoreOptions

logger = logging.getLogger(__name__)
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Title
    description: str = Field(default="", max_length=4000)
    brand: str = Field(default="", max_length=120)
    color: str = Field(default="", max_length=120)
    bullet: str = Field(default="", max_length=1000)


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Query
    candidates: list[ProductInput] = Field(min_length=2, max_length=40)

    @model_validator(mode="after")
    def validate_searchable_text(self) -> RankRequest:
        if not normalize_query(self.query):
            raise ValueError("The query must contain at least one letter or number.")
        titles = [normalize_query(product.title) for product in self.candidates]
        if not all(titles):
            raise ValueError("Each title must contain at least one letter or number.")
        if len(set(titles)) != len(titles):
            raise ValueError("Each product title must be distinct.")
        return self


class RankResponse(BaseModel):
    query: str
    candidate_count: int
    products: list[RankedProduct]


def create_app(model_dir: Path, site_dir: Path, storefront: StoreOptions | None = None) -> FastAPI:
    prediction_lock = Lock()
    ranker = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal ranker
        try:
            ranker = Ranker.load(model_dir)
        except FileNotFoundError:
            logger.warning("Model files are missing in %s. Live inference is disabled.", model_dir)
        engine = None
        if storefront is not None:
            from .relations import PairRankers
            from .store import StoreEngine

            engine = StoreEngine(
                storefront.directory / "catalog.json",
                storefront.index,
                storefront.cache,
                ranker,
                prediction_lock,
                PairRankers.load(model_dir),
            )
            _app.state.store = engine
        try:
            yield
        finally:
            if engine is not None:
                engine.close()

    app = FastAPI(
        title="Product Search Reranker",
        description="Compare BM25 with LambdaRank on an explicitly supplied candidate set.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ready": ranker is not None}

    @app.post("/api/rank")
    def rank(request: RankRequest) -> RankResponse:
        if ranker is None:
            raise HTTPException(503, "The trained model is unavailable.")
        candidates = [
            Candidate(f"{index:03d}", **product.model_dump())
            for index, product in enumerate(request.candidates, 1)
        ]
        with prediction_lock:
            results = ranker.rank(request.query, candidates)
        return RankResponse(query=request.query, candidate_count=len(candidates), products=results)

    if storefront is not None:
        from .store_web import mount_store_assets, store_router

        app.include_router(store_router(storefront))
        mount_store_assets(app, storefront)

    app.mount("/", StaticFiles(directory=site_dir, html=True), name="demo")
    return app
