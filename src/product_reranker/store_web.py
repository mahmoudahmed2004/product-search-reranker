"""HTTP API and storefront page routes, mounted alongside the V1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from .store import COLLECTIONS, Product, Recommendation


@dataclass(frozen=True, slots=True)
class StoreOptions:
    directory: Path = Path("storefront")
    index: Path = Path("data/store-index")
    cache: Path = Path("data/store-models")


class Collection(BaseModel):
    id: str
    label: str
    count: int


class CatalogResponse(BaseModel):
    products: list[Product]
    collections: list[Collection]
    total: int


class HomeContext(BaseModel):
    source: Literal["search", "view", "none"]
    collection: str | None = None
    label: str = ""
    products: list[Product] = Field(default_factory=list)


class ProductPage(BaseModel):
    product: Product
    similar: list[Recommendation]
    complements: list[Recommendation]


def store_router(options: StoreOptions) -> APIRouter:
    router = APIRouter()

    @router.get("/api/store/catalog", response_model=CatalogResponse, tags=["Store"])
    def catalog(
        request: Request,
        q: Annotated[str, Query(max_length=160)] = "",
        collection: Annotated[str, Query(max_length=30)] = "",
        rank: Literal["hybrid", "reranked"] = "hybrid",
    ) -> CatalogResponse:
        if collection and collection not in COLLECTIONS:
            raise HTTPException(422, "Unknown collection.")
        engine = request.app.state.store
        products = (
            engine.search(q, collection, strategy=rank) if q.strip() else engine.catalog(collection)
        )
        return CatalogResponse(
            products=products,
            total=len(products),
            collections=[
                Collection(id=key, label=label, count=len(engine.catalog(key)))
                for key, label in COLLECTIONS.items()
            ],
        )

    @router.get("/api/store/home-context", response_model=HomeContext, tags=["Store"])
    def home_context(
        request: Request,
        q: Annotated[str, Query(max_length=160)] = "",
        viewed: Annotated[str, Query(max_length=180)] = "",
        collection: Annotated[str, Query(max_length=30)] = "",
    ) -> HomeContext:
        if collection and collection not in COLLECTIONS:
            raise HTTPException(422, "Unknown collection.")
        engine = request.app.state.store
        if q.strip():
            topic, products = engine.home_search(q, collection)
            return HomeContext(
                source="search", collection=topic, label=q.strip(), products=products
            )
        anchor = engine.by_slug.get(viewed)
        if anchor:
            similar, complements = engine.recommendations(anchor)
            return HomeContext(
                source="view",
                collection=anchor.collection,
                label=anchor.title,
                products=[row.product for row in (*similar, *complements)][:4],
            )
        return HomeContext(source="none")

    @router.get("/api/store/products/{slug}", response_model=ProductPage, tags=["Store"])
    def product_page(request: Request, slug: str) -> ProductPage:
        engine = request.app.state.store
        product = engine.by_slug.get(slug)
        if product is None:
            raise HTTPException(404, "Product not found.")
        similar, complements = engine.recommendations(product)
        return ProductPage(product=product, similar=similar, complements=complements)

    @router.get("/store", include_in_schema=False)
    def storefront_redirect() -> RedirectResponse:
        return RedirectResponse("/store/")

    @router.get("/store/", include_in_schema=False)
    def storefront() -> FileResponse:
        return FileResponse(options.directory / "index.html")

    @router.get("/store/p/{slug}", include_in_schema=False)
    def storefront_product(request: Request, slug: str) -> FileResponse:
        exists = slug in request.app.state.store.by_slug
        return FileResponse(options.directory / "index.html", status_code=200 if exists else 404)

    return router


def mount_store_assets(app, options: StoreOptions) -> None:
    app.mount(
        "/store-assets",
        StaticFiles(directory=options.directory / "assets"),
        name="store-assets",
    )
