# Assembly storefront and product recommendations

V2 of Assembly used a 39-product catalog, collection browsing, filtered search, product galleries, and an on-device shopping bag. Product pages show two recommendations for different shopping intents. **Similar products** are substitutes in the same product type. **Complete your setup** shows compatible accessories and items that can be used alongside the product.

[View the product page screenshot](storefront-product.png)

The current storefront and its expanded catalog are documented in the [V3 notes](storefront-v3.md). The figures below are the original V2 experiments.

## Run locally

From the repository root with Python 3.11 or later and a virtual environment active:

```sh
python -m pip install -e ".[store]"
python -m product_reranker serve-store
```

Open [Assembly](http://127.0.0.1:8000/store/) or go directly to the [iPhone 13 Pro](http://127.0.0.1:8000/store/p/iphone-13-pro) and [MacBook Pro](http://127.0.0.1:8000/store/p/apple-macbook-pro-14-inch-space-grey) pages. The original ESCI benchmark is at the server root. The first store launch downloads the encoder and creates the local index. Product photography is loaded from the catalog CDN. Sample prices are in USD and the shopping bag is a local demo.

## Retrieval and recommendation design

1. **Search.** Qdrant stores product descriptions as 384-dimensional Granite embeddings and sparse BM25 vectors. It combines retrieval ranks using Reciprocal Rank Fusion. Hybrid is the default on the reviewed catalog. Search results offer a Hybrid + reranker option that adds the V1 ESCI LambdaRank checkpoint and gives exact titles precedence.
2. **Similar products.** Retrieve stocked products of the same type using product-to-product cosine similarity. Rerank them with a supervised substitute checkpoint trained on external ESCI judgments.
3. **Complements.** Apply catalog eligibility before vector search and supervised reranking. A device-specific accessory needs explicit compatibility with its anchor device. The product page lists one of each eligible type before showing more from a repeated type. Reasons come from the product relationship policy.

Qdrant runs in persistent local mode in `data/store-index`. Its directory serves one process at a time. The demo uses English catalog text and uses the product as context. It contains no shopper behavioral data.

## Train the V2 relation models

```sh
python -m product_reranker download
python -m product_reranker train-store --queries 780
python -m product_reranker audit-store --index-dir data/store-audit-index
```

Training samples complete US Task 1 query groups from the pinned Amazon ESCI snapshot. One judged exact product anchors each group. Exact and substitute labels supply graded positives for the similar-item ranker, and complement labels supply positives for the complement ranker. The train, validation, and test sets have disjoint anchor IDs. Reversed pairs that occur across splits are removed from the earlier training partitions. LightGBM LambdaRank learns separately for each task from character and token overlap, brand, model numbers, and accessory cues. Granite stays frozen and supplies the vector retrieval stage. The relationship labels are shopping-query projections and can differ from direct item-pair judgments. The evaluation uses title-character cosine on all eligible held-out groups and a separate, smaller comparison against Granite cosine.

The command writes `v2-similar.txt`, `v2-complements.txt`, and a JSON training report under `artifacts/store-training`. The released models are already in `models/`. To serve new runs, copy the two new checkpoints into `models/` alongside the existing V1 assets, or use a separate model directory containing both releases. The original `train` command still reproduces the V1 search checkpoint.

## Results

The held-out relationship benchmark compares the trained rankers with title-character cosine on the same candidate lists.

| Relationship | Held-out groups | Character cosine nDCG@5 | Learned nDCG@5 |
| --- | ---: | ---: | ---: |
| Similar products | 255 | 0.7445 | 0.7300 |
| Complements | 65 | 0.1970 | 0.3714 |

The supervised model improved complements on the full sampled benchmark. Character overlap was stronger for similar products. On a separate 14-group-per-task comparison against Granite cosine, the learned models scored 0.8515 vs 0.7863 for similar items and 0.4215 vs 0.2217 for complements. Those estimates come from a small slice. The storefront combines learned and retrieval ranks, with product-type and compatibility filters around candidate generation. [Training report](storefront-training.json)

A separate review of **14 catalog search queries** scored 0.9211 with BM25, 0.9291 with dense search, 0.9611 with hybrid retrieval, and 0.9550 with the additional V1 reranker. These are manually judged demo cases. The recommendation audit checked all **39 products** for self-matches, unavailable items, and invalid add-ons. It found zero violations and passed the eight hand-reviewed cases for each recommendation type. [Store audit](storefront-audit.json)

## Data and references

- The storefront is a curated snapshot of [DummyJSON products](https://dummyjson.com/docs/products). Its [source revision](https://github.com/Ovi/DummyJSON/tree/3d78fed5f35b7f1087db1c162edc412f2a736722) and [license](../storefront/catalog-license.txt) are recorded in the repository. Run `python scripts/snapshot_catalog.py` to regenerate the snapshot.
- Reference data from [Apple's MagSafe Battery Pack specifications](https://support.apple.com/en-sa/111857) verifies the iPhone 13 Pro pairing. The iPhone 12 case is scoped to the model named in its product description.
- Training uses the [Amazon Shopping Queries Dataset](https://github.com/amazon-science/esci-data). The embedding checkpoint is [IBM Granite Embedding Small English R2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2). The retrieval design follows [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/).
