# Assembly V3 · Contextual storefront

The storefront now has 149 products across 13 departments. Kitchen, sports, home, beauty, style, pantry, and pet supplies are in the same inventory as the V2 tech catalog. Individual kinds keep unrelated products in each department apart. A tennis racket pairs with tennis balls and a kitchen pan pairs with serving tools, while the existing phone-fit constraints continue to apply.

[Browse view](storefront-v3-home.png) · [Contextual view](storefront-v3-context.png)

## Try the homepage

1. Start the store with `python -m product_reranker serve-store` and open [Assembly](http://127.0.0.1:8000/store/).
2. Search for *tennis practice gear* and return to the homepage. You get the ball and racket, without products from other sports.
3. Search for *carbon steel wok* and return home. The homepage follows the newest search even if you open a product from a different department in between.
4. Search for *renew my passport* and return home. The page explains that there are no close matches. Use **Clear activity** to return to department browsing.

The browser uses the last successful search as its first signal, with the last viewed product as a fallback. Both expire after 24 hours. The site sends the active query or product to the recommendation API on demand. The demo uses no user account or server-side profile database.

## Ranking pipeline

1. Retrieve candidates with Granite embeddings and sparse BM25 in Qdrant.
2. Resolve the likely department using agreement between the retrievers, weighted title evidence, exact product names, and a separate check for semantically convergent product kinds. Withhold suggestions when the evidence is weak or a name can refer to unrelated departments.
3. Filter to the inferred department and remove candidates with weak evidence for the specific intent. Keep an exact product and its same-task alternatives together when applicable. Use the supervised ESCI LambdaRank checkpoint after this gate to reorder eligible candidates. This model shares the V1 feature implementation and model weights. Keep the leading retrieval result when a learned promotion would discard a substantial part of its title and tag evidence.
4. Limit the homepage to four relevant items. Use the V2 similar-item and complement models when the last viewed product supplies the context.

The [V2 notes](storefront.md) document the trained pair models. V3 extends the store and the query-side use of the supervised ranker. It uses the same ESCI checkpoints.

## Reproduce the audit

```sh
python -m pip install -e ".[dev,store]"
python -m pytest -q
node --test tests/test_home_state.mjs
python -m product_reranker audit-store --index-dir data/store-v3-audit
```

The audit compares raw hybrid search with the gated homepage, then compares ordering before and after LambdaRank on the same eligible candidates. The reviewed cases cover unrelated departments, ambiguous terms and queries outside the inventory. They measure this demo catalog. The larger ESCI benchmarks remain separate.


## Review results

The homepage selected the correct department for **21 of 22** matching searches. All **46** suggestions it displayed for these searches were in the expected department, compared with **64 of 88** from raw hybrid retrieval. It withheld recommendations for all **8** reviewed ambiguous or out-of-catalog searches. Raw hybrid retrieval returned products for all eight.

On the same homepage candidates, the ordering results were:

| Ordering | nDCG@4 across the 22 matching searches |
| --- | ---: |
| Gated retrieval | 0.9332 |
| Supervised reranking | 0.9271 |
| Supervised reranking with query evidence preserved | 0.9439 |

The model helped floral perfume. The query-evidence rule preserved the black product for black sunglasses. The example set was used to review and refine the rule and is too small to estimate production performance. Product-detail checks covered all 149 products and found zero violations of the recorded self-match, availability, and compatibility rules. [Full review data](storefront-v3-audit.json)
