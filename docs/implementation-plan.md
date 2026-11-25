# Product Search Reranker implementation plan

This project compares two ways to order a fixed set of products for a shopping query. It uses the English (US) Amazon ESCI Task 1 benchmark and runs on CPU.

## 1. Prepare data and establish the baseline

- Download the examples and products Parquet files from the pinned Amazon revision. Verify their Git LFS SHA-256 values before reuse. Keep raw data out of Git.
- Filter to US Task 1 and join on locale and product ID. Keep each original candidate list intact. Normalize query text with Unicode NFKC, case folding, and whitespace and punctuation handling. Give official test queries precedence when their normalized text overlaps training. Reserve validation groups from official training with a stable hash. Record original and effective counts.
- Fit BM25 statistics using unique training products only. Score title and description separately and weight the title twice as high. Sort tied scores by product ID.

## 2. Train and evaluate LambdaRank

- Reuse the same candidate sets and feature extraction for both evaluation and inference. Train on BM25, query coverage in title and description, phrase, brand, color, and product-type overlap using the available catalog text. ESCI supplies no clean category field.
- Use ordered labels I, C, S, E and gains 0, 1, 10, 100 to match the relative weights of the ESCI challenge. Select the number of boosting rounds on validation. Score the held-out set once using nDCG@10, a paired bootstrap interval, and predefined one- or two-token and exact-title slices. Treat official query IDs as ranking groups and normalized text as split groups.
- Test gain ordering and cutoff, split isolation, candidate parity, and model serialization. Save the trained model and run outputs locally. Publish aggregate results and a small credited set of held-out cases for the demo.

## 3. Build the public demo

- Put a static demo in this docs directory so it can be served locally and on GitHub Pages. Let visitors choose a held-out query, compare the two ranked lists, and see each product's relevance judgment and change in rank.
- Let visitors send their own query and product titles to the Python server to score them with the saved model. Keep the recorded benchmark comparisons available on static hosts. Explain the candidate list boundary where visitors choose inputs.
- Use slate text, cool paper, and a restrained cobalt accent. Give the comparison most of the screen. Show rank movement next to each product so the effect stays legible when visitors switch views on mobile.

## 4. Verify the release

- Run pytest, Ruff, package build, and the full pinned benchmark. Verify browser behavior at desktop and mobile sizes. Document exact reproduction commands, data provenance, and observed gains and losses in the README.
