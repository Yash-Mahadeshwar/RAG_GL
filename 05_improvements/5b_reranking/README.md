# Step 05b - Add a reranker (~15 min)

## What we're building
A two-stage retriever. First, vector search over-fetches 10 candidate chunks.
Then a small **cross-encoder** reranker (FlashRank, running locally on your
CPU) re-scores each candidate against the question, and we keep the top 3.
The script prints both rankings so you can watch chunks move.

## Why this step exists
Vector search embeds the question and each chunk *separately*, then compares
the two vectors. That's fast enough to search millions of chunks, but it's
lossy. A cross-encoder reads the question and chunk *together* in one forward
pass, so it can see whether this passage actually answers this question.
That's far more accurate, but too slow to run on every chunk. So we combine
them: cheap retrieval to get a shortlist, then expensive precision on the
shortlist. This **retrieve-then-rerank** pattern is standard in production
search systems. Step 06 measures how much it helps.

## What to fill in
- **TODO 1** (`main`): over-fetch with `vectorstore.similarity_search_with_score(question, k=FETCH_K)`.
- **TODO 2** (`rerank`): convert Documents to FlashRank passages, call `ranker.rerank(...)`, then map the results back to Documents.

## How to run it
```bash
python 05_improvements/5b_reranking/starter.py
python 05_improvements/5b_reranking/starter.py -q "What is the dropout rate?"
```
The first run downloads the reranker model (~22 MB) into `.flashrank_cache/`.

## What success looks like
```
Q: How long did it take to train the big model?

  Vector search top 10:
   #1  dist 0.391  [p. 1] ...BLEU score of 41.8 after training for 3.5 days on eight GPUs...
   #2  dist 0.402  [p. 8] ...big model ... outperforms the best previously reported models...
   ...
   #6  dist 0.455  [p. 7] (3.5 days). 5.3 Optimizer We used the Adam optimizer...

  After reranking, top 3:
   (was #6 ) score 0.981  [p. 7] (3.5 days). 5.3 Optimizer We used the Adam optimizer...
   (was #1 ) score 0.912  [p. 1] ...BLEU score of 41.8 after training for 3.5 days...
   (was #2 ) score 0.204  [p. 8] ...
```
Look for `(was #6)`-style jumps: a chunk vector search ranked low that the
reranker recognised as the real answer. Not every question shows a big
change. When vector search already got it right, the reranker mostly agrees.

## Peek-ahead
The `"id"` you put in each passage dict comes back on each result, so
`docs[r["id"]]` gets you the original Document.
Still stuck? → [`solutions/05b_reranking.py`](../../solutions/05b_reranking.py)
