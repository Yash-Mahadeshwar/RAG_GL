# Step 05d - Hybrid search (theory, ~5-10 min discussion)

> **No starter file here.** This is a read-and-discuss section. There's an
> optional code sketch at the bottom for anyone who finishes early.

## The problem: embeddings are bad at exact matches

Everything so far has used **dense retrieval**: embed the question, find the
nearest chunk vectors. It's great at *meaning*. "sign in" finds "login
credentials" even though they share no words.

But embeddings deliberately blur surface form, and sometimes the surface form
is the whole point:

| Query contains | Example | Why dense search struggles |
|---|---|---|
| Identifiers / codes | `ERR_CONN_4012`, `CVE-2024-3094`, order `#88231` | Random strings have no "meaning" for an embedding model to capture |
| Rare technical terms | `warmup_steps`, `d_ff`, `BM25` | Seen rarely in training, so they get a vague vector |
| Exact numbers | "the 28.4 BLEU result" | Embeddings treat 28.4 and 27.3 as near-identical |
| Names | a colleague's surname, a product codename | Same as identifiers |
| Very short queries | "P100" | Not enough text to embed meaningfully |

The classic keyword approach (**sparse retrieval**) is the opposite: excellent
at exact tokens, useless at synonyms. "automobile" never matches "car".

**Hybrid search runs both and merges the results.** Each covers the other's
blind spot.

## Sparse retrieval in one minute: BM25

BM25 is the standard keyword-ranking function behind Elasticsearch, Lucene
and most search boxes you've used. For each query word it asks three
questions about a chunk:

1. **Term frequency.** How often does the word appear in this chunk? More is
   better, with *diminishing returns*: 10 mentions isn't 10× better than 1.
2. **Inverse document frequency (IDF).** How rare is the word across *all*
   chunks? "the" appears everywhere, so it's worth almost nothing. "warmup_steps"
   appears in one chunk, so it's worth a lot.
3. **Length normalisation.** Is the chunk unusually long? Long chunks match
   more words by chance, so they're penalised slightly.

The score for a chunk is the sum over query words of
`IDF(word) × saturated_TF(word, chunk)`. No neural network or GPU is involved,
and it's very fast. The "sparse" name comes from representing each chunk as a
vocabulary-sized vector that is almost all zeros.

| | Dense (embeddings) | Sparse (BM25) |
|---|---|---|
| Matches | Meaning, paraphrases, other languages | Exact tokens |
| Fails on | IDs, codes, rare terms, numbers | Synonyms, rephrasing |
| Needs | An embedding model (API call or GPU) | Just tokenising the text |
| Index | Vector DB (HNSW, etc.) | Inverted index |

## Merging the two lists: Reciprocal Rank Fusion

You now have two ranked lists. You can't just add the scores, because they're
on different scales: cosine similarity is roughly 0-1, while BM25 scores are
unbounded (3.2, 17.8, ...).

**Reciprocal Rank Fusion (RRF)** ignores the raw scores entirely and uses only
*rank positions*:

```
RRF_score(chunk) = Σ over each list   1 / (k + rank_in_that_list)
```

`k` is a smoothing constant, conventionally **60**. A chunk that ranks well in
*both* lists beats one that ranks first in only one.

Worked example (k = 60):

| Chunk | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | - (not retrieved) | 1/61 = **0.0164** |
| B | 3 | 2 | 1/63 + 1/62 = **0.0320** |
| C | 2 | 5 | 1/62 + 1/65 = **0.0315** |
| D | - | 1 | 1/61 = **0.0164** |

Final order: **B, C, A/D**. B wins because both retrievers agree on it.

Alternatives: normalise both score lists to 0-1 and take a weighted blend
`α·dense + (1-α)·sparse`. That's more tunable, but you have to pick α, and
the normalisation is fragile. RRF is the usual default because it has
almost nothing to tune.

## Where it fits in the pipeline

```
               ┌─> dense search  (top 20) ─┐
question ──────┤                           ├─> RRF fuse ─> rerank (top 3) ─> prompt ─> LLM
               └─> BM25 search   (top 20) ─┘
```

Hybrid search improves **recall** (getting the right chunk into the
shortlist), and the reranker from 5b improves **precision** (putting it at the
top). They're complementary, and production systems often use both.

## When is it worth it?

- **Yes:** technical docs, legal/medical text, codebases, support tickets, product
  catalogues, or anything where users type IDs, error messages or jargon.
- **Probably not:** conversational, paraphrase-heavy questions over prose
  (like Paul Graham's essay). Dense search alone is usually fine there.
- **Always:** measure it. Add a few identifier-style questions to your step-06
  eval set and compare hit-rate with and without hybrid.

## Discuss

1. Look at the step-06 eval set. Which questions would BM25 probably get
   right on its own? Which would it miss?
2. A user searches your company wiki for `JIRA-4471`. Walk through what dense
   search returns, and what BM25 returns.
3. Why does RRF use ranks instead of scores? What information does it throw away?

## Where to find it in real tools

- **LangChain:** `BM25Retriever` and `EnsembleRetriever` (which does weighted
  RRF). These live in the older `langchain-community` / `langchain-classic`
  packages, so check current docs before relying on them.
- **Search engines and vector DBs with built-in hybrid:** Elasticsearch and
  OpenSearch, Weaviate, Qdrant (via sparse vectors), and Postgres (`pgvector`
  for dense plus built-in full-text search for sparse). Other hosted stores
  are adding it, so check your store's docs.
- **Learned sparse models** (SPLADE and similar) sit between the two: sparse
  vectors produced by a neural model, so you get exact-match behaviour *with*
  some synonym expansion.

---

## Optional: hybrid in ~25 lines (for early finishers)

Needs one extra package: `pip install rank-bm25`. Add this to a copy of
`solutions/06_evaluation.py` and evaluate `hybrid` alongside `vector_only`
and `with_rerank`:

```python
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

# Build the BM25 index over the same chunks that are in Chroma.
stored = vectorstore.get()  # {"ids": [...], "documents": [...], "metadatas": [...]}
all_chunks = [Document(page_content=t, metadata=m) for t, m in zip(stored["documents"], stored["metadatas"])]
bm25 = BM25Okapi([tokenize(d.page_content) for d in all_chunks])

def rrf(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1 / (k + rank)
            by_text[doc.page_content] = doc
    return [by_text[text] for text in sorted(scores, key=scores.get, reverse=True)]

def hybrid(question: str, k: int) -> list[Document]:
    dense = vectorstore.similarity_search(question, k=20)
    sparse = bm25.get_top_n(tokenize(question), all_chunks, n=20)
    return rrf([dense, sparse])[:k]
```
