# Step 03 - Embeddings and similarity (~25 min)

## What we're building
A script that turns six sentences into embedding vectors, prints a cosine
similarity matrix, and finds the sentences closest in meaning to a query, all
with numpy and no vector database yet.

## Why this step exists
RAG has to answer one question very fast: *which pieces of my documents are
relevant to this question?* Keyword search fails when the words differ ("sign
in" vs "login credentials"). An **embedding** model maps text to a vector (here
1,536 numbers) so that texts with similar *meaning* point in similar
directions. **Cosine similarity** measures that direction. Once you've written
`most_similar()` yourself, you know what a vector database does: the same
thing, with indexing tricks to make it fast over millions of vectors.

## What to fill in
- **TODO 1** (`embed`): call `client.embeddings.create(...)` on a list of texts and return a 2-D numpy array.
- **TODO 2** (`cosine_similarity`): `dot(a, b) / (|a| * |b|)` with numpy.
- **TODO 3** (`most_similar`): score every row against the query, sort and return the top k.

## How to run it
```bash
python 03_embeddings/starter.py
python 03_embeddings/starter.py "my laptop won't turn on"
```

## What success looks like
```
Embedded 6 sentences -> matrix of shape (6, 1536)

Cosine similarity matrix (look for the bright pairs: S0/S1, S2/S3, S4/S5):
         S0     S1     S2     S3     S4     S5
   S0  1.00   0.62   0.08   0.07   0.06   0.05
   S1  0.62   1.00   0.07   0.09   0.06   0.07
   S2  0.08   0.07   1.00   0.58   0.04   0.03
   ...
Query: "My account won't let me sign in"
  0.52  S5: I forgot my login credentials.
  0.47  S4: How do I reset my password?
  0.08  S0: The cat sat on the mat.
```
Exact numbers vary. The pattern is what matters: paraphrase pairs score
clearly higher than unrelated pairs, and the query finds the login sentences
even though it shares almost no words with them.

**Discuss:** with `text-embedding-3-small` even unrelated sentences rarely score
below ~0.0-0.1, and "similar" is often only 0.5-0.7. Only the *relative* order
matters, never the absolute number.

**Try it:** add a sentence that's a trap, like "The cat logged into the stock
exchange". Where does it land?

## Peek-ahead
`np.array([item.embedding for item in response.data])` for TODO 1;
`sorted(enumerate(scores), key=lambda p: p[1], reverse=True)[:k]` for TODO 3.
Still stuck? → [`solutions/03_embeddings.py`](../solutions/03_embeddings.py)
