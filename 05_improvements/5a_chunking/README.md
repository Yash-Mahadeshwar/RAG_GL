# Step 05a - Smarter chunking (~15 min)

## What we're building
A side-by-side comparison of two chunkers on the same PDF, with the same size
budget: the fixed-size splitter from step 04 and LangChain's
`RecursiveCharacterTextSplitter`. We look at where each one cuts, then index
both and compare what they retrieve for the same questions.

## Why this step exists
Chunking is the most underrated stage of RAG. The LLM can only answer from
the chunks you hand it. If a chunk starts with "...ers. The encod" and ends
mid-sentence, its embedding is muddier and the model gets broken context. The
recursive splitter tries paragraph breaks first, then line breaks, then
spaces, and only cuts mid-word as a last resort. It's a small change and
usually the cheapest quality win in RAG. It's also your first taste of
LangChain: a `Document` is your `{"text", "page"}` dict, and a splitter is
your `chunk_pages()`.

## What to fill in
- **TODO 1** (`recursive_split`): create a `RecursiveCharacterTextSplitter` and call `.split_documents()`.
- **TODO 2** (`build_vectorstore`): `vectorstore.add_documents(chunks)`, which embeds and stores in one line.
- **TODO 3** (`main`): `store.similarity_search(question, k=3)`.

## How to run it
```bash
python 05_improvements/5a_chunking/starter.py
python 05_improvements/5a_chunking/starter.py -q "What is label smoothing?"
```

## What success looks like
```
  naive        63 chunks, avg 703 chars
  recursive    63 chunks, avg 677 chars

Where chunks get cut (look for words sliced in half):
  naive:
    chunk ends:   ...der. The best performing models also connect the encoder and
    next starts:  orks that include an encoder and a decoder. The best perform...
  recursive:
    chunk ends:   ...ral networks that include an encoder and a decoder. The best
    next starts:  convolutional neural networks that include an encoder and a ...

Q: What is multi-head attention?
  --- naive
    [p. 5] ens to vectors of dimensiondmodel. We also use the usual learned...
  --- recursive
    [p. 4] Multi-Head Attention Instead of performing a single attention function...
```
Naive chunks start with fragments like `orks` and `ens`; recursive chunks
start on whole words. On a short paper the retrieved *pages* are often the
same. The difference shows up in how readable and self-contained the chunks
are, and it grows on long, messy documents.

**Try it:** set `CHUNK_SIZE = 300` and then `2000`. Small chunks are precise but
lose context; big chunks keep context but dilute the embedding. There's no
universal right answer, which is why step 06 (evaluation) exists.

## Peek-ahead
`RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)`.
Still stuck? → [`solutions/05a_chunking.py`](../../solutions/05a_chunking.py)
