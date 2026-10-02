# Step 04 - Naive RAG in plain Python (~30 min)

## What we're building
A complete RAG system in one file with no framework. It reads a PDF, cuts it
into chunks, embeds them, stores them in a local Chroma database, then
answers questions by retrieving the most relevant chunks and handing them to
the LLM.

```
PDF → chunk_pages → embed_texts → Chroma ──┐
                                           │
question → embed → collection.query → top-4 chunks → build_prompt → LLM → answer
```
(Full diagram: [`docs/rag_flow.md`](../docs/rag_flow.md))

## Why this step exists
This is the core of the workshop. Every RAG framework, LangChain included, is
a wrapper around these six operations. Once you've written each one, you can
debug any RAG system by asking which stage is failing: bad chunks, wrong
chunks retrieved, or a good context with a bad answer. It's called *naive*
because each stage is the simplest thing that works. Step 05 upgrades them
one at a time.

## What to fill in
Work top to bottom; each TODO is one pipeline stage.
- **TODO 1** (`chunk_pages`): fixed-size, overlapping character windows, tagged with page numbers.
- **TODO 2** (`embed_texts`): embed one batch with `client.embeddings.create`.
- **TODO 3** (`index_chunks`): `collection.add(ids=..., documents=..., embeddings=..., metadatas=...)`.
- **TODO 4** (`retrieve`): embed the question, `collection.query(...)`, unpack the results.
- **TODO 5** (`build_prompt`): instruction + page-labelled context + question.
- **TODO 6** (`generate_answer`): call the chat model with `temperature=0`.

## How to run it
```bash
python 04_naive_rag/starter.py                      # interactive Q&A
python 04_naive_rag/starter.py -q "What is multi-head attention?"
python 04_naive_rag/starter.py sample_docs/how_to_do_great_work.pdf
```
The index is rebuilt on every run. That's deliberate: change `CHUNK_SIZE` and
you see the effect straight away. It's stored in `chroma_db/` (gitignored).

## What success looks like
```
Loaded attention_is_all_you_need.pdf: 15 pages -> 65 chunks
Embedded 65 chunks (1536 dimensions each)
Stored 65 chunks in Chroma collection 'naive_attention_is_all_you_need'

> How many attention heads does the base model use?

Retrieved chunks:
  [page  5 | distance 0.412] dmodel×dv andW O∈ Rhdv×dmodel. In this work we employ h = 8 parallel...
  [page  4 | distance 0.455] Multi-Head Attention Instead of performing a single attention function...
  ...
Answer: The base model uses 8 attention heads (h = 8).
```
**Try these:** "What hardware was used for training?", "What is the capital
of France?" (does it refuse?), and a question whose answer straddles two
chunks. Look at the retrieved chunks: do you see words cut in half?

## Peek-ahead
For TODO 4, Chroma's results are lists of lists (one per query), so you want
`results["documents"][0]` and friends, zipped together.
Still stuck? → [`solutions/04_naive_rag.py`](../solutions/04_naive_rag.py)
