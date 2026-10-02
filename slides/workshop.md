---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 34px; }
  h1 { font-size: 64px; }
  h2 { font-size: 50px; }
  code { font-size: 0.85em; }
  section.lead { text-align: center; justify-content: center; }
  section.lead h1 { font-size: 80px; }
---

<!-- _class: lead -->

# Build RAG from scratch

Retrieval-Augmented Generation, by hand, then with LangChain

NMIMS · B.Tech Data Science · 3 hours

`github.com/Yash-Mahadeshwar/RAG_GL`

---

## By the end of today you will

- Call an LLM from Python
- Watch it hallucinate, and understand why
- Build a full RAG pipeline **by hand**
- Improve it with LangChain
- **Measure** whether it actually got better

---

## The arc

| | Step | |
|---|---|---|
| 01 | Hello LLM | 10m |
| 02 | Hallucination | 10m |
| 03 | Embeddings | 25m |
| 04 | Naive RAG (plain Python) | 30m |
| 05 | Improvements (LangChain) | 45m |
| 06 | Evaluation | 20m |
| 07 | LangChain bridge | 10m |

---

## How each step works

1. Read the step's `README.md`
2. Open `starter.py` and find the `# TODO`s
3. Write the code and run it
4. Stuck? Read the hint, then `solutions/`

Run everything **from the repo root**.

---

<!-- _class: lead -->

# 01 · Hello, LLM

Prove the key works.

---

## An LLM call is just a function call

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are..."},
        {"role": "user",   "content": "question"},
    ],
)
```

Messages in, text out. You pay per **token**.

---

<!-- _class: lead -->

# 02 · Hallucination

The problem we're solving.

---

## LLMs don't know your documents

- Private docs: **never seen** them
- Famous docs: **half-remember** them
- Either way, they answer **confidently**

Right and wrong answers look **identical**.

---

## The fix, in its crudest form

> Paste the facts into the prompt.

It works! But a 10,000-document library won't fit in a prompt.

**We need to send only the relevant bits.**

---

<!-- _class: lead -->

# 03 · Embeddings

How do we find "the relevant bits"?

---

## Meaning → vectors

```
"How do I reset my password?"   → [0.01, -0.03, ..., 0.02]   (1,536 numbers)
"I forgot my login credentials" → [0.02, -0.02, ..., 0.01]
```

Similar meaning → similar **direction**

Cosine similarity = how aligned two vectors are

---

## Keyword search vs embeddings

"sign in" vs "login credentials"

- Keywords: **no match** (zero shared words)
- Embeddings: **close** (same meaning)

---

<!-- _class: lead -->

# ☕ Break · 10 min

Next: build RAG by hand.

---

<!-- _class: lead -->

# 04 · Naive RAG

Six functions. No framework.

---

## The whole pipeline

**Indexing** (once):
PDF → **chunk** → **embed** → **store** in Chroma

**Querying** (every question):
question → **embed** → **retrieve** top-k → **prompt** → **LLM**

Every RAG framework is a wrapper around this.

---

## Debugging RAG = finding the broken stage

- Bad **chunks**? (cut mid-sentence)
- Wrong chunks **retrieved**?
- Right chunks, bad **answer**?

---

<!-- _class: lead -->

# 05 · Improvements

LangChain starts here.

---

## What LangChain gives you

| You wrote | LangChain |
|---|---|
| `chunk_pages()` | `RecursiveCharacterTextSplitter` |
| `embed_texts()` | `OpenAIEmbeddings` |
| `index_chunks()` | `Chroma.add_documents` |
| `retrieve()` | `.as_retriever()` |
| `build_prompt()` | `ChatPromptTemplate` |
| function calls | `prompt \| llm \| parser` |

---

## 5a · Chunking

Naive: cut every 800 characters
→ `"...the encod" | "er is composed..."`

Recursive: paragraph → line → word
→ clean, self-contained chunks

The cheapest quality win in RAG.

---

## 5b · Reranking

Vector search: **fast**, but blunt

Cross-encoder: **accurate**, but slow

So use both:
fetch 10 with vectors → rerank → keep 3

---

## 5c · Prompting

Retrieval decides what the model **sees**

The prompt decides what it **does**

- Use ONLY the context
- Cite pages `[p. 3]`
- Exact phrase: *"I don't know based on the provided document."*

---

## 5d · Hybrid search (theory)

Embeddings miss **exact tokens**:
`ERR_4012` · `warmup_steps` · `P100`

BM25 keyword search misses **synonyms**

Run both and merge with **Reciprocal Rank Fusion**:

`score = Σ 1 / (60 + rank)`

---

<!-- _class: lead -->

# ☕ Break · 10 min

Next: is it actually better?

---

<!-- _class: lead -->

# 06 · Evaluation

"Looks right" is not a measurement.

---

## Measure the two halves separately

**Retrieval hit-rate@k**
Is the right chunk in the top k?

**Answer accuracy**
Is the answer correct? (LLM-as-judge)

Low hit-rate → fix retrieval
Good hit-rate, bad answers → fix the prompt

---

<!-- _class: lead -->

# 07 · The LangChain bridge

Everything, in ~30 lines.

---

## One pipe

```python
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.stream("What optimizer did they use?")
```

You know what every piece does, because you wrote it.

---

## Where to next

- **LangGraph**: agents that decide *when* to retrieve
- **Hybrid search**: BM25 + vectors
- **Hosted vector DBs**: Pinecone, Qdrant, Weaviate, pgvector
- **Real evals**: Ragas, LangSmith, real user questions

---

<!-- _class: lead -->

# Thank you

Build something with your own documents this week.

`github.com/Yash-Mahadeshwar/RAG_GL`
