# Rag_GL - Build RAG from scratch, then with LangChain

## 1. What this is

A hands-on, 3-hour workshop on **Retrieval-Augmented Generation (RAG)**: getting
an LLM to answer questions about *your* documents instead of making things up.
You start by calling an LLM directly and watching it hallucinate. Then you
build a full RAG pipeline by hand in plain Python (chunking, embeddings, a
vector database, retrieval and prompting), so nothing is magic. After that you
improve it and measure it with LangChain. Each step is a small program with
gaps (`# TODO`) that you fill in yourself.

## 2. The workshop arc

| Step | Folder | Time | What you do |
|---|---|---|---|
| 1 | [`01_hello_llm`](01_hello_llm/) | 10 min | Call an LLM from Python and prove your API key works. |
| 2 | [`02_hallucination`](02_hallucination/) | 10 min | Ask about documents the LLM hasn't seen and watch it invent answers. |
| 3 | [`03_embeddings`](03_embeddings/) | 25 min | Turn text into vectors and show that similar meanings land close together. |
| ☕ | | 10 min | Break |
| 4 | [`04_naive_rag`](04_naive_rag/) | 30 min | Plain-Python RAG end to end: chunk → embed → Chroma → retrieve → answer. |
| 5 | [`05_improvements`](05_improvements/) | 45 min | **LangChain starts here.** Better chunking, a reranker, better prompts, plus hybrid search theory. |
| ☕ | | 10 min | Break |
| 6 | [`06_evaluation`](06_evaluation/) | 20 min | A tiny eval set: measure retrieval hit-rate and answer accuracy. |
| 7 | [`07_langchain_bridge`](07_langchain_bridge/) | 10 min | The whole pipeline in ~30 lines of LCEL, and where to go next. |

Steps 01-04 use **plain Python only** (OpenAI SDK, numpy, chromadb). Steps 05-07
use **LangChain**. The diagrams are in [`docs/rag_flow.md`](docs/rag_flow.md) and
the slides in [`slides/workshop.md`](slides/workshop.md).

## 3. Prerequisites

You should already have done these from the setup message sent before the
workshop. Quick checklist:

- **Python 3.10 or 3.11**: check with `python --version`. (3.12+ will probably work but hasn't been tested.)
- **An IDE**: VS Code, PyCharm, or similar.
- **Git**: check with `git --version`.
- **An API key** for OpenAI, or another OpenAI-compatible provider (see [Using another provider](#using-another-provider)), **with credit on the account**. A brand-new key on an account with no billing set up will fail.
- **Comfortable Python**: functions, lists, dicts, f-strings. No ML background needed.

## 4. Setup

```bash
# 1. Clone
git clone git@github.com:Yash-Mahadeshwar/RAG_GL.git
cd RAG_GL

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (cmd / PowerShell)

# 3. Install dependencies (takes 1-3 minutes - chromadb is the big one)
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env               # Windows: copy .env.example .env
# then open .env and paste your key after OPENAI_API_KEY=

# 5. Download the sample PDF (see sample_docs/README.md for the others)
curl -L -o sample_docs/attention_is_all_you_need.pdf https://arxiv.org/pdf/1706.03762

# 6. Smoke test - should print an answer from the model
python solutions/01_hello_llm.py
```

> **Always run scripts from the repo root**, e.g. `python 04_naive_rag/starter.py`.
> (They also work from other folders, but keeping to one place avoids confusion.)

### Using another provider

The code uses the OpenAI SDK, which also talks to any OpenAI-compatible API.
To switch, set `OPENAI_BASE_URL` in `.env` and change `CHAT_MODEL` /
`EMBEDDING_MODEL` at the top of each script. Every file has a comment block
showing where. Note that RAG needs an **embeddings** endpoint as well as chat,
and not every provider offers one, so check before switching.

## 5. How to work through this

For each step:

1. **Read the step's README** to see what you're building and why.
2. **Open `starter.py`** and find the `# TODO` comments. Each one explains what
   to write. The line underneath is a `raise NotImplementedError(...)`;
   replace it with your code.
3. **Run it**: `python 0X_step_name/starter.py`. If a TODO isn't done yet, you'll
   get a friendly `[not done yet] line N: TODO ...` message pointing at it.
   Fill in TODOs one at a time and rerun after each.
4. **Compare with "What success looks like"** in the README.
5. **Stuck?** Read the README's *Peek-ahead* hint first. If still stuck, open the
   matching file in [`solutions/`](solutions/). No shame in it; the goal is
   to understand the pipeline, not to memorise APIs.

**Fallen behind?** Every step is self-contained: no step imports code from a
previous one. Run the solution to see the result (`python solutions/04_naive_rag.py`)
and move on to the next step with everyone else. You can come back later.

## 6. Troubleshooting

**`pip install` fails on chromadb**
- Check `python --version`: it must be 3.10 or 3.11. On 3.13+ some wheels may be missing.
- Upgrade pip first: `python -m pip install --upgrade pip`, then retry.
- Windows: if it complains about C++ build tools, you're almost certainly on an
  unsupported Python version. Install 3.11 from python.org.
- macOS: `xcode-select --install` if you see compiler errors.

**`ModuleNotFoundError: No module named 'openai'` (or any other package)**
Your virtual environment isn't active. Run `source .venv/bin/activate` (or the
Windows equivalent) and check that your IDE uses the `.venv` interpreter.

**`OPENAI_API_KEY is not set`**
The `.env` file is missing or in the wrong place. It must sit in the repo
root, next to `requirements.txt`, and be named exactly `.env` (not `.env.txt`;
Windows hides extensions, so check carefully).

**`AuthenticationError` / `401` / "Incorrect API key"**
Usually a typo, a stray space or quotes around the key in `.env`. The line
should look like `OPENAI_API_KEY=sk-proj-abc123...` with nothing else on it.

**`RateLimitError` / `429` with `insufficient_quota` or "no credits remaining"**
The key is valid, but the account has no credit. Add billing on the provider's
dashboard. This is the most common day-of issue.

**`RateLimitError` / `429` without the quota message**
Too many requests too fast. That's common when a whole room shares one org
key. Wait 20-30 seconds and rerun.

**`PDF not found`**
Download the sample PDF (setup step 5), or pass your own path as the first
argument.

**`APIConnectionError`**
Network problem. Campus Wi-Fi sometimes blocks API endpoints, so try a phone
hotspot.

**Chroma errors about dimensions or a corrupted collection**
Usually caused by switching embedding model or provider. Delete the index and
rebuild: `rm -rf chroma_db` (Windows: `rmdir /s chroma_db`). It's regenerated
on the next run.

**05b/06 hang on first run**
They're downloading the ~22 MB reranker model into `.flashrank_cache/`. This
happens once.

**`langchain` import errors**
LangChain is split into several packages. Install exactly what's in
`requirements.txt`; don't `pip install langchain` on top, because mismatched
versions break imports.

## 7. Where to go next

- **Agents with [LangGraph](https://langchain-ai.github.io/langgraph/).** Let
  the model decide *when* to retrieve, rewrite the query, or call other tools,
  in a loop. "Agentic RAG" is the natural next build.
- **Hybrid search.** Combine keyword search (BM25) with vectors. Theory is in
  [`05_improvements/5d_hybrid_search`](05_improvements/5d_hybrid_search/).
- **Hosted vector stores.** For production scale and multi-user access, look
  at Pinecone, Weaviate, Qdrant, Chroma Cloud, or `pgvector` if you already
  run Postgres.
- **Serious evaluation.** [Ragas](https://docs.ragas.io/) (faithfulness, context
  precision), [LangSmith](https://docs.smith.langchain.com/) (tracing and
  datasets), and building a larger, honest eval set from real user questions.
- **Better ingestion.** Tables, images and scanned PDFs need layout-aware
  parsers ([Unstructured](https://unstructured.io/), [Docling](https://github.com/docling-project/docling)).
