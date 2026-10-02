"""Step 04 - Naive RAG in plain Python.

The whole pipeline, no frameworks:
    PDF -> chunks -> embeddings -> Chroma -> retrieve top-k -> prompt -> answer

Run from the repo root:
    python 04_naive_rag/starter.py                      # interactive, default PDF
    python 04_naive_rag/starter.py my.pdf -q "What is this document about?"

Work through the TODOs in order (1 -> 6). Each one is a single stage of the
pipeline. Stuck? Peek at solutions/04_naive_rag.py
"""

# Default: OpenAI. To use Mistral/OpenRouter/other, change these two lines:
#   client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=...)
#   model = "mistral-small-latest"  # or whichever model
# (Embeddings need an embedding model too, e.g. EMBEDDING_MODEL = "mistral-embed".)

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 800     # characters, roughly 200 tokens
CHUNK_OVERLAP = 100  # so a sentence cut at a boundary still appears whole in one chunk
TOP_K = 4


# ---------------------------------------------------------------------------
# Setup helpers (already done for you)
# ---------------------------------------------------------------------------

def make_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key in.")
    return OpenAI()


def resolve_pdf_path(cli_arg: str | None) -> Path:
    """CLI argument wins, then the RAG_PDF_PATH env var, then the default sample."""
    path = Path(cli_arg or os.getenv("RAG_PDF_PATH") or DEFAULT_PDF)
    if not path.exists():
        sys.exit(f"PDF not found: {path}\nDownload the sample docs first - see sample_docs/README.md")
    return path


def load_pdf_pages(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def collection_name_for(pdf_path: Path) -> str:
    # Chroma names: 3-63 chars of [a-zA-Z0-9._-], starting and ending alphanumeric.
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem).strip("_-")[:40] or "doc"
    return f"naive_{slug}"


# ---------------------------------------------------------------------------
# 1. Chunk
# ---------------------------------------------------------------------------

def chunk_pages(pages: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Fixed-size character windows, per page, so every chunk knows its page number."""
    chunks = []
    step = chunk_size - overlap
    for page_number, text in enumerate(pages, start=1):
        text = " ".join(text.split())  # PDF text is full of stray newlines

        # TODO 1: Slide a window of `chunk_size` characters across `text`,
        #         moving forward by `step` each time (so windows overlap).
        #   - for start in range(0, len(text), step):
        #   -     piece = text[start : start + chunk_size]
        #   - Skip pieces that are only whitespace.
        #   - Append {"text": piece, "page": page_number} to `chunks`.
        # Yes, this cuts words in half. That's the "naive" part - step 05a fixes it.
        raise NotImplementedError("TODO 1: split each page into overlapping fixed-size chunks")
    return chunks


# ---------------------------------------------------------------------------
# 2. Embed
# ---------------------------------------------------------------------------

def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    # Batching: one API call per 100 texts instead of one per text. A long PDF
    # can have thousands of chunks, and APIs cap how much you send per request.
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # TODO 2: Embed `batch` and add the vectors to `embeddings`.
        #   - Same call as step 03: client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        #   - embeddings.extend(item.embedding for item in response.data)
        #   (Plain lists are fine here - Chroma doesn't need numpy arrays.)
        raise NotImplementedError("TODO 2: embed one batch of texts")
    return embeddings


# ---------------------------------------------------------------------------
# 3. Store
# ---------------------------------------------------------------------------

def fresh_collection(chroma_client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    # Rebuilding every run keeps things simple: if you change CHUNK_SIZE you
    # see the effect immediately, and embedding one paper costs a fraction of a cent.
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet
    # Cosine distance matches the similarity we computed by hand in step 03.
    return chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})


def index_chunks(collection: chromadb.Collection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    # TODO 3: Store every chunk in Chroma with collection.add(...). It takes
    #         four parallel lists - item i of each describes chunk i:
    #   - ids=         unique strings, e.g. [f"chunk-{i}" for i in range(len(chunks))]
    #   - documents=   the chunk texts
    #   - embeddings=  the vectors from TODO 2 (we computed them ourselves, so
    #                  Chroma doesn't need to embed anything)
    #   - metadatas=   [{"page": c["page"]} for c in chunks] - kept for citations
    raise NotImplementedError("TODO 3: add chunks, embeddings and metadata to the collection")


# ---------------------------------------------------------------------------
# 4. Retrieve
# ---------------------------------------------------------------------------

def retrieve(client: OpenAI, collection: chromadb.Collection, question: str, k: int = TOP_K) -> list[dict]:
    # TODO 4: Find the k chunks closest to the question.
    #   a) Embed the question with embed_texts(client, [question])[0].
    #      It MUST use the same model as the chunks, or the vectors live in
    #      different spaces and the distances mean nothing.
    #   b) results = collection.query(query_embeddings=[that_vector], n_results=k)
    #   c) Chroma returns one list per query, and we sent one query, so the data
    #      is in results["documents"][0], results["metadatas"][0] and
    #      results["distances"][0]. zip them together and return a list of
    #      {"text": ..., "page": meta["page"], "distance": ...} dicts.
    raise NotImplementedError("TODO 4: embed the question and query Chroma")


# ---------------------------------------------------------------------------
# 5. Prompt + generate
# ---------------------------------------------------------------------------

def build_prompt(question: str, chunks: list[dict]) -> str:
    # TODO 5: Build the prompt string the LLM will see.
    #   - Join the chunks into one context string, labelling each with its page:
    #       "[page 3] chunk text...\n\n[page 5] chunk text..."
    #   - Then return something like:
    #       "Answer the question using only the context below. If the answer is
    #        not in the context, say you don't know.
    #
    #        Context:
    #        <context>
    #
    #        Question: <question>
    #        Answer:"
    raise NotImplementedError("TODO 5: assemble instruction + context + question")


def generate_answer(client: OpenAI, prompt: str) -> str:
    # TODO 6: Send the prompt to CHAT_MODEL as one user message and return the text.
    #   - Exactly like step 02, but use temperature=0: for factual Q&A we want
    #     the most likely answer, not a creative one.
    raise NotImplementedError("TODO 6: call the chat model with the prompt")


# ---------------------------------------------------------------------------
# Main (already done - read it to see how the pieces connect)
# ---------------------------------------------------------------------------

def answer_question(client: OpenAI, collection: chromadb.Collection, question: str) -> None:
    chunks = retrieve(client, collection, question)
    print("\nRetrieved chunks:")
    for c in chunks:
        print(f"  [page {c['page']:>2} | distance {c['distance']:.3f}] {c['text'][:90]}...")
    answer = generate_answer(client, build_prompt(question, chunks))
    print(f"\nAnswer: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to index (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", help="ask one question and exit (otherwise interactive)")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    client = make_client()

    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)
    print(f"Loaded {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks")

    embeddings = embed_texts(client, [c["text"] for c in chunks])
    print(f"Embedded {len(embeddings)} chunks ({len(embeddings[0])} dimensions each)")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = fresh_collection(chroma_client, collection_name_for(pdf_path))
    index_chunks(collection, chunks, embeddings)
    print(f"Stored {collection.count()} chunks in Chroma collection '{collection.name}'")

    if args.question:
        answer_question(client, collection, args.question)
        return

    print("\nAsk questions about the document (empty line or 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C
            break
        if question.lower() in {"", "quit", "exit"}:
            break
        answer_question(client, collection, question)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
