"""Step 05a - Smarter chunking with LangChain.

Compare the fixed-size chunker from step 04 with LangChain's
RecursiveCharacterTextSplitter, which prefers to cut at paragraph, line and
word boundaries. Then see whether cleaner chunks retrieve better.

Run from the repo root:
    python 05_improvements/5a_chunking/starter.py
    python 05_improvements/5a_chunking/starter.py my.pdf -q "a question about my.pdf"

Stuck? Peek at solutions/05a_chunking.py
"""

# Default: OpenAI. To use Mistral/OpenRouter/other, change these two lines:
#   client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=...)
#   model = "mistral-small-latest"  # or whichever model
# In LangChain that means: OpenAIEmbeddings(model=..., base_url=..., api_key=...,
# check_embedding_ctx_length=False) and ChatOpenAI(model=..., base_url=..., api_key=...).

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

DEFAULT_QUESTIONS = [
    "What is multi-head attention?",
    "Why did the authors use sinusoidal positional encodings?",
    "What hardware was the model trained on, and for how long?",
]


def check_api_key() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key in.")


def resolve_pdf_path(cli_arg: str | None) -> Path:
    """CLI argument wins, then the RAG_PDF_PATH env var, then the default sample."""
    path = Path(cli_arg or os.getenv("RAG_PDF_PATH") or DEFAULT_PDF)
    if not path.exists():
        sys.exit(f"PDF not found: {path}\nDownload the sample docs first - see sample_docs/README.md")
    return path


def load_pdf_documents(path: Path) -> list[Document]:
    # A LangChain Document is just text + a metadata dict - the same
    # {"text": ..., "page": ...} shape you built by hand in step 04.
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": number})
        for number, page in enumerate(PdfReader(path).pages, start=1)
    ]


def slug(pdf_path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem).strip("_-")[:40] or "doc"


def naive_split(docs: list[Document]) -> list[Document]:
    # separator="" means "cut anywhere": the same fixed-size windows as step 04.
    splitter = CharacterTextSplitter(separator="", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def recursive_split(docs: list[Document]) -> list[Document]:
    # TODO 1: Split with RecursiveCharacterTextSplitter instead.
    #   - splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    #     (same size budget as naive_split, so the comparison is fair)
    #   - return splitter.split_documents(docs)
    # How it works: it tries to cut on "\n\n" (paragraphs) first, then "\n",
    # then " " (words), and only cuts mid-word if a piece is still too big.
    # Page metadata is copied onto every chunk automatically.
    raise NotImplementedError("TODO 1: split docs with RecursiveCharacterTextSplitter")


def build_vectorstore(name: str, chunks: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    vectorstore = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.reset_collection()  # start clean so re-runs don't stack duplicates

    # TODO 2: Add the chunks to the vector store.
    #   - vectorstore.add_documents(chunks)
    # That one line replaces embed_texts() + index_chunks() from step 04: it
    # embeds every chunk with `embeddings` and stores text, vector and
    # metadata together.
    raise NotImplementedError("TODO 2: add the chunks to the vector store")
    return vectorstore


def show_boundaries(label: str, chunks: list[Document], count: int = 2) -> None:
    # Print where chunk N ends and chunk N+1 begins, so you can see mid-word
    # cuts. Skip page breaks - both splitters cut there, so they look identical.
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["page"] == b.metadata["page"]]
    print(f"  {label}:")
    for a, b in pairs[:count]:
        # Overlap means b repeats the tail of a; show where b *starts* instead.
        end = " ".join(a.page_content.split())[-60:]
        start = " ".join(b.page_content.split())[:60]
        print(f"    chunk ends:   ...{end}\n    next starts:  {start}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", action="append", help="question(s) to compare retrieval on")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    check_api_key()

    docs = load_pdf_documents(pdf_path)
    naive_chunks = naive_split(docs)
    recursive_chunks = recursive_split(docs)

    print(f"{pdf_path.name}: {len(docs)} pages\n")
    for label, chunks in [("naive", naive_chunks), ("recursive", recursive_chunks)]:
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"  {label:<10} {len(chunks):>4} chunks, avg {avg:.0f} chars")
    print("\nWhere chunks get cut (look for words sliced in half):")
    show_boundaries("naive", naive_chunks)
    show_boundaries("recursive", recursive_chunks)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    naive_store = build_vectorstore(f"cmp_naive_{slug(pdf_path)}", naive_chunks, embeddings)
    recursive_store = build_vectorstore(f"cmp_recursive_{slug(pdf_path)}", recursive_chunks, embeddings)

    for question in args.question or DEFAULT_QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {question}")
        for label, store in [("naive", naive_store), ("recursive", recursive_store)]:
            print(f"  --- {label}")
            # TODO 3: Get the 3 most similar chunks for `question` from `store`.
            #   - results = store.similarity_search(question, k=3)
            #   This replaces your retrieve() from step 04: it embeds the
            #   question and queries Chroma in one call, returning Documents.
            results = []
            raise NotImplementedError("TODO 3: run a similarity search on the store")
            for doc in results:
                text = " ".join(doc.page_content.split())
                print(f"    [p.{doc.metadata['page']:>2}] {text[:100]}...")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
