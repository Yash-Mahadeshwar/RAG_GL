"""Step 05b - Add a reranker.

Vector search is fast but blunt: it compares two vectors computed separately.
A cross-encoder reranker reads the question and each chunk *together* and
scores how well the chunk answers it. Standard recipe: over-fetch with vector
search (say 10), rerank, keep the best few (say 3).

Run from the repo root:
    python 05_improvements/5b_reranking/starter.py
    python 05_improvements/5b_reranking/starter.py my.pdf -q "a question about my.pdf"

The first run downloads a ~22 MB reranker model into .flashrank_cache/.
Stuck? Peek at solutions/05b_reranking.py
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
from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # small cross-encoder, runs on CPU
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
FETCH_K = 10  # candidates from vector search
TOP_N = 3     # what survives reranking

DEFAULT_QUESTIONS = [
    "How long did it take to train the big model?",
    "Why is self-attention faster than recurrent layers?",
    "What regularization techniques were used during training?",
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
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": number})
        for number, page in enumerate(PdfReader(path).pages, start=1)
    ]


def get_vectorstore(pdf_path: Path, rebuild: bool = False) -> Chroma:
    """Open the persistent index for this PDF, building it only if it's empty."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem).strip("_-")[:40] or "doc"
    vectorstore = Chroma(
        collection_name=f"lc_{slug}_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if rebuild:
        vectorstore.reset_collection()
    if not vectorstore.get(limit=1)["ids"]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(load_pdf_documents(pdf_path))
        vectorstore.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {pdf_path.name}")
    return vectorstore


def rerank(ranker: Ranker, question: str, docs: list[Document], top_n: int = TOP_N) -> list[tuple[Document, float]]:
    # TODO 2: Rerank `docs` against `question` and return the best `top_n`
    #         as (Document, score) tuples, best first.
    #   a) FlashRank wants plain dicts, not Documents:
    #        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    #      The "id" is how we find the original Document again afterwards.
    #   b) results = ranker.rerank(RerankRequest(query=question, passages=passages))
    #      -> a list of dicts, already sorted best-first, each with "id" and "score".
    #   c) Return [(docs[r["id"]], float(r["score"])) for r in results[:top_n]]
    raise NotImplementedError("TODO 2: rerank the candidates with FlashRank")


def preview(doc: Document, width: int = 80) -> str:
    return f"[p.{doc.metadata['page']:>2}] {' '.join(doc.page_content.split())[:width]}..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", action="append", help="question(s) to test")
    parser.add_argument("--rebuild", action="store_true", help="re-embed the PDF even if an index exists")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    check_api_key()
    vectorstore = get_vectorstore(pdf_path, rebuild=args.rebuild)
    # log_level: FlashRank otherwise switches on INFO logging for every library.
    ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(REPO_ROOT / ".flashrank_cache"), log_level="WARNING")

    for question in args.question or DEFAULT_QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {question}")

        # TODO 1: Over-fetch FETCH_K candidates WITH their scores.
        #   - candidates = vectorstore.similarity_search_with_score(question, k=FETCH_K)
        #   - That returns a list of (Document, distance) tuples. With cosine,
        #     distance is lower = closer (it's 1 - cosine similarity).
        # Why fetch 10 when we only want 3? The right chunk is often at #6 or
        # #8 in vector search. The reranker can only promote what it's given.
        candidates = []
        raise NotImplementedError("TODO 1: fetch FETCH_K candidates with similarity_search_with_score")

        docs = [doc for doc, _ in candidates]
        print(f"\n  Vector search top {FETCH_K}:")
        for rank, (doc, distance) in enumerate(candidates, start=1):
            print(f"   #{rank:<2} dist {distance:.3f}  {preview(doc)}")

        print(f"\n  After reranking, top {TOP_N}:")
        for doc, score in rerank(ranker, question, docs):
            was = docs.index(doc) + 1
            print(f"   (was #{was:<2}) score {score:.3f}  {preview(doc)}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
