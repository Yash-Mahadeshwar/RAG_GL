"""Step 06 - Evaluate your RAG pipeline.

"It looks right when I try it" is not a measurement. Build a tiny labelled
eval set and measure two things separately:
  1. Retrieval hit-rate@k - did the right chunk make it into the top k?
  2. Answer accuracy      - is the final answer correct? (graded by an LLM judge)
Separating them tells you WHICH half of the pipeline to fix.

Run from the repo root:
    python 06_evaluation/starter.py
(The eval set below is written for the Attention paper. For your own PDF,
write your own questions - that is the real exercise.)

Stuck? Peek at solutions/06_evaluation.py
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
from typing import Callable

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
FETCH_K = 20       # candidates handed to the reranker
ANSWER_TOP_K = 4   # chunks given to the LLM when answering

IDK = "I don't know based on the provided document."

# Each item: the question, a reference answer, and an "evidence" string that
# appears (case-insensitive) in the chunk that contains the answer. Evidence
# of None marks a question the document cannot answer.
EVAL_SET = [
    {"question": "How many identical layers are in the Transformer encoder stack?",
     "expected": "6 identical layers (N = 6).", "evidence": "identical layers"},
    {"question": "How many attention heads does the base Transformer use?",
     "expected": "8 heads (h = 8 parallel attention layers).", "evidence": "parallel attention layers"},
    {"question": "What BLEU score did the Transformer get on WMT 2014 English-to-German?",
     "expected": "28.4 BLEU.", "evidence": "28.4"},
    {"question": "What hardware were the models trained on?",
     "expected": "One machine with 8 NVIDIA P100 GPUs.", "evidence": "p100"},
    {"question": "What value of warmup_steps was used in the learning rate schedule?",
     "expected": "4000.", "evidence": "warmup_steps"},
    {"question": "Which optimizer was used for training?",
     "expected": "Adam, with beta1 = 0.9, beta2 = 0.98 and epsilon = 1e-9.", "evidence": "adam optimizer"},
    {"question": "Who won the 2018 FIFA World Cup?",
     "expected": IDK, "evidence": None},
    # TODO 4: Add at least two questions of your own. Open the PDF, find a
    #   fact, and copy a short exact phrase from that sentence as "evidence".
    #   Pick phrases that are distinctive (a number, a technical term) - a
    #   common word like "model" would match almost every chunk.
    #   Example shape:
    #   {"question": "...?", "expected": "...", "evidence": "exact phrase"},
]

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful research assistant answering questions about one document.\n"
     "Use ONLY the context. If the context does not contain the answer, reply exactly: "
     f'"{IDK}"\nCite page numbers like [p. 3]. Be concise.'),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading a question-answering system against a reference answer.

Question: {question}
Reference answer: {expected}
System answer: {actual}

The system answer is CORRECT if it agrees with the reference on the key facts
(extra correct detail and citations are fine). If the reference says the
document doesn't contain the answer, the system answer is CORRECT only if it
also declines to answer.

Reply with exactly one word: CORRECT or INCORRECT."""
)

Retriever = Callable[[str, int], list[Document]]


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


def rerank(ranker: Ranker, question: str, docs: list[Document], top_n: int) -> list[Document]:
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    results = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [docs[r["id"]] for r in results[:top_n]]


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[p. {d.metadata['page']}]\n{d.page_content}" for d in docs)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def is_hit(docs: list[Document], evidence: str) -> bool:
    # TODO 1: Return True if `evidence` appears in ANY of the docs.
    #   - Compare lowercase to lowercase.
    #   - Collapse whitespace in the chunk first: " ".join(text.split())
    #     PDF text breaks lines in the middle of phrases, so "label\nsmoothing"
    #     would otherwise fail to match "label smoothing".
    #   - any(...) over the docs is a neat one-liner.
    raise NotImplementedError("TODO 1: check whether the evidence is in any retrieved chunk")


def hit_rate(retrieve: Retriever, eval_set: list[dict], k: int) -> float:
    # TODO 2: Fraction of ANSWERABLE questions whose evidence is in the top k.
    #   - Skip items whose "evidence" is None (nothing to find for those).
    #   - For each remaining item: docs = retrieve(item["question"], k)
    #     and check is_hit(docs, item["evidence"]).
    #   - Return hits / number_of_answerable_items  (a float between 0 and 1).
    # `retrieve` is a function you're handed - this lets us score vector-only
    # and reranked retrieval with the same code.
    raise NotImplementedError("TODO 2: compute hit-rate@k over the answerable questions")


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

def judge(judge_chain, question: str, expected: str, actual: str) -> bool:
    # TODO 3: Ask the judge LLM whether `actual` matches `expected`.
    #   - verdict = judge_chain.invoke({"question": ..., "expected": ..., "actual": ...})
    #     (the keys must match the {placeholders} in JUDGE_PROMPT)
    #   - The judge replies "CORRECT" or "INCORRECT". Careful: "CORRECT" is a
    #     substring of "INCORRECT", so check for INCORRECT first.
    raise NotImplementedError("TODO 3: call the judge chain and parse its verdict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("--rebuild", action="store_true", help="re-embed the PDF even if an index exists")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    check_api_key()
    vectorstore = get_vectorstore(pdf_path, rebuild=args.rebuild)
    # log_level: FlashRank otherwise switches on INFO logging for every library.
    ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(REPO_ROOT / ".flashrank_cache"), log_level="WARNING")

    def vector_only(question: str, k: int) -> list[Document]:
        return vectorstore.similarity_search(question, k=k)

    def with_rerank(question: str, k: int) -> list[Document]:
        return rerank(ranker, question, vectorstore.similarity_search(question, k=FETCH_K), top_n=k)

    print(f"\nRetrieval hit-rate on {sum(i['evidence'] is not None for i in EVAL_SET)} answerable questions")
    print(f"  {'k':>3}  {'vector only':>12}  {'+ rerank':>9}")
    for k in (1, 3, 5):
        print(f"  {k:>3}  {hit_rate(vector_only, EVAL_SET, k):>12.0%}  {hit_rate(with_rerank, EVAL_SET, k):>9.0%}")

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

    print(f"\nAnswer accuracy (reranked top {ANSWER_TOP_K}, judged by {CHAT_MODEL})")
    correct = 0
    for item in EVAL_SET:
        docs = with_rerank(item["question"], ANSWER_TOP_K)
        answer = answer_chain.invoke({"context": format_docs(docs), "question": item["question"]})
        ok = judge(judge_chain, item["question"], item["expected"], answer)
        correct += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {item['question'][:55]:<55} -> {' '.join(answer.split())[:60]}")
    print(f"\n  Accuracy: {correct}/{len(EVAL_SET)} = {correct / len(EVAL_SET):.0%}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
