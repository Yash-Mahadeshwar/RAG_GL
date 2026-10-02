"""Step 05c - Better prompting and "I don't know".

Retrieval decides WHAT the model sees; the prompt decides what it DOES with
it. Compare a lazy prompt with a strict one on questions the document can and
can't answer.

Run from the repo root:
    python 05_improvements/5c_prompting/starter.py
    python 05_improvements/5c_prompting/starter.py my.pdf -q "a question about my.pdf"

Stuck? Peek at solutions/05c_prompting.py
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

DEFAULT_QUESTIONS = [
    "How many attention heads does the base model use?",  # answerable
    "What is the capital of Australia?",                  # off-topic: should decline
    "How many GPUs were used to train GPT-4?",            # plausible but not in the doc
]

# The "before": no rules, so the model happily mixes in what it already knows.
WEAK_PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}"
)

IDK = "I don't know based on the provided document."

# TODO 1: Write a system prompt that fixes the weak prompt's problems.
#   Good RAG system prompts usually include:
#   - a role ("You are a careful research assistant answering questions about one document")
#   - "use ONLY the context, never outside knowledge"
#   - an EXACT refusal phrase for when the answer isn't there. Use the IDK
#     constant above (it's an f-string, so write {IDK}). A fixed phrase is easy
#     to detect in code and in step 06's evaluation.
#   - "cite the page number(s) you used, like [p. 3]" so humans can verify
#   - a length limit
#   Numbered rules work well. Replace the placeholder string below.
STRONG_SYSTEM_PROMPT = f"""TODO: write your system prompt here. Refusal phrase: "{IDK}" """

STRONG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRONG_SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


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


def format_docs(docs: list[Document]) -> str:
    # TODO 2: Turn retrieved Documents into one context string.
    #   - Label each chunk with its page so the model CAN cite it:
    #       "[p. 3]\n<chunk text>"
    #   - The page number is in d.metadata["page"], the text in d.page_content
    #   - Join chunks with a clear separator, e.g. "\n\n---\n\n"
    raise NotImplementedError("TODO 2: format the docs into a context string with page labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", action="append", help="question(s) to test")
    parser.add_argument("--rebuild", action="store_true", help="re-embed the PDF even if an index exists")
    args = parser.parse_args()

    if STRONG_SYSTEM_PROMPT.startswith("TODO"):
        raise NotImplementedError("TODO 1: write STRONG_SYSTEM_PROMPT near the top of the file")

    pdf_path = resolve_pdf_path(args.pdf)
    check_api_key()
    retriever = get_vectorstore(pdf_path, rebuild=args.rebuild).as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # LCEL: the | operator pipes one step's output into the next.
    # prompt fills in the template -> llm generates -> parser pulls out the text.
    weak_chain = WEAK_PROMPT | llm | StrOutputParser()

    # TODO 3: Build the same chain, but with STRONG_PROMPT.
    #   - Copy the line above and swap the prompt.
    #   Notice the chain doesn't care which prompt it gets - that's the point
    #   of composable pieces.
    strong_chain = None
    raise NotImplementedError("TODO 3: build strong_chain with STRONG_PROMPT")

    for question in args.question or DEFAULT_QUESTIONS:
        docs = retriever.invoke(question)
        inputs = {"context": format_docs(docs), "question": question}
        print(f"\n{'=' * 70}\nQ: {question}")
        print(f"  pages retrieved: {[d.metadata['page'] for d in docs]}")
        print(f"\n  WEAK prompt:\n    {weak_chain.invoke(inputs)}")
        print(f"\n  STRONG prompt:\n    {strong_chain.invoke(inputs)}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
