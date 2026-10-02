"""Step 07 - The whole thing in LangChain.

Everything you built by hand, as one LCEL pipeline. Read build_rag_chain()
top to bottom and name the step-04 function each line replaces.

Run from the repo root:
    python 07_langchain_bridge/starter.py
    python 07_langchain_bridge/starter.py my.pdf -q "What is this document about?"

Stuck? Peek at solutions/07_langchain_bridge.py
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
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHROMA_PATH = str(REPO_ROOT / "chroma_db")
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = (
    "Answer using ONLY the context. If it doesn't contain the answer, say "
    "\"I don't know based on the provided document.\" Cite pages like [p. 3]."
)


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


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[p. {d.metadata['page']}] {d.page_content}" for d in docs)


# ------------------------- the ~30 lines that matter -------------------------
def build_rag_chain(pdf_path: Path) -> Runnable:
    # load_pdf_pages()  ->  Documents (text + metadata)
    pages = [
        Document(page_content=p.extract_text() or "", metadata={"page": n})
        for n, p in enumerate(PdfReader(pdf_path).pages, start=1)
    ]
    # chunk_pages()  ->  a splitter
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(pages)

    # embed_texts() + fresh_collection() + index_chunks()  ->  a vector store
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem).strip("_-")[:40] or "doc"
    vectorstore = Chroma(
        collection_name=f"bridge_{slug}",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PATH,
    )
    if not vectorstore.get(limit=1)["ids"]:  # only embed on the first run
        vectorstore.add_documents(chunks)

    # retrieve()  ->  a retriever
    # TODO 1: Turn the vector store into a retriever that returns 4 chunks.
    #   - retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # A retriever is a Runnable: it has .invoke(question) -> list[Document],
    # which means it can be dropped into an LCEL pipeline with |.
    retriever = None
    raise NotImplementedError("TODO 1: create a retriever from the vector store")

    # build_prompt()  ->  a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    # generate_answer()  ->  a chat model + output parser
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # TODO 2: Compose the full pipeline and return it. The input is a plain
    #   question string; the output should be the answer string.
    #
    #   The first stage is a dict. LCEL runs every value on the SAME input and
    #   collects the results under those keys - exactly the {"context", "question"}
    #   that the prompt template needs:
    #       {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #   - retriever | format_docs : question -> Documents -> one context string
    #   - RunnablePassthrough()    : hands the question through unchanged
    #
    #   Then pipe that into: prompt | llm | StrOutputParser()
    #   Wrap the whole expression in parentheses so it can span several lines.
    raise NotImplementedError("TODO 2: compose the LCEL rag chain")
# ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", help="ask one question and exit (otherwise interactive)")
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    check_api_key()
    rag_chain = build_rag_chain(pdf_path)

    if args.question:
        stream_answer(rag_chain, args.question)
        return

    print("Ask questions about the document (empty line or 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C
            break
        if question.lower() in {"", "quit", "exit"}:
            break
        stream_answer(rag_chain, question)


def stream_answer(rag_chain: Runnable, question: str) -> None:
    # Every LCEL chain gets .invoke(), .batch() and .stream() for free.
    # TODO 3 (small): print the answer token-by-token as it arrives.
    #   - for token in rag_chain.stream(question):
    #         print(token, end="", flush=True)
    #   - then print() once at the end for a newline.
    #   (Swap .stream for .invoke to see the difference: same answer, all at once.)
    raise NotImplementedError("TODO 3: stream the answer with rag_chain.stream(question)")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
