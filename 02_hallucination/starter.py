"""Step 02 - Watch the LLM make things up.

Ask questions about documents the model has never seen (or only half
remembers) and compare with what happens when we paste the facts in.

Run from the repo root:
    python 02_hallucination/starter.py
    python 02_hallucination/starter.py path/to/your.pdf -q "A question about your PDF"

Stuck? Peek at solutions/02_hallucination.py
"""

# Default: OpenAI. To use Mistral/OpenRouter/other, change these two lines:
#   client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=...)
#   model = "mistral-small-latest"  # or whichever model

import argparse
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "sample_docs" / "attention_is_all_you_need.pdf"
CHAT_MODEL = "gpt-4o-mini"
PRICE_PER_MILLION_INPUT_TOKENS = 0.15  # gpt-4o-mini, USD - check current pricing

# A made-up company handbook. The model cannot possibly know this, which makes
# it a clean test: any confident answer without context is invented.
PRIVATE_DOC = """\
Nimbus Analytics - Employee Handbook, Section 4.2 (Remote Work)
Employees may work remotely up to 3 days per week after completing their
90-day probation period. Remote days must be logged in the "Skyline" portal
before 10 AM. The company reimburses up to INR 1,500 per month for home
internet. Interns are not eligible for remote work."""

PRIVATE_QUESTION = (
    "According to the Nimbus Analytics employee handbook, how many days a week "
    "can employees work remotely, and how much is the monthly internet reimbursement?"
)

# Questions about the default sample PDF. The model has probably read this
# paper during training, so some answers will be right and some subtly wrong -
# and you can't tell which is which without the source.
PAPER_QUESTIONS = [
    "In the paper 'Attention Is All You Need', what dropout rate was used for the "
    "big Transformer model trained on English-to-French?",
    "In the paper 'Attention Is All You Need', what value of warmup_steps was used, "
    "and what was the exact BLEU score on WMT 2014 English-to-French?",
]


def make_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key in.")
    return OpenAI()


def ask_without_context(client: OpenAI, question: str) -> str:
    # TODO 1: Ask the model the question with NO extra information.
    #   - Same call as step 01: client.chat.completions.create(...)
    #   - This time skip the system message: messages=[{"role": "user", "content": question}]
    #   - temperature=0.2
    #   - Return the answer TEXT (response.choices[0].message.content), not the response.
    raise NotImplementedError("TODO 1: ask the model without any context")


def ask_with_context(client: OpenAI, question: str, context: str) -> str:
    # TODO 2: Same question, but this time paste the facts into the prompt.
    #   - Build one prompt string that contains, in this order:
    #       1. an instruction: "Answer the question using only the context below."
    #       2. the context (label it "Context:")
    #       3. the question (label it "Question:")
    #   - Send it as a single user message, return the answer text.
    #   This is RAG in its most primitive form. Steps 03-04 automate *finding*
    #   the right context instead of pasting it by hand.
    raise NotImplementedError("TODO 2: build a prompt with the context and ask again")


def resolve_pdf_path(cli_arg: str | None) -> Path:
    """CLI argument wins, then the RAG_PDF_PATH env var, then the default sample."""
    return Path(cli_arg or os.getenv("RAG_PDF_PATH") or DEFAULT_PDF)


def why_not_paste_everything(pdf_path: Path) -> None:
    if not pdf_path.exists():
        print(f"(Skipping: {pdf_path} not found - see sample_docs/README.md)")
        return
    pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
    chars = sum(len(p) for p in pages)
    approx_tokens = chars // 4  # rule of thumb for English text
    cost = approx_tokens / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS
    print(f"{pdf_path.name}: {len(pages)} pages, {chars:,} characters, ~{approx_tokens:,} tokens")
    print(f"Pasting all of it costs ~${cost:.4f} per question. Fine for one paper...")
    print(f"...but a 10,000-document knowledge base would be ~${cost * 10_000:,.0f} per question")
    print("and would not fit in any context window. We need to send only the relevant bits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", nargs="?", help="PDF to use (default: $RAG_PDF_PATH or the Attention paper)")
    parser.add_argument("-q", "--question", action="append", help="extra question to ask without context")
    args = parser.parse_args()

    client = make_client()

    print("=" * 70)
    print("PART A - a private document the model has never seen")
    print("=" * 70)
    print(f"Q: {PRIVATE_QUESTION}\n")
    print(f"Without context:\n  {ask_without_context(client, PRIVATE_QUESTION)}\n")
    print(f"With the handbook pasted in:\n  {ask_with_context(client, PRIVATE_QUESTION, PRIVATE_DOC)}\n")

    print("=" * 70)
    print("PART B - a famous paper the model half-remembers")
    print("=" * 70)
    for question in PAPER_QUESTIONS + (args.question or []):
        print(f"Q: {question}")
        print(f"A: {ask_without_context(client, question)}\n")
    print("Check these against the PDF. Ground truth: dropout 0.1, warmup_steps 4000, BLEU 41.8.\n")

    print("=" * 70)
    print("PART C - why not just paste the whole document every time?")
    print("=" * 70)
    why_not_paste_everything(resolve_pdf_path(args.pdf))


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
