"""Step 01 - Hello, LLM.

Send one message to a chat model and print the reply. If this runs, your API
key, network and Python environment are all working.

Run from the repo root:
    python 01_hello_llm/starter.py
    python 01_hello_llm/starter.py "Explain embeddings to a 10 year old"

Stuck? Peek at solutions/01_hello_llm.py
"""

# Default: OpenAI. To use Mistral/OpenRouter/other, change these two lines:
#   client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=...)
#   model = "mistral-small-latest"  # or whichever model

import os
import sys
import traceback
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAT_MODEL = "gpt-4o-mini"


def make_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key in.")

    # TODO 1: Create and return an OpenAI client.
    #   - Call OpenAI() with NO arguments. It reads OPENAI_API_KEY from the
    #     environment (load_dotenv above put it there from your .env file).
    #   - Why not pass the key directly? Keys in source code end up on GitHub.
    raise NotImplementedError("TODO 1: return an OpenAI() client")


def call_llm(client: OpenAI, question: str):
    # TODO 2: Send the question to the model and return the full response.
    #   - Use client.chat.completions.create(model=..., messages=[...], temperature=...)
    #   - model: CHAT_MODEL
    #   - messages is a LIST of dicts, each with "role" and "content":
    #       {"role": "system", "content": "You are a friendly teaching assistant. Answer in 2-3 sentences."}
    #       {"role": "user",   "content": question}
    #     The system message sets behaviour; the user message is the request.
    #   - temperature=0.2 (low = more repeatable answers)
    raise NotImplementedError("TODO 2: call client.chat.completions.create(...)")


def explain_api_error(exc: openai.APIError) -> str:
    # The raw tracebacks for these are long and scary; this is what they mean.
    if isinstance(exc, openai.AuthenticationError):
        return "Your API key was rejected. Check .env for typos, quotes or spaces, or create a new key."
    if isinstance(exc, openai.RateLimitError):
        if "quota" in str(exc) or "credit" in str(exc):
            return ("Your key is valid but the account has no credits. Add billing at "
                    "https://platform.openai.com/settings/organization/billing/ "
                    "or switch provider (see .env.example).")
        return "Rate limited: too many requests at once. Wait ~20 seconds and try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Couldn't reach the API. Check your internet, proxy or VPN (campus Wi-Fi sometimes blocks it)."
    if isinstance(exc, openai.NotFoundError):
        return f"Model not found. Does your account or provider offer '{CHAT_MODEL}'?"
    return f"API error: {exc}"


def main() -> None:
    question = " ".join(sys.argv[1:]) or "In one sentence, what is retrieval-augmented generation?"

    client = make_client()
    try:
        response = call_llm(client, question)
    except openai.APIError as exc:
        sys.exit(explain_api_error(exc))

    # TODO 3: Pull the answer text out of the response.
    #   - The response holds a list of `choices`; we asked for one.
    #   - The text lives at: response.choices[0].message.content
    #   - Tip: print(response) once to see the whole structure.
    answer = None
    raise NotImplementedError("TODO 3: extract the answer text from response")

    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    # Tokens are what you pay for - worth building the habit of looking.
    usage = response.usage
    print(f"[{CHAT_MODEL}] prompt tokens: {usage.prompt_tokens}, "
          f"completion tokens: {usage.completion_tokens}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
